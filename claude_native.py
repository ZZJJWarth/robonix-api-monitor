"""Claude Messages bridge for Pilot; protocol conversion never executes actions."""
import base64
import copy
import hashlib
import http.client
import json
import socket
import time

import jsonschema

from claude_schema import adapt_schema, select_capabilities
from monitor import Analysis, SafeStream, now, write_json, is_pilot_request


CALL_PREFIX = 'call_claude_'


def tool_id(blocks, index):
    raw = json.dumps({'blocks': blocks, 'index': index}, ensure_ascii=False, separators=(',', ':')).encode()
    return CALL_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip('=')


def signed_blocks(call):
    token = call['id'][len(CALL_PREFIX):]
    state = json.loads(base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)))
    block = state['blocks'][state['index']]
    if (block['type'] != 'tool_use' or block['name'] != call['function']['name']
            or block['input'] != json.loads(call['function']['arguments'])):
        raise ValueError('Claude tool call no longer matches its signed history')
    return state['blocks'], block['id']


def content_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    blocks = []
    for part in content:
        if part.get('type') == 'text':
            blocks.append({'type': 'text', 'text': part['text'], **({'cache_control': copy.deepcopy(part['cache_control'])} if 'cache_control' in part else {})})
        elif part.get('type') == 'image_url':
            url = part['image_url']['url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise ValueError('Claude bridge accepts inline base64 images only')
            mime, data = url[5:].split(';base64,', 1)
            blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': data},
                           **({'cache_control': copy.deepcopy(part['cache_control'])} if 'cache_control' in part else {})})
        else:
            raise ValueError('Unsupported Claude message content')
    return blocks


def build_request(chat, plan_depth=1, cap_limit=1, schema_mode="structured", guidance=None,
                  guidance_version=None):
    if schema_mode not in ('structured', 'json'):
        raise ValueError('Unsupported Claude schema mode')
    allowed = {'model', 'messages', 'stream', 'stream_options', 'response_format', 'tools', 'tool_choice',
               'max_tokens', 'max_completion_tokens', 'temperature', 'top_p', 'top_k', 'stop', 'n', 'user',
               'reasoning_effort', 'prompt_cache_key', 'prompt_cache_options'}
    if set(chat) - allowed or chat.get('n', 1) != 1:
        raise ValueError('Unsupported Chat option in Claude bridge')
    if chat.get('reasoning_effort') not in (None, 'none'):
        raise ValueError('Explicit Claude thinking configuration is not yet supported')
    messages, system, calls = [], [], {}
    for message in chat['messages']:
        role = message['role']
        if role in ('system', 'developer'):
            system.extend(content_blocks(message.get('content')))
            continue
        if role == 'tool':
            native_id = calls[message['tool_call_id']]
            text = message.get('content') or ''
            block = {'type': 'tool_result', 'tool_use_id': native_id, 'content': text}
            if messages and messages[-1]['role'] == 'user' and all(p['type'] == 'tool_result' for p in messages[-1]['content']):
                messages[-1]['content'].append(block)
            else:
                messages.append({'role': 'user', 'content': [block]})
            continue
        if role not in ('user', 'assistant'):
            raise ValueError('Unsupported Claude message role')
        blocks = content_blocks(message.get('content'))
        tool_calls = message.get('tool_calls') or []
        signed = [c for c in tool_calls if c['id'].startswith(CALL_PREFIX)]
        if signed:
            if len(signed) != len(tool_calls):
                raise ValueError('Mixed signed and unsigned Claude tool calls')
            blocks, _ = signed_blocks(signed[0])
            for call in signed:
                same, native_id = signed_blocks(call)
                if same != blocks:
                    raise ValueError('Claude parallel tool calls belong to different turns')
                calls[call['id']] = native_id
            if len([b for b in blocks if b['type'] == 'tool_use']) != len(tool_calls):
                raise ValueError('Incomplete signed Claude tool-call group')
            original_text = ''.join(b['text'] for b in blocks if b['type'] == 'text')
            if original_text != (message.get('content') or ''):
                raise ValueError('Claude assistant text no longer matches signed history')
        else:
            for call in tool_calls:
                native_id = 'toolu_' + hashlib.sha256(call['id'].encode()).hexdigest()[:32]
                calls[call['id']] = native_id
                args = json.loads(call['function']['arguments'])
                if not isinstance(args, dict):
                    raise ValueError('Tool arguments must be an object')
                blocks.append({'type': 'tool_use', 'id': native_id, 'name': call['function']['name'], 'input': args})
        if blocks:
            messages.append({'role': role, 'content': blocks})
    native = {'model': chat['model'], 'max_tokens': chat.get('max_completion_tokens', chat.get('max_tokens', 8192)),
              'messages': messages, 'stream': bool(chat.get('stream', False))}
    if system:
        native['system'] = system
    for field in ('temperature', 'top_p', 'top_k'):
        if field in chat:
            native[field] = chat[field]
    if 'stop' in chat:
        native['stop_sequences'] = [chat['stop']] if isinstance(chat['stop'], str) else chat['stop']
    metadata = {'transport': 'claude_messages', 'claude_schema_mode': schema_mode, 'untranslated_cache_controls': [
        k for k in ('prompt_cache_key', 'prompt_cache_options') if k in chat]}
    if is_pilot_request(chat) and guidance:
        native.setdefault('system', []).append({'type': 'text', 'text': guidance})
        if guidance_version:
            metadata['pilot_guidance'] = guidance_version
    shape = chat.get('response_format') or {}
    if shape.get('type') in ('json_schema', 'json_object') and schema_mode == 'json':
        native.setdefault('system', []).append({'type': 'text', 'text': (
            'Response transport for this round: return exactly ONE valid JSON object, beginning with { and ending with }. '
            'No Markdown fences, preamble, XML or text outside the object. Follow the Pilot RTDL envelope and capability catalog. '
            'All capabilities in the current catalog are available in THIS round; prior single-capability transport restrictions no longer apply. '
            'Compose the known dependent actions in a sequence or independent actions in parallel as the task requires. '
            'Keep the original task and total requested quantity; describe only what the actual calls perform. '
            'Do not output an empty plan for unfinished work when no actual plan is running.')})
        metadata['json_object_prompt_only'] = True
    elif shape.get('type') == 'json_schema':
        named = shape['json_schema']
        schema, adaptation = adapt_schema(named['schema'], plan_depth, rtdl=named.get('name') == 'rtdl_envelope')
        native['output_config'] = {'format': {'type': 'json_schema', 'schema': schema}}
        metadata['claude_schema'] = adaptation
        if named.get('name') == 'rtdl_envelope':
            branches = named['schema']['$defs']['node']['anyOf']
            capabilities = [b['properties']['cap']['enum'][0] for b in branches if 'cap' in b.get('properties', {})]
            metadata['claude_capability_selection'] = {'required': cap_limit > 0 and len(capabilities) > cap_limit,
                                                       'limit': cap_limit, 'catalog_count': len(capabilities)}
    elif shape.get('type') == 'json_object':
        native['messages'].append({'role': 'user', 'content': [{'type': 'text', 'text': 'Return exactly one valid JSON object for this response.'}]})
        metadata['json_object_prompt_only'] = True
    elif shape.get('type') not in (None, 'text'):
        raise ValueError('Unsupported Claude response format')
    for tool in chat.get('tools') or []:
        if tool.get('type') != 'function':
            raise ValueError('Unsupported Claude tool type')
        function = tool['function']
        parameters = copy.deepcopy(function.get('parameters', {'type': 'object', 'properties': {}}))
        entry = {'name': function['name'], 'description': function.get('description', ''), 'input_schema': parameters}
        if function.get('strict'):
            entry['input_schema'], _ = adapt_schema(parameters)
            entry['strict'] = True
        native.setdefault('tools', []).append(entry)
    if 'tool_choice' in chat:
        choice = chat['tool_choice']
        native['tool_choice'] = ({'type': 'tool', 'name': choice['function']['name']} if isinstance(choice, dict)
                                 else {'type': {'auto': 'auto', 'none': 'none', 'required': 'any'}[choice]})
    return native, metadata


def chat_response(message, requested_model):
    if message.get('stop_reason') not in ('end_turn', 'tool_use', 'stop_sequence'):
        raise ValueError('Claude response was refused, truncated, or incomplete')
    blocks = message.get('content') or []
    text, reasoning, tools = [], [], []
    for index, block in enumerate(blocks):
        kind = block['type']
        if kind == 'text':
            text.append(block['text'])
        elif kind == 'thinking':
            reasoning.append(block['thinking'])
        elif kind == 'redacted_thinking':
            continue
        elif kind == 'tool_use':
            tools.append({'id': tool_id(blocks, index), 'type': 'function', 'function': {
                'name': block['name'], 'arguments': json.dumps(block['input'], ensure_ascii=False, separators=(',', ':'))}})
        else:
            raise ValueError('Unsupported Claude output block')
    if not text and not tools:
        raise ValueError('Claude returned no usable output')
    reply = {'role': 'assistant', 'content': ''.join(text) or None}
    if reasoning:
        reply['reasoning_content'] = ''.join(reasoning)
    if tools:
        reply['tool_calls'] = tools
    raw = message.get('usage') or {}
    cached = raw.get('cache_read_input_tokens', 0)
    prompt = raw.get('input_tokens', 0) + cached + raw.get('cache_creation_input_tokens', 0)
    completion = raw.get('output_tokens', 0)
    return {'id': message['id'], 'model': message.get('model', requested_model), 'object': 'chat.completion',
            'created': int(time.time()), 'choices': [{'index': 0, 'message': reply, 'finish_reason': 'tool_calls' if tools else 'stop'}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion,
                      'prompt_tokens_details': {'cached_tokens': cached}}}


class NativeStream:
    def __init__(self):
        self.pending, self.message, self.blocks = b'', None, {}
        self.inputs, self.complete, self.events = {}, False, 0
        self.open_blocks = set()

    def feed(self, data, final=False):
        self.pending += data
        lines = self.pending.split(b'\n')
        self.pending = b'' if final else lines.pop()
        reasoning = []
        for line in lines:
            if not line.startswith(b'data:'):
                continue
            event = json.loads(line[5:].strip())
            self.events += 1
            kind = event['type']
            if kind == 'error':
                raise ValueError('Claude SSE returned an error event')
            if kind == 'message_start':
                if self.message is not None:
                    raise ValueError('Duplicate Claude message_start')
                self.message = copy.deepcopy(event['message'])
            elif kind == 'content_block_start':
                index = event['index']
                if index in self.blocks:
                    raise ValueError('Duplicate Claude content block')
                self.blocks[index] = copy.deepcopy(event['content_block'])
                self.open_blocks.add(index)
            elif kind == 'content_block_delta':
                index, delta = event['index'], event['delta']
                block = self.blocks[index]
                field = {'text_delta': 'text', 'thinking_delta': 'thinking', 'signature_delta': 'signature'}.get(delta['type'])
                if field:
                    block[field] = block.get(field, '') + delta[field]
                    if field == 'thinking':
                        reasoning.append(delta[field])
                elif delta['type'] == 'input_json_delta':
                    self.inputs[index] = self.inputs.get(index, '') + delta['partial_json']
                else:
                    raise ValueError('Unsupported Claude SSE content delta')
            elif kind == 'message_delta':
                if self.message is None:
                    raise ValueError('Claude message_delta before start')
                self.message.update(event['delta'])
                self.message.setdefault('usage', {}).update(event.get('usage') or {})
            elif kind == 'content_block_stop':
                self.open_blocks.remove(event['index'])
            elif kind == 'message_stop':
                if self.open_blocks:
                    raise ValueError('Claude message stopped with incomplete content blocks')
                self.complete = True
            elif kind != 'ping':
                raise ValueError('Unsupported Claude SSE event')
        return reasoning

    def completed(self):
        if not self.complete or self.message is None:
            raise ValueError('Claude stream ended before message_stop')
        message = copy.deepcopy(self.message)
        for i, raw in self.inputs.items():
            self.blocks[i]['input'] = json.loads(raw)
        message['content'] = [self.blocks[i] for i in sorted(self.blocks)]
        return message


def validate_result(result, chat, native):
    message = result['choices'][0]['message']
    tool_schemas = {t['function']['name']: t['function'].get('parameters', {}) for t in chat.get('tools') or []}
    for call in message.get('tool_calls', []):
        name = call['function']['name']
        if name not in tool_schemas:
            raise ValueError('Claude returned an undeclared tool')
        jsonschema.Draft202012Validator(tool_schemas[name]).validate(json.loads(call['function']['arguments']))
    text = message.get('content')
    shape = chat.get('response_format') or {}
    if shape.get('type') not in ('json_object', 'json_schema'):
        return None
    if not text and message.get('tool_calls'):
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected one JSON object')
    if shape.get('type') == 'json_schema':
        upstream_schema = native.get('output_config', {}).get('format', {}).get('schema')
        if upstream_schema is not None:
            jsonschema.Draft202012Validator(upstream_schema).validate(value)
        jsonschema.Draft202012Validator(shape['json_schema']['schema']).validate(value)
    return True


def encode_response(result, stream, omit_reasoning=False):
    if not stream:
        return json.dumps(result, ensure_ascii=False).encode(), 'application/json'
    reply = copy.deepcopy(result['choices'][0]['message'])
    if omit_reasoning:
        reply.pop('reasoning_content', None)
    for i, call in enumerate(reply.get('tool_calls', [])):
        call['index'] = i
    base = {k: result[k] for k in ('id', 'model', 'created')}
    base['object'] = 'chat.completion.chunk'
    events = [{**base, 'choices': [{'index': 0, 'delta': reply, 'finish_reason': None}]},
              {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': result['choices'][0]['finish_reason']}], 'usage': result['usage']}]
    raw = b''.join(b'data: ' + json.dumps(e, ensure_ascii=False).encode() + b'\n\n' for e in events) + b'data: [DONE]\n\n'
    return raw, 'text/event-stream; charset=utf-8'


def select_round(handler, directory, chat, native, optimization, redactor):
    selection = optimization.get('claude_capability_selection', {})
    if not selection.get('required'):
        return native
    original = chat['response_format']['json_schema']['schema']
    branches = original['$defs']['node']['anyOf']
    names = [b['properties']['cap']['enum'][0] for b in branches if 'cap' in b.get('properties', {})]
    schema = {'type': 'object', 'properties': {'capabilities': {'type': 'array', 'items': {'type': 'string', 'enum': names}}},
              'required': ['capabilities'], 'additionalProperties': False}
    selector = copy.deepcopy(native)
    selector.pop('tools', None)
    selector.pop('tool_choice', None)
    selector['stream'] = False
    selector['max_tokens'] = 8192
    selector['thinking'] = {'type': 'disabled'}
    selector['output_config'] = {'format': {'type': 'json_schema', 'schema': schema}}
    selector.setdefault('system', []).append({'type': 'text', 'text': (
        'Transport capability-selection stage ONLY. For this call, replace the RTDL envelope output format with '
        '{"capabilities":["exact catalog name"]}. Choose the capability needed for the NEXT executable plan, '
        'not every capability useful for the whole task. Choose at most ' + str(selection['limit']) + ' unique names. '
        'Choose [] only for legitimate waiting, task completion, clarification, or plan control. '
        'Every catalog capability will be available for selection again next round. Select from actual task state '
        'and feedback; do not invent observations. This selection itself executes no action.')})
    folder = directory / 'capability-selection'
    folder.mkdir()
    write_json(folder / 'request.json', {'path': handler.server.claude_path, 'body': selector}, redactor)
    upstream = handler.server.upstream
    cls = http.client.HTTPSConnection if upstream.scheme == 'https' else http.client.HTTPConnection
    conn = cls(upstream.hostname, upstream.port, timeout=90)
    auth = handler.headers.get('Authorization', '')
    key = auth.split(' ', 1)[1] if auth.startswith('Bearer ') else auth
    start = time.monotonic()
    try:
        conn.request('POST', handler.server.claude_path, json.dumps(selector, ensure_ascii=False).encode(),
                     {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01',
                      'Authorization': auth, 'x-api-key': key})
        response = conn.getresponse()
        raw = response.read()
        (folder / 'response.body').write_bytes(redactor.bytes(raw))
        selection['status'] = response.status
        selection['duration_s'] = round(time.monotonic()-start, 4)
        if response.status != 200:
            raise ValueError('Claude capability selection API rejected request')
        value = json.loads(raw)
        selection['stop_reason'] = value.get('stop_reason')
        selection['usage'] = value.get('usage')
        if value.get('stop_reason') != 'end_turn':
            raise ValueError('Claude capability selection did not complete: ' + str(value.get('stop_reason')))
        text = ''.join(p['text'] for p in value['content'] if p['type'] == 'text')
        result = json.loads(text)
        jsonschema.Draft202012Validator(schema).validate(result)
        chosen = result['capabilities']
        if len(chosen) > selection['limit'] or len(set(chosen)) != len(chosen):
            raise ValueError('Claude selected too many or duplicate capabilities')
        selection['selected'] = chosen
        selection['usage'] = value.get('usage')
        depth = getattr(handler.server, 'claude_plan_depth', 1)
        adapted, stats = adapt_schema(select_capabilities(original, chosen), depth, rtdl=True)
        native = copy.deepcopy(native)
        native['output_config']['format']['schema'] = adapted
        native.setdefault('system', []).append({'type': 'text', 'text': (
            'Transport planning stage: this round may use only the capabilities you selected: '
            + json.dumps(chosen) + '. Every catalog capability is available for selection next round. '
            'Emit only actions genuinely performed by these capabilities; do not substitute reading a manual '
            'for executing an action. Defer other actions to subsequent rounds and describe only the actions '
            'actually in this plan. The schema permits at most ' + str(depth) + ' nested sequence/parallel containers. '
            'Do not return an empty plan for unfinished work when there is no actual running plan to wait for.')})
        optimization['claude_schema'] = stats
        return native
    finally:
        conn.close()
        write_json(folder / 'summary.json', selection, redactor)


def forward(handler, directory, chat, native, optimization, redactor, request_id):
    upstream = handler.server.upstream
    cls = http.client.HTTPSConnection if upstream.scheme == 'https' else http.client.HTTPConnection
    conn = cls(upstream.hostname, upstream.port, timeout=180)
    started = time.monotonic()
    stream, headers_sent, delivered, complete = bool(chat.get('stream')), False, False, False
    status, error, schema_valid, first_byte = None, None, None, None
    analysis, parser = Analysis(), NativeStream()
    stream_id, stream_created = 'chatcmpl-monitor-' + request_id, int(time.time())
    result, validation_error = None, None
    forwarded = (directory / 'forwarded.response.body').open('wb')
    def send_chunk(data):
        forwarded.write(redactor.bytes(data))
        forwarded.flush()
        analysis.feed(data)
        handler.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')
        handler.wfile.flush()
    try:
        if stream:
            # Return headers immediately so the Pilot's stream-open deadline
            # does not include Claude grammar compilation/generation time.
            handler.send_response(200)
            handler.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            handler.send_header('Transfer-Encoding', 'chunked')
            handler.end_headers()
            headers_sent = True
            send_chunk(b'data: ' + json.dumps({'id': stream_id, 'created': stream_created, 'model': chat['model'],
                'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}).encode() + b'\n\n')
        native = select_round(handler, directory, chat, native, optimization, redactor)
        record = json.loads((directory / 'request.json').read_text())
        record.update(body=native, optimization=optimization, request_stage='planning')
        write_json(directory / 'request.json', record, redactor)
        authorization = handler.headers.get('Authorization', '')
        key = authorization.split(' ', 1)[1] if authorization.startswith('Bearer ') else authorization
        headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream' if stream else 'application/json',
                   'anthropic-version': '2023-06-01'}
        if key:
            headers['x-api-key'] = key
            headers['Authorization'] = authorization
        conn.request('POST', handler.server.claude_path, json.dumps(native, ensure_ascii=False).encode(), headers)
        response = conn.getresponse()
        status = response.status
        write_json(directory / 'response.json', {'status': status, 'path': handler.server.claude_path,
                   'headers_at': now()}, redactor)
        is_sse = 'text/event-stream' in response.getheader('Content-Type', '')
        raw = bytearray()
        with (directory / 'response.body').open('wb') as target:
            recorder = SafeStream(target, redactor)
            try:
                while data := response.read1(65536):
                    if first_byte is None:
                        first_byte = time.monotonic() - started
                    recorder.write(data)
                    if status == 200 and is_sse:
                        for thought in parser.feed(data):
                            if stream:
                                send_chunk(b'data: ' + json.dumps({'id': stream_id, 'created': stream_created, 'model': chat['model'], 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'reasoning_content': thought}}]}, ensure_ascii=False).encode() + b'\n\n')
                    else:
                        raw.extend(data)
                if response.length not in (None, 0):
                    raise http.client.HTTPException('Incomplete Claude response body')
            finally:
                recorder.write(b'', final=True)
        complete = True
        if status != 200:
            if stream:
                raise ValueError('Claude upstream rejected request; see original response')
            handler.send_response(status)
            handler.send_header('Content-Type', 'application/json')
            handler.send_header('Content-Length', str(len(raw)))
            handler.end_headers()
            headers_sent = True
            handler.wfile.write(raw)
            return
        if is_sse:
            parser.feed(b'', final=True)
            message = parser.completed()
        else:
            message = json.loads(raw)
        result = chat_response(message, chat['model'])
        if stream:
            result.update(id=stream_id, created=stream_created)
        schema_valid = validate_result(result, chat, native)
        payload, mime = encode_response(result, stream, omit_reasoning=stream and is_sse)
        if stream:
            send_chunk(payload)
            handler.wfile.write(b'0\r\n\r\n')
        else:
            forwarded.write(redactor.bytes(payload))
            forwarded.flush()
            handler.send_response(200)
            handler.send_header('Content-Type', mime)
            handler.send_header('Content-Length', str(len(payload)))
            handler.end_headers()
            headers_sent = True
            handler.wfile.write(payload)
            analysis.feed(encode_response(result, True)[0], final=True)
        handler.wfile.flush()
        delivered = True
    except Exception as exc:
        error = type(exc).__name__
        if isinstance(exc, jsonschema.ValidationError):
            schema_valid = False
            validation_error = {'path': list(exc.absolute_path), 'message': exc.message[:400]}
        elif isinstance(exc, ValueError) and result is not None and chat.get('response_format', {}).get('type') in ('json_schema', 'json_object'):
            schema_valid = False
        handler.close_connection = True
        if headers_sent:
            # Do not end chunked framing: the client must receive a transport
            # error, never a successful partial plan. No action text was sent.
            try:
                handler.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        else:
            try:
                handler.send_error(502, 'Claude Messages bridge failed; see monitor records')
            except OSError:
                pass
    finally:
        conn.close()
        forwarded.close()
        analysis.feed(b'', final=True)
        summary = dict(request_id=request_id, finished_at=now(), status=status,
                       duration_s=round(time.monotonic()-started, 4), first_byte_s=first_byte,
                       delivered=delivered, body_complete=complete, error_type=error, schema_valid=schema_valid,
                       transport='claude_messages', native_events=parser.events, validation_error=validation_error, optimization=optimization, **analysis.summary())
        write_json(directory / 'summary.json', summary, redactor)
        with handler.server.index_lock:
            with (handler.server.record_root / 'index.jsonl').open('a') as target:
                target.write(json.dumps(redactor.value(summary), ensure_ascii=False) + '\n')
