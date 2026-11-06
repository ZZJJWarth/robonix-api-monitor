"""Chat/Gemini native protocol bridge; never repairs or executes model plans."""
import base64
import copy
import http.client
import json
import time
from urllib.parse import quote

import jsonschema

from monitor import Analysis, now, write_json


CALL_PREFIX = 'call_gemini_'


def call_id(part):
    # Pilot preserves IDs but drops provider extension fields. Carry the exact
    # native functionCall/signature in its opaque ID so it survives restarts.
    raw = json.dumps(part, ensure_ascii=False, separators=(',', ':')).encode()
    return CALL_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip('=')


def original_call(identifier, function):
    if not identifier.startswith(CALL_PREFIX):
        raise ValueError('Gemini tool history requires its original signed call ID')
    token = identifier[len(CALL_PREFIX):]
    part = json.loads(base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)))
    native = part['functionCall']
    if native['name'] != function['name'] or native.get('args', {}) != json.loads(function['arguments']):
        raise ValueError('Gemini tool history no longer matches the original call')
    return part


def parts(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{'text': content}] if content else []
    result = []
    for item in content:
        if item['type'] == 'text':
            result.append({'text': item['text']})
        elif item['type'] == 'image_url':
            url = item['image_url']['url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise ValueError('Gemini bridge accepts inline base64 images only')
            mime, data = url[5:].split(';base64,', 1)
            result.append({'inlineData': {'mimeType': mime, 'data': data}})
        else:
            raise ValueError('Unsupported Gemini message content')
    return result


def build_request(body, default_thinking=None):
    supported = {'model', 'messages', 'stream', 'stream_options', 'response_format',
                 'tools', 'tool_choice', 'temperature', 'top_p', 'max_tokens',
                 'max_completion_tokens', 'stop', 'n', 'user', 'reasoning_effort',
                 'prompt_cache_key', 'prompt_cache_options'}
    if set(body) - supported or body.get('n', 1) != 1:
        raise ValueError('Unsupported Chat parameter in Gemini bridge')
    result, contents, instructions, calls = {}, [], [], {}
    for message in body['messages']:
        role = message['role']
        if role in ('system', 'developer'):
            instructions.extend(parts(message.get('content')))
            continue
        if role == 'tool':
            call = calls[message['tool_call_id']]['functionCall']
            output = message.get('content') or ''
            try:
                output = json.loads(output)
            except (ValueError, TypeError):
                pass
            response = {'name': call['name'], 'response': output if isinstance(output, dict) else {'output': output}}
            if 'id' in call:
                response['id'] = call['id']
            native_parts = [{'functionResponse': response}]
            native_role = 'user'
        elif role in ('user', 'assistant'):
            native_role = 'model' if role == 'assistant' else 'user'
            native_parts = parts(message.get('content'))
            for call in message.get('tool_calls') or []:
                native = original_call(call['id'], call['function'])
                calls[call['id']] = native
                native_parts.append(native)
        else:
            raise ValueError('Unsupported Gemini message role')
        if native_parts:
            # Parallel tool responses must be one user turn.
            if role == 'tool' and contents and contents[-1]['role'] == 'user' and all('functionResponse' in p for p in contents[-1]['parts']):
                contents[-1]['parts'].extend(native_parts)
            else:
                contents.append({'role': native_role, 'parts': native_parts})
    result['contents'] = contents
    if instructions:
        result['systemInstruction'] = {'parts': instructions}
    config = {}
    for source, target in [('temperature', 'temperature'), ('top_p', 'topP'),
                           ('max_tokens', 'maxOutputTokens'), ('max_completion_tokens', 'maxOutputTokens')]:
        if source in body:
            config[target] = body[source]
    if 'stop' in body:
        config['stopSequences'] = [body['stop']] if isinstance(body['stop'], str) else body['stop']
    shape = body.get('response_format') or {}
    if shape.get('type') == 'json_schema':
        config['responseMimeType'] = 'application/json'
        config['responseJsonSchema'] = copy.deepcopy(shape['json_schema']['schema'])
    elif shape.get('type') == 'json_object':
        config['responseMimeType'] = 'application/json'
    elif shape.get('type') not in (None, 'text'):
        raise ValueError('Unsupported response format')
    if 'reasoning_effort' in body or default_thinking is not None:
        effort = body.get('reasoning_effort', default_thinking)
        if effort not in ('none', 'minimal', 'low', 'medium', 'high'):
            raise ValueError('Unsupported Gemini thinking effort')
        config['thinkingConfig'] = {'thinkingLevel': 'minimal' if effort == 'none' else effort}
    if config:
        result['generationConfig'] = config
    declarations = []
    for tool in body.get('tools') or []:
        if tool.get('type') != 'function':
            raise ValueError('Unsupported Gemini tool type')
        function = tool['function']
        declarations.append({k: copy.deepcopy(function[k]) for k in ('name', 'description') if k in function} |
                            {'parametersJsonSchema': copy.deepcopy(function.get('parameters', {'type': 'object'}))})
    if declarations:
        result['tools'] = [{'functionDeclarations': declarations}]
    if 'tool_choice' in body:
        choice = body['tool_choice']
        if isinstance(choice, dict):
            tool_config = {'mode': 'ANY', 'allowedFunctionNames': [choice['function']['name']]}
        else:
            tool_config = {'mode': {'auto': 'AUTO', 'none': 'NONE', 'required': 'ANY'}[choice]}
        result['toolConfig'] = {'functionCallingConfig': tool_config}
    return result


def chat_response(native, model):
    if native.get('promptFeedback', {}).get('blockReason'):
        raise ValueError('Gemini blocked the prompt')
    candidates = native.get('candidates') or []
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Gemini response did not complete successfully')
    message, tool_calls, texts, thoughts = {'role': 'assistant'}, [], [], []
    for part in candidates[0].get('content', {}).get('parts', []):
        if 'text' in part:
            (thoughts if part.get('thought') else texts).append(part['text'])
        elif 'functionCall' in part:
            function = part['functionCall']
            tool_calls.append({'id': call_id(part), 'type': 'function', 'function': {
                'name': function['name'], 'arguments': json.dumps(function.get('args', {}), ensure_ascii=False, separators=(',', ':'))}})
        else:
            raise ValueError('Unsupported Gemini output part')
    message['content'] = ''.join(texts) or None
    if tool_calls:
        message['tool_calls'] = tool_calls
    if thoughts:
        message['reasoning_content'] = ''.join(thoughts)
    if not texts and not tool_calls:
        raise ValueError('Gemini returned no usable output')
    usage = native.get('usageMetadata') or {}
    thought_count = usage.get('thoughtsTokenCount', 0)
    result = {'id': native.get('responseId', 'gemini-response'), 'object': 'chat.completion',
              'created': int(time.time()), 'model': native.get('modelVersion', model),
              'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}],
              'usage': {'prompt_tokens': usage.get('promptTokenCount', 0),
                        'completion_tokens': usage.get('candidatesTokenCount', 0) + thought_count,
                        'total_tokens': usage.get('totalTokenCount', 0),
                        'completion_tokens_details': {'reasoning_tokens': thought_count}}}
    return result


def encode_response(result, stream):
    if not stream:
        return json.dumps(result, ensure_ascii=False).encode(), 'application/json'
    choice = result['choices'][0]
    delta = copy.deepcopy(choice['message'])
    for i, call in enumerate(delta.get('tool_calls', [])):
        call['index'] = i
    base = {k: result[k] for k in ('id', 'created', 'model')}
    base['object'] = 'chat.completion.chunk'
    events = [{**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
              {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}], 'usage': result['usage']}]
    return b''.join(b'data: ' + json.dumps(e, ensure_ascii=False).encode() + b'\n\n' for e in events) + b'data: [DONE]\n\n', 'text/event-stream; charset=utf-8'


def forward(handler, directory, chat, native, optimization, redactor, request_id):
    upstream = handler.server.upstream
    # --upstream is normally the OpenAI base; native Gemini is a sibling at the
    # same origin. An explicit --gemini-prefix supports other gateway layouts.
    model = chat['model'].removeprefix('models/')
    path = handler.server.gemini_prefix.rstrip('/') + '/models/' + quote(model, safe='') + ':generateContent'
    cls = http.client.HTTPSConnection if upstream.scheme == 'https' else http.client.HTTPConnection
    conn = cls(upstream.hostname, upstream.port, timeout=180)
    started = time.monotonic()
    status, error, schema_valid, delivered, complete = None, None, None, False, False
    analysis = Analysis()
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if handler.headers.get('Authorization'):
            headers['Authorization'] = handler.headers['Authorization']
        conn.request('POST', path, json.dumps(native, ensure_ascii=False).encode(), headers)
        response = conn.getresponse()
        status = response.status
        raw = response.read()
        if response.length not in (None, 0):
            raise http.client.HTTPException('Incomplete Gemini response')
        complete = True
        (directory / 'response.body').write_bytes(redactor.bytes(raw))
        write_json(directory / 'response.json', {'status': status, 'path': path, 'headers_at': now()}, redactor)
        if status != 200:
            handler.send_response(status)
            handler.send_header('Content-Type', 'application/json')
            handler.send_header('Content-Length', str(len(raw)))
            handler.end_headers()
            handler.wfile.write(raw)
            return
        result = chat_response(json.loads(raw), model)
        content = result['choices'][0]['message'].get('content')
        schema = (chat.get('response_format') or {}).get('json_schema', {}).get('schema')
        if schema is not None and content:
            try:
                schema_valid = not list(jsonschema.Draft202012Validator(schema).iter_errors(json.loads(content)))
            except ValueError:
                schema_valid = False
        payload, content_type = encode_response(result, chat.get('stream', False))
        (directory / 'forwarded.response.body').write_bytes(redactor.bytes(payload))
        analysis.feed(encode_response(result, True)[0], final=True)
        handler.send_response(200)
        handler.send_header('Content-Type', content_type)
        handler.send_header('Content-Length', str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        handler.wfile.flush()
        delivered = True
    except Exception as exc:
        error = type(exc).__name__
        handler.close_connection = True
        if not delivered:
            try:
                handler.send_error(502, 'Gemini native bridge failed; see monitor records')
            except OSError:
                pass
    finally:
        conn.close()
        summary = dict(request_id=request_id, finished_at=now(), status=status,
                       duration_s=round(time.monotonic() - started, 4), body_complete=complete,
                       delivered=delivered, error_type=error, schema_valid=schema_valid,
                       optimization=optimization, transport='gemini_native', **analysis.summary())
        write_json(directory / 'summary.json', summary, redactor)
        with handler.server.index_lock:
            with (handler.server.record_root / 'index.jsonl').open('a') as target:
                target.write(json.dumps(redactor.value(summary), ensure_ascii=False) + '\n')
