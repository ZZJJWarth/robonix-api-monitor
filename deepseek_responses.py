"""External Chat/Responses bridge. It never repairs or executes model content."""
import copy
import http.client
import json
import time

import jsonschema

from monitor import Analysis, SafeStream, now, write_json


def content_parts(content, role):
    if isinstance(content, str):
        return content
    result = []
    for part in content or []:
        kind = part.get('type')
        if kind == 'text':
            result.append({'type': 'output_text' if role == 'assistant' else 'input_text',
                           'text': part['text']})
        elif kind == 'image_url':
            image = part['image_url']
            result.append({'type': 'input_image', 'image_url': image['url'],
                           **({'detail': image['detail']} if 'detail' in image else {})})
        else:
            raise ValueError('Unsupported Chat content part in Responses bridge')
    return result


def build_request(original, optimized):
    shape = original['response_format']['json_schema']
    if not isinstance(shape.get('schema'), dict):
        raise ValueError('Responses Schema mode requires the original JSON Schema')
    items = []
    for message in optimized['messages']:
        role = message['role']
        content = message.get('content')
        if role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                          'output': content_parts(content, role) if content is not None else ''})
            continue
        if content is not None:
            items.append({'role': role, 'content': content_parts(content, role)})
        for call in message.get('tool_calls') or []:
            function = call['function']
            items.append({'type': 'function_call', 'call_id': call['id'],
                          'name': function['name'], 'arguments': function['arguments']})
    result = {'model': original['model'], 'input': items, 'stream': True,
              'reasoning': {'effort': 'none'},
              'text': {'format': {'type': 'json_schema', 'name': shape['name'],
                                  'schema': copy.deepcopy(shape['schema'])}}}
    # Preserve explicit limits; otherwise use DeepSeek's non-thinking default.
    result['max_output_tokens'] = original.get('max_completion_tokens') or original.get('max_tokens') or 8192
    for field in ('temperature', 'top_p', 'user'):
        if field in original:
            result[field] = copy.deepcopy(original[field])
    if original.get('tools'):
        tools = []
        for tool in original['tools']:
            if tool.get('type') != 'function':
                raise ValueError('Unsupported tool type in Responses bridge')
            function = tool['function']
            tools.append({'type': 'function', **{k: copy.deepcopy(function[k])
                          for k in ('name', 'description', 'parameters') if k in function}})
        result['tools'] = tools
    if 'tool_choice' in original:
        choice = original['tool_choice']
        result['tool_choice'] = ({'type': 'function', 'name': choice['function']['name']}
                                 if isinstance(choice, dict) else choice)
    return result


class CompletedResponse:
    """Accept byte-split SSE; release content only after observed completion."""
    def __init__(self):
        self.pending = b''
        self.response = None
        self.terminal = None
        self.events = 0

    def feed(self, data, final=False):
        self.pending += data
        lines = self.pending.split(b'\n')
        self.pending = b'' if final else lines.pop()
        for line in lines:
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if not payload or payload == b'[DONE]':
                continue
            event = json.loads(payload)
            self.events += 1
            kind = event.get('type')
            if kind in ('response.failed', 'response.incomplete', 'error'):
                raise ValueError('Responses API returned an unsuccessful terminal event')
            if kind == 'response.completed':
                self.terminal = kind
                self.response = event['response']

    def completed(self):
        if self.terminal != 'response.completed' or not isinstance(self.response, dict):
            raise ValueError('Responses stream ended without a completed response')
        if self.response.get('status') != 'completed' or self.response.get('error'):
            raise ValueError('Responses API did not complete successfully')
        return self.response


def chat_stream(response):
    """Translate envelope/protocol only; output text and arguments stay exact."""
    base = {'id': response['id'], 'object': 'chat.completion.chunk',
            'created': response.get('created_at', 0), 'model': response.get('model', '')}
    chunks = []
    def emit(delta, finish=None, usage=None):
        value = {**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
        if usage is not None:
            value['usage'] = usage
        chunks.append(b'data: ' + json.dumps(value, ensure_ascii=False).encode() + b'\n\n')
    emit({'role': 'assistant'})
    content = ''
    tool_index = 0
    for item in response.get('output') or []:
        kind = item.get('type')
        if kind == 'message':
            for part in item.get('content') or []:
                if part.get('type') != 'output_text':
                    raise ValueError('Unsupported Responses message content')
                text = part['text']
                content += text
                emit({'content': text})
        elif kind == 'function_call':
            emit({'tool_calls': [{'index': tool_index, 'id': item['call_id'], 'type': 'function',
                                  'function': {'name': item['name'], 'arguments': item['arguments']}}]})
            tool_index += 1
        elif kind != 'reasoning':
            raise ValueError('Unsupported Responses output item')
    raw_usage = response.get('usage') or {}
    usage = {'prompt_tokens': raw_usage.get('input_tokens', 0),
             'completion_tokens': raw_usage.get('output_tokens', 0),
             'total_tokens': raw_usage.get('total_tokens', 0)}
    if 'input_tokens_details' in raw_usage:
        usage['prompt_tokens_details'] = raw_usage['input_tokens_details']
    if 'output_tokens_details' in raw_usage:
        usage['completion_tokens_details'] = raw_usage['output_tokens_details']
    emit({}, 'tool_calls' if tool_index else 'stop', usage)
    chunks.append(b'data: [DONE]\n\n')
    return b''.join(chunks), content


def forward(handler, directory, request_body, optimization, redactor, request_id):
    upstream = handler.server.upstream
    conn_type = http.client.HTTPSConnection if upstream.scheme == 'https' else http.client.HTTPConnection
    conn = conn_type(upstream.hostname, upstream.port, timeout=30)
    started = time.monotonic()
    status = None
    first_byte = None
    error = None
    body_complete = False
    delivered = False
    schema_valid = None
    analysis = Analysis()
    buffer = CompletedResponse()
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                   'Accept-Encoding': 'identity'}
        if handler.headers.get('Authorization'):
            headers['Authorization'] = handler.headers['Authorization']
        conn.request('POST', upstream.path.rstrip('/') + '/responses',
                     json.dumps(request_body, ensure_ascii=False).encode(), headers)
        response = conn.getresponse()
        status = response.status
        write_json(directory / 'response.json', {
            'status': status, 'headers_at': now(), 'path': '/responses',
            'headers': {k: v for k, v in response.getheaders()
                        if k.lower() in ('content-type', 'x-request-id', 'request-id', 'x-ds-request-id')},
        }, redactor)
        with (directory / 'response.body').open('wb') as target:
            recorder = SafeStream(target, redactor)
            try:
                while data := response.read1(65536):
                    if first_byte is None:
                        first_byte = time.monotonic() - started
                    recorder.write(data)
                    if status == 200:
                        buffer.feed(data)
                    if time.monotonic() - started > 180:
                        raise TimeoutError('Responses bridge request budget exceeded')
            finally:
                recorder.write(b'', final=True)
        body_complete = True
        if status != 200:
            raise ValueError('Responses API rejected request; see recorded response')
        buffer.feed(b'', final=True)
        result = buffer.completed()
        payload, content = chat_stream(result)
        try:
            value = json.loads(content)
            schema_valid = not list(jsonschema.Draft202012Validator(
                request_body['text']['format']['schema']).iter_errors(value))
        except ValueError:
            schema_valid = False
        (directory / 'forwarded.response.body').write_bytes(redactor.bytes(payload))
        analysis.feed(payload, final=True)
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream; charset=utf-8')
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
                # Do not trigger Pilot's schema-to-JSON compatibility fallback.
                handler.send_error(502, 'Responses Schema bridge failed; see monitor records')
            except OSError:
                pass
    finally:
        conn.close()
        summary = dict(request_id=request_id, finished_at=now(), status=status,
                       duration_s=round(time.monotonic() - started, 4), first_byte_s=first_byte,
                       body_complete=body_complete, delivered=delivered, error_type=error,
                       optimization=optimization, transport='responses_schema',
                       schema_valid=schema_valid, responses_events=buffer.events,
                       completed_event_seen=buffer.terminal == 'response.completed', **analysis.summary())
        write_json(directory / 'summary.json', summary, redactor)
        with handler.server.index_lock:
            with (handler.server.record_root / 'index.jsonl').open('a') as target:
                target.write(json.dumps(redactor.value(summary), ensure_ascii=False) + '\n')
