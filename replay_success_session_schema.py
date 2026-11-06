#!/usr/bin/env python3
"""Paired API replay from a successful robot session; never execute plans."""
import copy
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import time

import jsonschema

from monitor import Redactor, SafeStream, write_json
from replay import credentials, inspect_response
from replay_responses import inspect_stream


ROOT = Path(__file__).resolve().parent
SESSION = '566d33dd-4187-4e51-856b-f11d4b0de298'
SAMPLES = [
    ('depart', '出发前往三楼', '20261025-213135-a8a9ea3085'),
    ('search', '检查302后继续搜索', '20261025-213840-f05bbb0443'),
    ('withdraw', '405箱子确认有面包后取物', '20261025-220345-6bc7dcd677'),
    ('return', '取得面包后返回一楼', '20261025-220348-fb0423e541'),
    ('deposit', '一楼箱子确认可存放后投递', '20261025-220724-ee7ba127c1'),
    ('complete', '投递成功后确认完成', '20261025-220727-4ac6f8e329'),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def response_content(raw, mode):
    text = ''
    final = {}
    for line in raw.splitlines():
        if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
            continue
        event = json.loads(line[5:])
        if mode == 'responses_schema':
            if event.get('type') == 'response.output_text.delta':
                text += event.get('delta') or ''
            if isinstance(event.get('response'), dict):
                final = event['response']
        else:
            for choice in event.get('choices', []):
                if choice.get('index', 0) == 0:
                    text += (choice.get('delta') or choice.get('message') or {}).get('content') or ''
    if not text and final:
        text = ''.join(part.get('text', '') for item in final.get('output', [])
                       if item.get('type') == 'message'
                       for part in item.get('content', []) if part.get('type') == 'output_text')
    return text


def analyze(raw, mode, schema):
    row = inspect_stream(raw, schema) if mode == 'responses_schema' else inspect_response(raw)
    content = response_content(raw, mode)
    row['content_preview'] = content[:500]
    try:
        value = json.loads(content)
    except ValueError:
        row.update(json_valid=False, schema_valid=False, schema_errors=[])
        return row, content
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
    row.update(json_valid=True, schema_valid=not errors,
               schema_errors=[{'path': list(e.absolute_path), 'message': e.message} for e in errors])
    if isinstance(value, dict):
        row['task_updates'] = value.get('task_updates')
        leaves = []
        def walk(node):
            if not isinstance(node, dict):
                return
            if node.get('op') == 'do':
                leaves.append({'cap': node.get('cap'), 'args': node.get('args')})
            for child in node.get('children') or []:
                walk(child)
        walk(value.get('rtdl'))
        row['action_leaves'] = leaves
    return row, content


def main():
    os.umask(0o077)
    key = os.environ.get('VLM_API_KEY') or credentials()
    redactor = Redactor('Bearer ' + key)
    output = ROOT / 'successful-session-schema' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, mode=0o700)
    print('Evidence:', output, flush=True)
    configured = os.environ.get('RBNX_ROBOT_DIR', '')
    robot = Path(configured).expanduser() if configured else None
    if robot is None or not robot.is_dir():
        raise RuntimeError('Set RBNX_ROBOT_DIR to the robot workspace that recorded the session.')
    session = robot / 'rbnx-boot/sessions' / SESSION / 'transcript.jsonl'
    final = None
    for line in session.read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') == 'task_state' and any(
            t.get('status') == 'done' for t in event.get('task', {}).get('tasks', [])
        ):
            final = event
    if not final:
        raise RuntimeError('Selected session has no recorded completed task.')
    write_json(output / 'successful-session.json', {'session_id': SESSION,
               'transcript_path': str(session), 'final_task_state': final}, redactor)
    provenance = []
    requests = {}
    schema = None
    baselines = []
    for name, label, source_id in SAMPLES:
        folder = ROOT / 'records' / source_id
        capture = json.loads((folder / 'request.json').read_text())
        chat = copy.deepcopy(capture['body'])
        original = capture.get('received_body', capture['body'])
        candidate = original['response_format']['json_schema']['schema']
        if schema is None:
            schema = candidate
            jsonschema.Draft202012Validator.check_schema(schema)
        assert candidate == schema
        assert chat['response_format'] == {'type': 'json_object'}
        assert chat.get('thinking') == {'type': 'disabled'}
        assert not chat.get('tools')
        assert chat['model'] == 'deepseek-flash'
        for field in ('prompt_cache_key', 'prompt_cache_options'):
            chat.pop(field, None)
        chat['max_tokens'] = 8192
        responses = {
            'model': chat['model'], 'input': copy.deepcopy(chat['messages']), 'stream': True,
            'reasoning': {'effort': 'none'}, 'max_output_tokens': 8192,
            'text': {'format': {'type': 'json_schema', 'name': 'rtdl_envelope', 'schema': schema}},
        }
        assert responses['input'] == chat['messages']
        provenance.append({'name': name, 'label': label, 'source': str(folder / 'request.json'),
                           'message_count': len(chat['messages']), 'messages_sha256': digest(chat['messages']),
                           'schema_sha256': digest(schema), 'messages_modified': False})
        for mode, path, body in [('chat_json', '/chat/completions', chat),
                                 ('responses_schema', '/responses', responses)]:
            requests[(name, mode)] = (path, json.dumps(body, ensure_ascii=False).encode())
            write_json(output / f'{name}.{mode}.request.json', body, redactor)
        baseline, _ = analyze((folder / 'response.body').read_bytes(), 'chat_json', schema)
        baselines.append({'name': name, **baseline})
    write_json(output / 'provenance.json', provenance, redactor)
    write_json(output / 'schema.json', schema, redactor)
    write_json(output / 'logged-baseline-results.json', baselines, redactor)
    rows = []
    for trial in range(1, 4):
        modes = ('responses_schema', 'chat_json') if trial % 2 else ('chat_json', 'responses_schema')
        for name, label, _ in SAMPLES:
            for mode in modes:
                path, encoded = requests[(name, mode)]
                started = time.monotonic()
                row = {'sample': name, 'label': label, 'mode': mode, 'trial': trial}
                stem = f'{name}.{mode}.{trial}'
                conn = http.client.HTTPSConnection('api.deepseek.com', timeout=30)
                try:
                    conn.request('POST', path, encoded, {'Authorization': 'Bearer ' + key,
                                 'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
                    response = conn.getresponse()
                    row['http_status'] = response.status
                    chunks = []
                    with (output / f'{stem}.response.body').open('wb') as target:
                        recorder = SafeStream(target, redactor)
                        try:
                            while data := response.read1(65536):
                                chunks.append(data)
                                recorder.write(data)
                                if time.monotonic() - started > 150:
                                    raise TimeoutError('total request budget exceeded')
                        finally:
                            recorder.write(b'', final=True)
                    raw = b''.join(chunks)
                    if response.status == 200:
                        analysis, content = analyze(raw, mode, schema)
                        row.update(analysis)
                        (output / f'{stem}.content.txt').write_bytes(redactor.bytes(content.encode()))
                    else:
                        try:
                            row['api_error'] = json.loads(raw).get('error')
                        except ValueError:
                            row['api_error'] = 'Non-JSON HTTP error; see response body.'
                except Exception as exc:
                    row['client_error'] = type(exc).__name__
                finally:
                    conn.close()
                row['duration_s'] = round(time.monotonic() - started, 3)
                rows.append(row)
                write_json(output / 'results.json', rows, redactor)
                print(json.dumps(redactor.value({k: row[k] for k in (
                    'sample', 'mode', 'trial', 'http_status', 'schema_valid', 'whitespace_only',
                    'duration_s', 'api_error', 'client_error') if k in row}), ensure_ascii=False), flush=True)
                if row.get('http_status') in (401, 403, 429):
                    print('Stopped on authentication or quota error.', flush=True)
                    return
    print('Completed 36 independent API replays; no plans executed.', flush=True)


if __name__ == '__main__':
    main()
