#!/usr/bin/env python3
"""Standalone DeepSeek schema probe. Returned plans are saved, never executed."""
import argparse
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
from replay import credentials
from replay_responses import inspect_stream


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--active-only', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    key = os.environ.get('VLM_API_KEY') or credentials()
    redactor = Redactor('Bearer ' + key)
    output = ROOT / 'responses-schema-probe' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, mode=0o700)
    print('Evidence:', output, flush=True)
    latest = ROOT / 'records/20261025-220727-4ac6f8e329/request.json'
    if args.active_only:
        latest = ROOT / 'records/20261025-220325-0af2e9736c/request.json'
    original = ROOT / 'records/20261025-175556-bcf93b9d15/request.json'
    captured = json.loads(latest.read_text())
    schema = captured['received_body']['response_format']['json_schema']['schema']
    jsonschema.Draft202012Validator.check_schema(schema)
    write_json(output / 'schema.json', schema, redactor)
    write_json(output / 'provenance.json', {
        'latest_source': str(latest), 'original_source': str(original),
        'schema_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        'endpoint': 'https://api.deepseek.com/responses',
        'model': 'deepseek-flash', 'reasoning_effort': 'none',
        'max_output_tokens': 8192, 'robot_execution': False,
        'production_changes': False,
    }, redactor)
    cases = []
    control_schema = {
        'type': 'object', 'properties': {'ok': {'type': 'boolean', 'enum': [True]}},
        'required': ['ok'], 'additionalProperties': False,
    }
    for trial in range(1, 4):
        cases.append((f'conflict-strict-{trial}', {
            'model': 'deepseek-flash', 'stream': True,
            'reasoning': {'effort': 'none'}, 'max_output_tokens': 256,
            'input': [{'role': 'user', 'content':
                       'Reply with exactly the plain text SCHEMA_IGNORED. Do not output JSON.'}],
            'text': {'format': {'type': 'json_schema', 'name': 'constraint_probe',
                                'strict': True, 'schema': control_schema}},
        }, control_schema))
    body = copy.deepcopy(cases[0][1])
    body['input'] = [{'role': 'user', 'content': 'Return JSON with the field ok set to true.'}]
    cases.append(('compliant-strict', body, control_schema))
    for group, source in [('original', original), ('latest', latest)]:
        record = json.loads(source.read_text())
        messages = record['body']['messages']
        body = {
            'model': record['body']['model'], 'input': messages, 'stream': True,
            'reasoning': {'effort': 'none'}, 'max_output_tokens': 8192,
            'text': {'format': {'type': 'json_schema', 'name': 'rtdl_envelope', 'schema': schema}},
        }
        assert body['input'] == record['body']['messages']
        for trial in range(1, 4):
            cases.append((f'{group}-rtdl-{trial}', copy.deepcopy(body), schema))
        if group == 'latest':
            strict = copy.deepcopy(body)
            strict['text']['format']['strict'] = True
            cases.append(('latest-rtdl-strict', strict, schema))
    if args.active_only:
        cases = [(name.replace('latest-', 'active-'), body, expected)
                 for name, body, expected in cases
                 if name.startswith('latest-rtdl-') and not name.endswith('strict')]
    rows = []
    for name, body, expected_schema in cases:
        write_json(output / f'{name}.request.json', body, redactor)
        started = time.monotonic()
        row = {'name': name}
        conn = http.client.HTTPSConnection('api.deepseek.com', timeout=30)
        chunks = []
        try:
            conn.request('POST', '/responses', json.dumps(body, ensure_ascii=False).encode(), {
                'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            })
            response = conn.getresponse()
            row['http_status'] = response.status
            with (output / f'{name}.response.body').open('wb') as target:
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
                row.update(inspect_stream(raw, expected_schema))
                content = ''
                final = None
                for line in raw.splitlines():
                    if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                        continue
                    event = json.loads(line[5:])
                    if event.get('type') == 'response.output_text.delta':
                        content += event.get('delta') or ''
                    if isinstance(event.get('response'), dict):
                        final = event['response']
                if not content and final:
                    content = ''.join(part.get('text', '') for item in final.get('output', [])
                                      if item.get('type') == 'message'
                                      for part in item.get('content', []) if part.get('type') == 'output_text')
                (output / f'{name}.content.txt').write_bytes(redactor.bytes(content.encode()))
                row['content_preview'] = content[:600]
                row['plain_conflict_text_returned'] = content.strip() == 'SCHEMA_IGNORED'
                if final:
                    row['echoed_text_format'] = (final.get('text') or {}).get('format')
            else:
                try:
                    row['api_error'] = json.loads(raw).get('error')
                except ValueError:
                    row['api_error'] = 'Non-JSON HTTP response; see evidence body.'
        except Exception as exc:
            row['client_error'] = type(exc).__name__
        finally:
            conn.close()
        row['duration_s'] = round(time.monotonic() - started, 3)
        rows.append(row)
        write_json(output / 'results.json', rows, redactor)
        print(json.dumps(redactor.value({k: row[k] for k in (
            'name', 'http_status', 'duration_s', 'response_status', 'completed_event_seen',
            'json_valid', 'schema_valid', 'whitespace_only', 'plain_conflict_text_returned',
            'api_error', 'client_error') if k in row}), ensure_ascii=False), flush=True)
        if row.get('http_status') in (401, 403, 429):
            print('Stopping on authentication or quota error.', flush=True)
            break
    print('Finished; no Pilot/Executor requests or robot actions.', flush=True)


if __name__ == '__main__':
    main()
