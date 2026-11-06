#!/usr/bin/env python3
"""Test the captured RTDL schema with Responses; never execute returned plans."""
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


ROOT = Path(__file__).resolve().parent


def inspect_stream(raw, schema):
    streamed_text = ''
    reasoning_chars = 0
    response = None
    event_counts = {}
    errors = []
    for line in raw.splitlines():
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if data == b'[DONE]':
            continue
        event = json.loads(data)
        kind = event.get('type', '')
        event_counts[kind] = event_counts.get(kind, 0) + 1
        if kind == 'response.output_text.delta':
            streamed_text += event.get('delta') or ''
        if 'reasoning' in kind and kind.endswith('.delta'):
            reasoning_chars += len(event.get('delta') or '')
        if kind in ('error', 'response.failed', 'response.incomplete'):
            errors.append(event)
        if isinstance(event.get('response'), dict):
            response = event['response']
    response = response or {}
    final_text = ''.join(
        part.get('text', '') for item in response.get('output') or []
        if item.get('type') == 'message' for part in item.get('content') or []
        if part.get('type') == 'output_text'
    )
    text = streamed_text or final_text
    try:
        obj = json.loads(text)
        json_valid = True
        schema_errors = [dict(path=list(e.absolute_path), message=e.message)
                         for e in jsonschema.Draft202012Validator(schema).iter_errors(obj)]
    except ValueError:
        json_valid = False
        schema_errors = []
    return dict(response_id=response.get('id'), response_status=response.get('status'),
                completed_event_seen='response.completed' in event_counts,
                event_counts=event_counts, content_chars=len(text),
                content_non_whitespace_chars=sum(not c.isspace() for c in text),
                whitespace_only=not text.strip(), reasoning_chars=reasoning_chars,
                json_valid=json_valid, schema_valid=json_valid and not schema_errors,
                schema_errors=schema_errors, stream_matches_final=(streamed_text == final_text),
                usage=response.get('usage'), stream_errors=errors)


def main():
    os.umask(0o077)
    key = credentials()
    redactor = Redactor('Bearer ' + key)
    root = ROOT / 'responses-replays' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    root.mkdir(parents=True, mode=0o700)
    format_source = ROOT / 'records/20261025-175556-bcf93b9d15/request.json'
    envelope = json.loads(format_source.read_text())['body']['response_format']['json_schema']
    schema = envelope['schema']
    jsonschema.Draft202012Validator.check_schema(schema)
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    write_json(root / 'schema.json', schema, redactor)
    print('Responses replay evidence:', root, flush=True)
    rows = []
    for group, source_id, count in [
        ('original', '20261025-175556-c3c4dd2bac', 5),
        ('retry', '20261025-175558-23ef9bcaac', 3),
    ]:
        source = ROOT / 'records' / source_id / 'request.json'
        original = json.loads(source.read_text())['body']
        body = dict(model=original['model'], input=original['messages'], stream=True,
                    text=dict(format=dict(type='json_schema', name=envelope['name'], schema=schema)))
        assert body['input'] == original['messages']
        assert body['text']['format']['schema'] == schema
        encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        write_json(root / f'{group}.request.json', body, redactor)
        write_json(root / f'{group}.source.json', dict(
            source=str(source), schema_source=str(format_source), schema_sha256=schema_hash,
            message_count=len(original['messages']), repeats=count,
            messages_sha256=hashlib.sha256(json.dumps(original['messages'], ensure_ascii=False).encode()).hexdigest(),
            converted_fields={'messages': 'input', 'response_format': 'text.format'},
            omitted_chat_only_fields=['stream_options', 'prompt_cache_key', 'prompt_cache_options'],
            sampling_parameters_added=False,
        ), redactor)
        for trial in range(1, count + 1):
            started = time.monotonic()
            path = root / f'{group}-{trial}.response.body'
            c = http.client.HTTPSConnection('api.deepseek.com', timeout=180)
            row = dict(group=group, trial=trial)
            try:
                c.request('POST', '/responses', encoded, {
                    'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                })
                response = c.getresponse()
                row['status'] = response.status
                chunks = []
                with path.open('wb') as target:
                    recorder = SafeStream(target, redactor)
                    try:
                        while data := response.read1(65536):
                            chunks.append(data)
                            recorder.write(data)
                    finally:
                        recorder.write(b'', final=True)
                raw = b''.join(chunks)
                row['response_body_bytes'] = len(raw)
                if response.status == 200:
                    row.update(inspect_stream(raw, schema))
                else:
                    try:
                        row['api_error'] = json.loads(raw).get('error')
                    except ValueError:
                        row['api_error'] = 'Non-JSON HTTP error; see response body.'
            except Exception as exc:
                row['client_error'] = type(exc).__name__
            finally:
                c.close()
            row['duration_s'] = round(time.monotonic() - started, 3)
            rows.append(row)
            write_json(root / 'results.json', rows, redactor)
            print(json.dumps(redactor.value({k: row.get(k) for k in [
                'group', 'trial', 'status', 'response_status', 'completed_event_seen',
                'whitespace_only', 'content_chars', 'reasoning_chars', 'json_valid',
                'schema_valid', 'schema_errors', 'duration_s', 'api_error', 'client_error',
            ] if k in row}), ensure_ascii=False), flush=True)
            if row.get('status') in (400, 401, 403, 404, 422):
                print('Stopping: this request was rejected; identical retries would not test generation.', flush=True)
                return
    print('Completed', len(rows), 'Responses requests; no Executor calls.', flush=True)


if __name__ == '__main__':
    main()
