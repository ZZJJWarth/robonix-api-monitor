"""API-only Claude schema probes; credentials stay in environment/memory."""
import concurrent.futures
import copy
import datetime
import http.client
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import jsonschema


MODEL = 'claude-haiku-5-5'
SIMPLE_SCHEMA = {'type': 'object', 'properties': {'result': {'type': 'string', 'enum': ['SCHEMA_OK']}},
                 'required': ['result'], 'additionalProperties': False}


def make_request(protocol, prompt, schema=SIMPLE_SCHEMA):
    body = {'model': MODEL, 'max_tokens': 512, 'messages': [{'role': 'user', 'content': prompt}]}
    if schema is not None:
        if protocol == 'messages':
            body['output_config'] = {'format': {'type': 'json_schema', 'schema': copy.deepcopy(schema)}}
        else:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'probe', 'strict': True, 'schema': copy.deepcopy(schema)}}
    return body


def call(protocol, case, body, directory, schema=None, expected_reject=False):
    key = os.environ['VLM_API_KEY']
    upstream = urlsplit(os.environ.get('VLM_BASE_URL', 'https://api.gptsapi.net'))
    prefix = upstream.path.rstrip('/')
    path = prefix + '/chat/completions' if protocol == 'chat' else prefix + ('/messages' if prefix.endswith('/v1') else '/v1/messages')
    conn_cls = http.client.HTTPSConnection if upstream.scheme == 'https' else http.client.HTTPConnection
    conn = conn_cls(upstream.hostname, upstream.port, timeout=90)
    run = directory / f'{protocol}-{case}'
    run.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        raw = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
        (run / name).write_text(raw.replace(key, '[REDACTED]'))
    save('request.json', {'protocol': protocol, 'path': path, 'body': body})
    started = time.monotonic()
    status, raw, error, parsed, text = None, '', None, None, ''
    try:
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
        if protocol == 'messages':
            headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
        conn.request('POST', path, body=json.dumps(body, ensure_ascii=False).encode(), headers=headers)
        response = conn.getresponse()
        status = response.status
        raw = response.read().decode()
        save('response.body', raw)
        try:
            parsed = json.loads(raw)
            if protocol == 'messages':
                text = ''.join(p.get('text', '') for p in parsed.get('content', []) if p.get('type') == 'text')
            else:
                text = (parsed.get('choices') or [{}])[0].get('message', {}).get('content') or ''
        except (ValueError, TypeError, AttributeError):
            pass
    except Exception as exc:
        error = type(exc).__name__
    finally:
        conn.close()
    errors, valid = [], None
    if status == 200 and schema is not None and not expected_reject:
        try:
            value = json.loads(text)
            errors = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(value)]
            valid = not errors
        except (ValueError, jsonschema.SchemaError) as exc:
            errors = [type(exc).__name__]
            valid = False
    result = {'protocol': protocol, 'case': case, 'status': status, 'duration_s': round(time.monotonic()-started, 3),
              'error_type': error, 'schema_valid': valid, 'validation_errors': errors[:10],
              'expected_reject': expected_reject, 'model': parsed.get('model') if isinstance(parsed, dict) else None,
              'stop_reason': parsed.get('stop_reason') if isinstance(parsed, dict) else None,
              'text_preview': text[:1000], 'error_preview': raw[:1500] if status != 200 else None}
    save('summary.json', result)
    print(json.dumps(result, ensure_ascii=False).replace(key, '[REDACTED]'), flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['discovery', 'compliance'], required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('VLM_API_KEY'):
        raise SystemExit('VLM_API_KEY environment is required')
    os.umask(0o077)
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.phase == 'discovery':
        jobs = [(p, 'cooperative', make_request(p, 'Return the JSON object {"result":"SCHEMA_OK"}.'), SIMPLE_SCHEMA, False) for p in ['chat', 'messages']]
    else:
        discovery = json.loads((args.directory / 'discovery.json').read_text())
        protocols = [r['protocol'] for r in discovery if r['status'] == 200]
        if not protocols:
            raise SystemExit('No working schema interface discovered')
        cases = [
            ('plain_text_control', 'Reply with exactly 你好, without JSON or punctuation.', None, False),
            ('conflicting_plain_text_1', 'Reply with exactly 你好, without JSON or punctuation.', SIMPLE_SCHEMA, False),
            ('conflicting_plain_text_2', 'Reply with exactly 你好, without JSON or punctuation.', SIMPLE_SCHEMA, False),
            ('conflicting_fields', 'Return exactly {"result":7,"extra":true}. Do not change these fields.', SIMPLE_SCHEMA, False),
            ('invalid_schema', 'Reply with OK.', {'type': 'definitely_not_a_json_schema_type'}, True),
        ]
        jobs = [(p, name, make_request(p, prompt, schema), schema, reject) for p in protocols for name, prompt, schema, reject in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, p, name, body, args.directory, schema, reject) for p, name, body, schema, reject in jobs]
        results = [future.result() for future in futures]
    (args.directory / (args.phase + '.json')).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
