#!/usr/bin/env python3
"""Replay one observed decision with thinking off/on; never execute plans."""
import copy
import datetime
import http.client
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import jsonschema

from monitor import Handler, ThreadingHTTPServer
from replay import credentials, inspect_response


def main():
    os.umask(0o077)
    root = Path(__file__).resolve().parent
    source = root / 'records/20261025-195120-ec0ddd3fda/request.json'
    captured = json.loads(source.read_text())
    baseline = captured['body']
    schema = captured['received_body']['response_format']['json_schema']['schema']
    key = credentials()
    output = root / 'thinking-comparison' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, mode=0o700)
    records = output / 'records'
    records.mkdir(mode=0o700)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.upstream = urlsplit('https://api.deepseek.com')
    server.record_root = records
    server.index_lock = threading.Lock()
    server.deepseek_rtdl = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = []
    print('Evidence:', output, flush=True)
    try:
        for trial in range(1, 4):
            for mode in ('disabled', 'enabled'):
                body = copy.deepcopy(baseline)
                body['thinking'] = {'type': mode}
                conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=180)
                started = time.monotonic()
                row = {'trial': trial, 'thinking': mode}
                try:
                    conn.request('POST', '/chat/completions',
                                 json.dumps(body, ensure_ascii=False).encode(),
                                 {'Authorization': 'Bearer ' + key,
                                  'Content-Type': 'application/json',
                                  'Accept': 'text/event-stream'})
                    response = conn.getresponse()
                    raw = response.read()
                    row['status'] = response.status
                    if response.status == 200:
                        row.update(inspect_response(raw))
                        parts = []
                        for line in raw.splitlines():
                            if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                                continue
                            event = json.loads(line[5:])
                            for choice in event.get('choices', []):
                                text = (choice.get('delta') or {}).get('content')
                                if isinstance(text, str):
                                    parts.append(text)
                        content = ''.join(parts)
                        (output / f'{trial}-{mode}.content.txt').write_text(content)
                        try:
                            parsed = json.loads(content)
                            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(parsed))
                            row['schema_valid'] = not errors
                            row['schema_errors'] = [e.message[:250] for e in errors[:3]]
                            row['content'] = parsed.get('content') if isinstance(parsed, dict) else None
                            row['rtdl'] = parsed.get('rtdl') if isinstance(parsed, dict) else None
                        except ValueError:
                            row['schema_valid'] = False
                    else:
                        row['error'] = 'upstream_http_error'
                except Exception as exc:
                    row['error'] = type(exc).__name__
                finally:
                    conn.close()
                row['duration_s'] = round(time.monotonic() - started, 3)
                results.append(row)
                (output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
                print(json.dumps({k: row[k] for k in ('trial', 'thinking', 'status', 'duration_s',
                                 'reasoning_chars', 'whitespace_only', 'schema_valid', 'content', 'error')
                                  if k in row}, ensure_ascii=False), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print('Finished independent API replay. No Executor or robot actions.', flush=True)


if __name__ == '__main__':
    main()
