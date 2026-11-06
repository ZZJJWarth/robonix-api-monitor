#!/usr/bin/env python3
"""Replay captured model requests without sending any plan to Executor."""
import argparse
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent


def credentials():
    configured = os.environ.get('RBNX_ROBOT_DIR', '')
    robot = Path(configured).expanduser() if configured else None
    if robot is None or not robot.is_dir():
        raise RuntimeError('Set RBNX_ROBOT_DIR to the running robot workspace first.')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if ((process / 'comm').read_text().strip() != 'robonix-pilot'
                    or (process / 'cwd').resolve() != robot):
                continue
            args = [part.decode() for part in (process / 'cmdline').read_bytes().split(b'\0') if part]
            def arg(name):
                for index, part in enumerate(args):
                    if part == name and index + 1 < len(args):
                        return args[index + 1]
                    if part.startswith(name + '='):
                        return part.split('=', 1)[1]
                return None
            config = json.loads(arg('--config-json') or '{}')
            key = arg('--vlm-api-key') or config.get('vlm', {}).get('api_key')
            if key:
                return key
        except (OSError, ValueError):
            continue
    raise RuntimeError('Running robot Pilot credentials are unavailable; no key file was read.')


def inspect_response(data):
    content = ''
    reasoning_chars = 0
    finish_reasons = []
    usage = None
    upstream_id = None
    done = False
    for line in data.splitlines():
        if not line.startswith(b'data:'):
            continue
        payload = line[5:].strip()
        if payload == b'[DONE]':
            done = True
            continue
        event = json.loads(payload)
        upstream_id = upstream_id or event.get('id')
        usage = event.get('usage') or usage
        for choice in event.get('choices', []):
            if choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta', {})
            content += delta.get('content') or ''
            reasoning_chars += len(delta.get('reasoning_content') or delta.get('reasoning') or '')
            if choice.get('finish_reason') is not None:
                finish_reasons.append(choice['finish_reason'])
    try:
        obj = json.loads(content)
        json_valid = True
    except ValueError:
        obj = None
        json_valid = False
    fields = {'content', 'message_updates', 'rtdl_parent_task_id', 'rtdl_description', 'rtdl', 'task_updates'}
    envelope_keys_valid = isinstance(obj, dict) and set(obj) == fields
    return dict(upstream_id=upstream_id, content_chars=len(content),
                content_non_whitespace_chars=sum(not c.isspace() for c in content),
                whitespace_only=not content.strip(), reasoning_chars=reasoning_chars,
                finish_reasons=finish_reasons, done_marker_seen=done,
                json_valid=json_valid, envelope_keys_valid=envelope_keys_valid, usage=usage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-count', type=int, default=5)
    parser.add_argument('--retry-count', type=int, default=3)
    args = parser.parse_args()
    os.umask(0o077)
    key = credentials()
    output = ROOT / 'replays' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, mode=0o700)
    plans = [('original', '20261025-175556-c3c4dd2bac', args.original_count),
             ('retry', '20261025-175558-23ef9bcaac', args.retry_count)]
    results = []
    print('Replay evidence:', output, flush=True)
    for group, source_id, count in plans:
        source = ROOT / 'records' / source_id / 'request.json'
        body = json.loads(source.read_text())['body']
        encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        assert b'[REDACTED]' not in encoded
        digest = hashlib.sha256(encoded).hexdigest()
        (output / f'{group}.source.json').write_text(json.dumps(dict(
            source=str(source), request_sha256=digest, model=body['model'],
            response_format=body.get('response_format'),
            message_count=len(body['messages']), repeats=count), ensure_ascii=False, indent=2) + '\n')
        for trial in range(1, count + 1):
            started = time.monotonic()
            known = {p.name for p in (ROOT / 'records').iterdir() if p.is_dir()}
            c = http.client.HTTPConnection('127.0.0.1', 54090, timeout=180)
            try:
                c.request('POST', '/chat/completions', encoded, {
                    'Authorization': 'Bearer ' + key, 'Accept': 'text/event-stream',
                    'Content-Type': 'application/json',
                })
                response = c.getresponse()
                data = response.read()
                row = dict(group=group, trial=trial, status=response.status,
                           duration_s=round(time.monotonic() - started, 3),
                           request_sha256=digest)
                if response.status == 200:
                    row.update(inspect_response(data))
                else:
                    row['error'] = 'upstream_http_error'
                for path in (ROOT / 'records').iterdir():
                    if path.is_dir() and path.name not in known:
                        p = path / 'request.json'
                        if p.exists() and json.loads(p.read_text()).get('body') == body:
                            row['monitor_record'] = path.name
                            break
            except Exception as exc:
                row = dict(group=group, trial=trial, error=type(exc).__name__,
                           duration_s=round(time.monotonic() - started, 3))
            finally:
                c.close()
            results.append(row)
            (output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({k: row.get(k) for k in (
                'group', 'trial', 'status', 'whitespace_only', 'content_chars',
                'reasoning_chars', 'finish_reasons', 'json_valid', 'envelope_keys_valid',
                'duration_s', 'error') if k in row}, ensure_ascii=False), flush=True)
    print('Completed', len(results), 'independent API requests; no Executor calls.', flush=True)


if __name__ == '__main__':
    main()
