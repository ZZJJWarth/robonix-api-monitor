#!/usr/bin/env python3
"""Loopback Chat Completions recorder; forwards response bytes as they arrive."""
import argparse
import datetime
import http.client
import json
import os
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlsplit
import uuid

from message_content import text_content
from observation_hints import DEFAULT_DOMAIN, DomainContext


def now():
    return datetime.datetime.now().astimezone().isoformat()


class Redactor:
    def __init__(self, authorization):
        self.secrets = []
        if authorization:
            self.secrets.append(authorization.encode())
            parts = authorization.split(None, 1)
            if len(parts) == 2 and parts[1]:
                self.secrets.append(parts[1].encode())
        self.secrets.sort(key=len, reverse=True)

    def bytes(self, data):
        for secret in self.secrets:
            data = data.replace(secret, b'[REDACTED]')
        return data

    def value(self, value):
        if isinstance(value, dict):
            return {
                key: '[REDACTED]' if re.fullmatch(
                    r'(?:authorization|api[_-]?key|access[_-]?token|password|secret)',
                    key, re.I,
                ) else self.value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.bytes(value.encode()).decode()
        return value


class SafeStream:
    """Hold a secret-length suffix so split credentials cannot reach disk."""
    def __init__(self, target, redactor):
        self.target = target
        self.redactor = redactor
        self.pending = b''
        self.keep = max(map(len, redactor.secrets), default=1) - 1

    def write(self, data, final=False):
        self.pending += data
        boundary = len(self.pending) if final else max(0, len(self.pending) - self.keep)
        for secret in self.redactor.secrets:
            start = self.pending.find(secret)
            while start >= 0:
                if start < boundary < start + len(secret):
                    boundary = start
                start = self.pending.find(secret, start + 1)
        self.target.write(self.redactor.bytes(self.pending[:boundary]))
        self.target.flush()
        self.pending = self.pending[boundary:]


class Analysis:
    def __init__(self):
        self.pending = b''
        self.choices = {}
        self.usage = None
        self.model = None
        self.done = False
        self.events = 0
        self.invalid_events = 0

    def feed(self, data, final=False):
        self.pending += data
        lines = self.pending.split(b'\n')
        self.pending = b'' if final else lines.pop()
        for line in lines:
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                self.done = True
                continue
            try:
                event = json.loads(payload)
                if not isinstance(event, dict):
                    raise ValueError('non-object event')
            except (ValueError, UnicodeError):
                self.invalid_events += 1
                continue
            self.events += 1
            if event.get('usage') is not None:
                self.usage = event['usage']
            if event.get('model'):
                self.model = event['model']
            for choice in event.get('choices') or []:
                state = self.choices.setdefault(str(choice.get('index', 0)), {
                    'content_chars': 0, 'content_non_whitespace_chars': 0,
                    'reasoning_chars': 0, 'tool_call_deltas': 0,
                    'finish_reasons': [], 'delta_fields': [],
                })
                delta = choice.get('delta') or choice.get('message') or {}
                state['delta_fields'] = sorted(set(state['delta_fields']) | set(delta))
                content = delta.get('content')
                if isinstance(content, str):
                    state['content_chars'] += len(content)
                    state['content_non_whitespace_chars'] += sum(not c.isspace() for c in content)
                for field in ('reasoning_content', 'reasoning'):
                    if isinstance(delta.get(field), str):
                        state['reasoning_chars'] += len(delta[field])
                state['tool_call_deltas'] += len(delta.get('tool_calls') or [])
                if choice.get('finish_reason') is not None:
                    state['finish_reasons'].append(choice['finish_reason'])

    def summary(self):
        return dict(model=self.model, usage=self.usage, choices=self.choices,
                    sse_events=self.events, invalid_sse_events=self.invalid_events,
                    done_marker_seen=self.done)


def write_json(path, value, redactor):
    path.write_text(json.dumps(redactor.value(value), ensure_ascii=False, indent=2) + '\n')


def is_pilot_request(body):
    """Limit model-independent history compaction to recognizable Pilot requests."""
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return False
    return any(isinstance(m, dict) and m.get('role') == 'system'
               and (text := text_content(m.get('content'))) is not None
               and 'You are the Robonix Pilot' in text
               and '## RTDL output' in text for m in body['messages'])


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == '/healthz':
            profile = getattr(self.server, 'profile', None)
            data = (json.dumps({'ok': True, 'recorder': 'rbnx-api-monitor',
                                'profile': getattr(profile, 'name', 'generic'),
                                'profile_source': getattr(profile, 'source', None),
                                'empty_slot_items': list(getattr(profile, 'empty_slot_items', ())),
                                'inventory_schema': dict(getattr(profile, 'inventory_schema', {})),
                                'claude_guidance': bool(getattr(profile, 'claude_guidance', None)),
                                'claude_guidance_version': getattr(profile, 'claude_guidance_version', None),
                                'rtdl_domain_reminder': bool(getattr(profile, 'rtdl_domain_reminder', None)),
                                'deepseek_rtdl': getattr(self.server, 'deepseek_rtdl', False),
                                'responses_schema': getattr(self.server, 'responses_schema', False),
                                'context_compaction': getattr(self.server, 'compact_context', False),
                                'empty_observation_hint': getattr(self.server, 'empty_observation_hint', False),
                                'uninspected_object_hint': getattr(self.server, 'uninspected_object_hint', False),
                                'explore_followup_hint': getattr(self.server, 'explore_followup_hint', False),
                                'embedded_error_hint': getattr(self.server, 'embedded_error_hint', False),
                                'holding_delivery_hint': getattr(self.server, 'holding_delivery_hint', False),
                                'repeat_failure_hint': getattr(self.server, 'repeat_failure_hint', False),
                                'refine_rtdl_schema': getattr(self.server, 'refine_rtdl_schema', False),
                                'provider': getattr(self.server, 'provider', 'custom'),
                                'claude_schema_mode': getattr(self.server, 'claude_schema_mode', 'structured'),
                                'claude_plan_depth': getattr(self.server, 'claude_plan_depth', None),
                                'claude_schema_cap_limit': getattr(self.server, 'claude_schema_cap_limit', None),
                                'upstream': self.server.upstream.geturl()}) + '\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        path = urlsplit(self.path)
        if (path.query or path.fragment or not (
            self.command == 'POST' and path.path == '/chat/completions'
            or self.command == 'GET' and (
                path.path == '/models' or path.path.startswith('/models/')
            )
        )):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.send_error(400)
            return
        if length < 0 or length > 32 * 1024 * 1024 or self.headers.get('Transfer-Encoding'):
            self.send_error(413)
            return
        body = self.rfile.read(length)
        redactor = Redactor(self.headers.get('Authorization', ''))
        request_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:10]
        directory = self.server.record_root / request_id
        directory.mkdir(mode=0o700)
        try:
            request_body = json.loads(body) if body else None
        except ValueError:
            request_body = {'invalid_json': True, 'body_bytes': len(body)}
        received_body = request_body
        profile = getattr(self.server, 'profile', None)
        domain = getattr(self.server, 'domain', None)
        if domain is None and profile is not None:
            domain = DomainContext(profile.contracts, profile.empty_slot_items,
                                   profile.hint_notes, profile.inventory_schema)
        domain = domain or DEFAULT_DOMAIN
        refinement = None
        if path.path == '/chat/completions' and getattr(self.server, 'refine_rtdl_schema', False):
            from rtdl_schema import refine_request, SchemaRefinementError
            try:
                request_body, refinement = refine_request(request_body)
            except SchemaRefinementError as error:
                refinement = {'enabled': True, 'applied': False, 'error': str(error)}
                write_json(directory / 'request.json', {
                    'started_at': now(), 'method': self.command, 'path': path.path,
                    'body': received_body, 'optimization': {'schema_refinement': refinement}}, redactor)
                write_json(directory / 'summary.json', {
                    'request_id': request_id, 'finished_at': now(), 'status': 400,
                    'delivered': False, 'error_type': 'SchemaRefinementError',
                    'optimization': {'schema_refinement': refinement}}, redactor)
                self.send_error(400, 'RTDL schema refinement failed; see monitor request record')
                return
        schema_source = request_body
        optimization = {'applied': False}
        if (path.path == '/chat/completions'
                and getattr(self.server, 'deepseek_rtdl', False)):
            from deepseek_rtdl import optimize_request
            request_body, optimization = optimize_request(
                request_body, domain_reminder=getattr(profile, 'rtdl_domain_reminder', None))
        if refinement is not None:
            optimization['schema_refinement'] = refinement
        if (path.path == '/chat/completions'
                and getattr(self.server, 'compact_context', False)
                and is_pilot_request(request_body)):
            from context_compaction import compact_request
            request_body, compaction = compact_request(
                request_body,
                empty_slot_items=getattr(profile, 'empty_slot_items', None),
                inventory_schema=getattr(profile, 'inventory_schema', None))
            optimization['context_compaction'] = compaction
        if (path.path == '/chat/completions'
                and getattr(self.server, 'embedded_error_hint', False)
                and (profile is None or profile.hint_enabled('embedded_error_hint'))):
            from observation_hints import add_embedded_error_notes
            request_body, hints = add_embedded_error_notes(request_body, domain=domain)
            if hints['embedded_error_notes']:
                optimization['embedded_error_hint'] = hints
        if (path.path == '/chat/completions'
                and getattr(self.server, 'repeat_failure_hint', False)
                and (profile is None or profile.hint_enabled('repeat_failure_hint'))):
            from observation_hints import add_repeat_failure_notes
            request_body, hints = add_repeat_failure_notes(request_body, domain=domain)
            if hints['repeat_failure_notes']:
                optimization['repeat_failure_hint'] = hints
        if (path.path == '/chat/completions'
                and getattr(self.server, 'empty_observation_hint', False)
                and (profile is None or profile.hint_enabled('empty_observation_hint'))):
            from observation_hints import add_empty_observation_notes
            request_body, hints = add_empty_observation_notes(request_body, domain=domain)
            if hints['empty_observation_notes']:
                optimization['empty_observation_hint'] = hints
        if (path.path == '/chat/completions'
                and getattr(self.server, 'uninspected_object_hint', False)
                and (profile is None or profile.hint_enabled('uninspected_object_hint'))):
            from observation_hints import add_uninspected_object_notes
            request_body, hints = add_uninspected_object_notes(request_body, domain=domain)
            if hints['uninspected_object_notes']:
                optimization['uninspected_object_hint'] = hints
        if (path.path == '/chat/completions'
                and getattr(self.server, 'explore_followup_hint', False)
                and (profile is None or profile.hint_enabled('explore_followup_hint'))):
            from observation_hints import add_explore_followup_notes
            request_body, hints = add_explore_followup_notes(request_body, domain=domain)
            if hints['explore_followup_notes']:
                optimization['explore_followup_hint'] = hints
        if (path.path == '/chat/completions'
                and getattr(self.server, 'holding_delivery_hint', False)
                and (profile is None or profile.hint_enabled('holding_delivery_hint'))):
            from observation_hints import add_holding_delivery_notes
            request_body, hints = add_holding_delivery_notes(request_body, domain=domain)
            if hints['holding_delivery_notes']:
                optimization['holding_delivery_hint'] = hints
        modified = (optimization['applied'] or optimization.get('context_compaction', {}).get('applied', False)
                    or optimization.get('empty_observation_hint', {}).get('empty_observation_notes', 0)
                    or optimization.get('uninspected_object_hint', {}).get('uninspected_object_notes', 0)
                    or optimization.get('explore_followup_hint', {}).get('explore_followup_notes', 0)
                    or optimization.get('embedded_error_hint', {}).get('embedded_error_notes', 0)
                    or optimization.get('repeat_failure_hint', {}).get('repeat_failure_notes', 0)
                    or optimization.get('holding_delivery_hint', {}).get('holding_delivery_notes', 0)
                    or optimization.get('schema_refinement', {}).get('applied', False))
        if modified:
            body = json.dumps(request_body, ensure_ascii=False, separators=(',', ':')).encode()
        bridge = (getattr(self.server, 'responses_schema', False)
                  and optimization['applied'])
        if bridge:
            from deepseek_responses import build_request
            try:
                request_body = build_request(schema_source, request_body)
                optimization = {**optimization, 'response_format': 'json_schema',
                                'transport': 'responses_schema'}
            except (KeyError, TypeError, ValueError):
                self.send_error(502, 'Responses Schema mode requires a valid original schema request')
                return
        gemini = (getattr(self.server, 'provider', '') in ('gemini', 'auto')
                  and path.path == '/chat/completions' and isinstance(request_body, dict)
                  and str(request_body.get('model', '')).removeprefix('models/').lower().startswith('gemini'))
        claude = (getattr(self.server, 'provider', '') in ('claude', 'auto')
                  and path.path == '/chat/completions' and isinstance(request_body, dict)
                  and str(request_body.get('model', '')).lower().startswith('claude'))
        chat_request = request_body
        if gemini:
            from gemini_native import build_request
            try:
                request_body = build_request(chat_request, default_thinking=getattr(self.server, 'gemini_thinking_level', None))
                optimization = {**optimization, 'transport': 'gemini_native',
                                'untranslated_cache_controls': [k for k in ('prompt_cache_key', 'prompt_cache_options') if k in chat_request]}
            except (KeyError, TypeError, ValueError):
                self.send_error(400, 'Unsupported Chat request for Gemini native bridge')
                return
        if claude:
            from claude_native import build_request
            try:
                request_body, metadata = build_request(chat_request,
                    plan_depth=getattr(self.server, 'claude_plan_depth', 1),
                    cap_limit=getattr(self.server, 'claude_schema_cap_limit', 1),
                    schema_mode=getattr(self.server, 'claude_schema_mode', 'structured'),
                    guidance=getattr(profile, 'claude_guidance', None),
                    guidance_version=getattr(profile, 'claude_guidance_version', None))
                optimization = {**optimization, **metadata}
            except (KeyError, TypeError, ValueError) as error:
                optimization['claude_request_error'] = str(error)[:600]
                write_json(directory / 'request.json', {'started_at': now(), 'path': path.path,
                    'body': received_body, 'optimization': optimization}, redactor)
                write_json(directory / 'summary.json', {'request_id': request_id, 'finished_at': now(),
                    'status': 400, 'delivered': False, 'error_type': 'ClaudeRequestError', 'optimization': optimization}, redactor)
                self.send_error(400, 'Unsupported Claude request; see monitor record')
                return
        write_json(directory / 'request.json', {
            'started_at': now(), 'method': self.command, 'path': path.path,
            **({'upstream_path': '/responses'} if bridge else
               {'upstream_path': self.server.gemini_prefix.rstrip('/') + '/models/' +
                quote(str(chat_request['model']).removeprefix('models/'), safe='') + ':generateContent'} if gemini else
               {'upstream_path': self.server.claude_path, 'request_stage': 'prepared'} if claude else {}),
            'body': request_body, 'optimization': optimization,
            **({'received_body': received_body} if modified or gemini or claude else {}),
        }, redactor)
        if bridge:
            from deepseek_responses import forward
            forward(self, directory, request_body, optimization, redactor, request_id)
            return
        if gemini:
            from gemini_native import forward
            forward(self, directory, chat_request, request_body, optimization, redactor, request_id)
            return
        if claude:
            from claude_native import forward
            forward(self, directory, chat_request, request_body, optimization, redactor, request_id)
            return
        upstream = self.server.upstream
        cls = http.client.HTTPSConnection if upstream.scheme == 'https' else http.client.HTTPConnection
        conn = cls(upstream.hostname, upstream.port, timeout=30)
        headers = {'Content-Type': self.headers.get('Content-Type', 'application/json'),
                   'Accept': self.headers.get('Accept', 'application/json'),
                   'Accept-Encoding': 'identity'}
        if self.headers.get('Authorization'):
            headers['Authorization'] = self.headers['Authorization']
        analysis = Analysis()
        started = time.monotonic()
        total_bytes = 0
        status = None
        error_type = None
        sent_headers = False
        complete = False
        first_byte_s = None
        try:
            conn.connect()
            conn.sock.settimeout(600)
            conn.request(self.command, upstream.path.rstrip('/') + path.path,
                         body=body if length else None, headers=headers)
            response = conn.getresponse()
            status = response.status
            response_headers = {k: v for k, v in response.getheaders() if k.lower() in {
                'content-type', 'content-encoding', 'x-request-id', 'request-id',
                'x-ds-request-id', 'cf-ray', 'openai-processing-ms',
            }}
            write_json(directory / 'response.json', {
                'status': status, 'headers': response_headers,
                'headers_at': now(),
            }, redactor)
            self.send_response(status)
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            sent_headers = True
            with (directory / 'response.body').open('wb') as target:
                safe_stream = SafeStream(target, redactor)
                try:
                    while data := response.read1(65536):
                        if first_byte_s is None:
                            first_byte_s = time.monotonic() - started
                        total_bytes += len(data)
                        safe_stream.write(data)
                        if 'text/event-stream' in response.getheader('Content-Type', ''):
                            analysis.feed(data)
                        self.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')
                        self.wfile.flush()
                    if response.length not in (None, 0):
                        raise http.client.HTTPException('Incomplete upstream body')
                    complete = True
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                finally:
                    safe_stream.write(b'', final=True)
        except Exception as exc:
            error_type = type(exc).__name__
            self.close_connection = True
            if not sent_headers:
                try:
                    self.send_error(502, 'Monitor could not complete upstream request')
                except OSError:
                    pass
        finally:
            conn.close()
            analysis.feed(b'', final=True)
            summary = dict(request_id=request_id, finished_at=now(), status=status,
                           duration_s=round(time.monotonic() - started, 4),
                           first_byte_s=first_byte_s, response_body_bytes=total_bytes,
                           body_complete=complete, error_type=error_type,
                           optimization=optimization,
                           **analysis.summary())
            write_json(directory / 'summary.json', summary, redactor)
            with self.server.index_lock:
                with (self.server.record_root / 'index.jsonl').open('a') as target:
                    target.write(json.dumps(redactor.value(summary), ensure_ascii=False) + '\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=54090)
    parser.add_argument('--provider', choices=('custom', 'gpt', 'gemini', 'deepseek', 'claude', 'auto'),
                        default='custom', help='Provider profile; gemini models use native generateContent transport')
    parser.add_argument('--upstream', default='https://api.deepseek.com')
    parser.add_argument('--profile', default='auto',
                        help='Domain profile: auto, generic, or a JSON profile path (default: auto)')
    parser.add_argument('--gemini-thinking-level', choices=('auto', 'minimal', 'low', 'medium', 'high'),
                        default='auto', help='Default Gemini thinking level; explicit request setting takes priority')
    parser.add_argument('--gemini-prefix', default='/v1beta',
                        help='Native Gemini path at upstream origin (default: /v1beta)')
    parser.add_argument('--claude-path', default='/v1/messages', help='Claude Messages path at upstream origin')
    parser.add_argument('--claude-schema-mode', choices=('structured', 'json'), default='structured',
                        help='structured sends native Schema; json uses prompt-only JSON without capability selection')
    parser.add_argument('--claude-plan-depth', type=int, choices=range(1, 5), default=1, help='Finite RTDL container nesting depth')
    parser.add_argument('--claude-schema-cap-limit', type=int, choices=range(0, 5), default=1,
                        help='Let Claude choose this many capability types per round; 0 sends the full schema')
    parser.add_argument('--records', type=Path, default=Path(__file__).parent / 'records')
    parser.add_argument('--deepseek-rtdl', action='store_true',
                        help='Adapt DeepSeek RTDL planning requests; response bytes remain unchanged')
    parser.add_argument('--responses-schema', action='store_true',
                        help='Bridge DeepSeek RTDL to Responses JSON Schema; requires --deepseek-rtdl')
    parser.add_argument('--compact-context', action=argparse.BooleanOptionalAction, default=False,
                        help='Compact repeated Pilot state and retain structured execution evidence (default: off)')
    parser.add_argument('--empty-observation-hint', action='store_true',
                        help='Note that an empty observation result does not establish absence')
    parser.add_argument('--uninspected-object-hint', action='store_true',
                        help='Note that an observation hit gives a position, not inspected contents')
    parser.add_argument('--explore-followup-hint', action='store_true',
                        help='Note that an exploration result is not a search verdict')
    parser.add_argument('--embedded-error-hint', action='store_true',
                        help='Surface an error nested inside a result marked successful')
    parser.add_argument('--holding-delivery-hint', action='store_true',
                        help='Note that items carried in inventory are not yet delivered')
    parser.add_argument('--repeat-failure-hint', action='store_true',
                        help='Note that an identical call has just failed repeatedly')
    parser.add_argument('--refine-rtdl-schema', action=argparse.BooleanOptionalAction, default=False,
                        help='Optionally bind RTDL node/argument schemas to the current capability catalog (default: off)')
    args = parser.parse_args(argv)
    if args.provider in ('gpt', 'gemini', 'claude', 'auto') and (args.deepseek_rtdl or args.responses_schema):
        parser.error('This provider profile cannot use DeepSeek adaptation flags')
    if args.refine_rtdl_schema and args.deepseek_rtdl and not args.responses_schema:
        parser.error('RTDL schema refinement with DeepSeek adaptation requires --responses-schema')
    if args.responses_schema and not args.deepseek_rtdl:
        parser.error('--responses-schema requires --deepseek-rtdl')
    if not args.gemini_prefix.startswith('/') or any(c in args.gemini_prefix for c in ('?', '#', '..')):
        parser.error('gemini-prefix must be an absolute API path without query or traversal')
    if not args.claude_path.startswith('/') or any(c in args.claude_path for c in ('?', '#', '..')):
        parser.error('claude-path must be an absolute API path without query or traversal')
    upstream = urlsplit(args.upstream)
    if (upstream.scheme not in ('https', 'http') or not upstream.hostname
        or upstream.username or upstream.password or upstream.query or upstream.fragment):
        parser.error('upstream must be an HTTP(S) base URL without credentials or query')
    from profiles import validate_args
    args.domain = validate_args(parser, args)
    return args


def configure_server(server, args):
    server.upstream = urlsplit(args.upstream)
    server.provider = args.provider
    server.claude_path = args.claude_path
    server.claude_schema_mode = args.claude_schema_mode
    server.claude_plan_depth = args.claude_plan_depth
    server.claude_schema_cap_limit = args.claude_schema_cap_limit
    server.gemini_prefix = args.gemini_prefix
    server.gemini_thinking_level = None if args.gemini_thinking_level == 'auto' else args.gemini_thinking_level
    server.record_root = args.records
    server.index_lock = threading.Lock()
    server.profile = args.domain
    server.domain = DomainContext(args.domain.contracts, args.domain.empty_slot_items,
                                  args.domain.hint_notes, args.domain.inventory_schema)
    server.deepseek_rtdl = args.deepseek_rtdl
    server.responses_schema = args.responses_schema
    server.compact_context = args.compact_context
    server.empty_observation_hint = args.empty_observation_hint
    server.uninspected_object_hint = args.uninspected_object_hint
    server.explore_followup_hint = args.explore_followup_hint
    server.embedded_error_hint = args.embedded_error_hint
    server.holding_delivery_hint = args.holding_delivery_hint
    server.repeat_failure_hint = args.repeat_failure_hint
    server.refine_rtdl_schema = args.refine_rtdl_schema


def main(argv=None):
    args = parse_args(argv)
    os.umask(0o077)
    args.records.mkdir(parents=True, exist_ok=True, mode=0o700)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    configure_server(server, args)
    print(f'Monitor listening on 127.0.0.1:{args.port}; records: {args.records}; '
          f'profile: {args.domain.name}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
