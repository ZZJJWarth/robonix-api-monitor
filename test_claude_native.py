import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.parse import urlsplit

import jsonschema

from claude_native import NativeStream, build_request, chat_response
from claude_schema import adapt_schema, select_capabilities, ClaudeSchemaError
from monitor import Handler
from observation_hints import DomainContext
from profiles import DomainSpec
from rtdl_schema import refine_request
from test_rtdl_schema import request as base_request, action, envelope, catalog_message

DEMO_GUIDANCE = 'Follow the demo execution rules.'
DEMO_GUIDANCE_VERSION = 'demo-v1'
DEMO_PROFILE = DomainSpec(
    name='demo', source='test', hints=frozenset(), hint_notes={}, contracts={},
    empty_slot_items=(), inventory_schema={}, claude_guidance=DEMO_GUIDANCE,
    claude_guidance_version=DEMO_GUIDANCE_VERSION, rtdl_domain_reminder=None)


def request():
    body = base_request()
    props = body['response_format']['json_schema']['schema']['properties']
    props['message_updates']['items'] = {'type': 'object', 'properties': {
        'message_id': {'type': 'string'}, 'parent_task_id': {'type': 'string'}}, 'required': ['message_id', 'parent_task_id']}
    props['task_updates']['items'] = {'type': 'object', 'properties': {
        'task_id': {'type': 'string'}, 'goal': {'type': 'string'}, 'success_criterion': {'type': 'string'},
        'status': {'type': 'string'}, 'status_reason': {'type': 'string'}}, 'required': ['task_id']}
    return body


def native_message(text=None, blocks=None, stop='end_turn'):
    return {'id': 'msg-test', 'type': 'message', 'role': 'assistant', 'model': 'claude-haiku-5-5',
            'content': blocks if blocks is not None else [{'type': 'text', 'text': text or json.dumps(envelope(action()), ensure_ascii=False)}],
            'stop_reason': stop, 'usage': {'input_tokens': 10, 'output_tokens': 20, 'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 3}}


def native_sse(message, complete=True):
    start = copy.deepcopy(message)
    start['content'] = []
    start['stop_reason'] = None
    events = [{'type': 'message_start', 'message': start}]
    for i, block in enumerate(message['content']):
        if block['type'] == 'text':
            events += [{'type': 'content_block_start', 'index': i, 'content_block': {'type': 'text', 'text': ''}},
                       {'type': 'content_block_delta', 'index': i, 'delta': {'type': 'text_delta', 'text': block['text']}}]
        elif block['type'] == 'tool_use':
            events += [{'type': 'content_block_start', 'index': i, 'content_block': {**block, 'input': {}}},
                       {'type': 'content_block_delta', 'index': i, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'], ensure_ascii=False)}}]
        elif block['type'] == 'thinking':
            events += [{'type': 'content_block_start', 'index': i, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''}},
                       {'type': 'content_block_delta', 'index': i, 'delta': {'type': 'thinking_delta', 'thinking': block['thinking']}},
                       {'type': 'content_block_delta', 'index': i, 'delta': {'type': 'signature_delta', 'signature': block['signature']}}]
        events.append({'type': 'content_block_stop', 'index': i})
    if complete:
        events += [{'type': 'message_delta', 'delta': {'stop_reason': message['stop_reason']}, 'usage': message['usage']}, {'type': 'message_stop'}]
    return b''.join(b'event: ' + e['type'].encode() + b'\ndata: ' + json.dumps(e, ensure_ascii=False).encode() + b'\n\n' for e in events)


class ClaudeSchemaTest(unittest.TestCase):
    def setUp(self):
        self.body, _ = refine_request(request())
        self.original = self.body['response_format']['json_schema']['schema']

    def test_finite_depth_closed_objects_and_original_not_mutated(self):
        before = copy.deepcopy(self.original)
        adapted, meta = adapt_schema(self.original, 1, rtdl=True)
        self.assertEqual(self.original, before)
        validator = jsonschema.Draft202012Validator(adapted)
        one = {'op': 'sequence', 'op_id': 0, 'description': 'one', 'children': [action()]}
        self.assertTrue(validator.is_valid(envelope(one)))
        self.assertFalse(validator.is_valid(envelope({'op': 'parallel', 'op_id': 0, 'description': 'two', 'children': [one]})))
        self.assertTrue(validator.is_valid(envelope()))
        self.assertEqual(meta['max_container_depth'], 1)
        self.assertTrue(meta['closed_object_count'])

    def test_numeric_constraints_remain_in_original_local_contract(self):
        adapted, meta = adapt_schema(self.original, rtdl=True)
        invalid = envelope(action('navigation.navigation_set_speed_limit', {'percentage': 101, 'run_id': '', 'persist': True}))
        self.assertTrue(jsonschema.Draft202012Validator(adapted).is_valid(invalid))
        self.assertFalse(jsonschema.Draft202012Validator(self.original).is_valid(invalid))
        self.assertTrue(meta['locally_validated_constraints'])

    def test_selection_prunes_unused_schemas_without_changing_original_catalog(self):
        selected = select_capabilities(self.original, ['executor.builtin_read_capability_doc'])
        adapted, _ = adapt_schema(selected, 1, rtdl=True)
        validator = jsonschema.Draft202012Validator(adapted)
        self.assertTrue(validator.is_valid(envelope(action())))
        self.assertFalse(validator.is_valid(envelope(action('scene.scene_list_regions', {}))))
        self.assertFalse(validator.is_valid(envelope(action(args={}))))
        self.assertEqual(sum(k.startswith('monitor_cap_args_') for k in selected['$defs']), 1)
        self.assertEqual(sum(k.startswith('monitor_cap_args_') for k in self.original['$defs']), 4)
        for names in [['unknown'], ['executor.builtin_read_capability_doc'] * 2]:
            with self.assertRaises(ClaudeSchemaError):
                select_capabilities(self.original, names)

    def test_zero_selected_capabilities_can_wait_without_inventing_an_action(self):
        selected = select_capabilities(self.original, [])
        adapted, _ = adapt_schema(selected, 1, rtdl=True)
        validator = jsonschema.Draft202012Validator(adapted)
        self.assertTrue(validator.is_valid(envelope()))
        self.assertFalse(validator.is_valid(envelope(action())))

    def test_unconstrained_or_recursive_argument_objects_fail_explicitly(self):
        for schema in [{'type': 'object'}, {'type': 'object', 'properties': {'next': {'$ref': '#/definitions/self'}},
                                         'definitions': {'self': {'type': 'object', 'properties': {'next': {'$ref': '#/definitions/self'}}}}}]:
            with self.assertRaises(ClaudeSchemaError):
                adapt_schema(schema)


class ClaudeProtocolTest(unittest.TestCase):
    def test_messages_tools_images_and_options_preserve_input(self):
        body = {'model': 'claude-haiku-5-5', 'messages': [{'role': 'system', 'content': 'rules'},
                {'role': 'developer', 'content': 'more rules'}, {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}],
                'max_tokens': 2048, 'temperature': .2, 'stream': True,
                'tools': [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {}}}}],
                'tool_choice': {'type': 'function', 'function': {'name': 'read'}}}
        before = copy.deepcopy(body)
        native, _ = build_request(body)
        self.assertEqual(body, before)
        self.assertEqual(native['system'], [{'type': 'text', 'text': 'rules'}, {'type': 'text', 'text': 'more rules'}])
        self.assertEqual(native['messages'][0]['content'][1]['source']['data'], 'AA==')
        self.assertEqual(native['tools'][0]['name'], 'read')
        self.assertEqual(native['tool_choice'], {'type': 'tool', 'name': 'read'})
        self.assertTrue(native['stream'])

    def test_signed_thinking_and_parallel_tool_results_survive_round_trip(self):
        blocks = [{'type': 'thinking', 'thinking': 'real thought', 'signature': 'actual signature'},
                  {'type': 'text', 'text': 'read manuals'},
                  {'type': 'tool_use', 'id': 'toolu_1', 'name': 'read', 'input': {'name': 'elevator'}},
                  {'type': 'tool_use', 'id': 'toolu_2', 'name': 'read', 'input': {'name': 'navigation'}}]
        result = chat_response(native_message(blocks=blocks, stop='tool_use'), 'claude-haiku-5-5')
        msg = result['choices'][0]['message']
        body = {'model': 'claude-haiku-5-5', 'messages': [msg] + [
            {'role': 'tool', 'tool_call_id': call['id'], 'content': 'manual result'} for call in msg['tool_calls']]}
        native, _ = build_request(body)
        self.assertEqual(native['messages'][0]['content'], blocks)
        self.assertEqual([p['tool_use_id'] for p in native['messages'][1]['content']], ['toolu_1', 'toolu_2'])
        self.assertEqual(result['usage']['prompt_tokens'], 18)
        msg['tool_calls'][0]['function']['arguments'] = '{}'
        with self.assertRaises(ValueError):
            build_request(body)

    def test_sse_split_at_every_byte_and_incomplete_events(self):
        message = native_message(blocks=[{'type': 'thinking', 'thinking': '观察', 'signature': 'sig'},
                                        {'type': 'tool_use', 'id': 't1', 'name': 'read', 'input': {'room': '401'}}], stop='tool_use')
        parser = NativeStream()
        for byte in native_sse(message):
            parser.feed(bytes([byte]))
        parser.feed(b'', final=True)
        self.assertEqual(parser.completed()['content'], message['content'])
        parser = NativeStream()
        parser.feed(native_sse(message, complete=False), final=True)
        with self.assertRaises(ValueError):
            parser.completed()
        for stop in ['max_tokens', 'refusal', None]:
            with self.assertRaises(ValueError):
                chat_response(native_message(stop=stop), 'claude-haiku-5-5')


class ClaudeHttpTest(unittest.TestCase):
    def exercise(self, invalid=False, upstream_error=False, nonstream=False, incomplete=False, cached=False, json_mode=False):
        captured, release = [], threading.Event()
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((self.path, self.headers.get('x-api-key'), body))
                release.wait(3)
                if 'capabilities' in body.get('output_config', {}).get('format', {}).get('schema', {}).get('properties', {}):
                    raw = json.dumps(native_message(text='{"capabilities":["executor.builtin_read_capability_doc"]}')).encode()
                    mime, status = 'application/json', 200
                elif upstream_error:
                    raw, mime, status = b'{"error":{"message":"schema rejected"}}', 'application/json', 400
                else:
                    answer = envelope({'op': 'sequence', 'op_id': 0, 'description': 'two capabilities', 'children': [action(), action('scene.scene_list_regions', {})]}) if json_mode else envelope(action(args={} if invalid else None))
                    message = native_message(text=json.dumps(answer, ensure_ascii=False))
                    raw = json.dumps(message).encode() if nonstream else native_sse(message, complete=not incomplete)
                    mime, status = ('application/json' if nonstream else 'text/event-stream'), 200
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        with tempfile.TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
            proxy = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            proxy.provider, proxy.claude_path = 'auto', '/v1/messages'
            proxy.claude_schema_mode = 'json' if json_mode else 'structured'
            proxy.claude_plan_depth, proxy.claude_schema_cap_limit = 1, 1
            proxy.refine_rtdl_schema, proxy.compact_context = True, True
            proxy.profile, proxy.domain = DEMO_PROFILE, DomainContext()
            proxy.upstream = urlsplit(f'http://127.0.0.1:{upstream.server_port}/v1')
            proxy.record_root, proxy.index_lock = Path(directory), threading.Lock()
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
            for t in threads:
                t.start()
            conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
            try:
                body = request()
                body.update(model='claude-haiku-5-5', stream=not nonstream)
                # Use a real-shaped structured history to exercise compaction.
                from test_context_compaction import fixture
                body['messages'] = fixture()['messages']
                body['messages'][0]['content'] = 'You are the Robonix Pilot\n## RTDL output'
                body['messages'][2] = catalog_message()
                body['messages'][3]['content'] += 'old telemetry ' * 5000
                if cached:
                    for i, msg in enumerate(body['messages']):
                        if isinstance(msg.get('content'), str):
                            part = {'type': 'text', 'text': msg['content']}
                            if i in (0, len(body['messages'])-2, len(body['messages'])-3):
                                part['cache_control'] = {'type': 'ephemeral'}
                            msg['content'] = [part]
                conn.request('POST', '/chat/completions', json.dumps(body), {'Authorization': 'Bearer private-test-key', 'Content-Type': 'application/json'})
                if nonstream:
                    release.set()
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                if not nonstream:
                    # Headers must arrive before the upstream is allowed to answer.
                    self.assertFalse(release.is_set())
                    release.set()
                if invalid or upstream_error or incomplete:
                    with self.assertRaises(http.client.IncompleteRead) as caught:
                        response.read()
                    self.assertNotIn(b'"content"', caught.exception.partial)
                    self.assertNotIn(b'[DONE]', caught.exception.partial)
                else:
                    raw = response.read()
                    if nonstream:
                        content = json.loads(raw)['choices'][0]['message']['content']
                    else:
                        content = ''.join(c.get('delta', {}).get('content', '') for line in raw.splitlines() if line.startswith(b'data: {')
                                          for c in json.loads(line[6:]).get('choices', []))
                        self.assertIn(b'[DONE]', raw)
                    expected = envelope({'op': 'sequence', 'op_id': 0, 'description': 'two capabilities', 'children': [action(), action('scene.scene_list_regions', {})]}) if json_mode else envelope(action())
                    self.assertEqual(json.loads(content), expected)
                deadline = time.monotonic()+2
                while True:
                    try:
                        summary = json.loads(next(Path(directory).glob('*/summary.json')).read_text())
                        break
                    except (StopIteration, json.JSONDecodeError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.01)
                self.assertEqual(len(captured), 1 if json_mode else 2)
                if json_mode:
                    self.assertNotIn('output_config', captured[0][2])
                    self.assertNotIn('claude_capability_selection', summary['optimization'])
                    self.assertEqual(summary['optimization']['claude_schema_mode'], 'json')
                else:
                    self.assertEqual(captured[0][2]['thinking'], {'type': 'disabled'})
                    self.assertEqual(captured[0][2]['max_tokens'], 8192)
                for _, _, request_body in captured:
                    self.assertEqual(sum(block.get('text') == DEMO_GUIDANCE for block in request_body.get('system', [])), 1)
                self.assertEqual(summary['optimization']['pilot_guidance'], DEMO_GUIDANCE_VERSION)
                self.assertTrue(all(p == '/v1/messages' and key == 'private-test-key' for p, key, _ in captured))
                self.assertEqual(summary['delivered'], not (invalid or upstream_error or incomplete))
                self.assertTrue(summary['optimization']['context_compaction']['applied'])
                self.assertTrue(summary['optimization']['schema_refinement']['applied'])
                if cached:
                    self.assertEqual(captured[-1][2]['system'][0]['cache_control'], {'type': 'ephemeral'})
                if not json_mode:
                    self.assertEqual(summary['optimization']['claude_capability_selection']['selected'], ['executor.builtin_read_capability_doc'])
                record = json.loads(next(Path(directory).glob('*/request.json')).read_text())
                self.assertEqual(record['received_body'], body)
                self.assertEqual(record['body'], captured[-1][2])
                self.assertFalse(any(b'private-test-key' in p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()))
            finally:
                release.set()
                conn.close()
                for server in (proxy, upstream):
                    server.shutdown()
                    server.server_close()
                for t in threads:
                    t.join(timeout=2)

    def test_prompt_only_mode_uses_one_api_and_allows_multiple_capabilities(self):
        self.exercise(json_mode=True)

    def test_stream_headers_selection_schema_and_compaction(self):
        self.exercise()

    def test_nonstream_supported(self):
        self.exercise(nonstream=True)

    def test_missing_parameters_never_delivered_as_a_plan(self):
        self.exercise(invalid=True)

    def test_upstream_rejection_never_delivered_as_partial_success(self):
        self.exercise(upstream_error=True)

    def test_truncated_native_stream_never_delivered_as_a_plan(self):
        self.exercise(incomplete=True)


if __name__ == '__main__':
    unittest.main()
