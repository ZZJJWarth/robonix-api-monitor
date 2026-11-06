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

from monitor import Handler
from rtdl_schema import current_catalog, refine_request, SchemaRefinementError


CATALOG = {
    'executor.builtin_read_capability_doc': {'type': 'object', 'properties': {'provider_id': {'type': 'string'}}, 'required': ['provider_id']},
    'navigation.navigation_set_speed_limit': {'type': 'object', 'properties': {
        'percentage': {'type': 'number', 'minimum': 0, 'maximum': 100}, 'run_id': {'type': 'string'}, 'persist': {'type': 'boolean'}},
        'required': ['percentage', 'run_id', 'persist'], 'additionalProperties': False},
    'elevator_robot.navigation_ride_elevator': {'type': 'object', 'properties': {'target_floor': {'type': 'string'}, 'timeout_s': {'type': 'number'}},
        'required': ['target_floor', 'timeout_s'], 'additionalProperties': False},
    'scene.scene_list_regions': {'type': 'object', 'properties': {}, 'additionalProperties': False}}


def catalog_message(catalog=CATALOG):
    return {'role': 'user', 'content': '\n\n## Available capabilities\nLatest complete catalog.\n\n' + '\n'.join(
        f'- capability_name: {name}\n  description: "provider capability"\n  args_schema: {json.dumps(schema)}' for name, schema in catalog.items())}


def request():
    schema = {'type': 'object', 'properties': {
        'content': {'type': 'string'}, 'message_updates': {'type': 'array', 'items': {'type': 'object'}},
        'rtdl_parent_task_id': {'type': ['string', 'null']}, 'rtdl_description': {'type': 'string'},
        'rtdl': {'anyOf': [{'$ref': '#/$defs/node'}, {'$ref': '#/$defs/meta'}]},
        'task_updates': {'type': 'array', 'items': {'type': 'object'}}},
        'required': ['content', 'message_updates', 'rtdl_parent_task_id', 'rtdl_description', 'rtdl', 'task_updates'],
        '$defs': {'node': {'type': 'object', 'properties': {'op': {'type': 'string'}, 'op_id': {'type': 'integer'},
            'description': {'type': 'string'}, 'children': {'type': 'array'}}},
            'meta': {'anyOf': [{'type': 'object', 'properties': {'op': {'enum': ['cancel_plan']}, 'plan_id': {'type': 'string'}}, 'required': ['op', 'plan_id']}]}}}
    return {'model': 'gpt-6-luna', 'stream': True, 'temperature': .2,
            'messages': [{'role': 'system', 'content': 'You are the Robonix Pilot\n## RTDL output'},
                         {'role': 'user', 'content': 'User message m1: bring food'}, catalog_message()],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'rtdl_envelope', 'strict': False, 'schema': schema}}}


def envelope(rtdl=None):
    return {'content': 'test', 'message_updates': [], 'rtdl_parent_task_id': 'T1', 'rtdl_description': 'test',
            'rtdl': rtdl if rtdl is not None else {'op': 'sequence', 'op_id': 0, 'description': 'done', 'children': []}, 'task_updates': []}


def action(cap='executor.builtin_read_capability_doc', args=None):
    return {'op': 'do', 'op_id': 0, 'description': 'call', 'cap': cap, 'args': args if args is not None else {'provider_id': 'elevator_robot'}}


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.original = request()
        self.result, self.meta = refine_request(self.original)
        self.validator = jsonschema.Draft202012Validator(self.result['response_format']['json_schema']['schema'])

    def test_preserves_request_and_argument_semantics(self):
        before = copy.deepcopy(self.original)
        result, meta = refine_request(self.original)
        self.assertEqual(self.original, before)
        for field in ('messages', 'model', 'stream', 'temperature'):
            self.assertEqual(result[field], before[field])
        self.assertEqual(result['response_format']['json_schema']['strict'], False)
        self.assertEqual(meta['capability_count'], 4)
        self.assertTrue(self.validator.is_valid(envelope(action())))
        self.assertTrue(self.validator.is_valid(envelope(action('scene.scene_list_regions', {}))))
        self.assertTrue(self.validator.is_valid(envelope()))
        self.assertTrue(self.validator.is_valid(envelope({'op': 'cancel_plan', 'plan_id': 'p1'})))

    def test_real_missing_args_and_wrong_tree_shapes_are_rejected(self):
        bad = [action(args={}), action('navigation.navigation_set_speed_limit', {}),
               action('elevator_robot.navigation_ride_elevator', {}), {k: v for k, v in action().items() if k != 'args'},
               action() | {'children': []}, action() | {'children': [action()]}, action('unknown.cap', {}),
               {'op': 'sequence', 'op_id': 0, 'description': 'bad'},
               {'op': 'sequence', 'op_id': 0, 'description': 'bad', 'children': [], 'args': {}},
               {'op': 'sequence', 'op_id': 0, 'description': 'bad', 'children': [{'op': 'cancel_plan', 'plan_id': 'p1'}]}]
        for node in bad:
            with self.subTest(node=node):
                self.assertFalse(self.validator.is_valid(envelope(node)))

    def test_optional_fields_types_ranges_and_nested_calls_follow_catalog(self):
        valid = action('navigation.navigation_set_speed_limit', {'percentage': 100, 'run_id': '', 'persist': True})
        self.assertTrue(self.validator.is_valid(envelope(valid)))
        for args in [{'percentage': '100', 'run_id': '', 'persist': True}, {'percentage': 101, 'run_id': '', 'persist': True},
                     {'percentage': 50, 'run_id': '', 'persist': True, 'extra': 1}]:
            self.assertFalse(self.validator.is_valid(envelope(action('navigation.navigation_set_speed_limit', args))))
        tree = {'op': 'parallel', 'op_id': 0, 'description': 'batch', 'children': [valid, action()]}
        self.assertTrue(self.validator.is_valid(envelope(tree)))
        tree['children'][0]['children'] = []
        self.assertFalse(self.validator.is_valid(envelope(tree)))

    def test_latest_catalog_replaces_old_catalog_without_adding_removed_capabilities(self):
        new = {'scene.scene_list_regions': CATALOG['scene.scene_list_regions']}
        self.original['messages'].append(catalog_message(new))
        result, meta = refine_request(self.original)
        validator = jsonschema.Draft202012Validator(result['response_format']['json_schema']['schema'])
        self.assertEqual(meta['capability_count'], 1)
        self.assertFalse(validator.is_valid(envelope(action())))
        self.assertTrue(validator.is_valid(envelope(action('scene.scene_list_regions', {}))))

    def test_malformed_duplicate_or_empty_current_catalog_does_not_fall_back(self):
        for message in [catalog_message({}), catalog_message() | {'content': catalog_message()['content'].replace('args_schema:', 'missing_schema:', 1)},
                        {'role': 'user', 'content': catalog_message()['content'] + catalog_message()['content'].split('Latest complete catalog.')[1]}]:
            body = request()
            body['messages'].append(message)
            with self.subTest(message=message), self.assertRaises(SchemaRefinementError):
                refine_request(body)

    def test_local_refs_are_rebased_and_opaque_instance_data_is_not_rewritten(self):
        args = {'type': 'object', 'properties': {'value': {'$ref': '#/$defs/Value'}, '$id': {'type': 'string'}}, 'required': ['value'],
                '$defs': {'Value': {'type': 'string', 'enum': ['ok'], 'examples': [{'$ref': 'this is data'}]}}}
        body = request()
        body['messages'].append(catalog_message({'ref.cap': args}))
        result, _ = refine_request(body)
        validator = jsonschema.Draft202012Validator(result['response_format']['json_schema']['schema'])
        self.assertTrue(validator.is_valid(envelope(action('ref.cap', {'value': 'ok', '$id': 'plain field'}))))
        self.assertFalse(validator.is_valid(envelope(action('ref.cap', {'value': 'bad'}))))
        args['properties']['value'] = {'$ref': '#/$defs/missing'}
        body['messages'][-1] = catalog_message({'ref.cap': args})
        with self.assertRaises(SchemaRefinementError):
            refine_request(body)
        args['properties']['value'] = {'$ref': 'https://invalid.example/schema'}
        body['messages'][-1] = catalog_message({'ref.cap': args})
        with self.assertRaises(SchemaRefinementError):
            refine_request(body)

    def test_non_rtdl_requests_and_native_tools_remain_unchanged(self):
        for body in [{'model': 'gemini', 'messages': []}, request() | {'response_format': {'type': 'json_object'}},
                     request() | {'messages': [{'role': 'user', 'content': 'normal request'}]}]:
            result, meta = refine_request(body)
            self.assertIs(result, body)
            self.assertFalse(meta['applied'])


class HttpTest(unittest.TestCase):
    def exercise(self, enabled=False, provider='gpt', invalid_catalog=False):
        captured = []
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                if provider == 'gemini':
                    from test_gemini_native import native_reply
                    raw = json.dumps(native_reply([{'text': json.dumps(envelope(action()))}])).encode()
                    mime = 'application/json'
                elif provider == 'deepseek':
                    from test_deepseek_responses import completed, sse
                    raw = sse(completed(json.dumps(envelope(action()))))
                    mime = 'text/event-stream'
                else:
                    raw = b'data: {"choices":[{"index":0,"delta":{"content":"unmodified text"}}]}\n\ndata: [DONE]\n\n'
                    mime = 'text/event-stream'
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        with tempfile.TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
            proxy = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            proxy.provider, proxy.gemini_prefix = provider, '/v1beta'
            proxy.refine_rtdl_schema = enabled
            proxy.deepseek_rtdl = proxy.responses_schema = provider == 'deepseek'
            proxy.upstream = urlsplit(f'http://127.0.0.1:{upstream.server_port}')
            proxy.record_root, proxy.index_lock = Path(directory), threading.Lock()
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
            for t in threads:
                t.start()
            c = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=5)
            try:
                c.request('GET', '/healthz')
                self.assertEqual(json.loads(c.getresponse().read())['refine_rtdl_schema'], enabled)
                body = request()
                body['model'] = {'gpt': 'gpt-6-luna', 'gemini': 'gemini-test', 'deepseek': 'deepseek-flash'}[provider]
                if invalid_catalog:
                    body['messages'][-1] = catalog_message({})
                c.request('POST', '/chat/completions', json.dumps(body), {'Content-Type': 'application/json', 'Authorization': 'Bearer test-private-key'})
                response = c.getresponse()
                raw = response.read()
                if invalid_catalog:
                    self.assertEqual(response.status, 400)
                    self.assertEqual(captured, [])
                    self.assertTrue(list(Path(directory).glob('*/summary.json')))
                    return
                self.assertEqual(response.status, 200)
                sent = captured[0][1]
                schema = sent['generationConfig']['responseJsonSchema'] if provider == 'gemini' else sent['text']['format']['schema'] if provider == 'deepseek' else sent['response_format']['json_schema']['schema']
                validator = jsonschema.Draft202012Validator(schema)
                self.assertEqual(validator.is_valid(envelope(action(args={}))), not enabled)
                self.assertTrue(validator.is_valid(envelope(action())))
                if provider == 'gpt':
                    self.assertIn(b'unmodified text', raw)
                    self.assertEqual(sent['messages'], body['messages'])
                    if not enabled:
                        self.assertEqual(sent, body)
                deadline = time.monotonic() + 2
                while True:
                    try:
                        record = json.loads(next(Path(directory).glob('*/request.json')).read_text())
                        summary = json.loads(next(Path(directory).glob('*/summary.json')).read_text())
                        break
                    except (StopIteration, json.JSONDecodeError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.01)
                if enabled:
                    self.assertEqual(record['received_body'], body)
                    self.assertTrue(record['optimization']['schema_refinement']['applied'])
                self.assertFalse(any(b'test-private-key' in p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()))
            finally:
                c.close()
                for server in (proxy, upstream):
                    server.shutdown()
                    server.server_close()
                for t in threads:
                    t.join(timeout=2)

    def test_default_off_passes_through_unchanged(self):
        self.exercise()

    def test_gpt_gemini_and_deepseek_receive_refined_schema(self):
        for provider in ('gpt', 'gemini', 'deepseek'):
            with self.subTest(provider=provider):
                self.exercise(enabled=True, provider=provider)

    def test_enabled_mode_rejects_bad_catalog_before_upstream_call(self):
        self.exercise(enabled=True, invalid_catalog=True)


if __name__ == '__main__':
    unittest.main()
