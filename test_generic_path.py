import http.client
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import monitor
from claude_native import build_request
from observation_hints import RESULT

PILOT_SYSTEM = 'You are the Robonix Pilot\n## RTDL output'
PILOT_BODY = {
    'model': 'test-model',
    'stream': False,
    'messages': [
        {'role': 'system', 'content': PILOT_SYSTEM},
        {'role': 'user', 'content': 'find the bread'},
    ],
}
PLAIN_BODY = {
    'model': 'test-model',
    'stream': False,
    'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                 {'role': 'user', 'content': 'hello'}],
}


class ProxyHarness:
    def __init__(self, extra_args=(), profile='generic'):
        self.captured = []
        captured = self.captured

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                captured.append((self.path, raw))
                payload = json.dumps({
                    'id': 'test', 'model': 'test-model',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'},
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.directory = tempfile.TemporaryDirectory()
        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.upstream.daemon_threads = True
        args = monitor.parse_args([
            '--profile', profile, '--port', '0',
            '--upstream', f'http://127.0.0.1:{self.upstream.server_port}',
            '--records', str(Path(self.directory.name) / 'records'),
            *extra_args,
        ])
        (Path(self.directory.name) / 'records').mkdir()
        self.proxy = ThreadingHTTPServer(('127.0.0.1', 0), monitor.Handler)
        self.proxy.daemon_threads = True
        monitor.configure_server(self.proxy, args)
        self.profile = args.domain
        self.threads = [threading.Thread(target=server.serve_forever, daemon=True)
                        for server in (self.upstream, self.proxy)]
        for thread in self.threads:
            thread.start()

    def close(self):
        for server in (self.proxy, self.upstream):
            server.shutdown()
            server.server_close()
        self.directory.cleanup()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.proxy.server_port, timeout=5)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer private'}
            connection.request(method, path, payload, headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    @property
    def records(self):
        return Path(self.directory.name) / 'records'


class GenericPathTest(unittest.TestCase):
    def test_healthz_reports_generic_profile(self):
        harness = ProxyHarness()
        try:
            status, raw = harness.request('GET', '/healthz')
            self.assertEqual(status, 200)
            health = json.loads(raw)
            self.assertEqual(health['profile'], 'generic')
            self.assertEqual(health['profile_source'], 'generic.json')
            self.assertEqual(health['empty_slot_items'], [])
            self.assertEqual(health['inventory_schema'], {})
            self.assertFalse(health['claude_guidance'])
            self.assertFalse(health['rtdl_domain_reminder'])
            self.assertFalse(health['context_compaction'])
        finally:
            harness.close()

    def test_non_pilot_request_is_forwarded_unchanged(self):
        harness = ProxyHarness(['--compact-context'])
        try:
            status, _ = harness.request('POST', '/chat/completions', PLAIN_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(len(harness.captured), 1)
            self.assertEqual(json.loads(harness.captured[0][1]), PLAIN_BODY)
            record = json.loads(next(harness.records.glob('*/request.json')).read_text())
            self.assertNotIn('context_compaction', record['optimization'])
            self.assertNotIn('received_body', record)
            self.assertTrue(next(harness.records.glob('*/summary.json')).is_file())
        finally:
            harness.close()

    def test_pilot_request_is_compacted_in_generic_profile(self):
        calls = []

        def fake_compact(body, **kwargs):
            calls.append(kwargs)
            return body, {'applied': True, 'probe': True,
                          'empty_slot_items': list(kwargs.get('empty_slot_items') or ())}

        harness = ProxyHarness(['--compact-context'])
        try:
            with mock.patch('context_compaction.compact_request', side_effect=fake_compact):
                status, _ = harness.request('POST', '/chat/completions', PILOT_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].get('empty_slot_items'), ())
            self.assertEqual(calls[0].get('inventory_schema'), {})
            self.assertEqual(json.loads(harness.captured[0][1]), PILOT_BODY)
            record = json.loads(next(harness.records.glob('*/request.json')).read_text())
            self.assertTrue(record['optimization']['context_compaction']['probe'])
        finally:
            harness.close()

    def test_generic_profile_never_runs_domain_hints(self):
        harness = ProxyHarness()
        try:
            harness.proxy.empty_observation_hint = True
            with mock.patch('observation_hints.add_empty_observation_notes',
                            side_effect=AssertionError('hint must stay disabled')):
                status, _ = harness.request('POST', '/chat/completions', PILOT_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(harness.captured[0][1]), PILOT_BODY)
        finally:
            harness.close()

    def test_claude_guidance_follows_profile(self):
        chat = {'model': 'claude-test', 'stream': False,
                'messages': [{'role': 'system', 'content': PILOT_SYSTEM},
                             {'role': 'user', 'content': 'do it'}]}
        native, metadata = build_request(chat, guidance=None)
        self.assertEqual([part['text'] for part in native['system']], [PILOT_SYSTEM])
        self.assertNotIn('pilot_guidance', metadata)
        native, metadata = build_request(chat, guidance='Demo guidance.',
                                         guidance_version='demo-v1')
        self.assertIn('Demo guidance.', [part['text'] for part in native['system']])
        self.assertEqual(metadata['pilot_guidance'], 'demo-v1')

    def test_a_custom_profile_enables_its_own_domain_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo.json'
            path.write_text(json.dumps({
                'name': 'demo',
                'hints': ['empty_observation_hint'],
                'hint_notes': {'empty_observation': 'No match; absence is not established.'},
                'contracts': {'observation': ['demo/scene/find']},
                'empty_slot_items': ['demo:air'],
                'inventory_schema': {'transaction_count_field': 'held_after'},
                'claude_guidance': 'Demo guidance.',
                'claude_guidance_version': 'demo-v1',
                'rtdl_domain_reminder': 'Demo domain rule.',
            }))
            harness = ProxyHarness(['--empty-observation-hint'], profile=str(path))
            try:
                status, raw = harness.request('GET', '/healthz')
                health = json.loads(raw)
                self.assertEqual(status, 200)
                self.assertEqual(health['profile'], 'demo')
                self.assertEqual(health['empty_slot_items'], ['demo:air'])
                self.assertEqual(health['inventory_schema'],
                                 {'transaction_count_field': 'held_after'})
                self.assertTrue(health['claude_guidance'])
                self.assertTrue(health['rtdl_domain_reminder'])
                request = {'model': 'test-model', 'stream': False, 'messages': [
                    {'role': 'user', 'content': RESULT + json.dumps({
                        'leaf_result': {'call_id': 'c1', 'contract_id': 'demo/scene/find',
                                        'success': True, 'output': {'objects': []}}})}]}
                status, _ = harness.request('POST', '/chat/completions', request)
                self.assertEqual(status, 200)
                forwarded = json.loads(harness.captured[0][1])
                leaf = json.loads(forwarded['messages'][0]['content'][
                    len(RESULT):])['leaf_result']
                self.assertEqual(leaf['monitor_note'],
                                 'No match; absence is not established.')
            finally:
                harness.close()


if __name__ == '__main__':
    unittest.main()
