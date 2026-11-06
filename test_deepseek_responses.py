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

from deepseek_responses import CompletedResponse, build_request, chat_stream
from deepseek_rtdl import optimize_request
from monitor import Handler


def request():
    return {'model': 'deepseek-flash', 'messages': [
        {'role': 'system', 'content': 'You are the Robonix Pilot\n## RTDL output'},
        {'role': 'user', 'content': '去房间找食物'},
    ], 'stream': True, 'response_format': {'type': 'json_schema', 'json_schema': {
        'name': 'rtdl_envelope', 'strict': False,
        'schema': {'type': 'object', 'properties': {'content': {'type': 'string'}},
                   'required': ['content']}}}}


def completed(text='{"content":"找到箱子"}', tools=False):
    output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]
    if tools:
        output.append({'type': 'function_call', 'call_id': 'c1', 'name': 'read_manual',
                       'arguments': '{ "name": "电梯" }'})
    return {'id': 'resp-test', 'created_at': 123, 'model': 'deepseek-flash',
            'status': 'completed', 'output': output,
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30}}


def sse(response):
    return ('event: response.completed\ndata: ' + json.dumps({
        'type': 'response.completed', 'response': response}, ensure_ascii=False) + '\n\n').encode()


class ProtocolTest(unittest.TestCase):
    def test_schema_and_multimodal_tool_history_survive_without_mutating_request(self):
        original = request()
        original['messages'] += [
            {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'c1', 'type': 'function', 'function': {'name': 'read_manual', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'manual result'},
            {'role': 'user', 'content': [{'type': 'text', 'text': '看看'},
             {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA==', 'detail': 'low'}}]},
        ]
        original['tools'] = [{'type': 'function', 'function': {
            'name': 'read_manual', 'parameters': {'type': 'object'}}}]
        before = copy.deepcopy(original)
        optimized, _ = optimize_request(original)
        result = build_request(original, optimized)
        self.assertEqual(original, before)
        self.assertEqual(result['text']['format']['schema'], original['response_format']['json_schema']['schema'])
        self.assertEqual(result['input'][2], {'type': 'function_call', 'call_id': 'c1', 'name': 'read_manual', 'arguments': '{}'})
        self.assertEqual(result['input'][3]['output'], 'manual result')
        self.assertEqual(result['input'][4]['content'][1]['type'], 'input_image')
        self.assertEqual(result['input'][-1]['content'], optimized['messages'][-1]['content'])
        self.assertNotIn('strict', result['text']['format'])
        self.assertEqual(result['tools'][0]['name'], 'read_manual')

    def test_sse_accepts_unicode_split_at_every_byte(self):
        response = completed()
        reader = CompletedResponse()
        for byte in sse(response):
            reader.feed(bytes([byte]))
        reader.feed(b'', final=True)
        payload, content = chat_stream(reader.completed())
        self.assertEqual(content, response['output'][0]['content'][0]['text'])
        self.assertIn(b'data: [DONE]', payload)

    def test_failed_incomplete_and_unterminated_streams_are_not_success(self):
        for kind in ['response.failed', 'response.incomplete', 'error']:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                CompletedResponse().feed(b'data: ' + json.dumps({'type': kind}).encode() + b'\n')
        with self.assertRaises(ValueError):
            CompletedResponse().completed()

    def test_tool_call_arguments_and_content_are_preserved_exactly(self):
        response = completed(tools=True)
        payload, text = chat_stream(response)
        events = [json.loads(line[5:]) for line in payload.splitlines()
                  if line.startswith(b'data:') and line != b'data: [DONE]']
        calls = [c for e in events for c in e['choices'][0]['delta'].get('tool_calls', [])]
        self.assertEqual(calls[0]['function']['arguments'], response['output'][1]['arguments'])
        self.assertEqual(calls[0]['id'], 'c1')
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(events[-1]['usage']['prompt_tokens'], 10)


class HttpBridgeTest(unittest.TestCase):
    def exercise(self, upstream_status=200, incomplete=False, schema=True, compact=False, gpt=False, passthrough_model=None, base_path=""):
        captured = []
        self_outer = self
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                self_outer.assertEqual(self.headers.get('Authorization'), 'Bearer test-private-key')
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                if upstream_status != 200:
                    body = b'{"error":{"message":"schema rejected"}}'
                elif not schema or gpt:
                    body = b'data: {"choices":[{"index":0,"delta":{"content":"summary"}}]}\n\ndata: [DONE]\n\n'
                elif incomplete:
                    body = b'data: {"type":"response.incomplete","response":{}}\n\n'
                else:
                    body = sse(completed())
                self.send_response(upstream_status)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        with tempfile.TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
            proxy = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            proxy.upstream = urlsplit(f'http://127.0.0.1:{upstream.server_port}{base_path}')
            proxy.record_root = Path(directory)
            proxy.index_lock = threading.Lock()
            proxy.deepseek_rtdl = not gpt
            proxy.responses_schema = not gpt
            proxy.compact_context = compact
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
            for thread in threads:
                thread.start()
            conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=5)
            try:
                body = request()
                if compact:
                    from test_context_compaction import fixture
                    body['messages'] = fixture()['messages']
                    body['messages'][0]['content'] = 'You are the Robonix Pilot\n## RTDL output'
                    body['messages'][3]['content'] += '\n' + 'old telemetry ' * 5000
                if gpt:
                    body['model'] = passthrough_model or 'gpt-6-luna'
                    body['reasoning_effort'] = 'medium'
                    body['tools'] = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]
                if not schema:
                    body.pop('response_format')
                conn.request('POST', '/chat/completions', json.dumps(body), {
                    'Authorization': 'Bearer test-private-key', 'Content-Type': 'application/json'})
                response = conn.getresponse()
                raw = response.read()
                files = [p for p in Path(directory).rglob('*') if p.is_file()]
                self.assertTrue(files)
                self.assertFalse(any(b'test-private-key' in p.read_bytes() for p in files))
                deadline = time.monotonic() + 2
                while True:
                    try:
                        summary = json.loads(next(Path(directory).glob('*/summary.json')).read_text())
                        break
                    except (StopIteration, json.JSONDecodeError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
                if gpt:
                    self.assertEqual(captured[0][0], base_path + '/chat/completions')
                    expected = copy.deepcopy(body)
                    expected['messages'] = captured[0][1]['messages']
                    self.assertEqual(captured[0][1], expected)
                    self.assertFalse(summary['optimization']['applied'])
                    if compact:
                        self.assertTrue(summary['optimization']['context_compaction']['applied'])
                        self.assertLess(len(captured[0][1]['messages']), len(body['messages']))
                        self.assertFalse(any(m.get('content') == __import__('deepseek_rtdl').FORMAT_REMINDER for m in captured[0][1]['messages']))
                        record = json.loads(next(Path(directory).glob('*/request.json')).read_text())
                        self.assertEqual(record['received_body'], body)
                    else:
                        self.assertEqual(captured[0][1], body)
                    self.assertEqual(raw, b'data: {"choices":[{"index":0,"delta":{"content":"summary"}}]}\n\ndata: [DONE]\n\n')
                elif schema:
                    self.assertEqual(captured[0][0], '/responses')
                    self.assertEqual(captured[0][1]['text']['format']['type'], 'json_schema')
                    self.assertEqual(captured[0][1]['text']['format']['schema'], body['response_format']['json_schema']['schema'])
                    self.assertEqual(summary['transport'], 'responses_schema')
                    if compact:
                        self.assertTrue(summary['optimization']['context_compaction']['applied'])
                        self.assertLess(len(captured[0][1]['input']), len(body['messages']))
                else:
                    self.assertEqual(captured[0][0], base_path + '/chat/completions')
                return response.status, raw, summary
            finally:
                conn.close()
                for server in (proxy, upstream):
                    server.shutdown()
                    server.server_close()
                for thread in threads:
                    thread.join(timeout=2)

    def test_gpt_passthrough_compacts_only_history_and_preserves_response(self):
        status, _, _ = self.exercise(gpt=True, compact=True)
        self.assertEqual(status, 200)

    def test_gpt_short_request_passes_through_unchanged(self):
        status, _, _ = self.exercise(gpt=True)
        self.assertEqual(status, 200)

    def test_gemini_passthrough_with_v1_base_compacts_and_keeps_schema(self):
        status, _, _ = self.exercise(gpt=True, compact=True,
                                     passthrough_model='gemini-test', base_path='/v1')
        self.assertEqual(status, 200)

    def test_gemini_short_request_is_unchanged(self):
        status, _, _ = self.exercise(gpt=True, passthrough_model='gemini-test')
        self.assertEqual(status, 200)

    def test_actual_http_translation_and_schema_record(self):
        status, raw, summary = self.exercise()
        self.assertEqual(status, 200)
        self.assertIn(b'[DONE]', raw)
        self.assertTrue(summary['schema_valid'])
        self.assertTrue(summary['delivered'])

    def test_context_is_compacted_before_responses_conversion(self):
        status, raw, summary = self.exercise(compact=True)
        self.assertEqual(status, 200)
        self.assertTrue(summary['schema_valid'])

    def test_upstream_rejection_is_not_silently_downgraded(self):
        status, raw, summary = self.exercise(upstream_status=400)
        self.assertEqual(status, 502)
        self.assertNotIn(b'[DONE]', raw)
        self.assertFalse(summary['delivered'])

    def test_incomplete_stream_does_not_deliver_a_partial_plan(self):
        status, raw, summary = self.exercise(incomplete=True)
        self.assertEqual(status, 502)
        self.assertNotIn(b'[DONE]', raw)
        self.assertFalse(summary['delivered'])

    def test_non_rtdl_requests_still_pass_through_chat(self):
        status, raw, summary = self.exercise(schema=False)
        self.assertEqual(status, 200)
        self.assertIn(b'summary', raw)
        self.assertFalse(summary['optimization']['applied'])


if __name__ == '__main__':
    unittest.main()
