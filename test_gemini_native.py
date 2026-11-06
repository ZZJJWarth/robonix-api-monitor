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

from gemini_native import build_request, chat_response, encode_response
from monitor import Handler
from test_context_compaction import fixture


def native_reply(parts=None, finish='STOP'):
    return {'responseId': 'native-id', 'modelVersion': 'gemini-3.8-flash',
            'candidates': [{'finishReason': finish, 'content': {'role': 'model', 'parts': parts or [{'text': '{"content":"测试"}'}]}}],
            'usageMetadata': {'promptTokenCount': 20, 'candidatesTokenCount': 5, 'thoughtsTokenCount': 3, 'totalTokenCount': 28}}


class GeminiProtocolTest(unittest.TestCase):
    def test_schema_and_inline_image_are_preserved_without_mutating_input(self):
        schema = {'type': 'object', 'properties': {'content': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}}, 'additionalProperties': False}
        chat = {'model': 'gemini-3.8-flash', 'messages': [
            {'role': 'system', 'content': 'exact system rules'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'plan', 'schema': schema}},
            'max_completion_tokens': 1000, 'reasoning_effort': 'low'}
        before = copy.deepcopy(chat)
        converted = build_request(chat)
        self.assertEqual(chat, before)
        self.assertEqual(converted['generationConfig']['responseJsonSchema'], schema)
        self.assertEqual(converted['systemInstruction']['parts'][0]['text'], 'exact system rules')
        self.assertEqual(converted['contents'][0]['parts'][1]['inlineData']['data'], 'AA==')
        self.assertEqual(converted['generationConfig']['maxOutputTokens'], 1000)

    def test_signed_parallel_tool_history_survives_round_trip(self):
        native_parts = [{'functionCall': {'name': 'read_manual', 'args': {'name': 'elevator'}, 'id': 'native-call'}, 'thoughtSignature': 'original-signature'},
                        {'functionCall': {'name': 'read_other', 'args': {}}}]
        reply = chat_response(native_reply(native_parts), 'gemini-3.8-flash')
        message = reply['choices'][0]['message']
        chat = {'model': 'gemini-3.8-flash', 'messages': [message]}
        for call in message['tool_calls']:
            chat['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'content': '{"result":"manual text"}'})
        result = build_request(chat)
        self.assertEqual(result['contents'][0]['parts'], native_parts)
        self.assertEqual(len(result['contents'][1]['parts']), 2)
        self.assertEqual(result['contents'][1]['parts'][0]['functionResponse']['id'], 'native-call')
        self.assertEqual(result['contents'][1]['parts'][0]['functionResponse']['response']['result'], 'manual text')
        message['tool_calls'][0]['function']['arguments'] = '{}'
        with self.assertRaises(ValueError):
            build_request(chat)

    def test_incomplete_blocked_and_empty_output_are_rejected(self):
        for response in [native_reply(finish='MAX_TOKENS'), {'promptFeedback': {'blockReason': 'SAFETY'}}, {'candidates': []}]:
            with self.assertRaises(ValueError):
                chat_response(response, 'gemini-3.8-flash')

    def test_tool_schema_and_output_text_survive_conversion(self):
        chat = {'messages': [{'role': 'user', 'content': 'read'}],
                'tools': [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}],
                'tool_choice': {'type': 'function', 'function': {'name': 'read'}}}
        native = build_request(chat)
        self.assertEqual(native['toolConfig']['functionCallingConfig']['allowedFunctionNames'], ['read'])
        text = ' { "content" : "原文" } '
        result = chat_response(native_reply([{'text': text}]), 'gemini-3.8-flash')
        self.assertEqual(result['choices'][0]['message']['content'], text)
        self.assertEqual(result['usage']['completion_tokens'], 8)
        self.assertIn(b'[DONE]', encode_response(result, True)[0])
        self.assertEqual(json.loads(encode_response(result, False)[0]), result)

    def test_unmapped_semantic_options_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_request({'messages': [], 'frequency_penalty': 1})


class GeminiHttpTest(unittest.TestCase):
    def test_real_handler_compacts_then_uses_native_route_and_logs_original(self):
        captured = []
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                captured.append((self.path, self.headers.get('Authorization'), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                raw = json.dumps(native_reply()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        with tempfile.TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
            proxy = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            proxy.upstream = urlsplit(f'http://127.0.0.1:{upstream.server_port}/v1')
            proxy.provider, proxy.gemini_prefix = 'gemini', '/v1beta'
            proxy.compact_context = True
            proxy.record_root, proxy.index_lock = Path(directory), threading.Lock()
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
            for t in threads:
                t.start()
            conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=5)
            try:
                chat = fixture()
                chat['messages'][0]['content'] = 'You are the Robonix Pilot\n## RTDL output'
                chat['messages'][3]['content'] += 'old telemetry ' * 5000
                chat.update(model='gemini-3.8-flash', stream=True, prompt_cache_key='cache-hint',
                            response_format={'type': 'json_schema', 'json_schema': {'name': 'test', 'schema': {'type': 'object', 'required': ['content']}}})
                conn.request('POST', '/chat/completions', json.dumps(chat), {'Content-Type': 'application/json', 'Authorization': 'Bearer secret-test-key'})
                response = conn.getresponse()
                raw = response.read()
                self.assertEqual(response.status, 200)
                self.assertIn(b'[DONE]', raw)
                self.assertEqual(captured[0][0], '/v1beta/models/gemini-3.8-flash:generateContent')
                self.assertEqual(captured[0][1], 'Bearer secret-test-key')
                self.assertEqual(captured[0][2]['generationConfig']['responseJsonSchema'], chat['response_format']['json_schema']['schema'])
                self.assertLess(len(captured[0][2]['contents']), len(chat['messages']))
                deadline = time.monotonic() + 2
                while True:
                    try:
                        summary = json.loads(next(Path(directory).glob('*/summary.json')).read_text())
                        break
                    except (StopIteration, json.JSONDecodeError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.01)
                self.assertTrue(summary['schema_valid'])
                self.assertTrue(summary['optimization']['context_compaction']['applied'])
                record = json.loads(next(Path(directory).glob('*/request.json')).read_text())
                self.assertEqual(record['received_body'], chat)
                self.assertFalse(any(b'secret-test-key' in p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()))
            finally:
                conn.close()
                for server in (proxy, upstream):
                    server.shutdown()
                    server.server_close()
                for t in threads:
                    t.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
