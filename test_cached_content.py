import copy
import json
import unittest

from claude_native import build_request
from context_compaction import compact_request
from message_content import text_content, slice_content
from monitor import is_pilot_request
from rtdl_schema import refine_request
from test_context_compaction import fixture, snapshot
from test_claude_native import request
import test_claude_native as claude_tests


class CachedContentTest(unittest.TestCase):
    def test_system_text_blocks_are_identified_without_mutating_cache_metadata(self):
        body = request()
        original = body['messages'][0]['content']
        body['messages'][0]['content'] = [
            {'type': 'text', 'text': original[:12]},
            {'type': 'text', 'text': original[12:], 'cache_control': {'type': 'ephemeral'}}]
        before = copy.deepcopy(body)
        self.assertTrue(is_pilot_request(body))
        self.assertEqual(body, before)
        body['messages'][0]['content'].append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}})
        self.assertFalse(is_pilot_request(body))

    def test_catalog_and_schema_refinement_accept_cached_text_blocks(self):
        body = request()
        for i in (0, 2):
            text = body['messages'][i]['content']
            body['messages'][i]['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
        before = copy.deepcopy(body)
        refined, meta = refine_request(body)
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['capability_count'], 4)
        self.assertEqual(refined['messages'], before['messages'])
        native, _ = build_request(refined)
        self.assertEqual(native['system'][0]['cache_control'], {'type': 'ephemeral'})
        catalog = next(m for m in native['messages'] if 'Available capabilities' in m['content'][0].get('text', ''))
        self.assertEqual(catalog['content'][0]['cache_control'], {'type': 'ephemeral'})

    def test_latest_cached_live_state_supersedes_old_running_snapshot(self):
        body = fixture()
        latest = snapshot('LATEST_CACHED_FLOOR', running=False)
        text = latest['content']
        latest['content'] = [{'type': 'text', 'text': text[:40]},
                             {'type': 'text', 'text': text[40:], 'cache_control': {'type': 'ephemeral'}, 'opaque': 'preserve'}]
        body['messages'].insert(-1, latest)
        before = copy.deepcopy(body)
        compacted, meta = compact_request(body, min_chars=0)
        self.assertTrue(meta['applied'])
        combined = '\n'.join(text_content(m.get('content')) or '' for m in compacted['messages'])
        self.assertIn('LATEST_CACHED_FLOOR', combined)
        self.assertIn('"count":0', combined)
        self.assertNotIn('OLD_FLOOR', combined)
        self.assertNotIn('CURRENT_FLOOR_4', combined)
        self.assertNotIn('boundary_op=77', combined)
        self.assertEqual(body, before)
        cached = [p for m in compacted['messages'] if isinstance(m.get('content'), list) for p in m['content'] if p.get('cache_control')]
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]['opaque'], 'preserve')

    def test_all_text_wrapped_harness_results_survive_compaction(self):
        body = fixture()
        for m in body['messages']:
            if isinstance(m.get('content'), str):
                m['content'] = [{'type': 'text', 'text': m['content']}]
        compacted, meta = compact_request(body, min_chars=0)
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['historical_results'], 4)
        text = '\n'.join(text_content(m.get('content')) or '' for m in compacted['messages'])
        self.assertIn('CURRENT_FLOOR_4', text)
        self.assertIn('navigation timed out', text)
        self.assertNotIn('HISTORICAL_FLOOR', text)

    def test_slicing_preserves_cache_blocks_without_duplicating_markers(self):
        content = [{'type': 'text', 'text': 'keepA-drop-keepB', 'cache_control': {'type': 'ephemeral'}}]
        result = slice_content(content, [(0, 5), (11, 16)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['cache_control'], content[0]['cache_control'])
        self.assertNotIn('drop', text_content(result))
        self.assertIsNone(text_content([{'type': 'text', 'text': 'known'}, {'type': 'image_url', 'image_url': {'url': 'x'}}]))


class CachedHttpTest(unittest.TestCase):
    # Keep the integration fixture used by actual Claude tests, but represent
    # every textual role as blocks, as the cache-enabled Pilot does.
    def test_cached_pilot_runs_schema_selection_compaction_and_native_bridge(self):
        claude_tests.ClaudeHttpTest.exercise(self, cached=True)


if __name__ == '__main__':
    unittest.main()
