import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import context_compaction
import monitor
from observation_hints import DEFAULT_DOMAIN, DomainContext
from profiles import ProfileError, load_profile

DEMO = {
    'name': 'demo',
    'hints': ['empty_observation_hint'],
    'hint_notes': {'empty_observation': 'No match; absence is not established.'},
    'contracts': {'observation': ['demo/scene/find'], 'inventory': 'demo/inventory'},
    'empty_slot_items': ['demo:air'],
    'inventory_schema': {'transaction_count_field': 'held_after',
                         'slots_field': 'items', 'slot_field': 'index',
                         'item_field': 'kind', 'count_field': 'amount'},
    'claude_guidance': 'Use the demo capabilities only.',
    'claude_guidance_version': 'demo-v1',
    'rtdl_domain_reminder': 'The demo requires an explicit target.',
}


def parse(argv):
    return monitor.parse_args(argv)


def write_profile(directory, **overrides):
    path = Path(directory) / 'demo.json'
    path.write_text(json.dumps({**DEMO, **overrides}))
    return path


class ProfileResolutionTest(unittest.TestCase):
    def test_auto_defaults_to_generic(self):
        args = parse([])
        self.assertEqual(args.domain.name, 'generic')
        self.assertEqual(args.domain.hints, frozenset())
        self.assertEqual(args.domain.empty_slot_items, ())
        self.assertIsNone(args.domain.claude_guidance)
        self.assertIsNone(args.domain.rtdl_domain_reminder)

    def test_auto_with_a_hint_flag_is_rejected(self):
        for flag in ('--empty-observation-hint', '--uninspected-object-hint',
                     '--explore-followup-hint', '--embedded-error-hint',
                     '--holding-delivery-hint', '--repeat-failure-hint'):
            with self.subTest(flag=flag):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        parse([flag])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('does not enable these hints', stderr.getvalue())

    def test_generic_with_a_hint_flag_is_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                parse(['--profile', 'generic', '--empty-observation-hint'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("'generic' does not enable", stderr.getvalue())

    def test_custom_profile_loads_every_domain_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(directory)
            spec = parse(['--profile', str(path), '--empty-observation-hint']).domain
            self.assertEqual(spec.name, 'demo')
            self.assertEqual(spec.source, str(path))
            self.assertEqual(spec.hints, frozenset({'empty_observation_hint'}))
            self.assertEqual(spec.hint_notes, DEMO['hint_notes'])
            self.assertEqual(spec.contracts['observation'], ('demo/scene/find',))
            self.assertEqual(spec.contracts['inventory'], 'demo/inventory')
            self.assertIsNone(spec.contracts['inspect'])
            self.assertEqual(spec.empty_slot_items, ('demo:air',))
            self.assertEqual(spec.inventory_schema, DEMO['inventory_schema'])
            self.assertEqual(spec.claude_guidance, DEMO['claude_guidance'])
            self.assertEqual(spec.claude_guidance_version, 'demo-v1')
            self.assertEqual(spec.rtdl_domain_reminder, DEMO['rtdl_domain_reminder'])

    def test_profile_name_defaults_to_the_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unnamed.json'
            path.write_text('{}')
            self.assertEqual(load_profile(str(path)).name, str(path))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ProfileError):
            load_profile('does-not-exist')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse(['--profile', 'nope'])

    def test_invalid_profile_fields_are_rejected(self):
        cases = [
            {'unknown': True},
            {'name': ''},
            {'hints': ['made_up_hint']},
            {'hints': 'empty_observation_hint'},
            {'hints': ['empty_observation_hint'], 'hint_notes': {}},
            {'hints': ['empty_observation_hint'],
             'hint_notes': {'empty_observation': 'bad {placeholder}'}},
            {'hints': ['holding_delivery_hint'],
             'hint_notes': {'holding_delivery': 'Holding {wrong}.'}},
            {'hints': ['embedded_error_hint'],
             'hint_notes': {'embedded_error': 3}},
            {'hints': ['empty_observation_hint'],
             'hint_notes': {'empty_observation': 'ok', 'unknown': 'x'}},
            {'empty_slot_items': [1]},
            {'contracts': {'inspect': 3}},
            {'contracts': {'made_up': 'x'}},
            {'contracts': {'observation': [1]}},
            {'inventory_schema': {'made_up': 'x'}},
            {'inventory_schema': {'slots_field': ''}},
            {'claude_guidance': 3},
            {'claude_guidance_version': 'v1'},
            {'rtdl_domain_reminder': 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, profile in enumerate(cases):
                with self.subTest(profile=profile):
                    path = Path(directory) / f'bad-{index}.json'
                    path.write_text(json.dumps(profile))
                    with self.assertRaises(ProfileError):
                        load_profile(str(path))

    def test_unreadable_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.json'
            path.write_text('{')
            with self.assertRaises(ProfileError):
                load_profile(str(path))


class DomainContextTest(unittest.TestCase):
    def test_contract_matching_uses_profile_values(self):
        domain = DomainContext(contracts={'observation': ['demo/scene/find'],
                                          'inspect': 'demo/inspect'},
                               empty_slot_items=['demo:air'])
        self.assertTrue(domain.is_observation('x/demo/scene/find'))
        self.assertFalse(domain.is_observation('other/scene/find'))
        self.assertTrue(domain.is_inspect('x/demo/inspect'))
        self.assertFalse(domain.is_explore('demo/explore'))
        self.assertTrue(domain.is_empty_item('demo:air'))
        self.assertFalse(domain.is_empty_item('other:air'))

    def test_notes_are_formatted_only_when_configured(self):
        domain = DomainContext(note_texts={'holding_delivery': 'Holding {count}.'})
        self.assertEqual(domain.note('holding_delivery', count=2), 'Holding 2.')
        self.assertIsNone(domain.note('empty_observation'))
        self.assertIsNone(domain.note('holding_delivery'))

    def test_the_default_domain_never_annotates(self):
        request = {'messages': [{'role': 'user', 'content': 'plain request'}]}
        before = json.loads(json.dumps(request))
        for function in ('add_empty_observation_notes', 'add_uninspected_object_notes',
                         'add_embedded_error_notes', 'add_explore_followup_notes',
                         'add_repeat_failure_notes', 'add_holding_delivery_notes'):
            module = __import__('observation_hints', fromlist=[function])
            result, info = getattr(module, function)(request)
            self.assertIs(result, request)
            self.assertTrue(all(value == 0 for value in info.values()), (function, info))
        self.assertEqual(request, before)
        self.assertEqual(DEFAULT_DOMAIN.empty_slot_items, ())

    def test_compaction_uses_profile_empty_slot_items_and_inventory_fields(self):
        value = {'items': [{'index': 0, 'kind': 'demo:air', 'amount': 0},
                           {'index': 1, 'kind': 'demo:air', 'amount': 2}],
                 'held_after': 2}
        default = context_compaction.compact_output(value)
        self.assertEqual(len(default['items']), 2)
        custom = context_compaction.compact_output(
            value, empty_slot_items=('demo:air',),
            inventory_schema=DEMO['inventory_schema'])
        self.assertEqual([slot['index'] for slot in custom['items']], [1])
        self.assertEqual(custom['_monitor_observed_empty_slots'], [0])


if __name__ == '__main__':
    unittest.main()
