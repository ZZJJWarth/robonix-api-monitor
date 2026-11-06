import json
import unittest

from observation_hints import (DISPATCH, DomainContext, MEMORY, RESULT,
                               add_embedded_error_notes, add_empty_observation_notes,
                               add_explore_followup_notes, add_holding_delivery_notes,
                               add_repeat_failure_notes, add_uninspected_object_notes)


NOTE = 'The latest observation matched no record; absence is not established.'
UNINSPECTED_NOTE = 'The observation gives a location, not the object contents.'
EXPLORE_NOTE = 'An exploration result is not a search verdict.'
HOLDING_NOTE = 'Holding {count} item(s); not yet delivered.'
REPEAT_FAILURE_NOTE = 'The same call failed {count} times.'
REPEAT_FAILURE_ERROR_NOTE = 'The same call failed {count} times: {error}'

DOMAIN = DomainContext(
    contracts={'observation': ['demo/scene/find', 'demo/scene/list_objects'],
               'inspect': 'demo/inspect', 'explore': 'demo/explore',
               'withdraw': 'demo/withdraw', 'deposit': 'demo/deposit',
               'inventory': 'demo/inventory'},
    empty_slot_items=('demo:air', 'air'),
    note_texts={'empty_observation': NOTE, 'uninspected_object': UNINSPECTED_NOTE,
                'explore_followup': EXPLORE_NOTE,
                'embedded_error': 'Embedded error: {error}',
                'holding_delivery': HOLDING_NOTE,
                'repeat_failure': REPEAT_FAILURE_NOTE,
                'repeat_failure_with_error': REPEAT_FAILURE_ERROR_NOTE},
    inventory_schema={'transaction_count_field': 'held_after',
                      'slots_field': 'slots', 'slot_field': 'slot',
                      'item_field': 'item', 'count_field': 'count'},
)


def body(*messages):
    return {'model': 'deepseek-flash', 'messages': list(messages)}


def feedback(contract, output, success=True, call_id='c1'):
    return {'role': 'user', 'content': RESULT + json.dumps(
        {'leaf_result': {'call_id': call_id, 'contract_id': contract,
                         'success': success, 'output': output}}, ensure_ascii=False)}


def memory(plans=None, continues=None):
    document = {'note': 'memory', 'plans': plans or []}
    if continues:
        document['continues'] = continues
    return {'role': 'user', 'content': MEMORY + json.dumps(document, ensure_ascii=False)}


def dispatch(calls):
    return {'role': 'user', 'content': DISPATCH + json.dumps({'plan_id': 'p1', 'calls': calls})}


def inspect(object_id, call_id='p1:9', success=True):
    return {'call_id': call_id, 'contract_id': 'demo/inspect',
            'args': {'object_id': object_id},
            'results': [{'success': success, 'output': {'container_type': 'demo:box'}}]}


def hit(*ids):
    objects = [{'id': object_id, 'label': 'demo:box'} for object_id in ids]
    return json.dumps({'objects': objects, 'detail': ''})


def find(output, call_id='p1:0'):
    return {'call_id': call_id, 'contract_id': 'demo/scene/find',
            'results': [{'success': True, 'output': output}]}


def empty_output():
    return '{\n  "objects": [],\n  "detail": "no object matches \'chest\'"\n}'


def leaf_of(message):
    return json.loads(message['content'][len(RESULT):])['leaf_result']


class EmptyObservationNoteTest(unittest.TestCase):
    def test_an_empty_scene_query_carries_the_note(self):
        request = body(feedback('demo/scene/find', empty_output()))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 1)
        self.assertGreater(info['empty_observation_chars'], 0)
        leaf = leaf_of(result['messages'][0])
        self.assertEqual(leaf['monitor_note'], NOTE)
        self.assertEqual(leaf['contract_id'], 'demo/scene/find')
        self.assertIn('no object matches', leaf['output'])
        self.assertNotIn('map@', result['messages'][0]['content'])

    def test_a_non_empty_scene_query_is_untouched(self):
        request = body(feedback('demo/scene/find',
                                '{"objects": [{"label": "demo:box"}], "detail": ""}'))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 0)
        self.assertIs(result, request)

    def test_a_later_hit_suppresses_the_note_on_an_earlier_empty_result(self):
        request = body(feedback('demo/scene/find', empty_output(), call_id='c1'),
                       feedback('demo/scene/find',
                                '{"objects": [{"label": "demo:box"}]}', call_id='c2'))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 0)
        self.assertIs(result, request)

    def test_only_the_latest_empty_result_is_noted(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            find({'objects': [], 'detail': 'none'}, 'p1:0'),
            find({'objects': [], 'detail': 'none again'}, 'p1:1')]}]))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 1)
        document = json.loads(result['messages'][0]['content'][len(MEMORY):])
        first, second = document['plans'][0]['calls'][0]['results'][0], document['plans'][0]['calls'][1]['results'][0]
        self.assertNotIn('monitor_note', first)
        self.assertEqual(second['monitor_note'], NOTE)

    def test_a_non_scene_empty_result_is_untouched(self):
        request = body(feedback('demo/inventory', '{"slots": []}'))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 0)
        self.assertIs(result, request)

    def test_a_failed_scene_query_is_untouched(self):
        request = body(feedback('demo/scene/find', empty_output(), success=False))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 0)
        self.assertIs(result, request)

    def test_memory_uses_the_call_contract_and_notes_continues(self):
        plans = [{'plan_id': 'p1', 'calls': [
            {'call_id': 'p1:0', 'contract_id': 'demo/scene/find',
             'results': [{'success': True, 'output': {'objects': [], 'detail': 'none'}}]},
            {'call_id': 'p1:1', 'contract_id': 'demo/inventory',
             'results': [{'success': True, 'output': {'slots': []}}]}]}]
        continues = [{'plan_id': 'p0', 'result': {
            'success': True, 'contract_id': 'demo/scene/list_objects',
            'output': {'objects': []}}}]
        request = body(memory(plans, continues))
        result, info = add_empty_observation_notes(request, domain=DOMAIN)
        self.assertEqual(info['empty_observation_notes'], 1)
        document = json.loads(result['messages'][0]['content'][len(MEMORY):])
        self.assertEqual(document['continues'][0]['result']['monitor_note'], NOTE)
        self.assertNotIn('monitor_note', document['plans'][0]['calls'][0]['results'][0])

    def test_a_second_pass_changes_nothing(self):
        request = body(feedback('demo/scene/find', empty_output()))
        once, first = add_empty_observation_notes(request, domain=DOMAIN)
        twice, second = add_empty_observation_notes(once, domain=DOMAIN)
        self.assertEqual(first['empty_observation_notes'], 1)
        self.assertEqual(second['empty_observation_notes'], 0)
        self.assertEqual(once, twice)


class UninspectedObjectNoteTest(unittest.TestCase):
    def test_a_location_hit_for_an_uninspected_object_carries_the_note(self):
        request = body(feedback('demo/scene/find', hit('crate@1')))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 1)
        self.assertGreater(info['uninspected_object_chars'], 0)
        leaf = leaf_of(result['messages'][0])
        self.assertEqual(leaf['monitor_note'], UNINSPECTED_NOTE)
        self.assertIn('crate@1', leaf['output'])

    def test_an_inspected_object_in_memory_is_untouched(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            inspect('crate@1'),
            find(hit('crate@1'), 'p1:1')]}]))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)
        self.assertIs(result, request)

    def test_one_uninspected_object_among_several_still_warns(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            inspect('crate@1'),
            find(hit('crate@1', 'crate@2'), 'p1:1')]}]))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 1)

    def test_an_inspection_in_a_raw_dispatch_record_counts(self):
        request = body(dispatch([inspect('crate@1')]),
                       feedback('demo/scene/find', hit('crate@1')))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)
        self.assertIs(result, request)

    def test_a_failed_inspection_still_counts_as_an_attempt(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            inspect('crate@1', success=False),
            find(hit('crate@1'), 'p1:1')]}]))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)

    def test_an_empty_result_is_left_to_the_empty_note(self):
        request = body(feedback('demo/scene/find', empty_output()))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)
        self.assertIs(result, request)

    def test_only_the_latest_observation_is_considered(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            find(hit('crate@1'), 'p1:0'),
            inspect('crate@2', 'p1:2'),
            find(hit('crate@2'), 'p1:3')]}]))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)
        self.assertIs(result, request)

    def test_a_non_scene_hit_is_untouched(self):
        request = body(feedback('demo/inventory',
                                '{"slots": [{"slot": 0, "item": "bread", "count": 2}]}'))
        result, info = add_uninspected_object_notes(request, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)
        self.assertIs(result, request)

    def test_a_second_pass_changes_nothing(self):
        request = body(feedback('demo/scene/find', hit('crate@1')))
        once, first = add_uninspected_object_notes(request, domain=DOMAIN)
        twice, second = add_uninspected_object_notes(once, domain=DOMAIN)
        self.assertEqual(first['uninspected_object_notes'], 1)
        self.assertEqual(second['uninspected_object_notes'], 0)
        self.assertEqual(once, twice)


class ExploreFollowupNoteTest(unittest.TestCase):
    def test_the_latest_explore_result_carries_the_note(self):
        request = body(feedback('demo/explore', '', success=False))
        result, info = add_explore_followup_notes(request, domain=DOMAIN)
        self.assertEqual(info['explore_followup_notes'], 1)
        self.assertGreater(info['explore_followup_chars'], 0)
        leaf = leaf_of(result['messages'][0])
        self.assertEqual(leaf['monitor_note'], EXPLORE_NOTE)
        self.assertEqual(leaf['contract_id'], 'demo/explore')

    def test_a_later_non_explore_result_suppresses_the_note(self):
        request = body(feedback('demo/explore', '', success=False, call_id='c1'),
                       feedback('demo/scene/find', '{"objects": []}', call_id='c2'))
        result, info = add_explore_followup_notes(request, domain=DOMAIN)
        self.assertEqual(info['explore_followup_notes'], 0)
        self.assertIs(result, request)

    def test_memory_uses_the_call_contract_when_the_result_omits_it(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            {'call_id': 'p1:0', 'contract_id': 'demo/explore',
             'results': [{'success': False, 'output': '',
                          'error': 'search view budget exhausted; target absence is not established'}]}]}]))
        result, info = add_explore_followup_notes(request, domain=DOMAIN)
        self.assertEqual(info['explore_followup_notes'], 1)
        document = json.loads(result['messages'][0]['content'][len(MEMORY):])
        evidence = document['plans'][0]['calls'][0]['results'][0]
        self.assertEqual(evidence['monitor_note'], EXPLORE_NOTE)
        self.assertIn('target absence is not established', evidence['error'])

    def test_a_later_memory_result_wins_over_an_explore_in_an_earlier_call(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            {'call_id': 'p1:0', 'contract_id': 'demo/explore',
             'results': [{'success': True, 'output': ''}]},
            {'call_id': 'p1:1', 'contract_id': 'demo/inventory',
             'results': [{'success': True, 'output': {'slots': []}}]}]}]))
        result, info = add_explore_followup_notes(request, domain=DOMAIN)
        self.assertEqual(info['explore_followup_notes'], 0)
        self.assertIs(result, request)

    def test_explore_status_results_are_not_a_search_result(self):
        request = body(feedback('demo/explore/status', '{"state": "running"}'))
        result, info = add_explore_followup_notes(request, domain=DOMAIN)
        self.assertEqual(info['explore_followup_notes'], 0)
        self.assertIs(result, request)

    def test_an_explore_result_already_noted_is_untouched(self):
        request = body(feedback('demo/explore', '', success=False))
        once, first = add_explore_followup_notes(request, domain=DOMAIN)
        twice, second = add_explore_followup_notes(once, domain=DOMAIN)
        self.assertEqual(first['explore_followup_notes'], 1)
        self.assertEqual(second['explore_followup_notes'], 0)
        self.assertEqual(once, twice)


class EmbeddedErrorNoteTest(unittest.TestCase):
    NESTED = '{"status": {"data": "{\\"error\\": \\"rotate_deg cannot be combined with velocity mode fields\\"}"}}'

    def test_an_embedded_error_in_a_successful_result_carries_the_note(self):
        request = body(feedback('demo/move', self.NESTED))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 1)
        self.assertGreater(info['embedded_error_chars'], 0)
        leaf = leaf_of(result['messages'][0])
        self.assertTrue(leaf['success'])
        self.assertEqual(leaf['contract_id'], 'demo/move')
        self.assertIn('cannot be combined', leaf['output'])
        self.assertIn('Embedded error: rotate_deg cannot be combined',
                      leaf['monitor_note'])

    def test_a_string_encoded_top_level_error_is_still_hidden(self):
        request = body(feedback('demo/move', '{"error": "no such block"}'))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 1)
        self.assertIn('Embedded error: no such block', leaf_of(result['messages'][0])['monitor_note'])

    def test_a_visible_top_level_error_is_untouched(self):
        request = body(feedback('demo/deposit',
                                {'error': 'container is full'}))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 0)
        self.assertIs(result, request)

    def test_a_failed_call_is_untouched(self):
        request = body(feedback('demo/move', self.NESTED, success=False))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 0)
        self.assertIs(result, request)

    def test_a_truncated_preview_is_not_parsed(self):
        preview = '{"status": {"data": "{\\"error\\": \\"rotate_deg cannot be co'
        request = body(feedback('demo/move', {
            '_robonix_truncation': {'truncated': True, 'complete': False}, 'preview': preview}))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 0)
        self.assertIs(result, request)

    def test_an_error_free_result_is_untouched(self):
        request = body(feedback('demo/scene/find',
                                '{"objects": [{"label": "demo:box"}], "detail": ""}'))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 0)
        self.assertIs(result, request)

    def test_memory_results_are_noted(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            {'call_id': 'p1:0', 'contract_id': 'demo/move',
             'results': [{'success': True,
                          'output': {'status': {'data': '{"error": "forward_m cannot be combined"}'}}}]}]}]))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 1)
        document = json.loads(result['messages'][0]['content'][len(MEMORY):])
        evidence = document['plans'][0]['calls'][0]['results'][0]
        self.assertIn('Embedded error: forward_m cannot be combined', evidence['monitor_note'])

    def test_an_existing_note_is_not_replaced(self):
        request = body(feedback('demo/move', self.NESTED))
        request['messages'][0]['content'] = RESULT + json.dumps({'leaf_result': {
            'call_id': 'c1', 'contract_id': 'demo/move', 'success': True,
            'output': json.loads(self.NESTED), 'monitor_note': 'other note'}})
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 0)
        self.assertEqual(leaf_of(result['messages'][0])['monitor_note'], 'other note')

    def test_only_the_latest_result_for_a_contract_is_noted(self):
        request = body(feedback('demo/move', self.NESTED, call_id='c1'),
                       feedback('demo/move', self.NESTED, call_id='c2'))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 1)
        self.assertNotIn('monitor_note', leaf_of(result['messages'][0]))
        self.assertIn('monitor_note', leaf_of(result['messages'][1]))

    def test_a_later_result_for_the_same_contract_replaces_the_error(self):
        request = body(feedback('demo/move', self.NESTED, call_id='c1'),
                       feedback('demo/move',
                                '{"status": {"data": "{\\"status\\": \\"done\\"}"}}', call_id='c2'))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 0)
        self.assertIs(result, request)

    def test_a_later_result_for_another_contract_keeps_the_error(self):
        request = body(feedback('demo/move', self.NESTED, call_id='c1'),
                       feedback('demo/scene/find', '{"objects": []}', call_id='c2'))
        result, info = add_embedded_error_notes(request, domain=DOMAIN)
        self.assertEqual(info['embedded_error_notes'], 1)
        self.assertIn('monitor_note', leaf_of(result['messages'][0]))
        self.assertNotIn('monitor_note', leaf_of(result['messages'][1]))

    def test_a_second_pass_changes_nothing(self):
        request = body(feedback('demo/move', self.NESTED))
        once, first = add_embedded_error_notes(request, domain=DOMAIN)
        twice, second = add_embedded_error_notes(once, domain=DOMAIN)
        self.assertEqual(first['embedded_error_notes'], 1)
        self.assertEqual(second['embedded_error_notes'], 0)
        self.assertEqual(once, twice)


class BothHintsTest(unittest.TestCase):
    def test_an_empty_latest_gets_only_the_empty_note(self):
        request = body(feedback('demo/scene/find', empty_output()))
        once, _ = add_empty_observation_notes(request, domain=DOMAIN)
        twice, info = add_uninspected_object_notes(once, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 0)
        self.assertEqual(leaf_of(twice['messages'][0])['monitor_note'], NOTE)

    def test_a_hit_latest_gets_only_the_location_note(self):
        request = body(feedback('demo/scene/find', hit('crate@1')))
        once, _ = add_empty_observation_notes(request, domain=DOMAIN)
        twice, info = add_uninspected_object_notes(once, domain=DOMAIN)
        self.assertEqual(info['uninspected_object_notes'], 1)
        self.assertEqual(leaf_of(twice['messages'][0])['monitor_note'], UNINSPECTED_NOTE)


class HoldingDeliveryNoteTest(unittest.TestCase):
    def withdraw(self, count, success=True):
        return feedback('demo/withdraw',
                        {'transferred_count': count, 'player_count_before': 0,
                         'held_after': count, 'container_count_before': 52,
                         'container_count_after': 52 - count}, success=success)

    def deposit(self, before, after):
        return feedback('demo/deposit',
                        {'transferred_count': before - after, 'player_count_before': before,
                         'held_after': after})

    def inventory(self, slots, success=True):
        return feedback('demo/inventory',
                        {'selected_slot': 0, 'slots': slots}, success=success)

    def test_a_later_result_while_holding_carries_the_note(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            {'call_id': 'p1:0', 'contract_id': 'demo/withdraw',
             'results': [{'success': True, 'output': {'held_after': 2}}]}]}]),
            feedback('demo/travel',
                     {'known': True, 'state': 'SUCCEEDED', 'current_floor': 'lab_1f'}, call_id='p2:0'))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 1)
        self.assertEqual(leaf_of(result['messages'][-1])['monitor_note'],
                         HOLDING_NOTE.format(count=2))

    def test_a_deposit_that_empties_the_inventory_suppresses_the_note(self):
        request = body(self.withdraw(2), self.deposit(2, 0),
                       feedback('demo/travel', {'state': 'SUCCEEDED'},
                                call_id='p3:0'))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 0)
        self.assertIs(result, request)

    def test_a_nonempty_inventory_result_establishes_holding(self):
        request = body(self.inventory([{'slot': 9, 'item': 'demo:bread', 'count': 2}]),
                       feedback('demo/scene/find', hit('crate@1'), call_id='p2:0'))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 1)
        self.assertEqual(leaf_of(result['messages'][-1])['monitor_note'],
                         HOLDING_NOTE.format(count=2))

    def test_an_inventory_of_air_only_does_not_establish_holding(self):
        request = body(self.inventory([{'slot': 9, 'item': 'demo:air', 'count': 0}]))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 0)
        self.assertIs(result, request)

    def test_a_failed_withdraw_does_not_establish_holding(self):
        request = body(self.withdraw(2, success=False))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 0)
        self.assertIs(result, request)

    def test_a_later_continue_that_empties_the_inventory_wins(self):
        request = body(memory([{'plan_id': 'p1', 'calls': [
            {'call_id': 'p1:0', 'contract_id': 'demo/withdraw',
             'results': [{'success': True, 'output': {'held_after': 2}}]}]}],
            continues=[{'plan_id': 'p1', 'result': {
                'success': True, 'contract_id': 'demo/deposit',
                'output': {'held_after': 0}}}]), self.deposit(2, 0))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 0)
        self.assertIs(result, request)

    def test_an_existing_note_is_not_replaced(self):
        request = body(self.withdraw(2),
                       feedback('demo/explore', {'state': 'SUCCEEDED'}, call_id='p2:0'))
        once, _ = add_explore_followup_notes(request, domain=DOMAIN)
        twice, info = add_holding_delivery_notes(once, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 0)
        self.assertIs(twice, once)

    def test_a_second_pass_changes_nothing(self):
        request = body(self.withdraw(2),
                       feedback('demo/travel', {'state': 'SUCCEEDED'},
                                call_id='p2:0'))
        once, first = add_holding_delivery_notes(request, domain=DOMAIN)
        twice, second = add_holding_delivery_notes(once, domain=DOMAIN)
        self.assertEqual(first['holding_delivery_notes'], 1)
        self.assertEqual(second['holding_delivery_notes'], 0)
        self.assertEqual(once, twice)

    def test_without_item_evidence_there_is_no_note(self):
        request = body(feedback('demo/travel', {'state': 'SUCCEEDED'}))
        result, info = add_holding_delivery_notes(request, domain=DOMAIN)
        self.assertEqual(info['holding_delivery_notes'], 0)
        self.assertIs(result, request)


NAV = 'demo/navigate'


def nav_call(call_id, room='scene.region.anno.1', success=False,
             error='room navigation timed out', timeout=180):
    return {'call_id': call_id, 'contract_id': NAV,
            'args': {'room_id': room, 'timeout_s': timeout},
            'results': [{'success': success, 'output': '{"state": "FAILED"}', 'error': error}]}


def nav_memory(*calls):
    return body(memory([{'plan_id': f'p{n}', 'calls': [call]} for n, call in enumerate(calls)]))


def memory_note(request, plan=0, call=0, result=0):
    document = json.loads(next(m['content'][len(MEMORY):] for m in request['messages']
                               if isinstance(m.get('content'), str)
                               and m['content'].startswith(MEMORY)))
    return document['plans'][plan]['calls'][call]['results'][result].get('monitor_note')


class RepeatFailureNoteTest(unittest.TestCase):
    def test_three_identical_failures_carry_the_note(self):
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'), nav_call('p2:0'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 1)
        self.assertGreater(info['repeat_failure_chars'], 0)
        note = memory_note(result, plan=2)
        self.assertEqual(note, REPEAT_FAILURE_ERROR_NOTE.format(
            count=3, error='room navigation timed out'))

    def test_two_failures_are_not_yet_repeated(self):
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)
        self.assertIs(result, request)

    def test_failures_of_different_arguments_are_not_identical(self):
        request = nav_memory(nav_call('p0:0', room='a'), nav_call('p1:0', room='a'),
                             nav_call('p2:0', room='b'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)

    def test_a_later_success_resets_the_streak(self):
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'), nav_call('p2:0'),
                             nav_call('p3:0'), nav_call('p4:0'),
                             nav_call('p5:0', success=True, error=''))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'),
                             nav_call('p2:0', success=True, error=''), nav_call('p3:0'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)

    def test_failures_after_a_success_still_form_a_streak(self):
        request = nav_memory(nav_call('p0:0', success=True, error=''),
                             nav_call('p1:0'), nav_call('p2:0'), nav_call('p3:0'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 1)
        self.assertIn('3 times', memory_note(result, plan=3))

    def test_a_failure_without_error_text_still_counts(self):
        request = nav_memory(nav_call('p0:0', error=''), nav_call('p1:0', error=''),
                             nav_call('p2:0', error=''))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 1)
        self.assertEqual(memory_note(result, plan=2), REPEAT_FAILURE_NOTE.format(count=3))

    def test_a_changed_timeout_is_a_different_call(self):
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'),
                             nav_call('p2:0', timeout=120))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)

    def test_the_note_lands_on_the_latest_result_of_the_streak(self):
        request = body(memory([{'plan_id': 'p0', 'calls': [
            {**nav_call('p0:0'), 'results': [nav_call('p0:0')['results'][0],
                                             nav_call('p1:0')['results'][0]]},
            nav_call('p2:0')]}]))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 1)
        self.assertIsNone(memory_note(result, plan=0, call=0, result=0))
        self.assertIn('3 times', memory_note(result, plan=0, call=1))

    def test_raw_result_messages_resolve_arguments_from_dispatch(self):
        calls = [nav_call('p0:0'), nav_call('p1:0'), nav_call('p2:0')]
        request = body(
            dispatch([{'call_id': 'p0:0', 'contract_id': NAV, 'args': calls[0]['args']}]),
            feedback(NAV, '{"state": "FAILED"}', success=False, call_id='p0:0'),
            dispatch([{'call_id': 'p1:0', 'contract_id': NAV, 'args': calls[1]['args']}]),
            feedback(NAV, '{"state": "FAILED"}', success=False, call_id='p1:0'),
            dispatch([{'call_id': 'p2:0', 'contract_id': NAV, 'args': calls[2]['args']}]),
            feedback(NAV, '{"state": "FAILED"}', success=False, call_id='p2:0'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 1)
        leaf = leaf_of(result['messages'][-1])
        self.assertIn('3 times', leaf['monitor_note'])

    def test_raw_results_without_arguments_are_not_guessed(self):
        request = body(feedback(NAV, '{"state": "FAILED"}', success=False, call_id='a'),
                       feedback(NAV, '{"state": "FAILED"}', success=False, call_id='b'),
                       feedback(NAV, '{"state": "FAILED"}', success=False, call_id='c'))
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)

    def test_an_existing_note_is_not_replaced(self):
        latest = nav_call('p2:0')
        latest['results'][0]['monitor_note'] = 'something else'
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'), latest)
        result, info = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(info['repeat_failure_notes'], 0)
        self.assertEqual(memory_note(result, plan=2), 'something else')

    def test_a_second_pass_changes_nothing(self):
        request = nav_memory(nav_call('p0:0'), nav_call('p1:0'), nav_call('p2:0'))
        _, first = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(first['repeat_failure_notes'], 1)
        _, second = add_repeat_failure_notes(request, domain=DOMAIN)
        self.assertEqual(second['repeat_failure_notes'], 0)
        self.assertEqual(second['repeat_failure_chars'], 0)


if __name__ == '__main__':
    unittest.main()
