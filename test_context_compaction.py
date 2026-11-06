import copy
import json
import unittest

from context_compaction import (DISPATCH, MEMORY, OBJECT_PROJECTION, OBJECTS_CONTRACT, RESULT, TASK,
                                compact_catalog_boilerplate,
                                compact_catalog_provider_descriptions, compact_output,
                                compact_request, project_observation)


def message(content, role='user', **extra):
    return {'role': role, 'content': content, **extra}


def encoded(prefix, value):
    return message(prefix + json.dumps(value, ensure_ascii=False))


def snapshot(name, running=False):
    return message(('## In-flight trees\nplan live-plan boundary_op=77\n\n' if running else '') +
                   '## Executor active plans (authoritative live snapshot)\n' +
                   ('snapshot_json: {"count":1,"plans":[{"plan_id":"live-plan"}]}' if running
                    else 'snapshot_json: {"count":0,"plans":[]}') +
                   '\n\n## Current environment and provider state\n' + name)


def fixture():
    msgs = [message('SYSTEM FORMAT RULES', 'system'),
            message('User message m1 (filed under T1): find two food items'),
            message('## Available capabilities\nEXACT CATALOG SCHEMA\n\n'
                    '## Capability docs\nEXACT MANUAL\n\n'
                    '## Executor active plans (authoritative live snapshot)\nold catalog state'),
            snapshot('OLD_FLOOR', running=True)]
    for i in range(4):
        msgs += [encoded(TASK, {'task_id': 'T1', 'goal': 'find food', 'status': 'in_progress', 'revision': i}),
                 encoded(DISPATCH, {'plan_id': f'p{i}', 'description': f'room {i}', 'calls': [
                     {'call_id': f'p{i}:0', 'op_id': i, 'contract_id': 'scene/find',
                      'args': {'region': str(401 + i), 'label': 'chest'}}]}),
                 message(f'Executor feedback scope: plan_id=p{i}'),
                 encoded(RESULT, {'leaf_result': {'call_id': f'p{i}:0', 'contract_id': 'scene/find',
                     'success': i != 1, 'output': json.dumps({'room': str(401 + i),
                     'objects': [{'id': f'box-{i}', 'label': 'demo:box'}]}),
                     'error': 'navigation timed out' if i == 1 else ''}}),
                 message('I will inspect the box.', 'assistant'), snapshot('HISTORICAL_FLOOR')]
    msgs += [snapshot('CURRENT_FLOOR_4'),
             message('## Tasks (current status board)\nT1 in_progress: find two food items'),
             message('Final response format contract for this round:\nJSON only')]
    return {'model': 'deepseek-flash', 'messages': msgs, 'response_format': {'schema_marker': 42}}


def memory(body):
    return json.loads(next(m['content'][len(MEMORY):] for m in body['messages']
                          if isinstance(m.get('content'), str) and m['content'].startswith(MEMORY)))


def objects_body(observations):
    msgs = [message('SYSTEM FORMAT RULES', 'system'),
            message('User message m1 (filed under T1): find the bread chest')]
    for n, (map_id, objects) in enumerate(observations):
        call_id = f'o{n}:0'
        output = objects if isinstance(objects, dict) else {'map_id': map_id, 'objects': objects}
        msgs += [encoded(DISPATCH, {'plan_id': f'o{n}', 'description': f'look {n}', 'calls': [
                     {'call_id': call_id, 'contract_id': OBJECTS_CONTRACT, 'args': {}}]}),
                 encoded(RESULT, {'leaf_result': {'call_id': call_id, 'contract_id': OBJECTS_CONTRACT,
                     'success': True, 'output': json.dumps(output), 'error': ''}}),
                 snapshot(f'FLOOR_{n}')]
    msgs += [snapshot('CURRENT_FLOOR'),
             message('Final response format contract for this round:\nJSON only')]
    return {'model': 'deepseek-flash', 'messages': msgs}


def task_board_body(board_state='succeeded', board_plans=None,
                    record_state='succeeded', record_extra=None):
    body = fixture()
    messages = body['messages']
    for i, item in enumerate(messages):
        content = item.get('content')
        if isinstance(content, str) and content.startswith(TASK):
            data = json.loads(content[len(TASK):])
            data.update({'goal': 'find food',
                         'success_criterion': 'two food items in the lobby chest',
                         'messages': ['m1'],
                         'plans': [{'plan_id': f'p{j}', 'state': record_state}
                                   for j in range(4)]})
            if record_extra:
                data.update(record_extra)
            messages[i] = message(TASK + json.dumps(data, ensure_ascii=False))
    plans = board_plans if board_plans is not None else [f'p{j}' for j in range(4)]
    messages[-2] = message(
        '## Tasks (current status board)\n- T1: in_progress\n'
        '  goal: find food\n'
        '  success criterion: two food items in the lobby chest\n'
        '  messages: m1\n' +
        ''.join(f'  plan {p} ({board_state}): room {p[1:]}\n' for p in plans))
    return body


def chest(name, **extra):
    return {'id': name, 'label': 'demo:box', 'x': 61.5, 'y': -50.0, 'z': 40.5,
            'frame_id': 'map', 'observation_count': 31, 'last_seen_unix': 1793463165.65, **extra}


def object_results(body):
    document = memory(body)
    outputs = {}
    for plan in document['plans']:
        for call in plan['calls']:
            for evidence in call.get('results', []):
                outputs[call['call_id']] = evidence['output']
    for item in document.get('continues', []):
        outputs[item['result']['call_id']] = item['result']['output']
    return outputs


class ContextCompactionTest(unittest.TestCase):
    def test_preserves_instructions_user_catalog_schema_and_all_results(self):
        original = fixture()
        before = copy.deepcopy(original)
        compacted, meta = compact_request(original, min_chars=0)
        self.assertEqual(original, before)
        self.assertTrue(meta['applied'])
        self.assertEqual(compacted['response_format'], original['response_format'])
        self.assertEqual(compacted['messages'][:2], original['messages'][:2])
        text = '\n'.join(m['content'] for m in compacted['messages'])
        self.assertIn('EXACT CATALOG SCHEMA', text)
        self.assertIn('EXACT MANUAL', text)
        self.assertIn('CURRENT_FLOOR_4', text)
        self.assertNotIn('OLD_FLOOR', text)
        self.assertNotIn('HISTORICAL_FLOOR', text)
        self.assertNotIn('boundary_op=77', text)
        plans = memory(compacted)['plans']
        self.assertEqual(len(plans), 4)
        for i, plan in enumerate(plans):
            call = plan['calls'][0]
            self.assertEqual(call['args']['region'], str(401 + i))
            self.assertEqual(call['results'][0]['output']['room'], str(401 + i))
        failed = plans[1]['calls'][0]['results'][0]
        self.assertFalse(failed['success'])
        self.assertEqual(failed['error'], 'navigation timed out')

    def test_result_values_duplicated_from_the_call_are_omitted(self):
        original = fixture()
        for i, item in enumerate(original['messages']):
            content = item.get('content')
            if isinstance(content, str) and content.startswith(RESULT):
                data = json.loads(content[len(RESULT):])
                leaf = data['leaf_result']
                output = json.loads(leaf['output'])
                output['region'] = str(401 + int(leaf['call_id'][1]))
                output['kept'] = 'not an argument'
                leaf['output'] = json.dumps(output)
                original['messages'][i] = message(RESULT + json.dumps(data, ensure_ascii=False))
        compacted, meta = compact_request(original, min_chars=0)
        self.assertEqual(meta['result_argument_fields_omitted'], 4)
        document = memory(compacted)
        for plan in document['plans']:
            output = plan['calls'][0]['results'][0]['output']
            self.assertNotIn('region', output)
            self.assertEqual(output['kept'], 'not an argument')

    def test_long_contiguous_empty_slot_runs_use_an_inclusive_range(self):
        value = {'slots': [{'slot': i, 'item': 'demo:air', 'count': 0} for i in range(26)],
                 'slot_count': 27}
        result = compact_output(json.dumps(value), empty_slot_items=('demo:air', 'air'))
        self.assertEqual(result['_monitor_observed_empty_slots'], '0-25')
        small = {'slots': [{'slot': i, 'item': 'demo:air', 'count': 0} for i in range(3)]}
        self.assertEqual(compact_output(json.dumps(small), empty_slot_items=('demo:air', 'air'))['_monitor_observed_empty_slots'], [0, 1, 2])
        split = {'slots': [{'slot': i, 'item': 'demo:air', 'count': 0} for i in [0, 1, 2, 4]]}
        self.assertEqual(compact_output(json.dumps(split), empty_slot_items=('demo:air', 'air'))['_monitor_observed_empty_slots'], [0, 1, 2, 4])

    def test_result_labels_duplicated_from_the_call_are_omitted(self):
        original = fixture()
        compacted, meta = compact_request(original, min_chars=0)
        self.assertGreater(meta['result_fields_omitted'], 0)
        document = memory(compacted)
        self.assertIn('do not repeat that call', document['note'])
        for plan in document['plans']:
            call = plan['calls'][0]
            result = call['results'][0]
            self.assertEqual(result['success'], plan['plan_id'] != 'p1')
            self.assertNotIn('call_id', result)
            self.assertNotIn('contract_id', result)
            self.assertNotIn('_monitor_history_index', result)
            if result['success']:
                self.assertNotIn('error', result)
        failed = document['plans'][1]['calls'][0]['results'][0]
        self.assertEqual(failed['error'], 'navigation timed out')

    def test_catalog_doc_pointers_collapse_to_one_note(self):
        catalog = ('## Available capabilities\n'
                   'This is the complete catalog; it replaces any earlier catalog in this conversation.\n'
                   '- capability_name: x.ping\n  description: "Ping"\n'
                   '  more: call `read_capability_doc` with provider_id `x` for this capability\'s full description\n'
                   '- capability_name: y.ping\n  description: "Ping"\n'
                   '  more: call `read_capability_doc` with provider_id `y` for this capability\'s full description\n\n'
                   '## Capability docs\n- `x` — use when: ping')
        text, removed = compact_catalog_boilerplate(catalog)
        self.assertEqual(removed, 1)
        self.assertIn('Providers with a manual', text)
        self.assertNotIn('provider_id `x`', text)
        self.assertIn('provider_id `y`', text)
        self.assertIn('- `x` — use when: ping', text)
        again, removed = compact_catalog_boilerplate(text)
        self.assertEqual(removed, 0)
        self.assertEqual(again, text)

    def test_a_provider_wide_description_is_stated_once(self):
        shared = 'Turn sensor streams into a map and persist it for re-localization.'
        catalog = ('## Available capabilities\n'
                   'This is the complete catalog; it replaces any earlier catalog in this conversation.\n'
                   f'- capability_name: m.get\n  description: {json.dumps(shared)}\n'
                   '  args_schema: {"type":"object","properties":{}}\n'
                   f'- capability_name: m.set\n  description: {json.dumps(shared)}\n'
                   '  args_schema: {"type":"object","properties":{"on":{"type":"boolean"}}}\n'
                   '- capability_name: scene.find\n  description: "Find one object."\n'
                   '  args_schema: {"type":"object","properties":{"label":{"type":"string"}}}\n')
        text, hoisted, saved = compact_catalog_provider_descriptions(catalog)
        self.assertEqual(hoisted, 2)
        self.assertEqual(saved, 0)
        self.assertEqual(text.count(json.dumps(shared)), 1)
        self.assertIn('- provider_description: m\n', text)
        self.assertEqual(text.count('- capability_name: m.'), 2)
        self.assertIn('args_schema: {"type":"object","properties":{"on":{"type":"boolean"}}}', text)
        self.assertIn('- capability_name: scene.find\n  description: "Find one object."', text)
        self.assertIn('provider_description', text)
        again, hoisted, saved = compact_catalog_provider_descriptions(text)
        self.assertEqual((again, hoisted, saved), (text, 0, 0))

    def test_a_description_repeated_verbatim_is_written_once(self):
        half = 'SLAM mapping builds a live occupancy grid and persists named maps.'
        catalog = ('## Available capabilities\n'
                   'This is the complete catalog; it replaces any earlier catalog in this conversation.\n'
                   f'- capability_name: m.get\n  description: {json.dumps(half + " " + half)}\n'
                   '  args_schema: {"type":"object","properties":{}}\n')
        text, hoisted, saved = compact_catalog_provider_descriptions(catalog)
        self.assertEqual(hoisted, 0)
        self.assertEqual(saved, len(half) + 1)
        self.assertIn(json.dumps(half), text)
        self.assertNotIn(json.dumps(half + " " + half), text)
        again, hoisted, saved = compact_catalog_provider_descriptions(text)
        self.assertEqual((again, hoisted, saved), (text, 0, 0))

    def test_distinct_descriptions_and_the_docs_section_are_untouched(self):
        catalog = ('## Available capabilities\n'
                   'This is the complete catalog; it replaces any earlier catalog in this conversation.\n'
                   '- capability_name: a.one\n  description: "First."\n'
                   '  args_schema: {"type":"object","properties":{}}\n'
                   '- capability_name: a.two\n  description: "Second."\n'
                   '  args_schema: {"type":"object","properties":{}}\n\n'
                   '## Capability docs\n- `a` — use when: anything')
        text, hoisted, saved = compact_catalog_provider_descriptions(catalog)
        self.assertEqual((text, hoisted, saved), (catalog, 0, 0))
        self.assertNotIn('provider_description', text)

    def test_task_record_fields_already_on_the_board_are_omitted(self):
        original = task_board_body()
        compacted, meta = compact_request(original, min_chars=0)
        self.assertEqual(meta['task_record_fields_omitted'], 4)
        records = [m['content'] for m in compacted['messages']
                   if isinstance(m.get('content'), str) and m['content'].startswith(TASK)]
        self.assertEqual(len(records), 1)
        record = records[0]
        payload = json.loads(record[record.find('{'):])
        self.assertEqual(payload['task_id'], 'T1')
        self.assertEqual(payload['status'], 'in_progress')
        self.assertEqual(payload['revision'], 3)
        for key in ('plans', 'goal', 'success_criterion', 'messages'):
            self.assertNotIn(key, payload)
        self.assertIn('Tasks (current status board)', record)
        text = '\n'.join(m['content'] for m in compacted['messages'])
        self.assertIn('  plan p0 (succeeded): room 0', text)
        self.assertEqual(meta['plan_descriptions_omitted'], 4)
        for plan in memory(compacted)['plans']:
            self.assertNotIn('description', plan)
        self.assertIn('plan description identical', memory(compacted)['note'])
        self.assertEqual(memory(compacted)['plans'][0]['plan_id'], 'p0')
        again, meta = compact_request(compacted, min_chars=0)
        self.assertIn('Tasks (current status board)',
                      next(m['content'] for m in again['messages']
                           if isinstance(m.get('content'), str)
                           and m['content'].startswith(TASK)))

    def test_a_plan_state_that_differs_from_the_board_is_kept(self):
        original = task_board_body(board_state='failed')
        compacted, meta = compact_request(original, min_chars=0)
        self.assertEqual(meta['task_record_fields_omitted'], 3)
        record = next(m['content'] for m in compacted['messages']
                      if isinstance(m.get('content'), str) and m['content'].startswith(TASK))
        payload = json.loads(record[record.find('{'):])
        self.assertEqual([p['plan_id'] for p in payload['plans']], ['p0', 'p1', 'p2', 'p3'])
        self.assertNotIn('goal', payload)

    def test_a_plan_missing_from_the_board_keeps_the_record_plans(self):
        original = task_board_body(board_plans=['p0', 'p1', 'p2'])
        compacted, meta = compact_request(original, min_chars=0)
        record = next(m['content'] for m in compacted['messages']
                      if isinstance(m.get('content'), str) and m['content'].startswith(TASK))
        payload = json.loads(record[record.find('{'):])
        self.assertEqual(len(payload['plans']), 4)

    def test_a_plan_description_absent_from_the_board_is_kept(self):
        original = task_board_body(board_plans=['p0', 'p1', 'p2'])
        compacted, meta = compact_request(original, min_chars=0)
        self.assertEqual(meta['plan_descriptions_omitted'], 3)
        plans = memory(compacted)['plans']
        self.assertNotIn('description', plans[0])
        self.assertEqual(plans[3]['description'], 'room 3')

    def test_current_running_plan_and_semantic_stop_boundary_are_preserved(self):
        original = fixture()
        original['messages'].insert(-1, snapshot('CURRENT_ACTIVE_FLOOR', running=True))
        compacted, _ = compact_request(original, min_chars=0)
        text = '\n'.join(m['content'] for m in compacted['messages'])
        self.assertIn('boundary_op=77', text)
        self.assertIn('"count":1', text)
        self.assertNotIn('CURRENT_FLOOR_4', text)

    def test_user_corrections_and_multiple_task_states_are_preserved(self):
        original = fixture()
        original['messages'] += [message('User message m2 (filed under T1): stop taking food; return now'),
                                  encoded(TASK, {'task_id': 'T2', 'status': 'in_progress', 'goal': 'greet'})]
        compacted, _ = compact_request(original, min_chars=0)
        text = '\n'.join(m['content'] for m in compacted['messages'])
        self.assertIn('stop taking food; return now', text)
        self.assertIn('"task_id": "T2"', text)
        self.assertIn('"revision": 3', text)
        self.assertNotIn('"revision": 0', text)

    def test_empty_slots_retain_exact_slot_ids_and_nonempty_counts(self):
        value = {'slots': [{'slot': 0, 'item': 'demo:bread', 'count': 2},
                           {'slot': 1, 'item': 'demo:air', 'count': 0},
                           {'slot': 2, 'item': 'demo:air', 'count': 0, 'unknown_data': True}],
                 'slot_count': 27, 'status': 'true', 'room': '401'}
        result = compact_output(json.dumps(value), empty_slot_items=('demo:air', 'air'))
        self.assertEqual(result['_monitor_observed_empty_slots'], [1])
        self.assertEqual(result['slots'], [value['slots'][0], value['slots'][2]])
        self.assertEqual(result['slot_count'], 27)
        self.assertEqual(result['room'], '401')
        self.assertEqual(result['status'], 'true')
        partial = {'_robonix_truncation': {'complete': False}, 'slots': value['slots'], 'preview': 'partial'}
        self.assertEqual(compact_output(json.dumps(partial), empty_slot_items=('demo:air', 'air')), partial)

    def test_unknown_messages_and_native_tool_call_groups_are_preserved(self):
        original = fixture()
        additions = [message('opaque feedback must survive'),
                     message(None, 'assistant', tool_calls=[{'id': 'tc', 'type': 'function',
                             'function': {'name': 'read', 'arguments': '{}'}}]),
                     message('manual text', 'tool', tool_call_id='tc')]
        original['messages'][5:5] = additions
        compacted, _ = compact_request(original, min_chars=0)
        for item in additions:
            self.assertIn(item, compacted['messages'])

    def test_unmatched_or_malformed_feedback_is_not_discarded(self):
        original = fixture()
        additions = [message('Executor feedback scope: unknown'),
                     message(RESULT + '{malformed'), encoded(TASK, {'unknown_task_format': True}),
                     encoded(RESULT, {'leaf_result': {'call_id': 'unknown', 'contract_id': 'scene/find',
                                                      'success': False, 'error': 'unknown failure'}})]
        original['messages'][4:4] = additions
        compacted, _ = compact_request(original, min_chars=0)
        for item in additions:
            self.assertIn(item, compacted['messages'])

    def test_successive_feedback_and_unfinished_dispatch_are_not_fabricated(self):
        original = fixture()
        original['messages'] += [encoded(RESULT, {'leaf_result': {'call_id': 'p0:0',
            'contract_id': 'scene/find', 'success': False, 'error': 'provider went offline'}}),
            encoded(DISPATCH, {'plan_id': 'pending', 'calls': [{'call_id': 'pending:0',
                'contract_id': 'ride', 'args': {'target_floor': 'lab_1f'}}]})]
        compacted, _ = compact_request(original, min_chars=0)
        plans = memory(compacted)['plans']
        self.assertEqual(len(plans[0]['calls'][0]['results']), 2)
        self.assertNotIn('results', plans[-1]['calls'][0])

    def test_only_current_retry_is_retained(self):
        original = fixture()
        old = message('Your previous RTDL response was rejected by Pilot. OLD')
        new = message('Your previous RTDL response was rejected by Pilot. CURRENT')
        original['messages'].insert(4, old)
        original['messages'].append(new)
        compacted, _ = compact_request(original, min_chars=0)
        self.assertNotIn(old, compacted['messages'])
        self.assertIn(new, compacted['messages'])

    def test_ambiguous_call_ids_are_kept_in_original_messages(self):
        original = fixture()
        duplicate = encoded(DISPATCH, {'plan_id': 'duplicate', 'calls': [
            {'call_id': 'p0:0', 'contract_id': 'scene/find', 'args': {'region': 'different'}}]})
        original['messages'].append(duplicate)
        compacted, meta = compact_request(original, min_chars=0)
        self.assertTrue(meta['applied'])
        self.assertIn(duplicate, compacted['messages'])
        self.assertNotIn('p0', [p['plan_id'] for p in memory(compacted)['plans']])
        self.assertTrue(any(m.get('content', '').startswith(RESULT) and 'p0:0' in m['content']
                            for m in compacted['messages']))

    def test_short_history_bypass_and_idempotence(self):
        original = fixture()
        bypass, meta = compact_request(original)
        self.assertIs(bypass, original)
        self.assertFalse(meta['applied'])
        compacted, meta = compact_request(original, min_chars=0)
        again, meta = compact_request(compacted, min_chars=0)
        self.assertIs(again, compacted)
        self.assertFalse(meta['applied'])

    def test_a_later_round_folds_into_a_second_block_without_touching_the_first(self):
        compacted, _ = compact_request(fixture(), min_chars=0)
        first = next(m['content'] for m in compacted['messages']
                     if isinstance(m.get('content'), str) and m['content'].startswith(MEMORY))
        compacted['messages'] += [snapshot('SUPERSEDED_FLOOR_9'),
                                  message('I will search the next room. ' + 'n' * 2000, 'assistant'),
                                  encoded(TASK, {'task_id': 'T1', 'revision': 9, 'status': 'in_progress'}),
                                  encoded(DISPATCH, {'plan_id': 'p9', 'description': 'room 9', 'calls': [
                                      {'call_id': 'p9:0', 'op_id': 9, 'contract_id': 'scene/find',
                                       'args': {'region': '409', 'label': 'chest'}},
                                      {'call_id': 'p9:1', 'op_id': 10, 'contract_id': 'scene/find',
                                       'args': {'region': '410', 'label': 'chest'}}]}),
                                  message('Executor feedback scope: plan_id=p9'),
                                  encoded(RESULT, {'leaf_result': {'call_id': 'p9:0', 'contract_id': 'scene/find',
                                      'success': True, 'output': json.dumps({'room': '409', 'objects': []}),
                                      'error': ''}}),
                                  encoded(RESULT, {'leaf_result': {'call_id': 'p9:1', 'contract_id': 'scene/find',
                                      'success': True, 'output': json.dumps({'room': '410', 'objects': []}),
                                      'error': ''}}),
                                  snapshot('CURRENT_FLOOR_9')]
        again, meta = compact_request(compacted, min_chars=0)
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['memory_blocks'], 2)
        self.assertEqual(meta['historical_plans'], 1)
        memories = [m['content'] for m in again['messages']
                    if isinstance(m.get('content'), str) and m['content'].startswith(MEMORY)]
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0], first)
        self.assertIn('"plan_id":"p9"', memories[1])
        self.assertNotIn('"plan_id":"p0"', memories[1])
        text = '\n'.join(m['content'] for m in again['messages'])
        self.assertIn('CURRENT_FLOOR_9', text)
        self.assertNotIn('SUPERSEDED_FLOOR_9', text)
        self.assertNotIn('I will search the next room', text)

    def test_results_for_an_earlier_block_are_kept_as_continuations(self):
        compacted, _ = compact_request(fixture(), min_chars=0)
        compacted['messages'] += [snapshot('SUPERSEDED_FLOOR_5'),
                                  message('Waiting for the earlier plan. ' + 'w' * 1500, 'assistant'),
                                  snapshot('CURRENT_FLOOR_5'),
                                  encoded(RESULT, {'leaf_result': {'call_id': 'p0:0', 'contract_id': 'scene/find',
                                      'success': False, 'error': 'later timeout'}}),
                                  message('Executor feedback scope: plan_id=p0'),
                                  encoded(RESULT, {'leaf_result': {'call_id': 'p1:0', 'contract_id': 'scene/find',
                                      'success': True, 'output': json.dumps({'room': '402'})}})]
        again, meta = compact_request(compacted, min_chars=0)
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['continued_results'], 2)
        document = only_memory(again, last=True)
        self.assertEqual(document['plans'], [])
        self.assertEqual([item['plan_id'] for item in document['continues']], ['p0', 'p1'])
        self.assertEqual(document['continues'][0]['result']['error'], 'later timeout')
        self.assertEqual(document['continues'][1]['result']['output'], {'room': '402'})
        for item in document['continues']:
            self.assertIn('call_id', item['result'])
            self.assertIn('contract_id', item['result'])

    def test_repeat_without_new_execution_evidence_is_a_noop(self):
        compacted, _ = compact_request(fixture(), min_chars=0)
        compacted['messages'] += [snapshot('CURRENT_FLOOR_6'),
                                  message('I am still waiting for the plan.', 'assistant')]
        again, meta = compact_request(compacted, min_chars=0)
        self.assertIs(again, compacted)
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'insufficient_new_evidence')

    def test_unreadable_memory_blocks_rewrites(self):
        original = fixture()
        original['messages'].append(message(MEMORY + '{not json'))
        body, meta = compact_request(original, min_chars=0)
        self.assertIs(body, original)
        self.assertEqual(meta['reason'], 'unreadable_memory')

    def test_superseded_object_lists_become_deltas_but_the_latest_stays_full(self):
        a, b, c = chest('chest-a'), chest('chest-b'), chest('chest-c')
        body = objects_body([('lab_3f', [a, b]), ('lab_3f', [a, b, c]), ('lab_3f', [a, b, c]),
                             ('lab_4f', [chest('chest-d')])])
        compacted, meta = compact_request(body, min_chars=0)
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['projected_observations'], 1)
        self.assertGreater(meta['projection_saved_chars'], 0)
        outputs = object_results(compacted)
        self.assertNotIn('_monitor_projection', outputs['o0:0'])
        self.assertEqual(outputs['o0:0']['objects'], [a, b])
        self.assertEqual(outputs['o1:0']['_monitor_projection'], OBJECT_PROJECTION)
        self.assertEqual([e['id'] for e in outputs['o1:0']['new_objects']], ['chest-c'])
        self.assertEqual(outputs['o1:0']['observed_object_count'], 3)
        self.assertNotIn('_monitor_projection', outputs['o2:0'])
        self.assertEqual([e['id'] for e in outputs['o2:0']['objects']],
                         ['chest-a', 'chest-b', 'chest-c'])
        self.assertNotIn('_monitor_projection', outputs['o3:0'])
        state = set()
        for call_id in ('o0:0', 'o1:0', 'o2:0'):
            output = outputs[call_id]
            if 'objects' in output:
                state = {e['id'] for e in output['objects']}
            else:
                state |= {e['id'] for e in output['new_objects']}
                state -= set(output.get('removed_object_ids', []))
        self.assertEqual(state, {'chest-a', 'chest-b', 'chest-c'})

    def test_object_updates_and_removals_are_explicit(self):
        a = chest('chest-a')
        a2 = chest('chest-a', x=1.5)
        b = chest('chest-b')
        c = chest('chest-c')
        body = objects_body([('lab_3f', [a, b, c]), ('lab_3f', [a2, b, c]),
                             ('lab_3f', [a2, c]), ('lab_4f', [chest('chest-z')]),
                             ('lab_3f', [a2, c])])
        compacted, _ = compact_request(body, min_chars=0)
        outputs = object_results(compacted)
        self.assertEqual(outputs['o1:0']['_monitor_projection'], OBJECT_PROJECTION)
        self.assertEqual(outputs['o1:0']['new_objects'], [])
        self.assertEqual(outputs['o1:0']['updated_objects'], [a2])
        self.assertEqual(outputs['o1:0']['observed_object_count'], 3)
        self.assertEqual(outputs['o2:0']['_monitor_projection'], OBJECT_PROJECTION)
        self.assertEqual(outputs['o2:0']['new_objects'], [])
        self.assertEqual(outputs['o2:0']['removed_object_ids'], ['chest-b'])
        self.assertEqual(outputs['o2:0']['observed_object_count'], 2)
        self.assertNotIn('_monitor_projection', outputs['o4:0'])
        self.assertEqual(outputs['o4:0']['objects'], [a2, c])

    def test_object_state_is_per_map(self):
        shared, a1 = chest('shared'), chest('only-3f')
        a2 = chest('only-4f')
        b1, b2 = chest('new-3f'), chest('new-4f')
        body = objects_body([('lab_3f', [shared, a1]), ('lab_4f', [shared, a2]),
                             ('lab_3f', [shared, a1, b1]), ('lab_4f', [shared, a2, b2]),
                             ('lab_3f', [shared, a1, b1])])
        compacted, _ = compact_request(body, min_chars=0)
        outputs = object_results(compacted)
        self.assertEqual([e['id'] for e in outputs['o2:0']['new_objects']], ['new-3f'])
        self.assertNotIn('_monitor_projection', outputs['o3:0'])
        self.assertEqual([e['id'] for e in outputs['o3:0']['objects']],
                         ['shared', 'only-4f', 'new-4f'])
        self.assertNotIn('_monitor_projection', outputs['o4:0'])

    def test_truncated_and_malformed_object_lists_are_untouched(self):
        truncated = {'map_id': 'lab_3f', '_robonix_truncation': {'complete': False},
                     'objects': [{'id': 'maybe'}]}
        partial_index = {'map_id': 'lab_3f', 'objects': [{'id': 'maybe'}, {'id': 'maybe-2'}],
                         '_robonix_truncation': {'truncated': True, 'complete_record_index': False}}
        malformed = {'map_id': 'lab_4f', 'objects': [{'label': 'no-id'}]}
        a = chest('chest-a')
        body = objects_body([('lab_3f', truncated), ('lab_4f', malformed), ('lab_3f', [a]),
                             ('lab_3f', [a, chest('chest-b')]), ('lab_4f', [chest('chest-z')]),
                             ('lab_3f', partial_index)])
        compacted, meta = compact_request(body, min_chars=0)
        outputs = object_results(compacted)
        self.assertEqual(outputs['o0:0'], truncated)
        self.assertEqual(outputs['o1:0'], malformed)
        self.assertEqual(outputs['o5:0'], partial_index)
        self.assertNotIn('_monitor_projection', outputs['o2:0'])
        self.assertEqual([e['id'] for e in outputs['o2:0']['objects']], ['chest-a'])
        self.assertEqual(meta['projected_observations'], 0)

    def test_an_index_complete_truncation_still_allows_deltas(self):
        def index_only(objects):
            return {'map_id': 'lab_3f', 'stamp_unix': 1.0, 'objects': objects,
                    '_robonix_truncation': {'truncated': True, 'complete_record_index': True,
                                            'omitted_fields': True, 'total_records': len(objects),
                                            'returned_records': len(objects)}}
        small = chest('chest-a')
        other = chest('chest-b')
        body = objects_body([('lab_3f', index_only([small])),
                             ('lab_3f', index_only([small, other])),
                             ('lab_3f', index_only([small, other])),
                             ('lab_4f', [chest('chest-z')])])
        compacted, meta = compact_request(body, min_chars=0)
        outputs = object_results(compacted)
        self.assertEqual(outputs['o1:0']['_monitor_projection'], OBJECT_PROJECTION)
        self.assertEqual([e['id'] for e in outputs['o1:0']['new_objects']], ['chest-b'])
        self.assertEqual(outputs['o1:0']['observed_object_count'], 2)
        self.assertNotIn('_monitor_projection', outputs['o2:0'])
        self.assertEqual(len(outputs['o2:0']['objects']), 2)
        self.assertEqual(meta['projected_observations'], 1)

    def test_a_new_block_projects_against_the_earlier_block(self):
        a, b = chest('chest-a'), chest('chest-b')
        body = objects_body([('lab_3f', [a]), ('lab_3f', [a, b]), ('lab_3f', [a, b]),
                             ('lab_4f', [chest('chest-z')])])
        compacted, _ = compact_request(body, min_chars=0)
        first_content = next(m['content'] for m in compacted['messages']
                             if isinstance(m.get('content'), str) and m['content'].startswith(MEMORY))
        compacted['messages'] += [
            snapshot('SUPERSEDED_FLOOR_9'),
            message('I am checking the same room again. ' + 'r' * 2000, 'assistant'),
            encoded(DISPATCH, {'plan_id': 'later', 'calls': [
                {'call_id': 'later:0', 'contract_id': OBJECTS_CONTRACT, 'args': {}},
                {'call_id': 'later:1', 'contract_id': OBJECTS_CONTRACT, 'args': {}}]}),
            encoded(RESULT, {'leaf_result': {'call_id': 'later:0', 'contract_id': OBJECTS_CONTRACT,
                'success': True, 'output': json.dumps({'map_id': 'lab_3f', 'objects': [a, b]})}}),
            encoded(RESULT, {'leaf_result': {'call_id': 'later:1', 'contract_id': OBJECTS_CONTRACT,
                'success': True, 'output': json.dumps({'map_id': 'lab_3f',
                    'objects': [a, b, chest('chest-c')]})}}),
            snapshot('CURRENT_FLOOR_9')]
        again, meta = compact_request(compacted, min_chars=0)
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['memory_blocks'], 2)
        memories = [m['content'] for m in again['messages']
                    if isinstance(m.get('content'), str) and m['content'].startswith(MEMORY)]
        self.assertEqual(memories[0], first_content)
        document = only_memory(again, last=True)
        first_output = document['plans'][0]['calls'][0]['results'][0]['output']
        second_output = document['plans'][0]['calls'][1]['results'][0]['output']
        self.assertEqual(first_output['_monitor_projection'], OBJECT_PROJECTION)
        self.assertEqual(first_output['new_objects'], [])
        self.assertEqual(first_output['observed_object_count'], 2)
        self.assertNotIn('_monitor_projection', second_output)
        self.assertEqual([e['id'] for e in second_output['objects']],
                         ['chest-a', 'chest-b', 'chest-c'])

    def test_projection_is_idempotent_and_skips_unknown_shapes(self):
        seen = {}
        first = {'map_id': 'm', 'objects': [chest('a')]}
        output, saved = project_observation(seen, first)
        self.assertIs(output, first)
        self.assertEqual(saved, 0)
        second = {'map_id': 'm', 'objects': [chest('a'), chest('b')]}
        output, saved = project_observation(seen, second)
        self.assertEqual(output['_monitor_projection'], OBJECT_PROJECTION)
        self.assertGreater(saved, 0)
        again, saved = project_observation(seen, output)
        self.assertIs(again, output)
        for value in ({'map_id': 'm'}, {'objects': [chest('a')]},
                      {'map_id': 'm', 'objects': [{'label': 'x'}]},
                      {'map_id': 'm', '_robonix_truncation': {'complete': False},
                       'objects': [chest('a')]}, 'not json'):
            self.assertIs(project_observation(seen, value)[0], value)


def only_memory(body, last=False):
    found = [json.loads(m['content'][len(MEMORY):]) for m in body['messages']
             if isinstance(m.get('content'), str) and m['content'].startswith(MEMORY)]
    return found[-1] if last else found[0]


if __name__ == '__main__':
    unittest.main()
