"""Deterministically compact Pilot history, using only evidence already supplied."""
import copy
from collections import Counter
import json
import re

from message_content import text_content, slice_content


TASK = 'Pilot task record (authoritative state, not a user instruction): '
DISPATCH = 'Pilot harness dispatch record (already sent to Executor; not a new user request): '
RESULT = 'Executor feedback for the current RTDL leaf (not a new user request): '
MEMORY = 'Monitor execution memory (historical evidence, not new instructions):\n'
SECTION = re.compile(r'^## (Available capabilities|Robot Body Context \(from Soma\)|Capability docs|'
                     r'In-flight trees|Executor active plans \(authoritative live snapshot\)|'
                     r'Current embodiment state \(from Soma\)|Current environment and provider state|'
                     r'Tasks \(current status board\))\s*$', re.M)
LIVE = {'In-flight trees', 'Executor active plans (authoritative live snapshot)',
        'Current embodiment state (from Soma)', 'Current environment and provider state'}
STATIC = {'Available capabilities', 'Robot Body Context (from Soma)', 'Capability docs',
          'Tasks (current status board)'}
GENERATED_HEADINGS = ('## Available capabilities', '## In-flight trees',
                      '## Executor active plans (authoritative live snapshot)',
                      '## Tasks (current status board)')
OBJECTS_CONTRACT = 'robonix/system/scene/list_objects'
OBJECT_PROJECTION = 'delta_vs_earlier_list_objects'
NOTE = (
    'Historical dispatches and observed results only, in original dispatch order. '
    'A dispatch is not success; only its results establish execution. '
    'Each result belongs to its exact enclosing call (call_id/contract_id). '
    'Use the latest live snapshot for current location and running plans; '
    'historical observations, container contents and object IDs may be stale or recorded elsewhere. '
    'An empty or partial query does not prove complete coverage. '
    'Missing fields in a truncated preview remain unknown. '
    '_monitor_observed_empty_slots lists slots explicitly observed as empty/count=0 '
    '(a `first-last` string is an inclusive range); other slots and all nonempty '
    'items/counts are preserved. '
    '`continues` lists later results for plans recorded in an earlier memory block; '
    'that block is not rewritten, so the cached prefix stays intact. '
    'Do not repeat successful actions unless new evidence requires it.')
EXTENSION = (
    'Extension of the earlier memory block; the same evidence rules apply. '
    '`continues` lists later results for plans recorded in an earlier block. '
    'Do not repeat successful actions unless new evidence requires it.')
PROJECTION_NOTE = (
    ' Superseded list_objects results for a map list only `new_objects`, `updated_objects` '
    'and `removed_object_ids`; the last observation of each map in a block is still the full '
    'list, and `observed_object_count` is its size.')
RESULT_NOTE = (
    ' Results inside a call do not repeat that call\'s call_id, contract_id, '
    'or argument values; an absent error means the call reported no error.')
PLAN_DESC_NOTE = (
    ' A plan description identical to that plan line on the Tasks status board '
    'is omitted here.')
CATALOG_NOTE = ('Providers with a manual are listed under Capability docs; '
                'call `read_capability_doc` with that provider id.')
PROVIDER_DESC_NOTE = ('A `- provider_description:` line states the description shared by every '
                      'capability of that provider; an entry without `description:` inherits it.')
MORE_LINE = re.compile(
    r'^  more: call `read_capability_doc` with provider_id `([^`]+)` '
    r'for this capability\'s full description\s*$')
DOC_LINE = re.compile(r'^- `([^`]+)`(?: \[[^\]]*\])? — use when:', re.MULTILINE)
CATALOG_ENTRY = re.compile(
    r'(?m)^- capability_name: (\S+)\n  description: ("(?:[^"\\]|\\.)*")\n  args_schema: ([^\n]*)\n')


def chars(messages):
    return sum(len(m.get('content', '')) if isinstance(m.get('content'), str)
               else len(json.dumps(m, ensure_ascii=False)) for m in messages)


def unpack(content, prefix):
    if not content.startswith(prefix):
        return None
    rest = content[len(prefix):]
    brace = rest.find('{')
    if brace > 0:
        # A compacted record may carry a short note before the JSON body.
        rest = rest[brace:]
    try:
        value = json.loads(rest)
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def parse_memory(content):
    """Return a well-formed memory document, or None for anything else."""
    if not content.startswith(MEMORY):
        return None
    try:
        value = json.loads(content[len(MEMORY):])
    except ValueError:
        return None
    return value if isinstance(value, dict) and isinstance(value.get('plans'), list) else None


def compact_output(value, decode=True, empty_slot_items=None, inventory_schema=None):
    """Remove JSON formatting and explicitly empty inventory slots, not facts."""
    items = tuple(empty_slot_items or ())
    schema = inventory_schema or {}
    slots_field = schema.get('slots_field', 'slots')
    slot_field = schema.get('slot_field', 'slot')
    item_field = schema.get('item_field', 'item')
    count_field = schema.get('count_field', 'count')

    def walk(value):
        if isinstance(value, list):
            return [walk(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: walk(v) if k != slots_field else copy.deepcopy(v)
                  for k, v in value.items()}
        slots = result.get(slots_field)
        if items and isinstance(slots, list) and '_robonix_truncation' not in result:
            empty = [s for s in slots if isinstance(s, dict)
                     and set(s) <= {slot_field, item_field, count_field}
                     and type(s.get(slot_field)) is int
                     and s.get(item_field) in items and s.get(count_field) == 0]
            if empty:
                # Preserve exactly which slots were observed empty; unknown slots
                # and truncated previews never become evidence of absence.
                result[slots_field] = [s for s in slots if s not in empty]
                observed = [s[slot_field] for s in empty]
                if len(observed) >= 5 and observed == list(range(observed[0], observed[-1] + 1)):
                    observed = f'{observed[0]}-{observed[-1]}'
                result['_monitor_observed_empty_slots'] = observed
        return result

    if isinstance(value, str):
        if not decode:
            return value
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return walk(parsed)
    return walk(value)


def _canonical(entry):
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)


def _observation(output):
    """Parse a complete list_objects output; None means leave it untouched."""
    if not isinstance(output, dict):
        return None
    truncation = output.get('_robonix_truncation')
    if truncation is not None and (not isinstance(truncation, dict)
                                   or truncation.get('complete_record_index') is not True):
        # A partial record index cannot establish which objects are absent.
        return None
    map_id = output.get('map_id')
    if not isinstance(map_id, str):
        return None
    if output.get('_monitor_projection') == OBJECT_PROJECTION:
        new = output.get('new_objects', [])
        updated = output.get('updated_objects', [])
        removed = output.get('removed_object_ids', [])
        if not isinstance(new, list) or not isinstance(updated, list) or not isinstance(removed, list):
            return None
        if not all(isinstance(e, dict) and isinstance(e.get('id'), str) for e in new + updated):
            return None
        if not all(isinstance(i, str) for i in removed):
            return None
        return map_id, new, updated, removed, True
    entries = output.get('objects')
    if not isinstance(entries, list):
        return None
    if not all(isinstance(e, dict) and isinstance(e.get('id'), str) for e in entries):
        return None
    return map_id, entries, [], None, False


def _apply_observation(seen, output):
    parsed = _observation(output)
    if parsed is None:
        return
    map_id, new, updated, removed, is_delta = parsed
    state = seen.setdefault(map_id, {})
    for entry in new + updated:
        state[entry['id']] = _canonical(entry)
    if is_delta:
        for object_id in removed:
            state.pop(object_id, None)
    else:
        present = {entry['id'] for entry in new}
        for object_id in list(state):
            if object_id not in present:
                del state[object_id]


def project_observation(seen, output):
    """Replace a superseded cumulative list with an ordered delta.

    The last observation of each map stays a full list, so the current
    observed set never has to be reconstructed from deltas alone.
    """
    parsed = _observation(output)
    if parsed is None:
        return output, 0
    map_id, entries, _, _, is_delta = parsed
    state = seen.get(map_id)
    if is_delta or state is None:
        _apply_observation(seen, output)
        return output, 0
    current = {entry['id']: entry for entry in entries}
    added = [entry for entry in entries if entry['id'] not in state]
    updated = [entry for entry in entries
               if entry['id'] in state and state[entry['id']] != _canonical(entry)]
    removed = [object_id for object_id in state if object_id not in current]
    projected = {k: copy.deepcopy(v) for k, v in output.items() if k != 'objects'}
    projected['new_objects'] = copy.deepcopy(added)
    projected['observed_object_count'] = len(entries)
    if updated:
        projected['updated_objects'] = copy.deepcopy(updated)
    if removed:
        projected['removed_object_ids'] = removed
    projected['_monitor_projection'] = OBJECT_PROJECTION
    before = len(json.dumps(output, ensure_ascii=False))
    after = len(json.dumps(projected, ensure_ascii=False))
    _apply_observation(seen, output)
    return (output, 0) if after >= before else (projected, before - after)


def project_scene_objects(prior_plans, plans, continued):
    """Project every list_objects result after the last full one per map."""
    seen = {}
    for plan in prior_plans:
        if not isinstance(plan, dict) or not isinstance(plan.get('calls'), list):
            continue
        for call in plan['calls']:
            if not isinstance(call, dict) or call.get('contract_id') != OBJECTS_CONTRACT:
                continue
            for evidence in call.get('results', []) if isinstance(call.get('results'), list) else []:
                if isinstance(evidence, dict):
                    _apply_observation(seen, evidence.get('output'))
    events = []
    for plan in plans:
        if not isinstance(plan, dict) or not isinstance(plan.get('calls'), list):
            continue
        for call in plan['calls']:
            if isinstance(call, dict) and isinstance(call.get('results'), list):
                events.extend(e for e in call['results'] if isinstance(e, dict))
    events.extend(item['result'] for item in continued
                  if isinstance(item, dict) and isinstance(item.get('result'), dict))
    events.sort(key=lambda e: e.get('_monitor_history_index', -1))
    latest = {}
    for evidence in events:
        if evidence.get('contract_id') != OBJECTS_CONTRACT:
            continue
        parsed = _observation(evidence.get('output'))
        if parsed is not None and not parsed[4]:
            latest[parsed[0]] = evidence.get('_monitor_history_index', -1)
    projected, saved = 0, 0
    for evidence in events:
        if evidence.get('contract_id') != OBJECTS_CONTRACT:
            continue
        output = evidence.get('output')
        parsed = _observation(output)
        if parsed is None:
            continue
        if not parsed[4] and latest.get(parsed[0]) == evidence.get('_monitor_history_index', -1):
            _apply_observation(seen, output)
            continue
        evidence['output'], delta = project_observation(seen, output)
        if evidence['output'] is not output:
            projected += 1
            saved += delta
    return projected, saved


def compact_result_labels(plans):
    """Drop fields that only repeat the enclosing call or a bookkeeping index."""
    omitted = 0
    arguments = 0
    for plan in plans:
        if not isinstance(plan, dict) or not isinstance(plan.get('calls'), list):
            continue
        for call in plan['calls']:
            if not isinstance(call, dict) or not isinstance(call.get('results'), list):
                continue
            args = call.get('args')
            for evidence in call['results']:
                if not isinstance(evidence, dict):
                    continue
                for field in ('call_id', 'contract_id'):
                    if field in evidence and evidence[field] == call.get(field):
                        del evidence[field]
                        omitted += 1
                if '_monitor_history_index' in evidence:
                    del evidence['_monitor_history_index']
                    omitted += 1
                if 'error' in evidence and not evidence['error']:
                    del evidence['error']
                    omitted += 1
                output = evidence.get('output')
                if isinstance(args, dict) and isinstance(output, dict):
                    # A result value byte-identical to the call's own argument
                    # adds no evidence; the call record still states it.
                    for field in [f for f in output if f in args and output[f] == args[f]]:
                        del output[field]
                        omitted += 1
                        arguments += 1
    return omitted, arguments


def compact_catalog_boilerplate(content):
    """Replace per-entry doc pointers with one pointer the docs section already supports."""
    if '## Available capabilities' not in content:
        return content, 0
    providers = set(DOC_LINE.findall(content))
    if not providers:
        return content, 0
    kept, removed = [], 0
    for line in content.splitlines(keepends=True):
        match = MORE_LINE.match(line.rstrip('\r\n'))
        if match is not None and match.group(1) in providers:
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return content, 0
    text = ''.join(kept)
    intro = 'This is the complete catalog; it replaces any earlier catalog in this conversation.'
    if CATALOG_NOTE not in text:
        if intro in text:
            text = text.replace(intro, intro + '\n' + CATALOG_NOTE, 1)
        else:
            text = text.replace('## Available capabilities',
                                '## Available capabilities\n' + CATALOG_NOTE, 1)
    return text, removed


def _collapse_repeated_text(text, min_half=40):
    """Undo a description that is its own first half repeated verbatim."""
    n = len(text)
    for sep in (' ', '\n'):
        for i in range(max(min_half, n // 2 - 2), n // 2 + 3):
            if 0 < i < n - 1 and text[i] == sep and text[:i] == text[i + 1:]:
                return text[:i], n - i
    if n >= 2 * min_half and n % 2 == 0 and text[:n // 2] == text[n // 2:]:
        return text[:n // 2], n // 2
    return text, 0


def compact_catalog_provider_descriptions(content):
    """State a provider-wide description once instead of on every capability."""
    if '## Available capabilities' not in content:
        return content, 0, 0
    entries = []
    for match in CATALOG_ENTRY.finditer(content):
        name = match.group(1)
        text, saved = _collapse_repeated_text(json.loads(match.group(2)))
        entries.append({
            'match': match, 'name': name, 'provider': name.split('.', 1)[0],
            'description': text, 'args_schema': match.group(3), 'saved': saved,
        })
    if not entries:
        return content, 0, 0
    by_provider = {}
    for entry in entries:
        by_provider.setdefault(entry['provider'], []).append(entry)
    shared = {provider: group[0]['description'] for provider, group in by_provider.items()
              if len(group) > 1
              and len({entry['description'] for entry in group}) == 1}
    pieces, last, hoisted, saved_chars = [], 0, 0, 0
    emitted = set()
    for entry in entries:
        match = entry['match']
        pieces.append(content[last:match.start()])
        last = match.end()
        saved_chars += entry['saved']
        if entry['provider'] in shared:
            if entry['provider'] not in emitted:
                emitted.add(entry['provider'])
                pieces.append('- provider_description: {}\n  description: {}\n'.format(
                    entry['provider'],
                    json.dumps(shared[entry['provider']], ensure_ascii=False)))
            pieces.append('- capability_name: {}\n  args_schema: {}\n'.format(
                entry['name'], entry['args_schema']))
            hoisted += 1
        else:
            pieces.append('- capability_name: {}\n  description: {}\n  args_schema: {}\n'.format(
                entry['name'], json.dumps(entry['description'], ensure_ascii=False),
                entry['args_schema']))
    pieces.append(content[last:])
    text = ''.join(pieces)
    if text == content:
        return content, 0, 0
    if PROVIDER_DESC_NOTE not in text:
        intro = 'This is the complete catalog; it replaces any earlier catalog in this conversation.'
        if CATALOG_NOTE in text:
            text = text.replace(CATALOG_NOTE, CATALOG_NOTE + '\n' + PROVIDER_DESC_NOTE, 1)
        elif intro in text:
            text = text.replace(intro, intro + '\n' + PROVIDER_DESC_NOTE, 1)
        else:
            text = text.replace('## Available capabilities',
                                '## Available capabilities\n' + PROVIDER_DESC_NOTE, 1)
    return text, hoisted, saved_chars


BOARD_TASK = re.compile(r'(?m)^- (\S+): (\S+)\n((?:  [^\n]*(?:\n|$))*)')
BOARD_FIELD = re.compile(r'(?m)^  (goal|success criterion|messages): ?(.*)$')
BOARD_PLAN = re.compile(r'(?m)^  plan (\S+) \(([^)]+)\): ?(.*)$')
TASK_OMIT_NOTE = ('Fields stated verbatim on the Tasks (current status board) are omitted here: '
                  '{}.')


def parse_task_board(text):
    """Map each board task to the field values it states verbatim."""
    if not text:
        return {}
    tasks = {}
    for match in BOARD_TASK.finditer(text):
        task = {'status': match.group(2), 'plans': {}, 'plan_descriptions': {}}
        body = match.group(3)
        for field in BOARD_FIELD.finditer(body):
            task[field.group(1)] = field.group(2)
        for plan in BOARD_PLAN.finditer(body):
            task['plans'][plan.group(1)] = plan.group(2)
            task['plan_descriptions'][plan.group(1)] = plan.group(3)
        tasks[match.group(1)] = task
    return tasks


def compact_task_record(content, board_tasks):
    """Drop record fields that the task board already states with the same value."""
    rest = content[len(TASK):] if content.startswith(TASK) else content
    brace = rest.find('{')
    if brace < 0:
        return content, 0
    try:
        record = json.loads(rest[brace:])
    except ValueError:
        return content, 0
    if not isinstance(record, dict):
        return content, 0
    task = board_tasks.get(str(record.get('task_id')))
    if task is None:
        return content, 0
    omitted = []
    plans = record.get('plans')
    if (isinstance(plans, list) and plans and task['plans']
            and all(isinstance(p, dict) and isinstance(p.get('plan_id'), str)
                    and task['plans'].get(p['plan_id']) == p.get('state') for p in plans)
            and len(task['plans']) == len(plans)):
        record.pop('plans')
        omitted.append('plans')
    for field, key in (('goal', 'goal'), ('success criterion', 'success_criterion'),
                       ('messages', 'messages')):
        value = record.get(key)
        if key == 'messages' and isinstance(value, list):
            value = ', '.join(str(v) for v in value)
        if isinstance(value, str) and task.get(field) == value:
            record.pop(key)
            omitted.append(key)
    if not omitted:
        return content, 0
    return (TASK + TASK_OMIT_NOTE.format(', '.join(omitted)) + ' '
            + json.dumps(record, ensure_ascii=False, separators=(',', ':'))), len(omitted)


def compact_memory_plan_descriptions(plans, board_tasks):
    """Drop plan descriptions the task board repeats word for word."""
    descriptions = {}
    for task in board_tasks.values():
        descriptions.update(task['plan_descriptions'])
    omitted = 0
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        plan_id = plan.get('plan_id')
        if (isinstance(plan_id, str) and 'description' in plan
                and descriptions.get(plan_id) == plan['description']):
            plan.pop('description')
            omitted += 1
    return omitted


def compact_request(body, min_chars=48000, empty_slot_items=None, inventory_schema=None):
    """Fold the history after the last memory block into a new memory block.

    Only the suffix after the previous memory is rewritten, so the cached
    prefix (including earlier memory) stays byte-identical across rounds. A
    result for a plan recorded in an earlier block is kept in `continues`
    rather than reopening that block.
    """
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body, {'applied': False, 'reason': 'no_messages'}
    before = chars(messages)
    if before < min_chars:
        return body, {'applied': False, 'reason': 'below_threshold', 'before_chars': before}
    prior, start, memory_count = None, 0, 0
    for i, message in enumerate(messages):
        content = text_content(message.get('content'))
        if isinstance(content, str) and content.startswith(MEMORY):
            prior = parse_memory(content)
            if prior is None:
                return body, {'applied': False, 'reason': 'unreadable_memory',
                              'before_chars': before}
            memory_count += 1
            start = i + 1

    tasks, dispatches, results, sections = {}, [], [], {}
    correction, warning, last_user, last_live = None, None, -1, -1
    board_text = ''
    latest_sections = {}
    snapshot_count = segment_snapshots = 0
    for i, message in enumerate(messages):
        content = text_content(message.get('content'))
        if message.get('role') != 'user' or content is None:
            continue
        task = unpack(content, TASK)
        dispatch = unpack(content, DISPATCH)
        feedback = unpack(content, RESULT)
        if task is not None and isinstance(task.get('task_id'), (str, int)):
            tasks[task['task_id']] = i
        elif dispatch is not None and isinstance(dispatch.get('plan_id'), str) and isinstance(dispatch.get('calls'), list):
            if i >= start:
                dispatches.append((i, dispatch))
        elif feedback is not None and isinstance(feedback.get('leaf_result'), dict):
            leaf = feedback['leaf_result']
            if i >= start and isinstance(leaf.get('call_id'), str):
                results.append((i, leaf))
        elif content.startswith(('Your previous RTDL response was rejected by Pilot.',
                                 'The previous plan output was rejected by Pilot\'s JSON or RTDL validator.')):
            correction = i
        elif content.startswith('Pilot harness feedback:'):
            warning = i
        elif content.startswith('User message '):
            last_user = i
        elif content.lstrip().startswith(GENERATED_HEADINGS):
            matches = list(SECTION.finditer(content))
            if not matches:
                continue
            # Keep any unrecognized leading text rather than silently dropping it.
            if content[:matches[0].start()].strip():
                continue
            blocks = []
            for j, match in enumerate(matches):
                end = matches[j + 1].start() if j + 1 < len(matches) else len(content)
                blocks.append((match.group(1), match.start(), end))
            sections[i] = blocks
            if any(name in LIVE for name, _, _ in blocks):
                last_live = i
                snapshot_count += 1
                if i >= start:
                    segment_snapshots += 1
            for name, begin, end in blocks:
                if name in STATIC:
                    latest_sections[name] = i
                    if name == 'Tasks (current status board)':
                        board_text = content[begin:end]

    # Match results to exact dispatch IDs and contracts, including parallel plans.
    # Do not infer arguments from prose or from the current state. Calls from an
    # earlier memory block are matched by exact call_id/contract as well, so a
    # long-running plan's later result is still attributed and preserved.
    board_tasks = parse_task_board(board_text)
    prior_calls = {}
    for plan in prior['plans'] if prior is not None else []:
        if not isinstance(plan, dict) or not isinstance(plan.get('calls'), list):
            continue
        for call in plan['calls']:
            if isinstance(call, dict) and isinstance(call.get('call_id'), str):
                prior_calls[call['call_id']] = (plan.get('plan_id'), call.get('contract_id'))
    plans, calls, continued = [], {}, []
    dispatch_indices = set()
    call_counts = Counter(c['call_id'] for _, d in dispatches for c in d['calls']
                          if isinstance(c, dict) and isinstance(c.get('call_id'), str))
    for i, dispatch in dispatches:
        entries = dispatch['calls']
        if (not all(isinstance(c, dict) and isinstance(c.get('call_id'), str) for c in entries)
                or len({c['call_id'] for c in entries}) != len(entries)
                or any(call_counts[c['call_id']] != 1 for c in entries)
                or any(c['call_id'] in calls or c['call_id'] in prior_calls for c in entries)):
            continue
        plan = copy.deepcopy(dispatch)
        for call in plan['calls']:
            calls[call['call_id']] = call
        plans.append(plan)
        dispatch_indices.add(i)
    result_indices = set()
    for i, leaf in results:
        call = calls.get(leaf['call_id'])
        evidence = copy.deepcopy(leaf)
        evidence['_monitor_history_index'] = i
        if 'output' in evidence:
            evidence['output'] = compact_output(
                evidence['output'], empty_slot_items=empty_slot_items,
                inventory_schema=inventory_schema)
        if call is not None and call.get('contract_id') == leaf.get('contract_id'):
            # Keep repeated feedback as ordered events, not just the last state.
            call.setdefault('results', []).append(evidence)
            result_indices.add(i)
            continue
        prior_call = prior_calls.get(leaf['call_id'])
        if prior_call is not None and prior_call[1] == leaf.get('contract_id'):
            continued.append({'plan_id': prior_call[0], 'result': evidence})
            result_indices.add(i)
    if prior is None:
        if len(results) < 4 or snapshot_count < 2 or not dispatches:
            return body, {'applied': False, 'reason': 'insufficient_structured_history',
                          'before_chars': before}
    elif len(result_indices) < 2 or segment_snapshots < 1 or not (plans or continued):
        return body, {'applied': False, 'reason': 'insufficient_new_evidence',
                      'before_chars': before}
    if not result_indices:
        return body, {'applied': False, 'reason': 'unmatched_feedback', 'before_chars': before}

    projected_observations, projection_saved = project_scene_objects(
        prior['plans'] if prior is not None and isinstance(prior.get('plans'), list) else [],
        plans, continued)
    result_fields_omitted, result_argument_fields_omitted = compact_result_labels(plans)
    plan_descriptions_omitted = compact_memory_plan_descriptions(plans, board_tasks)

    last_dispatch = max(dispatch_indices) if dispatch_indices else -1
    last_result = max(result_indices)
    note = NOTE if prior is None else EXTENSION
    if projected_observations:
        note += PROJECTION_NOTE
    if result_fields_omitted:
        note += RESULT_NOTE
    if plan_descriptions_omitted:
        note += PLAN_DESC_NOTE
    document = {'note': note, 'plans': plans}
    if continued:
        document['continues'] = continued
    memory_message = {'role': 'user', 'content': MEMORY + json.dumps(document, ensure_ascii=False,
                                                                     separators=(',', ':'))}
    compacted = []
    removed = {}
    catalog_lines_removed = 0
    catalog_descriptions_hoisted = 0
    catalog_description_chars_saved = 0
    task_record_fields_omitted = 0
    def drop(reason):
        removed[reason] = removed.get(reason, 0) + 1
    for i, message in enumerate(messages):
        if i < start:
            compacted.append(copy.deepcopy(message))
            continue
        if i == last_result:
            compacted.append(memory_message)
        content = text_content(message.get('content'))
        role = message.get('role')
        if role == 'user' and isinstance(content, str):
            if i in sections:
                kept = [(begin, end) for name, begin, end in sections[i]
                        if name in LIVE and i == last_live
                        or name in STATIC and latest_sections.get(name) == i]
                if kept:
                    payload = slice_content(message['content'], kept)
                    if latest_sections.get('Available capabilities') == i:
                        payload, catalog_removed = compact_catalog_boilerplate(payload)
                        catalog_lines_removed += catalog_removed
                        payload, desc_hoisted, desc_saved = (
                            compact_catalog_provider_descriptions(payload))
                        catalog_descriptions_hoisted += desc_hoisted
                        catalog_description_chars_saved += desc_saved
                    compacted.append({**copy.deepcopy(message), 'content': payload})
                else:
                    drop('superseded_snapshots')
                continue
            task = unpack(content, TASK)
            if task is not None and isinstance(task.get('task_id'), (str, int)):
                if i in tasks.values():
                    payload, omitted = compact_task_record(content, board_tasks)
                    task_record_fields_omitted += omitted
                    compacted.append({**copy.deepcopy(message), 'content': payload})
                else:
                    drop('superseded_task_records')
                continue
            if i in dispatch_indices or i in result_indices:
                drop('execution_wrappers_consolidated')
                continue
            if content.startswith('Executor feedback scope:'):
                # Unknown/unmatched feedback still needs its original scope.
                if i + 1 in result_indices:
                    drop('repeated_feedback_scope')
                    continue
            if content.startswith(('Your previous RTDL response was rejected by Pilot.',
                                   'The previous plan output was rejected by Pilot\'s JSON or RTDL validator.')):
                if i != correction or i < max(last_dispatch, last_result, last_user):
                    drop('obsolete_corrections')
                    continue
            if content.startswith('Pilot harness feedback:'):
                if i != warning or i < max(last_dispatch, last_result, last_user):
                    drop('obsolete_duplicate_call_notices')
                    continue
        # Assistant narration is not authoritative execution evidence. Preserve
        # assistant tool calls and every tool response together and unchanged.
        if role == 'assistant' and not message.get('tool_calls') and isinstance(content, str):
            drop('assistant_narration')
            continue
        compacted.append(copy.deepcopy(message))
    after = chars(compacted)
    if after >= before:
        return body, {'applied': False, 'reason': 'no_size_reduction', 'before_chars': before}
    result = copy.deepcopy(body)
    result['messages'] = compacted
    return result, {'applied': True, 'method': 'structured_execution_history_v4',
                    'before_messages': len(messages), 'after_messages': len(compacted),
                    'before_chars': before, 'after_chars': after,
                    'reduction_percent': round(100 * (before - after) / before, 1),
                    'memory_blocks': memory_count + 1,
                    'prior_plans': len(prior['plans']) if prior is not None else 0,
                    'historical_plans': len(plans), 'historical_results': len(result_indices),
                    'continued_results': len(continued),
                    'projected_observations': projected_observations,
                    'projection_saved_chars': projection_saved,
                    'result_fields_omitted': result_fields_omitted,
                    'result_argument_fields_omitted': result_argument_fields_omitted,
                    'catalog_lines_removed': catalog_lines_removed,
                    'catalog_provider_descriptions_hoisted': catalog_descriptions_hoisted,
                    'catalog_description_chars_saved': catalog_description_chars_saved,
                    'task_record_fields_omitted': task_record_fields_omitted,
                    'plan_descriptions_omitted': plan_descriptions_omitted,
                    'removed_messages': removed,
                    'preserved': ['system_and_schema', 'user_requests', 'latest_catalog_and_docs',
                                  'all_matched_execution_results_and_arguments', 'latest_task_records',
                                  'latest_live_snapshot', 'latest_task_board', 'current_correction',
                                  'earlier_memory_blocks', 'scene_object_observations',
                                  'unknown_messages_and_tool_call_pairs']}
