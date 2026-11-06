"""Note scope limits and evidence gaps in already recorded execution results.

The passes in this module are domain-neutral: which contracts they recognize,
which inventory fields they read, and the note wording all come from a
DomainContext. Without one, every pass is a no-op, so the proxy never injects
wording for an embodiment it was not configured for. A pass only annotates the
latest relevant result; it never adds coordinates, contents, or task answers.
"""
import json


RESULT = 'Executor feedback for the current RTDL leaf (not a new user request): '
MEMORY = 'Monitor execution memory (historical evidence, not new instructions):\n'
DISPATCH = 'Pilot harness dispatch record (already sent to Executor; not a new user request): '


class DomainContext:
    """Contract, inventory, and note configuration for one domain profile."""

    def __init__(self, contracts=None, empty_slot_items=None, note_texts=None,
                 inventory_schema=None):
        contracts = contracts or {}
        self.observation = tuple(contracts.get('observation') or ())
        self.inspect = contracts.get('inspect')
        self.explore = contracts.get('explore')
        self.withdraw = contracts.get('withdraw')
        self.deposit = contracts.get('deposit')
        self.inventory = contracts.get('inventory')
        self.empty_slot_items = tuple(empty_slot_items or ())
        self.note_texts = dict(note_texts or {})
        schema = inventory_schema or {}
        self.transaction_count_field = schema.get('transaction_count_field')
        self.slots_field = schema.get('slots_field', 'slots')
        self.slot_field = schema.get('slot_field', 'slot')
        self.item_field = schema.get('item_field', 'item')
        self.count_field = schema.get('count_field', 'count')

    def _matches(self, contract, expected):
        return isinstance(contract, str) and bool(expected) and contract.endswith(expected)

    def is_observation(self, contract):
        return self._matches(contract, self.observation)

    def is_inspect(self, contract):
        return self._matches(contract, self.inspect)

    def is_explore(self, contract):
        return self._matches(contract, self.explore)

    def is_withdraw(self, contract):
        return self._matches(contract, self.withdraw)

    def is_deposit(self, contract):
        return self._matches(contract, self.deposit)

    def is_inventory(self, contract):
        return self._matches(contract, self.inventory)

    def is_empty_item(self, item):
        return item in self.empty_slot_items

    def note(self, key, **values):
        template = self.note_texts.get(key)
        if not template:
            return None
        try:
            return template.format(**values)
        except (KeyError, IndexError, ValueError):
            return None


DEFAULT_DOMAIN = DomainContext()


def _is_observation(contract, domain=DEFAULT_DOMAIN):
    return domain.is_observation(contract)


def _is_inspect(contract, domain=DEFAULT_DOMAIN):
    return domain.is_inspect(contract)


def _is_explore(contract, domain=DEFAULT_DOMAIN):
    return domain.is_explore(contract)


def _json_after(content, prefix):
    try:
        value = json.loads(content[len(prefix):])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _is_empty(output):
    value = output
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return False
    return (isinstance(value, dict) and isinstance(value.get('objects'), list)
            and not value['objects'])


def _object_ids(output):
    value = output
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, dict) or not isinstance(value.get('objects'), list):
        return []
    return [item['id'] for item in value['objects']
            if isinstance(item, dict) and isinstance(item.get('id'), str)]


def _memory_observations(document, domain=DEFAULT_DOMAIN):
    plans = document.get('plans')
    for plan in plans if isinstance(plans, list) else []:
        calls = plan.get('calls') if isinstance(plan, dict) else None
        for call in calls if isinstance(calls, list) else []:
            contract = call.get('contract_id') if isinstance(call, dict) else None
            results = call.get('results') if isinstance(call, dict) else None
            for evidence in results if isinstance(results, list) else []:
                if isinstance(evidence, dict) and _is_observation(
                        evidence.get('contract_id') or contract, domain):
                    yield evidence
    continues = document.get('continues')
    for entry in continues if isinstance(continues, list) else []:
        result = entry.get('result') if isinstance(entry, dict) else None
        if isinstance(result, dict) and _is_observation(result.get('contract_id'), domain):
            yield result


def _memory_results(document):
    for call in _memory_calls(document):
        results = call.get('results')
        for evidence in results if isinstance(results, list) else []:
            if isinstance(evidence, dict):
                yield evidence, call.get('contract_id')
    continues = document.get('continues')
    for entry in continues if isinstance(continues, list) else []:
        result = entry.get('result') if isinstance(entry, dict) else None
        if isinstance(result, dict):
            yield result, result.get('contract_id')


def _latest_result(messages):
    latest = None  # (message index, prefix, parsed root, evidence, contract)
    for index, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        if content.startswith(RESULT):
            root = _json_after(content, RESULT)
            leaf = root.get('leaf_result') if root else None
            if isinstance(leaf, dict):
                latest = (index, RESULT, root, leaf, leaf.get('contract_id'))
        elif content.startswith(MEMORY):
            root = _json_after(content, MEMORY)
            for evidence, contract in _memory_results(root or {}):
                latest = (index, MEMORY, root, evidence,
                          evidence.get('contract_id') or contract)
    return latest


def _memory_calls(document):
    plans = document.get('plans')
    for plan in plans if isinstance(plans, list) else []:
        calls = plan.get('calls') if isinstance(plan, dict) else None
        for call in calls if isinstance(calls, list) else []:
            if isinstance(call, dict):
                yield call


def _dispatch_calls(content):
    document = _json_after(content, DISPATCH)
    calls = document.get('calls') if document else None
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, dict):
            yield call


def _inspected_ids(messages, domain=DEFAULT_DOMAIN):
    ids = set()
    for message in messages:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        if content.startswith(MEMORY):
            calls = _memory_calls(_json_after(content, MEMORY) or {})
        elif content.startswith(DISPATCH):
            calls = _dispatch_calls(content)
        else:
            continue
        for call in calls:
            if _is_inspect(call.get('contract_id'), domain):
                args = call.get('args')
                object_id = args.get('object_id') if isinstance(args, dict) else None
                if isinstance(object_id, str):
                    ids.add(object_id)
    return ids


def _latest_observation(messages, domain=DEFAULT_DOMAIN):
    latest = None  # (message index, prefix, parsed root, evidence)
    for index, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        if content.startswith(RESULT):
            root = _json_after(content, RESULT)
            leaf = root.get('leaf_result') if root else None
            if isinstance(leaf, dict) and _is_observation(leaf.get('contract_id'), domain):
                latest = (index, RESULT, root, leaf)
        elif content.startswith(MEMORY):
            root = _json_after(content, MEMORY)
            for evidence in _memory_observations(root or {}, domain):
                latest = (index, MEMORY, root, evidence)
    return latest


def add_empty_observation_notes(body, domain=DEFAULT_DOMAIN):
    """Note the scope of the latest scene observation when it was empty."""
    empty = {'empty_observation_notes': 0, 'empty_observation_chars': 0}
    note = domain.note('empty_observation')
    if note is None or not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body, empty
    messages = body['messages']
    latest = _latest_observation(messages, domain)
    if latest is None:
        return body, empty
    index, prefix, root, evidence = latest
    if ('monitor_note' in evidence or evidence.get('success') is False
            or not _is_empty(evidence.get('output'))):
        return body, empty
    before = len(messages[index].get('content') or '')
    evidence['monitor_note'] = note
    content = prefix + json.dumps(root, ensure_ascii=False, separators=(',', ':'))
    messages[index] = {**messages[index], 'content': content}
    return body, {'empty_observation_notes': 1, 'empty_observation_chars': len(content) - before}


def add_uninspected_object_notes(body, domain=DEFAULT_DOMAIN):
    """Note that a location result does not show contents of its objects."""
    result = {'uninspected_object_notes': 0, 'uninspected_object_chars': 0}
    note = domain.note('uninspected_object')
    if note is None or not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body, result
    messages = body['messages']
    latest = _latest_observation(messages, domain)
    if latest is None:
        return body, result
    index, prefix, root, evidence = latest
    if 'monitor_note' in evidence or evidence.get('success') is False:
        return body, result
    ids = _object_ids(evidence.get('output'))
    if not ids:
        return body, result
    inspected = _inspected_ids(messages, domain)
    if all(object_id in inspected for object_id in ids):
        return body, result
    before = len(messages[index].get('content') or '')
    evidence['monitor_note'] = note
    content = prefix + json.dumps(root, ensure_ascii=False, separators=(',', ':'))
    messages[index] = {**messages[index], 'content': content}
    return body, {'uninspected_object_notes': 1, 'uninspected_object_chars': len(content) - before}


def _embedded_error(value, depth=0):
    """Return an error hidden inside a successful result, if any.

    A string that parses as JSON counts as one level deeper: the model sees an
    escaped payload, not a directly readable field. Truncated previews do not
    parse and are left alone rather than guessed at.
    """
    if depth > 5:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith(('{', '[')):
            return None
        try:
            value = json.loads(text)
        except ValueError:
            return None
        depth += 1
    if isinstance(value, dict):
        error = value.get('error')
        if depth > 0 and isinstance(error, str) and error.strip():
            return error.strip()
        for child in value.values():
            found = _embedded_error(child, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _embedded_error(child, depth + 1)
            if found:
                return found
    return None


def _result_entries(messages):
    for index, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        if content.startswith(RESULT):
            root = _json_after(content, RESULT)
            leaf = root.get('leaf_result') if root else None
            if isinstance(leaf, dict):
                yield index, RESULT, root, leaf, leaf.get('contract_id')
        elif content.startswith(MEMORY):
            root = _json_after(content, MEMORY)
            for evidence, contract in _memory_results(root or {}):
                yield (index, MEMORY, root, evidence,
                       evidence.get('contract_id') or contract)


def add_embedded_error_notes(body, domain=DEFAULT_DOMAIN):
    """Surface an error nested inside the latest result for its own contract.

    A later result for the same contract replaces the note, so a recovered
    action does not keep carrying an old error.
    """
    info = {'embedded_error_notes': 0, 'embedded_error_chars': 0}
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body, info
    messages = body['messages']
    entries = list(_result_entries(messages))
    last = {}
    for position, (*_, contract) in enumerate(entries):
        if isinstance(contract, str):
            last[contract] = position
    touched = {}
    for position, (index, prefix, root, evidence, contract) in enumerate(entries):
        if not isinstance(contract, str) or last.get(contract) != position:
            continue
        if 'monitor_note' in evidence or evidence.get('success') is False:
            continue
        error = _embedded_error(evidence.get('output'))
        if error is None:
            continue
        note = domain.note('embedded_error', error=error.rstrip('.')[:300])
        if note is None:
            continue
        evidence['monitor_note'] = note
        info['embedded_error_notes'] += 1
        touched[index] = (prefix, root)
    for index, (prefix, root) in sorted(touched.items()):
        before = len(messages[index].get('content') or '')
        content = prefix + json.dumps(root, ensure_ascii=False, separators=(',', ':'))
        messages[index] = {**messages[index], 'content': content}
        info['embedded_error_chars'] += len(content) - before
    return body, info


def add_explore_followup_notes(body, domain=DEFAULT_DOMAIN):
    """Note that an exploration result is not itself a search verdict."""
    result = {'explore_followup_notes': 0, 'explore_followup_chars': 0}
    note = domain.note('explore_followup')
    if note is None or not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body, result
    messages = body['messages']
    latest = _latest_result(messages)
    if latest is None:
        return body, result
    index, prefix, root, evidence, contract = latest
    if 'monitor_note' in evidence or not _is_explore(contract, domain):
        return body, result
    before = len(messages[index].get('content') or '')
    evidence['monitor_note'] = note
    content = prefix + json.dumps(root, ensure_ascii=False, separators=(',', ':'))
    messages[index] = {**messages[index], 'content': content}
    return body, {'explore_followup_notes': 1, 'explore_followup_chars': len(content) - before}


def _item_count(contract, evidence, domain=DEFAULT_DOMAIN):
    """Return the held-item count a result establishes, or None when unreadable."""
    if evidence.get('success') is False:
        return None
    output = evidence.get('output')
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    if not isinstance(output, dict):
        return None
    if domain.is_withdraw(contract) or domain.is_deposit(contract):
        field = domain.transaction_count_field
        if not field:
            return None
        count = output.get(field)
        return count if type(count) is int and count >= 0 else None
    if domain.is_inventory(contract):
        slots = output.get(domain.slots_field)
        if not isinstance(slots, list):
            return None
        total = 0
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            item = slot.get(domain.item_field)
            count = slot.get(domain.count_field)
            if domain.is_empty_item(item) or type(count) is not int:
                continue
            total += count
        return total
    return None


def _held_items(entries, domain=DEFAULT_DOMAIN):
    held = None
    for index, prefix, root, evidence, contract in entries:
        count = _item_count(contract, evidence, domain)
        if count is not None:
            held = count
    return held


def _call_identity(contract, args):
    if not isinstance(contract, str) or not isinstance(args, dict):
        return None
    return contract + '|' + json.dumps(args, sort_keys=True, ensure_ascii=False,
                                        separators=(',', ':'))


def _call_identities(messages):
    identities = {}
    for message in messages:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        if content.startswith(DISPATCH):
            calls = _dispatch_calls(content)
        elif content.startswith(MEMORY):
            calls = _memory_calls(_json_after(content, MEMORY) or {})
        else:
            continue
        for call in calls:
            call_id = call.get('call_id')
            identity = _call_identity(call.get('contract_id'), call.get('args'))
            if isinstance(call_id, str) and identity is not None:
                identities[call_id] = identity
    return identities


def _identity_entries(messages):
    identities = _call_identities(messages)
    for index, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        if content.startswith(RESULT):
            root = _json_after(content, RESULT)
            leaf = root.get('leaf_result') if root else None
            if not isinstance(leaf, dict):
                continue
            identity = identities.get(leaf.get('call_id'))
            if identity is None:
                identity = _call_identity(leaf.get('contract_id'), leaf.get('args'))
            yield index, RESULT, root, leaf, identity
        elif content.startswith(MEMORY):
            root = _json_after(content, MEMORY)
            for call in _memory_calls(root or {}):
                identity = _call_identity(call.get('contract_id'), call.get('args'))
                for evidence in call.get('results') or []:
                    if isinstance(evidence, dict):
                        yield index, MEMORY, root, evidence, identity


def _failure_text(evidence):
    error = evidence.get('error')
    if isinstance(error, str) and error.strip():
        return error.strip()
    output = evidence.get('output')
    if isinstance(output, dict):
        error = output.get('error')
        if isinstance(error, str) and error.strip():
            return error.strip()
    return ''


def _failed(evidence):
    return evidence.get('success') is False or bool(_failure_text(evidence))


def add_repeat_failure_notes(body, domain=DEFAULT_DOMAIN):
    """Note when the latest result repeats an identical call that keeps failing."""
    result = {'repeat_failure_notes': 0, 'repeat_failure_chars': 0}
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body, result
    messages = body['messages']
    entries = list(_identity_entries(messages))
    if not entries:
        return body, result
    index, prefix, root, evidence, identity = entries[-1]
    if identity is None or 'monitor_note' in evidence or not _failed(evidence):
        return body, result
    streak = 0
    for *_, prior, prior_identity in entries:
        if prior_identity == identity:
            streak = streak + 1 if _failed(prior) else 0
    if streak < 3:
        return body, result
    error = _failure_text(evidence).rstrip('.')[:200]
    note = (domain.note('repeat_failure_with_error', count=streak, error=error) if error
            else domain.note('repeat_failure', count=streak))
    if note is None:
        return body, result
    before = len(messages[index].get('content') or '')
    evidence['monitor_note'] = note
    content = prefix + json.dumps(root, ensure_ascii=False, separators=(',', ':'))
    messages[index] = {**messages[index], 'content': content}
    return body, {'repeat_failure_notes': 1, 'repeat_failure_chars': len(content) - before}


def add_holding_delivery_notes(body, domain=DEFAULT_DOMAIN):
    """Note that items carried in inventory are not yet delivered anywhere."""
    result = {'holding_delivery_notes': 0, 'holding_delivery_chars': 0}
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body, result
    messages = body['messages']
    entries = list(_result_entries(messages))
    if not entries:
        return body, result
    held = _held_items(entries, domain)
    if not held:
        return body, result
    index, prefix, root, evidence, contract = entries[-1]
    if 'monitor_note' in evidence:
        return body, result
    note = domain.note('holding_delivery', count=held)
    if note is None:
        return body, result
    before = len(messages[index].get('content') or '')
    evidence['monitor_note'] = note
    content = prefix + json.dumps(root, ensure_ascii=False, separators=(',', ':'))
    messages[index] = {**messages[index], 'content': content}
    return body, {'holding_delivery_notes': 1, 'holding_delivery_chars': len(content) - before}
