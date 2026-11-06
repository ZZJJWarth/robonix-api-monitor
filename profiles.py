"""Domain profiles keep the proxy neutral and load embodiment semantics from JSON.

A profile decides which observation hints are available, which contract ids
those hints recognize, the hint wording, inventory field names, empty-slot item
ids, optional Claude guidance, and an optional DeepSeek domain reminder. None of
those values are hardcoded in the proxy.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import string


PROFILE_DIR = Path(__file__).resolve().parent / 'profiles'
BUILTIN_PROFILES = ('generic',)
HINT_FLAGS = (
    'empty_observation_hint',
    'uninspected_object_hint',
    'explore_followup_hint',
    'embedded_error_hint',
    'holding_delivery_hint',
    'repeat_failure_hint',
)
HINT_NOTE_KEYS = {
    'empty_observation_hint': ('empty_observation',),
    'uninspected_object_hint': ('uninspected_object',),
    'explore_followup_hint': ('explore_followup',),
    'embedded_error_hint': ('embedded_error',),
    'holding_delivery_hint': ('holding_delivery',),
    'repeat_failure_hint': ('repeat_failure', 'repeat_failure_with_error'),
}
NOTE_FIELDS = {
    'empty_observation': frozenset(),
    'uninspected_object': frozenset(),
    'explore_followup': frozenset(),
    'embedded_error': frozenset({'error'}),
    'holding_delivery': frozenset({'count'}),
    'repeat_failure': frozenset({'count'}),
    'repeat_failure_with_error': frozenset({'count', 'error'}),
}
SCALAR_CONTRACTS = ('inspect', 'explore', 'withdraw', 'deposit', 'inventory')
INVENTORY_FIELDS = ('transaction_count_field', 'slots_field', 'slot_field', 'item_field', 'count_field')
PROFILE_KEYS = {
    'name', 'description', 'hints', 'hint_notes', 'contracts', 'empty_slot_items',
    'inventory_schema', 'claude_guidance', 'claude_guidance_version',
    'rtdl_domain_reminder',
}


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class DomainSpec:
    name: str
    source: str
    hints: frozenset
    hint_notes: dict
    contracts: dict
    empty_slot_items: tuple
    inventory_schema: dict
    claude_guidance: object
    claude_guidance_version: object
    rtdl_domain_reminder: object

    def hint_enabled(self, name):
        return name in self.hints


def _optional_text(value, field):
    if value is not None and (not isinstance(value, str) or not value):
        raise ProfileError(f'{field} must be null or a non-empty string')
    return value


def _check_placeholders(text, allowed, key):
    for _, field_name, _, _ in string.Formatter().parse(text):
        if field_name is None:
            continue
        root = field_name.split('.', 1)[0].split('[', 1)[0]
        if root not in allowed:
            raise ProfileError(f'hint note {key!r} uses unknown placeholder {{{field_name}}}')


def _hint_notes(value, hints):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ProfileError('hint_notes must be an object')
    unknown = set(value) - set(NOTE_FIELDS)
    if unknown:
        raise ProfileError(f'unknown hint note keys: {", ".join(sorted(unknown))}')
    notes = {}
    for key, text in value.items():
        if not isinstance(text, str) or not text:
            raise ProfileError(f'hint_notes.{key} must be a non-empty string')
        _check_placeholders(text, NOTE_FIELDS[key], key)
        notes[key] = text
    for hint in hints:
        for key in HINT_NOTE_KEYS[hint]:
            if key not in notes:
                raise ProfileError(f'hint {hint!r} requires hint_notes.{key}')
    return notes


def _contracts(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ProfileError('contracts must be an object')
    unknown = set(value) - {'observation', *SCALAR_CONTRACTS}
    if unknown:
        raise ProfileError(f'unknown contract fields: {", ".join(sorted(unknown))}')
    result = {}
    observation = value.get('observation', [])
    if isinstance(observation, str):
        observation = [observation]
    if not isinstance(observation, list) or not all(isinstance(item, str) and item for item in observation):
        raise ProfileError('contracts.observation must be a list of contract ids')
    result['observation'] = tuple(observation)
    for name in SCALAR_CONTRACTS:
        contract = value.get(name)
        if contract is not None and (not isinstance(contract, str) or not contract):
            raise ProfileError(f'contracts.{name} must be a contract id string')
        result[name] = contract
    return result


def _inventory_schema(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ProfileError('inventory_schema must be an object')
    unknown = set(value) - set(INVENTORY_FIELDS)
    if unknown:
        raise ProfileError(f'unknown inventory_schema fields: {", ".join(sorted(unknown))}')
    schema = {}
    for key, field in value.items():
        if not isinstance(field, str) or not field:
            raise ProfileError(f'inventory_schema.{key} must be a non-empty field name')
        schema[key] = field
    return schema


def _spec(selection, source, data):
    if not isinstance(data, dict):
        raise ProfileError('profile must be a JSON object')
    unknown = set(data) - PROFILE_KEYS
    if unknown:
        raise ProfileError(f'unknown profile fields: {", ".join(sorted(unknown))}')
    given_name = data.get('name')
    if given_name is not None and (not isinstance(given_name, str) or not given_name):
        raise ProfileError('name must be a non-empty string')
    hints = data.get('hints', [])
    if not isinstance(hints, list) or not all(isinstance(item, str) for item in hints):
        raise ProfileError('hints must be a list of hint names')
    unknown_hints = set(hints) - set(HINT_FLAGS)
    if unknown_hints:
        raise ProfileError(f'unknown hints: {", ".join(sorted(unknown_hints))}')
    empty_slot_items = data.get('empty_slot_items', [])
    if (not isinstance(empty_slot_items, list)
            or not all(isinstance(item, str) and item for item in empty_slot_items)):
        raise ProfileError('empty_slot_items must be a list of item ids')
    guidance = _optional_text(data.get('claude_guidance'), 'claude_guidance')
    guidance_version = _optional_text(data.get('claude_guidance_version'),
                                      'claude_guidance_version')
    if guidance_version and not guidance:
        raise ProfileError('claude_guidance_version requires claude_guidance')
    return DomainSpec(
        name=given_name or selection,
        source=source,
        hints=frozenset(hints),
        hint_notes=_hint_notes(data.get('hint_notes'), hints),
        contracts=_contracts(data.get('contracts')),
        empty_slot_items=tuple(empty_slot_items),
        inventory_schema=_inventory_schema(data.get('inventory_schema')),
        claude_guidance=guidance,
        claude_guidance_version=guidance_version,
        rtdl_domain_reminder=_optional_text(data.get('rtdl_domain_reminder'),
                                            'rtdl_domain_reminder'),
    )


def load_profile(selection):
    """Load the built-in generic profile or a path to a profile JSON file."""
    if selection in BUILTIN_PROFILES:
        path = PROFILE_DIR / f'{selection}.json'
        source = path.name
    else:
        path = Path(selection).expanduser()
        source = str(path)
        if not path.is_file():
            raise ProfileError(f'unknown profile {selection!r}; use generic or a JSON path')
    try:
        data = json.loads(path.read_text())
    except OSError as error:
        raise ProfileError(f'cannot read profile {source}: {error}') from error
    except ValueError as error:
        raise ProfileError(f'profile {source} is not valid JSON: {error}') from error
    return _spec(selection, source, data)


def resolve_profile(args):
    selection = getattr(args, 'profile', 'auto')
    if selection == 'auto':
        selection = 'generic'
    return load_profile(selection)


def validate_args(parser, args):
    """Reject hint flags that the selected profile does not enable."""
    try:
        spec = resolve_profile(args)
    except ProfileError as error:
        parser.error(str(error))
    missing = [flag for flag in HINT_FLAGS if getattr(args, flag, False) and flag not in spec.hints]
    if missing:
        parser.error('profile ' + repr(spec.name) + ' does not enable these hints: '
                     + ', '.join('--' + flag.replace('_', '-') for flag in missing)
                     + '; pass a JSON profile that enables them')
    return spec
