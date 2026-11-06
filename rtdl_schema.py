"""Optional RTDL schema refinement from the request's current capability catalog."""
import copy
import hashlib
import json
import re

import jsonschema

from message_content import text_content


ENVELOPE = {'content', 'message_updates', 'rtdl_parent_task_id', 'rtdl_description', 'rtdl', 'task_updates'}
HEADING = re.compile(r'^## Available capabilities[ \t]*$', re.M)
SECTION = re.compile(r'^## ', re.M)
ENTRY = re.compile(r'^- capability_name:[ \t]*([A-Za-z0-9_.:/-]+)[ \t]*$', re.M)
ARGS = re.compile(r'^[ \t]+args_schema:[ \t]*', re.M)


class SchemaRefinementError(ValueError):
    pass


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def current_catalog(messages):
    """Latest complete generated catalog replaces all earlier catalogs."""
    section = None
    for message in messages:
        text = text_content(message.get('content'))
        if message.get('role') != 'user' or not isinstance(text, str):
            continue
        if HEADING.match(text.lstrip()) is None:
            continue
        heading = HEADING.search(text)
        if heading is None:
            continue
        end = SECTION.search(text, heading.end())
        section = text[heading.end():end.start() if end else len(text)]
    if section is None:
        raise SchemaRefinementError('No current generated capability catalog')
    entries = list(ENTRY.finditer(section))
    if not entries or len(entries) != len(re.findall(r'^- capability_name:', section, re.M)):
        raise SchemaRefinementError('Missing or malformed capability names in current catalog')
    catalog = {}
    for i, match in enumerate(entries):
        name = match.group(1)
        block = section[match.end():entries[i + 1].start() if i + 1 < len(entries) else len(section)]
        fields = list(ARGS.finditer(block))
        if name in catalog or len(fields) != 1:
            raise SchemaRefinementError('Duplicate capability or missing/ambiguous argument schema')
        try:
            raw_schema = block[fields[0].end():].lstrip()
            schema, consumed = json.JSONDecoder().raw_decode(raw_schema)
            if raw_schema[consumed:].split('\n', 1)[0].strip():
                raise ValueError('Unexpected text after argument schema')
            if not isinstance(schema, dict) or schema.get('type') != 'object':
                raise ValueError('Capability arguments must be an object schema')
            jsonschema.Draft202012Validator.check_schema(schema)
        except (ValueError, jsonschema.SchemaError) as error:
            raise SchemaRefinementError('Invalid capability argument schema') from error
        catalog[name] = schema
    return catalog


def relocate(schema, base):
    """Rebase local argument-schema references when embedding under $defs."""
    def check_pointer(reference):
        target = schema
        try:
            for token in reference[2:].split('/'):
                if re.search(r'~(?![01])', token):
                    raise KeyError('Invalid JSON Pointer escape')
                token = token.replace('~1', '/').replace('~0', '~')
                target = target[int(token)] if isinstance(target, list) else target[token]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise SchemaRefinementError('Unresolved local argument-schema reference') from error

    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        result = {}
        for key, item in value.items():
            if key in ('$id', '$anchor', '$dynamicAnchor', '$dynamicRef'):
                raise SchemaRefinementError('Scoped schema identifiers are not supported in argument schemas')
            if key == '$ref':
                if item == '#':
                    result[key] = base
                elif isinstance(item, str) and item.startswith('#/'):
                    check_pointer(item)
                    result[key] = base + item[1:]
                else:
                    raise SchemaRefinementError('Argument schemas must use local JSON Pointer references')
            elif key in ('properties', '$defs', 'definitions', 'patternProperties'):
                # Property/definition names are data, including names such as $id.
                result[key] = {name: visit(child) for name, child in item.items()}
            elif key in ('const', 'enum', 'default', 'examples'):
                result[key] = copy.deepcopy(item)
            else:
                result[key] = visit(item)
        return result
    return visit(schema)


def refine_request(body):
    shape = body.get('response_format') if isinstance(body, dict) else None
    if not isinstance(shape, dict) or shape.get('type') != 'json_schema':
        return body, {'enabled': True, 'applied': False, 'reason': 'not_schema_request'}
    named = shape.get('json_schema')
    if not isinstance(named, dict) or named.get('name') != 'rtdl_envelope':
        return body, {'enabled': True, 'applied': False, 'reason': 'not_rtdl_envelope'}
    from monitor import is_pilot_request
    if not is_pilot_request(body):
        return body, {'enabled': True, 'applied': False, 'reason': 'not_pilot_request'}
    original = named.get('schema')
    try:
        jsonschema.Draft202012Validator.check_schema(original)
    except jsonschema.SchemaError as error:
        raise SchemaRefinementError('Invalid original RTDL envelope schema') from error
    if (not isinstance(original, dict) or original.get('type') != 'object'
            or set(original.get('properties', {})) != ENVELOPE
            or set(original.get('required', [])) != ENVELOPE
            or not isinstance(original.get('$defs', {}).get('node'), dict)
            or not isinstance(original.get('$defs', {}).get('meta'), dict)):
        raise SchemaRefinementError('Unrecognized RTDL envelope schema; original request not forwarded')
    catalog = current_catalog(body['messages'])
    schema = copy.deepcopy(original)
    definitions = schema['$defs']
    node = definitions['node']
    properties = node.get('properties', {})
    if not all(k in properties for k in ('op_id', 'description', 'children')):
        raise SchemaRefinementError('Unrecognized RTDL node schema')
    branches = []
    for op in ('sequence', 'parallel'):
        branches.append({'type': 'object', 'properties': {
            'op': {'type': 'string', 'enum': [op]},
            'op_id': copy.deepcopy(properties['op_id']),
            'description': copy.deepcopy(properties['description']),
            'children': {'type': 'array', 'items': {'$ref': '#/$defs/node'}}},
            'required': ['op', 'op_id', 'description', 'children'], 'additionalProperties': False})
    for i, (name, args) in enumerate(catalog.items()):
        key = f'monitor_cap_args_{i}'
        if key in definitions:
            raise SchemaRefinementError('Reserved monitor schema definition already exists')
        definitions[key] = relocate(args, f'#/$defs/{key}')
        branches.append({'type': 'object', 'properties': {
            'op': {'type': 'string', 'enum': ['do']},
            'op_id': copy.deepcopy(properties['op_id']),
            'description': copy.deepcopy(properties['description']),
            'cap': {'type': 'string', 'enum': [name]},
            'args': {'$ref': f'#/$defs/{key}'}},
            'required': ['op', 'op_id', 'description', 'cap', 'args'], 'additionalProperties': False})
    definitions['node'] = {'anyOf': branches}
    # Cancellation/control remains at the root, never a child physical action.
    controls = definitions['meta'].get('anyOf')
    if not isinstance(controls, list) or not controls:
        raise SchemaRefinementError('Unrecognized RTDL plan-control schema')
    for branch in controls:
        if not isinstance(branch, dict) or branch.get('type') != 'object':
            raise SchemaRefinementError('Unrecognized RTDL plan-control schema')
        branch['additionalProperties'] = False
    schema['additionalProperties'] = False
    jsonschema.Draft202012Validator.check_schema(schema)
    result = copy.deepcopy(body)
    result['response_format']['json_schema']['schema'] = schema
    return result, {'enabled': True, 'applied': True, 'method': 'capability_bound_rtdl_v1',
                    'capability_count': len(catalog), 'node_branches': len(branches),
                    'original_schema_sha256': digest(original), 'schema_sha256': digest(schema),
                    'catalog_sha256': digest(catalog),
                    'original_schema_chars': len(json.dumps(original, ensure_ascii=False)),
                    'schema_chars': len(json.dumps(schema, ensure_ascii=False))}
