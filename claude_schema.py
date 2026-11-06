"""Finite RTDL trees and Claude's supported JSON Schema subset.

The original schema remains the local validation contract. No model output is
rewritten and unsupported constraints are recorded, not silently discarded.
"""
import copy
import json

import jsonschema


class ClaudeSchemaError(ValueError):
    pass


def adapt_schema(original, plan_depth=2, rtdl=False):
    if not 1 <= plan_depth <= 4:
        raise ClaudeSchemaError('Claude plan depth must be between 1 and 4')
    schema = copy.deepcopy(original)
    jsonschema.Draft202012Validator.check_schema(schema)
    if rtdl:
        defs = schema.get('$defs', {})
        node = defs.get('node', {})
        branches = node.get('anyOf')
        if not isinstance(branches, list):
            raise ClaudeSchemaError('Claude RTDL Schema mode requires --refine-rtdl-schema')
        leaves = [b for b in branches if b.get('properties', {}).get('op', {}).get('enum') == ['do']]
        containers = [b for b in branches if b.get('properties', {}).get('op', {}).get('enum') in (['sequence'], ['parallel'])]
        if len(containers) != 2 or len(leaves) + 2 != len(branches):
            raise ClaudeSchemaError('Unrecognized capability-bound RTDL schema')
        for level in range(plan_depth + 1):
            key = f'monitor_claude_node_{level}'
            if key in defs:
                raise ClaudeSchemaError('Reserved Claude schema definition already exists')
            alternatives = copy.deepcopy(leaves)
            if not leaves and level == 0:
                # Without a selected action, only empty containers are valid
                # under the original schema; local validation rejects nulls.
                alternatives = [{'type': 'null'}]
            if level:
                for branch in containers:
                    branch = copy.deepcopy(branch)
                    branch['properties']['children']['items'] = {'$ref': f'#/$defs/monitor_claude_node_{level - 1}'}
                    alternatives.append(branch)
            defs[key] = {'anyOf': alternatives}
        defs['node'] = {'$ref': f'#/$defs/monitor_claude_node_{plan_depth}'}

    removed, closed = [], []
    numerical = {'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'}
    lengths = {'minLength', 'maxLength', 'maxItems', 'uniqueItems', 'minProperties', 'maxProperties'}
    unsupported = numerical | lengths
    def visit(value, path):
        if isinstance(value, list):
            return [visit(v, path + '/' + str(i)) for i, v in enumerate(value)]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        out, notes = {}, []
        for key, item in value.items():
            if key in unsupported or key == 'minItems' and item not in (0, 1):
                removed.append({'path': path, 'keyword': key, 'value': item})
                notes.append(key + '=' + json.dumps(item, ensure_ascii=False))
            elif key in ('$id', '$anchor', '$dynamicRef', '$dynamicAnchor'):
                raise ClaudeSchemaError('Scoped/dynamic schema identifiers are not supported')
            elif key in ('const', 'enum', 'default', 'examples'):
                out[key] = copy.deepcopy(item)
            elif key in ('properties', '$defs', 'definitions', 'patternProperties'):
                out[key] = {name: visit(child, path + '/' + key + '/' + name) for name, child in item.items()}
            else:
                out[key] = visit(item, path + '/' + key)
        kind = value.get('type')
        if kind == 'object' or isinstance(kind, list) and 'object' in kind:
            if not isinstance(out.get('properties'), dict):
                raise ClaudeSchemaError('Unconstrained object requires an explicit property schema for Claude')
            additional = value.get('additionalProperties', True)
            if additional is not False:
                if isinstance(additional, dict):
                    raise ClaudeSchemaError('Dynamic additionalProperties cannot be represented by Claude Schema')
                out['additionalProperties'] = False
                closed.append(path)
        if notes:
            description = out.get('description', '')
            out['description'] = (description + '\nConstraints validated locally: ' + '; '.join(notes)).strip()
        return out
    schema = visit(schema, '#')

    # Check reachable AND unused definitions: Claude rejects cycles in either.
    def resolve(ref):
        if not isinstance(ref, str) or not ref.startswith('#/'):
            raise ClaudeSchemaError('Only local JSON Pointer schema references are supported')
        target = schema
        try:
            for token in ref[2:].split('/'):
                token = token.replace('~1', '/').replace('~0', '~')
                target = target[int(token)] if isinstance(target, list) else target[token]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ClaudeSchemaError('Unresolved Claude schema reference') from exc
        return target
    visited, stack = set(), set()
    def acyclic(value):
        if isinstance(value, list):
            for v in value:
                acyclic(v)
            return
        if not isinstance(value, dict):
            return
        ref = value.get('$ref')
        if ref:
            if ref in stack:
                raise ClaudeSchemaError('Recursive argument schemas are not supported by Claude')
            if ref not in visited:
                stack.add(ref)
                acyclic(resolve(ref))
                stack.remove(ref)
                visited.add(ref)
        for key, item in value.items():
            if key in ('const', 'enum', 'default', 'examples', '$ref'):
                continue
            if key in ('properties', '$defs', 'definitions', 'patternProperties'):
                for child in item.values():
                    acyclic(child)
            else:
                acyclic(item)
    acyclic(schema)
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema, {'applied': True, 'rtdl': rtdl, 'max_container_depth': plan_depth if rtdl else None,
                    'closed_object_count': len(closed), 'locally_validated_constraints': removed,
                    'schema_chars': len(json.dumps(schema, ensure_ascii=False))}


def select_capabilities(original, names):
    """Restrict this round's schema only; selection is made by Claude, not us."""
    schema = copy.deepcopy(original)
    branches = schema.get('$defs', {}).get('node', {}).get('anyOf')
    if not isinstance(branches, list):
        raise ClaudeSchemaError('Claude RTDL selection requires --refine-rtdl-schema')
    known = {b['properties']['cap']['enum'][0] for b in branches if 'cap' in b.get('properties', {})}
    if len(names) != len(set(names)) or set(names) - known:
        raise ClaudeSchemaError('Claude selected unknown or duplicate capabilities')
    schema['$defs']['node']['anyOf'] = [b for b in branches if 'cap' not in b.get('properties', {}) or b['properties']['cap']['enum'][0] in names]
    # Prune unused argument definitions so the compiler sees this round only.
    used = set()
    def refs(value):
        if isinstance(value, list):
            for child in value:
                refs(child)
        elif isinstance(value, dict):
            ref = value.get('$ref', '')
            if ref.startswith('#/$defs/'):
                key = ref.split('/')[2].replace('~1', '/').replace('~0', '~')
                if key not in used:
                    used.add(key)
                    refs(schema['$defs'][key])
            for key, child in value.items():
                if key not in ('$defs', '$ref', 'const', 'enum', 'default', 'examples'):
                    refs(child)
    refs({k: v for k, v in schema.items() if k != '$defs'})
    schema['$defs'] = {k: v for k, v in schema['$defs'].items() if k in used}
    return schema
