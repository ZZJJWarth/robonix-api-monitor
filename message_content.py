"""Read text-only Chat content without losing structured cache annotations."""
import copy


def text_content(content):
    if isinstance(content, str):
        return content
    if (isinstance(content, list) and content
            and all(isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str) for p in content)):
        return ''.join(p['text'] for p in content)
    return None


def slice_content(content, ranges):
    """Keep ordered text ranges and each surviving block's original metadata."""
    if isinstance(content, str):
        return '\n\n'.join(content[a:b] for a, b in ranges)
    if text_content(content) is None:
        raise ValueError('Only text-only content may be sliced')
    result, offset = [], 0
    for part in content:
        end = offset + len(part['text'])
        fragments = [part['text'][max(a, offset)-offset:min(b, end)-offset]
                     for a, b in ranges if max(a, offset) < min(b, end)]
        if fragments:
            result.append({**copy.deepcopy(part), 'text': '\n\n'.join(fragments)})
        offset = end
    return result
