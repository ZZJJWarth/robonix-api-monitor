"""Replay one actual Pilot prompt against Claude; never dispatch any plan."""
import concurrent.futures
import copy
import json
from pathlib import Path

from probe_claude_schema import MODEL, call
from rtdl_schema import refine_request


def main():
    root = Path(__file__).parent
    directory = root / 'claude-schema-validation' / MODEL
    original = json.loads((root / 'records/20261026-185818-b32bbcf11d/request.json').read_text())['received_body']
    refined, metadata = refine_request(original)
    schema = refined['response_format']['json_schema']['schema']
    (directory / 'refined-rtdl-schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    chat = {'model': MODEL, 'max_tokens': 4096, 'messages': copy.deepcopy(original['messages']),
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'rtdl_envelope', 'strict': True, 'schema': schema}}}
    system, messages = [], []
    for message in original['messages']:
        if message['role'] in ('system', 'developer'):
            system.append(message['content'])
        else:
            messages.append(copy.deepcopy(message))
    native = {'model': MODEL, 'max_tokens': 4096, 'system': '\n\n'.join(system), 'messages': messages,
              'output_config': {'format': {'type': 'json_schema', 'schema': schema}}}
    jobs = [('chat', chat), ('messages', native)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, protocol, 'actual_rtdl_refined', body, directory, schema) for protocol, body in jobs]
        results = [f.result() for f in futures]
    for result in results:
        result['source_record'] = '20261026-185818-b32bbcf11d'
        result['capability_count'] = metadata['capability_count']
    (directory / 'rtdl.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
