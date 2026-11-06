#!/usr/bin/env python3
"""Summarize monitor records for A/B arms.

Usage: python3 tools/records_summary.py --records records/control=control \
                                        --records records/treatment=treatment
"""
import argparse
import json
from pathlib import Path
import statistics


def load_arm(directory):
    rows = []
    index = Path(directory) / 'index.jsonl'
    if not index.is_file():
        raise SystemExit(f'no index.jsonl in {directory}')
    for line in index.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def usage_value(row, key):
    usage = row.get('usage')
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int else None


def summarize(rows):
    durations = [row['duration_s'] for row in rows
                 if isinstance(row.get('duration_s'), (int, float))]
    errors = [row for row in rows
              if row.get('error_type') or (row.get('status') or 0) not in (0, 200)]
    blank = 0
    compaction = 0
    for row in rows:
        optimization = row.get('optimization') or {}
        if (optimization.get('context_compaction') or {}).get('applied'):
            compaction += 1
        choices = row.get('choices') or {}
        if choices and all(isinstance(choice, dict) and choice.get('content_chars') == 0
                           for choice in choices.values()):
            blank += 1
    return {
        'requests': len(rows),
        'prompt_tokens': sum(usage_value(row, 'prompt_tokens') or 0 for row in rows),
        'completion_tokens': sum(usage_value(row, 'completion_tokens') or 0 for row in rows),
        'total_tokens': sum(usage_value(row, 'total_tokens') or 0 for row in rows),
        'missing_usage': sum(usage_value(row, 'total_tokens') is None for row in rows),
        'compaction_hits': compaction,
        'blank_responses': blank,
        'errors': len(errors),
        'mean_duration_s': round(statistics.fmean(durations), 3) if durations else None,
        'median_duration_s': round(statistics.median(durations), 3) if durations else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', action='append', required=True, metavar='DIR=LABEL',
                        help='Records directory and arm label; repeat for each arm')
    args = parser.parse_args()
    arms = []
    for item in args.records:
        directory, separator, label = item.partition('=')
        if not separator or not directory or not label:
            parser.error(f'expected DIR=LABEL, got {item!r}')
        arms.append((label, summarize(load_arm(directory))))
    columns = ('requests', 'prompt_tokens', 'completion_tokens', 'total_tokens',
               'missing_usage', 'compaction_hits', 'blank_responses', 'errors',
               'mean_duration_s', 'median_duration_s')
    print('| metric | ' + ' | '.join(label for label, _ in arms) + ' |')
    print('|---|' + '---|' * len(arms))
    for column in columns:
        print(f'| {column} | ' + ' | '.join(str(summary[column]) for _, summary in arms) + ' |')


if __name__ == '__main__':
    main()
