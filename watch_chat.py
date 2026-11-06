#!/usr/bin/env python3
"""Read-only tmux view of the independently running Liaison chat stream."""
import json
import os
from pathlib import Path
import time

ROOT = Path(os.environ.get('RBNX_CHAT_ROOT', Path(__file__).parent / 'chat-logs'))


def format_event(event, state):
    out = []
    text = event.get('text_chunk')
    if text:
        state['last_text'] = text
        out.append('[机器人] ' + text)
    if event.get('final_text'):
        text = event['final_text']
        if text != state.get('last_text'):
            out.append('[机器人] ' + text)
        state['last_text'] = text
    plan = event.get('plan')
    if plan:
        for node in plan.get('nodes', []):
            if node.get('node_kind') == 2:
                out.append('[动作] ' + node.get('description', node.get('call', {}).get('contract_id', '')))
    nodes = [event['node_state']] if 'node_state' in event else event.get('batch_result', {}).get('results', [])
    for node in nodes:
        leaf = node.get('leaf_result')
        if not leaf or node.get('state') not in (2, 3):
            continue
        key = (leaf.get('call_id'), node.get('state'))
        if key in state.setdefault('seen', set()):
            continue
        state['seen'].add(key)
        label = '动作完成' if node['state'] == 2 else '动作失败'
        out.append('[' + label + '] ' + node.get('description', leaf.get('contract_id', '')) +
                   (': ' + leaf['error'] if leaf.get('error') else ''))
    task = event.get('task_state')
    if task and task.get('status') != state.get('task_status'):
        state['task_status'] = task.get('status')
        out.append('[任务] ' + {'in_progress': '进行中', 'done': '已完成', 'failed': '失败', 'superseded': '已替换'}.get(task.get('status'), str(task.get('status'))))
    status = event.get('status', {}).get('message', '')
    if status and status != state.get('pilot_status') and not status.startswith('turn_id='):
        state['pilot_status'] = status
        out.append('[状态] ' + {'Planning the next step': '正在规划下一步', 'VLM response delayed; retrying once': '模型响应失败或延迟，正在重试一次'}.get(status, status))
    return out


def main():
    print('机器人聊天输出（只读；自动跟随最新测试，不发送任务）\n', flush=True)
    run, offset, state, ending = None, 0, {}, None
    while True:
        try:
            latest = Path(json.loads((ROOT / 'latest.json').read_text())['run_dir'])
            meta = json.loads((latest / 'task.json').read_text())
            if latest != run:
                run, offset, state, ending = latest, 0, {}, None
                print('\n' + '=' * 64, flush=True)
                print('[模型] ' + meta.get('model', ''), flush=True)
                print('[用户] ' + meta.get('prompt', ''), flush=True)
                if meta.get('original_goal'):
                    print('[原任务] ' + meta['original_goal'], flush=True)
            path = run / 'events.jsonl'
            if path.exists():
                with path.open() as stream:
                    stream.seek(offset)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith('\n'):
                            offset = start
                            break
                        offset = stream.tell()
                        for text in format_event(json.loads(line), state):
                            print(text, flush=True)
            status = meta.get('status')
            if status in ('stream_completed', 'stream_error') and status != ending:
                ending = status
                if status == 'stream_error':
                    print('[对话结束] 发生错误：' + meta.get('error_type', '见原始日志'), flush=True)
                else:
                    print('[对话结束] ' + ('任务已完成' if state.get('task_status') == 'done' else '本轮停止，任务尚未完成'), flush=True)
        except (FileNotFoundError, ValueError, KeyError):
            pass
        time.sleep(.5)


if __name__ == '__main__':
    main()
