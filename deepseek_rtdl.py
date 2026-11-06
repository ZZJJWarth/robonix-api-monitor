"""Request-only DeepSeek adaptation; Pilot still validates every returned plan."""
from copy import deepcopy


FORMAT_REMINDER = '''Final response format contract for this round:
Return exactly ONE JSON object with exactly these six top-level fields: content, message_updates, rtdl_parent_task_id, rtdl_description, rtdl, task_updates. Begin with { and end with }. All user-facing sentences belong inside the content string. Put task changes only in task_updates. Put the next executable plan only in rtdl, using the existing RTDL grammar and current capability catalog.
Do not output a standalone task-state object, a separate preamble, XML, tool-protocol markup, Markdown fences, or any other text outside that object. Earlier rejected-response previews are invalid examples; do not imitate their structure.
Keep the original task and its success criterion unchanged. Choose the next action from the actual feedback. An empty plan is allowed only when waiting or legitimately concluding; do not use it to avoid planning pending work. Before emitting the final answer, ensure the entire final content parses as one JSON object and all six required fields are present. Do not claim execution or success that has not been observed.'''

CORRECTION = '''The previous plan output was rejected by Pilot's JSON or RTDL validator.
Correct the same task using the current capability catalog and argument schemas.
Return one complete RTDL envelope. Do not repeat already successful actions.
Keep the task unfinished unless its success criterion has actually been verified.'''


def reminder_text(domain_reminder=None):
    """Protocol rules always; embodiment rules only when a profile supplies them."""
    if domain_reminder:
        return FORMAT_REMINDER + '\n\n' + domain_reminder
    return FORMAT_REMINDER


def optimize_request(body, domain_reminder=None):
    if not isinstance(body, dict):
        return body, {'applied': False}
    model = str(body.get('model', '')).lower().rsplit('/', 1)[-1]
    messages = body.get('messages')
    shape = body.get('response_format')
    if (not model.startswith('deepseek') or not isinstance(messages, list)
            or not all(isinstance(message, dict) for message in messages)
            or not isinstance(shape, dict)):
        return body, {'applied': False}
    schema = shape.get('json_schema') or {}
    named_schema = (shape.get('type') == 'json_schema'
                    and isinstance(schema, dict) and schema.get('name') == 'rtdl_envelope')
    rtdl_prompt = any(
        isinstance(message, dict) and message.get('role') == 'system'
        and isinstance(message.get('content'), str)
        and 'You are the Robonix Pilot' in message['content']
        and '## RTDL output' in message['content']
        for message in messages
    )
    if not (named_schema or shape.get('type') == 'json_object' and rtdl_prompt):
        return body, {'applied': False}

    result = deepcopy(body)
    result['response_format'] = {'type': 'json_object'}
    result['thinking'] = {'type': 'disabled'}
    if 'reasoning_effort' in result:
        result['reasoning_effort'] = 'none'
    cleaned = 0
    for message in result['messages']:
        content = message.get('content') if isinstance(message, dict) else None
        if (message.get('role') == 'user' and isinstance(content, str)
                and content.startswith('Your previous RTDL response was rejected by Pilot.')):
            message['content'] = CORRECTION
            cleaned += 1
    reminder = {'role': 'user', 'content': reminder_text(domain_reminder)}
    if not result['messages'] or result['messages'][-1] != reminder:
        result['messages'].append(reminder)
    return result, {'applied': True, 'thinking': 'disabled', 'response_format': 'json_object',
                    'error_previews_removed': cleaned}
