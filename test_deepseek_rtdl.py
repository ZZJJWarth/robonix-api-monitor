import copy
import unittest

from deepseek_rtdl import CORRECTION, FORMAT_REMINDER, optimize_request


def request(model='deepseek-flash'):
    return dict(model=model, response_format={
        'type': 'json_schema', 'json_schema': {'name': 'rtdl_envelope', 'schema': {}}},
        messages=[{'role': 'system', 'content': 'You are the Robonix Pilot\n## RTDL output'},
                  {'role': 'user', 'content': 'An arbitrary user task.'}])


class DeepSeekOptimizationTest(unittest.TestCase):
    def test_real_plan_inputs_are_preserved_and_schema_request_becomes_json_object(self):
        original = request()
        before = copy.deepcopy(original)
        result, meta = optimize_request(original)
        self.assertEqual(original, before)
        self.assertTrue(meta['applied'])
        self.assertEqual(result['model'], original['model'])
        self.assertEqual(result['messages'][:-1], original['messages'])
        self.assertEqual(result['response_format'], {'type': 'json_object'})
        self.assertEqual(result['thinking'], {'type': 'disabled'})
        self.assertEqual(result['messages'][-1]['content'], FORMAT_REMINDER)

    def test_domain_reminder_is_appended_only_when_supplied(self):
        plain, _ = optimize_request(request())
        self.assertEqual(plain['messages'][-1]['content'], FORMAT_REMINDER)
        self.assertNotIn('Demo domain rule.', plain['messages'][-1]['content'])
        result, _ = optimize_request(request(), domain_reminder='Demo domain rule.')
        self.assertEqual(result['messages'][-1]['content'],
                         FORMAT_REMINDER + '\n\nDemo domain rule.')

    def test_only_generated_correction_messages_are_cleaned(self):
        original = request()
        original['messages'] += [
            {'role': 'user', 'content': 'Your previous RTDL response was rejected by Pilot.\nError: invalid example </context_update>'},
            {'role': 'user', 'content': 'A user mentions </context_update> in their own request.'},
        ]
        result, meta = optimize_request(original)
        self.assertEqual(meta['error_previews_removed'], 1)
        self.assertEqual(result['messages'][2]['content'], CORRECTION)
        self.assertEqual(result['messages'][3], original['messages'][3])

    def test_openai_and_other_models_are_unchanged(self):
        for model in ['gpt-6-luna', 'anthropic/claude', 'qwen-model']:
            original = request(model)
            result, meta = optimize_request(original)
            self.assertIs(result, original)
            self.assertFalse(meta['applied'])

    def test_deepseek_prose_and_unrelated_json_requests_are_unchanged(self):
        for shape in [None, {'type': 'text'}, {'type': 'json_object'},
                      {'type': 'json_schema', 'json_schema': {'name': 'other'}}]:
            original = request()
            original['messages'][0]['content'] = 'Summarize this conversation.'
            if shape is None:
                original.pop('response_format')
            else:
                original['response_format'] = shape
            result, meta = optimize_request(original)
            self.assertIs(result, original)
            self.assertFalse(meta['applied'])

    def test_fallback_and_second_pass_do_not_duplicate_reminder(self):
        original = request('deepseek/deepseek-flash')
        original['response_format'] = {'type': 'json_object'}
        original['reasoning_effort'] = 'high'
        result, meta = optimize_request(original)
        self.assertTrue(meta['applied'])
        again, _ = optimize_request(result)
        self.assertEqual(again, result)
        self.assertEqual(result['reasoning_effort'], 'none')

    def test_invalid_messages_are_left_to_upstream_validation(self):
        original = request()
        original['messages'].append('not a message object')
        result, meta = optimize_request(original)
        self.assertIs(result, original)
        self.assertFalse(meta['applied'])


if __name__ == '__main__':
    unittest.main()
