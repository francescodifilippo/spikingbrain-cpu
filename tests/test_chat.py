import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spikingbrain_cpu.chat import ChatTemplate, parse_tool_calls  # noqa: E402

# Shortened copy of the checkpoint template's tool section, enough to check
# rendering of tools, assistant tool calls and tool responses.
TEMPLATE = (
    "{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0].role == 'system' %}"
    "{{- messages[0].content + '\\n\\n' }}{%- endif %}{{- '<tools>' }}{%- for tool in tools %}"
    "{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools><|im_end|>\\n' }}{%- endif %}"
    "{%- for message in messages %}{%- if message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\\n' + (message.content or '') }}"
    "{%- for tool_call in message.tool_calls or [] %}{%- set tool_call = tool_call.function %}"
    "{{- '<tool_call>\\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' + tool_call.arguments + '}\\n</tool_call>' }}"
    "{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


class ChatTest(unittest.TestCase):
    def test_render_tools_unicode_and_turns(self):
        tools = [{"type": "function", "function": {"name": "events_list", "description": "Événements naïfs",
                                                    "parameters": {"type": "object", "properties": {}}}}]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "events_list", "arguments": "{\"a\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        ]
        text = ChatTemplate(TEMPLATE).render(messages, tools)
        self.assertIn('"description": "Événements naïfs"', text)
        self.assertIn('<tool_call>\n{"name": "events_list", "arguments": {"a": 1}}\n</tool_call>', text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))

    def test_parse_tool_calls(self):
        text = 'thinking\n<tool_call>\n{"name": "pods_list", "arguments": {"namespace": "x"}}\n</tool_call>' \
               '<tool_call>\n{"name": "events_list", "arguments": "{}"}\n</tool_call>'
        content, calls, recovered = parse_tool_calls(text, {"pods_list", "events_list"})
        self.assertEqual(content, "thinking")
        self.assertEqual(recovered, 0)
        self.assertEqual([c["function"]["name"] for c in calls], ["pods_list", "events_list"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"namespace": "x"})
        self.assertTrue(all(c["id"].startswith("call_") for c in calls))

    def test_invalid_or_truncated_calls_stay_raw(self):
        for text in ('<tool_call>\n{"name": broken}\n</tool_call>', '<tool_call>\n{"name": "x", "argu'):
            content, calls, _ = parse_tool_calls(text)
            self.assertEqual(calls, [])
            self.assertEqual(content, text.strip())

    def test_bare_json_recovery_only_for_declared_tools(self):
        text = 'agger\n{"name": "pods_list_in_namespace", "arguments": {"namespace": "monitoring"}}\nagger'
        content, calls, recovered = parse_tool_calls(text, {"pods_list_in_namespace"})
        self.assertEqual((content, recovered), (None, 1))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"namespace": "monitoring"})
        # undeclared tool names, inline JSON and tool-less requests stay raw
        for names, raw in (({"other"}, text), ({"pods_list_in_namespace"}, 'see {"name": "pods_list_in_namespace", "arguments": {}}'),
                           (None, text)):
            content, calls, recovered = parse_tool_calls(raw, names)
            self.assertEqual((calls, recovered, content), ([], 0, raw))

    def test_bare_json_keeps_prose_and_tagged_calls_win(self):
        text = 'Controllo i pod.\n{"name": "a", "arguments": {}}\n{"name": "b", "arguments": {"x": 1}}'
        content, calls, recovered = parse_tool_calls(text, {"a", "b"})
        self.assertEqual(content, "Controllo i pod.")
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        tagged = '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n{"name": "b", "arguments": {}}'
        content, calls, recovered = parse_tool_calls(tagged, {"a", "b"})
        self.assertEqual(([c["function"]["name"] for c in calls], recovered), (["a"], 0))


if __name__ == "__main__":
    unittest.main()
