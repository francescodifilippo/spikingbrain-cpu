"""Chat template rendering and tool-call extraction (Hermes/Qwen style tags)."""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime

from jinja2.sandbox import ImmutableSandboxedEnvironment

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
REQUIRED_TOOL_PREFIX = '<tool_call>\n{"name": "'


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise ValueError(message)


class ChatTemplate:
    """Renders the checkpoint's Jinja chat template like Hugging Face tokenizers do."""

    def __init__(self, source: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)

    def render(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> str:
        return self.template.render(messages=messages, tools=tools or None, add_generation_prompt=True, **kwargs)


def _call(name: str, arguments: dict) -> dict:
    return {
        "id": "call_" + secrets.token_hex(12),
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _shape(body) -> tuple[str, dict]:
    name = body["name"]
    arguments = body.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("bad tool call shape")
    return name, arguments


# A delimiter line the model emitted instead of the tag: one short word.
DELIMITER = re.compile(r"[^\s{}]{1,16}")


def _bare_calls(text: str, tool_names: set[str]) -> tuple[str, list[dict]]:
    """Find ``{"name": <declared tool>, "arguments": {...}}`` objects outside tags.

    Some checkpoints choose a wrong token instead of ``<tool_call>`` while still
    writing a well-formed call. Only objects that start a line, name a declared
    tool and carry an argument object are accepted; a one-word line directly
    before or after such an object is treated as the misplaced delimiter.
    """
    decoder = json.JSONDecoder()
    lines = text.split("\n")
    calls, keep = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("{"):
            rest = "\n".join(lines[i:]).lstrip()
            try:
                body, end = decoder.raw_decode(rest)
                name, arguments = _shape(body)
            except (ValueError, KeyError, TypeError, AttributeError):
                keep.append(line)
                i += 1
                continue
            if name not in tool_names:
                keep.append(line)
                i += 1
                continue
            calls.append(_call(name, arguments))
            if keep and DELIMITER.fullmatch(keep[-1].strip()):
                keep.pop()
            i += rest[:end].count("\n") + 1
            if i < len(lines) and DELIMITER.fullmatch(lines[i].strip()):
                i += 1
            continue
        keep.append(line)
        i += 1
    return "\n".join(keep), calls


def parse_tool_calls(text: str, tool_names: set[str] | None = None) -> tuple[str | None, list[dict], int]:
    """Split generated text into plain content and OpenAI-style tool calls.

    Returns (content, calls, recovered) where ``recovered`` counts calls found
    by the bare-JSON fallback (only when no tagged call exists and
    ``tool_names`` is given). Blocks that are not valid
    ``{"name": str, "arguments": object}`` JSON stay in the content unchanged.
    """
    calls = []
    kept = []
    last = 0
    for match in TOOL_CALL.finditer(text):
        try:
            name, arguments = _shape(json.loads(match.group(1)))
        except (ValueError, KeyError, TypeError):
            continue
        kept.append(text[last:match.start()])
        last = match.end()
        calls.append(_call(name, arguments))
    kept.append(text[last:])
    content = "".join(kept)
    recovered = 0
    if not calls and tool_names:
        content, calls = _bare_calls(content, tool_names)
        recovered = len(calls)
    content = content.strip()
    return (content or None), calls, recovered
