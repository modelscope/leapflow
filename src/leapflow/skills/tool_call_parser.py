# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool call parsing utilities for LLM response content.

Extracts structured tool calls from free-form LLM text output. Supports
multiple formats: markdown JSON code blocks, XML-style ``<tool_call>``
wrappers, inline JSON with ``"name"``/``"tool"`` keys, and legacy
``{"tool": ..., "params": {...}}`` format.

Also provides a repetition detector that aborts when the LLM is stuck
in a degenerate output loop.

These utilities are shared between the legacy ReAct skill executor
(``tool_executor.py``) and the unified agent tool dispatch engine
(``tool_dispatch_engine.py``).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from leapflow.skills.tool_types import ToolCall

_TOOL_CALL_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
)
_INLINE_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:', re.DOTALL
)


def detect_repetition(content: str, threshold: int = 10) -> bool:
    """Detect if LLM output is stuck in a repetitive pattern."""
    if len(content) < 100:
        return False
    # Check for repeated closing tags (common failure mode)
    repeated_patterns = ["</invoke>", "</tool_call>", "```\n```"]
    for pattern in repeated_patterns:
        if content.count(pattern) >= threshold:
            return True
    # Check last 200 chars for character-level repetition
    tail = content[-200:]
    if len(set(tail.split())) <= 3 and len(tail) > 50:
        return True
    return False


def parse_tool_call(content: str) -> Optional[ToolCall]:
    """Extract a tool call JSON from LLM response text.

    Supports:
    - ```json {"name": ..., "arguments": {...}} ``` (primary)
    - Inline {"name": ...} patterns
    - <tool_call>{"name": ..., "arguments": {...}}</tool_call> patterns
    - Legacy {"tool": ..., "params": {...}} format
    """
    # 1. Standard markdown code block
    match = _TOOL_CALL_PATTERN.search(content)
    if match:
        result = _try_parse_json(match.group(1))
        if result:
            return result

    # 2. <tool_call> XML-style wrapper
    tc_match = re.search(
        r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|</invoke>)', content, re.DOTALL
    )
    if tc_match:
        result = _try_parse_json(tc_match.group(1))
        if result:
            return result

    # 3. Inline JSON with "name" or "tool" key
    idx = -1
    for pattern_str in ['"name"', '"tool"']:
        search = content.find('{')
        while search != -1:
            # Check if this { starts a valid tool call JSON
            if pattern_str in content[search:search + 50]:
                idx = search
                break
            search = content.find('{', search + 1)
        if idx != -1:
            break

    if idx == -1:
        match2 = _INLINE_JSON_PATTERN.search(content)
        if match2:
            idx = match2.start()

    if idx != -1:
        depth = 0
        end = idx
        for i in range(idx, min(len(content), idx + 2000)):  # limit scan to 2000 chars
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth == 0:
            return _try_parse_json(content[idx:end])

    return None


def _try_parse_json(text: str) -> Optional[ToolCall]:
    """Parse a JSON string into a ToolCall (OpenAI function calling format)."""
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        # Primary: OpenAI function calling format {"name": ..., "arguments": {...}}
        if "name" in data:
            name = data["name"]
            params = data.get("arguments", data.get("params", data.get("parameters", {})))
            if isinstance(params, str):
                # Sometimes arguments is a JSON string
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {"raw": params}
            return ToolCall(name=str(name), params=params if isinstance(params, dict) else {})

        # Fallback: legacy {"tool": ..., "params": {...}} format
        if "tool" in data:
            return ToolCall(
                name=data["tool"],
                params=data.get("params", data.get("arguments", data.get("parameters", {}))) or {},
            )

    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None
