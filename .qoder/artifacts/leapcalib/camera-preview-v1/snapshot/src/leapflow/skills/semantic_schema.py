# Copyright (c) Alibaba, Inc. and its affiliates.
"""Semantic tool schema support — metadata and conversion for desktop tools.

Semantic desktop tools (observe_ui, click, switch_app, ...) are registered by
the ``desktop_semantic`` plugin, which builds OpenAI function-calling schemas
from registration-style parameter strings. This module owns the shared pieces:

- ``SEMANTIC_TOOL_NAMES``: the fixed set of SemanticAdapter-backed tools that
  may be disclosed (execution-port defaults such as file_list/shell/launch_app
  are excluded — they overlap with the plugin catalog).
- ``parse_param_spec``: parses registration parameter strings, e.g.
  ``"string (optional, default=30) — max seconds to wait"``.
- ``semantic_tool_to_openai``: converts one registration-style definition
  into an OpenAI schema carrying ``x_leapflow`` metadata (used by context
  disclosure tests and fixture builders).
- ``semantic_requires_approval``: the mutating-tool classification consumed
  by the engine's desktop approval gate.

Risk metadata is declared explicitly per tool: the disclosure planner treats
missing metadata as fail-closed for core admission, and ``desktop`` is not a
category its risk table recognizes, so every entry states category, risk,
schema cost, and approval requirement. All desktop tools are non-core
(schema_cost="high"); the model obtains them via ``capability_expand``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional

# Parameter spec head, e.g. "string" or "number (optional, default=30)".
_HEAD_RE = re.compile(r"^(?P<type>[A-Za-z]+)\s*(?:\((?P<flags>[^)]*)\))?$")

# JSON schema types the bridge registration format uses; anything else maps to string.
_TYPE_MAP = {"string": "string", "number": "number", "boolean": "boolean", "int": "integer"}


@dataclass(frozen=True)
class ParamSpec:
    """Parsed registration-style parameter description."""

    type: str
    required: bool
    description: str


SEMANTIC_TOOL_NAMES: FrozenSet[str] = frozenset({
    "observe_ui",
    "click",
    "type_text",
    "shortcut",
    "switch_app",
    "list_apps",
    "list_windows",
    "open_url",
    "get_clipboard",
    "set_clipboard",
    "read_text",
    "wait",
    "wait_until",
    "wait_until_stable",
    "scroll",
    "select_text",
    "right_click",
    "screenshot",
})

_OBSERVATION_TOOLS: FrozenSet[str] = frozenset({
    "observe_ui", "list_apps", "list_windows", "read_text", "get_clipboard", "screenshot",
})
_WAIT_TOOLS: FrozenSet[str] = frozenset({
    "wait", "wait_until", "wait_until_stable",
})
_MUTATING_TOOLS: FrozenSet[str] = frozenset({
    "click", "type_text", "shortcut", "switch_app", "open_url",
    "set_clipboard", "scroll", "select_text", "right_click",
})

DESKTOP_CATEGORY = "desktop"


def _metadata_for(name: str) -> Dict[str, Any]:
    """Build x_leapflow metadata for one semantic tool.

    Observation and wait tools are read-only and run without approval;
    mutating tools are medium-risk and gated by the desktop approval gate.
    schema_cost is high for every desktop tool so none qualifies as core
    (core admission requires read_only risk AND non-high schema cost).
    """
    if name in _MUTATING_TOOLS:
        risk, approval = "medium", True
    else:
        risk, approval = "read_only", False
    return {
        "category": DESKTOP_CATEGORY,
        "risk_level": risk,
        "schema_cost": "high",
        "requires_approval": approval,
    }


def semantic_requires_approval(name: str) -> bool:
    """Return whether executing this semantic tool requires desktop approval."""
    return name in _MUTATING_TOOLS


def parse_param_spec(spec: str) -> ParamSpec:
    """Parse a registration parameter string into a structured spec.

    Accepts the registration format used by the desktop_semantic plugin and
    the skill executor's tool definitions, e.g.
    ``"string (required) — text to type"`` or ``"number (optional, default=30)
    — max seconds to wait"``. Tolerates a missing flags group, a missing
    description, and empty input.
    """
    text = (spec or "").strip()
    if not text:
        return ParamSpec(type="string", required=False, description="")

    head, sep, description = text.partition("\u2014")
    head = head.strip()
    description = description.strip() if sep else ""

    match = _HEAD_RE.match(head)
    if match is None:
        return ParamSpec(type="string", required=False, description=text)

    raw_type = match.group("type").lower()
    flags = (match.group("flags") or "").lower()
    required = "required" in flags and "optional" not in flags
    return ParamSpec(
        type=_TYPE_MAP.get(raw_type, "string"),
        required=required,
        description=description,
    )


def semantic_tool_to_openai(definition: Any) -> Optional[Dict[str, Any]]:
    """Convert one registration-style definition into an OpenAI function schema.

    Returns None for tools outside SEMANTIC_TOOL_NAMES so callers can pass
    arbitrary tool definitions without pre-filtering.
    """
    name = str(getattr(definition, "name", "") or "")
    if name not in SEMANTIC_TOOL_NAMES:
        return None

    parameters = getattr(definition, "parameters", None) or {}
    properties: Dict[str, Any] = {}
    required_names: List[str] = []
    for param_name, param_spec in parameters.items():
        parsed = parse_param_spec(str(param_spec))
        properties[str(param_name)] = {
            "type": parsed.type,
            "description": parsed.description,
        }
        if parsed.required:
            required_names.append(str(param_name))

    schema: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            "description": str(getattr(definition, "description", "") or ""),
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
        "x_leapflow": _metadata_for(name),
    }
    if required_names:
        schema["function"]["parameters"]["required"] = required_names
    return schema
