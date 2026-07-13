"""Feishu/Lark action pack — loaded from feishu.yaml.

All action definitions live in feishu.yaml next to this file.
Add new Feishu capabilities there without changing Python code.
"""
from __future__ import annotations

from pathlib import Path

from leapflow.gateway.connectors.action_registry import ActionRegistry
from leapflow.gateway.connectors.protocol import ActionSpec

_YAML_PATH = Path(__file__).parent / "feishu.yaml"
_registry = ActionRegistry.from_yaml(_YAML_PATH)

# ACTION_SPECS is the canonical export consumed by ActionRegistry.from_module().
ACTION_SPECS: dict[str, ActionSpec] = dict(_registry.all())
# Legacy alias kept for any direct imports that reference FEISHU_ACTION_SPECS.
FEISHU_ACTION_SPECS = ACTION_SPECS


def get_action_spec(action: str) -> ActionSpec | None:
    """Return a Feishu action spec by domain.operation name."""
    return _registry.get(action)
