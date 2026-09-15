# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for cua-driver tool → Capability mapping in the VSI facade.

The mapping previously used informal strings ("ax_tree", "input", ...) that
matched no Capability enum value, so every derived manifest silently carried
an empty capability set. These tests pin the mapping to real enum members.
"""

from __future__ import annotations

from types import SimpleNamespace

from leapflow.domain.platform import Capability
from leapflow.platform.facade import (
    _CUA_TOOL_TO_CAPABILITIES,
    _manifest_from_cua_tools,
)


def _fake_rpc(tool_names: list[str]) -> SimpleNamespace:
    session = SimpleNamespace(
        available_tools={name: set() for name in tool_names},
        capability_version="test",
    )
    return SimpleNamespace(_session=session)


def test_mapping_values_are_capability_members() -> None:
    """Every mapped value must be a real Capability member — a string here
    would silently drop out and reintroduce the empty-manifest bug."""
    for tool, caps in _CUA_TOOL_TO_CAPABILITIES.items():
        assert caps, f"{tool} maps to an empty capability list"
        for cap in caps:
            assert isinstance(cap, Capability), f"{tool} maps to non-enum {cap!r}"


def test_manifest_derives_capabilities_from_tools() -> None:
    manifest = _manifest_from_cua_tools(_fake_rpc([
        "get_window_state", "click", "screenshot", "launch_app", "list_apps",
    ]))
    assert manifest.supports(Capability.AX_TREE_READ)
    assert manifest.supports(Capability.AX_PERFORM_ACTION)
    assert manifest.supports(Capability.SCREEN_CAPTURE)
    assert manifest.supports(Capability.APP_LAUNCH)
    assert manifest.supports(Capability.APP_ACTIVATE)
    assert len(manifest.capabilities) > 0


def test_manifest_with_unknown_tools_only_is_empty() -> None:
    manifest = _manifest_from_cua_tools(_fake_rpc(["some_future_tool"]))
    assert len(manifest.capabilities) == 0
    # Tools still surface in metadata even without a capability mapping
    assert manifest.metadata["tools"] == ["some_future_tool"]
