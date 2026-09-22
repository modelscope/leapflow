# Copyright (c) Alibaba, Inc. and its affiliates.
"""Behavioral tests for ActionDescriptor: signature stability, _normalize_detail
rules, device resource formatting and effect mapping, MCP trust boundary and
description truncation, network_fetch origin scoping, and platform action
effect inference.

Deterministic and offline; no IO beyond object construction.
"""

from __future__ import annotations

import pytest

from leapflow.security.actions import (
    ActionDescriptor,
    ActionEffect,
    ActionKind,
    _normalize_detail,
)


# ═══════════════════════════════════════════════════════════════════
# Signature stability
# ═══════════════════════════════════════════════════════════════════


class TestSignatureStability:
    """Signature must be deterministic and sensitive to identity fields."""

    def test_same_descriptor_yields_same_signature(self) -> None:
        desc = ActionDescriptor.shell("ls -la", cwd="/tmp")
        assert desc.signature() == desc.signature()

    def test_different_resource_yields_different_signature(self) -> None:
        a = ActionDescriptor.file_read("/a")
        b = ActionDescriptor.file_read("/b")
        assert a.signature() != b.signature()

    def test_different_effect_yields_different_signature(self) -> None:
        base = ActionDescriptor(
            kind="test.kind", summary="s", detail="d",
            effect=ActionEffect.READ.value, resource="r",
        )
        mutated = ActionDescriptor(
            kind="test.kind", summary="s", detail="d",
            effect=ActionEffect.WRITE.value, resource="r",
        )
        assert base.signature() != mutated.signature()

    def test_different_origin_yields_different_signature(self) -> None:
        a = ActionDescriptor.shell("echo hi", origin="agent_tool")
        b = ActionDescriptor.shell("echo hi", origin="scheduler")
        assert a.signature() != b.signature()


# ═══════════════════════════════════════════════════════════════════
# _normalize_detail
# ═══════════════════════════════════════════════════════════════════


class TestNormalizeDetail:
    """Detail normalization rules by action kind."""

    def test_platform_action_collapses_to_placeholder(self) -> None:
        assert _normalize_detail(ActionKind.PLATFORM_ACTION.value, "long json") == "<platform-payload>"

    def test_gateway_send_collapses_to_placeholder(self) -> None:
        assert _normalize_detail(ActionKind.GATEWAY_SEND.value, "msg body") == "<platform-payload>"

    def test_network_fetch_collapses(self) -> None:
        assert _normalize_detail(ActionKind.NETWORK_FETCH.value, "https://x.com/long/path") == "<network-target>"

    def test_mcp_tool_collapses(self) -> None:
        assert _normalize_detail(ActionKind.MCP_TOOL.value, "arbitrary args") == "<mcp-invocation>"

    def test_device_kinds_collapse(self) -> None:
        for kind in (
            ActionKind.DEVICE_READ, ActionKind.DEVICE_ACTUATE,
            ActionKind.DEVICE_CONFIGURE, ActionKind.DEVICE_DISPENSE,
        ):
            assert _normalize_detail(kind.value, "set 42") == "<device-command>"

    def test_default_kind_preserves_text_truncated(self) -> None:
        long_text = "x" * 5000
        result = _normalize_detail(ActionKind.FILE_WRITE.value, long_text)
        assert len(result) <= 4000


# ═══════════════════════════════════════════════════════════════════
# ActionDescriptor.device
# ═══════════════════════════════════════════════════════════════════


class TestDeviceDescriptor:
    """Device resource formatting and effect mapping."""

    def test_resource_includes_band(self) -> None:
        desc = ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id="pump-1", channel_id="flow",
            envelope_band="0-100ml/min",
        )
        assert desc.resource == "pump-1:flow@0-100ml/min"

    def test_resource_without_band(self) -> None:
        desc = ActionDescriptor.device(
            kind=ActionKind.DEVICE_READ.value,
            device_id="sensor-1", channel_id="temp",
        )
        assert desc.resource == "sensor-1:temp"

    def test_effect_mapping(self) -> None:
        mapping = {
            ActionKind.DEVICE_READ.value: ActionEffect.READ.value,
            ActionKind.DEVICE_CONFIGURE.value: ActionEffect.CONFIGURE.value,
            ActionKind.DEVICE_ACTUATE.value: ActionEffect.EXECUTE.value,
            ActionKind.DEVICE_DISPENSE.value: ActionEffect.EXECUTE.value,
            ActionKind.DEVICE_ESTOP.value: ActionEffect.EXECUTE.value,
        }
        for kind, expected_effect in mapping.items():
            desc = ActionDescriptor.device(
                kind=kind, device_id="d", channel_id="c",
            )
            assert desc.effect == expected_effect, f"kind={kind}"

    def test_location_appears_in_summary(self) -> None:
        desc = ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id="arm-1", channel_id="joint",
            location="Lab B",
        )
        assert "Lab B" in desc.summary


# ═══════════════════════════════════════════════════════════════════
# ActionDescriptor.mcp_tool
# ═══════════════════════════════════════════════════════════════════


class TestMcpToolDescriptor:
    """MCP tool trust boundary, resource, and description truncation."""

    def test_resource_is_server_colon_tool(self) -> None:
        desc = ActionDescriptor.mcp_tool(server="my-server", tool="search")
        assert desc.resource == "my-server:search"

    def test_read_only_sets_read_effect(self) -> None:
        desc = ActionDescriptor.mcp_tool(server="s", tool="t", read_only=True)
        assert desc.effect == ActionEffect.READ.value

    def test_mutating_sets_execute_effect(self) -> None:
        desc = ActionDescriptor.mcp_tool(server="s", tool="t", read_only=False)
        assert desc.effect == ActionEffect.EXECUTE.value

    def test_description_truncated_at_400_chars(self) -> None:
        long_desc = "A" * 600
        desc = ActionDescriptor.mcp_tool(server="s", tool="t", description=long_desc)
        # The description appears in detail after a prefix; only 400 chars of it.
        desc_portion = desc.detail.split(": ", 1)[-1]
        assert len(desc_portion) <= 400

    def test_server_leads_summary(self) -> None:
        desc = ActionDescriptor.mcp_tool(server="acme-mcp", tool="do_thing")
        assert "acme-mcp" in desc.summary


# ═══════════════════════════════════════════════════════════════════
# ActionDescriptor.network_fetch
# ═══════════════════════════════════════════════════════════════════


class TestNetworkFetchDescriptor:
    """network_fetch uses origin (host), not full URL, as the resource."""

    def test_resource_is_origin_not_full_url(self) -> None:
        desc = ActionDescriptor.network_fetch(
            "https://api.example.com/v1/data?q=test",
            origin="https://api.example.com",
        )
        assert desc.resource == "https://api.example.com"

    def test_full_url_stored_in_detail(self) -> None:
        url = "https://api.example.com/v1/data?q=test"
        desc = ActionDescriptor.network_fetch(url, origin="https://api.example.com")
        assert desc.detail == url


# ═══════════════════════════════════════════════════════════════════
# Platform action effect inference
# ═══════════════════════════════════════════════════════════════════


class TestPlatformActionEffect:
    """Effect inference for representative read/write/send/delete verbs."""

    @pytest.mark.parametrize(
        "action,expected",
        [
            ("get_users", ActionEffect.READ.value),
            ("list_items", ActionEffect.READ.value),
            ("send_message", ActionEffect.SEND.value),
            ("reply_text", ActionEffect.SEND.value),
            ("create_doc", ActionEffect.WRITE.value),
            ("update_record", ActionEffect.WRITE.value),
            ("delete_item", ActionEffect.DELETE.value),
            ("remove_member", ActionEffect.DELETE.value),
        ],
    )
    def test_effect_for_action_verb(self, action: str, expected: str) -> None:
        desc = ActionDescriptor.platform_action("feishu", action, {"x": 1})
        assert desc.effect == expected
