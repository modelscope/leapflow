# Copyright (c) Alibaba, Inc. and its affiliates.
"""Platform adapter return-shape contracts (cua-driver, verified against 0.6.8).

Locks the response side of the wire contract: get_window_state's flat
elements array becomes a UISnapshot (records verbatim, no tree), dict
payloads gain an ok envelope with images preserved, the local clipboard
dispatch returns the PerceptionPort dict shape, screenshots land on disk
via screenshot_out_file, and exec_shell runs locally instead of
masquerading as an AX action.

Driven through MockBridge, so this file never contacts a real driver;
tests/test_darwin_adapter.py covers that side.
"""

from __future__ import annotations

import sys

import pytest

from leapflow.domain.platform import PlatformManifest
from leapflow.platform.adapters.darwin import (
    DarwinExecutionAdapter,
    DarwinPerceptionAdapter,
    _snapshot_from_payload,
)
from leapflow.platform.cua_client import CuaDriverClient, _local_clipboard_get
from leapflow.platform.mock import MockBridge
from leapflow.platform.protocol import RpcError


def _manifest() -> PlatformManifest:
    return PlatformManifest.default_darwin()


# ── R1: flat elements → UISnapshot ───────────────────────────────────────────

def test_snapshot_from_payload_parses_records_verbatim() -> None:
    payload = {
        "snapshot_id": "s00000042",
        "elements_complete": False,
        "total_element_count": 3,
        "capture_coverage": {
            "browser_chrome": {"status": "not_observable_in_window_scope"},
        },
        "elements": [
            {
                "element_index": 0,
                "element_token": "s00000042:0",
                "role": "Button",
                "label": "Send",
                "enabled": True,
                "frame": {"x": 1, "y": 2, "w": 30, "h": 20},
                "depth": 5,
            },
            {
                # Unlabeled Edit without frame — a real driver variant.
                "element_index": 1,
                "element_token": "s00000042:1",
                "role": "Edit",
                "value": "hello",
                "depth": 8,
            },
            {
                "element_index": 2,
                "element_token": "s00000042:2",
                "role": "TabItem",
                "label": "Docs",
                "selected": True,
                "parent_index": 1,
                "depth": 9,
            },
        ],
    }

    snapshot = _snapshot_from_payload(payload, pid=844, window_id=10725)

    assert (snapshot.pid, snapshot.window_id) == (844, 10725)
    assert snapshot.snapshot_id == "s00000042"
    assert snapshot.elements_complete is False
    assert snapshot.coverage["browser_chrome"]["status"] == "not_observable_in_window_scope"

    button, edit, tab = snapshot.elements
    assert button.target == "s00000042:0"
    assert button.frame == {"x": 1.0, "y": 2.0, "w": 30.0, "h": 20.0}
    assert edit.label == "" and edit.value == "hello" and edit.frame is None
    assert tab.selected is True and tab.parent_index == 1
    assert snapshot.find(2) is tab and snapshot.find(99) is None


def test_snapshot_from_payload_surfaces_degraded_diagnostics() -> None:
    payload = {"elements": [], "degraded": True, "degraded_reason": "ax_tree_empty"}
    snapshot = _snapshot_from_payload(payload, pid=1, window_id=2)
    assert snapshot.elements == ()
    assert snapshot.degraded is True
    assert snapshot.degraded_reason == "ax_tree_empty"


@pytest.mark.asyncio
async def test_read_window_state_over_mock_bridge_is_index_addressable() -> None:
    """MockBridge's AX_TREE payload must survive the real parse path."""
    perception = DarwinPerceptionAdapter(MockBridge(), _manifest())
    snapshot = await perception.read_window_state(100, 1)
    labels = {el.label for el in snapshot.elements}
    assert {"Save", "Cancel"} <= labels
    save = next(el for el in snapshot.elements if el.label == "Save")
    assert save.target == "s00000001:0"  # token, ready for element actions


# ── R4: ok envelope + image preservation ─────────────────────────────────────

def test_unwrap_injects_ok_and_keeps_images() -> None:
    result = CuaDriverClient._unwrap_result({
        "data": None,
        "images": ["b64=="],
        "structuredContent": {"pid": 844, "effect": "unverifiable"},
        "isError": False,
    })
    assert result["ok"] is True
    assert result["images"] == ["b64=="]
    assert result["effect"] == "unverifiable"  # effect verdicts flow through


def test_unwrap_raises_structured_error() -> None:
    with pytest.raises(RpcError):
        CuaDriverClient._unwrap_result(
            {"data": "boom", "images": [], "structuredContent": None, "isError": True}
        )


def test_unwrap_preserves_existing_ok() -> None:
    result = CuaDriverClient._unwrap_result({
        "data": {"ok": False, "error": "denied"},
        "images": [],
        "structuredContent": None,
        "isError": False,
    })
    assert result["ok"] is False and result["error"] == "denied"


# ── R2: local clipboard contract shape ───────────────────────────────────────

def test_local_clipboard_get_returns_contract_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "leapflow.platform.cua_client._clipboard_get", lambda: "copied text"
    )
    result = _local_clipboard_get({})
    assert result["text"] == "copied text"
    assert "change_count" in result and "change_ts" in result


# ── R5: screenshot lands on disk ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_capture_screenshot_returns_path() -> None:
    perception = DarwinPerceptionAdapter(MockBridge(), _manifest())
    result = await perception.capture_screenshot(pid=100, window_id=1)
    assert result["ok"] is True
    assert result["path"].endswith(".png")


# ── R7: exec_shell runs locally ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exec_shell_runs_locally_with_contract_shape() -> None:
    execution = DarwinExecutionAdapter(MockBridge(), _manifest())
    result = await execution.exec_shell("echo hello")
    assert result["ok"] is True
    assert "hello" in result["stdout"]
    assert result["exit_code"] == 0


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="exit-code shell idiom differs")
async def test_exec_shell_reports_failure() -> None:
    execution = DarwinExecutionAdapter(MockBridge(), _manifest())
    result = await execution.exec_shell("exit 3")
    assert result["ok"] is False
    assert result["exit_code"] == 3
