# Copyright (c) Alibaba, Inc. and its affiliates.
"""SemanticAdapter window-target and element_index addressing.

Drives the real SemanticAdapter over the mock perception/execution adapters
(no object.__new__ shortcuts) to lock the wiring: list_windows supplies
pid/window_id, observe_ui snapshots a window with indexed elements, action
tools resolve element_index against the latest snapshot and address the
driver via element_token, and untargeted calls return structured guidance.
"""

from __future__ import annotations

import pytest

from leapflow.platform.adapters.mock import MockExecutionAdapter, MockPerceptionAdapter
from leapflow.skills.semantic_adapter import SemanticAdapter


def _adapter() -> SemanticAdapter:
    return SemanticAdapter(
        perception=MockPerceptionAdapter(),
        execution=MockExecutionAdapter(),
        settle_delay=0.0,
    )


@pytest.mark.asyncio
async def test_list_windows_supplies_pid_and_window_id() -> None:
    adapter = _adapter()
    result = await adapter.list_windows({})
    assert result["ok"] is True
    record = result["windows"][0]
    assert isinstance(record["pid"], int) and isinstance(record["window_id"], int)


@pytest.mark.asyncio
async def test_observe_ui_requires_window_target() -> None:
    adapter = _adapter()
    result = await adapter.observe_ui({})
    assert result["ok"] is False
    assert result["error"] == "missing_window_target"
    assert "list_windows" in result["suggestion"]


@pytest.mark.asyncio
async def test_observe_ui_indexes_elements() -> None:
    adapter = _adapter()
    observed = await adapter.observe_ui({"pid": 100, "window_id": 1})
    assert observed["ok"] is True
    assert (observed["pid"], observed["window_id"]) == (100, 1)
    indices = [el["element_index"] for el in observed["elements"]]
    assert indices == [0, 1, 2, 3]
    tab = observed["elements"][3]
    assert tab["selected"] is True  # driver state fields reach the model


@pytest.mark.asyncio
async def test_click_addresses_element_by_index_via_token() -> None:
    adapter = _adapter()
    await adapter.observe_ui({"pid": 100, "window_id": 1})
    clicked = await adapter.click({"element_index": 0})
    assert clicked["ok"] is True
    assert "state_after" in clicked

    execution: MockExecutionAdapter = adapter._execution  # type: ignore[assignment]
    record = next(r for r in execution.history if r["type"] == "ui_action")
    assert record["node_id"] == "s00000001:0"  # element_token, not a selector
    assert record["action"] == "press"


@pytest.mark.asyncio
async def test_actions_without_snapshot_return_guidance() -> None:
    adapter = _adapter()
    result = await adapter.click({"element_index": 0})
    assert result["ok"] is False and result["error"] == "no_snapshot"

    stale = await adapter.observe_ui({"pid": 100, "window_id": 1})
    assert stale["ok"] is True
    missing = await adapter.click({"element_index": 99})
    assert missing["ok"] is False
    assert missing["error"] == "element_not_found: 99"
    assert "observe_ui" in missing["suggestion"]


@pytest.mark.asyncio
async def test_scroll_requires_window_target() -> None:
    adapter = _adapter()
    result = await adapter.scroll({"direction": "down"})
    assert result["ok"] is False and result["error"] == "missing_window_target"


@pytest.mark.asyncio
async def test_scroll_carries_window_pid_for_keystroke_path() -> None:
    adapter = _adapter()
    await adapter.observe_ui({"pid": 100, "window_id": 1})
    result = await adapter.scroll({"direction": "up", "amount": 5})
    assert result["ok"] is True

    execution: MockExecutionAdapter = adapter._execution  # type: ignore[assignment]
    record = next(r for r in execution.history if r["type"] == "scroll")
    assert record["node_id"] == ""  # no element target — keystroke path
    assert record["direction"] == "up" and record["amount"] == 5
    assert (record["pid"], record["window_id"]) == (100, 1)  # driver requires pid


@pytest.mark.asyncio
async def test_wait_until_falls_back_to_last_snapshot_target() -> None:
    adapter = _adapter()
    missing = await adapter.wait_until({"condition": "Save"})
    assert missing["ok"] is False and missing["error"] == "missing_window_target"

    await adapter.observe_ui({"pid": 100, "window_id": 1})
    met = await adapter.wait_until(
        {"condition": "Save", "timeout": 1, "poll_interval": 0.5}
    )
    assert met["ok"] is True and met["met"] is True


@pytest.mark.asyncio
async def test_switch_app_uses_launch_response_target() -> None:
    adapter = _adapter()
    result = await adapter.switch_app({"app_id": "com.mock.app"})
    assert result["ok"] is True
    assert (result["pid"], result["window_id"]) == (100, 1)

    execution: MockExecutionAdapter = adapter._execution  # type: ignore[assignment]
    activation = next(r for r in execution.history if r["type"] == "activate_app")
    assert activation["pid"] == 100


@pytest.mark.asyncio
async def test_list_apps_filters_locally() -> None:
    adapter = _adapter()
    everything = await adapter.list_apps({})
    assert [r["name"] for r in everything["apps"]] == ["Mock App"]

    none_match = await adapter.list_apps({"filter": "absent"})
    assert none_match["apps"] == []

    running = await adapter.list_apps({"running_only": True})
    assert [r["bundle_id"] for r in running["apps"]] == ["com.mock.app"]


@pytest.mark.asyncio
async def test_type_text_has_no_method_knob() -> None:
    adapter = _adapter()
    result = await adapter.type_text({"text": "hello"})
    assert result["ok"] is True

    execution: MockExecutionAdapter = adapter._execution  # type: ignore[assignment]
    record = next(r for r in execution.history if r["type"] == "type_text")
    assert record == {"type": "type_text", "text": "hello"}


@pytest.mark.asyncio
async def test_describe_element_resolves_semantics_for_policy() -> None:
    adapter = _adapter()
    assert adapter.describe_element({"element_index": 0}) == ""  # no snapshot yet

    await adapter.observe_ui({"pid": 100, "window_id": 1})
    assert adapter.describe_element({"element_index": 0}) == "Button Save"
    assert adapter.describe_element({"element_index": 99}) == ""
    assert adapter.describe_element({}) == ""


@pytest.mark.asyncio
async def test_read_text_reads_from_snapshot() -> None:
    adapter = _adapter()
    await adapter.observe_ui({"pid": 100, "window_id": 1})
    result = await adapter.read_text({"element_index": 2})
    assert result["ok"] is True and result["text"] == "hello"
