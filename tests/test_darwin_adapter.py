# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the Darwin platform adapter.

Verifies that each port exposed by DarwinPerceptionAdapter and
DarwinExecutionAdapter (src/leapflow/platform/adapters/darwin.py) returns
the expected values when driven against a real cua-driver, including:

- perception: fs subscription, UI tree parsing, clipboard, screenshots,
  and event streaming
- execution: file operations with undo/backup bookkeeping, UI actions,
  app launch/activate, intents, shell, input, and multi-step rollback

The whole module is skipped when the cua-driver binary is not installed
on PATH; a module-scoped fixture boots one shared client/VSI/adapter
stack for all tests.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import AsyncIterator

import pytest

from leapflow.cli.commands.host import _cua_driver_installed
from leapflow.domain.events import UISnapshot
from leapflow.platform.adapters.darwin import DarwinExecutionAdapter, DarwinPerceptionAdapter
from leapflow.platform.adapters.mock import MockExecutionAdapter, MockPerceptionAdapter
from leapflow.platform.cua_client import CuaDriverClient
from leapflow.platform.facade import VirtualSystemInterface

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not _cua_driver_installed(),
    reason="cua-driver binary not found on PATH",
)


def _is_browser_window(app_name: str):
    browsers = ["chrome", "firefox", "safari", "edge", "brave"]
    return any(browser in app_name.lower() for browser in browsers)


@dataclass(frozen=True)
class Adapters:
    """Bundle of a live cua-driver stack shared by all tests in this module."""

    rpc: CuaDriverClient
    darwin_execution: DarwinExecutionAdapter
    darwin_perception: DarwinPerceptionAdapter
    mock_execution: MockExecutionAdapter
    mock_perception: MockPerceptionAdapter


@pytest.fixture(scope="module")
async def adapters() -> AsyncIterator[Adapters]:
    rpc = CuaDriverClient()
    # hack timeout mapping
    rpc._timeout_map = {
        "ping": 3.0,
        "ax": 60.0,
        "app": 30.0,
        "app.list": 60.0,
        "input": 60.0,
        "screen": 60.0,
        "recording": 10.0,
        "clipboard": 3.0,
        "file": 15.0,
        "system": 5.0,
    }
    try:
        rpc.start()
        manifest = await VirtualSystemInterface(rpc).handshake()
    except Exception:
        try:
            rpc.stop()
        except Exception:
            logger.debug("CuaDriverClient cleanup failed after start error", exc_info=True)
        raise
    try:
        yield Adapters(
            rpc=rpc,
            darwin_execution=DarwinExecutionAdapter(rpc, manifest),
            darwin_perception=DarwinPerceptionAdapter(rpc, manifest),
            mock_execution=MockExecutionAdapter(),
            mock_perception=MockPerceptionAdapter(),
        )
    finally:
        rpc.stop()


async def test_perception_subscribe_fs(adapters: Adapters) -> None:
    """subscribe_fs() returns a str subscription id for the watched paths."""
    result1 = await adapters.darwin_perception.subscribe_fs([])
    assert isinstance(result1, str), type(result1)

    with tempfile.TemporaryDirectory() as tmpdir:
        result2 = await adapters.darwin_perception.subscribe_fs([tmpdir])
        assert isinstance(result2, str), type(result2)


async def test_perception_windows(adapters: Adapters) -> None:
    windows_info = await adapters.darwin_perception.list_windows()
    assert isinstance(windows_info, dict)
    assert "ok" in windows_info
    assert windows_info["ok"] is True
    assert "windows" in windows_info
    assert isinstance(windows_info["windows"], list)

    windows = windows_info.get("windows", [])

    # find a browser window
    windows = [window for window in windows if _is_browser_window(window.get("app_name", ""))]
    if len(windows) == 0:
        pytest.skip("No browser windows found to test read_window_state()")
    window = windows[0]

    assert isinstance(window, dict)
    assert "pid" in window
    assert "window_id" in window

    result = await adapters.darwin_perception.read_window_state(
        window["pid"], window["window_id"]
    )
    assert isinstance(result, UISnapshot)
    assert (result.pid, result.window_id) == (window["pid"], window["window_id"])
    for element in result.elements:
        assert isinstance(element.element_index, int)
        assert element.target  # token or index — always addressable


async def test_perception_execution_clipboard(adapters: Adapters) -> None:
    text = "I am the text for testing clipboard"
    await adapters.darwin_execution.set_clipboard(text)
    result = await adapters.darwin_perception.get_clipboard()

    assert isinstance(result, dict)
    assert "ok" in result
    assert result["ok"] is True
    assert "text" in result
    assert result["text"] == text


async def test_perception_capture_screenshot(adapters: Adapters) -> None:
    """Window-scoped capture writes a PNG; a targetless request is refused.

    cua-driver exposes no full-display capture tool, so capture_screenshot()
    needs a (pid, window_id) pair. A targetless call used to be mapped onto
    get_desktop_state and came back from the driver as "Unknown tool".

    list_windows reports every layer-0 surface (~200 here), most of them
    offscreen service windows that produce neither an AX tree nor an image.
    Selecting on is_on_screen plus a real size is what makes this deterministic;
    the smallest qualifying window is used because the AX walk that accompanies
    the capture scales with element count.
    """
    from leapflow.platform.protocol import RpcError

    windows_info = await adapters.darwin_perception.list_windows()

    def _area(window: dict) -> float:
        bounds = window.get("bounds") or {}
        return float(bounds.get("width", 0)) * float(bounds.get("height", 0))

    candidates = sorted(
        (
            w
            for w in windows_info.get("windows", [])
            if w.get("is_on_screen")
            and float((w.get("bounds") or {}).get("width", 0)) >= 200
            and float((w.get("bounds") or {}).get("height", 0)) >= 200
        ),
        key=_area,
    )
    if not candidates:
        pytest.skip("no on-screen window large enough to capture")
    window = candidates[0]

    result = await adapters.darwin_perception.capture_screenshot(
        pid=window["pid"], window_id=window["window_id"]
    )
    assert isinstance(result, dict)
    assert result["ok"] is True
    # The image is routed to disk; a base64 payload must never reach context.
    assert result["path"] == result["screenshot_file_path"]
    assert os.path.exists(result["path"])
    assert os.path.getsize(result["path"]) > 0

    with pytest.raises(RpcError) as excinfo:
        await adapters.darwin_perception.capture_screenshot()
    assert excinfo.value.code == "invalid_params"


async def test_execution_perform_file_op(adapters: Adapters) -> None:
    """perform_file_op() list/move/copy/delete all return ok dicts with result paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, world!")

        result1 = await adapters.darwin_execution.perform_file_op(
            "list", {"path": tmpdir}
        )
        assert isinstance(result1, dict)
        assert "ok" in result1
        assert result1["ok"] is True
        assert "result" in result1
        assert isinstance(result1["result"], list)

        names = [entry["name"] for entry in result1["result"]]
        assert "test.txt" in names

        result2 = await adapters.darwin_execution.perform_file_op(
            "move", {"source": test_file, "destination": os.path.join(tmpdir, "test2.txt")}
        )
        assert isinstance(result2, dict)
        assert result2["ok"] is True
        assert "moved" in result2
        assert result2["moved"] == os.path.join(tmpdir, "test2.txt")

        result3 = await adapters.darwin_execution.perform_file_op(
            "copy", {"source": result2["moved"], "destination": os.path.join(tmpdir, "test3.txt")}
        )
        assert isinstance(result3, dict)
        assert result3["ok"] is True
        assert "copied" in result3
        assert result3["copied"] == os.path.join(tmpdir, "test3.txt")

        result4 = await adapters.darwin_execution.perform_file_op(
            "delete", {"path": result3["copied"]}
        )
        assert isinstance(result4, dict)
        assert result4["ok"] is True
        assert "deleted" in result4
        assert result4["deleted"] == os.path.join(tmpdir, "test3.txt")


async def test_execution_undo_restores_deleted_file(adapters: Adapters) -> None:
    """undo_last() after a delete restores the file from the pre-delete backup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "undo_me.txt")
        with open(test_file, "w") as f:
            f.write("precious content")

        depth_before = adapters.darwin_execution.undo_depth
        result = await adapters.darwin_execution.perform_file_op(
            "delete", {"path": test_file}
        )
        assert result["ok"] is True
        assert result.get("deleted") == test_file
        assert not os.path.exists(test_file)
        assert adapters.darwin_execution.undo_depth == depth_before + 1

        undone = await adapters.darwin_execution.undo_last()
        assert undone.get("ok") is True, undone
        assert os.path.exists(test_file)
        with open(test_file) as f:
            assert f.read() == "precious content"


async def test_execution_list_apps(adapters: Adapters) -> None:
    """list_apps() returns app records carrying name and an integer pid field."""
    result = await adapters.darwin_execution.list_apps()
    assert isinstance(result, dict)
    assert result.get("ok") is True
    apps = result.get("apps")
    assert isinstance(apps, list) and len(apps) > 0
    for record in apps[:5]:
        assert isinstance(record, dict)
        assert "name" in record
        assert isinstance(record.get("pid"), int)


async def test_execution_exec_shell(adapters: Adapters) -> None:
    """exec_shell() runs locally and reports stdout plus a zero exit code."""
    result = await adapters.darwin_execution.exec_shell("echo hello-leapflow")
    assert result["ok"] is True
    assert "hello-leapflow" in result["stdout"]
    assert result["exit_code"] == 0

    failed = await adapters.darwin_execution.exec_shell(
        "exit 3" if sys.platform != "win32" else "cmd /c exit 3"
    )
    assert failed["ok"] is False
    assert failed["exit_code"] == 3


async def test_execution_run_intent_degrades_without_capability(adapters: Adapters) -> None:
    """run_intent() reports intents_not_supported when the manifest lacks the capability."""
    result = await adapters.darwin_execution.run_intent("open_note", {})
    assert result == {"ok": False, "error": "intents_not_supported"}


async def test_execution_perform_ui_action_rejects_stale_token(adapters: Adapters) -> None:
    """perform_ui_action() with a fabricated element_token is refused by the driver."""
    from leapflow.platform.protocol import RpcError

    with pytest.raises(RpcError):
        await adapters.darwin_execution.perform_ui_action("s99999999:0", "press")


@pytest.mark.skipif(sys.platform != "win32", reason="notepad round-trip is Windows-only")
async def test_execution_notepad_input_roundtrip(adapters: Adapters) -> None:
    """launch/activate/type_text/shortcut/scroll round-trip, verified via clipboard."""
    text = "leapflow input roundtrip input test"

    # Modern Windows Notepad restores the previous session; open a fresh
    # empty file so the focused tab is clean and select-all captures only
    # what this test types. Close the handle so notepad can open the file.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    temp_file.close()

    launch = await adapters.darwin_execution.launch_app(
        "notepad", urls=[temp_file.name]
    )
    assert launch.get("ok") is True, launch
    pid = launch.get("pid")
    assert isinstance(pid, int) and pid > 0, launch

    try:
        # The launch response carries the windows array; fall back to
        # polling list_windows until the window is registered.
        window_id = None
        for record in launch.get("windows") or []:
            if isinstance(record.get("window_id"), int):
                window_id = record["window_id"]
                break
        for _ in range(10):
            if window_id is not None:
                break
            await asyncio.sleep(0.5)
            listed = await adapters.darwin_perception.list_windows()
            for record in listed.get("windows", []):
                if record.get("pid") == pid and isinstance(record.get("window_id"), int):
                    window_id = record["window_id"]
                    break
        assert window_id is not None, "notepad window never appeared"

        activated = await adapters.darwin_execution.activate_app(pid, window_id)
        assert isinstance(activated, dict), activated
        await asyncio.sleep(1.0)

        typed = await adapters.darwin_execution.type_text(text)
        assert typed.get("ok") is True, typed
        await asyncio.sleep(0.5)

        # Select-all + copy, then verify the typed text through the clipboard.
        await adapters.darwin_execution.send_shortcut("ctrl+a")
        await asyncio.sleep(0.3)
        await adapters.darwin_execution.send_shortcut("ctrl+c")
        await asyncio.sleep(0.5)
        clipboard = await adapters.darwin_perception.get_clipboard()
        assert clipboard.get("text") == text, clipboard

        # Targetless scroll drives the window's focused scroller (keystroke
        # path); the driver requires pid/window_id even without an element.
        scrolled = await adapters.darwin_execution.scroll(
            "", "down", 2, pid=pid, window_id=window_id
        )
        assert scrolled.get("ok") is True, scrolled
    finally:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True, timeout=10,
        )
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass

