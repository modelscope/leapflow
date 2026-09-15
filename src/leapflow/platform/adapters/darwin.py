# Copyright (c) Alibaba, Inc. and its affiliates.
"""macOS adapter — maps VSI ports to HostRpc calls targeting CuaDriver."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import uuid
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from leapflow.domain.events import SystemEvent, UIElement, UISnapshot
from leapflow.domain.platform import Capability, PlatformManifest
from leapflow.platform.protocol import HostRpc, Methods

logger = logging.getLogger(__name__)


class DarwinPerceptionAdapter:
    """PerceptionPort implementation backed by CuaDriver RPC."""

    def __init__(self, rpc: HostRpc, manifest: PlatformManifest) -> None:
        self._rpc = rpc
        self._manifest = manifest
        self._event_queue: asyncio.Queue[SystemEvent] = asyncio.Queue(maxsize=512)

    async def subscribe_fs(self, paths: List[str]) -> str:
        result = await self._rpc.call(Methods.FS_SUBSCRIBE, {"path": paths[0] if paths else "~"})
        return str(result.get("subscription_id", ""))

    async def read_window_state(
        self, pid: int, window_id: int, query: str = ""
    ) -> UISnapshot:
        params: Dict[str, Any] = {
            "pid": pid,
            "window_id": window_id,
            "include_screenshot": False,
        }
        if query:
            params["query"] = query
        result = await self._rpc.call(Methods.AX_TREE, params)
        payload = result if isinstance(result, dict) else {}
        return _snapshot_from_payload(payload, pid=pid, window_id=window_id)

    async def list_windows(self) -> Dict[str, Any]:
        """List top-level windows; the source of pid/window_id targets."""
        result = await self._rpc.call(Methods.AX_LIST, {})
        return result if isinstance(result, dict) else {"windows": result}

    async def get_clipboard(self) -> Dict[str, Any]:
        result = await self._rpc.call(Methods.CLIPBOARD_GET, {})
        if isinstance(result, dict):
            return result
        return {"text": str(result or ""), "change_count": 0, "change_ts": None}

    async def capture_screenshot(
        self, pid: Optional[int] = None, window_id: Optional[int] = None
    ) -> Dict[str, Any]:
        # Route the image to disk: base64 payloads must never enter context.
        out_file = str(
            Path(tempfile.gettempdir()) / f"leapflow_screenshot_{uuid.uuid4().hex[:8]}.png"
        )
        params: Dict[str, Any] = {"screenshot_out_file": out_file}
        if pid is not None and window_id is not None:
            params["pid"] = pid
            params["window_id"] = window_id
        result = await self._rpc.call(Methods.SCREEN_CAPTURE_FRAME, params)
        if not isinstance(result, dict):
            return {"ok": True, "path": out_file}
        result.setdefault("path", result.get("screenshot_file_path") or out_file)
        return result

    async def stream_events(self) -> AsyncIterator[SystemEvent]:
        while True:
            event = await self._event_queue.get()
            yield event

    def enqueue_event(self, event: SystemEvent) -> None:
        """Called by EventBus to feed normalized events into the stream."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._event_queue.get_nowait()
            self._event_queue.put_nowait(event)


class DarwinExecutionAdapter:
    """ExecutionPort implementation backed by CuaDriver RPC.

    Features a bounded undo stack supporting multi-step rollback and
    pre-delete backup for file recovery.
    """

    def __init__(
        self, rpc: HostRpc, manifest: PlatformManifest, *, undo_capacity: int = 20
    ) -> None:
        self._rpc = rpc
        self._manifest = manifest
        self._undo_stack: deque[Dict[str, Any]] = deque(maxlen=undo_capacity)

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    async def perform_file_op(self, op: str, params: Dict[str, Any]) -> Dict[str, Any]:
        method_map = {
            "list": Methods.FILE_LIST,
            "move": Methods.FILE_MOVE,
            "copy": Methods.FILE_COPY,
            "delete": Methods.FILE_DELETE,
        }
        method = method_map.get(op)
        if method is None:
            return {"ok": False, "error": f"unsupported_file_op:{op}"}

        # Pre-delete backup (must happen before the delete RPC)
        backup = ""
        if op == "delete":
            backup = await self._backup_for_undo(params.get("path", ""))

        result = await self._rpc.call(method, params)

        # Push undo record only AFTER successful RPC execution
        if op == "delete":
            self._undo_stack.append({
                "type": "file_delete",
                "backup": backup,
                "original": params.get("path", ""),
            })
        elif op in ("move", "copy"):
            self._undo_stack.append({
                "type": f"file_{op}",
                "params": dict(params),
            })

        return result if isinstance(result, dict) else {"ok": True, "result": result}

    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self._manifest.supports(Capability.APP_INTENTS_PERFORM):
            return await self._rpc.call(
                "intent.perform",
                {"node_id": node_id, "action": action, **(params or {})},
            )
        return await self._rpc.call(
            Methods.AX_PERFORM,
            {"node_id": node_id, "action": action, **(params or {})},
        )

    async def launch_app(
        self, app_id: str, urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"bundle_id": app_id}
        if urls:
            params["urls"] = list(urls)
        result = await self._rpc.call(Methods.APP_LAUNCH, params)
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    async def run_intent(self, intent_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._manifest.supports(Capability.APP_INTENTS_PERFORM):
            return {"ok": False, "error": "intents_not_supported"}
        return await self._rpc.call(
            "intent.perform", {"intent": intent_name, "params": params}
        )

    async def activate_app(
        self, pid: int, window_id: Optional[int] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"pid": pid}
        if window_id is not None:
            params["window_id"] = window_id
        return await self._rpc.call(Methods.APP_ACTIVATE, params)

    async def open_url(self, url: str) -> Dict[str, Any]:
        """Open a URL in the default browser (local OS dispatch)."""
        result = await self._rpc.call(Methods.OPEN_URL, {"url": url})
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    async def list_apps(self) -> Dict[str, Any]:
        """List available applications on the system."""
        return await self._rpc.call(Methods.APP_LIST, {})

    async def exec_shell(self, command: str) -> Dict[str, Any]:
        """Run a shell command locally — cua-driver exposes no shell tool."""
        timeout = float(os.environ.get("LEAPFLOW_SHELL_TIMEOUT", "60.0"))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "ok": False,
                    "error": f"timeout after {timeout}s",
                    "stdout": "",
                    "stderr": "",
                }
            return {
                "ok": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "exit_code": proc.returncode,
            }
        except Exception as exc:  # noqa: BLE001 - boundary: report, never crash the turn
            return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}

    async def set_clipboard(self, text: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.CLIPBOARD_SET, {"text": text})

    async def type_text(self, text: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.INPUT_TYPE_TEXT, {"text": text})

    async def send_shortcut(self, keys: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.INPUT_SHORTCUT, {"keys": keys})

    async def scroll(
        self,
        node_id: str,
        direction: str,
        amount: int = 3,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        # The driver's keystroke scroll path requires pid even without an
        # element target (observed behaviour, despite the schema marking it
        # optional) — it must know which process receives the keystrokes.
        params: Dict[str, Any] = {"direction": direction, "amount": amount}
        if node_id:
            params["node_id"] = node_id
        if pid is not None:
            params["pid"] = pid
        if window_id is not None:
            params["window_id"] = window_id
        return await self._rpc.call(Methods.AX_SCROLL, params)

    async def capture_screenshot(
        self, pid: Optional[int] = None, window_id: Optional[int] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if pid is not None and window_id is not None:
            params["pid"] = pid
            params["window_id"] = window_id
        return await self._rpc.call(Methods.SCREEN_CAPTURE_FRAME, params)

    async def undo(self, steps: int = 1) -> List[Dict[str, Any]]:
        """Undo the last N file operations from the stack."""
        results: List[Dict[str, Any]] = []
        for _ in range(min(steps, len(self._undo_stack))):
            record = self._undo_stack.pop()
            results.append(await self._reverse_op(record))
        return results

    async def undo_last(self) -> Dict[str, Any]:
        """Backward-compatible single-step undo."""
        results = await self.undo(1)
        return results[0] if results else {"ok": False, "error": "nothing_to_undo"}

    async def _backup_for_undo(self, path: str) -> str:
        """Copy file to temp before deletion for potential recovery."""
        filename = Path(path).name if path else "file"
        backup = str(
            Path(tempfile.gettempdir()) / f"leap_undo_{uuid.uuid4().hex[:8]}_{filename}"
        )
        try:
            await self._rpc.call(Methods.FILE_COPY, {"source": path, "destination": backup})
        except Exception as exc:
            logger.warning("undo_backup_failed path=%s error=%s", path, exc)
            backup = ""
        return backup

    async def _reverse_op(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Reverse a single tracked operation."""
        op_type = record.get("type", "")

        if op_type == "file_move":
            params = record.get("params", {})
            return await self._rpc.call(Methods.FILE_MOVE, {
                "source": params.get("destination", ""),
                "destination": params.get("source", ""),
            })

        if op_type == "file_copy":
            params = record.get("params", {})
            dest = params.get("destination", "")
            if dest:
                return await self._rpc.call(Methods.FILE_DELETE, {"path": dest})
            return {"ok": False, "error": "no_destination_to_undo"}

        if op_type == "file_delete":
            backup = record.get("backup", "")
            original = record.get("original", "")
            if backup and original:
                return await self._rpc.call(Methods.FILE_MOVE, {
                    "source": backup,
                    "destination": original,
                })
            return {"ok": False, "error": "no_backup_available"}

        return {"ok": False, "error": f"not_reversible:{op_type}"}


def _snapshot_from_payload(
    payload: Dict[str, Any], *, pid: int, window_id: int
) -> UISnapshot:
    """Parse a get_window_state payload into a flat UISnapshot.

    The driver already returns the filtered, actionable-only element list;
    records are taken verbatim. ``elements_complete`` and ``capture_coverage``
    are the driver's own statements about blind spots (e.g. browser page
    content is not observable in window scope) and must reach the caller.
    """
    raw_elements = payload.get("elements")
    records = [r for r in raw_elements if isinstance(r, dict)] if isinstance(
        raw_elements, list
    ) else []

    elements: List[UIElement] = []
    for record in records:
        index = record.get("element_index")
        if not isinstance(index, int):
            continue
        frame_raw = record.get("frame")
        frame = (
            {"x": float(frame_raw["x"]), "y": float(frame_raw["y"]),
             "w": float(frame_raw["w"]), "h": float(frame_raw["h"])}
            if isinstance(frame_raw, dict)
            and all(k in frame_raw for k in ("x", "y", "w", "h"))
            else None
        )
        value = record.get("value")
        parent = record.get("parent_index")
        selected = record.get("selected")
        elements.append(UIElement(
            element_index=index,
            role=str(record.get("role", "") or ""),
            label=str(record.get("label", "") or ""),
            value="" if value is None else str(value),
            element_token=str(record.get("element_token", "") or ""),
            enabled=bool(record.get("enabled", True)),
            selected=bool(selected) if isinstance(selected, bool) else None,
            depth=record.get("depth", 0) if isinstance(record.get("depth"), int) else 0,
            parent_index=parent if isinstance(parent, int) else None,
            frame=frame,
        ))

    coverage = payload.get("capture_coverage")
    total = payload.get("total_element_count", payload.get("element_count", len(elements)))
    return UISnapshot(
        pid=pid,
        window_id=window_id,
        snapshot_id=str(payload.get("snapshot_id", "") or ""),
        elements=tuple(elements),
        elements_complete=bool(payload.get("elements_complete", True)),
        total_element_count=total if isinstance(total, int) else len(elements),
        degraded=bool(payload.get("degraded", False)),
        degraded_reason=str(payload.get("degraded_reason", "") or ""),
        coverage=coverage if isinstance(coverage, dict) else {},
    )
