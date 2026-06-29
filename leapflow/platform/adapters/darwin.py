"""macOS adapter — maps VSI ports to HostRpc calls targeting the Swift OSHost."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

from leapflow.domain.events import SystemEvent, UINode
from leapflow.domain.platform import Capability, PlatformManifest
from leapflow.platform.protocol import HostRpc, Methods


class DarwinPerceptionAdapter:
    """PerceptionPort implementation backed by macOS OSHost RPC."""

    def __init__(self, rpc: HostRpc, manifest: PlatformManifest) -> None:
        self._rpc = rpc
        self._manifest = manifest
        self._event_queue: asyncio.Queue[SystemEvent] = asyncio.Queue(maxsize=512)

    async def subscribe_fs(self, paths: List[str]) -> str:
        result = await self._rpc.call(Methods.FS_SUBSCRIBE, {"path": paths[0] if paths else "~"})
        return str(result.get("subscription_id", ""))

    async def read_ui_tree(self, app_id: Optional[str] = None) -> UINode:
        params: Dict[str, Any] = {}
        if app_id:
            params["bundle_id"] = app_id

        if self._manifest.supports(Capability.APP_INTENTS_DISCOVER):
            params["prefer_intents"] = True

        result = await self._rpc.call(Methods.AX_TREE, params)
        return _parse_ui_tree(result.get("root", {}))

    async def get_clipboard(self) -> Dict[str, Any]:
        return await self._rpc.call(Methods.CLIPBOARD_GET, {})

    async def capture_screenshot(self, region: str = "", app_id: str = "") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if app_id:
            params["bundle_id"] = app_id
        elif region:
            params["region"] = region
        return await self._rpc.call(Methods.SCREEN_CAPTURE_FRAME, params)

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
    """ExecutionPort implementation backed by macOS OSHost RPC.

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

    async def launch_app(self, app_id: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.APP_LAUNCH, {"bundle_id": app_id})

    async def run_intent(self, intent_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._manifest.supports(Capability.APP_INTENTS_PERFORM):
            return {"ok": False, "error": "intents_not_supported"}
        return await self._rpc.call(
            "intent.perform", {"intent": intent_name, "params": params}
        )

    async def activate_app(self, app_id: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.APP_ACTIVATE, {"bundle_id": app_id})

    async def list_apps(self, filter: str = "", running_only: bool = False) -> Dict[str, Any]:
        """List available applications on the system."""
        return await self._rpc.call(
            Methods.APP_LIST, {"filter": filter, "running_only": running_only}
        )

    async def exec_shell(self, command: str) -> Dict[str, Any]:
        return await self._rpc.call(
            Methods.AX_PERFORM,
            {"commands": [{"type": "shell", "cmd": command}]},
        )

    async def set_clipboard(self, text: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.CLIPBOARD_SET, {"text": text})

    async def type_text(self, text: str, method: str = "paste") -> Dict[str, Any]:
        return await self._rpc.call(
            Methods.INPUT_TYPE_TEXT, {"text": text, "method": method}
        )

    async def send_shortcut(self, keys: str) -> Dict[str, Any]:
        return await self._rpc.call(Methods.INPUT_SHORTCUT, {"keys": keys})

    async def scroll(self, node_id: str, delta_x: int, delta_y: int) -> Dict[str, Any]:
        return await self._rpc.call(Methods.AX_SCROLL, {
            "node_id": node_id, "delta_x": delta_x, "delta_y": delta_y,
        })

    async def capture_screenshot(self, region: str = "", app_id: str = "") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if app_id:
            params["bundle_id"] = app_id
        elif region:
            params["region"] = region
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
        backup = f"/tmp/leap_undo_{uuid.uuid4().hex[:8]}_{filename}"
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


def _parse_ui_tree(raw: Dict[str, Any]) -> UINode:
    """Recursively parse a raw AX tree dict into UINode."""
    children_raw = raw.get("children", [])
    children = [_parse_ui_tree(c) for c in children_raw] if children_raw else []

    frame = raw.get("frame")
    frame_dict = (
        {"x": frame["x"], "y": frame["y"], "w": frame["w"], "h": frame["h"]}
        if isinstance(frame, dict)
        else None
    )

    ax_props_raw = raw.get("ax_props") or {}
    ax_props = ax_props_raw if isinstance(ax_props_raw, dict) else {}

    return UINode(
        node_id=str(raw.get("id", raw.get("role", ""))),
        role=str(raw.get("role", "")),
        label=str(raw.get("title", "")),
        value=str(raw.get("value", "")),
        children=children,
        actions=raw.get("actions", []),
        frame=frame_dict,
        ax_props=ax_props,
    )
