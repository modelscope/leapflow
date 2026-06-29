"""In-process mock OSHost with event simulation for demos without Swift."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from leapflow.platform.protocol import EventHandler, EventTypes, HostRpc, Methods


class MockBridge(HostRpc):
    """Deterministic stub that mimics OSHost RPC and can simulate pushed events."""

    def __init__(self) -> None:
        self._vfs: Dict[str, Dict[str, Any]] = {}
        self._clipboard = ""
        self._clipboard_change_ts: float = time.time()
        self._file_events: List[Dict[str, Any]] = []
        self._event_handlers: List[EventHandler] = []
        self._seed_demo_fs()

    def _seed_demo_fs(self) -> None:
        downloads = str(Path.home() / "Downloads" / "_leapflow_demo")
        for name in ("readme.pdf", "invoice.pdf", "spec.pdf"):
            path = str(Path(downloads) / name)
            self._vfs[path] = {
                "path": path,
                "size": 1200,
                "mtime": time.time() - 60,
                "is_dir": False,
                "name": name,
            }

    def on_event(self, handler: EventHandler) -> None:
        """Register an event handler (mirrors BridgeClient interface)."""
        self._event_handlers.append(handler)

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        p = params or {}
        if method == Methods.PING:
            return {"pong": True, "mock": True}
        if method == Methods.SYSTEM_MANIFEST:
            return {
                "platform_id": "darwin_15",
                "os_version": "15.0.0",
                "capabilities": [
                    "fs.watch", "ax.tree_read", "ax.perform_action",
                    "clipboard.read", "clipboard.watch", "file.ops",
                    "app.launch", "app.activate", "shell.exec",
                ],
                "metadata": {"mock": "true"},
            }
        if method == Methods.SYSTEM_INFO:
            return {"platform": "darwin", "mock": True}
        if method == Methods.CLIPBOARD_GET:
            return {
                "text": self._clipboard,
                "change_count": 1,
                "change_ts": self._clipboard_change_ts,
            }
        if method == Methods.CLIPBOARD_LAST_CHANGE:
            return {"change_count": 1, "change_ts": self._clipboard_change_ts}
        if method == Methods.FILE_LIST:
            directory = str(p.get("path", "")).rstrip("/")
            entries: List[Dict[str, Any]] = []
            for path, meta in self._vfs.items():
                parent = str(Path(path).parent)
                if directory and parent == directory:
                    entries.append(meta)
            return {"entries": entries, "path": directory}
        if method == Methods.FILE_MOVE:
            src = str(p["src"])
            dst = str(p["dst"])
            item = self._vfs.pop(src, None)
            if item is None:
                return {"ok": False, "error": "not_found", "src": src}
            item["path"] = dst
            self._vfs[dst] = item
            ev = {"path": dst, "action": "move", "ts": time.time(), "src": src}
            self._file_events.append(ev)
            await self._emit_event(EventTypes.FS_CHANGE, {"path": dst, "flags": 0x00001000, "ts": time.time()})
            return {"ok": True, "dst": dst}
        if method == Methods.FILE_COPY:
            src = str(p["src"])
            dst = str(p["dst"])
            item = self._vfs.get(src)
            if item is None:
                return {"ok": False, "error": "not_found", "src": src}
            self._vfs[dst] = dict(item, path=dst)
            return {"ok": True, "dst": dst}
        if method == Methods.FILE_DELETE:
            path = str(p["path"])
            existed = self._vfs.pop(path, None) is not None
            if existed:
                await self._emit_event(EventTypes.FS_CHANGE, {"path": path, "flags": 0x00000200, "ts": time.time()})
            return {"ok": existed, "path": path}
        if method == Methods.APP_LAUNCH:
            return {"ok": True, "bundle_id": p.get("bundle_id"), "mock": True}
        if method == Methods.APP_ACTIVATE:
            return {"ok": True, "bundle_id": p.get("bundle_id"), "mock": True}
        if method == Methods.AX_TREE:
            return {
                "root": {"role": "mock_window", "title": "Mock UI", "children": []},
                "mock": True,
            }
        if method == Methods.AX_PERFORM:
            return {"ok": True, "mock": True, "action": p}
        if method == Methods.FS_SUBSCRIBE:
            return {"subscription_id": str(uuid.uuid4()), "mock": True}
        if method == Methods.RECORDING_START:
            return {"ok": True, "sequence_start": 0, "mock": True}
        if method == Methods.RECORDING_STOP:
            return {"ok": True, "event_count": 0, "mock": True}

        return {"ok": False, "error": "unsupported_method", "method": method}

    def set_clipboard(self, text: str) -> None:
        """Simulate a clipboard change (triggers event emission)."""
        self._clipboard = text
        self._clipboard_change_ts = time.time()
        asyncio.ensure_future(
            self._emit_event(
                EventTypes.CLIPBOARD_CHANGE,
                {"text": text, "change_count": 1, "change_ts": self._clipboard_change_ts},
            )
        )

    def simulate_fs_event(self, path: str, flags: int = 0x00000100) -> None:
        """Manually push an FSEvent for testing."""
        asyncio.ensure_future(
            self._emit_event(EventTypes.FS_CHANGE, {"path": path, "flags": flags, "ts": time.time()})
        )

    def simulate_app_focus(self, bundle_id: str, app_name: str = "") -> None:
        """Manually push an app focus change event for testing."""
        asyncio.ensure_future(
            self._emit_event(EventTypes.APP_FOCUS_CHANGE, {
                "bundle_id": bundle_id,
                "app_name": app_name or bundle_id,
                "pid": 0,
                "ts": time.time(),
            })
        )

    async def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        for handler in self._event_handlers:
            try:
                await handler(event_type, payload)
            except Exception:
                pass

    def file_events(self) -> List[Dict[str, Any]]:
        return list(self._file_events)
