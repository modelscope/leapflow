"""Mock adapter for testing without a native host process."""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from leapflow.domain.events import SystemEvent, UINode

_URL_OPEN_RE = re.compile(
    r"^\s*(?:open|xdg-open|start)\s+['\"]?https?://", re.IGNORECASE
)


def _is_url_open_command(command: str) -> bool:
    """Detect shell commands that open URLs in a browser."""
    return bool(_URL_OPEN_RE.match(command))


class MockPerceptionAdapter:
    """In-memory perception stub for tests and demos."""

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[SystemEvent] = asyncio.Queue(maxsize=128)

    async def subscribe_fs(self, paths: List[str]) -> str:
        return "mock-sub-001"

    async def read_ui_tree(self, app_id: Optional[str] = None) -> UINode:
        return UINode(
            node_id="mock-root",
            role="AXWindow",
            label="Mock Window",
            children=[
                UINode(node_id="mock-btn-save", role="AXButton", label="Save", actions=["AXPress"]),
                UINode(node_id="mock-btn-cancel", role="AXButton", label="Cancel", actions=["AXPress"]),
                UINode(
                    node_id="mock-scroll-area",
                    role="AXScrollArea",
                    label="",
                    children=[
                        UINode(node_id="mock-text-field", role="AXTextField", label="Input", value="hello"),
                    ],
                ),
            ],
            actions=["AXPress", "AXRaise"],
        )

    async def get_clipboard(self) -> Dict[str, Any]:
        return {"text": "", "change_count": 0, "change_ts": time.time()}

    async def capture_screenshot(self, region: str = "") -> Dict[str, Any]:
        return {"ok": True, "path": "/tmp/mock_screenshot.png", "region": region}

    async def stream_events(self) -> AsyncIterator[SystemEvent]:
        while True:
            event = await self._event_queue.get()
            yield event

    def inject_event(self, event: SystemEvent) -> None:
        """Push a synthetic event for testing."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


class MockExecutionAdapter:
    """In-memory execution stub that records actions for assertion."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    async def perform_file_op(self, op: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import shutil
        from pathlib import Path

        record = {"type": "file_op", "op": op, "params": params}
        self.history.append(record)
        try:
            if op == "list":
                p = Path(params.get("path", "."))
                entries = [{"name": e.name, "is_dir": e.is_dir()} for e in p.iterdir()]
                return {"ok": True, "entries": entries}
            elif op == "move":
                shutil.move(params["source"], params["destination"])
                return {"ok": True}
            elif op == "copy":
                src = Path(params["source"])
                dst = Path(params["destination"])
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                return {"ok": True}
            elif op == "delete":
                p = Path(params["path"])
                if p.is_dir():
                    shutil.rmtree(str(p))
                else:
                    p.unlink()
                return {"ok": True}
            else:
                return {"ok": False, "error": f"unknown op: {op}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = {"type": "ui_action", "node_id": node_id, "action": action}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def launch_app(self, app_id: str) -> Dict[str, Any]:
        record = {"type": "launch_app", "app_id": app_id}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def run_intent(self, intent_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        record = {"type": "intent", "intent": intent_name, "params": params}
        self.history.append(record)
        return {"ok": True, "mock": True, "stub": True}

    async def activate_app(self, app_id: str) -> Dict[str, Any]:
        record = {"type": "activate_app", "app_id": app_id}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def list_apps(self, filter: str = "", running_only: bool = False) -> Dict[str, Any]:
        """Return empty app list in mock mode."""
        record = {"type": "list_apps", "filter": filter, "running_only": running_only}
        self.history.append(record)
        return {"ok": True, "apps": []}

    async def exec_shell(self, command: str) -> Dict[str, Any]:
        record = {"type": "shell", "command": command}
        self.history.append(record)
        if _is_url_open_command(command):
            return {"ok": False, "error": "blocked: use open_url tool instead of shell for URLs", "stdout": "", "stderr": ""}
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return {
                "ok": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "exit_code": proc.returncode,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}

    async def open_url(self, url: str) -> Dict[str, Any]:
        record = {"type": "open_url", "url": url}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def set_clipboard(self, text: str) -> Dict[str, Any]:
        record = {"type": "set_clipboard", "text": text}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def type_text(self, text: str, method: str = "paste") -> Dict[str, Any]:
        record = {"type": "type_text", "text": text, "method": method}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def send_shortcut(self, keys: str) -> Dict[str, Any]:
        record = {"type": "send_shortcut", "keys": keys}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def scroll(self, node_id: str, delta_x: int, delta_y: int) -> Dict[str, Any]:
        record = {"type": "scroll", "node_id": node_id, "delta_x": delta_x, "delta_y": delta_y}
        self.history.append(record)
        return {"ok": True, "mock": True}

    async def undo_last(self) -> Dict[str, Any]:
        if self.history:
            undone = self.history.pop()
            return {"ok": True, "mock": True, "undone": undone}
        return {"ok": False, "error": "nothing_to_undo"}
