"""Core event types shared across all layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

# ── Event priority levels ──
# Higher value = higher urgency. Used by downstream consumers (queues,
# schedulers) to bias processing without changing event semantics.
PRIORITY_CRITICAL: int = 5   # User interaction: click, keyboard, shortcut, drag
PRIORITY_HIGH: int = 4       # Workflow boundaries: app focus, context change
PRIORITY_NORMAL: int = 3     # Default: clipboard, scroll, etc.
PRIORITY_LOW: int = 2        # Background: filesystem changes
PRIORITY_DEFERRED: int = 1   # System: unmapped / internal


@dataclass(frozen=True)
class SystemEvent:
    """Normalized system event — uniform across all platforms."""

    event_type: str
    source: str
    payload: Dict[str, Any]
    timestamp: float
    platform_hint: str = ""
    priority: int = PRIORITY_NORMAL


@dataclass
class UINode:
    """Normalized UI element tree node."""

    node_id: str
    role: str
    label: str
    value: str = ""
    children: List["UINode"] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    frame: Optional[Dict[str, float]] = None
    ax_props: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PerceptionPort(Protocol):
    """Unified perception interface — Engine depends only on this."""

    async def subscribe_fs(self, paths: List[str]) -> str: ...

    async def read_ui_tree(self, app_id: Optional[str] = None) -> UINode: ...

    async def get_clipboard(self) -> Dict[str, Any]: ...

    async def stream_events(self) -> AsyncIterator[SystemEvent]: ...


@runtime_checkable
class ExecutionPort(Protocol):
    """Unified execution interface."""

    async def perform_file_op(
        self, op: str, params: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...

    async def launch_app(self, app_id: str) -> Dict[str, Any]: ...

    async def activate_app(self, app_id: str) -> Dict[str, Any]: ...

    async def run_intent(
        self, intent_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    async def exec_shell(self, command: str) -> Dict[str, Any]: ...

    async def set_clipboard(self, text: str) -> Dict[str, Any]: ...

    async def type_text(self, text: str, method: str = "paste") -> Dict[str, Any]: ...

    async def send_shortcut(self, keys: str) -> Dict[str, Any]: ...

    async def undo(self, steps: int = 1) -> List[Dict[str, Any]]: ...

    async def undo_last(self) -> Dict[str, Any]: ...
