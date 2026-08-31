"""Core event types shared across all layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# ── Event priority levels ──
# Higher value = higher urgency. Used by downstream consumers (queues,
# schedulers) to bias processing without changing event semantics.
PRIORITY_CRITICAL: int = 5   # User interaction: click, keyboard, shortcut, drag
PRIORITY_HIGH: int = 4       # Workflow boundaries: app focus, context change
PRIORITY_NORMAL: int = 3     # Default: clipboard, scroll, etc.
PRIORITY_LOW: int = 2        # Background: filesystem changes
PRIORITY_DEFERRED: int = 1   # System: unmapped / internal

PRE_NORMALIZED_EVENT_PREFIXES: tuple[str, ...] = ("gateway.", "daemon.", "hw.")
"""Event-type prefixes whose producers already emit normalized types.

Every normalizer passes these through unchanged, because downstream consumers --
watch triggers, the board's family grouping -- match on the original type. Anything
not listed here collapses to ``internal.unmapped``, which silently discards the
family a producer chose; kept in one place so adding a source is one edit rather
than a matching pair that can drift apart.
"""


@dataclass(frozen=True)
class SystemEvent:
    """Normalized system event — uniform across all platforms.

    Timebase convention:
    - ``timestamp``: wall-clock time (``time.time()`` epoch seconds). Suitable
      for persistence and display but **not** for ordering or causal inference
      because wall-clock can jump (NTP, suspend/resume, manual adjustment).
    - ``payload["_mono_ts"]``: monotonic origin time (``time.monotonic()``)
      injected by each observer at event creation. Used by
      :class:`~leapflow.platform.reorder_buffer.EventReorderBuffer` for
      arrival-order correction and temporal sorting.
    """

    event_type: str
    source: str
    payload: Dict[str, Any]
    timestamp: float  # Wall-clock (time.time()); for storage/display only, NOT for ordering.
    platform_hint: str = ""
    priority: int = PRIORITY_NORMAL


@dataclass(frozen=True)
class UIElement:
    """One actionable UI element row from a window snapshot.

    Mirrors the driver's get_window_state element record. ``element_index``
    and ``element_token`` are the action addressing handles; the token is
    preferred because the driver validates its staleness on every action.
    """

    element_index: int
    role: str
    label: str = ""
    value: str = ""
    element_token: str = ""
    enabled: bool = True
    selected: Optional[bool] = None
    depth: int = 0
    parent_index: Optional[int] = None
    frame: Optional[Dict[str, float]] = None

    @property
    def target(self) -> str:
        """Preferred action target: element_token, else the index."""
        return self.element_token or str(self.element_index)


@dataclass(frozen=True)
class UISnapshot:
    """Immutable snapshot of one window's actionable elements.

    A snapshot is scoped to (pid, window_id) and superseded by the next
    read of the same window; ``elements_complete`` and ``coverage`` carry
    the driver's own statements about what this snapshot cannot see
    (e.g. browser page content in window scope).
    """

    pid: int
    window_id: int
    snapshot_id: str = ""
    elements: Tuple[UIElement, ...] = ()
    elements_complete: bool = True
    total_element_count: int = 0
    degraded: bool = False
    degraded_reason: str = ""
    coverage: Dict[str, Any] = field(default_factory=dict)

    def find(self, element_index: int) -> Optional[UIElement]:
        """Look up an element by its index."""
        for element in self.elements:
            if element.element_index == element_index:
                return element
        return None


@runtime_checkable
class PerceptionPort(Protocol):
    """Unified perception interface — Engine depends only on this."""

    async def subscribe_fs(self, paths: List[str]) -> str: ...

    async def read_window_state(
        self, pid: int, window_id: int, query: str = ""
    ) -> UISnapshot: ...

    async def list_windows(self) -> Dict[str, Any]: ...

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

    async def launch_app(
        self, app_id: str, urls: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...

    async def activate_app(
        self, pid: int, window_id: Optional[int] = None
    ) -> Dict[str, Any]: ...

    async def run_intent(
        self, intent_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    async def exec_shell(self, command: str) -> Dict[str, Any]: ...

    async def set_clipboard(self, text: str) -> Dict[str, Any]: ...

    async def type_text(self, text: str) -> Dict[str, Any]: ...

    async def send_shortcut(self, keys: str) -> Dict[str, Any]: ...

    async def undo(self, steps: int = 1) -> List[Dict[str, Any]]: ...

    async def undo_last(self) -> Dict[str, Any]: ...
