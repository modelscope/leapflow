"""Cross-platform event observers for passive signal collection.

Each observer implements the Observer Protocol and publishes events
through EventBus.handle_event(). Platform-specific implementations
are selected transparently based on sys.platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    """Lightweight passive signal observer.

    Contract:
    - start() is idempotent: calling on a running observer is a no-op.
    - stop() is idempotent: calling on a stopped observer is a no-op.
    - Exceptions inside observer loops MUST be caught internally.
    - Events are published via EventBus.handle_event().
    """

    async def start(self) -> None:
        """Begin observing. Idempotent."""
        ...

    async def stop(self) -> None:
        """Stop observing and release resources. Idempotent."""
        ...

    @property
    def running(self) -> bool:
        """Whether the observer is actively collecting signals."""
        ...


@dataclass
class ObserverConfig:
    """Configuration for the observation subsystem."""

    # Which observers to enable (key = observer name)
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        "fs_watcher": True,
        "app_focus": True,
        "clipboard": True,
        "input_tap": False,  # Requires accessibility permissions
    })

    # fs_watcher settings
    fs_watch_paths: List[str] = field(default_factory=list)
    fs_debounce_ms: int = 500

    # clipboard settings
    clipboard_poll_interval_s: float = 1.0

    # input_tap settings
    input_throttle_ms: int = 50


from leapflow.platform.observers.fs_watcher import FileSystemObserver  # noqa: E402
from leapflow.platform.observers.app_focus import AppFocusObserver  # noqa: E402
from leapflow.platform.observers.clipboard import ClipboardObserver  # noqa: E402
from leapflow.platform.observers.input_tap import InputTapObserver  # noqa: E402
from leapflow.platform.observers.daemon import ObservationDaemon  # noqa: E402

__all__ = [
    "Observer",
    "ObserverConfig",
    "FileSystemObserver",
    "AppFocusObserver",
    "ClipboardObserver",
    "InputTapObserver",
    "ObservationDaemon",
]
