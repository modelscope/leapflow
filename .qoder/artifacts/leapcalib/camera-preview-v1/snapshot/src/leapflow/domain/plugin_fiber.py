# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plugin lifecycle state machine (PluginFiber).

Manages the runtime lifecycle of a single plugin instance through a
seven-state finite automaton with generation tracking:

    PENDING → DRAFT → ACTIVE → UNLOADING → DISPOSED
    PENDING → LOADING → ACTIVE
              LOADING → FAILED → LOADING (retry)
    PENDING → ACTIVE (fast path for trusted built-ins)
    PENDING/DRAFT/LOADING/FAILED → DISPOSED (early cleanup)

Each fiber owns an EffectScope; dispose() always cascades scope cleanup.
Generation counter provides identity across reload cycles.

States:
    PENDING   — created, awaiting activation or async init
    DRAFT     — isolated and testable, not published to the live registry
    LOADING   — async initialization in progress (dependency resolution)
    ACTIVE    — fully operational, tools available
    FAILED    — initialization failed, retryable via retry()/begin_loading()
    UNLOADING — graceful teardown in progress
    DISPOSED  — terminal, scope cleaned, no longer usable
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from leapflow.domain.effect_scope import EffectScope


# Module-level monotonic counter for PluginFiber generation IDs.
# Safe under LeapFlow's single-threaded asyncio model; if multi-threaded
# plugin lifecycle management is added later, this counter must be guarded
# by a threading.Lock or moved to a thread-safe primitive.
_generation_counter: int = 0


def _next_generation() -> int:
    """Return the next monotonic generation number."""
    global _generation_counter
    _generation_counter += 1
    return _generation_counter


class FiberState(enum.Enum):
    """Plugin fiber lifecycle states."""
    PENDING = "pending"
    DRAFT = "draft"
    LOADING = "loading"
    ACTIVE = "active"
    FAILED = "failed"
    UNLOADING = "unloading"
    DISPOSED = "disposed"


class IllegalStateTransition(RuntimeError):
    """Raised when a fiber state transition is invalid."""


_VALID_TRANSITIONS: dict[FiberState, set[FiberState]] = {
    FiberState.PENDING: {
        FiberState.DRAFT,
        FiberState.ACTIVE,
        FiberState.LOADING,
        FiberState.DISPOSED,
    },
    FiberState.DRAFT: {FiberState.ACTIVE, FiberState.DISPOSED},
    FiberState.LOADING: {FiberState.ACTIVE, FiberState.FAILED, FiberState.DISPOSED},
    FiberState.ACTIVE: {FiberState.UNLOADING},
    FiberState.FAILED: {FiberState.LOADING, FiberState.DISPOSED},
    FiberState.UNLOADING: {FiberState.DISPOSED},
    FiberState.DISPOSED: set(),
}


@dataclass
class PluginFiber:
    """Lifecycle state machine for a plugin instance.

    Usage:
        scope = EffectScope("my-plugin")
        fiber = PluginFiber(plugin_id="my-plugin", scope=scope)
        fiber.activate()   # PENDING → ACTIVE
        # ... plugin operates ...
        fiber.begin_unload()  # ACTIVE → UNLOADING
        fiber.dispose()    # UNLOADING → DISPOSED + scope.dispose()
    """

    plugin_id: str
    scope: EffectScope = field(repr=False)
    state: FiberState = FiberState.PENDING
    generation: int = field(default_factory=_next_generation)
    _error: Optional[Exception] = field(default=None, repr=False, init=False)

    @property
    def is_active(self) -> bool:
        """Whether the fiber is in ACTIVE state."""
        return self.state == FiberState.ACTIVE

    @property
    def is_disposed(self) -> bool:
        """Whether the fiber has been fully disposed."""
        return self.state == FiberState.DISPOSED

    @property
    def is_loading(self) -> bool:
        """Whether the fiber is in LOADING state."""
        return self.state == FiberState.LOADING

    @property
    def is_failed(self) -> bool:
        """Whether the fiber is in FAILED state."""
        return self.state == FiberState.FAILED

    @property
    def error(self) -> Optional[Exception]:
        """The stored error from a failed loading attempt, if any."""
        return self._error

    def mark_draft(self) -> None:
        """Transition a newly created fiber into isolated validation."""
        self._transition(FiberState.DRAFT)

    def activate(self) -> None:
        """Transition from PENDING, DRAFT, or LOADING to ACTIVE."""
        self._transition(FiberState.ACTIVE)

    def begin_loading(self) -> None:
        """Transition PENDING→LOADING or FAILED→LOADING. Clears stored error."""
        self._transition(FiberState.LOADING)
        self._error = None

    def fail(self, error: Exception) -> None:
        """Transition LOADING→FAILED and store the error reference."""
        self._transition(FiberState.FAILED)
        self._error = error

    def retry(self) -> None:
        """Convenience alias for begin_loading() from FAILED state."""
        self.begin_loading()

    def begin_unload(self) -> None:
        """Transition from ACTIVE to UNLOADING."""
        self._transition(FiberState.UNLOADING)

    def dispose(self) -> None:
        """Transition to DISPOSED and dispose the owned scope."""
        self._transition(FiberState.DISPOSED)
        self._error = None
        self.scope.dispose()

    def _transition(self, target: FiberState) -> None:
        if target not in _VALID_TRANSITIONS[self.state]:
            raise IllegalStateTransition(
                f"Cannot transition fiber '{self.plugin_id}' "
                f"from {self.state.value} to {target.value}"
            )
        self.state = target
