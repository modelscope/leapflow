# Copyright (c) Alibaba, Inc. and its affiliates.
"""Reversible effect tracking for plugin lifecycle management."""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


class ScopeState(enum.Enum):
    """State of an EffectScope instance."""
    ACTIVE = "active"
    DISPOSING = "disposing"
    DISPOSED = "disposed"


class EffectScope:
    """Hierarchical scope collecting cleanup callbacks, disposed in LIFO order.

    Usage:
        scope = EffectScope("my-plugin")
        scope.effect(lambda: registry.unregister("my-plugin"))
        scope.effect(lambda: event_bus.unsubscribe(callback))
        # ... later ...
        scope.dispose()  # runs all cleanups in reverse order

    EffectScope supports parent-child hierarchies: disposing a parent
    cascades to all children in reverse creation order.

    Design principle: "cold path tracking, hot path zero overhead" —
    EffectScope only operates during register/unregister (cold path);
    runtime dispatch is unaffected.
    """

    def __init__(self, name: str, *, parent: Optional["EffectScope"] = None) -> None:
        self.name = name
        self.parent = parent
        self.state = ScopeState.ACTIVE
        self._effects: list[Callable[[], None]] = []
        self._async_effects: List[Callable[[], Awaitable[None]]] = []
        self._children: list["EffectScope"] = []
        if parent is not None:
            parent._children.append(self)

    @property
    def is_active(self) -> bool:
        """Whether effects can still be registered on this scope."""
        return self.state == ScopeState.ACTIVE

    @property
    def is_disposed(self) -> bool:
        """Whether this scope has been fully disposed."""
        return self.state == ScopeState.DISPOSED

    def effect(self, cleanup: Callable[[], None]) -> None:
        """Register a sync cleanup callback. Raises if scope is not active."""
        if self.state != ScopeState.ACTIVE:
            raise RuntimeError(
                f"Cannot register effect on {self.state.value} scope '{self.name}'"
            )
        self._effects.append(cleanup)

    def async_effect(self, cleanup: Callable[[], Awaitable[None]]) -> None:
        """Register an async cleanup callback (for network/IO teardown).

        Async effects are awaited during async_dispose() and handled
        gracefully (with fallback) during sync dispose().
        Raises if scope is not active.
        """
        if self.state != ScopeState.ACTIVE:
            raise RuntimeError(
                f"Cannot register async effect on {self.state.value} scope '{self.name}'"
            )
        self._async_effects.append(cleanup)

    def child(self, name: str) -> "EffectScope":
        """Create a child scope. Disposing parent cascades to children."""
        if self.state != ScopeState.ACTIVE:
            raise RuntimeError(
                f"Cannot create child scope on {self.state.value} scope '{self.name}'"
            )
        return EffectScope(name, parent=self)

    def dispose(self) -> None:
        """Dispose this scope: children first (LIFO), then own effects (LIFO).

        Idempotent: calling dispose() on an already-disposed scope is a no-op.
        Exception-safe: a failing cleanup logs a warning but does not prevent
        remaining cleanups from executing.

        Async effects are handled gracefully:
        - If no event loop is running, each is executed via asyncio.run().
        - If a loop IS running, they are scheduled as fire-and-forget tasks
          with a warning (prefer async_dispose() in async contexts).
        """
        if self.state == ScopeState.DISPOSED:
            return  # idempotent
        self.state = ScopeState.DISPOSING
        # Children in reverse order
        for child_scope in reversed(self._children):
            child_scope.dispose()
        # Async effects (graceful degradation in sync context)
        if self._async_effects:
            self._dispose_async_effects_sync()
        # Own sync effects in reverse order (catch-and-continue)
        for cleanup in reversed(self._effects):
            try:
                cleanup()
            except Exception as exc:
                logger.warning(
                    "Effect cleanup failed in scope '%s': %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
        self._effects.clear()
        self._async_effects.clear()
        self._children.clear()
        self.state = ScopeState.DISPOSED

    def _dispose_async_effects_sync(self) -> None:
        """Best-effort execution of async effects from a sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for cleanup in reversed(self._async_effects):
            if loop is not None and loop.is_running():
                # Cannot await — schedule as fire-and-forget task
                logger.warning(
                    "Scope '%s': scheduling async effect as background task "
                    "(prefer async_dispose() in async contexts)",
                    self.name,
                )
                loop.create_task(self._safe_async_cleanup(cleanup))
            else:
                # No running loop — safe to use asyncio.run()
                try:
                    asyncio.run(cleanup())
                except Exception as exc:
                    logger.warning(
                        "Async effect cleanup failed in scope '%s': %s",
                        self.name,
                        exc,
                        exc_info=True,
                    )

    async def _safe_async_cleanup(self, cleanup: Callable[[], Awaitable[None]]) -> None:
        """Await a single async cleanup with exception suppression."""
        try:
            await cleanup()
        except Exception as exc:
            logger.warning(
                "Async effect cleanup failed in scope '%s': %s",
                self.name,
                exc,
                exc_info=True,
            )

    async def async_dispose(self) -> None:
        """Async-aware disposal — awaits async effects, calls sync effects.

        Idempotent. Exception-safe. Disposes children first (LIFO), then
        async effects (LIFO), then sync effects (LIFO).
        """
        if self.state == ScopeState.DISPOSED:
            return
        self.state = ScopeState.DISPOSING
        # Children in reverse order (async)
        for child_scope in reversed(self._children):
            await child_scope.async_dispose()
        # Async effects in reverse order
        for cleanup in reversed(self._async_effects):
            try:
                await cleanup()
            except Exception as exc:
                logger.warning(
                    "Async effect cleanup failed in scope '%s': %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
        # Sync effects in reverse order
        for cleanup in reversed(self._effects):
            try:
                cleanup()
            except Exception as exc:
                logger.warning(
                    "Effect cleanup failed in scope '%s': %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
        self._effects.clear()
        self._async_effects.clear()
        self._children.clear()
        self.state = ScopeState.DISPOSED

    @property
    def effect_count(self) -> int:
        """Number of registered sync effects (for diagnostics)."""
        return len(self._effects)

    @property
    def async_effect_count(self) -> int:
        """Number of registered async effects (for diagnostics)."""
        return len(self._async_effects)

    @property
    def child_count(self) -> int:
        """Number of child scopes (for diagnostics)."""
        return len(self._children)

    def __enter__(self) -> "EffectScope":
        return self

    def __exit__(self, *_: object) -> None:
        self.dispose()

    def __repr__(self) -> str:
        return (
            f"EffectScope(name={self.name!r}, state={self.state.value}, "
            f"effects={len(self._effects)}, children={len(self._children)})"
        )
