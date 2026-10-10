# Copyright (c) Alibaba, Inc. and its affiliates.
"""Benchmark adapter registry with built-in and entry-point discovery.

First-wins name arbitration: if two adapters claim the same ``adapter_id``
the incumbent keeps the slot and the challenger is recorded as a conflict
(never silently overwritten).  A bad entry-point import cannot crash the
registry or prevent other adapters from loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from benchmarks.protocol import BenchmarkAdapter

logger = logging.getLogger(__name__)

_EP_GROUP = "leapflow.benchmarks.adapters"


@dataclass(frozen=True)
class AdapterConflict:
    """A rejected duplicate adapter-id claim."""

    adapter_id: str
    kept_type: str
    rejected_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "kept_type": self.kept_type,
            "rejected_type": self.rejected_type,
        }


class AdapterRegistry:
    """Registry of ``BenchmarkAdapter`` instances, keyed by adapter_id.

    Discovery flow:
    1. ``register()`` for built-in adapters.
    2. ``discover_entry_points()`` for installed third-party adapters.
    3. ``get()`` / ``list_available()`` for consumer queries.
    """

    def __init__(self, builtins: Sequence[BenchmarkAdapter] = ()) -> None:
        self._adapters: dict[str, BenchmarkAdapter] = {}
        self._conflicts: list[AdapterConflict] = []
        self._ep_scanned: bool = False
        for adapter in builtins:
            self.register(adapter)

    # ── Registration ──────────────────────────────────────────

    def register(self, adapter: BenchmarkAdapter) -> bool:
        """Register an adapter.  Returns True on success, False on conflict.

        First-wins: the incumbent keeps the name, the challenger is
        rejected and recorded as a conflict.
        """
        aid = adapter.adapter_id
        if aid in self._adapters:
            self._conflicts.append(AdapterConflict(
                adapter_id=aid,
                kept_type=type(self._adapters[aid]).__name__,
                rejected_type=type(adapter).__name__,
            ))
            logger.warning(
                "adapter conflict on %r: kept %s, rejected %s",
                aid,
                type(self._adapters[aid]).__name__,
                type(adapter).__name__,
            )
            return False
        self._adapters[aid] = adapter
        logger.debug("registered benchmark adapter: %s", aid)
        return True

    # ── Entry-point discovery ─────────────────────────────────

    def discover_entry_points(self) -> int:
        """Discover adapters registered via setuptools entry_points.

        Entry points in the ``leapflow.benchmarks.adapters`` group should
        resolve to a callable that returns a ``BenchmarkAdapter`` instance.

        Import-safe: a broken entry point is logged and skipped.
        """
        if self._ep_scanned:
            return 0
        self._ep_scanned = True
        discovered = 0

        try:
            from importlib.metadata import entry_points
        except ImportError:
            logger.debug("importlib.metadata not available; skipping entry_points")
            return 0

        try:
            eps = entry_points(group=_EP_GROUP)
        except TypeError:
            eps = entry_points().get(_EP_GROUP, [])  # type: ignore[arg-type,union-attr]

        for ep in eps:
            try:
                obj = ep.load()
                adapter = obj() if callable(obj) and not isinstance(obj, BenchmarkAdapter) else obj
                if isinstance(adapter, BenchmarkAdapter):
                    if self.register(adapter):
                        discovered += 1
                else:
                    logger.warning(
                        "entry point %r does not satisfy BenchmarkAdapter", ep.name,
                    )
            except Exception:
                logger.warning(
                    "failed to load benchmark adapter entry point %r",
                    ep.name, exc_info=True,
                )

        logger.debug("entry-point discovery: %d adapter(s) loaded", discovered)
        return discovered

    # ── Queries ───────────────────────────────────────────────

    def get(self, adapter_id: str) -> BenchmarkAdapter | None:
        """Return the registered adapter, or None."""
        self._ensure_discovered()
        return self._adapters.get(adapter_id)

    def list_available(self) -> tuple[str, ...]:
        """Return sorted tuple of all registered adapter ids."""
        self._ensure_discovered()
        return tuple(sorted(self._adapters))

    def list_adapters(self) -> tuple[BenchmarkAdapter, ...]:
        """Return all adapters ordered by adapter_id."""
        self._ensure_discovered()
        return tuple(self._adapters[k] for k in sorted(self._adapters))

    @property
    def conflicts(self) -> list[AdapterConflict]:
        return list(self._conflicts)

    def _ensure_discovered(self) -> None:
        """Trigger entry-point discovery if not yet done."""
        if not self._ep_scanned:
            self.discover_entry_points()


def default_registry() -> AdapterRegistry:
    """Create a registry pre-populated with all built-in adapters.

    Import-safe: adapter modules are imported lazily inside
    ``benchmarks.adapters.builtin_adapters()`` and
    ``benchmarks.native.native_adapters()``.
    """
    from benchmarks.adapters import builtin_adapters
    builtins = list(builtin_adapters())
    try:
        from benchmarks.native import native_adapters
        builtins.extend(native_adapters())
    except Exception:
        logger.warning("failed to load native adapters", exc_info=True)
    return AdapterRegistry(builtins=builtins)


__all__ = [
    "AdapterConflict",
    "AdapterRegistry",
    "default_registry",
]
