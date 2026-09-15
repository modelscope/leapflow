# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bridge between EventBus and MonitorManager event-triggered watches.

Subscribes to EventBus as a callback. When a SystemEvent arrives,
checks all registered EventTrigger instances for pattern matches
and activates matched triggers (marking corresponding watches as DUE).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, Set

from leapflow.domain.events import SystemEvent
from leapflow.scheduler.triggers.event import EventTrigger

logger = logging.getLogger(__name__)


class EventBridge:
    """Adapter that fans out EventBus notifications to registered EventTriggers.

    Usage:
        bridge = EventBridge()
        event_bus.subscribe(bridge.on_event)
        bridge.register(watch_id, trigger)

    When EventBus delivers a SystemEvent, the bridge checks each registered
    trigger's pattern against `event.event_type`. Matched triggers are marked
    as fired so the scheduler's next tick picks them up immediately.
    """

    def __init__(
        self,
        *,
        default_debounce_s: float = 1.0,
        scheduler_wake: Optional[Callable[[], None]] = None,
        mark_due: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._triggers: Dict[str, EventTrigger] = {}
        self._active_patterns: Set[str] = set()
        self._debounce_windows: Dict[str, float] = {}
        self._last_triggered: Dict[str, float] = {}
        self._debounced_count: Dict[str, int] = {}
        self._default_debounce_s = default_debounce_s
        self._scheduler_wake = scheduler_wake
        self._mark_due = mark_due

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self, watch_id: str, trigger: EventTrigger, *, debounce_s: float | None = None
    ) -> None:
        """Register an event trigger for a watch.

        Args:
            watch_id: Unique identifier of the watch owning this trigger.
            trigger: The EventTrigger instance to check on each event.
            debounce_s: Minimum interval in seconds between successive
                trigger activations for this watch.  When *None*, uses the
                instance-level ``default_debounce_s`` (default: 1.0).
        """
        effective_debounce = debounce_s if debounce_s is not None else self._default_debounce_s
        self._triggers[watch_id] = trigger
        self._debounce_windows[watch_id] = effective_debounce
        self._active_patterns.add(trigger.event_pattern)
        logger.debug(
            "EventBridge: registered watch=%s pattern=%s (total=%d)",
            watch_id[:8],
            trigger.event_pattern,
            len(self._triggers),
        )

    def unregister(self, watch_id: str) -> None:
        """Remove an event trigger for a watch.

        Args:
            watch_id: The watch whose trigger should be removed.
        """
        trigger = self._triggers.pop(watch_id, None)
        if trigger is not None:
            # Rebuild active_patterns from remaining triggers
            self._active_patterns = {
                t.event_pattern for t in self._triggers.values()
            }
            self._debounce_windows.pop(watch_id, None)
            self._last_triggered.pop(watch_id, None)
            self._debounced_count.pop(watch_id, None)
            logger.debug(
                "EventBridge: unregistered watch=%s (total=%d)",
                watch_id[:8],
                len(self._triggers),
            )

    # ------------------------------------------------------------------
    # EventBus subscriber callback
    # ------------------------------------------------------------------

    def on_event(self, event: SystemEvent) -> None:
        """EventBus subscriber callback: fan out event to registered triggers.

        This is a synchronous method matching the EventBus subscriber contract
        (`Callable[[SystemEvent], None]`).

        Short-circuits immediately when no triggers are registered.
        Applies per-watch debounce: if a trigger fires within its debounce
        window since the last activation, the event is suppressed.

        Args:
            event: The normalized SystemEvent from EventBus.
        """
        if not self._triggers:
            return

        now = time.monotonic()
        event_name = event.event_type
        for watch_id, trigger in self._triggers.items():
            if trigger.matches(event_name):
                debounce_s = self._debounce_windows.get(watch_id, 1.0)
                last = self._last_triggered.get(watch_id)
                if last is not None and (now - last) < debounce_s:
                    # Within debounce window — suppress
                    self._debounced_count[watch_id] = (
                        self._debounced_count.get(watch_id, 0) + 1
                    )
                    logger.debug(
                        "EventBridge: debounced watch=%s event=%s "
                        "(suppressed=%d)",
                        watch_id[:8],
                        event_name,
                        self._debounced_count[watch_id],
                    )
                    continue
                # First trigger or outside debounce window — fire
                trigger.notify(event_name)
                self._last_triggered[watch_id] = now
                logger.debug(
                    "EventBridge: trigger fired watch=%s event=%s pattern=%s",
                    watch_id[:8],
                    event_name,
                    trigger.event_pattern,
                )
                # Persist the due time so the scheduler SQL query finds this task.
                if self._mark_due is not None:
                    try:
                        self._mark_due(watch_id, trigger.next_due_at)
                    except Exception:  # noqa: BLE001
                        logger.debug("EventBridge: mark_due failed for %s", watch_id[:8], exc_info=True)
                if self._scheduler_wake is not None:
                    self._scheduler_wake()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def debounced_count(self, watch_id: str) -> int:
        """Return the number of events suppressed by debounce for a watch.

        Args:
            watch_id: The watch to query.

        Returns:
            Count of suppressed events, 0 if none or watch not found.
        """
        return self._debounced_count.get(watch_id, 0)

    @property
    def active_count(self) -> int:
        """Number of currently registered event triggers."""
        return len(self._triggers)
