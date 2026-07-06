"""Event consumer protocol — defines the interface for consuming events from EventBus.

EventConsumers are registered with EventBus and receive batched events
for downstream processing (e.g., pattern mining, analytics).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent


@runtime_checkable
class EventConsumer(Protocol):
    """Protocol for event consumers that process batched events.

    Implementations should be idempotent and fault-tolerant —
    a consumer failure must not block other consumers or the EventBus.
    """

    @property
    def consumer_id(self) -> str:
        """Unique identifier for this consumer (used in logging and dedup)."""
        ...

    async def on_events_batch(self, events: list["SystemEvent"]) -> None:
        """Process a batch of events.

        Called by EventBus when the batch buffer is full or on flush interval.
        Must not raise — exceptions are caught and logged by the bus.
        """
        ...

    @property
    def enabled(self) -> bool:
        """Whether this consumer is currently active."""
        ...
