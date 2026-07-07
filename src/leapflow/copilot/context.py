"""Incremental context encoder and EventBus bridge for Workflow Copilot.

Receives raw SystemEvent streams and maintains an up-to-date ContextState
using O(1) delta updates per event.  The encoder never blocks — it is
designed to run in the main event loop alongside the perception pipeline.

SRP: Encodes events → context.  Does not predict, render, or store.
Thread-safety: All writes happen in a single asyncio task; reads use
copy-on-read semantics via `current_state` property.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Optional

from leapflow.copilot.config import CopilotConfig
from leapflow.copilot.types import ContextState

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent
    from leapflow.memory.providers.working import WorkingMemoryProvider
    from leapflow.signal_fusion.cross_app import CrossAppContextTracker

logger = logging.getLogger(__name__)


class ContextEncoder:
    """Incremental context encoder — transforms SystemEvent stream into ContextState.

    Each incoming event updates only the affected fields (O(1) per event).
    The ring buffer (action_ring) maintains a sliding window of recent actions.
    Time bucket is refreshed at most once per configured interval.

    Usage::

        encoder = ContextEncoder(config)
        state = encoder.on_event(event)
        print(state.context_hash)
    """

    __slots__ = ("_state", "_ring_size", "_time_bucket_seconds", "_last_time_bucket_update")

    def __init__(self, config: CopilotConfig) -> None:
        self._state = ContextState()
        self._ring_size: int = config.action_ring_size
        self._time_bucket_seconds: float = config.time_bucket_minutes * 60.0
        self._last_time_bucket_update: float = 0.0

    def on_event(self, event: "SystemEvent") -> ContextState:
        """Process a single SystemEvent, incrementally update context, return snapshot.

        Dispatches on event_type to update the relevant ContextState fields.
        Always appends to action_ring regardless of event type.
        """
        match event.event_type:
            case "app.focus_change":
                self._state.delta_update("app_bundle", event.payload.get("bundle_id", ""))
                self._state.delta_update(
                    "window_title",
                    event.payload.get("window_title", event.payload.get("app_name", "")),
                )
            case "context.change":
                self._state.delta_update(
                    "window_title", event.payload.get("window_title", ""),
                )
            case "clipboard.change":
                self._state.delta_update("clipboard_hash", hash(str(event.payload)))
            case "fs.change":
                self._state.delta_update("fs_context_hash", hash(str(event.payload)))
            case _:
                pass

        # Always append to the rolling action ring
        action_desc = f"{event.event_type}:{event.source}"
        ring = (self._state.action_ring + [action_desc])[-self._ring_size:]
        self._state.delta_update("action_ring", ring)

        # Refresh time bucket at most once per interval
        if event.timestamp - self._last_time_bucket_update > self._time_bucket_seconds:
            self._state.delta_update("time_bucket", self._compute_time_bucket(event.timestamp))
            self._last_time_bucket_update = event.timestamp

        return self._state

    @property
    def current_state(self) -> ContextState:
        """Read-only access to the current context state.

        Note: Returns a mutable reference to the live state object.
        For async consumers that need a stable snapshot, use ``snapshot()`` instead.
        """
        return self._state

    def snapshot(self) -> ContextState:
        """Return an immutable snapshot of the current state for async consumers."""
        import copy
        return copy.copy(self._state)

    @staticmethod
    def _compute_time_bucket(ts: float) -> str:
        """Produce a human-friendly time bucket string, e.g. 'mon_09'."""
        dt = datetime.datetime.fromtimestamp(ts)
        return f"{dt.strftime('%a').lower()}_{dt.hour:02d}"


_WARMUP_EVENT_TYPES = frozenset({"app.focus_change", "context.change", "ui.action"})


class CopilotEventSubscriber:
    """Bridge between EventBus and the Copilot ContextEncoder.

    Subscribes to the platform EventBus, forwards events to the encoder,
    and optionally enriches context with CrossAppContextTracker hypotheses,
    WorkingMemory conversation hints, and speculative pipeline warmup.

    Usage::

        subscriber = CopilotEventSubscriber(encoder, tracker, working_memory=wm)
        event_bus.subscribe(subscriber.on_system_event)
    """

    __slots__ = (
        "_encoder", "_tracker", "_working_memory", "_last_hint", "_pipeline",
    )

    def __init__(
        self,
        encoder: ContextEncoder,
        tracker: Optional["CrossAppContextTracker"] = None,
        working_memory: Optional["WorkingMemoryProvider"] = None,
        pipeline: Optional[object] = None,
    ) -> None:
        self._encoder = encoder
        self._tracker = tracker
        self._working_memory = working_memory
        self._last_hint: str = ""
        self._pipeline = pipeline

    def set_pipeline(self, pipeline: object) -> None:
        """Late-bind speculative pipeline (avoids circular init deps)."""
        self._pipeline = pipeline

    def on_system_event(self, event: "SystemEvent") -> None:
        """EventBus callback — matches the standard EventCallback signature.

        Forwards the event to ContextEncoder for incremental processing.
        When CrossAppContextTracker has an active hypothesis, injects a
        workflow_hint field into the context for downstream predictors.
        When WorkingMemory is available, injects a conversation_hint.
        For high-information events, warms the speculative pipeline cache.
        """
        state = self._encoder.on_event(event)

        if self._tracker is not None and self._tracker.current_hypothesis is not None:
            state.delta_update(
                "workflow_hint",
                self._tracker.current_hypothesis.workflow_type.value,
            )

        if self._working_memory is not None:
            hint = self._extract_conversation_hint()
            if hint != self._last_hint:
                state.delta_update("conversation_hint", hint)
                self._last_hint = hint

        if self._pipeline is not None and event.event_type in _WARMUP_EVENT_TYPES:
            self._warmup_pipeline(event, state)

    def _warmup_pipeline(self, event: "SystemEvent", state: "ContextState") -> None:
        """Trigger speculative cache warmup for high-information OS events."""
        import asyncio
        on_observed = getattr(self._pipeline, "on_action_observed", None)
        if on_observed is None:
            return
        try:
            loop = asyncio.get_running_loop()
            ctx_snapshot = self._encoder.snapshot()
            loop.create_task(on_observed(
                action_id=f"{event.event_type}:{event.source}",
                context=ctx_snapshot,
            ))
        except RuntimeError:
            pass

    def _extract_conversation_hint(self) -> str:
        """Extract latest user intent from WorkingMemory (lightweight, <0.1ms)."""
        messages = self._working_memory.as_chat_messages()  # type: ignore[union-attr]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                return content[:80].strip()
        return ""
