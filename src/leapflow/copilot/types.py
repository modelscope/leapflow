"""Core protocol and data-type definitions for the Workflow Copilot module.

Defines the shared vocabulary of immutable data objects and structural
protocols used across ContextEncoder, PredictionEngine, FeedbackLoop, etc.

SRP: Only declares types — no behaviour, no I/O.
OCP: Consumers depend on Protocols; new implementations require no changes here.
"""

from __future__ import annotations

import hashlib
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent  # noqa: F401


# ────────────────────────────────────────────────────────────────────────────
# Signal Protocol — unified abstraction for any signal source
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Signal(Protocol):
    """Unified signal abstraction — future sensors/devices/network all conform.

    Duck-type compatible with SystemEvent (event_type, source, payload, timestamp).
    """

    @property
    def event_type(self) -> str: ...

    @property
    def timestamp(self) -> float: ...

    @property
    def payload(self) -> Dict[str, Any]: ...

    @property
    def source(self) -> str: ...


# ────────────────────────────────────────────────────────────────────────────
# ContextState — incremental operational context snapshot
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ContextState:
    """Incremental operational context snapshot — only mutated fields are updated.

    The context hash provides O(1) hot-path lookup into historical predictions
    stored in DuckDB (SkillLibrary).  Use `delta_update` to change individual
    fields so that `dirty_fields` accurately tracks what changed since last
    consumer read.
    """

    app_bundle: str = ""
    window_title: str = ""
    action_ring: List[str] = field(default_factory=list)
    clipboard_hash: int = 0
    time_bucket: str = ""
    fs_context_hash: int = 0
    # metadata
    last_update_ts: float = 0.0
    dirty_fields: Set[str] = field(default_factory=set)

    @property
    def context_hash(self) -> str:
        """Deterministic MD5[:16] hash for O(1) hot-path cache/index lookup."""
        key = f"{self.app_bundle}|{'→'.join(self.action_ring[-3:])}|{self.time_bucket}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def delta_update(self, field_name: str, value: Any) -> None:
        """Update a single field incrementally, marking it dirty."""
        setattr(self, field_name, value)
        self.dirty_fields.add(field_name)
        self.last_update_ts = _time.time()

    def clear_dirty(self) -> None:
        """Clear dirty-field set after consumers have processed deltas."""
        self.dirty_fields.clear()


# ────────────────────────────────────────────────────────────────────────────
# PredictionCandidate — immutable prediction suggestion
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionCandidate:
    """A single prediction suggestion produced by any PredictorLayer.

    Immutable by design — once produced, a candidate is never mutated.
    The `confidence` field is in [0.0, 1.0].
    """

    action_description: str
    confidence: float
    source_layer: str  # "L0" | "L1" | "L2" | "L3"
    context_hash: str
    display_delay_ms: int
    is_destructive: bool = False
    skill_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    expire_ts: float = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Feedback types
# ────────────────────────────────────────────────────────────────────────────


class FeedbackType(Enum):
    """User feedback classification for a displayed prediction."""

    ACCEPT = "accept"
    IGNORE = "ignore"
    CORRECT = "correct"
    EXPLICIT_REJECT = "reject"


@dataclass(frozen=True)
class FeedbackSignal:
    """Structured feedback signal capturing how the user responded to a hint.

    Produced by FeedbackCollector, consumed by EvolutionLoop and each
    PredictorLayer's `on_feedback` method.
    """

    feedback_type: FeedbackType
    candidate: PredictionCandidate
    actual_action: Optional[str] = None
    response_latency_ms: int = 0
    context_at_feedback: Optional[ContextState] = None
    timestamp: float = 0.0


# ────────────────────────────────────────────────────────────────────────────
# SignalChannel Protocol — dynamically registerable signal source
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SignalChannel(Protocol):
    """Dynamically registerable signal channel.

    Future sensors, network streams, and device inputs all expose this
    interface to plug into the Copilot signal bus.
    """

    @property
    def channel_id(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def subscribe(self, handler: Callable[[Signal], None]) -> None: ...


# ────────────────────────────────────────────────────────────────────────────
# PredictorLayer Protocol — unified prediction layer interface
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class PredictorLayer(Protocol):
    """Unified prediction layer interface.

    Any new prediction algorithm registers with PredictionEngine by
    satisfying this Protocol.  Each layer owns its internal state and
    learning logic; the engine only schedules and aggregates.
    """

    @property
    def layer_id(self) -> str: ...

    @property
    def priority(self) -> int:
        """Lower value = higher priority (L0=0, L1=1, …)."""
        ...

    @property
    def timeout_ms(self) -> int: ...

    async def predict(self, context: ContextState) -> List[PredictionCandidate]: ...

    async def on_feedback(self, signal: FeedbackSignal) -> None: ...


# ────────────────────────────────────────────────────────────────────────────
# HintRenderer Protocol — suggestion display abstraction
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class HintRenderer(Protocol):
    """Ghost-hint rendering abstraction — TUI / GUI / Overlay implementations.

    The renderer is stateless w.r.t. prediction logic; it only knows how
    to show and dismiss visual hints.
    """

    async def show(self, candidate: PredictionCandidate) -> None: ...

    async def dismiss(self) -> None: ...

    @property
    def is_visible(self) -> bool: ...
