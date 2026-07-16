"""Domain-neutral monitoring subsystem: Watch -> Finding contract and runtime.

Public surface:
- Contract types (``WatchSpec``, ``Finding``, ``Severity``, ``MonitorProducer``)
- ``FindingStore`` for persistence
- ``ProducerRegistry`` for per-domain observation logic
- ``MonitorManager`` orchestrating watch lifecycle, persistence, and push
"""

from leapflow.monitor.finding_store import FindingStore
from leapflow.monitor.manager import EmitFn, MonitorManager
from leapflow.monitor.producers import ProducerRegistry
from leapflow.monitor.session_producer import (
    SessionAnalysisProducer,
    SessionAnalysisServices,
    ensure_session_watch,
    session_watch_params,
)
from leapflow.monitor.types import (
    EVENT_ERROR,
    EVENT_FINDING,
    EVENT_HEARTBEAT,
    EVENT_WATCH_STATE,
    WATCH_KIND,
    Evidence,
    Finding,
    MonitorProducer,
    ProducerContext,
    Severity,
    SuggestedAction,
    WatchSpec,
    WatchView,
)

__all__ = [
    "EVENT_FINDING",
    "EVENT_WATCH_STATE",
    "EVENT_ERROR",
    "EVENT_HEARTBEAT",
    "WATCH_KIND",
    "Severity",
    "Evidence",
    "SuggestedAction",
    "Finding",
    "WatchSpec",
    "WatchView",
    "ProducerContext",
    "MonitorProducer",
    "FindingStore",
    "ProducerRegistry",
    "MonitorManager",
    "EmitFn",
    "SessionAnalysisProducer",
    "SessionAnalysisServices",
    "ensure_session_watch",
    "session_watch_params",
]
