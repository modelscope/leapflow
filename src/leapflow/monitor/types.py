"""Domain-neutral contract for the monitoring subsystem.

A ``Watch`` is a persistent, proactive monitor that periodically observes a
source and emits ``Finding`` objects. The contract is domain-agnostic: finance,
sentiment, research, and session analysis differ only by ``domain`` and the
shape of ``Finding.payload`` -- never by branching in core logic.

Watches are persisted as scheduler ``ArmedTask`` rows (``kind=watch``); this
module owns only the domain vocabulary and the serialization helpers that map a
``WatchSpec`` to/from an ``ArmedTask``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

# ── Event names broadcast on the daemon NotificationBus ──────────────────
EVENT_FINDING = "monitor.finding"
EVENT_WATCH_STATE = "watch.state"
EVENT_ERROR = "monitor.error"
EVENT_HEARTBEAT = "monitor.heartbeat"

# ── ArmedTask.metadata markers ───────────────────────────────────────────
WATCH_KIND = "watch"
METADATA_KIND_KEY = "kind"
METADATA_MUTED_KEY = "muted"
# Client-coupled watches (e.g. session analysis) only make sense while an
# interactive client is present; they must NOT keep the daemon alive on their own.
METADATA_CLIENT_COUPLED_KEY = "client_coupled"

_SEVERITY_RANK = {"info": 0, "notable": 1, "alert": 2}


class Severity(str, Enum):
    """Finding importance, driving disclosure and push gating."""

    INFO = "info"        # persist only (memory-level)
    NOTABLE = "notable"  # persist + passive push
    ALERT = "alert"      # persist + push + eligible for escalation

    @property
    def rank(self) -> int:
        """Return an ordinal for threshold comparisons."""
        return _SEVERITY_RANK[self.value]

    @classmethod
    def coerce(cls, value: Any, default: "Severity | None" = None) -> "Severity":
        """Return a Severity from a loose string, falling back to ``default``.

        ``default`` resolves to ``Severity.INFO`` when not supplied (it cannot be
        referenced as a class-body default before the enum members exist).
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return default if default is not None else cls.INFO


@dataclass(frozen=True)
class Evidence:
    """A single citation/link/snippet backing a finding."""

    kind: str = "text"  # text | link | quote | metric
    label: str = ""
    value: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "value": self.value, "url": self.url}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        return cls(
            kind=str(data.get("kind", "text")),
            label=str(data.get("label", "")),
            value=str(data.get("value", "")),
            url=str(data.get("url", "")),
        )


@dataclass(frozen=True)
class SuggestedAction:
    """A proposed next action a user can take on a finding.

    ``kind`` mirrors the dashboard action protocol so a suggested action can be
    dispatched directly by a view client (nav | rpc | intent | approval).
    """

    name: str
    label: str = ""
    kind: str = "intent"  # nav | rpc | intent | approval
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label or self.name,
            "kind": self.kind,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuggestedAction":
        return cls(
            name=str(data.get("name", "")),
            label=str(data.get("label", "")),
            kind=str(data.get("kind", "intent")),
            params=dict(data.get("params") or {}),
        )


@dataclass(frozen=True)
class Finding:
    """An immutable unit of observation produced by a watch.

    ``payload`` is a domain-private escape hatch (e.g. OHLC series for finance,
    author/abstract for papers). Core code never inspects it; only the matching
    dashboard renderer does.
    """

    watch_id: str
    domain: str
    title: str
    summary: str = ""
    severity: Severity = Severity.INFO
    score: float = 0.0
    ts: float = field(default_factory=time.time)
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    evidence: tuple[Evidence, ...] = ()
    tags: tuple[str, ...] = ()
    suggested_actions: tuple[SuggestedAction, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    dedup_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "watch_id": self.watch_id,
            "domain": self.domain,
            "title": self.title,
            "summary": self.summary,
            "severity": self.severity.value,
            "score": self.score,
            "ts": self.ts,
            "evidence": [item.to_dict() for item in self.evidence],
            "tags": list(self.tags),
            "suggested_actions": [action.to_dict() for action in self.suggested_actions],
            "payload": dict(self.payload),
            "dedup_key": self.dedup_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Finding":
        return cls(
            finding_id=str(data.get("finding_id") or uuid.uuid4().hex),
            watch_id=str(data.get("watch_id", "")),
            domain=str(data.get("domain", "")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            severity=Severity.coerce(data.get("severity")),
            score=float(data.get("score", 0.0) or 0.0),
            ts=float(data.get("ts", 0.0) or time.time()),
            evidence=tuple(Evidence.from_dict(item) for item in (data.get("evidence") or [])),
            tags=tuple(str(tag) for tag in (data.get("tags") or [])),
            suggested_actions=tuple(
                SuggestedAction.from_dict(item) for item in (data.get("suggested_actions") or [])
            ),
            payload=dict(data.get("payload") or {}),
            dedup_key=str(data.get("dedup_key", "")),
        )


@dataclass(frozen=True)
class WatchSpec:
    """Declarative definition of a monitor, persisted inside an ArmedTask.

    ``sensitivity`` is the minimum severity that will be pushed to view clients;
    lower-severity findings are still persisted (memory-level).
    """

    name: str
    domain: str
    trigger_expr: str = "10m"
    source: Mapping[str, Any] = field(default_factory=dict)
    lens: Mapping[str, Any] = field(default_factory=dict)
    sensitivity: str = "notable"  # info | notable | alert
    params: Mapping[str, Any] = field(default_factory=dict)
    max_runs: int = -1
    execution_tier: str = "local"
    watch_id: str = ""
    muted: bool = False
    client_coupled: bool = False

    def push_threshold(self) -> Severity:
        """Return the minimum severity that should be pushed to clients."""
        return Severity.coerce(self.sensitivity, default=Severity.NOTABLE)

    def to_task_parameters(self) -> dict[str, Any]:
        """Serialize spec content into ``ArmedTask.parameters``."""
        return {
            "watch_id": self.watch_id,
            "name": self.name,
            "domain": self.domain,
            "trigger_expr": self.trigger_expr,
            "source": dict(self.source),
            "lens": dict(self.lens),
            "sensitivity": self.sensitivity,
            "params": dict(self.params),
        }

    @classmethod
    def from_params(cls, parameters: Mapping[str, Any], *, muted: bool = False) -> "WatchSpec":
        """Reconstruct a WatchSpec from ``ArmedTask.parameters``."""
        params = parameters if isinstance(parameters, Mapping) else {}
        return cls(
            watch_id=str(params.get("watch_id", "")),
            name=str(params.get("name", "")),
            domain=str(params.get("domain", "")),
            trigger_expr=str(params.get("trigger_expr", "10m")),
            source=dict(params.get("source") or {}),
            lens=dict(params.get("lens") or {}),
            sensitivity=str(params.get("sensitivity", "notable")),
            params=dict(params.get("params") or {}),
            muted=bool(muted),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WatchSpec":
        """Build a spec from a client request dict (RPC/CLI/natural language)."""
        data = data if isinstance(data, Mapping) else {}
        return cls(
            name=str(data.get("name", "")).strip() or "watch",
            domain=str(data.get("domain", "")).strip(),
            trigger_expr=str(data.get("trigger_expr") or data.get("trigger") or "10m").strip(),
            source=dict(data.get("source") or {}),
            lens=dict(data.get("lens") or {}),
            sensitivity=str(data.get("sensitivity", "notable")).strip() or "notable",
            params=dict(data.get("params") or {}),
            max_runs=int(data.get("max_runs", -1) or -1),
            execution_tier=str(data.get("execution_tier", "local")).strip() or "local",
            watch_id=str(data.get("watch_id", "")),
            muted=bool(data.get("muted", False)),
            client_coupled=bool(data.get("client_coupled", False)),
        )


@dataclass(frozen=True)
class WatchView:
    """Runtime snapshot of a watch for listing and status reporting."""

    watch_id: str
    name: str
    domain: str
    trigger: str
    state: str
    muted: bool
    run_count: int
    next_due_at: float
    last_run_at: float
    finding_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "watch_id": self.watch_id,
            "name": self.name,
            "domain": self.domain,
            "trigger": self.trigger,
            "state": self.state,
            "muted": self.muted,
            "run_count": self.run_count,
            "next_due_at": self.next_due_at,
            "last_run_at": self.last_run_at,
            "finding_count": self.finding_count,
        }


@dataclass(frozen=True)
class ProducerContext:
    """Inputs handed to a producer for one observation cycle.

    ``services`` is an opaque, daemon-provided facade exposing capabilities a
    producer may need (skill execution, session history, LLM). It is optional so
    producers and tests can run without a live runtime.
    """

    spec: WatchSpec
    now: float
    run_count: int = 0
    last_run_at: float = 0.0
    services: Any = None
    force: bool = False


@runtime_checkable
class MonitorProducer(Protocol):
    """Per-domain observation logic: observe a source, emit findings.

    Producers own the domain-specific observe -> normalize -> score -> dedup
    steps; the ``Finding`` schema and the emit/persist path stay universal.
    """

    @property
    def domain(self) -> str:
        """Domain key this producer serves (matches ``WatchSpec.domain``)."""
        ...

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        """Return findings for this cycle (possibly empty)."""
        ...


__all__ = [
    "EVENT_FINDING",
    "EVENT_WATCH_STATE",
    "EVENT_ERROR",
    "EVENT_HEARTBEAT",
    "WATCH_KIND",
    "METADATA_KIND_KEY",
    "METADATA_MUTED_KEY",
    "METADATA_CLIENT_COUPLED_KEY",
    "Severity",
    "Evidence",
    "SuggestedAction",
    "Finding",
    "WatchSpec",
    "WatchView",
    "ProducerContext",
    "MonitorProducer",
]
