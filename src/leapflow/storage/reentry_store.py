"""DuckDB-backed persistence for event-driven re-entry (S2, phase N1).

Enables "finalize + Orient-seeded re-entry": a task can finalize a turn while
registering a ``ReentryTrigger`` ("wake me when <time/event/signal>") together
with an ``OrientSnapshot`` (its accumulated orientation). When the trigger
fires, the orchestrator (later phases) seeds a fresh, bounded ``engine.run()``
from that snapshot so the task continues from its accumulated orientation
rather than from scratch -- infinite OODA via bounded frame串联, not an
unbounded live loop.

N1 is the pure storage layer: data models + a DuckDB store, with no engine or
gateway wiring. CAS/claim semantics are single-process (daemon-serialized),
matching ``RecoveryCheckpoint``'s in-memory store and ``scheduler/store``.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder

logger = logging.getLogger(__name__)

_TABLE = "reentry_triggers"


class ReentryKind(str, Enum):
    """What kind of signal wakes the task."""

    TIME = "time"       # due_at reached
    EVENT = "event"     # a matching gateway event arrived
    SIGNAL = "signal"   # an external/environment signal


class ReentryState(str, Enum):
    """Lifecycle of a re-entry trigger."""

    ARMED = "armed"          # waiting to fire (re-queryable)
    EXHAUSTED = "exhausted"  # budget spent, terminal
    CANCELLED = "cancelled"  # explicitly cancelled, terminal


@dataclass(frozen=True)
class OrientSnapshot:
    """Accumulated orientation carried across a re-entry (minimal set, N1)."""

    task_id: str
    ledger_state: Dict[str, Any] = field(default_factory=dict)   # ResearchLedger.to_state()
    task_contract: Dict[str, Any] = field(default_factory=dict)  # goal / workspace / constraints
    continuation_summary: str = ""                               # compact carry-over for the first turn
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["OrientSnapshot"]:
        if not data:
            return None
        return cls(
            task_id=str(data.get("task_id", "")),
            ledger_state=dict(data.get("ledger_state") or {}),
            task_contract=dict(data.get("task_contract") or {}),
            continuation_summary=str(data.get("continuation_summary", "")),
            created_at=float(data.get("created_at", 0.0) or 0.0),
        )


@dataclass
class ReentryTrigger:
    """A registered wake-up: when to re-enter + the orientation to seed with."""

    task_id: str
    kind: str = ReentryKind.TIME.value
    trigger_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    due_at: float = 0.0                                   # kind=time
    event_match: Dict[str, Any] = field(default_factory=dict)  # kind=event (trigger_policy-style match)
    orient: Optional[OrientSnapshot] = None
    state: str = ReentryState.ARMED.value
    max_reentries: int = 1                                # budget guardrail
    reentries_used: int = 0
    deadline: float = 0.0                                 # 0 = no deadline
    created_at: float = field(default_factory=time.time)
    fired_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_consumable(self) -> bool:
        return (
            self.state == ReentryState.ARMED.value
            and self.reentries_used < self.max_reentries
        )


class ReentryStore:
    """Persist / query / claim re-entry triggers (DuckDB)."""

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder = source
        self._ensured = False

    def _ensure_schema(self) -> None:
        if self._ensured:
            return
        self._holder.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                trigger_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                kind TEXT NOT NULL,
                due_at DOUBLE DEFAULT 0.0,
                event_match TEXT DEFAULT '{{}}',
                orient_json TEXT DEFAULT '{{}}',
                state TEXT DEFAULT 'armed',
                max_reentries INTEGER DEFAULT 1,
                reentries_used INTEGER DEFAULT 0,
                deadline DOUBLE DEFAULT 0.0,
                created_at DOUBLE NOT NULL,
                fired_at DOUBLE DEFAULT 0.0,
                metadata TEXT DEFAULT '{{}}'
            )
            """
        )
        self._ensured = True

    @property
    def _con(self) -> Any:
        self._ensure_schema()
        return self._holder.connection

    def save(self, trigger: ReentryTrigger) -> None:
        """Insert or replace a trigger (with its embedded orient snapshot)."""
        orient_json = json.dumps(
            trigger.orient.to_dict() if trigger.orient else {}, ensure_ascii=False
        )
        self._con.execute(
            f"INSERT OR REPLACE INTO {_TABLE} (trigger_id, task_id, session_id, kind, "
            f"due_at, event_match, orient_json, state, max_reentries, reentries_used, "
            f"deadline, created_at, fired_at, metadata) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                trigger.trigger_id, trigger.task_id, trigger.session_id, trigger.kind,
                trigger.due_at, json.dumps(trigger.event_match, ensure_ascii=False),
                orient_json, trigger.state, trigger.max_reentries, trigger.reentries_used,
                trigger.deadline, trigger.created_at, trigger.fired_at,
                json.dumps(trigger.metadata, ensure_ascii=False),
            ],
        )

    def load(self, trigger_id: str) -> Optional[ReentryTrigger]:
        if not trigger_id:
            return None
        row = self._con.execute(
            f"SELECT * FROM {_TABLE} WHERE trigger_id = ?", [trigger_id]
        ).fetchone()
        return self._row_to_trigger(row) if row else None

    def list_due(self, now: Optional[float] = None) -> List[ReentryTrigger]:
        """Armed TIME triggers whose due time has arrived (0 < due_at <= now)."""
        now = time.time() if now is None else now
        rows = self._con.execute(
            f"SELECT * FROM {_TABLE} WHERE kind = ? AND state = ? "
            f"AND due_at > 0 AND due_at <= ? ORDER BY due_at ASC",
            [ReentryKind.TIME.value, ReentryState.ARMED.value, now],
        ).fetchall()
        return [self._row_to_trigger(r) for r in rows]

    def list_armed_events(self) -> List[ReentryTrigger]:
        """Armed EVENT triggers (matched against inbound gateway events later)."""
        rows = self._con.execute(
            f"SELECT * FROM {_TABLE} WHERE kind = ? AND state = ?",
            [ReentryKind.EVENT.value, ReentryState.ARMED.value],
        ).fetchall()
        return [self._row_to_trigger(r) for r in rows]

    def fire(self, trigger_id: str, now: Optional[float] = None) -> Optional[ReentryTrigger]:
        """Atomically claim a trigger for one re-entry (single-process CAS).

        Returns the claimed trigger (with updated counts/state) or None when it
        is not consumable (missing, cancelled, exhausted, or past deadline).
        Budget: on the final allowed claim the trigger becomes EXHAUSTED; while
        budget remains it stays ARMED (a recurring driver advances due_at).
        """
        now = time.time() if now is None else now
        trig = self.load(trigger_id)
        if trig is None or not trig.is_consumable:
            return None
        if trig.deadline > 0 and now > trig.deadline:
            trig.state = ReentryState.EXHAUSTED.value
            self.save(trig)
            return None
        trig.reentries_used += 1
        trig.fired_at = now
        if trig.reentries_used >= trig.max_reentries:
            trig.state = ReentryState.EXHAUSTED.value
        self.save(trig)
        return trig

    def advance_due(self, trigger_id: str, new_due_at: float) -> None:
        """Advance an armed recurring trigger's next due time."""
        self._con.execute(
            f"UPDATE {_TABLE} SET due_at = ? WHERE trigger_id = ? AND state = ?",
            [new_due_at, trigger_id, ReentryState.ARMED.value],
        )

    def cancel(self, trigger_id: str) -> bool:
        """Cancel an armed trigger. Returns True if it was armed."""
        trig = self.load(trigger_id)
        if trig is None or trig.state != ReentryState.ARMED.value:
            return False
        trig.state = ReentryState.CANCELLED.value
        self.save(trig)
        return True

    def cleanup(self, now: Optional[float] = None) -> int:
        """Delete terminal (exhausted/cancelled) and deadline-passed triggers."""
        now = time.time() if now is None else now
        before = self._con.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()[0]
        self._con.execute(
            f"DELETE FROM {_TABLE} WHERE state IN (?, ?) "
            f"OR (deadline > 0 AND deadline < ?)",
            [ReentryState.EXHAUSTED.value, ReentryState.CANCELLED.value, now],
        )
        after = self._con.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()[0]
        return int(before - after)

    def close(self) -> None:
        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass

    @staticmethod
    def _row_to_trigger(row: tuple) -> ReentryTrigger:
        return ReentryTrigger(
            trigger_id=row[0],
            task_id=row[1],
            session_id=row[2],
            kind=row[3],
            due_at=row[4],
            event_match=_safe_json(row[5]),
            orient=OrientSnapshot.from_dict(_safe_json(row[6])),
            state=row[7],
            max_reentries=row[8],
            reentries_used=row[9],
            deadline=row[10],
            created_at=row[11],
            fired_at=row[12],
            metadata=_safe_json(row[13]),
        )


def _safe_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def build_reentry_trigger(
    *,
    task_id: str,
    session_id: str = "",
    ledger_state: Optional[Dict[str, Any]] = None,
    task_contract: Optional[Dict[str, Any]] = None,
    continuation_summary: str = "",
    kind: str = ReentryKind.TIME.value,
    delay_seconds: float = 0.0,
    event_match: Optional[Dict[str, Any]] = None,
    max_reentries: int = 1,
    deadline_seconds: float = 0.0,
    now: Optional[float] = None,
) -> ReentryTrigger:
    """Pure factory: assemble an OrientSnapshot + a ReentryTrigger.

    Kept side-effect-free (no store, no clock unless defaulted) so the core
    registration logic is unit-testable; the engine gathers live state and
    persists the result.
    """
    now = time.time() if now is None else now
    kind = kind if kind in (ReentryKind.TIME.value, ReentryKind.EVENT.value) else ReentryKind.TIME.value
    snapshot = OrientSnapshot(
        task_id=task_id,
        ledger_state=dict(ledger_state or {}),
        task_contract=dict(task_contract or {}),
        continuation_summary=(continuation_summary or "").strip()[:2000],
    )
    due_at = now + max(0.0, float(delay_seconds)) if kind == ReentryKind.TIME.value else 0.0
    deadline = now + float(deadline_seconds) if deadline_seconds and float(deadline_seconds) > 0 else 0.0
    return ReentryTrigger(
        task_id=task_id,
        session_id=session_id,
        kind=kind,
        due_at=due_at,
        event_match=dict(event_match or {}),
        orient=snapshot,
        max_reentries=max(1, int(max_reentries)),
        deadline=deadline,
    )
