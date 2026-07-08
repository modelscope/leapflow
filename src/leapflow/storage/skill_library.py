"""Persistent skill library and update suggestion store.

Stores distilled skills durably so the active learning system can compare
new observations against the existing skill repertoire.  Follows the same
DuckDB patterns as ``imitation/store.py``.

Extended to support the parameterized Skill framework (P2.1).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import duckdb

from leapflow.domain.skill_types import (
    AnchorCandidate,
    DistillationCandidate,
    RecoveryEvent,
)
from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.write_buffer import execute_with_retry

if TYPE_CHECKING:
    from leapflow.skills.registry import Skill

logger = logging.getLogger(__name__)


# ── Data models ──


@dataclass
class StoredSkill:
    """A distilled skill persisted in the library."""

    skill_id: str = ""
    title: str = ""
    trigger_phrases: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    parameters: List[Dict[str, str]] = field(default_factory=list)
    pre_conditions: List[str] = field(default_factory=list)
    post_conditions: List[str] = field(default_factory=list)
    app_sequence: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)
    source_trajectory_id: str = ""
    source_episode_id: str = ""
    confidence: float = 0.0
    version: int = 1
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SkillExecution:
    """Records one feedback observation of a stored skill."""

    execution_id: str = ""
    skill_id: str = ""
    trajectory_id: str = ""
    episode_id: str = ""
    similarity_score: float = 0.0
    diff_hash: str = ""
    diff_summary: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    created_at: float = 0.0


@dataclass
class SkillUpdateSuggestion:
    """A pending suggestion to update an existing skill."""

    suggestion_id: str = ""
    existing_skill_id: str = ""
    existing_skill_title: str = ""
    new_candidate_json: str = ""
    similarity_score: float = 0.0
    similarity_details: Dict[str, Any] = field(default_factory=dict)
    suggestion_type: str = "update_existing"
    proposed_changes: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    source_trajectory_id: str = ""
    source_episode_id: str = ""
    created_at: float = 0.0
    resolved_at: Optional[float] = None


# ── Store ──


class SkillLibraryStore:
    """DuckDB-backed CRUD for the skill library and suggestion queue.

    Accepts ``ConnectionHolder`` (shared) or legacy ``Path``.
    """

    def __init__(
        self,
        source: Union[ConnectionHolder, Path],
        *,
        audit_logger: Optional[Any] = None,
    ) -> None:
        self._owns_holder = isinstance(source, Path)
        if self._owns_holder:
            source = LocalConnectionHolder(source)
        self._holder = source
        self._con = self._holder.connection
        self._audit_logger = audit_logger
        self._init_schema()

    def close(self) -> None:
        if self._owns_holder:
            self._holder.close()

    # ── Schema ──

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_skill_library (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                trigger_phrases TEXT,
                steps TEXT,
                parameters TEXT,
                pre_conditions TEXT,
                post_conditions TEXT,
                app_sequence TEXT,
                action_names TEXT,
                source_trajectory_id TEXT,
                source_episode_id TEXT,
                confidence DOUBLE,
                version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                created_at DOUBLE NOT NULL,
                updated_at DOUBLE NOT NULL
            )
        """)
        # Extended schema for parameterized skill framework (P2.1)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_parameterized_skills (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                parameters TEXT,
                preconditions TEXT,
                postconditions TEXT,
                triggers TEXT,
                source TEXT DEFAULT 'builtin',
                source_trajectory_id TEXT,
                source_episode_id TEXT,
                confidence DOUBLE DEFAULT 1.0,
                version INTEGER DEFAULT 1,
                code TEXT,
                created_at DOUBLE,
                updated_at DOUBLE,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_param_skill_active "
            "ON leap_parameterized_skills(is_active)"
        )
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_skill_suggestion (
                id TEXT PRIMARY KEY,
                existing_skill_id TEXT NOT NULL,
                existing_skill_title TEXT,
                new_candidate_json TEXT NOT NULL,
                similarity_score DOUBLE NOT NULL,
                similarity_details TEXT,
                suggestion_type TEXT NOT NULL,
                proposed_changes TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                source_trajectory_id TEXT,
                source_episode_id TEXT,
                created_at DOUBLE NOT NULL,
                resolved_at DOUBLE
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_skill_execution (
                id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL,
                trajectory_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                similarity_score DOUBLE NOT NULL,
                diff_hash TEXT NOT NULL,
                diff_summary TEXT,
                verdict TEXT NOT NULL,
                created_at DOUBLE NOT NULL
            )
        """)
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_skill "
            "ON leap_skill_execution(skill_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_hash "
            "ON leap_skill_execution(diff_hash)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_status "
            "ON leap_skill_library(status)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_status "
            "ON leap_skill_suggestion(status)"
        )

    # ── Skill CRUD ──

    def save_skill(self, skill: StoredSkill) -> None:
        now = time.time()
        execute_with_retry(
            self._con,
            "INSERT OR REPLACE INTO leap_skill_library VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                skill.skill_id,
                skill.title,
                _dumps(skill.trigger_phrases),
                _dumps(skill.steps),
                _dumps(skill.parameters),
                _dumps(skill.pre_conditions),
                _dumps(skill.post_conditions),
                _dumps(skill.app_sequence),
                _dumps(skill.action_names),
                skill.source_trajectory_id,
                skill.source_episode_id,
                skill.confidence,
                skill.version,
                skill.status,
                skill.created_at or now,
                now,
            ],
        )

    def load_skill(self, skill_id: str) -> Optional[StoredSkill]:
        rows = self._con.execute(
            "SELECT * FROM leap_skill_library WHERE id = ?",
            [skill_id],
        ).fetchall()
        return _row_to_skill(rows[0]) if rows else None

    def load_all_active(self) -> List[StoredSkill]:
        rows = self._con.execute(
            "SELECT * FROM leap_skill_library WHERE status = 'active' "
            "ORDER BY updated_at DESC",
        ).fetchall()
        return [_row_to_skill(r) for r in rows]

    def load_skill_by_title(self, title: str) -> Optional[StoredSkill]:
        """Find a stored skill by title (case-insensitive)."""
        rows = self._con.execute(
            "SELECT * FROM leap_skill_library WHERE title ILIKE ? LIMIT 1",
            [title],
        ).fetchall()
        return _row_to_skill(rows[0]) if rows else None

    def update_skill(self, skill: StoredSkill) -> None:
        """Persist an updated skill (caller sets version and fields)."""
        self.save_skill(skill)

    def save_from_candidate(
        self,
        candidate: DistillationCandidate,
        app_sequence: List[str],
        action_names: List[str],
    ) -> StoredSkill:
        now = time.time()
        skill = StoredSkill(
            skill_id=uuid.uuid4().hex[:16],
            title=candidate.title,
            trigger_phrases=list(candidate.trigger_phrases),
            steps=list(candidate.steps),
            parameters=list(candidate.parameters),
            pre_conditions=list(candidate.pre_conditions),
            post_conditions=list(candidate.post_conditions),
            app_sequence=app_sequence,
            action_names=action_names,
            source_trajectory_id=candidate.source_trajectory_id,
            source_episode_id=candidate.source_episode_id,
            confidence=candidate.confidence,
            version=1,
            status="active",
            created_at=now,
            updated_at=now,
        )
        self.save_skill(skill)
        logger.info("skill_library.new id=%s title=%s", skill.skill_id, skill.title)
        return skill

    # ── Suggestion CRUD ──

    def save_suggestion(self, s: SkillUpdateSuggestion) -> None:
        execute_with_retry(
            self._con,
            "INSERT OR REPLACE INTO leap_skill_suggestion VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                s.suggestion_id,
                s.existing_skill_id,
                s.existing_skill_title,
                s.new_candidate_json,
                s.similarity_score,
                _dumps(s.similarity_details),
                s.suggestion_type,
                _dumps(s.proposed_changes),
                s.status,
                s.source_trajectory_id,
                s.source_episode_id,
                s.created_at or time.time(),
                s.resolved_at,
            ],
        )

    def load_pending_suggestions(self, *, limit: int = 20) -> List[SkillUpdateSuggestion]:
        rows = self._con.execute(
            "SELECT * FROM leap_skill_suggestion "
            "WHERE status = 'pending' ORDER BY created_at DESC "
            f"LIMIT {int(limit)}",
        ).fetchall()
        return [_row_to_suggestion(r) for r in rows]

    def query_suggestions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent skill update suggestions as plain dicts for display.

        Includes resolved + pending entries (most recent first).  Each dict
        carries the candidate title (parsed from the stored JSON) so callers
        can render a compact pending-suggestions list without rehydrating a
        full :class:`DistillationCandidate`.
        """
        rows = self._con.execute(
            "SELECT * FROM leap_skill_suggestion "
            "ORDER BY created_at DESC "
            f"LIMIT {int(limit)}",
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            sug = _row_to_suggestion(r)
            candidate_title = ""
            if sug.new_candidate_json:
                try:
                    candidate_title = json.loads(sug.new_candidate_json).get(
                        "title", ""
                    )
                except (json.JSONDecodeError, AttributeError):
                    candidate_title = ""
            out.append({
                "suggestion_id": sug.suggestion_id,
                "candidate_title": candidate_title,
                "existing_skill_id": sug.existing_skill_id,
                "existing_skill_title": sug.existing_skill_title,
                "similarity_score": sug.similarity_score,
                "suggestion_type": sug.suggestion_type,
                "status": sug.status,
                "created_at": sug.created_at,
            })
        return out

    def count_pending(self) -> int:
        result = self._con.execute(
            "SELECT COUNT(*) FROM leap_skill_suggestion WHERE status = 'pending'"
        ).fetchone()
        return int(result[0]) if result else 0

    def resolve_suggestion(self, suggestion_id: str, status: str) -> None:
        execute_with_retry(
            self._con,
            "UPDATE leap_skill_suggestion SET status = ?, resolved_at = ? WHERE id = ?",
            [status, time.time(), suggestion_id],
        )

    # ── Execution log CRUD ──

    def save_execution(self, execution: SkillExecution) -> None:
        execute_with_retry(
            self._con,
            "INSERT OR REPLACE INTO leap_skill_execution VALUES "
            "(?,?,?,?,?,?,?,?,?)",
            [
                execution.execution_id,
                execution.skill_id,
                execution.trajectory_id,
                execution.episode_id,
                execution.similarity_score,
                execution.diff_hash,
                _dumps(execution.diff_summary),
                execution.verdict,
                execution.created_at or time.time(),
            ],
        )

    def load_executions(
        self, skill_id: str, *, limit: int = 20
    ) -> List[SkillExecution]:
        rows = self._con.execute(
            "SELECT * FROM leap_skill_execution "
            "WHERE skill_id = ? ORDER BY created_at DESC "
            f"LIMIT {int(limit)}",
            [skill_id],
        ).fetchall()
        return [_row_to_execution(r) for r in rows]

    def count_by_diff_hash(self, skill_id: str, diff_hash: str) -> int:
        result = self._con.execute(
            "SELECT COUNT(*) FROM leap_skill_execution "
            "WHERE skill_id = ? AND diff_hash = ?",
            [skill_id, diff_hash],
        ).fetchone()
        return int(result[0]) if result else 0

    def compute_skill_health(
        self, skill_id: str, *, recent_window: int = 10
    ) -> Dict[str, Any]:
        """Aggregate execution history into a health profile for the skill.

        Returns dict with success_rate, recent_regression_count,
        context_diversity, total_executions. Safe to call on unknown skill_id.
        """
        execs = self.load_executions(skill_id, limit=200)
        if not execs:
            return {
                "success_rate": 0.0,
                "recent_regression_count": 0,
                "context_diversity": 0,
                "total_executions": 0,
            }

        positive_verdicts = ("improved", "unchanged")
        successes = sum(1 for e in execs if e.verdict in positive_verdicts)
        recent = execs[:recent_window]
        regressions = sum(1 for e in recent if e.verdict == "regressed")
        contexts = {
            e.diff_summary.get("app_context", "")
            for e in execs
            if isinstance(e.diff_summary, dict) and e.diff_summary.get("app_context")
        }

        return {
            "success_rate": successes / len(execs),
            "recent_regression_count": regressions,
            "context_diversity": len(contexts),
            "total_executions": len(execs),
        }

    # ── Parameterized Skill CRUD (P2.1) ──

    def save_parameterized_skill(
        self, skill: Skill, code: Optional[str] = None
    ) -> None:
        """Persist a parameterized Skill to durable storage."""
        now = time.time()
        params_json = _dumps(
            [{"name": p.name, "type": p.type, "required": p.required,
              "default": p.default, "description": p.description}
             for p in skill.parameters]
        )
        execute_with_retry(
            self._con,
            "INSERT OR REPLACE INTO leap_parameterized_skills VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                skill.name,
                skill.description,
                params_json,
                _dumps(skill.preconditions),
                _dumps(skill.postconditions),
                _dumps(skill.triggers),
                skill.metadata.source,
                skill.metadata.source_trajectory_id,
                skill.metadata.source_episode_id,
                skill.metadata.confidence,
                skill.metadata.version,
                code,
                skill.metadata.created_at,
                now,
                True,
            ],
        )
        logger.debug("skill_library.save_parameterized name=%s v%d", skill.name, skill.metadata.version)

    def load_parameterized_skill(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a parameterized skill record by name. Returns raw dict or None."""
        rows = self._con.execute(
            "SELECT * FROM leap_parameterized_skills WHERE name = ?",
            [name],
        ).fetchall()
        if not rows:
            return None
        return _row_to_param_skill(rows[0])

    def load_all_active_parameterized(self) -> List[Dict[str, Any]]:
        """Load all active parameterized skills."""
        rows = self._con.execute(
            "SELECT * FROM leap_parameterized_skills WHERE is_active = TRUE "
            "ORDER BY updated_at DESC",
        ).fetchall()
        return [_row_to_param_skill(r) for r in rows]

    def deactivate_parameterized(self, name: str) -> bool:
        """Soft-delete a parameterized skill. Returns True if it existed."""
        execute_with_retry(
            self._con,
            "UPDATE leap_parameterized_skills SET is_active = FALSE, updated_at = ? "
            "WHERE name = ? AND is_active = TRUE",
            [time.time(), name],
        )
        # Check if row existed
        check = self._con.execute(
            "SELECT 1 FROM leap_parameterized_skills WHERE name = ?", [name]
        ).fetchone()
        return check is not None

    def update_skill_confidence(self, name: str, confidence: float) -> None:
        """Update confidence for a parameterized skill by name."""
        execute_with_retry(
            self._con,
            "UPDATE leap_parameterized_skills "
            "SET confidence = ?, updated_at = ? WHERE name = ?",
            [confidence, time.time(), name],
        )

    def update_parameterized_version(
        self, name: str, new_code: str
    ) -> int:
        """Bump version and update code. Returns new version number."""
        row = self._con.execute(
            "SELECT version FROM leap_parameterized_skills WHERE name = ?",
            [name],
        ).fetchone()
        if row is None:
            raise KeyError(f"Parameterized skill '{name}' not found")
        new_version = row[0] + 1
        execute_with_retry(
            self._con,
            "UPDATE leap_parameterized_skills "
            "SET version = ?, code = ?, updated_at = ? WHERE name = ?",
            [new_version, new_code, time.time(), name],
        )
        return new_version

    def search_parameterized_by_trigger(self, phrase: str) -> List[Dict[str, Any]]:
        """Search active parameterized skills by trigger phrase token overlap."""
        all_active = self.load_all_active_parameterized()
        phrase_tokens = _tokenize_phrase(phrase)
        if not phrase_tokens:
            return all_active

        scored: List[tuple[float, Dict[str, Any]]] = []
        for rec in all_active:
            triggers = rec.get("triggers", [])
            if not triggers:
                continue
            best = 0.0
            for t in triggers:
                t_tokens = _tokenize_phrase(t)
                if t_tokens:
                    overlap = len(phrase_tokens & t_tokens) / min(
                        len(phrase_tokens), len(t_tokens)
                    )
                    best = max(best, overlap)
            if best > 0.0:
                scored.append((best, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored]

    # ── Audit history query ──

    def query_history(
        self, skill_name: Optional[str] = None, limit: int = 20
    ) -> List[dict]:
        """Recent execution events for a skill (or all skills) from audit log.

        Pulls records prefixed with ``skill.execute`` from the configured
        :class:`AuditLogger` and optionally filters by ``skill`` field.  Returns
        records in descending time order, capped at ``limit``.
        """
        if self._audit_logger is None:
            return []
        # Fetch a generous window to allow post-filtering by skill name.
        raw_limit = max(limit * 5, limit) if skill_name else limit
        try:
            records = self._audit_logger.query(
                event_prefix="skill.execute", limit=raw_limit,
            )
        except Exception as e:
            logger.debug("skill_library.query_history_failed error=%s", e)
            return []

        if skill_name:
            records = [r for r in records if r.get("skill") == skill_name]

        # AuditLogger returns chronological order; reverse for newest-first.
        records = list(reversed(records))
        return records[:limit]


# ── Helpers ──


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _loads(val: Any) -> Any:
    if val is None:
        return []
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return []


def _row_to_skill(r: tuple) -> StoredSkill:
    return StoredSkill(
        skill_id=r[0],
        title=r[1],
        trigger_phrases=_loads(r[2]),
        steps=_loads(r[3]),
        parameters=_loads(r[4]),
        pre_conditions=_loads(r[5]),
        post_conditions=_loads(r[6]),
        app_sequence=_loads(r[7]),
        action_names=_loads(r[8]),
        source_trajectory_id=r[9] or "",
        source_episode_id=r[10] or "",
        confidence=r[11] or 0.0,
        version=r[12] or 1,
        status=r[13] or "active",
        created_at=r[14] or 0.0,
        updated_at=r[15] or 0.0,
    )


def _row_to_suggestion(r: tuple) -> SkillUpdateSuggestion:
    return SkillUpdateSuggestion(
        suggestion_id=r[0],
        existing_skill_id=r[1],
        existing_skill_title=r[2] or "",
        new_candidate_json=r[3] or "",
        similarity_score=r[4] or 0.0,
        similarity_details=_loads(r[5]) if r[5] else {},
        suggestion_type=r[6] or "",
        proposed_changes=_loads(r[7]) if r[7] else {},
        status=r[8] or "pending",
        source_trajectory_id=r[9] or "",
        source_episode_id=r[10] or "",
        created_at=r[11] or 0.0,
        resolved_at=r[12],
    )


def _row_to_execution(r: tuple) -> SkillExecution:
    return SkillExecution(
        execution_id=r[0],
        skill_id=r[1],
        trajectory_id=r[2],
        episode_id=r[3],
        similarity_score=r[4] or 0.0,
        diff_hash=r[5] or "",
        diff_summary=_loads(r[6]) if r[6] else {},
        verdict=r[7] or "",
        created_at=r[8] or 0.0,
    )


def serialize_candidate(c: DistillationCandidate) -> str:
    return json.dumps({
        "title": c.title,
        "trigger_phrases": c.trigger_phrases,
        "steps": c.steps,
        "parameters": c.parameters,
        "pre_conditions": c.pre_conditions,
        "post_conditions": c.post_conditions,
        "source_trajectory_id": c.source_trajectory_id,
        "source_episode_id": c.source_episode_id,
        "confidence": c.confidence,
        "recovery_events": [
            {"pattern": r.pattern, "trigger_action": r.trigger_action,
             "recovery_action": r.recovery_action, "confidence": r.confidence}
            for r in c.recovery_events
        ],
        "anchor_candidates": [
            {"step_index": a.step_index, "element_label": a.element_label,
             "element_role": a.element_role, "app_bundle_id": a.app_bundle_id}
            for a in c.anchor_candidates
        ],
        "procedure_graph": c.procedure_graph,
        "error_handling": c.error_handling,
    }, ensure_ascii=False)


def deserialize_candidate(raw: str) -> DistillationCandidate:
    d = json.loads(raw)
    recovery_events = [
        RecoveryEvent(**r) for r in d.get("recovery_events", [])
        if isinstance(r, dict)
    ]
    anchor_candidates = [
        AnchorCandidate(**a) for a in d.get("anchor_candidates", [])
        if isinstance(a, dict)
    ]
    return DistillationCandidate(
        title=d.get("title", ""),
        trigger_phrases=d.get("trigger_phrases", []),
        steps=d.get("steps", []),
        parameters=d.get("parameters", []),
        pre_conditions=d.get("pre_conditions", []),
        post_conditions=d.get("post_conditions", []),
        source_trajectory_id=d.get("source_trajectory_id", ""),
        source_episode_id=d.get("source_episode_id", ""),
        confidence=d.get("confidence", 0.0),
        recovery_events=recovery_events,
        anchor_candidates=anchor_candidates,
        procedure_graph=d.get("procedure_graph", ""),
        error_handling=d.get("error_handling", []),
    )


def _row_to_param_skill(r: tuple) -> Dict[str, Any]:
    """Convert a parameterized_skills row to a dict."""
    return {
        "name": r[0],
        "description": r[1],
        "parameters": _loads(r[2]),
        "preconditions": _loads(r[3]),
        "postconditions": _loads(r[4]),
        "triggers": _loads(r[5]),
        "source": r[6] or "builtin",
        "source_trajectory_id": r[7],
        "source_episode_id": r[8],
        "confidence": r[9] or 1.0,
        "version": r[10] or 1,
        "code": r[11],
        "created_at": r[12] or 0.0,
        "updated_at": r[13] or 0.0,
        "is_active": r[14] if r[14] is not None else True,
    }


def _tokenize_phrase(text: str) -> set:
    """Tokenize a phrase for trigger matching."""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return {t for t in tokens if len(t) >= 1}
