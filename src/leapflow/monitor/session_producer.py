"""Session-analysis producer: model the current conversation as a Watch.

``SessionAnalysisProducer`` reuses the generic Watch -> Finding machinery: on each
cycle it reads the conversation transcript (through an injected services facade),
decides whether a refresh is warranted (batch threshold, model salience, or a
forced manual refresh), and emits one insight ``Finding`` carrying a structured
analysis payload (story / insights / decisions / action items / entities / ...).

The producer holds only in-memory checkpoints keyed by watch id; on daemon
restart it re-analyzes once, which is acceptable and self-correcting.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence, runtime_checkable

from leapflow.monitor.types import Finding, ProducerContext, Severity

logger = logging.getLogger(__name__)

DOMAIN = "session"

_DEFAULT_BATCH_TURNS = 6
_DEFAULT_BATCH_TOKENS = 4000
_DEFAULT_DEBOUNCE_S = 15.0
_DEFAULT_MAX_PER_MIN = 4


@runtime_checkable
class SessionAnalysisServices(Protocol):
    """Daemon-provided capabilities the session producer needs.

    Kept as a protocol so the producer is testable with a fake and never imports
    engine/LLM internals directly.
    """

    async def session_history(self) -> dict[str, Any]:
        """Return {session_id, turn_count, token_count, messages:[{role,content}]}."""
        ...

    async def analyze_session(
        self,
        messages: list[dict[str, Any]],
        *,
        prior: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a structured analysis payload for the transcript and artifacts."""
        ...

    async def should_refresh(self, messages: list[dict[str, Any]]) -> bool:
        """Return True when the model judges a refresh worthwhile (salience)."""
        ...


def _first_line(text: str, limit: int = 180) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1] + "\u2026"


def _artifact_fingerprint(artifacts: list[dict[str, Any]]) -> str:
    """Return a stable fingerprint for artifact state changes."""
    parts: list[str] = []
    for artifact in artifacts:
        path = str(artifact.get("path", ""))
        mtime = str(artifact.get("mtime", ""))
        size = str(artifact.get("size", ""))
        status = str(artifact.get("status", ""))
        parts.append(f"{path}:{mtime}:{size}:{status}")
    return "|".join(sorted(parts))


def _refresh_reason(
    *,
    force: bool,
    first: bool,
    artifact_changed: bool,
    turn_delta: int,
    token_delta: int,
    batch_turns: int,
    batch_tokens: int,
) -> str:
    if force:
        return "manual_refresh"
    if first:
        return "first_observation"
    if artifact_changed:
        return "artifact_changed"
    if turn_delta >= batch_turns:
        return "batch_turns"
    if token_delta >= batch_tokens:
        return "batch_tokens"
    return "model_salience"


def _observation_status(
    *,
    artifacts: list[dict[str, Any]],
    reason: str,
    now: float,
    turn_count: int,
    token_count: int,
    batch_turns: int,
    batch_tokens: int,
    last_turn: int,
) -> dict[str, Any]:
    included = [a for a in artifacts if a.get("status") == "included"]
    skipped = [a for a in artifacts if a.get("status") == "skipped"]
    observed_targets = ["conversation transcript", "tool results"]
    if artifacts:
        observed_targets.append("file artifacts")
    missing_items: list[str] = []
    if not artifacts:
        missing_items.append("No file-write artifacts detected in the current session yet.")
    for artifact in skipped[:5]:
        label = artifact.get("path") or artifact.get("name") or "artifact"
        missing_items.append(f"{label}: {artifact.get('reason', 'not included')}")
    coverage = 0.7
    if included:
        coverage += 0.2
    if artifacts and not skipped:
        coverage += 0.1
    if skipped:
        coverage -= min(0.3, len(skipped) * 0.08)
    coverage = max(0.1, min(1.0, coverage))
    return {
        "state": "watching",
        "refresh_reason": reason,
        "last_refresh_at": now,
        "observed_targets": observed_targets,
        "context_coverage_pct": round(coverage * 100),
        "missing_items": missing_items,
        "artifact_count": len(artifacts),
        "artifacts_included": len(included),
        "artifacts_skipped": len(skipped),
        "next_threshold": {
            "turns": batch_turns,
            "tokens": batch_tokens,
            "turns_since_last": max(0, turn_count - max(last_turn, 0)),
            "current_turns": turn_count,
            "current_tokens": token_count,
        },
    }


class SessionAnalysisProducer:
    """Produce session-analysis findings from the live conversation transcript."""

    def __init__(self) -> None:
        self._last_turn: dict[str, int] = {}
        self._last_tokens: dict[str, int] = {}
        self._last_artifact_fingerprint: dict[str, str] = {}
        self._last_run_at: dict[str, float] = {}
        self._recent_runs: dict[str, list[float]] = {}

    @property
    def domain(self) -> str:
        return DOMAIN

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        services = ctx.services
        if services is None:
            return []
        watch_id = ctx.spec.watch_id
        params = ctx.spec.params or {}
        batch_turns = int(params.get("batch_turns", _DEFAULT_BATCH_TURNS))
        batch_tokens = int(params.get("batch_tokens", _DEFAULT_BATCH_TOKENS))
        use_salience = bool(params.get("use_model_salience", False))
        debounce_s = float(params.get("debounce_s", _DEFAULT_DEBOUNCE_S))
        max_per_min = int(params.get("max_refresh_per_min", _DEFAULT_MAX_PER_MIN))

        try:
            history = await services.session_history()
        except Exception as exc:  # noqa: BLE001 - degrade if history unavailable
            logger.debug("session producer: history unavailable: %s", exc)
            return []
        turn_count = int(history.get("turn_count", 0) or 0)
        token_count = int(history.get("token_count", 0) or 0)
        session_id = str(history.get("session_id", "") or "")
        messages = list(history.get("messages") or [])
        artifacts = list(history.get("artifacts") or [])
        artifact_fingerprint = _artifact_fingerprint(artifacts)

        last_turn = self._last_turn.get(watch_id, -1)
        first = last_turn < 0
        artifact_changed = bool(artifact_fingerprint and artifact_fingerprint != self._last_artifact_fingerprint.get(watch_id, ""))

        if not ctx.force and not first and turn_count <= last_turn and not artifact_changed:
            return []  # no new turns or artifacts since last analysis

        if not ctx.force and not first:
            if (ctx.now - self._last_run_at.get(watch_id, 0.0)) < debounce_s:
                return []
            recent = [t for t in self._recent_runs.get(watch_id, []) if ctx.now - t < 60.0]
            if len(recent) >= max_per_min:
                return []

        should = (
            ctx.force
            or first
            or artifact_changed
            or (turn_count - last_turn >= batch_turns)
            or (token_count - self._last_tokens.get(watch_id, 0) >= batch_tokens)
        )
        if not should and use_salience and turn_count > last_turn:
            try:
                should = bool(await services.should_refresh(messages))
            except Exception:  # noqa: BLE001 - salience is best-effort
                should = False
        if not should:
            return []

        reason = _refresh_reason(
            force=ctx.force,
            first=first,
            artifact_changed=artifact_changed,
            turn_delta=turn_count - last_turn,
            token_delta=token_count - self._last_tokens.get(watch_id, 0),
            batch_turns=batch_turns,
            batch_tokens=batch_tokens,
        )
        try:
            analysis = dict(await services.analyze_session(messages, artifacts=artifacts))
        except TypeError:
            analysis = dict(await services.analyze_session(messages))
        except Exception as exc:  # noqa: BLE001 - surface as no-op, not crash
            logger.debug("session producer: analysis failed: %s", exc)
            return []

        self._last_turn[watch_id] = turn_count
        self._last_tokens[watch_id] = token_count
        self._last_artifact_fingerprint[watch_id] = artifact_fingerprint
        self._last_run_at[watch_id] = ctx.now
        self._recent_runs.setdefault(watch_id, []).append(ctx.now)

        analysis.setdefault("usage", {})
        analysis["usage"].update({"turns": turn_count, "tokens": token_count})
        analysis["artifact_context"] = artifacts
        analysis["observation_status"] = _observation_status(
            artifacts=artifacts,
            reason=reason,
            now=ctx.now,
            turn_count=turn_count,
            token_count=token_count,
            batch_turns=batch_turns,
            batch_tokens=batch_tokens,
            last_turn=last_turn,
        )
        story = str(analysis.get("story") or "")
        summary = _first_line(story) if story else f"{turn_count} turns analyzed"
        return [
            Finding(
                watch_id=watch_id,
                domain=DOMAIN,
                title=f"Session analysis \u00b7 {turn_count} turns",
                summary=summary,
                severity=Severity.NOTABLE,
                score=0.5,
                tags=("session",),
                payload=analysis,
                dedup_key=f"{watch_id}:{session_id}:{turn_count}",
            )
        ]


_SESSION_TRIGGER = "2m"


def session_watch_params(settings: Any) -> dict[str, Any]:
    """Build session-watch params from settings (falls back to producer defaults)."""
    if settings is None:
        return {}
    return {
        "batch_turns": int(getattr(settings, "monitor_session_batch_turns", _DEFAULT_BATCH_TURNS)),
        "batch_tokens": int(getattr(settings, "monitor_session_batch_tokens", _DEFAULT_BATCH_TOKENS)),
        "use_model_salience": bool(getattr(settings, "monitor_session_use_model_salience", False)),
        "debounce_s": float(getattr(settings, "monitor_session_debounce_s", _DEFAULT_DEBOUNCE_S)),
        "max_refresh_per_min": int(getattr(settings, "monitor_session_max_refresh_per_min", _DEFAULT_MAX_PER_MIN)),
    }


async def ensure_session_watch(monitors: Any, *, params: dict[str, Any] | None = None) -> str:
    """Find the active session watch, or arm a new client-coupled one; return its id.

    Single source of truth for the session watch shape (trigger, coupling) so the
    daemon RPC and the TUI slash handler cannot drift apart.
    """
    for view in monitors.list_watches():
        if view.domain == DOMAIN and view.state in ("armed", "watching"):
            return view.watch_id
    from leapflow.monitor.types import WatchSpec

    view = await monitors.arm_watch(WatchSpec(
        name="Session",
        domain=DOMAIN,
        trigger_expr=_SESSION_TRIGGER,
        sensitivity="notable",
        params=dict(params or {}),
        client_coupled=True,
    ))
    return view.watch_id


__all__ = [
    "SessionAnalysisProducer",
    "SessionAnalysisServices",
    "ensure_session_watch",
    "session_watch_params",
    "DOMAIN",
]
