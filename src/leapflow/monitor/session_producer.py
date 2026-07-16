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
        self, messages: list[dict[str, Any]], *, prior: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return a structured analysis payload for the transcript."""
        ...

    async def should_refresh(self, messages: list[dict[str, Any]]) -> bool:
        """Return True when the model judges a refresh worthwhile (salience)."""
        ...


def _first_line(text: str, limit: int = 180) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1] + "\u2026"


class SessionAnalysisProducer:
    """Produce session-analysis findings from the live conversation transcript."""

    def __init__(self) -> None:
        self._last_turn: dict[str, int] = {}
        self._last_tokens: dict[str, int] = {}
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

        last_turn = self._last_turn.get(watch_id, -1)
        first = last_turn < 0

        if not ctx.force and not first and turn_count <= last_turn:
            return []  # no new turns since last analysis

        if not ctx.force and not first:
            if (ctx.now - self._last_run_at.get(watch_id, 0.0)) < debounce_s:
                return []
            recent = [t for t in self._recent_runs.get(watch_id, []) if ctx.now - t < 60.0]
            if len(recent) >= max_per_min:
                return []

        should = (
            ctx.force
            or first
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

        try:
            analysis = dict(await services.analyze_session(messages))
        except Exception as exc:  # noqa: BLE001 - surface as no-op, not crash
            logger.debug("session producer: analysis failed: %s", exc)
            return []

        self._last_turn[watch_id] = turn_count
        self._last_tokens[watch_id] = token_count
        self._last_run_at[watch_id] = ctx.now
        self._recent_runs.setdefault(watch_id, []).append(ctx.now)

        analysis.setdefault("usage", {})
        analysis["usage"].update({"turns": turn_count, "tokens": token_count})
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
