"""Dispatches due re-entry triggers by seeding an Orient-seeded run (S2, phase N3).

Pure orchestration: reads due TIME triggers from a ``ReentryStore``, atomically
claims each (CAS ``fire`` -> at-most-once), and invokes an injected async runner
with the trigger's ``OrientSnapshot`` (typically ``engine.resume_from_orient``).
It does NOT touch the engine core loop -- the runner is an abstraction, and the
caller (daemon) is responsible for serializing dispatch via ``_engine_lock`` so
no concurrent engine runs occur.

Guardrails: bounded per-tick fan-out; a claim happens *before* running (so a
failed re-entry is not silently retried into a storm); recurring triggers that
remain armed after a claim have their next due time advanced to prevent
same-tick re-fire. Enablement is injected (default off via config).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Optional, Union

from leapflow.storage.reentry_store import OrientSnapshot, ReentryState, ReentryStore

logger = logging.getLogger(__name__)

ReentryRunner = Callable[[OrientSnapshot, Any], Awaitable[Any]]

_MAX_CTX_ITEMS = 12


def build_reentry_subagent_config(orient: OrientSnapshot) -> Any:
    """Turn an OrientSnapshot into an isolated subagent config (S2 N3b).

    Re-entry runs as an *isolated* subagent (fresh context, own budget) seeded
    with the orientation as text -- so an autonomous re-entry never pollutes the
    interactive engine's working memory or session. The original goal becomes
    the subagent goal; findings / open questions / next step / continuation
    summary become its context.
    """
    from leapflow.engine.subagent import SubagentConfig

    contract = orient.task_contract or {}
    goal = str(contract.get("original_request", "") or "").strip() or "Continue the task."
    ledger = orient.ledger_state or {}
    lines: list[str] = []
    if orient.continuation_summary:
        lines.append(f"Continue: {orient.continuation_summary}")
    findings = list(ledger.get("findings") or [])[:_MAX_CTX_ITEMS]
    if findings:
        lines.append("Findings so far:")
        lines.extend(f"- {item}" for item in findings)
    open_questions = list(ledger.get("open_questions") or [])[:_MAX_CTX_ITEMS]
    if open_questions:
        lines.append("Open questions to resolve:")
        lines.extend(f"- {item}" for item in open_questions)
    next_step = str(ledger.get("next_step") or "").strip()
    if next_step:
        lines.append(f"Next step: {next_step}")
    return SubagentConfig(
        goal=goal,
        context="\n".join(lines),
        metadata={"reentry": True, "task_id": orient.task_id},
    )


def event_matches(event_match: dict, *, platform: str = "", chat: str = "", text: str = "") -> bool:
    """Whether an inbound gateway message matches an EVENT trigger's filter.

    An empty filter matches *nothing* (safety: never fire on all traffic). Each
    present field must match: platform (exact), chat (exact, ``chat`` or
    ``chat_id``), keyword (case-insensitive substring of the message text).
    """
    if not event_match:
        return False
    want_platform = event_match.get("platform")
    if want_platform and str(want_platform) != str(platform):
        return False
    want_chat = event_match.get("chat") or event_match.get("chat_id")
    if want_chat and str(want_chat) != str(chat):
        return False
    keyword = event_match.get("keyword")
    if keyword and str(keyword).lower() not in str(text).lower():
        return False
    return True


class ReentryDriver:
    """Periodic dispatcher of due re-entry triggers (ticked by the caller)."""

    def __init__(
        self,
        *,
        store: ReentryStore,
        runner: ReentryRunner,
        enabled: Union[Callable[[], bool], bool] = True,
        max_per_tick: int = 4,
        recurring_interval_seconds: float = 3600.0,
    ) -> None:
        self._store = store
        self._runner = runner
        self._enabled = enabled
        self._max_per_tick = max(1, int(max_per_tick))
        self._recurring_interval = max(1.0, float(recurring_interval_seconds))

    def _is_enabled(self) -> bool:
        try:
            return self._enabled() if callable(self._enabled) else bool(self._enabled)
        except Exception:
            return False

    async def tick(self, now: Optional[float] = None) -> int:
        """Dispatch due TIME triggers. Returns the number successfully dispatched.

        For each due trigger (bounded by ``max_per_tick``): CAS-claim via
        ``fire()``; if claimed, run the injected runner with its OrientSnapshot.
        A recurring trigger still armed after the claim has its ``due_at``
        advanced so it does not re-fire within the same tick window.
        """
        if not self._is_enabled():
            return 0
        now = time.time() if now is None else now
        due = self._store.list_due(now)
        dispatched = 0
        for trig in due[: self._max_per_tick]:
            claimed = self._store.fire(trig.trigger_id, now=now)
            if claimed is None:
                continue  # lost the race / not consumable
            # Recurring: still armed after the claim -> push next due to avoid a storm.
            if claimed.state == ReentryState.ARMED.value:
                self._store.advance_due(claimed.trigger_id, now + self._recurring_interval)
            if claimed.orient is None:
                logger.warning("reentry trigger %s has no orient snapshot; skipping run",
                               claimed.trigger_id)
                continue
            try:
                await self._runner(claimed.orient, claimed)
                dispatched += 1
            except Exception:
                # Already claimed (at-most-once): a failed re-entry is logged, not retried.
                logger.error("reentry dispatch failed for %s", claimed.trigger_id, exc_info=True)
        return dispatched
