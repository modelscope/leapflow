# Copyright (c) Alibaba, Inc. and its affiliates.
"""Guardian LLM-assisted approval: bridge AuxiliaryClient.classify_risk to ApprovalOrchestrator.

The Guardian is an **optional enhancement** layered on top of static risk
classification.  It never replaces the rule-based ``DefaultRiskClassifier`` —
it augments the orchestrator's decision with an LLM-derived risk score when
the static classifier yields a borderline verdict (``PolicyVerdict.ASK``).

Three operating modes are supported via ``GuardianConfig.mode``:

- ``static_only``  — Guardian is dormant; orchestrator works exactly as before.
- ``llm_assisted``  — Guardian evaluates every ASK-tier request; the LLM score
  drives an auto-approve / auto-deny / escalate-to-human decision.
- ``hybrid`` (default) — Guardian evaluates ASK-tier requests, but a timeout or
  LLM failure silently degrades to the static path (ask the user).

Design invariants:

1. Static rules remain the first line of defence: hardline / CRITICAL actions
   are never sent to the Guardian — they are denied before the orchestrator
   builds a request.
2. The Guardian is advisory: its ``recommendation`` is consumed by the
   orchestrator, not by the user.  The human prompt remains the final authority
   for anything the Guardian does not auto-resolve.
3. ``DenialBreaker`` prevents infinite approval loops: if the Guardian or the
   user deny N consecutive requests in a turn, the breaker trips and all
   subsequent requests in that turn are fast-denied without further prompting.
4. Every Guardian decision is written to a durable DuckDB ``approval_decisions``
   audit table with full provenance.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardianConfig:
    """Tuning knobs for the Guardian LLM approval adapter.

    ``mode`` controls when the Guardian fires:
    - ``static_only``: never (passthrough to existing static rules)
    - ``llm_assisted``: always for ASK-tier requests
    - ``hybrid`` (default): same as ``llm_assisted`` but silently degrades
      to static-only on LLM timeout / error
    """

    mode: str = "hybrid"  # "static_only" | "llm_assisted" | "hybrid"
    risk_threshold_auto_approve: float = 0.3
    risk_threshold_auto_deny: float = 0.8
    max_consecutive_denials: int = 3
    timeout_seconds: float = 10.0


# ---------------------------------------------------------------------------
# Verdict (immutable value object)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardianVerdict:
    """Outcome of a single Guardian LLM evaluation."""

    risk_score: float       # 0.0–1.0 from AuxiliaryClient.classify_risk
    recommendation: str     # "approve" | "review" | "deny"
    reasoning: str          # short explanation from the adapter
    latency_ms: float       # wall-clock time spent in classify_risk


# ---------------------------------------------------------------------------
# DenialBreaker (circuit breaker for consecutive denials)
# ---------------------------------------------------------------------------

class DenialBreaker:
    """Prevent infinite approval loops by tripping after N consecutive denials.

    Once tripped, ``is_tripped`` returns True and the orchestrator fast-denies
    all subsequent requests without prompting.  A single approval resets the
    counter.  Intended as a per-turn safety net.
    """

    def __init__(self, max_consecutive_denials: int = 3) -> None:
        self._max = max(1, max_consecutive_denials)
        self._consecutive: int = 0
        self._tripped: bool = False

    def record_denial(self) -> bool:
        """Record a denial.  Returns True if the breaker just tripped."""
        self._consecutive += 1
        if self._consecutive >= self._max:
            self._tripped = True
        return self._tripped

    def record_approval(self) -> None:
        """Reset the consecutive denial counter on approval."""
        self._consecutive = 0
        # Note: once tripped, the breaker stays tripped for the remainder
        # of the turn.  An approval resets the counter but does NOT un-trip.

    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        """Full reset — call between turns."""
        self._consecutive = 0
        self._tripped = False


# ---------------------------------------------------------------------------
# Audit sink protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GuardianAuditSink(Protocol):
    """Persist a Guardian decision record to durable storage."""

    async def record_guardian_decision(
        self,
        *,
        session_id: str,
        tool_name: str,
        risk_score: float,
        recommendation: str,
        decision: str,
        reasoning: str,
        latency_ms: float,
        metadata: dict[str, Any],
    ) -> None: ...


class NullGuardianAuditSink:
    """No-op audit sink when DuckDB is unavailable or storage not injected."""

    async def record_guardian_decision(self, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# GuardianDecisionAdapter
# ---------------------------------------------------------------------------

class GuardianDecisionAdapter:
    """Bridge ``AuxiliaryClient.classify_risk`` into an ``ApprovalOrchestrator``
    compatible advisory verdict.

    The adapter does NOT make final decisions — it returns a
    ``GuardianVerdict`` that the orchestrator interprets according to the
    configured thresholds.
    """

    def __init__(
        self,
        auxiliary_client: Any,  # AuxiliaryClient — loosely typed to avoid import cycle
        config: GuardianConfig | None = None,
        *,
        audit_sink: GuardianAuditSink | None = None,
    ) -> None:
        self._client = auxiliary_client
        self._config = config or GuardianConfig()
        self._audit = audit_sink or NullGuardianAuditSink()

    @property
    def config(self) -> GuardianConfig:
        return self._config

    async def evaluate(
        self,
        *,
        tool_name: str,
        detail: str,
        risk_hint: float,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> GuardianVerdict:
        """Call the LLM to classify risk, map the score to a recommendation.

        The ``detail`` string is the action summary passed to
        ``classify_risk``.  ``risk_hint`` is the static classifier's score
        (used as context, not as a fallback).
        """
        context = f"[tool={tool_name}] [static_risk={risk_hint:.2f}] {detail[:3800]}"
        t0 = time.monotonic()
        try:
            score = await asyncio.wait_for(
                self._client.classify_risk(context, timeout_s=self._config.timeout_seconds),
                timeout=self._config.timeout_seconds + 2.0,  # outer safety net
            )
        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning("guardian.evaluate timed out after %.0fms", latency_ms)
            return self._timeout_verdict(latency_ms)
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning("guardian.evaluate failed: %s (%.0fms)", exc, latency_ms)
            return self._error_verdict(latency_ms, str(exc))

        latency_ms = (time.monotonic() - t0) * 1000
        recommendation = self._map_score(score)
        reasoning = (
            f"LLM risk score {score:.2f}: "
            f"{'auto-approve (below threshold)' if recommendation == 'approve' else ''}"
            f"{'escalate to human review' if recommendation == 'review' else ''}"
            f"{'auto-deny (above threshold)' if recommendation == 'deny' else ''}"
        )
        verdict = GuardianVerdict(
            risk_score=score,
            recommendation=recommendation,
            reasoning=reasoning.strip(),
            latency_ms=latency_ms,
        )

        # Best-effort audit — never fail the turn
        try:
            await self._audit.record_guardian_decision(
                session_id=session_id,
                tool_name=tool_name,
                risk_score=score,
                recommendation=recommendation,
                decision=recommendation,
                reasoning=verdict.reasoning,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        except Exception as audit_exc:
            logger.debug("guardian.audit write failed: %s", audit_exc)

        return verdict

    def _map_score(self, score: float) -> str:
        """Map a 0.0–1.0 score to a recommendation string."""
        if score <= self._config.risk_threshold_auto_approve:
            return "approve"
        if score >= self._config.risk_threshold_auto_deny:
            return "deny"
        return "review"

    @staticmethod
    def _timeout_verdict(latency_ms: float) -> GuardianVerdict:
        return GuardianVerdict(
            risk_score=0.5,
            recommendation="review",
            reasoning="LLM timed out — falling back to human review",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _error_verdict(latency_ms: float, error: str) -> GuardianVerdict:
        return GuardianVerdict(
            risk_score=0.5,
            recommendation="review",
            reasoning=f"LLM error — falling back to human review ({error[:120]})",
            latency_ms=latency_ms,
        )
