# Copyright (c) Alibaba, Inc. and its affiliates.
"""Difficulty/threshold calibration and prefix-commitment helpers.

Extracted from ``engine.py`` (Phase 3 refactor). Owns the online calibration
of budget difficulty (``scale_k``) and finalize posture threshold, periodic
recalibration, progress-marker fingerprinting, cost-ceiling nudges, and the
cache-aware prefix-commitment evaluation. Holds a back-reference to the owning
engine so every access reads the engine's live mutable state, preserving exact
runtime semantics.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from leapflow.engine.prefix_commitment import CommitmentStatus, _system_prompt_hash
from leapflow.engine.context.context_disclosure import CacheBoundary, DisclosureLevel
from leapflow.engine.turn_usage import cost_ceiling_exceeded

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.engine.engine import AgentEngine
    from leapflow.engine.budget import IterationBudget

logger = logging.getLogger(__name__)


class CalibrationManager:
    """Online calibration and prefix-commitment logic, held by composition."""

    def __init__(self, engine: "AgentEngine") -> None:
        self._engine = engine

    def recalibrate_difficulty(self, store: Any) -> Any:
        """S3-L3: apply offline calibration (S3-L2) to the difficulty weight.

        Bounded, gated, and reversible: reads recent turn signals from the
        evolution store and — only when ``agent.calibration_enabled`` — installs a
        clamped ``scale_k`` derived from the *baseline* weight. Default-off, so
        budget behavior is byte-identical unless explicitly enabled. Returns the
        ``CalibrationResult`` for observability.
        """
        from leapflow.learning.difficulty_calibration import (
            CalibrationResult,
            apply_calibration,
            build_calibration_report_from_store,
        )

        enabled = bool(getattr(self._engine._settings, "agent_calibration_enabled", False))
        if not enabled or store is None:
            return CalibrationResult(
                self._engine._baseline_scale_k,
                self._engine._budget_config.scale_k,
                False,
                "calibration disabled" if not enabled else "no evolution store",
            )
        try:
            report = build_calibration_report_from_store(store)
        except Exception:
            logger.debug("difficulty calibration: report build failed", exc_info=True)
            return CalibrationResult(
                self._engine._baseline_scale_k,
                self._engine._budget_config.scale_k,
                False,
                "report build failed",
            )
        configured_min = float(
            getattr(self._engine._settings, "agent_calibration_difficulty_min_k", 0.25)
        )
        configured_max = float(
            getattr(self._engine._settings, "agent_calibration_difficulty_max_k", 3.0)
        )
        k_min = min(3.0, max(0.25, configured_min))
        k_max = max(k_min, min(3.0, configured_max))
        result = apply_calibration(
            self._engine._baseline_scale_k,
            report,
            enabled=True,
            min_confidence=float(
                getattr(self._engine._settings, "agent_calibration_min_confidence", 0.3)
            ),
            k_min=k_min,
            k_max=k_max,
        )
        if result.applied:
            self._engine._budget_config = replace(
                self._engine._budget_config, scale_k=result.effective_k
            )
            self._record_calibration_event(
                "difficulty_scale",
                baseline=result.baseline_k,
                effective=result.effective_k,
                reason=result.reason,
                lower_bound=k_min,
                upper_bound=k_max,
            )
            logger.info(
                "difficulty calibration applied: scale_k %.3f -> %.3f (%s)",
                self._engine._baseline_scale_k,
                result.effective_k,
                result.reason,
            )
        return result

    def reset_calibration(self) -> None:
        """Revert any applied difficulty calibration to the configured baseline."""
        self._engine._budget_config = replace(
            self._engine._budget_config, scale_k=self._engine._baseline_scale_k
        )

    def recalibrate_thresholds(self, store: Any) -> Any:
        """S3-L4: tune the finalize posture threshold from stored signals.

        Same bounded/gated/reversible contract as :meth:`recalibrate_difficulty`,
        applied to ``context_finalizing_ratio`` (clamped to a safe band) and
        derived from the configured baseline. Default-off; rebuilds the governance
        controller so subsequent frames observe the calibrated threshold.
        """
        from leapflow.learning.difficulty_calibration import (
            CalibrationResult,
            apply_calibration,
            build_threshold_report_from_store,
        )

        baseline = self._engine._settings.context_finalizing_ratio
        current = self._engine._calibrated_finalizing_ratio or baseline
        enabled = bool(getattr(self._engine._settings, "agent_calibration_enabled", False))
        if not enabled or store is None:
            return CalibrationResult(
                baseline,
                current,
                False,
                "calibration disabled" if not enabled else "no evolution store",
            )
        try:
            report = build_threshold_report_from_store(store)
        except Exception:
            logger.debug("threshold calibration: report build failed", exc_info=True)
            return CalibrationResult(baseline, current, False, "report build failed")
        configured_min = float(
            getattr(self._engine._settings, "agent_calibration_finalizing_min_ratio", 0.6)
        )
        configured_max = float(
            getattr(self._engine._settings, "agent_calibration_finalizing_max_ratio", 0.98)
        )
        k_min = min(0.98, max(0.6, configured_min))
        k_max = max(k_min, min(0.98, configured_max))
        result = apply_calibration(
            baseline,
            report,
            enabled=True,
            min_confidence=float(
                getattr(self._engine._settings, "agent_calibration_min_confidence", 0.3)
            ),
            k_min=k_min,
            k_max=k_max,
        )
        if result.applied:
            self._engine._calibrated_finalizing_ratio = result.effective_k
            self._engine._context_governance_controller = self._engine._new_governance()
            self._record_calibration_event(
                "finalizing_ratio",
                baseline=result.baseline_k,
                effective=result.effective_k,
                reason=result.reason,
                lower_bound=k_min,
                upper_bound=k_max,
            )
            logger.info(
                "threshold calibration applied: finalizing_ratio %.3f -> %.3f (%s)",
                baseline,
                result.effective_k,
                result.reason,
            )
        return result

    def reset_threshold_calibration(self) -> None:
        """Revert any applied finalize-threshold calibration to the baseline."""
        self._engine._calibrated_finalizing_ratio = None
        self._engine._context_governance_controller = self._engine._new_governance()

    def _record_calibration_event(
        self,
        parameter: str,
        *,
        baseline: float,
        effective: float,
        reason: str,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        store = self._engine._calibration_event_store
        if store is None:
            return
        try:
            import time

            from leapflow.domain.event_types import EvolutionEventType
            from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent

            occurred_at = time.time()
            event = EvolutionEvent.create(
                EvolutionEventType.CALIBRATION_UPDATED,
                context=EvolutionContext(
                    profile_id=str(getattr(self._engine._settings, "profile", "default")),
                    correlation_id=f"calibration:{parameter}",
                ),
                payload={
                    "parameter": parameter,
                    "baseline": float(baseline),
                    "effective": float(effective),
                    "reason": str(reason),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                },
                producer="engine.online_calibration",
                privacy_class="profile",
                occurred_at=occurred_at,
                dedup_key=f"calibration.updated:{parameter}:{time.time_ns()}",
            )
            store.append(event)
        except Exception:  # noqa: BLE001 - calibration audit cannot break a turn
            logger.error("calibration decision could not be persisted", exc_info=True)

    def _maybe_periodic_recalibration(self) -> None:
        """S3-L3/L4 periodic re-calibration (opt-in via agent.calibration_interval_turns).

        The one-shot startup calibration already applies the learned adjustment;
        when a positive interval is set, re-run every N *root* turns so calibration
        tracks accumulating outcome data. Default 0 = one-shot only (no periodic).
        Bounded/gated/reversible like the underlying recalibration; never raises.
        """
        if not getattr(self._engine._settings, "agent_calibration_enabled", False):
            return
        interval = int(getattr(self._engine._settings, "agent_calibration_interval_turns", 0) or 0)
        if interval <= 0 or self._engine._calibration_store is None:
            return
        self._engine._turns_since_calibration += 1
        if self._engine._turns_since_calibration < interval:
            return
        self._engine._turns_since_calibration = 0
        try:
            self.recalibrate_difficulty(self._engine._calibration_store)
            self.recalibrate_thresholds(self._engine._calibration_store)
        except Exception:
            logger.debug("periodic recalibration failed", exc_info=True)

    def _widen_budget_for_difficulty(self, budget: "IterationBudget") -> None:
        """Raise the elastic iteration cap to match the observed difficulty.

        Reads the difficulty produced by the most recent ``_prepare_llm_messages``
        governance snapshot and retargets the budget toward the difficulty-scaled
        ceiling. No-op for fixed budgets and for difficulty 0 (baseline floor).
        This is how a hard task earns a wider horizon while a simple task stays
        near the floor and relies on self-stop / answer-ready convergence.
        """
        difficulty = float(self._engine._last_context_snapshot.get("difficulty", 0.0) or 0.0)
        budget.retarget(budget.elastic_max(difficulty))

    def _task_progress_marker(self) -> tuple:
        """Fingerprint of task progress for stall detection (P0).

        Combines the research-ledger shape (findings / open questions /
        decisions / next step) with governance evidence breadth (evidence count,
        distinct sources, repeated reads). A change between rounds means the task
        advanced; an unchanged marker across rounds indicates a stall. Including
        repeated_reads ensures that growing re-reads (with no other progress)
        keep the marker unchanged, so stalled_rounds increments correctly.
        """
        d = self._engine._research_ledger.as_dict()
        gov = self._engine._last_context_snapshot.get("context_governance", {}) or {}
        return (
            len(d.get("findings", [])),
            len(d.get("open_questions", [])),
            len(d.get("decisions", [])),
            d.get("next_step", ""),
            int(gov.get("evidence_count", 0) or 0),
            int(gov.get("sources_seen", 0) or 0),
            int(gov.get("repeated_reads", 0) or 0),
        )

    def _cost_ceiling_notice(self) -> str:
        """Soft finalize nudge when cumulative effective cost crosses the ceiling.

        Opt-in safety companion to the elastic iteration cap: bounds runaway cost
        on large-context long tasks. Soft (a nudge, not a hard stop) so no work is
        lost; the iteration ceiling remains the hard bound. Disabled by default
        (``agent_cost_ceiling_context_multiple`` = 0).
        """
        multiple = float(
            getattr(self._engine._settings, "agent_cost_ceiling_context_multiple", 0.0) or 0.0
        )
        if multiple <= 0:
            return ""
        effective = self._engine._usage_tracker.summary().effective_prompt_tokens()
        if not cost_ceiling_exceeded(
            effective_prompt_tokens=effective,
            context_length=self._engine._active_context_length(),
            context_multiple=multiple,
        ):
            return ""
        return (
            "SYSTEM: Cumulative cost budget reached. Synthesize and provide the final "
            "answer now from the evidence already gathered; do not start new exploratory "
            "tool calls unless strictly required."
        )

    def _full_tool_schema_tokens(self) -> int:
        """Cached token estimate of the full unified catalog schema.

        Invalidated whenever the unified catalog rebuilds (static registry
        growth or desktop plugin identity/version change).
        """
        if self._engine._full_tools_tokens is None:
            self._engine._full_tools_tokens = self._engine._context_controller.estimator.estimate_tools(
                self._engine._tool_dispatch._unified_tool_catalog()
            )
        return self._engine._full_tools_tokens

    def _cache_aware_plan_kwargs(self) -> dict:
        """Build keyword arguments for ``DisclosurePlanner.plan`` cache-aware path.

        Cold-path helper (once per round).  Three cases:

        1. **Already committed with enforcement** — pass the frozen disclosure
           snapshot so the planner reproduces a byte-stable prefix.
        2. **Uncommitted with positive projected savings** — pass
           ``cache_benefit=True`` so the planner emits a ``SOFT`` boundary,
           which instructs ``PrefixCacheOptimizer`` to reorder messages for
           prefix stability *before* formal commitment.
        3. **Otherwise** — return an empty dict (backward-compatible ``NONE``).

        SOFT does **not** freeze disclosure level or lock the tool set — it
        only influences message cache layout (PCD minimum-sufficiency preserved).
        """
        commitment = self._engine._prefix_commitment
        enforcement = commitment.enforcement

        # Case 1: already committed with active enforcement
        if commitment.committed and enforcement is not None:
            return {
                "commitment_status": CommitmentStatus.COMMITTED,
                "committed_level": DisclosureLevel(enforcement.frozen_level),
                "committed_tool_names": enforcement.frozen_tool_names,
            }

        # Case 2: uncommitted — evaluate cache benefit from prior-round snapshot
        snap = self._engine._last_context_snapshot
        if not snap or commitment.committed:
            return {}
        msg_tokens = int(snap.get("message_tokens", 0) or 0)
        disclosed_tool_tokens = int(snap.get("tool_schema_tokens", 0) or 0)
        if msg_tokens <= 0:
            return {}  # no prior-round data yet (first round)
        est_full = msg_tokens + self._full_tool_schema_tokens()
        est_pcd = msg_tokens + disclosed_tool_tokens
        # Use budget max_iterations as a generous upper bound for remaining;
        # the real commitment gate in _evaluate_prefix_commitment uses actual
        # budget.remaining, so this only controls the soft-benefit signal.
        remaining = max(1, self._engine._budget_config.max_iterations - 1)
        savings = commitment.projected_savings(
            remaining_rounds=remaining,
            est_full_prefix_tokens=est_full,
            est_pcd_prefix_tokens=est_pcd,
        )
        if savings > 0:
            return {
                "commitment_status": CommitmentStatus.UNCOMMITTED,
                "cache_benefit": True,
            }
        return {}

    def _evaluate_prefix_commitment(self, budget: "IterationBudget") -> None:
        """Evaluate the adaptive prefix-commitment decision and apply enforcement.

        Two phases run once per round on the cold path (never per token):

        1. **Observe** -- compute whether the task should commit to a stable,
           cacheable prefix and record the decision in the context snapshot for
           observability. Reuses the token counts already produced by
           ``_prepare_llm_messages`` plus the post-retarget budget headroom, so
           no message body is re-estimated.
        2. **Enforce** (W2 slice 3) -- once committed, freeze the disclosure
           snapshot via :meth:`PrefixCommitmentController.enforce` and switch the
           session onto the ``COMMITTED`` cache boundary so the marker
           application in ``_prepare_llm_messages`` / before ``achat`` can cache
           the stable prefix. When enforcement is absent (never committed, or
           broken via :meth:`break_commitment`) the boundary falls back to
           ``NONE`` and normal PCD dynamics resume next round.
        """
        snap = self._engine._last_context_snapshot
        if not snap:
            return
        difficulty = float(snap.get("difficulty", 0.0) or 0.0)
        posture = str(snap.get("context_posture") or "baseline")
        message_tokens = int(snap.get("message_tokens", 0) or 0)
        disclosed_tool_tokens = int(snap.get("tool_schema_tokens", 0) or 0)
        est_full = message_tokens + self._full_tool_schema_tokens()
        est_pcd = message_tokens + disclosed_tool_tokens
        state = self._engine._prefix_commitment.evaluate(
            difficulty=difficulty,
            posture=posture,
            round_number=budget.used,
            remaining_rounds=budget.remaining,
            est_full_prefix_tokens=est_full,
            est_pcd_prefix_tokens=est_pcd,
        )
        snap["prefix_commitment"] = state.as_dict()
        snap["prefix_committed"] = state.committed

        # Enforce (2c): freeze the disclosure snapshot and switch to the
        # committed cache boundary. ``enforce`` is idempotent while an
        # enforcement is active (returns the existing snapshot), so this is
        # cheap to call every round. The frozen values are the disclosure
        # decision this turn recorded in ``_last_disclosure_metadata`` plus the
        # hash of the system prompt actually assembled this turn.
        boundary = CacheBoundary.NONE
        if state.committed:
            meta = self._engine._last_disclosure_metadata
            enforcement = self._engine._prefix_commitment.enforce(
                str(meta.get("level", DisclosureLevel.CORE.value)),
                tuple(meta.get("tools", ()) or ()),
                _system_prompt_hash(self._engine._last_system_prompt),
                int(self._engine._session_turn_count),
            )
            if enforcement is not None:
                boundary = CacheBoundary.COMMITTED
                snap["prefix_enforcement"] = {
                    "frozen_level": enforcement.frozen_level,
                    "frozen_tool_count": len(enforcement.frozen_tool_names),
                    "committed_at_turn": enforcement.committed_at_turn,
                }
        else:
            # P0-OPT-2: promote to SOFT when projected savings are positive.
            # This lets PrefixCacheOptimizer stabilize the prefix layout in
            # pre-commitment rounds without freezing disclosure or tools.
            savings = self._engine._prefix_commitment.projected_savings(
                remaining_rounds=budget.remaining,
                est_full_prefix_tokens=est_full,
                est_pcd_prefix_tokens=est_pcd,
            )
            if savings > 0:
                boundary = CacheBoundary.SOFT
        self._engine._current_cache_boundary = boundary
        snap["cache_boundary"] = boundary.value

    def _maybe_break_commitment(
        self,
        *,
        posture_changed: bool = False,
        tool_error: bool = False,
        slash_command: bool = False,
        transform_retry: bool = False,
    ) -> bool:
        """Break prefix-commitment enforcement on a structural prefix disruption.

        Delegates the decision to
        :meth:`PrefixCommitmentController.should_break_commitment` and, when it
        fires, clears the enforcement (the commitment *decision* stays monotonic)
        and drops the cache boundary back to ``NONE`` so the next round assembles
        a fresh, non-frozen prefix. Returns whether a break occurred.
        """
        if not self._engine._prefix_commitment.enforcement:
            return False
        if not self._engine._prefix_commitment.should_break_commitment(
            posture_changed=posture_changed,
            tool_error=tool_error,
            slash_command=slash_command,
            transform_retry=transform_retry,
        ):
            return False
        self._engine._prefix_commitment.break_commitment()
        self._engine._current_cache_boundary = CacheBoundary.NONE
        return True
