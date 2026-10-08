# Copyright (c) Alibaba, Inc. and its affiliates.
"""Physical capability resolver: four-value adjudication for the physical domain.

When a physical operation fails or underperforms, the resolver decides how
the framework should adapt.  It maps physical capability gaps to the same
four-value verdict space the software evolution engine uses (absorb, rebind,
acquire, escalate), grounding each verdict in evidence: verification verdicts,
success rates, and environment outcomes rather than a single failure.

The verdict costs are ordered: absorb (adjust a threshold, no new code) is
cheapest, escalate (ask a human to teleoperate) is used only when nothing
cheaper applies.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from leapflow.domain.adaptation_verdict import AdaptationVerdict

__all__ = [
    "PhysicalGapType",
    "PhysicalCapabilityGap",
    "PhysicalCapabilityResolver",
]

# Success-rate window below the precision bar that is still considered a
# marginal, in-band deviation the verifier can safely widen tolerance for.
_DEFAULT_ABSORB_MARGIN = 0.15

# Health states that mean the device is connected but not operating correctly.
_DEGRADED_HEALTH = frozenset({"degraded", "stale", "unreachable"})

# Physical-domain risk ceiling requested by an ``acquire`` verdict: a downloaded
# policy ultimately actuates hardware.  Still clamped downstream by the trusted
# caller, exactly as a software ``acquire`` is.
_ACQUIRE_RISK_LEVEL = "mutating"

_SEGMENT_RE = re.compile(r"[^a-z0-9]+")


def _slug_segment(value: str, fallback: str) -> str:
    """Coerce *value* into one capability-name segment ``[a-z0-9][a-z0-9_]*``.

    Returns *fallback* when nothing usable survives normalisation, so the
    assembled name always satisfies ``is_capability_name``.
    """
    text = _SEGMENT_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return text[:32] or fallback


class PhysicalGapType(str, Enum):
    """Classification of a physical capability gap."""

    MISSING_SKILL = "missing_skill"  # no policy/strategy for the manipulation
    INSUFFICIENT_PRECISION = "insufficient_precision"  # policy exists but verification fails too often
    MISSING_DEVICE = "missing_device"  # required device type not available
    HARDWARE_DEGRADED = "hardware_degraded"  # device connected but probes/reads failing


@dataclass(frozen=True)
class PhysicalCapabilityGap:
    """A detected physical capability gap with supporting evidence."""

    gap_type: str  # PhysicalGapType
    device_id: str
    affordance: str = ""
    success_rate: float = 0.0
    sample_count: int = 0
    recent_failures: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_type": self.gap_type,
            "device_id": self.device_id,
            "affordance": self.affordance,
            "success_rate": self.success_rate,
            "sample_count": self.sample_count,
            "recent_failures": list(self.recent_failures),
            "detail": self.detail,
        }


class PhysicalCapabilityResolver:
    """Adjudicates physical capability gaps into four-value verdicts.

    Dependencies:
    - inference_registry: InferenceStrategyRegistry (for rebind candidate lookup)
    - evidence_store: EvidenceStore (for success rate evidence)
    - trust_gate: HardwareTrustGate (optional, for trust-aware decisions)
    - acquire_probe: optional callable overriding the HF Hub acquisition check,
      so the acquire path can be exercised without a network round-trip.
    """

    def __init__(
        self,
        *,
        inference_registry: Any = None,
        evidence_store: Any = None,
        trust_gate: Any = None,
        precision_threshold: float = 0.7,  # success rate below which precision is insufficient
        min_samples: int = 5,  # minimum operations before adjudicating
        absorb_margin: float = _DEFAULT_ABSORB_MARGIN,
        acquire_probe: Callable[["PhysicalCapabilityGap"], bool] | None = None,
    ) -> None:
        self._inference_registry = inference_registry
        self._evidence_store = evidence_store
        self._trust_gate = trust_gate
        self._precision_threshold = max(0.0, min(1.0, float(precision_threshold)))
        self._min_samples = max(1, int(min_samples))
        self._absorb_margin = max(0.0, float(absorb_margin))
        self._acquire_probe = acquire_probe

    # ── Detection ──

    async def detect_gaps(
        self, environment_outcome: Mapping[str, Any]
    ) -> tuple[PhysicalCapabilityGap, ...]:
        """Detect physical gaps from a PhysicalEnvironmentSource outcome observation.

        Reads per-device success rates and failure patterns, classifies each
        underperforming device/affordance into a PhysicalGapType.
        """
        devices = environment_outcome.get("devices") if environment_outcome else None
        if not isinstance(devices, Mapping):
            return ()
        gaps: list[PhysicalCapabilityGap] = []
        for device_id, stats in devices.items():
            if not isinstance(stats, Mapping):
                continue
            gaps.extend(self._classify_device(str(device_id), stats))
        return tuple(gaps)

    def _classify_device(
        self, device_id: str, stats: Mapping[str, Any]
    ) -> list[PhysicalCapabilityGap]:
        """Classify one device's outcome stats into zero or more gaps.

        Structural faults (a disconnected or degraded device) short-circuit:
        a device that cannot be read cannot also be judged on success rate.
        """
        # Structural gaps first — these are facts, not statistics.
        if not bool(stats.get("connected", True)):
            return [
                PhysicalCapabilityGap(
                    gap_type=PhysicalGapType.MISSING_DEVICE.value,
                    device_id=device_id,
                    affordance=str(stats.get("affordance", "")),
                    detail=f"Device '{device_id}' is not connected.",
                )
            ]
        health = str(stats.get("health", "ok")).strip().lower()
        if health in _DEGRADED_HEALTH:
            return [
                PhysicalCapabilityGap(
                    gap_type=PhysicalGapType.HARDWARE_DEGRADED.value,
                    device_id=device_id,
                    affordance=str(stats.get("affordance", "")),
                    success_rate=float(stats.get("success_rate", 0.0) or 0.0),
                    sample_count=int(stats.get("total_operations", 0) or 0),
                    recent_failures=_as_str_tuple(stats.get("recent_failures")),
                    detail=f"Device '{device_id}' health is '{health}'.",
                )
            ]

        gaps: list[PhysicalCapabilityGap] = []

        # Affordances that were requested but have no policy at all.
        for affordance in _as_str_tuple(stats.get("unsupported_affordances")):
            gaps.append(
                PhysicalCapabilityGap(
                    gap_type=PhysicalGapType.MISSING_SKILL.value,
                    device_id=device_id,
                    affordance=affordance,
                    sample_count=0,
                    detail=f"No policy provides affordance '{affordance}'.",
                )
            )

        # Underperformance, either per-affordance or at the device level.
        breakdown = stats.get("affordances")
        if isinstance(breakdown, Mapping) and breakdown:
            for affordance, aff_stats in breakdown.items():
                if isinstance(aff_stats, Mapping):
                    gap = self._classify_performance(
                        device_id, str(affordance), aff_stats
                    )
                    if gap is not None:
                        gaps.append(gap)
        else:
            gap = self._classify_performance(
                device_id, str(stats.get("affordance", "")), stats
            )
            if gap is not None:
                gaps.append(gap)
        return gaps

    def _classify_performance(
        self, device_id: str, affordance: str, stats: Mapping[str, Any]
    ) -> PhysicalCapabilityGap | None:
        """Return an INSUFFICIENT_PRECISION gap when a policy underperforms.

        Requires at least one recorded operation; a device with no evidence
        yields no gap here (missing skills come from an explicit declaration).
        """
        total = int(stats.get("total_operations", stats.get("sample_count", 0)) or 0)
        if total <= 0:
            return None
        success_rate = float(stats.get("success_rate", 0.0) or 0.0)
        if success_rate >= self._precision_threshold:
            return None
        return PhysicalCapabilityGap(
            gap_type=PhysicalGapType.INSUFFICIENT_PRECISION.value,
            device_id=device_id,
            affordance=affordance,
            success_rate=success_rate,
            sample_count=total,
            recent_failures=_as_str_tuple(stats.get("recent_failures")),
            detail=(
                f"Affordance '{affordance}' on '{device_id}' succeeds "
                f"{success_rate:.0%} of {total} operation(s)."
            ),
        )

    # ── Adjudication ──

    async def adjudicate(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict | None:
        """Return an AdaptationVerdict for the gap.

        Decision logic:
        - INSUFFICIENT_PRECISION with a small deviation → absorb (widen tolerance)
        - INSUFFICIENT_PRECISION with alternative strategy available → rebind
        - MISSING_SKILL with a downloadable policy on HF Hub → acquire
        - MISSING_DEVICE → escalate (human must connect hardware)
        - HARDWARE_DEGRADED → escalate (human must inspect device)
        - Anything unresolvable cheaply → escalate (request teleoperation)

        Returns ``None`` when the evidence is too thin to conclude anything —
        an underperforming policy is judged on repeated samples, never a single
        failure.  Structural gaps (missing/degraded hardware) are facts and are
        never gated on sample count.
        """
        gap_type = gap.gap_type

        if gap_type == PhysicalGapType.INSUFFICIENT_PRECISION.value:
            if gap.sample_count < self._min_samples:
                # Not enough evidence to conclude the policy is imprecise.
                return None
            return self._adjudicate_precision(gap)

        if gap_type == PhysicalGapType.MISSING_SKILL.value:
            return self._adjudicate_missing_skill(gap)

        if gap_type == PhysicalGapType.MISSING_DEVICE.value:
            return self._escalate_missing_device(gap)

        if gap_type == PhysicalGapType.HARDWARE_DEGRADED.value:
            return self._escalate_degraded(gap)

        # Unknown or otherwise unresolvable gap: fall back to teleoperation.
        return self._escalate_teleop(gap)

    def _adjudicate_precision(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict:
        """Cheapest-first: widen tolerance, else switch strategy, else escalate."""
        deviation = self._precision_threshold - gap.success_rate
        if deviation <= self._absorb_margin:
            return self._absorb(gap, deviation)
        candidate = self._find_rebind_candidate(gap)
        if candidate:
            return self._rebind(gap, candidate)
        return self._escalate_teleop(gap)

    def _adjudicate_missing_skill(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict:
        """Rebind to an installed alternative, else acquire, else escalate."""
        candidate = self._find_rebind_candidate(gap)
        if candidate:
            return self._rebind(gap, candidate)
        if self._can_acquire(gap):
            return self._acquire(gap)
        return self._escalate_teleop(gap)

    # ── Verdict builders ──

    def _absorb(self, gap: PhysicalCapabilityGap, deviation: float) -> AdaptationVerdict:
        return AdaptationVerdict.create(
            "absorb",
            self._capability(gap),
            knowledge=(
                f"Operations for affordance '{gap.affordance}' on device "
                f"'{gap.device_id}' succeed {gap.success_rate:.0%} of the time, "
                f"marginally below the {self._precision_threshold:.0%} precision bar; "
                "the deviation is within the tolerance the verifier can safely widen."
            ),
            rationale=(
                "Widening the verifier tolerance is the cheapest correct adaptation "
                "for a marginal, in-band deviation; no new code or strategy is warranted."
            ),
            confidence=self._evidence_confidence(gap),
            expected_effect="widen the verifier tolerance for the affected channel",
            target_affordance=gap.affordance,
        )

    def _rebind(self, gap: PhysicalCapabilityGap, candidate: str) -> AdaptationVerdict:
        return AdaptationVerdict.create(
            "rebind",
            self._capability(gap),
            knowledge=(
                f"An alternative inference strategy '{candidate}' is registered for "
                f"affordance '{gap.affordance}' and should serve device "
                f"'{gap.device_id}', where the current policy underperforms "
                f"({gap.success_rate:.0%} success)."
            ),
            rationale=(
                "A registered alternative strategy already covers this affordance, "
                "so switching to it is cheaper than acquiring new code."
            ),
            confidence=self._evidence_confidence(gap),
            target=candidate,
            expected_effect=f"route '{gap.affordance}' through strategy '{candidate}'",
            target_affordance=gap.affordance,
        )

    def _acquire(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict:
        return AdaptationVerdict.create(
            "acquire",
            self._capability(gap),
            knowledge=(
                f"No installed policy performs affordance '{gap.affordance}' on device "
                f"'{gap.device_id}', and a downloadable policy candidate exists that "
                "could fill this gap."
            ),
            rationale=(
                "No installed strategy covers this manipulation and a candidate policy "
                "is downloadable; acquiring is the only path that fills the gap."
            ),
            confidence=self._evidence_confidence(gap),
            max_risk_level=_ACQUIRE_RISK_LEVEL,  # type: ignore[arg-type]
            expected_effect=f"download and register a policy for '{gap.affordance}'",
            target_affordance=gap.affordance,
        )

    def _escalate_missing_device(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict:
        return AdaptationVerdict.create(
            "escalate",
            self._capability(gap),
            knowledge=(
                f"No device of the type required for affordance '{gap.affordance}' is "
                f"currently connected (device '{gap.device_id}' is unavailable); the "
                "manipulation cannot proceed without hardware."
            ),
            rationale=(
                "The framework cannot connect hardware on its own; a human must attach "
                "the required device."
            ),
            confidence=1.0,
            target=f"connect the hardware required for '{gap.affordance}'",
            target_affordance=gap.affordance,
        )

    def _escalate_degraded(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict:
        return AdaptationVerdict.create(
            "escalate",
            self._capability(gap),
            knowledge=(
                f"Device '{gap.device_id}' is connected but its probes/reads are "
                "failing; it needs physical inspection before automated operation "
                "resumes."
            ),
            rationale=(
                "A degraded device is a physical fault the framework cannot repair; "
                "a human must inspect it."
            ),
            confidence=1.0,
            target=f"inspect device '{gap.device_id}'",
            target_affordance=gap.affordance,
        )

    def _escalate_teleop(self, gap: PhysicalCapabilityGap) -> AdaptationVerdict:
        return AdaptationVerdict.create(
            "escalate",
            self._capability(gap),
            knowledge=(
                f"No cheaper adaptation resolves the '{gap.affordance}' gap on device "
                f"'{gap.device_id}'; a human operator must teleoperate to complete "
                "the task."
            ),
            rationale=(
                "Every cheaper verdict (absorb, rebind, acquire) was exhausted; "
                "teleoperation is the safe fallback."
            ),
            confidence=self._evidence_confidence(gap),
            target=f"teleoperate to complete '{gap.affordance}'",
            target_affordance=gap.affordance,
        )

    # ── Helpers ──

    def _find_rebind_candidate(self, gap: PhysicalCapabilityGap) -> str | None:
        """Look for an alternative inference strategy for the same affordance.

        Prefers a strategy that declares the affordance explicitly; falls back
        to an id substring match when strategies declare no affordances.
        """
        registry = self._inference_registry
        if registry is None:
            return None
        affordance = gap.affordance.strip().lower()
        if not affordance:
            return None
        list_strategies = getattr(registry, "list_strategies", None)
        if not callable(list_strategies):
            return None
        try:
            summaries = list_strategies()
        except Exception:  # noqa: BLE001 – a lookup failure is not a resolver failure
            return None

        get = getattr(registry, "get", None)
        substring_match: str | None = None
        for entry in summaries or ():
            strategy_id = str((entry or {}).get("strategy_id") or "")
            if not strategy_id:
                continue
            declared = self._declared_affordances(get, strategy_id)
            if declared:
                if affordance in declared:
                    return strategy_id
                continue
            if substring_match is None and affordance in strategy_id.lower():
                substring_match = strategy_id
        return substring_match

    @staticmethod
    def _declared_affordances(get: Any, strategy_id: str) -> set[str]:
        """Return the affordances a strategy declares, or an empty set."""
        if not callable(get):
            return set()
        try:
            strategy = get(strategy_id)
        except Exception:  # noqa: BLE001
            return set()
        declared = getattr(strategy, "affordances", ()) if strategy is not None else ()
        return {str(a).strip().lower() for a in (declared or ()) if str(a).strip()}

    def _can_acquire(self, gap: PhysicalCapabilityGap) -> bool:
        """Check whether a policy for this affordance could be downloaded.

        Best-effort: an injected ``acquire_probe`` decides when provided;
        otherwise the Hugging Face Hub is queried.  ``huggingface_hub`` is an
        optional dependency, so an unavailable hub (import error or query
        failure) yields ``False`` rather than raising.
        """
        if self._acquire_probe is not None:
            try:
                return bool(self._acquire_probe(gap))
            except Exception:  # noqa: BLE001 – a probe failure is a negative answer
                return False
        affordance = gap.affordance.strip()
        if not affordance:
            return False
        try:
            from huggingface_hub import HfApi  # optional dependency
        except ImportError:
            return False
        try:
            api = HfApi()
            results = api.list_models(search=affordance, limit=1)
            return any(True for _ in results)
        except Exception:  # noqa: BLE001 – network/hub failure ⇒ cannot acquire
            return False

    def _capability(self, gap: PhysicalCapabilityGap) -> str:
        """Build a valid dotted capability name for the gap.

        Shaped ``physical.<device>[.<affordance>]`` so it satisfies
        ``is_capability_name`` (2–4 lowercase dotted segments).
        """
        device = _slug_segment(gap.device_id, "device")
        affordance = _slug_segment(gap.affordance, "")
        if affordance:
            return f"physical.{device}.{affordance}"
        return f"physical.{device}"

    def _evidence_confidence(self, gap: PhysicalCapabilityGap) -> float:
        """Confidence in the verdict, scaled by how much evidence supports it."""
        if gap.sample_count <= 0:
            return 0.0
        return round(min(1.0, gap.sample_count / (self._min_samples * 2.0)), 3)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Coerce an optional sequence into a tuple of non-empty strings."""
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value if str(item))
    except TypeError:
        return ()
