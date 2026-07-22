"""S3-L2: offline difficulty calibration analysis (report-only).

Consumes the adaptive-depth learning signals captured per turn (S3-L1) from the
evolution episode store and relates the *predicted* difficulty to the *actual*
effort and outcome. Produces a bounded, report-only calibration suggestion; it
never mutates runtime weights (that is S3-L3's online step, gated + reviewed).

Core question: "Does the difficulty signal correctly predict how much effort a
turn needs?" If high-difficulty turns do not cost more (over-prediction) or
low-difficulty turns fail / cost a lot (under-prediction), the difficulty
sensitivity weight should be scaled. This module only *reports* the suggestion.

All analysis is pure and hermetic; ``build_calibration_report_from_store`` is a
thin offline consumer that reads persisted episodes on demand (never in the
hot loop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_SUCCESS_OUTCOMES = frozenset({"success", "completed"})
_LOW_MAX = 0.34
_HIGH_MIN = 0.66
_SCALE_MIN = 0.5
_SCALE_MAX = 1.5
_MIN_SAMPLES = 3
_CONFIDENCE_FULL_AT = 50.0


@dataclass(frozen=True)
class DifficultyBucket:
    """Aggregate effort/success for one predicted-difficulty band."""

    label: str
    sample_size: int
    avg_effort: float
    success_rate: float


@dataclass(frozen=True)
class DifficultyCalibrationReport:
    """Report-only calibration analysis over captured turn signals.

    ``suggested_weight_scale`` is a bounded multiplier (``[0.5, 1.5]``) for the
    difficulty sensitivity; ``1.0`` means "well calibrated, no change". It is a
    suggestion only — applying it is S3-L3's gated online step.
    """

    sample_size: int
    buckets: List[DifficultyBucket] = field(default_factory=list)
    effort_monotonic: bool = True
    suggested_weight_scale: float = 1.0
    confidence: float = 0.0
    rationale: str = "insufficient data"

    def summary(self) -> Dict[str, Any]:
        """Flat dict for logging / CLI / dashboard surfacing."""
        return {
            "sample_size": self.sample_size,
            "effort_monotonic": self.effort_monotonic,
            "suggested_weight_scale": round(self.suggested_weight_scale, 3),
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "buckets": [
                {
                    "label": b.label,
                    "n": b.sample_size,
                    "avg_effort": round(b.avg_effort, 2),
                    "success_rate": round(b.success_rate, 3),
                }
                for b in self.buckets
            ],
        }


def _effort(record: Dict[str, Any]) -> Optional[float]:
    """Actual effort proxy: explicit round count, else API calls, else actions."""
    ctx = record.get("context") or {}
    if "steps" in ctx:
        try:
            return float(ctx["steps"])
        except (TypeError, ValueError):
            pass
    if "api_retries" in ctx:  # api_calls == api_retries + 1 (see TurnUsageTracker)
        try:
            return float(ctx["api_retries"]) + 1.0
        except (TypeError, ValueError):
            pass
    actions = record.get("actions") or []
    if actions:
        return float(len(actions))
    return None


def _is_success(record: Dict[str, Any]) -> bool:
    try:
        if float(record.get("reward", 0.0)) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    return str(record.get("outcome", "")).lower() in _SUCCESS_OUTCOMES


def _difficulty(record: Dict[str, Any]) -> Optional[float]:
    val = (record.get("context") or {}).get("final_difficulty")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def analyze_difficulty_calibration(
    records: List[Dict[str, Any]],
) -> DifficultyCalibrationReport:
    """Relate predicted difficulty to actual effort/outcome (report-only).

    ``records`` are evolution episodes (as returned by the evolution store):
    each may carry the S3-L1 signal under ``context.final_difficulty`` plus an
    effort proxy and an outcome/reward. Episodes lacking the signal are skipped.
    """
    usable: List[tuple[float, float, bool]] = []
    for record in records:
        difficulty = _difficulty(record)
        effort = _effort(record)
        if difficulty is None or effort is None:
            continue
        usable.append((difficulty, effort, _is_success(record)))

    sample_size = len(usable)
    if sample_size < _MIN_SAMPLES:
        return DifficultyCalibrationReport(sample_size=sample_size)

    bands: Dict[str, List[tuple[float, bool]]] = {"low": [], "medium": [], "high": []}
    for difficulty, effort, ok in usable:
        band = "low" if difficulty < _LOW_MAX else ("high" if difficulty >= _HIGH_MIN else "medium")
        bands[band].append((effort, ok))

    buckets: List[DifficultyBucket] = []
    for label in ("low", "medium", "high"):
        items = bands[label]
        if not items:
            buckets.append(DifficultyBucket(label, 0, 0.0, 0.0))
            continue
        avg_effort = sum(effort for effort, _ in items) / len(items)
        success_rate = sum(1 for _, ok in items if ok) / len(items)
        buckets.append(DifficultyBucket(label, len(items), avg_effort, success_rate))

    by_label = {b.label: b for b in buckets}
    low, high = by_label["low"], by_label["high"]

    populated = [b for b in buckets if b.sample_size > 0]
    efforts = [b.avg_effort for b in populated]
    monotonic = all(efforts[i] <= efforts[i + 1] + 1e-9 for i in range(len(efforts) - 1))

    scale = 1.0
    rationale = "difficulty tracks effort; no adjustment suggested"
    if high.sample_size > 0 and low.sample_size > 0 and high.avg_effort <= low.avg_effort:
        scale = 0.8
        rationale = "high-difficulty turns cost no more than low ones (over-predicted): reduce sensitivity"
    elif low.sample_size > 0 and (
        low.success_rate < 0.5
        or (high.sample_size > 0 and low.avg_effort >= high.avg_effort)
    ):
        scale = 1.2
        rationale = "low-difficulty turns fail or cost like hard ones (under-predicted): raise sensitivity"
    elif not monotonic:
        scale = 0.9
        rationale = "difficulty is a noisy effort predictor: slightly reduce sensitivity"

    scale = max(_SCALE_MIN, min(_SCALE_MAX, scale))
    confidence = round(min(1.0, sample_size / _CONFIDENCE_FULL_AT), 3)

    return DifficultyCalibrationReport(
        sample_size=sample_size,
        buckets=buckets,
        effort_monotonic=monotonic,
        suggested_weight_scale=scale,
        confidence=confidence,
        rationale=rationale,
    )


def build_calibration_report_from_store(
    store: Any, *, limit: int = 500,
) -> DifficultyCalibrationReport:
    """Load recent episodes from an evolution store and analyze calibration.

    Thin offline consumer bridging the persisted S3-L1 signals to the pure
    analyzer (for on-demand reporting; never called in the hot loop).
    """
    records = store.load_recent_episodes(limit=limit)
    return analyze_difficulty_calibration(records)


_CALIB_MIN_SAMPLES = 10
_CALIB_MIN_CONFIDENCE = 0.3
_CALIB_K_MIN = 0.25
_CALIB_K_MAX = 3.0


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of applying a calibration report to the difficulty weight.

    ``effective_k`` is always derived from ``baseline_k`` (never the previously
    calibrated value), so repeated recalibration cannot compound or drift and
    resetting to baseline is exact.
    """

    baseline_k: float
    effective_k: float
    applied: bool
    reason: str


def apply_calibration(
    baseline_k: float,
    report: DifficultyCalibrationReport,
    *,
    enabled: bool,
    min_confidence: float = _CALIB_MIN_CONFIDENCE,
    min_samples: int = _CALIB_MIN_SAMPLES,
    k_min: float = _CALIB_K_MIN,
    k_max: float = _CALIB_K_MAX,
) -> CalibrationResult:
    """S3-L3: turn a calibration report into a bounded, gated weight adjustment.

    Applies ``report.suggested_weight_scale`` to ``baseline_k`` only when
    calibration is enabled, there is enough evidence (samples + confidence), and
    an adjustment is actually suggested. The result is clamped to ``[k_min,
    k_max]``. Pure and side-effect free — the caller decides whether to install
    ``effective_k`` (and can revert to ``baseline_k`` at any time).
    """
    if not enabled:
        return CalibrationResult(baseline_k, baseline_k, False, "calibration disabled")
    if report.sample_size < min_samples:
        return CalibrationResult(
            baseline_k, baseline_k, False,
            f"insufficient samples ({report.sample_size} < {min_samples})",
        )
    if report.confidence < min_confidence:
        return CalibrationResult(
            baseline_k, baseline_k, False,
            f"confidence too low ({report.confidence:.2f} < {min_confidence:.2f})",
        )
    if report.suggested_weight_scale == 1.0:
        return CalibrationResult(baseline_k, baseline_k, False, "already well calibrated")
    effective = max(k_min, min(k_max, baseline_k * report.suggested_weight_scale))
    return CalibrationResult(baseline_k, round(effective, 4), True, report.rationale)


_PREMATURE_HIGH = 0.4
_PREMATURE_LOW = 0.1


@dataclass(frozen=True)
class ThresholdCalibrationReport:
    """S3-L4: report on finalize-posture threshold health (report-only).

    Exposes ``sample_size`` / ``confidence`` / ``suggested_weight_scale`` with the
    same shape as :class:`DifficultyCalibrationReport`, so it reuses
    :func:`apply_calibration` (with ratio bounds) unchanged.
    """

    sample_size: int
    premature_finalize_rate: float = 0.0
    suggested_weight_scale: float = 1.0
    confidence: float = 0.0
    rationale: str = "insufficient data"

    def summary(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "premature_finalize_rate": round(self.premature_finalize_rate, 3),
            "suggested_weight_scale": round(self.suggested_weight_scale, 3),
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
        }


def analyze_threshold_calibration(
    records: List[Dict[str, Any]],
) -> ThresholdCalibrationReport:
    """Detect premature finalization to tune the finalize posture threshold.

    A turn whose final posture is ``finalizing`` but whose outcome failed likely
    finalized too early. A high premature-finalize rate suggests raising the
    finalize threshold (converge later); a very low rate allows finalizing a
    touch earlier. Report-only — applying is gated via :func:`apply_calibration`.
    """
    finalizing = [
        r for r in records
        if str((r.get("context") or {}).get("final_posture", "")).lower() == "finalizing"
    ]
    n = len(finalizing)
    if n < _MIN_SAMPLES:
        return ThresholdCalibrationReport(sample_size=n)
    failures = sum(1 for r in finalizing if not _is_success(r))
    rate = failures / n
    scale = 1.0
    rationale = "finalize threshold healthy"
    if rate >= _PREMATURE_HIGH:
        scale = 1.1
        rationale = f"high premature-finalize rate ({rate:.2f}): raise finalize threshold (converge later)"
    elif rate <= _PREMATURE_LOW:
        scale = 0.95
        rationale = f"low premature-finalize rate ({rate:.2f}): allow slightly earlier finalize"
    confidence = round(min(1.0, n / _CONFIDENCE_FULL_AT), 3)
    return ThresholdCalibrationReport(
        sample_size=n,
        premature_finalize_rate=round(rate, 3),
        suggested_weight_scale=scale,
        confidence=confidence,
        rationale=rationale,
    )


def build_threshold_report_from_store(
    store: Any, *, limit: int = 500,
) -> ThresholdCalibrationReport:
    """Load recent episodes and analyze finalize-threshold calibration (offline)."""
    return analyze_threshold_calibration(store.load_recent_episodes(limit=limit))
