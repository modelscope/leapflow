# Copyright (c) Alibaba, Inc. and its affiliates.
"""Frame lifecycle: install a measured candidate, or vouch for a frame by a check.

Two ways to make a frame usable without measuring it again from scratch:

* **Promotion** installs an estimate some calibration already measured and
  indexed (``calib_candidates``) but never activated -- a stationary bridge's
  shadow, for instance. Which stored shapes are candidates of which frame is
  answered by :class:`CandidateDecoder` plugins keyed by the stored schema, so a
  new estimation method publishes its own decoder under
  ``leapflow.hardware.frame_candidates`` instead of teaching this module its
  format. The installed record embeds the candidate it came from, which is the
  evidence ``provenance.CandidatePromotionEvidence`` checks.
* **Attestation** compares a frame against an independently verified reference
  frame through both cameras' simultaneous view of one stationary target. It
  commands no motion. A pass writes ``<frame>.verify`` bound to the exact
  version it checked; a disagreement larger than the frame's required accuracy
  writes a failed ``<frame>.attestation`` (the frame then reads as drifted); an
  agreement the reference is too coarse to vouch for writes nothing.

Neither path knows an arm model or a camera name: frames, parents, roles, and
accuracies all come from the workcell declaration.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.hardware.skills.frame_status import Accuracy, FrameCandidate, FrameStatus
from leapflow.hardware.skills.frames import attestation_procedure, verify_procedure
from leapflow.hardware.skills.geometry import Pose, Transform, norm, rotation_distance_deg, sub
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    SIDE_EFFECT_COMMITTED,
    SkillError,
    SkillPlan,
)
from leapflow.hardware.skills.provenance import (
    ACHIEVED_ACCURACY,
    METHOD_BRIDGE_SHADOW,
    METHOD_CANDIDATE_PROMOTION,
)
from leapflow.hardware.skills.workcell import DEPENDS_ON, WorkcellSpec

logger = logging.getLogger(__name__)

CANDIDATE_ENTRY_POINT_GROUP = "leapflow.hardware.frame_candidates"
METHOD_CROSS_VIEW = "cross_view"
PARENT_BASE = "base"
PARENT_FLANGE = "flange"
MAX_CANDIDATES = 32
_GRID = (0.15, 0.5, 0.85)  # board points the two views are compared at, as board fractions
_MATRIX_TOLERANCE = 1e-6
_UNREVIVABLE = (FrameStatus.MISSING, FrameStatus.DEPENDENCY_SUPERSEDED)


# ── Candidates ───────────────────────────────────────────────────────────────


@runtime_checkable
class CandidateDecoder(Protocol):
    """Reads one stored candidate schema as an estimate of one declared frame."""

    schema: str

    def decode(
        self, identity: str, value: Mapping[str, Any], created_at: float, spec: WorkcellSpec, store: Any
    ) -> FrameCandidate | None:
        """The frame candidate *value* is, or None when it is not admissible here."""
        ...


class BridgeShadowCandidates:
    """A stationary bridge's shadow: a base-to-fixed-camera estimate, once it passed."""

    schema = "leapflow.bridge_shadow.v1"

    def decode(
        self, identity: str, value: Mapping[str, Any], created_at: float, spec: WorkcellSpec, store: Any
    ) -> FrameCandidate | None:
        from leapcalib.contracts.identity import digest_of
        from leapcalib.geometry.se3 import Transform as SiTransform
        from leapflow.hardware.calibration.bridge_assessment import shadow_problem
        from leapflow.hardware.calibration.bridge_policy import BridgePolicy

        if value.get("workcell_id") != spec.workcell_id or shadow_problem(value):
            return None
        evaluation = value["evaluation"]
        if evaluation.get("verdict") != "numerical_pass" or evaluation.get("candidate") is None:
            return None
        try:
            policy = BridgePolicy.from_mapping(spec.calibration_bridge)
        except (TypeError, ValueError):
            return None
        fixed, wrist = policy.fixed_role, policy.wrist_role
        if fixed not in spec.frames or spec.frames[fixed].parent != PARENT_BASE or wrist not in spec.frames:
            return None
        dependency = value["dependency"]
        current = store.latest(spec.workcell_id, spec.frames[wrist].procedure) if store is not None else None
        if current is None or digest_of(vars(current)) != dependency.get("record_digest"):
            return None  # the wrist mount the estimate was solved through has been replaced
        transform = SiTransform.from_mapping(evaluation["candidate"])
        matrix = tuple(
            tuple(float(v) * (1000.0 if column == 3 and row < 3 else 1.0) for column, v in enumerate(cells))
            for row, cells in enumerate(transform.to_matrix())
        )
        translation_m, rotation_rad = evaluation["uncertainty_m_rad"]
        return FrameCandidate(
            digest=identity,
            frame=fixed,
            method=METHOD_BRIDGE_SHADOW,
            matrix=matrix,
            achieved=Accuracy(float(translation_m) * 1000.0, math.degrees(float(rotation_rad))),
            created_at=created_at,
            dependency={wrist: float(current.recorded_at)},
        )


BUILTIN_DECODERS: tuple[CandidateDecoder, ...] = (BridgeShadowCandidates(),)
_discovered: tuple[CandidateDecoder, ...] | None = None


def discover_decoders() -> tuple[CandidateDecoder, ...]:
    """Builtin decoders, then those advertised under ``leapflow.hardware.frame_candidates``."""
    global _discovered
    if _discovered is None:
        from importlib.metadata import entry_points

        found: list[CandidateDecoder] = list(BUILTIN_DECODERS)
        for entry in entry_points(group=CANDIDATE_ENTRY_POINT_GROUP):
            try:
                loaded = entry.load()()
                for decoder in (loaded,) if isinstance(loaded, CandidateDecoder) else loaded:
                    if isinstance(decoder, CandidateDecoder):
                        found.append(decoder)
            except Exception as exc:  # noqa: BLE001 - a broken plugin offers nothing, never stops the node
                logger.warning("frame candidate entry point %r failed to load: %s", entry.name, exc)
        _discovered = tuple(found)
    return _discovered


def admissible_candidates(
    registry: Any, spec: WorkcellSpec, decoders: Iterable[CandidateDecoder] | None = None
) -> dict[str, tuple[FrameCandidate, ...]]:
    """Indexed, uninstalled estimates of *spec*'s frames, newest first per frame.

    Read-only and contained: a listing must not fail because a candidate index
    is unreadable, so any failure reads as "no candidates" and is logged.
    """
    generations = getattr(registry, "calibration_generations", None)
    if generations is None or not hasattr(generations, "candidates"):
        return {}
    store = getattr(registry, "calibration_store", None)
    by_schema = {decoder.schema: decoder for decoder in (discover_decoders() if decoders is None else decoders)}
    found: dict[str, list[FrameCandidate]] = {}
    try:
        rows = generations.candidates(spec.workcell_id, limit=MAX_CANDIDATES)
    except Exception as exc:  # noqa: BLE001 - an unreadable index offers nothing
        logger.warning("frame candidates of %s are unreadable: %s", spec.workcell_id, exc, exc_info=True)
        return {}
    for identity, value, created_at in rows:
        decoder = by_schema.get(str(value.get("schema", ""))) if isinstance(value, Mapping) else None
        if decoder is None:
            continue
        try:
            candidate = decoder.decode(identity, value, created_at, spec, store)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.warning("candidate %s is not decodable: %s", identity[:12], exc, exc_info=True)
            continue
        if candidate is not None and candidate.frame in spec.frames:
            found.setdefault(candidate.frame, []).append(candidate)
    return {frame: tuple(values) for frame, values in found.items()}


# ── Port ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Chain:
    """How a frame reaches the base: directly, or through the arm's current flange."""

    frame: str
    parent: str
    mount: Transform

    def to_base(self, flange: Pose | None) -> Transform:
        if self.parent == PARENT_BASE:
            return self.mount
        if flange is None:
            raise SkillError(FAILURE_PLAN_STALE, f"{self.frame} rides the arm, but no flange pose was read")
        return Transform.from_pose(flange).compose(self.mount)


class WorkcellFrameLifecycle:
    """Promotion and attestation for one run context. Satisfies ``FrameLifecyclePort``."""

    def __init__(self, ctx: Any, registry: Any) -> None:
        self._ctx = ctx
        self._registry = registry

    # Promotion -----------------------------------------------------------------

    async def promotion_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        frame = self._frame(args.get("frame"))
        candidate = self._candidate(frame, str(args.get("candidate_digest") or ""))
        diagnosis = self._ctx.frames.diagnosis(frame)
        required = self._ctx.frames.required_accuracy(frame)
        predicted = (
            FrameStatus.VALID
            if candidate.achieved is None or required is None or candidate.achieved.within(required)
            else FrameStatus.INSUFFICIENT_ACCURACY
        )
        return {
            "frame": frame,
            "procedure": self._ctx.frames.procedure(frame),
            "candidate": {**candidate.to_dict(), "matrix": [list(row) for row in candidate.matrix]},
            "replaces": {"version": diagnosis.version, "status": diagnosis.status.value},
            "required": required.to_dict() if required else None,
            "predicted_status": predicted.value,
        }

    async def promote(self, plan: SkillPlan) -> Mapping[str, Any]:
        basis = plan.basis.get("promotion", {})
        frame = self._frame(basis.get("frame"))
        approved = basis.get("candidate", {})
        candidate = self._candidate(frame, str(approved.get("digest", "")))
        if not _same_matrix(candidate.matrix, approved.get("matrix")):
            raise SkillError(FAILURE_PLAN_STALE, "the candidate changed since it was approved; plan again")
        if self._ctx.frames.diagnosis(frame).version != basis.get("replaces", {}).get("version"):
            raise SkillError(FAILURE_PLAN_STALE, f"{frame} was recalibrated since planning; plan again")
        matrix = [list(row) for row in candidate.matrix]
        parameters: dict[str, Any] = {
            "method": METHOD_CANDIDATE_PROMOTION,
            "camera_role": frame,
            "mounting": _mounting(self._ctx.frames.parent(frame)),
            "units": "mm",
            "promotion": {
                "candidate_digest": candidate.digest,
                "source_method": candidate.method,
                "matrix": matrix,
                "candidate_created_at": candidate.created_at,
            },
            DEPENDS_ON: dict(candidate.dependency),
        }
        if candidate.achieved is not None:
            parameters[ACHIEVED_ACCURACY] = candidate.achieved.to_dict()
        recorded_at = self._ctx.record_calibration(
            basis["procedure"], parameters=parameters, matrix=matrix,
            notes=f"promoted candidate {candidate.digest[:12]}",
        )
        return {
            "frame": frame,
            "recorded_at": recorded_at,
            "candidate_digest": candidate.digest,
            "achieved": candidate.achieved.to_dict() if candidate.achieved else None,
            "predicted_status": basis.get("predicted_status"),
        }

    # Attestation ---------------------------------------------------------------

    async def attestation_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        frame, reference = self._frame(args.get("frame")), self._frame(args.get("reference"))
        if frame == reference:
            raise SkillError(FAILURE_INVALID_ARGS, "a frame cannot attest itself; name another camera as reference")
        subject, version = self._subject(frame)
        anchor = self._reference(reference, self._ctx.frames.required_accuracy(frame))
        flange = await self._flange(subject, anchor)
        preview = await self._compare(subject, anchor, flange, samples=1)
        required = self._ctx.frames.required_accuracy(frame)
        bound = self._bound(preview, reference)
        return {
            "frame": frame,
            "reference": reference,
            "version": version,
            "reference_version": self._ctx.frames.versions[reference],
            "samples": int(args.get("samples", 3)),
            "flange": flange.to_dict() if flange is not None else None,
            "required": required.to_dict() if required else None,
            "preview": {**_summary(preview), "bound": bound.to_dict()},
        }

    async def attest(self, plan: SkillPlan) -> Mapping[str, Any]:
        basis = plan.basis.get("attestation", {})
        frame, reference = self._frame(basis.get("frame")), self._frame(basis.get("reference"))
        subject, version = self._subject(frame)
        anchor = self._reference(reference, self._ctx.frames.required_accuracy(frame))
        if version != basis.get("version") or self._ctx.frames.versions.get(reference) != basis.get("reference_version"):
            raise SkillError(FAILURE_PLAN_STALE, "a frame was recalibrated since planning; plan again")
        start = Pose.from_mapping(basis["flange"]) if basis.get("flange") else None
        await self._still(start)
        measured = await self._compare(subject, anchor, start, samples=int(basis["samples"]))
        await self._still(start)
        required = self._ctx.frames.required_accuracy(frame)
        stats = _summary(measured)
        disagreement = Accuracy(stats["p95_mm"], stats["rotation_p95_deg"])
        bound = self._bound(measured, reference)
        depends = {frame: version, reference: basis["reference_version"]}
        common = {
            **stats,
            "method": METHOD_CROSS_VIEW,
            "camera_role": frame,
            "reference": reference,
            "errors_mm": [round(e, 3) for e in measured.translation_mm],
            "samples": int(basis["samples"]),
            DEPENDS_ON: depends,
        }
        if required is not None and not disagreement.within(required):
            reason = (
                f"{frame} and {reference} disagree by {disagreement.describe()} at p95 "
                f"(required {required.describe()})"
            )
            self._ctx.record_calibration(
                attestation_procedure(frame),
                parameters={**common, "passed": False, "reason": reason},
            )
            raise SkillError(
                FAILURE_CHECK_FAILED,
                f"{reason}; {frame} has moved or was mis-measured -- recalibrate it",
                side_effect_state=SIDE_EFFECT_COMMITTED,
                result=common,
            )
        if required is not None and not bound.within(required):
            raise SkillError(
                FAILURE_CHECK_FAILED,
                f"{frame} agrees with {reference} to {disagreement.describe()}, but {reference}'s own "
                f"accuracy leaves a bound of {bound.describe()} (required {required.describe()}); "
                f"tighten {reference} first or calibrate {frame} directly",
                result=common,
            )
        recorded_at = self._ctx.record_calibration(
            verify_procedure(frame),
            parameters={
                **common,
                "passed": True,
                "training_disjoint": True,
                "points": len(measured.translation_mm),
                ACHIEVED_ACCURACY: bound.to_dict(),
            },
        )
        return {**stats, "frame": frame, "reference": reference, "recorded_at": recorded_at,
                "achieved": bound.to_dict()}

    # Helpers -------------------------------------------------------------------

    def _frame(self, value: Any) -> str:
        frame = str(value or "")
        if not frame or not self._ctx.workcell.role(frame) or not self._ctx.frames.procedure(frame):
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"workcell {self._ctx.workcell.workcell_id!r} declares no calibrated device frame {frame!r}",
            )
        if self._ctx.frames.parent(frame) not in (PARENT_BASE, PARENT_FLANGE):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"{frame} must be expressed in base or flange")
        return frame

    def _candidate(self, frame: str, digest: str) -> FrameCandidate:
        offered = self._ctx.frames.candidates(frame)
        chosen = next((c for c in offered if c.digest == digest), None) if digest else (offered[0] if offered else None)
        if chosen is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"no admissible {frame} candidate" + (f" {digest[:12]}" if digest else "")
                + " is indexed against the current calibrations",
            )
        return chosen

    def _subject(self, frame: str) -> tuple[_Chain, float]:
        """The frame's latest record as measured -- even when it does not hold yet."""
        diagnosis = self._ctx.frames.diagnosis(frame)
        if diagnosis.status in _UNREVIVABLE or diagnosis.version is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"{frame} has no current record to attest ({diagnosis.status.value}); calibrate it",
            )
        store = getattr(self._registry, "calibration_store", None)
        record = store.latest(self._ctx.workcell.workcell_id, self._ctx.frames.procedure(frame)) if store else None
        if record is None or abs(float(record.recorded_at) - diagnosis.version) > _MATRIX_TOLERANCE:
            raise SkillError(FAILURE_PLAN_STALE, f"{frame} changed while planning; plan again")
        try:
            mount = (
                Transform.from_matrix(record.matrix)
                if record.matrix
                else Transform.from_pose(Pose.from_mapping(record.pose or {}))
            )
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"{frame}'s record is malformed: {exc}") from exc
        return _Chain(frame, self._ctx.frames.parent(frame), mount), float(record.recorded_at)

    def _reference(self, reference: str, required: Accuracy | None = None) -> _Chain:
        mount = self._ctx.frames.transform(reference)
        diagnosis = self._ctx.frames.diagnosis(reference)
        accuracy = diagnosis.achieved
        if mount is None or not diagnosis.ok:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"reference {reference} does not hold; verify it first")
        if (accuracy is None or not math.isfinite(accuracy.translation_mm)
                or accuracy.translation_mm <= 0 or not diagnosis.verified_by):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE,
                             f"reference {reference} needs independent evidence with a known positive accuracy bound; verify it first")
        if required is not None and required.rotation_deg is not None and accuracy.rotation_deg is None:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE,
                             f"reference {reference} has unknown rotational accuracy; independently verify its rotation first")
        return _Chain(reference, self._ctx.frames.parent(reference), mount)

    def _bound(self, measured: "_Comparison", reference: str) -> Accuracy:
        stats = _summary(measured)
        disagreement = Accuracy(stats["p95_mm"], stats["rotation_p95_deg"])
        self._reference(reference)  # never substitute zero for unknown reference error
        return disagreement.plus(self._ctx.frames.diagnosis(reference).achieved)

    async def _flange(self, *chains: _Chain) -> Pose | None:
        """The arm's flange when a compared frame rides it; None when both sit on the base."""
        if all(chain.parent == PARENT_BASE for chain in chains):
            return None
        return (await self._ctx.motion.state()).flange

    async def _still(self, start: Pose | None) -> None:
        if start is None:
            return
        current = (await self._ctx.motion.state()).flange
        limits = self._ctx.workcell.calibration
        moved_mm, moved_deg = current.distance_mm(start), current.rotation_to_deg(start)
        if moved_mm > limits["start_drift_mm"] or moved_deg > limits["start_drift_deg"]:
            raise SkillError(
                FAILURE_PLAN_STALE,
                f"the arm moved {moved_mm:.1f} mm / {moved_deg:.1f} deg; hold it still and plan again",
            )

    async def _compare(
        self, subject: _Chain, anchor: _Chain, flange: Pose | None, *, samples: int
    ) -> "_Comparison":
        from leapflow.hardware.skills.vision import CharucoTarget

        try:
            target = CharucoTarget.from_mapping(self._ctx.workcell.calibration_target)
        except (TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, f"attestation needs the declared calibration_target: {exc}"
            ) from exc
        views = (self._ctx.camera_vision(subject.frame), self._ctx.camera_vision(anchor.frame))
        width, height = target.squares_x * target.square_mm, target.squares_y * target.square_mm
        points = [(fx * width, fy * height, 0.0) for fy in _GRID for fx in _GRID]
        comparison = _Comparison([], [])
        for _ in range(max(1, samples)):
            seen, known = await asyncio.gather(views[0].observe_target(), views[1].observe_target())
            board_a = subject.to_base(flange).compose(seen.transform)
            board_b = anchor.to_base(flange).compose(known.transform)
            comparison.translation_mm.extend(norm(sub(board_a.apply(p), board_b.apply(p))) for p in points)
            comparison.rotation_deg.append(rotation_distance_deg(board_a.rotation, board_b.rotation))
        return comparison


@dataclass
class _Comparison:
    translation_mm: list[float]
    rotation_deg: list[float]


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0


def _summary(measured: _Comparison) -> dict[str, float]:
    errors = measured.translation_mm
    rms = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
    return {
        "p95_mm": round(_p95(errors), 3),
        "rms_mm": round(rms, 3),
        "max_mm": round(max(errors, default=0.0), 3),
        "rotation_p95_deg": round(_p95(measured.rotation_deg), 4),
    }


def _mounting(parent: str) -> str:
    return "eye_to_hand" if parent == PARENT_BASE else "eye_in_hand"


def _same_matrix(left: Sequence[Sequence[float]], right: Any) -> bool:
    if not isinstance(right, Sequence) or len(right) != len(left):
        return False
    try:
        return all(
            abs(float(a) - float(b)) <= _MATRIX_TOLERANCE
            for row_a, row_b in zip(left, right, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "BUILTIN_DECODERS",
    "CANDIDATE_ENTRY_POINT_GROUP",
    "METHOD_CROSS_VIEW",
    "BridgeShadowCandidates",
    "CandidateDecoder",
    "WorkcellFrameLifecycle",
    "admissible_candidates",
    "discover_decoders",
]
