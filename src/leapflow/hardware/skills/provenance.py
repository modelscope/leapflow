# Copyright (c) Alibaba, Inc. and its affiliates.
"""Provenance validators: does a frame record carry the evidence its method produces?

A mounted device's frame (a camera on a base or a flange) is checkable because
the device was observed from the poses it was fitted on, and that observation
leaves evidence behind. Which evidence depends on how it was measured, so the
check is a registry of validators keyed by the *shape* of a record's evidence
rather than by any procedure or skill name: a board fit names the target mount
it solved for, a board-free fit carries per-joint screw agreement, a promoted
candidate embeds the estimate it installed. A new measurement method ships its
own validator under the ``leapflow.hardware.provenance`` entry-point group --
no edit to the frame graph, and no name it has to match.

A record no validator recognises is judged by what it is: a mounted-device
frame that cannot prove how it was measured is unproven (it is what an import
or a hand-edited matrix leaves behind, and it survives the device being moved),
while a declared offset such as a TCP is the operator's statement and stands.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.hardware.skills.frame_status import Accuracy, FrameStatus
from leapflow.hardware.skills.protocol import METHOD_MARKERLESS_SCREW

logger = logging.getLogger(__name__)

PROVENANCE_ENTRY_POINT_GROUP = "leapflow.hardware.provenance"
METHOD_CANDIDATE_PROMOTION = "candidate_promotion"
METHOD_BRIDGE_SHADOW = "stationary_board_bridge.v1"
ACHIEVED_ACCURACY = "achieved_accuracy"
MIN_SCREW_PAIRS = 3  # two rotations share the vertical, so no mount comes out of them
_MATRIX_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ProvenanceVerdict:
    """What a record's own evidence proves about the frame it describes."""

    status: FrameStatus
    reason: str = ""
    method: str = ""
    achieved: Accuracy | None = None


@runtime_checkable
class ProvenanceValidator(Protocol):
    """Recognises one evidence shape and checks a record against it."""

    name: str

    def applies(self, parameters: Mapping[str, Any]) -> bool:
        """True when *parameters* carry (or claim) this validator's evidence."""
        ...

    def check(self, record: Any) -> ProvenanceVerdict:
        """Judge a record :meth:`applies` accepted."""
        ...


# ── Builtin validators ───────────────────────────────────────────────────────


class BridgeShadowEvidence:
    """A stationary-board bridge estimate is a shadow candidate, never an installed frame."""

    name = "bridge_shadow"

    def applies(self, parameters: Mapping[str, Any]) -> bool:
        return (
            parameters.get("method") == METHOD_BRIDGE_SHADOW
            or parameters.get("bridge_evidence") is not None
        )

    def check(self, record: Any) -> ProvenanceVerdict:
        from leapflow.hardware.calibration.bridge_assessment import shadow_problem

        reason = shadow_problem((record.parameters or {}).get("bridge_evidence"))
        return ProvenanceVerdict(
            FrameStatus.UNPROVEN,
            reason
            or "stationary bridge is a conditional shadow candidate; separate validated "
            "promotion is required",
            METHOD_BRIDGE_SHADOW,
        )


class CandidatePromotionEvidence:
    """A promoted candidate must install exactly the estimate it names."""

    name = "candidate_promotion"

    def applies(self, parameters: Mapping[str, Any]) -> bool:
        return parameters.get("method") == METHOD_CANDIDATE_PROMOTION

    def check(self, record: Any) -> ProvenanceVerdict:
        parameters = record.parameters or {}
        promotion = parameters.get("promotion")
        if not isinstance(promotion, Mapping):
            return _unproven("the promotion record names no candidate", self.name)
        digest = promotion.get("candidate_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            return _unproven("the promotion record names no candidate digest", self.name)
        if not _same_matrix(record.matrix, promotion.get("matrix")):
            return _unproven(
                "the installed matrix differs from the candidate it was promoted from", self.name
            )
        return ProvenanceVerdict(
            FrameStatus.VALID,
            method=self.name,
            achieved=Accuracy.from_mapping(parameters.get(ACHIEVED_ACCURACY)),
        )


class BoardTargetMountEvidence:
    """A board fit solves for the target mount; the record must name it."""

    name = "board_target_mount"

    def applies(self, parameters: Mapping[str, Any]) -> bool:
        return parameters.get("target_mount") is not None

    def check(self, record: Any) -> ProvenanceVerdict:
        parameters = record.parameters or {}
        if not _is_matrix(parameters.get("target_mount")):
            return _unproven("the record's measured target mount is malformed", self.name)
        return ProvenanceVerdict(
            FrameStatus.VALID,
            method=str(parameters.get("method") or self.name),
            achieved=Accuracy.from_mapping(parameters.get(ACHIEVED_ACCURACY)),
        )


class ScrewMotionEvidence:
    """A board-free fit carries at least three joints' screw agreement."""

    name = METHOD_MARKERLESS_SCREW

    def applies(self, parameters: Mapping[str, Any]) -> bool:
        return "screw_evidence" in parameters or parameters.get("method") == METHOD_MARKERLESS_SCREW

    def check(self, record: Any) -> ProvenanceVerdict:
        parameters = record.parameters or {}
        evidence = parameters.get("screw_evidence")
        if evidence is None:
            # What calib.screws wrote before it kept its evidence: the method is
            # named and the mount was fitted, but nothing left can be rechecked.
            return ProvenanceVerdict(
                FrameStatus.LEGACY_EVIDENCE,
                "the record names the markerless_screw method but predates the "
                "screw-motion evidence a record now carries",
                self.name,
            )
        reason = screw_evidence_problem(evidence)
        if reason:
            return _unproven(reason, self.name)
        return ProvenanceVerdict(
            FrameStatus.VALID,
            method=self.name,
            achieved=Accuracy.from_mapping(parameters.get(ACHIEVED_ACCURACY)),
        )


def screw_evidence_problem(evidence: Any) -> str:
    """Why a board-free record's own screw evidence is not a measurement; empty when it is."""
    if not isinstance(evidence, Mapping):
        return "the record carries no measured target mount and no screw-motion evidence"
    if evidence.get("method") != METHOD_MARKERLESS_SCREW:
        return "the record names no measurement method its evidence can be checked against"
    pairs = evidence.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        return "the record's screw-motion evidence lists no joints"
    if len(pairs) < MIN_SCREW_PAIRS:
        return (
            f"the record's screw-motion evidence covers {len(pairs)} joints "
            f"(need {MIN_SCREW_PAIRS})"
        )
    for pair in pairs:
        if not isinstance(pair, Mapping):
            return "the record's screw-motion evidence is malformed"
        values = [pair.get(key) for key in ("agreement", "shift_gap_mm", "angle_deg")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return "the record's screw-motion evidence is missing a joint's residuals"
        if not all(math.isfinite(value) for value in values):
            return "the record's screw-motion evidence is not finite"
    return ""


BUILTIN_VALIDATORS: tuple[ProvenanceValidator, ...] = (
    BridgeShadowEvidence(),
    CandidatePromotionEvidence(),
    BoardTargetMountEvidence(),
    ScrewMotionEvidence(),
)


# ── Registry ─────────────────────────────────────────────────────────────────


_discovered: tuple[ProvenanceValidator, ...] | None = None


def discover_validators() -> tuple[ProvenanceValidator, ...]:
    """Builtin validators, then those advertised under ``leapflow.hardware.provenance``.

    Each entry point is a zero-argument callable returning one validator or an
    iterable of them. One that fails to load is skipped with a warning; a
    record its method would have recognised then reads as unproven, which
    refuses rather than trusts.
    """
    global _discovered
    if _discovered is None:
        _discovered = (*BUILTIN_VALIDATORS, *_plugin_validators())
    return _discovered


def _plugin_validators() -> list[ProvenanceValidator]:
    from importlib.metadata import entry_points

    found: list[ProvenanceValidator] = []
    for entry in entry_points(group=PROVENANCE_ENTRY_POINT_GROUP):
        try:
            loaded = entry.load()()
            for validator in (loaded,) if isinstance(loaded, ProvenanceValidator) else loaded:
                if isinstance(validator, ProvenanceValidator):
                    found.append(validator)
                else:
                    logger.warning("provenance entry point %r returned a non-validator", entry.name)
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the node
            logger.warning("provenance entry point %r failed to load: %s", entry.name, exc)
    return found


def judge(
    record: Any,
    *,
    mounted: bool,
    validators: Iterable[ProvenanceValidator] | None = None,
) -> ProvenanceVerdict:
    """What *record*'s own evidence proves. *mounted* marks a frame that binds a device.

    The first validator whose evidence shape the record carries decides.
    """
    parameters = record.parameters or {}
    for validator in discover_validators() if validators is None else validators:
        try:
            if validator.applies(parameters):
                return validator.check(record)
        except Exception as exc:  # noqa: BLE001 - a broken validator refuses, never trusts
            logger.warning("provenance validator %r failed: %s", getattr(validator, "name", "?"), exc)
            return _unproven(f"its evidence could not be checked ({type(exc).__name__})", "")
    if not mounted:
        return ProvenanceVerdict(
            FrameStatus.VALID,
            method=str(parameters.get("method") or ""),
            achieved=Accuracy.from_mapping(parameters.get(ACHIEVED_ACCURACY)),
        )
    method = parameters.get("method")
    if isinstance(method, str) and method:
        return _unproven(f"no evidence rule is installed for its method {method!r}", method)
    return _unproven(
        "the record carries no measured target mount and no screw-motion evidence", ""
    )


def _unproven(reason: str, method: str) -> ProvenanceVerdict:
    return ProvenanceVerdict(FrameStatus.UNPROVEN, reason, method)


def _is_matrix(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return False
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            return False
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)) or not math.isfinite(cell):
                return False
    return True


def _same_matrix(left: Any, right: Any) -> bool:
    if not (_is_matrix(left) and _is_matrix(right)):
        return False
    return all(
        abs(float(a) - float(b)) <= _MATRIX_TOLERANCE
        for row_a, row_b in zip(left, right)
        for a, b in zip(row_a, row_b)
    )


__all__ = [
    "ACHIEVED_ACCURACY",
    "BUILTIN_VALIDATORS",
    "METHOD_BRIDGE_SHADOW",
    "METHOD_CANDIDATE_PROMOTION",
    "MIN_SCREW_PAIRS",
    "PROVENANCE_ENTRY_POINT_GROUP",
    "ProvenanceValidator",
    "ProvenanceVerdict",
    "discover_validators",
    "judge",
    "screw_evidence_problem",
]
