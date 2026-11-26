# Copyright (c) Alibaba, Inc. and its affiliates.
"""Make a measured frame usable without measuring it again: promote or attest.

``calib.promote`` installs an estimate some calibration already measured and
indexed as a candidate; ``calib.attest`` vouches for a frame's current record
by comparing it with an independently verified frame through both cameras'
simultaneous view of one stationary target. Neither commands motion; both
change what later skills may trust, so each run is consented.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.frame_status import Accuracy, FrameStatus
from leapflow.hardware.skills.ports import FrameLifecyclePort, SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    EFFECT_ACTUATE,
    FAILURE_SKILL_UNAVAILABLE,
    REQUIRE_CALIBRATION,
    REQUIRE_CANDIDATE,
    REQUIRE_RECORD,
    REQUIRE_ROLE,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

_CAMERA_FRAME = {"type": "string", "pattern": "^camera(\\.[a-z0-9_]+)*$"}
# Any bound device role: a candidate may estimate whichever frame its method measures.
_ROLE_FRAME = {"type": "string", "pattern": "^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)*$"}


def _port(ctx: SkillContext) -> FrameLifecyclePort:
    port = ctx.frame_lifecycle
    if not isinstance(port, FrameLifecyclePort):
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "this node has no frame lifecycle port")
    return port


class PromoteCandidate:
    """Install an indexed, admissible candidate estimate as a frame's calibration."""

    spec = SkillSpec(
        skill_id="calib.promote",
        version="1.0.0",
        card=(
            "Install an already measured, indexed candidate estimate of a device frame "
            "(for example a stationary bridge's shadow transform) as that frame's "
            "calibration. The candidate must still be solved through current calibrations; "
            "its measured uncertainty becomes the frame's accuracy. candidate_digest picks "
            "one candidate (default: the newest). No motion; each run is consented."
        ),
        args_schema={
            "type": "object",
            "required": ["frame"],
            "properties": {
                "frame": _ROLE_FRAME,
                "candidate_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        effect=EFFECT_ACTUATE,
        consent=CONSENT_EACH_RUN,
        motion_allowed=False,
        requires=(f"{REQUIRE_ROLE}{{frame}}", f"{REQUIRE_CANDIDATE}{{frame}}"),
        provides=(f"{REQUIRE_CALIBRATION}{{frame}}",),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await _port(ctx).promotion_basis(args)
        candidate = basis["candidate"]
        accuracy = candidate.get("achieved") or {}
        summary = (
            f"Install candidate {str(candidate['digest'])[:12]} ({candidate['method']}) as "
            f"{basis['frame']}'s calibration"
        )
        if accuracy:
            summary += f", accurate to {float(accuracy['translation_mm']):.1f} mm"
        if basis.get("predicted_status") != FrameStatus.VALID.value:
            summary += "; that is coarser than the workcell requires, so it will still need an independent check"
        return SkillPlan(
            skill_id=self.spec.skill_id, args=args, summary=summary + "; no motion",
            basis={"promotion": dict(basis)}, limits={"max_duration_s": 30.0},
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        result = await _port(ctx).promote(plan)
        return SkillOutcome.succeeded(
            result, message=f"{result['frame']} now uses candidate {str(result['candidate_digest'])[:12]}"
        )


class AttestFrame:
    """Vouch for a frame's current record against an independently verified frame."""

    spec = SkillSpec(
        skill_id="calib.attest",
        version="1.0.0",
        card=(
            "Check a device frame's current calibration against another, verified frame: "
            "both cameras observe the stationary calibration board at once and the board "
            "poses they imply in the base frame are compared over a grid of board points. "
            "A pass records an independent check bound to that exact calibration; a "
            "disagreement larger than the frame's required accuracy marks it drifted. "
            "The board must be visible to both cameras and the arm still. No motion; "
            "each run is consented."
        ),
        args_schema={
            "type": "object",
            "required": ["frame", "reference"],
            "properties": {
                "frame": _CAMERA_FRAME,
                "reference": _CAMERA_FRAME,
                "samples": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
        },
        effect=EFFECT_ACTUATE,
        consent=CONSENT_EACH_RUN,
        motion_allowed=False,
        requires=(
            f"{REQUIRE_ROLE}{{frame}}",
            f"{REQUIRE_ROLE}{{reference}}",
            f"{REQUIRE_CALIBRATION}{{reference}}",
        ),
        provides=(f"{REQUIRE_CALIBRATION}{{frame}}", f"{REQUIRE_RECORD}{{frame}}.verify"),
        provides_when=(
            FrameStatus.LEGACY_EVIDENCE.value,
            FrameStatus.UNPROVEN.value,
            FrameStatus.INSUFFICIENT_ACCURACY.value,
        ),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await _port(ctx).attestation_basis(args)
        preview = basis["preview"]
        required = Accuracy.from_mapping(basis["required"]) if basis.get("required") else None
        bound = Accuracy.from_mapping(preview["bound"])
        verdict = ""
        if required is not None:
            verdict = f"; it must meet {required.describe()}"
            if bound is not None and not bound.within(required):
                # The executor refuses on exactly this bound, so say it now: an approval
                # spent here can only record the frame as drifted, never restore it.
                verdict += (
                    f", which this bound cannot: the run can only record {basis['frame']} "
                    "as drifted and change nothing"
                )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Compare {basis['frame']} with {basis['reference']} over {basis['samples']} "
                f"simultaneous views of the board (a first look agrees to {preview['p95_mm']:.1f} mm, "
                f"bound {float(preview['bound']['translation_mm']):.1f} mm{verdict}); no motion"
            ),
            basis={"attestation": dict(basis)},
            limits={"max_duration_s": 60.0},
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        result = await _port(ctx).attest(plan)
        return SkillOutcome.succeeded(
            result,
            message=(
                f"{result['frame']} agrees with {result['reference']} to {result['p95_mm']:.1f} mm "
                f"at p95; it is accurate to {float(result['achieved']['translation_mm']):.1f} mm"
            ),
        )


def skills() -> tuple[PromoteCandidate, AttestFrame]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (PromoteCandidate(), AttestFrame())


__all__ = ["AttestFrame", "PromoteCandidate", "skills"]
