# Copyright (c) Alibaba, Inc. and its affiliates.
"""The unit and provenance boundary between LeapFlow hardware and LeapCalib.

LeapFlow's hardware layer speaks the declaration's units -- millimetres, degrees,
``Rz@Ry@Rx`` Euler angles -- because that is what device YAML, firmware channels
and operators use. LeapCalib speaks metres and radians, because that is what
manifold optimisation and covariance weighting need. Both are right, and the
conversion between them is exactly the kind of thing that gets done twice or not
at all when it is scattered through call sites. So it happens here, once, in
named functions, and nowhere else.

This module is also where preview-grade data is turned away. ``FrameTransport``
produces a bounded JPEG and ``RgbdFrame`` produces metric depth; only the second
can become an observation, and asking for the first raises rather than quietly
producing an observation whose relationship to millimetres is undefined.

What this module does not do: acquire frames, move anything, persist anything, or
activate anything. It converts values and maps failure codes. The existing
node-side owners keep approval, leasing, storage and admission, which is what
keeps there being one of each.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from leapcalib.contracts.failures import (
    FAILURE_BUDGET_EXCEEDED,
    FAILURE_CLOCK_DOMAIN_MISMATCH,
    FAILURE_DEADLINE,
    FAILURE_EVIDENCE_CORRUPT,
    FAILURE_GEOMETRY_ASSET_MISSING,
    FAILURE_INSUFFICIENT_EVIDENCE,
    FAILURE_NOT_IMPLEMENTED,
    FAILURE_PROVENANCE_REFUSED,
    FAILURE_SCHEMA,
    FAILURE_UNSUPPORTED_PARAMETER,
    FAILURE_WORKER_UNAVAILABLE,
    CalibrationFailure,
)
from leapcalib.contracts.identity import (
    ROLE_FIXED,
    SOURCE_FACTORY,
    SOURCE_VENDOR_DOCUMENT,
    ModelIdentity,
    StreamProfile,
    digest_of,
)
from leapcalib.contracts.observations import (
    CALIBRATION_PROVENANCE,
    MAX_SAMPLES,
    PROVENANCE_METRIC_RGBD,
    JointObservation,
    PixelDepthSample,
    RgbdObservation,
    Stamp,
)
from leapcalib.contracts.units import deg_to_rad, mm_to_m
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills import protocol as hw
from leapflow.hardware.skills.kinematics import DhChain

# How a LeapCalib refusal reads to LeapFlow's recovery path. The mapping is
# deliberately onto planning-time codes: nothing in LeapCalib moves an arm, so no
# refusal here may imply a side effect that has to be verified before a retry.
CALIBRATION_FAILURE_MAP: Mapping[str, str] = {
    # The evidence does not support the claim -- a measurement out of tolerance,
    # which is what check_failed already means for the bring-up checks.
    FAILURE_INSUFFICIENT_EVIDENCE: hw.FAILURE_CHECK_FAILED,
    FAILURE_UNSUPPORTED_PARAMETER: hw.FAILURE_CHECK_FAILED,
    FAILURE_BUDGET_EXCEEDED: hw.FAILURE_CHECK_FAILED,
    # The caller handed over something that was never admissible. Nothing was
    # attempted, so this is an argument problem, not a hardware one.
    FAILURE_PROVENANCE_REFUSED: hw.FAILURE_INVALID_ARGS,
    FAILURE_CLOCK_DOMAIN_MISMATCH: hw.FAILURE_INVALID_ARGS,
    # A declared model or asset the deployment never supplied.
    FAILURE_GEOMETRY_ASSET_MISSING: hw.FAILURE_MODEL_UNAVAILABLE,
    # Stored evidence from another schema version is stale; evidence that failed
    # its own digest cannot be read at all, so for a caller it is missing.
    FAILURE_SCHEMA: hw.FAILURE_CALIBRATION_STALE,
    FAILURE_EVIDENCE_CORRUPT: hw.FAILURE_CALIBRATION_MISSING,
    # A stage that was designed but not delivered.
    FAILURE_NOT_IMPLEMENTED: hw.FAILURE_SKILL_UNAVAILABLE,
    FAILURE_WORKER_UNAVAILABLE: hw.FAILURE_SKILL_UNAVAILABLE,
    FAILURE_DEADLINE: hw.FAILURE_TIMEOUT,
}


def hardware_failure_code(failure: CalibrationFailure) -> str:
    """The LeapFlow failure code for a LeapCalib refusal.

    An unmapped code becomes ``internal_defect`` rather than a plausible
    neighbour: a failure kind this adapter has never seen means LeapCalib grew a
    code and the mapping was not updated, which is a defect in LeapFlow and
    should stop the turn instead of being retried as a hardware condition.
    """
    return CALIBRATION_FAILURE_MAP.get(failure.code, hw.FAILURE_INTERNAL_DEFECT)


def stream_profile(
    *,
    device_id: str,
    stream: str,
    intrinsics: CameraIntrinsics,
    depth_scale_m: float,
    role: str = ROLE_FIXED,
    valid_min_m: float = 0.0,
    valid_max_m: float = 0.0,
    source: str = SOURCE_FACTORY,
) -> StreamProfile:
    """A :class:`StreamProfile` for the stream these *intrinsics* were read from.

    ``source`` defaults to factory because that is what the SDK reports: these
    are the vendor's numbers for the running profile, and they stay priors until
    a solve with its own evidence supersedes them.
    """
    return StreamProfile(
        device_id=device_id,
        stream=stream,
        width=intrinsics.width,
        height=intrinsics.height,
        fx_px=intrinsics.fx,
        fy_px=intrinsics.fy,
        ppx_px=intrinsics.ppx,
        ppy_px=intrinsics.ppy,
        depth_scale_m=depth_scale_m,
        distortion_model=intrinsics.model,
        distortion_coeffs=tuple(intrinsics.coeffs),
        valid_min_m=valid_min_m,
        valid_max_m=valid_max_m,
        role=role,
        source=source,
    )


def stream_profile_of(
    frame: RgbdFrame,
    *,
    stream: str = "depth_aligned",
    role: str = ROLE_FIXED,
    valid_min_m: float = 0.0,
    valid_max_m: float = 0.0,
) -> StreamProfile:
    """The profile of the stream *frame* came from, taken from the frame itself."""
    return stream_profile(
        device_id=frame.device_id,
        stream=stream,
        intrinsics=frame.intrinsics,
        depth_scale_m=frame.depth_scale_m,
        role=role,
        valid_min_m=valid_min_m,
        valid_max_m=valid_max_m,
    )


def rgbd_observation(
    frame: RgbdFrame,
    *,
    observation_id: str,
    pixels: Sequence[tuple[int, int]],
    profile: StreamProfile | None = None,
    stamp: Stamp | None = None,
    provenance: str = PROVENANCE_METRIC_RGBD,
    link_ids: Mapping[tuple[int, int], str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RgbdObservation:
    """Sample *frame* at *pixels* into a metric observation.

    Depth leaves ``RgbdFrame`` in millimetres and arrives in metres through
    :func:`~leapcalib.contracts.units.mm_to_m`, so the conversion is visible in
    the call rather than implied by a magnitude. A pixel with no measurement
    keeps its zero: dropping it here would hide how much of a view was unusable,
    which is exactly what :attr:`RgbdObservation.valid_ratio` reports.

    When *stamp* is omitted the frame's node monotonic receipt time is used, with
    the semantics it actually has -- receipt, not exposure. Callers who captured
    after measured stillness should say so by passing a
    :data:`~leapcalib.contracts.observations.STAMP_SETTLED_CAPTURE` stamp.
    """
    if provenance not in CALIBRATION_PROVENANCE:
        raise CalibrationFailure(
            FAILURE_PROVENANCE_REFUSED,
            f"{provenance} cannot be calibration evidence; a preview JPEG and a "
            "colourised depth image have been resampled and palette-quantised, "
            "so no residual built from them has a stated relationship to metres",
        )
    selected_profile = profile or stream_profile_of(frame)
    live_profile = stream_profile_of(frame)
    profile_fields = (
        "device_id",
        "width",
        "height",
        "fx_px",
        "fy_px",
        "ppx_px",
        "ppy_px",
        "depth_scale_m",
        "distortion_model",
        "distortion_coeffs",
    )
    if any(getattr(selected_profile, key) != getattr(live_profile, key) for key in profile_fields):
        raise CalibrationFailure(
            FAILURE_PROVENANCE_REFUSED, "profile does not match the live frame"
        )
    if len(pixels) > MAX_SAMPLES:
        raise ValueError(f"an observation holds at most {MAX_SAMPLES} samples")
    if any(
        type(u) is not int
        or type(v) is not int
        or not (0 <= u < frame.width and 0 <= v < frame.height)
        for u, v in pixels
    ):
        raise ValueError("sample pixel is outside the frame or is not an integer")
    labels = dict(link_ids or {})
    samples = tuple(
        PixelDepthSample(
            u_px=float(u),
            v_px=float(v),
            depth_m=mm_to_m(frame.depth_mm(u, v)),
            link_id=labels.get((u, v), ""),
        )
        for u, v in pixels
    )
    return RgbdObservation(
        observation_id=observation_id,
        stream=selected_profile,
        stamp=stamp or Stamp(seconds=frame.monotonic_at),
        provenance=provenance,
        samples=samples,
        metadata={**dict(metadata or {}), "frame_sequence": frame.sequence},
    )


def model_identity(
    chain: DhChain,
    *,
    model_id: str,
    source: str = SOURCE_VENDOR_DOCUMENT,
) -> ModelIdentity:
    """The SI identity of a declared DH table.

    The digest is taken over the chain *converted to metres and radians*, so the
    same physical geometry declared in different units digests identically and a
    correction cannot be silently reattached to a different table.
    """
    si_chain = {
        "convention": chain.convention,
        "links": [
            {
                "a_m": mm_to_m(link.a),
                "alpha_rad": deg_to_rad(link.alpha),
                "d_m": mm_to_m(link.d),
                "theta_offset_rad": deg_to_rad(link.theta_offset),
            }
            for link in chain.links
        ],
    }
    return ModelIdentity(
        model_id=model_id,
        dof=chain.dof,
        chain_digest=digest_of(si_chain),
        dh_convention=chain.convention,
        source=source,
    )


def joint_observation(
    joints_deg: Sequence[float],
    *,
    observation_id: str,
    model: ModelIdentity,
    stamp: Stamp,
    still: bool = False,
    max_speed_deg_s: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> JointObservation:
    """Raw joint feedback in degrees, as a radian observation.

    These remain *raw* firmware coordinates. Applying a correction here would put
    a second, invisible offset on the path that
    :mod:`leapcalib.geometry.conventions` exists to keep single and explicit.
    """
    return JointObservation(
        observation_id=observation_id,
        model=model,
        stamp=stamp,
        q_raw_rad=tuple(deg_to_rad(float(q)) for q in joints_deg),
        still=still,
        max_speed_rad_s=deg_to_rad(max_speed_deg_s),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "CALIBRATION_FAILURE_MAP",
    "hardware_failure_code",
    "joint_observation",
    "model_identity",
    "rgbd_observation",
    "stream_profile",
    "stream_profile_of",
]
