# Copyright (c) Alibaba, Inc. and its affiliates.
"""Operation verification: the evidence-based answer to 'did the physical
action actually succeed?'

Execution completion is not task success.  A motor controller that returns
OK has fulfilled its electrical contract, not the manipulation goal.  This
module provides the structural layer that separates *execution evidence*
(what the hardware reported) from *semantic verdict* (whether the physical
intent was achieved).

Verification is optional and mode-gated: ``off`` skips it entirely,
``audit`` records evidence without blocking, ``enforce`` blocks on failure,
``recovery`` feeds failure into RecoveryCoordinator.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.hardware.transport import FrameReading, Reading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerificationMode(str, Enum):
    """How verification results affect the execution pipeline."""

    OFF = "off"
    """Skip verification entirely."""

    AUDIT = "audit"
    """Record evidence, never block."""

    ENFORCE = "enforce"
    """Block pipeline on failure."""

    RECOVERY = "recovery"
    """Feed failure into RecoveryCoordinator."""


class VerdictStatus(str, Enum):
    """Outcome of a verification check."""

    SUCCESS = "success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Evidence and verdict data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceBundle:
    """Post-operation evidence collected for verification and audit.

    Captured after a write operation completes and the channel's declared
    settling time has elapsed.  Contains the commanded intent, the transport's
    own outcome, and post-settle sensor readings that a verifier uses to
    determine whether the physical goal was achieved.

    ``operation_id`` is a unique identifier for this operation instance,
    linking the evidence to the audit trail entry.  ``frames`` may be empty
    when no camera channel is available or relevant.
    """

    operation_id: str
    device_id: str
    channel_id: str
    timestamp: float
    intended_value: Any
    actual_outcome: Any  # WriteOutcome, but typed as Any to avoid circular import
    post_settle_readings: tuple[Reading, ...]
    frames: tuple[FrameReading, ...] = ()
    settle_delay_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the evidence as a plain dict for audit serialisation."""
        return {
            "operation_id": self.operation_id,
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "timestamp": self.timestamp,
            "intended_value": self.intended_value,
            "settle_delay_s": self.settle_delay_s,
            "post_settle_readings": [r.to_dict() for r in self.post_settle_readings],
            "frames": [f.to_metadata() for f in self.frames],
        }


@dataclass(frozen=True)
class OperationVerdict:
    """The semantic answer to 'did this operation achieve its goal?'

    ``confidence`` is 0.0--1.0 and reflects how much evidence the verifier
    had.  A position check with tight tolerance and immediate readback is
    high confidence; a grasp check with only force data and no visual
    confirmation is lower.

    ``deviation`` is the absolute difference between intended and actual
    in the channel's native units, when meaningful (e.g. position error
    in radians).  None when the metric is not numeric (e.g. grasp binary).

    ``detail`` is a human-readable (and LLM-readable) explanation of the
    verdict, suitable for the audit log and for feeding into
    RecoveryCoordinator context.
    """

    status: str  # VerdictStatus value
    confidence: float
    deviation: float | None = None
    detail: str = ""
    evidence: EvidenceBundle | None = None

    @property
    def is_success(self) -> bool:
        """Return True when the verification passed."""
        return self.status == VerdictStatus.SUCCESS.value

    @property
    def is_failure(self) -> bool:
        """Return True when the verification definitively failed."""
        return self.status == VerdictStatus.FAILURE.value

    def to_dict(self) -> dict[str, Any]:
        """Return the verdict as a plain dict for audit serialisation."""
        payload: dict[str, Any] = {
            "status": self.status,
            "confidence": self.confidence,
        }
        if self.deviation is not None:
            payload["deviation"] = self.deviation
        if self.detail:
            payload["detail"] = self.detail
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        return payload


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OperationVerifier(Protocol):
    """Judges whether a physical operation achieved its intended goal.

    Stateless by design: a verifier receives an evidence bundle and returns
    a verdict.  State (success/failure counts, trust progression) lives in
    the trust gate, not here.
    """

    @property
    def verifier_id(self) -> str:
        """Unique identifier for this verifier type."""
        ...

    @property
    def applicable_effects(self) -> frozenset[str]:
        """HardwareEffect values this verifier can judge.

        The dispatcher matches a channel's declared ``effect`` against this
        set to select the right verifier.
        """
        ...

    async def verify(self, bundle: EvidenceBundle) -> OperationVerdict:
        """Examine the evidence and return a verdict."""
        ...


# ---------------------------------------------------------------------------
# Concrete verifiers
# ---------------------------------------------------------------------------


class PositionVerifier:
    """Verifies that a joint position command achieved its target.

    Verification logic:

    1. Find the post-settle reading for the target channel.
    2. Compare with commanded value within the channel's declared tolerance
       (from Envelope), or a configured default tolerance.
    3. If deviation < tolerance: SUCCESS with high confidence.
    4. If deviation < 2 * tolerance: INCONCLUSIVE.
    5. Otherwise: FAILURE.

    Applicable to: ACTUATE and CONFIGURE effects on SCALAR channels.
    """

    def __init__(self, *, default_tolerance: float = 0.05) -> None:
        self._default_tolerance = max(0.0, default_tolerance)

    @property
    def verifier_id(self) -> str:
        return "position"

    @property
    def applicable_effects(self) -> frozenset[str]:
        return frozenset({"actuate", "configure"})

    async def verify(self, bundle: EvidenceBundle) -> OperationVerdict:
        """Check whether the post-settle reading matches the commanded value."""
        # Locate the reading for the target channel.
        reading = _find_channel_reading(bundle.post_settle_readings, bundle.channel_id)
        if reading is None:
            return OperationVerdict(
                status=VerdictStatus.INCONCLUSIVE.value,
                confidence=0.1,
                detail=(
                    f"No post-settle reading for channel '{bundle.channel_id}'; "
                    "cannot verify position."
                ),
                evidence=bundle,
            )

        intended = _as_float(bundle.intended_value)
        actual = _as_float(reading.value)
        if intended is None or actual is None:
            return OperationVerdict(
                status=VerdictStatus.INCONCLUSIVE.value,
                confidence=0.2,
                detail=(
                    f"Non-numeric values (intended={bundle.intended_value!r}, "
                    f"actual={reading.value!r}); position check requires numerics."
                ),
                evidence=bundle,
            )

        deviation = abs(actual - intended)
        tolerance = self._resolve_tolerance(bundle)

        if deviation < tolerance:
            return OperationVerdict(
                status=VerdictStatus.SUCCESS.value,
                confidence=0.95,
                deviation=deviation,
                detail=(
                    f"Position within tolerance: deviation={deviation:.6f}, "
                    f"tolerance={tolerance:.6f}."
                ),
                evidence=bundle,
            )

        if deviation < 2.0 * tolerance:
            return OperationVerdict(
                status=VerdictStatus.INCONCLUSIVE.value,
                confidence=0.5,
                deviation=deviation,
                detail=(
                    f"Position marginally outside tolerance: deviation={deviation:.6f}, "
                    f"tolerance={tolerance:.6f} (within 2x)."
                ),
                evidence=bundle,
            )

        return OperationVerdict(
            status=VerdictStatus.FAILURE.value,
            confidence=0.9,
            deviation=deviation,
            detail=(
                f"Position outside tolerance: deviation={deviation:.6f}, "
                f"tolerance={tolerance:.6f}."
            ),
            evidence=bundle,
        )

    def _resolve_tolerance(self, bundle: EvidenceBundle) -> float:
        """Return the effective tolerance from metadata or the default."""
        # The caller (collect_evidence or the tool layer) may inject the
        # channel's Envelope tolerance into ``bundle.metadata``.
        declared = bundle.metadata.get("tolerance")
        if declared is not None:
            val = _as_float(declared)
            if val is not None and val > 0.0:
                return val
        return self._default_tolerance


class GraspVerifier:
    """Verifies grasp success using force/torque readings.

    Verification logic:

    1. Look for force/torque readings in ``post_settle_readings``
       (channel quantity contains 'force' or 'torque').
    2. If force > force_threshold -> grip detected, SUCCESS.
    3. If force readings available but below threshold -> FAILURE.
    4. If no force readings available -> INCONCLUSIVE (low confidence).
    5. Optional: if frames available, note them in metadata for future
       VLM-based verification (not implemented in v0).

    Applicable to: ACTUATE effects on STATE channels (gripper).
    """

    def __init__(self, *, force_threshold: float = 1.0) -> None:
        self._force_threshold = max(0.0, force_threshold)

    @property
    def verifier_id(self) -> str:
        return "grasp"

    @property
    def applicable_effects(self) -> frozenset[str]:
        return frozenset({"actuate"})

    async def verify(self, bundle: EvidenceBundle) -> OperationVerdict:
        """Check whether force/torque readings indicate a successful grasp."""
        force_readings = [
            r for r in bundle.post_settle_readings
            if _is_force_quantity(r.quantity)
        ]

        if not force_readings:
            frame_note = ""
            if bundle.frames:
                frame_note = (
                    f" ({len(bundle.frames)} frame(s) available for future "
                    "VLM-based verification.)"
                )
            return OperationVerdict(
                status=VerdictStatus.INCONCLUSIVE.value,
                confidence=0.15,
                detail=(
                    f"No force/torque readings available for grasp verification."
                    f"{frame_note}"
                ),
                evidence=bundle,
            )

        # Use the maximum force among available readings.
        max_force_reading = max(force_readings, key=lambda r: _as_float(r.value) or 0.0)
        force_value = _as_float(max_force_reading.value)

        if force_value is None:
            return OperationVerdict(
                status=VerdictStatus.INCONCLUSIVE.value,
                confidence=0.2,
                detail=(
                    f"Force reading is non-numeric: {max_force_reading.value!r}."
                ),
                evidence=bundle,
            )

        if force_value >= self._force_threshold:
            return OperationVerdict(
                status=VerdictStatus.SUCCESS.value,
                confidence=0.75,
                detail=(
                    f"Grasp detected: force={force_value:.3f}, "
                    f"threshold={self._force_threshold:.3f}."
                ),
                evidence=bundle,
            )

        return OperationVerdict(
            status=VerdictStatus.FAILURE.value,
            confidence=0.7,
            detail=(
                f"Grasp not detected: force={force_value:.3f} below "
                f"threshold={self._force_threshold:.3f}."
            ),
            evidence=bundle,
        )


# ---------------------------------------------------------------------------
# Evidence collection helper
# ---------------------------------------------------------------------------


async def collect_evidence(
    registry: Any,  # HardwareRegistry, typed as Any to avoid circular import
    device_id: str,
    channel_id: str,
    *,
    intended_value: Any,
    outcome: Any,  # WriteOutcome
    settle_delay_s: float = 0.0,
    related_channels: tuple[str, ...] = (),
    capture_frame: bool = False,
) -> EvidenceBundle:
    """Collect post-operation evidence from a device.

    Waits ``settle_delay_s`` (from the channel's Envelope), then reads
    the target channel and any related channels.  Optionally captures
    a frame if the device has a ``FrameTransport``.

    The registry is typed as ``Any`` to avoid importing ``HardwareRegistry``
    from ``registry.py``, which would create a circular dependency.  The
    caller is expected to pass a registry instance that exposes ``read``
    and optionally ``read_frame`` for evidence collection.
    """
    operation_id = uuid.uuid4().hex[:16]

    # Wait for the physical system to settle before reading back.
    if settle_delay_s > 0.0:
        await asyncio.sleep(settle_delay_s)

    # Read the target channel.
    readings: list[Reading] = []
    target_reading = await _safe_read(registry, device_id, channel_id)
    if target_reading is not None:
        readings.append(target_reading)

    # Read related channels (e.g. adjacent force sensors for a grasp).
    for related_ch in related_channels:
        related_reading = await _safe_read(registry, device_id, related_ch)
        if related_reading is not None:
            readings.append(related_reading)

    # Optionally capture a frame for visual evidence.
    frames: list[FrameReading] = []
    if capture_frame:
        frame = await _safe_read_frame(registry, device_id, channel_id)
        if frame is not None:
            frames.append(frame)

    return EvidenceBundle(
        operation_id=operation_id,
        device_id=device_id,
        channel_id=channel_id,
        timestamp=time.time(),
        intended_value=intended_value,
        actual_outcome=outcome,
        post_settle_readings=tuple(readings),
        frames=tuple(frames),
        settle_delay_s=settle_delay_s,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_channel_reading(
    readings: tuple[Reading, ...],
    channel_id: str,
) -> Reading | None:
    """Return the first reading matching *channel_id*, or None."""
    return next((r for r in readings if r.channel_id == channel_id), None)


def _is_force_quantity(quantity: str) -> bool:
    """Return True when the quantity string indicates force or torque."""
    q = quantity.lower()
    return "force" in q or "torque" in q


def _as_float(value: Any) -> float | None:
    """Coerce *value* to a finite float, returning None on failure.

    Mirrors ``context.as_numeric`` but kept local to avoid coupling on a
    private-scope function signature.  Booleans, NaN, and infinity are
    rejected for the same reasons documented there.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    import math
    if math.isnan(result) or math.isinf(result):
        return None
    return result


async def _safe_read(
    registry: Any,
    device_id: str,
    channel_id: str,
) -> Reading | None:
    """Read a channel, returning None on any failure.

    Evidence collection must never raise: a failed readback is missing
    evidence, not a pipeline-stopping error.
    """
    try:
        return await registry.read(device_id, channel_id)
    except Exception:
        logger.debug(
            "Evidence collection: failed to read %s/%s",
            device_id, channel_id,
            exc_info=True,
        )
        return None


async def _safe_read_frame(
    registry: Any,
    device_id: str,
    channel_id: str,
) -> FrameReading | None:
    """Read a frame, returning None on any failure or if unsupported."""
    try:
        read_frame = getattr(registry, "read_frame", None)
        if read_frame is None:
            return None
        return await read_frame(device_id, channel_id)
    except Exception:
        logger.debug(
            "Evidence collection: failed to read frame %s/%s",
            device_id, channel_id,
            exc_info=True,
        )
        return None


__all__ = [
    "EvidenceBundle",
    "GraspVerifier",
    "OperationVerdict",
    "OperationVerifier",
    "PositionVerifier",
    "VerdictStatus",
    "VerificationMode",
    "collect_evidence",
]
