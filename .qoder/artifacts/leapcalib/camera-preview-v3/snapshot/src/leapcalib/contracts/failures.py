# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed calibration failures, so a refusal is readable instead of a bare ``ValueError``.

A calibration worker fails in kinds that call for different handling: evidence
that does not support the parameter asked for, data whose provenance disqualifies
it, two clocks that cannot be compared, a geometry asset that was never supplied.
A caller that only sees ``ValueError`` has to parse prose to tell "ask again with
more poses" from "this input was never admissible".

These codes stay LeapFlow-free. The integration adapter maps them onto the
hardware layer's existing failure vocabulary
(``leapflow.hardware.skills.protocol``) so recovery keeps happening in one place;
LeapCalib does not learn LeapFlow's codes, and LeapFlow does not learn
LeapCalib's exception type.

Construction errors in the value types stay ``ValueError``: a malformed frozen
dataclass is a programming defect at the boundary, not an estimation outcome.
"""

from __future__ import annotations

FAILURE_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
"""The observations do not support the requested parameter block at all."""
FAILURE_UNSUPPORTED_PARAMETER = "unsupported_parameter"
"""A supported-looking estimate was requested for a direction the data leave free."""
FAILURE_PROVENANCE_REFUSED = "provenance_refused"
"""The input was preview-grade or otherwise not metric calibration evidence."""
FAILURE_CLOCK_DOMAIN_MISMATCH = "clock_domain_mismatch"
"""Two stamps from unrelated clocks were asked to be compared or differenced."""
FAILURE_GEOMETRY_ASSET_MISSING = "geometry_asset_missing"
"""A method needs native surface geometry that was never declared or verified."""
FAILURE_BUDGET_EXCEEDED = "budget_exceeded"
"""The predicted error does not fit the task tolerance it was checked against."""
FAILURE_SCHEMA = "schema"
"""An evidence bundle or message did not match the schema version it claims."""
FAILURE_EVIDENCE_CORRUPT = "evidence_corrupt"
"""A stored bundle failed its own digest or exceeded its admission bound."""
FAILURE_NOT_IMPLEMENTED = "not_implemented"
"""A declared-but-undelivered stage was invoked; P0 ships no solver and no service."""

FAILURE_WORKER_UNAVAILABLE = "worker_unavailable"
"""Compute worker disabled, crashed, or exhausted its bounded fresh-launch allowance."""
FAILURE_DEADLINE = "deadline"
"""Compute request expired; any late output is discarded, never replaying motion."""

FAILURE_CODES = frozenset(
    {
        FAILURE_INSUFFICIENT_EVIDENCE,
        FAILURE_UNSUPPORTED_PARAMETER,
        FAILURE_PROVENANCE_REFUSED,
        FAILURE_CLOCK_DOMAIN_MISMATCH,
        FAILURE_GEOMETRY_ASSET_MISSING,
        FAILURE_BUDGET_EXCEEDED,
        FAILURE_SCHEMA,
        FAILURE_EVIDENCE_CORRUPT,
        FAILURE_NOT_IMPLEMENTED,
        FAILURE_WORKER_UNAVAILABLE,
        FAILURE_DEADLINE,
    }
)


class CalibrationFailure(Exception):
    """A calibration request that will not be answered, with a reason code.

    ``retryable`` means *the numerics* may be retried unchanged -- it never means
    a robot action may be replayed. Nothing in LeapCalib acquires actuator
    authority, so a retry here recomputes; it does not move anything.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown calibration failure code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


__all__ = [
    "FAILURE_BUDGET_EXCEEDED",
    "FAILURE_CLOCK_DOMAIN_MISMATCH",
    "FAILURE_CODES",
    "FAILURE_WORKER_UNAVAILABLE",
    "FAILURE_DEADLINE",
    "FAILURE_EVIDENCE_CORRUPT",
    "FAILURE_GEOMETRY_ASSET_MISSING",
    "FAILURE_INSUFFICIENT_EVIDENCE",
    "FAILURE_NOT_IMPLEMENTED",
    "FAILURE_PROVENANCE_REFUSED",
    "FAILURE_SCHEMA",
    "FAILURE_UNSUPPORTED_PARAMETER",
    "CalibrationFailure",
]
