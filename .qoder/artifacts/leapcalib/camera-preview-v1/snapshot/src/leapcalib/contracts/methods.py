# Copyright (c) Alibaba, Inc. and its affiliates.
"""Concrete calibration-method discovery is compute-only, never authority."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.failures import CalibrationFailure, FAILURE_NOT_IMPLEMENTED, FAILURE_SCHEMA

if TYPE_CHECKING:
    from leapcalib.validation.bundle import EvidenceBundle

METHOD_GROUP = "leapcalib.methods"


@runtime_checkable
class CalibrationMethod(Protocol):
    """A method owns data requirements, filtering, bounded solve and support report."""

    @property
    def method_id(self) -> str: ...

    def solve(self, evidence: EvidenceBundle, options: Mapping[str, Any]) -> EstimateBundle: ...


def discover_methods() -> dict[str, CalibrationMethod]:
    """Load installed implementations, rejecting broken or ambiguous discovery."""
    found: dict[str, CalibrationMethod] = {}
    for entry in entry_points(group=METHOD_GROUP):
        try:
            method = entry.load()()
        except (ImportError, AttributeError, TypeError) as error:
            raise CalibrationFailure(
                FAILURE_NOT_IMPLEMENTED, f"calibration method entry {entry.name!r} cannot load"
            ) from error
        if not isinstance(method, CalibrationMethod) or not isinstance(method.method_id, str):
            raise CalibrationFailure(FAILURE_SCHEMA, f"invalid calibration method {entry.name!r}")
        if method.method_id in found:
            raise CalibrationFailure(
                FAILURE_SCHEMA, f"duplicate calibration method {method.method_id!r}"
            )
        found[method.method_id] = method
    return found


__all__ = ["CalibrationMethod", "METHOD_GROUP", "discover_methods"]
