# Copyright (c) Alibaba, Inc. and its affiliates.
"""Estimates that cannot quietly overstate themselves.

The failure mode this module exists to prevent is an optimizer returning six
numbers for a transform the data only constrain in four directions, and those six
numbers travelling onward as a measurement. The remedy is in the type: an
estimate carries its support state, and the state constrains the values.

* ``retained`` and ``insufficient`` must equal the prior, component for
  component. "We could not establish this, so we kept the declared value" is
  then a checkable property rather than a promise in a report.
* ``partial`` must name exactly which components are supported -- a non-empty
  proper subset -- and every unsupported component must still equal its prior.
* ``supported`` must name a method and a non-zero count of accepted observations.
  Support is a claim about evidence, so it requires evidence to point at.

Uncertainty is never allowed to shrink below the prior's for a component that was
not estimated, which stops a prior-regularised solve from reporting confidence it
inherited rather than earned.

What this module deliberately does *not* have is a ``qualified`` flag. Held-out
qualification, validity domains and activation authority live elsewhere --
``leapcalib.validation`` and the LeapFlow admission path -- because a
statistically plausible estimate is not an approved one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from leapcalib.contracts.identity import TopologyIdentity, digest_of
from leapcalib.contracts.parameters import (
    PRIOR_ONLY_STATES,
    SUPPORT_PARTIAL,
    SUPPORT_RETAINED,
    SUPPORT_STATES,
    SUPPORT_SUPPORTED,
    ParameterBlock,
)
from leapcalib.contracts.units import canonical_number

SCHEMA_VERSION = "leapcalib.estimate.v0"
"""Schema identity of a candidate estimate bundle; support never grants activation."""


def _array(value: Any, *, maximum: int) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise ValueError("bounded estimate array required before decoding")
    return value


def _text(value: Any, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or len(value) > 4096 or (not allow_empty and not value.strip()):
        raise ValueError("bounded typed estimate identity/text required")


@dataclass(frozen=True)
class ParameterEstimate:
    """One parameter block's values, its prior, and exactly how far the data got."""

    block: ParameterBlock
    values: tuple[float, ...]
    prior: tuple[float, ...]
    support: str
    uncertainty: tuple[float, ...]
    prior_uncertainty: tuple[float, ...]
    supported_components: tuple[int, ...] = ()
    gauge_anchors: tuple[str, ...] = ()
    method_id: str = ""
    accepted_count: int = 0
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.support not in SUPPORT_STATES:
            raise ValueError(f"support must be one of {sorted(SUPPORT_STATES)}")
        for name in ("values", "prior", "uncertainty", "prior_uncertainty"):
            values = tuple(canonical_number(v) for v in getattr(self, name))
            if len(values) != self.block.dimension:
                raise ValueError(f"{name} must have {self.block.dimension} components")
            object.__setattr__(self, name, values)
        if any(u < 0 for u in self.uncertainty + self.prior_uncertainty):
            raise ValueError("uncertainties cannot be negative")
        object.__setattr__(self, "supported_components", tuple(self.supported_components))
        object.__setattr__(self, "gauge_anchors", tuple(self.gauge_anchors))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if type(self.accepted_count) is not int or self.accepted_count < 0:
            raise ValueError("accepted observation count must be a nonnegative integer")
        self._check_support()

    def _check_support(self) -> None:
        dimension = self.block.dimension
        indices = set(self.supported_components)
        if any(type(i) is not int or i < 0 or i >= dimension for i in indices):
            raise ValueError("supported components must index the block")
        if len(indices) != len(self.supported_components):
            raise ValueError("supported components must be unique")

        if self.support in PRIOR_ONLY_STATES:
            if indices:
                raise ValueError(f"a {self.support} estimate supports no component")
            if self.values != self.prior:
                raise ValueError(f"a {self.support} estimate must keep its prior values unchanged")
        elif self.support == SUPPORT_SUPPORTED:
            if indices != set(range(dimension)):
                raise ValueError("a supported estimate supports every component")
        elif self.support == SUPPORT_PARTIAL:
            if not indices or indices == set(range(dimension)):
                raise ValueError(
                    "a partial estimate supports a nonempty proper subset of components"
                )

        if self.support in (SUPPORT_SUPPORTED, SUPPORT_PARTIAL):
            if not self.method_id.strip():
                raise ValueError("a supported estimate must name the method that produced it")
            if self.accepted_count < 1:
                raise ValueError("a supported estimate must point at accepted observations")

        # An unestimated component keeps its prior value *and* its prior
        # uncertainty: carrying a tighter uncertainty for a component the data
        # never touched is how a prior ends up read as a measurement.
        for index in range(dimension):
            if index in indices:
                continue
            if self.values[index] != self.prior[index]:
                raise ValueError(
                    f"component {index} of {self.block.block_id} is unsupported "
                    "but differs from its prior"
                )
            if self.uncertainty[index] < self.prior_uncertainty[index]:
                raise ValueError(
                    f"component {index} of {self.block.block_id} is unsupported "
                    "but claims tighter uncertainty than its prior"
                )

    @property
    def retained_components(self) -> tuple[int, ...]:
        """Component indices still carrying their declared prior."""
        supported = set(self.supported_components)
        return tuple(i for i in range(self.block.dimension) if i not in supported)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block.to_dict(),
            "values": list(self.values),
            "prior": list(self.prior),
            "support": self.support,
            "uncertainty": list(self.uncertainty),
            "prior_uncertainty": list(self.prior_uncertainty),
            "supported_components": list(self.supported_components),
            "gauge_anchors": list(self.gauge_anchors),
            "method_id": self.method_id,
            "accepted_count": self.accepted_count,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ParameterEstimate":
        if not isinstance(data, Mapping) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("unknown parameter estimate fields")
        for name in ("values", "prior", "uncertainty", "prior_uncertainty"):
            _array(data.get(name), maximum=1024)
        for name in ("supported_components", "gauge_anchors", "limitations"):
            _array(data.get(name, ()), maximum=1024)
        _text(data.get("method_id", ""), allow_empty=True)
        for name in ("gauge_anchors", "limitations"):
            for value in data.get(name, ()):
                _text(value)
        return cls(
            block=ParameterBlock.from_mapping(data["block"]),
            values=tuple(data["values"]),
            prior=tuple(data["prior"]),
            support=data["support"],
            uncertainty=tuple(data["uncertainty"]),
            prior_uncertainty=tuple(data["prior_uncertainty"]),
            supported_components=tuple(data.get("supported_components", ())),
            gauge_anchors=tuple(data.get("gauge_anchors", ())),
            method_id=data.get("method_id", ""),
            accepted_count=data.get("accepted_count", 0),
            limitations=tuple(data.get("limitations", ())),
        )


def retained_estimate(
    block: ParameterBlock,
    prior: tuple[float, ...],
    prior_uncertainty: tuple[float, ...],
    *,
    reason: str,
    support: str = SUPPORT_RETAINED,
) -> ParameterEstimate:
    """A block the data did not establish, kept at its prior and labelled as kept.

    The normal constructor for an unsupported parameter: it is a result, not an
    error, and a bundle that reports one is more honest than a bundle that omits
    the block.
    """
    return ParameterEstimate(
        block=block,
        values=tuple(prior),
        prior=tuple(prior),
        support=support,
        uncertainty=tuple(prior_uncertainty),
        prior_uncertainty=tuple(prior_uncertainty),
        limitations=(reason,),
    )


@dataclass(frozen=True)
class EstimateBundle:
    """A solve's full result: identity, estimates, the evidence split, and limits.

    ``accepted_ids``, ``rejected_ids`` and ``holdout_ids`` must be disjoint, for
    the same reason the LeapFlow certificate requires it: a held-out observation
    that also trained the estimate measures nothing.
    """

    bundle_id: str
    topology: TopologyIdentity
    estimates: tuple[ParameterEstimate, ...]
    accepted_ids: tuple[str, ...] = ()
    rejected_ids: tuple[str, ...] = ()
    holdout_ids: tuple[str, ...] = ()
    method_id: str = ""
    schema_version: str = SCHEMA_VERSION
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.bundle_id)
        _text(self.method_id, allow_empty=True)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported estimate schema {self.schema_version!r}")
        object.__setattr__(self, "estimates", tuple(self.estimates))
        for name in ("accepted_ids", "rejected_ids", "holdout_ids", "limitations"):
            values = _array(getattr(self, name), maximum=4096)
            for value in values:
                _text(value)
            object.__setattr__(self, name, tuple(values))
        groups = (set(self.accepted_ids), set(self.rejected_ids), set(self.holdout_ids))
        names = (self.accepted_ids, self.rejected_ids, self.holdout_ids)
        if any(len(group) != len(values) for group, values in zip(groups, names)):
            raise ValueError("observation identifiers must be unique within each group")
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("accepted, rejected and held-out observations must be disjoint")
        block_ids = [e.block.block_id for e in self.estimates]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("one estimate per parameter block")

    @property
    def supported_blocks(self) -> tuple[str, ...]:
        return tuple(e.block.block_id for e in self.estimates if e.support == SUPPORT_SUPPORTED)

    @property
    def partial_blocks(self) -> tuple[str, ...]:
        return tuple(e.block.block_id for e in self.estimates if e.support == SUPPORT_PARTIAL)

    @property
    def retained_blocks(self) -> tuple[str, ...]:
        return tuple(e.block.block_id for e in self.estimates if e.support in PRIOR_ONLY_STATES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "schema_version": self.schema_version,
            "topology": self.topology.to_dict(),
            "estimates": [e.to_dict() for e in self.estimates],
            "accepted_ids": list(self.accepted_ids),
            "rejected_ids": list(self.rejected_ids),
            "holdout_ids": list(self.holdout_ids),
            "method_id": self.method_id,
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EstimateBundle":
        if not isinstance(data, Mapping) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("unknown estimate bundle fields")
        estimates = _array(data.get("estimates"), maximum=128)
        for name in ("accepted_ids", "rejected_ids", "holdout_ids", "limitations"):
            _array(data.get(name, ()), maximum=4096)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("estimate metadata must be a mapping")
        return cls(
            bundle_id=data["bundle_id"],
            topology=TopologyIdentity.from_mapping(data["topology"]),
            estimates=tuple(ParameterEstimate.from_mapping(e) for e in estimates),
            accepted_ids=tuple(data.get("accepted_ids", ())),
            rejected_ids=tuple(data.get("rejected_ids", ())),
            holdout_ids=tuple(data.get("holdout_ids", ())),
            method_id=data.get("method_id", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            limitations=tuple(data.get("limitations", ())),
            metadata=dict(metadata),
        )

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


__all__ = [
    "SCHEMA_VERSION",
    "EstimateBundle",
    "ParameterEstimate",
    "retained_estimate",
]
