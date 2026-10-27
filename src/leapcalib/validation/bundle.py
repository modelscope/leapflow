# Copyright (c) Alibaba, Inc. and its affiliates.
"""Evidence bundles: what was observed, against what hardware, with what caveats.

Replay is the only honest way to develop a calibration offline. Every claim this
package will eventually make -- that a parameter is supported, that a residual
fell, that an injected offset was recovered -- has to be reproducible from stored
observations, or it is a claim about one afternoon on one robot.

So a bundle is self-describing and integrity-checked:

* It carries the full topology identity, so a bundle recorded before a remount
  can be *detected* as such instead of being silently mixed into a later solve.
* Accepted, rejected and held-out observation ids are disjoint by construction.
  A held-out sample that also trained the estimate measures nothing.
* The file stores a digest of its own content. A truncated or edited bundle is
  refused rather than being solved against.
* Reading is bounded. An evidence file is attacker-adjacent only in the mildest
  sense, but a 2 GB JSON document on a Jetson is a denial of service either way.

Placement and lifetime belong to the caller. Raw evidence is sensitive, so this
module has no default directory, writes nowhere on import, never overwrites
without being asked, and defaults :class:`Retention` to non-syncable. The
deployment that owns the disk owns the quota.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from leapcalib.contracts.failures import (
    FAILURE_EVIDENCE_CORRUPT,
    FAILURE_SCHEMA,
    CalibrationFailure,
)
from leapcalib.contracts.identity import TopologyIdentity, digest_of
from leapcalib.contracts.units import canonical_number
from leapcalib.contracts.observations import (
    Observation,
    observation_from_mapping,
    settled_fraction,
    total_valid_ratio,
    worst_stamp_uncertainty_s,
)

SCHEMA_VERSION = "leapcalib.evidence.v0"
"""Schema identity of an evidence bundle; readers refuse a version they do not know."""

DEFAULT_MAX_BYTES = 16 * 1024 * 1024
"""Read bound. Bundles hold sampled points and regions, not full image streams."""

MAX_OBSERVATIONS = 100000
"""Per-bundle observation bound, so one file cannot be unbounded work to load."""


@dataclass(frozen=True)
class Retention:
    """How long evidence may be kept and whether it may leave the machine.

    Non-syncable by default: depth samples of a workspace are observations of
    somebody's room, and the default for data nobody has consented to share is
    that it stays put.
    """

    ttl_s: float
    syncable: bool = False
    sensitivity: str = "raw_evidence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ttl_s", canonical_number(self.ttl_s))
        if self.ttl_s <= 0:
            raise ValueError("evidence retention needs a positive ttl")
        if type(self.syncable) is not bool:
            raise ValueError("syncability must be an explicit boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttl_s": float(self.ttl_s),
            "syncable": self.syncable,
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Retention":
        return cls(
            ttl_s=data["ttl_s"],
            syncable=data.get("syncable", False),
            sensitivity=str(data.get("sensitivity") or "raw_evidence"),
        )


@dataclass(frozen=True)
class EvidenceBundle:
    """One recorded, replayable set of observations with its provenance."""

    bundle_id: str
    topology: TopologyIdentity
    observations: tuple[Observation, ...]
    created_at_s: float
    retention: Retention
    accepted_ids: tuple[str, ...] = ()
    rejected_ids: tuple[str, ...] = ()
    holdout_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("an evidence bundle needs a nonempty id")
        if self.schema_version != SCHEMA_VERSION:
            raise CalibrationFailure(
                FAILURE_SCHEMA, f"unsupported evidence schema {self.schema_version!r}"
            )
        object.__setattr__(self, "created_at_s", canonical_number(self.created_at_s))
        object.__setattr__(self, "observations", tuple(self.observations))
        if len(self.observations) > MAX_OBSERVATIONS:
            raise ValueError(f"a bundle holds at most {MAX_OBSERVATIONS} observations")
        ids = [o.observation_id for o in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("observation ids must be unique within a bundle")
        for name in ("accepted_ids", "rejected_ids", "holdout_ids", "limitations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        groups = (set(self.accepted_ids), set(self.rejected_ids), set(self.holdout_ids))
        listed = (self.accepted_ids, self.rejected_ids, self.holdout_ids)
        if any(len(group) != len(values) for group, values in zip(groups, listed)):
            raise ValueError("observation ids must be unique within each group")
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("accepted, rejected and held-out observations must be disjoint")
        known = set(ids)
        unknown = (groups[0] | groups[1] | groups[2]) - known
        if unknown:
            raise ValueError(f"split references absent observations {sorted(unknown)}")

    @property
    def content(self) -> dict[str, Any]:
        """The canonical payload the digest is taken over."""
        return {
            "bundle_id": self.bundle_id,
            "schema_version": self.schema_version,
            "created_at_s": self.created_at_s,
            "topology": self.topology.to_dict(),
            "retention": self.retention.to_dict(),
            "observations": [o.to_dict() for o in self.observations],
            "accepted_ids": list(self.accepted_ids),
            "rejected_ids": list(self.rejected_ids),
            "holdout_ids": list(self.holdout_ids),
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.content)

    @property
    def summary(self) -> dict[str, Any]:
        """Readable quality facts, for a human deciding whether to solve with this.

        Every number here is descriptive. A bundle with a high valid ratio and
        fully settled joints is *loadable*, not accurate.
        """
        return {
            "bundle_id": self.bundle_id,
            "digest": self.digest,
            "topology_digest": self.topology.digest,
            "observations": len(self.observations),
            "accepted": len(self.accepted_ids),
            "rejected": len(self.rejected_ids),
            "holdout": len(self.holdout_ids),
            "depth_valid_ratio": total_valid_ratio(self.observations),
            "settled_fraction": settled_fraction(self.observations),
            "worst_stamp_uncertainty_s": worst_stamp_uncertainty_s(self.observations),
            "provenances": sorted({getattr(o, "provenance", "n/a") for o in self.observations}),
            "clock_domains": sorted({o.stamp.domain for o in self.observations}),
            "stamp_semantics": sorted({o.stamp.semantics for o in self.observations}),
            "limitations": list(self.limitations),
        }

    def with_split(
        self,
        *,
        accepted: Sequence[str] = (),
        rejected: Sequence[str] = (),
        holdout: Sequence[str] = (),
    ) -> "EvidenceBundle":
        """The same evidence with a different accepted/rejected/held-out split."""
        return replace(
            self,
            accepted_ids=tuple(accepted),
            rejected_ids=tuple(rejected),
            holdout_ids=tuple(holdout),
        )

    def to_dict(self) -> dict[str, Any]:
        """The file payload: content plus the digest a reader verifies."""
        return {"content": self.content, "digest": self.digest}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceBundle":
        """Rebuild a bundle from :meth:`to_dict`, verifying the stored digest."""
        if not isinstance(payload, Mapping) or not isinstance(payload.get("content"), Mapping):
            raise CalibrationFailure(FAILURE_SCHEMA, "evidence payload has no content object")
        content = payload["content"]
        version = str(content.get("schema_version") or "")
        if version != SCHEMA_VERSION:
            raise CalibrationFailure(FAILURE_SCHEMA, f"unsupported evidence schema {version!r}")
        stored = payload.get("digest")
        if not isinstance(stored, str) or not stored:
            raise CalibrationFailure(FAILURE_EVIDENCE_CORRUPT, "evidence payload has no digest")
        try:
            actual = digest_of(content)
        except (TypeError, ValueError) as error:
            raise CalibrationFailure(
                FAILURE_SCHEMA, f"invalid evidence content: {error}"
            ) from error
        if stored != actual:
            raise CalibrationFailure(
                FAILURE_EVIDENCE_CORRUPT,
                f"evidence digest mismatch for {content.get('bundle_id')!r}",
            )
        try:
            return cls._from_content(content)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
            raise CalibrationFailure(
                FAILURE_SCHEMA, f"invalid evidence content: {error}"
            ) from error

    @classmethod
    def _from_content(cls, content: Mapping[str, Any]) -> "EvidenceBundle":
        return cls(
            bundle_id=str(content["bundle_id"]),
            topology=TopologyIdentity.from_mapping(content["topology"]),
            observations=tuple(
                observation_from_mapping(o) for o in content.get("observations") or ()
            ),
            created_at_s=content["created_at_s"],
            retention=Retention.from_mapping(content["retention"]),
            accepted_ids=tuple(str(i) for i in content.get("accepted_ids") or ()),
            rejected_ids=tuple(str(i) for i in content.get("rejected_ids") or ()),
            holdout_ids=tuple(str(i) for i in content.get("holdout_ids") or ()),
            limitations=tuple(str(x) for x in content.get("limitations") or ()),
            schema_version=str(content["schema_version"]),
            metadata=dict(content.get("metadata") or {}),
        )


def write_bundle(
    path: str | os.PathLike[str], bundle: EvidenceBundle, *, overwrite: bool = False
) -> Path:
    """Write *bundle* as canonical JSON to *path*, returning the path written.

    Refuses to replace an existing file unless asked: evidence is the input a
    later result is judged against, so quietly overwriting a recording is a way
    to lose the only copy of what actually happened.
    """
    target = Path(path)
    encoded = json.dumps(bundle.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise CalibrationFailure(
            FAILURE_EVIDENCE_CORRUPT, f"{target} already holds evidence; pass overwrite"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(encoded)
    return target


def read_bundle(
    path: str | os.PathLike[str], *, max_bytes: int = DEFAULT_MAX_BYTES
) -> EvidenceBundle:
    """Read and verify a bundle, bounding how much is loaded."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("a positive byte budget is required")
    source = Path(path)
    with source.open("rb") as input_file:
        encoded = input_file.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise CalibrationFailure(
            FAILURE_EVIDENCE_CORRUPT,
            f"{source} is over the {max_bytes}-byte budget",
        )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise CalibrationFailure(
            FAILURE_EVIDENCE_CORRUPT, f"{source} is not valid evidence JSON: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CalibrationFailure(FAILURE_SCHEMA, f"{source} is not an evidence object")
    return EvidenceBundle.from_mapping(payload)


class ReplaySource:
    """An :class:`~leapcalib.contracts.observations.ObservationSource` over a bundle.

    The P0 implementation of the acquisition seam, and the only one that can
    exist offline. Observations are yielded exactly as recorded, including their
    original provenance: the bundle is what marks this as a replay, so rewriting
    each sample's provenance would destroy the record of how it was first
    measured.
    """

    def __init__(self, bundle: EvidenceBundle, *, only_accepted: bool = False) -> None:
        self._bundle = bundle
        self._only_accepted = only_accepted

    @property
    def bundle(self) -> EvidenceBundle:
        return self._bundle

    @property
    def topology_digest(self) -> str:
        return self._bundle.topology.digest

    def observations(self) -> Iterator[Observation]:
        accepted = set(self._bundle.accepted_ids)
        for observation in self._bundle.observations:
            if self._only_accepted and observation.observation_id not in accepted:
                continue
            yield observation

    def __len__(self) -> int:
        return sum(1 for _ in self.observations())


__all__ = [
    "DEFAULT_MAX_BYTES",
    "MAX_OBSERVATIONS",
    "SCHEMA_VERSION",
    "EvidenceBundle",
    "ReplaySource",
    "Retention",
    "read_bundle",
    "write_bundle",
]
