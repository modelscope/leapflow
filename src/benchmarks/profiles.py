# Copyright (c) Alibaba, Inc. and its affiliates.
"""Stable benchmark tier profiles.

Tier taxonomy:
  0 — import and contract checks (no external dependencies)
  1 — deterministic unit/simulation benchmarks
  2 — integration benchmarks (services, sandboxes)
  3 — pre-hardware safety and fleet simulation
  4 — real hardware / production environment

The ``pre-hardware`` profile includes Tiers 0 through 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from benchmarks.models import BenchmarkManifest


@dataclass(frozen=True)
class BenchmarkProfile:
    """Named selection of benchmark tiers."""

    profile_id: str
    tiers: tuple[int, ...]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def includes(self, manifest: BenchmarkManifest) -> bool:
        """Return True if the manifest belongs to this profile."""
        if manifest.tier not in self.tiers:
            return False
        if not self.tags:
            return True
        return bool(set(self.tags).intersection(manifest.tags))


# Stable ordered profile IDs — do not reorder; CLI and CI depend on this.
PROFILE_IDS: tuple[str, ...] = (
    "tier0",
    "tier1",
    "tier2",
    "tier3",
    "tier4",
    "live-llm",
    "production-sim",
    "pre-hardware",
    "all",
)

_PROFILES: dict[str, BenchmarkProfile] = {
    "tier0": BenchmarkProfile(
        "tier0", (0,), "Import safety and protocol contract checks",
    ),
    "tier1": BenchmarkProfile(
        "tier1", (1,), "Deterministic unit and simulation benchmarks",
    ),
    "tier2": BenchmarkProfile(
        "tier2", (2,), "Service and sandbox integration benchmarks",
    ),
    "tier3": BenchmarkProfile(
        "tier3", (3,), "Pre-hardware safety and fleet simulation",
    ),
    "tier4": BenchmarkProfile(
        "tier4", (4,), "Real hardware and production environment",
    ),
    "live-llm": BenchmarkProfile(
        "live-llm", (5,), "Explicitly authorized live-provider evaluation",
    ),
    "production-sim": BenchmarkProfile(
        "production-sim", (1,), "Production hardware-settings simulation", ("production-sim",),
    ),
    "pre-hardware": BenchmarkProfile(
        "pre-hardware", (0, 1, 2, 3), "All tiers not requiring real hardware",
    ),
    "all": BenchmarkProfile(
        "all", (0, 1, 2, 3, 4), "All benchmark tiers",
    ),
}


def get_profile(profile_id: str) -> BenchmarkProfile | None:
    """Return a profile by id, accepting legacy hyphenated tier aliases."""
    normalized = profile_id.lower().replace("tier-", "tier")
    return _PROFILES.get(normalized)


def list_profiles() -> tuple[BenchmarkProfile, ...]:
    """Return all profiles in stable order."""
    return tuple(_PROFILES[pid] for pid in PROFILE_IDS)


def select_manifests(
    manifests: Sequence[BenchmarkManifest],
    profile_id: str,
) -> tuple[BenchmarkManifest, ...]:
    """Select manifests matching a profile, sorted by benchmark id."""
    profile = get_profile(profile_id)
    if profile is None:
        return ()
    return tuple(sorted((m for m in manifests if profile.includes(m)), key=lambda m: m.id))


__all__ = [
    "BenchmarkProfile",
    "PROFILE_IDS",
    "get_profile",
    "list_profiles",
    "select_manifests",
]
