# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable environment fingerprint used by adaptive capability resolution.

The fingerprint is a compact, stable view of structured facts: platform
capabilities reported by the host and explicit workspace markers supplied by a
probe caller. It deliberately does not infer intent from natural language.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from leapflow.domain.platform import Capability, PlatformManifest


def _capability_value(value: Capability | str) -> str:
    """Normalize Capability enum members and strings to their wire values."""
    if isinstance(value, Capability):
        return value.value
    return str(value)


def _freeze_strs(values: Iterable[Capability | str] | Capability | str | None = None) -> tuple[str, ...]:
    """Return a stable, de-duplicated tuple of strings."""
    if not values:
        return ()
    if isinstance(values, (Capability, str)):
        normalized = _capability_value(values)
        return (normalized,) if normalized else ()
    return tuple(sorted({_capability_value(v) for v in values if _capability_value(v)}))


def _freeze_metadata(metadata: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
    """Convert arbitrary metadata to a stable immutable string map."""
    if not metadata:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in metadata.items()))


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Stable snapshot of the environment facts a resolver may use."""

    platform_id: str
    os_version: str = ""
    platform_capabilities: tuple[str, ...] = field(default_factory=tuple)
    workspace_root: str = ""
    workspace_markers: tuple[str, ...] = field(default_factory=tuple)
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_platform_manifest(
        cls,
        manifest: PlatformManifest,
        *,
        workspace_root: str | Path = "",
        workspace_markers: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EnvironmentFingerprint":
        """Build a fingerprint from a host PlatformManifest."""
        return cls(
            platform_id=manifest.platform_id.value,
            os_version=str(manifest.os_version or ""),
            platform_capabilities=_freeze_strs(manifest.capabilities),
            workspace_root=str(workspace_root or ""),
            workspace_markers=_freeze_strs(workspace_markers),
            metadata=_freeze_metadata(metadata),
        )

    @property
    def fingerprint_id(self) -> str:
        """Stable content hash for comparing before/after environment state."""
        return self._content_hash()

    def supports_capability(self, capability: Capability | str) -> bool:
        """Return whether the host reports a platform capability."""
        return _capability_value(capability) in set(self.platform_capabilities)

    def supports_all(self, capabilities: Iterable[Capability | str]) -> bool:
        """Return whether all requested platform capabilities are present."""
        available = set(self.platform_capabilities)
        return all(_capability_value(c) in available for c in capabilities)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "platform_id": self.platform_id,
            "os_version": self.os_version,
            "platform_capabilities": list(self.platform_capabilities),
            "workspace_root": self.workspace_root,
            "workspace_markers": list(self.workspace_markers),
            "metadata": dict(self.metadata),
        }

    def _content_hash(self) -> str:
        """Hash the fields excluding the hash field itself."""
        payload = {
            "platform_id": self.platform_id,
            "os_version": self.os_version,
            "platform_capabilities": list(self.platform_capabilities),
            "workspace_root": self.workspace_root,
            "workspace_markers": list(self.workspace_markers),
            "metadata": dict(self.metadata),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
