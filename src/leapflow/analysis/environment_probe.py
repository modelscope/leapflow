# Copyright (c) Alibaba, Inc. and its affiliates.
"""Structured environment probing for adaptive capability selection.

The probe only observes explicit structural facts supplied by its caller:
platform capabilities from ``PlatformManifest`` and configured workspace marker
paths. It never classifies free-form user text or source text.
"""

from __future__ import annotations

import platform as platform_module
from pathlib import Path
from typing import Iterable

from leapflow.analysis.environment_catalog import EnvironmentCatalog
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.platform import PlatformID, PlatformManifest


class EnvironmentProbe:
    """Build EnvironmentFingerprint snapshots from structured inputs."""

    def __init__(self, workspace_markers: Iterable[str] = ()) -> None:
        self._workspace_markers = tuple(str(m) for m in workspace_markers if str(m))

    @classmethod
    def from_catalog(cls, catalog: EnvironmentCatalog) -> "EnvironmentProbe":
        """Build a probe from a declarative environment marker catalog."""
        return cls(catalog.marker_paths())

    def probe(
        self,
        *,
        platform_manifest: PlatformManifest | None = None,
        workspace_root: str | Path = "",
        catalog: EnvironmentCatalog | None = None,
    ) -> EnvironmentFingerprint:
        """Return a stable environment fingerprint.

        ``workspace_markers`` are explicit relative paths supplied at construction
        time. No built-in filename taxonomy is used here; callers can pass a
        config-derived marker set appropriate for their experiment or product
        surface.
        """
        manifest = platform_manifest or PlatformManifest(
            platform_id=PlatformID.resolve(),
            os_version=platform_module.platform(),
            capabilities=frozenset(),
        )
        root = Path(workspace_root).expanduser() if workspace_root else Path()
        marker_catalog = catalog or EnvironmentCatalog.from_markers(self._workspace_markers)
        present = (
            tuple(marker.path for marker in marker_catalog.present_markers(root))
            if workspace_root
            else ()
        )
        metadata = marker_catalog.metadata_for(root) if workspace_root else {}
        return EnvironmentFingerprint.from_platform_manifest(
            manifest,
            workspace_root=str(root) if workspace_root else "",
            workspace_markers=present,
            metadata=metadata,
        )

    def _present_markers(self, root: Path) -> tuple[str, ...]:
        """Return configured marker paths that exist under the workspace root."""
        found: list[str] = []
        for marker in self._workspace_markers:
            candidate = root / marker
            if candidate.exists():
                found.append(marker)
        return tuple(sorted(found))
