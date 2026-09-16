# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declarative environment marker catalog for adaptive capability selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EnvironmentMarker:
    """One explicit structural marker that may be present in a workspace."""

    path: str
    category: str = "workspace"
    source: str = "catalog"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "source": self.source,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentMarker":
        return cls(
            path=str(data.get("path") or ""),
            category=str(data.get("category") or "workspace"),
            source=str(data.get("source") or "catalog"),
            tags=tuple(str(item) for item in data.get("tags") or ()),
        )


@dataclass(frozen=True)
class EnvironmentCatalog:
    """A declarative list of workspace markers, independent of user text."""

    markers: tuple[EnvironmentMarker, ...] = field(default_factory=tuple)

    @classmethod
    def from_markers(
        cls, markers: Iterable[str | Mapping[str, Any] | EnvironmentMarker]
    ) -> "EnvironmentCatalog":
        parsed: list[EnvironmentMarker] = []
        for marker in markers:
            if isinstance(marker, EnvironmentMarker):
                parsed.append(marker)
            elif isinstance(marker, Mapping):
                item = EnvironmentMarker.from_dict(marker)
                if item.path:
                    parsed.append(item)
            else:
                text = str(marker or "")
                if text:
                    parsed.append(EnvironmentMarker(path=text))
        return cls(markers=tuple(parsed))

    def marker_paths(self) -> tuple[str, ...]:
        return tuple(marker.path for marker in self.markers if marker.path)

    def present_markers(self, workspace_root: str | Path) -> tuple[EnvironmentMarker, ...]:
        root = Path(workspace_root).expanduser()
        return tuple(marker for marker in self.markers if (root / marker.path).exists())

    def metadata_for(self, workspace_root: str | Path) -> dict[str, str]:
        present = self.present_markers(workspace_root)
        tags = sorted({tag for marker in present for tag in marker.tags})
        categories = sorted({marker.category for marker in present if marker.category})
        sources = sorted({marker.source for marker in present if marker.source})
        return {
            "environment_marker_tags": ",".join(tags),
            "environment_marker_categories": ",".join(categories),
            "environment_marker_sources": ",".join(sources),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"markers": [marker.to_dict() for marker in self.markers]}


__all__ = ["EnvironmentCatalog", "EnvironmentMarker"]
