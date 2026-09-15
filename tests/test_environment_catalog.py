# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for declarative environment marker catalogs."""

from __future__ import annotations

from leapflow.analysis.environment_catalog import EnvironmentCatalog, EnvironmentMarker
from leapflow.analysis.environment_probe import EnvironmentProbe
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest


def test_environment_catalog_reports_present_marker_metadata(tmp_path) -> None:
    (tmp_path / "package.json").write_text('{"name":"demo"}', encoding="utf-8")
    catalog = EnvironmentCatalog.from_markers(
        [
            EnvironmentMarker(
                path="package.json",
                category="runtime",
                source="test",
                tags=("node", "frontend"),
            ),
            EnvironmentMarker(
                path="pyproject.toml", category="runtime", source="test", tags=("python",)
            ),
        ]
    )

    present = catalog.present_markers(tmp_path)
    metadata = catalog.metadata_for(tmp_path)

    assert [marker.path for marker in present] == ["package.json"]
    assert metadata["environment_marker_tags"] == "frontend,node"
    assert metadata["environment_marker_categories"] == "runtime"


def test_environment_probe_from_catalog_includes_marker_metadata(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    catalog = EnvironmentCatalog.from_markers(
        [{"path": "pyproject.toml", "category": "language", "source": "unit", "tags": ["python"]}]
    )
    manifest = PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset({Capability.FILE_OPS}))

    fingerprint = EnvironmentProbe.from_catalog(catalog).probe(
        platform_manifest=manifest,
        workspace_root=tmp_path,
        catalog=catalog,
    )

    assert fingerprint.workspace_markers == ("pyproject.toml",)
    metadata = dict(fingerprint.metadata)
    assert metadata["environment_marker_tags"] == "python"
    assert metadata["environment_marker_categories"] == "language"
