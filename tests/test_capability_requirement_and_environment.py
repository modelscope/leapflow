# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for structured adaptive capability requirements and environments."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from leapflow.analysis.environment_probe import EnvironmentProbe
from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest


def _manifest(*caps: Capability) -> PlatformManifest:
    return PlatformManifest(
        platform_id=PlatformID.DARWIN_15,
        os_version="15.0",
        capabilities=frozenset(caps),
    )


def test_capability_requirement_is_normalized_and_serializable() -> None:
    req = CapabilityRequirement.create(
        "json.pretty",
        "explicit_request",
        evidence="operator asked for JSON formatting",
        required_platform_capabilities=["file.ops"],
        approval_mode="autonomous_allowed",
        metadata={"source": "unit"},
        requirement_id="req-json",
    )

    assert req.capability == "json.pretty"
    assert req.required_platform_capabilities == ("file.ops",)
    assert req.allows_autonomous_approval is True
    assert req.to_dict()["metadata"] == {"source": "unit"}


def test_capability_requirement_rejects_empty_capability() -> None:
    with pytest.raises(ValueError, match="capability is required"):
        CapabilityRequirement.create("", "explicit_request")


def test_environment_fingerprint_is_stable_and_supports_capabilities() -> None:
    manifest = _manifest(Capability.FILE_OPS, Capability.SHELL_EXEC)
    fp_a = EnvironmentFingerprint.from_platform_manifest(
        manifest,
        workspace_root="/tmp/work",
        workspace_markers=("pyproject.toml", "README.md"),
    )
    fp_b = EnvironmentFingerprint.from_platform_manifest(
        manifest,
        workspace_root="/tmp/work",
        workspace_markers=("README.md", "pyproject.toml"),
    )

    assert fp_a.fingerprint_id == fp_b.fingerprint_id
    assert fp_a.supports_capability(Capability.SHELL_EXEC)
    assert fp_a.supports_all(["file.ops", "shell.exec"])
    assert fp_a.workspace_markers == ("README.md", "pyproject.toml")


def test_environment_fingerprint_changes_when_environment_changes() -> None:
    base = EnvironmentFingerprint.from_platform_manifest(_manifest(Capability.FILE_OPS))
    changed = EnvironmentFingerprint.from_platform_manifest(_manifest(Capability.SHELL_EXEC))

    assert base.fingerprint_id != changed.fingerprint_id


def test_environment_fingerprint_is_frozen() -> None:
    fp = EnvironmentFingerprint.from_platform_manifest(_manifest(Capability.FILE_OPS))

    with pytest.raises(FrozenInstanceError):
        fp.platform_id = "mutated"  # type: ignore[misc]


def test_environment_probe_uses_explicit_workspace_markers(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    probe = EnvironmentProbe(workspace_markers=("pyproject.toml", "package.json", "src"))

    fp = probe.probe(platform_manifest=_manifest(Capability.FILE_OPS), workspace_root=tmp_path)

    assert fp.workspace_markers == ("pyproject.toml", "src")
    assert "package.json" not in fp.workspace_markers
    assert fp.platform_capabilities == ("file.ops",)
