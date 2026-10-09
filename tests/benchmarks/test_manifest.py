# Copyright (c) Alibaba, Inc. and its affiliates.
"""Strict manifest loading and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmarks.manifest import load_manifest, load_manifests


VALID_MANIFEST = """\
id: sample
version: "1.0.0"
adapter: sample_adapter
tier: 1
required: true
seeds: [42]
timeouts:
  trial: 10
scenarios:
  - scenario_id: first
    name: First scenario
"""


def _write(tmp_path: Path, text: str, name: str = "manifest.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid_manifest(tmp_path: Path) -> None:
    manifest, errors = load_manifest(_write(tmp_path, VALID_MANIFEST))

    assert errors == []
    assert manifest is not None
    assert manifest.id == "sample"
    assert manifest.adapter == "sample_adapter"
    assert manifest.scenarios[0].adapter_id == "sample_adapter"


def test_official_contract_is_preserved_and_must_be_a_mapping(tmp_path: Path) -> None:
    valid = VALID_MANIFEST + "official:\n  implementation: official/repo\n"
    manifest, errors = load_manifest(_write(tmp_path, valid))

    assert errors == []
    assert manifest is not None
    assert manifest.official == {"implementation": "official/repo"}
    assert manifest.to_dict()["official"] == {"implementation": "official/repo"}

    invalid, errors = load_manifest(_write(tmp_path, VALID_MANIFEST + "official: invalid\n"))
    assert invalid is None
    assert any(error.field == "official" for error in errors)


@pytest.mark.parametrize("field", ["id", "version", "adapter", "tier"])
def test_missing_required_field_is_rejected(tmp_path: Path, field: str) -> None:
    lines = [line for line in VALID_MANIFEST.splitlines() if not line.startswith(f"{field}:")]
    manifest, errors = load_manifest(_write(tmp_path, "\n".join(lines)))

    assert manifest is None
    assert any(error.field == field and "required" in error.message for error in errors)


def test_unknown_root_and_scenario_fields_are_rejected(tmp_path: Path) -> None:
    text = VALID_MANIFEST.replace("required: true", "required: true\nunknown_root: value")
    text = text.replace("    name: First scenario", "    name: First scenario\n    mystery: value")
    manifest, errors = load_manifest(_write(tmp_path, text))

    assert manifest is None
    assert {error.field for error in errors} >= {"unknown_root", "scenarios[0].mystery"}


def test_duplicate_scenario_id_is_rejected(tmp_path: Path) -> None:
    text = VALID_MANIFEST + """\
  - scenario_id: first
    name: Duplicate
"""
    manifest, errors = load_manifest(_write(tmp_path, text))

    assert manifest is None
    assert any("duplicate scenario id" in error.message for error in errors)


@pytest.mark.parametrize(
    ("replacement", "field"),
    [
        ("tier: one", "tier"),
        ("required: 1", "required"),
        ("seeds: [true]", "seeds"),
        ("timeouts:\n  trial: 0", "timeouts.trial"),
        ("scenarios: invalid", "scenarios"),
    ],
)
def test_strict_field_types_are_rejected(
    tmp_path: Path, replacement: str, field: str,
) -> None:
    key = replacement.split(":", 1)[0]
    lines = VALID_MANIFEST.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{key}:"))
    end = start + 1
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    lines[start:end] = replacement.splitlines()

    manifest, errors = load_manifest(_write(tmp_path, "\n".join(lines)))

    assert manifest is None
    assert any(error.field == field for error in errors)


def test_directory_load_is_stable_and_keeps_valid_files(tmp_path: Path) -> None:
    _write(tmp_path, VALID_MANIFEST.replace("sample", "zeta", 1), "z.yaml")
    _write(tmp_path, VALID_MANIFEST.replace("sample", "alpha", 1), "a.yml")
    _write(tmp_path, "id: broken\n", "broken.yaml")
    (tmp_path / "ignored.txt").write_text(VALID_MANIFEST, encoding="utf-8")

    manifests, errors = load_manifests(tmp_path)

    assert [manifest.id for manifest in manifests] == ["alpha", "zeta"]
    assert errors


def _repo_root() -> Path:
    """Locate the checkout without relying on the test process working directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "benchmarks").is_dir():
            return candidate
    raise AssertionError("repository root not found")


def _load_source_catalog(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    sources = data.get("sources")
    assert isinstance(sources, list)
    return {str(item["id"]): item for item in sources}


def test_all_shipped_manifests_are_valid() -> None:
    root = _repo_root() / "src" / "benchmarks" / "manifests"
    manifests = []
    errors = []
    for directory in (root / "native", root / "external", root / "hardware"):
        loaded, found_errors = load_manifests(directory)
        manifests.extend(loaded)
        errors.extend(found_errors)

    assert errors == []
    assert len(manifests) == 27
    assert len({manifest.id for manifest in manifests}) == 27


def test_external_manifests_match_shipped_source_catalog() -> None:
    root = _repo_root() / "src" / "benchmarks" / "manifests"
    catalog = _load_source_catalog(root / "sources.yaml")
    manifests, errors = load_manifests(root / "external")

    assert errors == []
    assert len(manifests) == 12
    catalog_manifests = [manifest for manifest in manifests if manifest.id != "live_llm"]
    assert len(catalog_manifests) == 11
    for manifest in catalog_manifests:
        assert manifest.id in catalog
        assert manifest.official, f"{manifest.id} must declare its official contract"
        assert manifest.official.get("implementation")
        assert manifest.official.get("dataset")
        assert manifest.official.get("result_contract")
        for scenario in manifest.scenarios:
            declared = scenario.parameters.get("official_metrics")
            assert declared, f"{manifest.id}/{scenario.scenario_id} lacks official metric contract"
            assert set(declared).issubset(manifest.metrics)
        source = catalog[manifest.id]
        for manifest_field, source_field in (
            ("source", "homepage"),
            ("repo", "repository"),
            ("paper", "paper"),
        ):
            expected = source.get(source_field)
            actual = getattr(manifest, manifest_field)
            if expected is None or str(expected).lower() == "unknown":
                assert actual in ("", "UNKNOWN")
            else:
                assert actual == expected
        assert manifest.license == source["manifest_license"]


def test_discussion_source_catalog_mirrors_shipped_catalog() -> None:
    repo_root = _repo_root()
    discussion_source = repo_root / "temp" / "embodied_agent" / "benchmarks" / "sources.yaml"
    if not discussion_source.is_file():
        pytest.skip("gitignored discussion mirror is not available")

    shipped_source = repo_root / "src" / "benchmarks" / "manifests" / "sources.yaml"
    assert yaml.safe_load(discussion_source.read_text(encoding="utf-8")) == yaml.safe_load(
        shipped_source.read_text(encoding="utf-8")
    )


def test_external_benchmarks_require_explicit_official_configuration() -> None:
    root = _repo_root() / "src" / "benchmarks" / "manifests" / "external"
    data_backed = {
        "asimov", "attackvla", "calvin", "embodyguard", "is_bench",
        "kinder", "robojailbench", "vlabench",
    }
    manifests, errors = load_manifests(root)

    assert errors == []
    for manifest in manifests:
        if manifest.id == "live_llm":
            continue
        assert f"config:benchmark.commands.{manifest.id}" in manifest.dependencies
        if manifest.id in data_backed:
            assert f"config:benchmark.data_roots.{manifest.id}" in manifest.dependencies
        assert "benchmark.commands" in manifest.setup_guidance
        assert manifest.official.get("result_contract")
