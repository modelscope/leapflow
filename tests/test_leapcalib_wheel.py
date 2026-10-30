# Copyright (c) Alibaba, Inc. and its affiliates.
"""Build and exercise an isolated distribution; packaging is not physical qualification."""

from __future__ import annotations

import configparser
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

pytestmark = pytest.mark.component
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project = tmp_path_factory.mktemp("calibration-wheel")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(
        ROOT / "src",
        project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('wheelhouse')",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = tuple((project / "wheelhouse").glob("*.whl"))
    assert len(wheels) == 1, result.stdout
    return wheels[0]


def test_wheel_contains_exact_calibration_sources_and_registered_entry_points(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for package in ("leapcalib", "leapflow/hardware/calibration"):
            for source in (ROOT / "src" / package).rglob("*.py"):
                relative = source.relative_to(ROOT / "src").as_posix()
                assert relative in names, relative
                assert (
                    hashlib.sha256(archive.read(relative)).digest()
                    == hashlib.sha256(source.read_bytes()).digest()
                )
        metadata_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(metadata_name).decode())
        assert entries["console_scripts"]["leap-calib"] == "leapcalib.cli:main"
        assert (
            entries["leapcalib.methods"]["raw_joint_response"]
            == "leapcalib.estimation.raw_response:RawResponseMethod"
        )
        assert (
            entries["leapflow.hardware.skills"]["builtin_governed_calibration"]
            == "leapflow.hardware.skills.builtin.governed_calibration:skills"
        )
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_numerical_and_vision_dependencies_remain_optional_in_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
    extras = metadata.get_all("Provides-Extra", [])
    assert "leapcalib" in extras and "vision" in extras
    dependencies = metadata.get_all("Requires-Dist", [])
    for dependency in dependencies:
        if dependency.lower().startswith(("numpy", "opencv", "torch", "scipy")):
            assert "extra ==" in dependency, dependency


def test_installed_wheel_discovery_is_lazy_and_compute_only(wheel: Path) -> None:
    code = """
import importlib, importlib.metadata, pkgutil, sys
from pathlib import Path
wheel = sys.argv[1]
sys.path.insert(0, wheel)
import leapcalib
assert leapcalib.__file__.startswith(wheel), leapcalib.__file__
for module in pkgutil.walk_packages(leapcalib.__path__, leapcalib.__name__ + '.'):
    importlib.import_module(module.name)
distribution = next(importlib.metadata.distributions(path=[wheel]))
methods = {entry.name: entry.load()() for entry in distribution.entry_points if entry.group == 'leapcalib.methods'}
assert set(methods) == {'metric_motion', 'task_response', 'raw_joint_response'}
from leapcalib.contracts.methods import CalibrationMethod
assert all(isinstance(method, CalibrationMethod) for method in methods.values())
assert not any(name.startswith('leapflow') for name in sys.modules)
import leapflow.hardware.calibration.worker
import leapflow.hardware.calibration.rollback
import leapflow.hardware.calibration.scene
from leapflow.hardware.skills.builtin.governed_calibration import skills
assert {skill.spec.skill_id for skill in skills()} >= {'calib.collect', 'calib.align_reference', 'calib.inspect', 'calib.rollback', 'calib.assess_motion', 'calib.collect_poses'}
assert not {'numpy', 'cv2', 'pyrealsense2', 'rclpy', 'torch'} & sys.modules.keys()
assert not any(name.startswith('leapflow.engine') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheel)],
        cwd=wheel.parent,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_preview_templates_refuse_unmeasured_operator_values() -> None:
    from leapflow.hardware.calibration.acquisition import CollectionProfile
    from leapflow.hardware.calibration.pose_policy import PoseCollectionPolicy

    tool = runpy.run_path(str(ROOT / "tools" / "calib_preview.py"))
    template = tool["camera_template"]()
    with pytest.raises((TypeError, ValueError)):
        CollectionProfile.from_mapping(template["calibration_acquisition"])
    with pytest.raises((TypeError, ValueError)):
        PoseCollectionPolicy.from_mapping(template["calibration_motion"])
    policy = template["calibration_motion"]
    assert policy["reference_acquisition"] is None
    assert policy["speed_pct_max"] == 10 and policy["max_moves"] == 8
    assert policy["geometry"]["minimum_clearance_mm"] == 30
    assert policy["geometry"]["boundary_margin_mm"] == 20
    assert tool["trial_template"]()["pairs"] is None
    assert "args" not in tool["trial_template"]()


@pytest.mark.parametrize(
    "content",
    [
        '<testsuites><testsuite tests="1" failures="1"/></testsuites>',
        '<testsuites><testsuite tests="0"/></testsuites>',
        '<!DOCTYPE testsuite [<!ENTITY payload "unsafe">]><testsuite tests="1"/>',
    ],
)
def test_preview_refuses_failed_empty_or_entity_reports(tmp_path: Path, content: str) -> None:
    tool = runpy.run_path(str(ROOT / "tools" / "calib_preview.py"))
    report = tmp_path / "verification.xml"
    report.write_text(content)
    with pytest.raises(ValueError):
        tool["verification"]([report])


def test_preview_binds_offline_preparation_and_refuses_drift(tmp_path: Path) -> None:
    prepare = runpy.run_path(str(ROOT / "tools/calib_prepare.py"))
    preview = runpy.run_path(str(ROOT / "tools/calib_preview.py"))
    from tests.test_calibration_preparation import probe

    source = tmp_path / "probe.json"
    source.write_text(json.dumps(probe()))
    draft = tmp_path / "draft"
    prepare["write_preparation"](draft, prepare["prepare"](probe(), {}), [source])
    manifest = draft / "preparation.json"
    summary, artifacts = preview["preparation_artifacts"](manifest)
    assert summary["pose_batches_generated"] == 0 and summary["physical_qualification"] is False
    assert summary["readiness_blockers"] and manifest in artifacts
    (draft / "collect.args.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        preview["preparation_artifacts"](manifest)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", ".."])
def test_preview_refuses_preparation_artifact_path_escape(tmp_path: Path, name: str) -> None:
    preview = runpy.run_path(str(ROOT / "tools/calib_preview.py"))
    manifest = tmp_path / "preparation.json"
    manifest.write_text(json.dumps({"schema": "leapflow.camera_preparation.v1", "physical_qualification": False,
        "live_configuration_written": False, "motion_performed": False, "artifact_sha256": {name: "a" * 64}}))
    with pytest.raises(ValueError, match="filenames"):
        preview["preparation_artifacts"](manifest)


def test_preview_records_exact_passing_verification_digest(tmp_path: Path) -> None:
    tool = runpy.run_path(str(ROOT / "tools" / "calib_preview.py"))
    report = tmp_path / "verification.xml"
    report.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>'
    )
    recorded = tool["verification"]([report])[0]
    assert recorded["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert recorded["tests"] == 3 and recorded["failures"] == 0 and recorded["skipped"] == 1
