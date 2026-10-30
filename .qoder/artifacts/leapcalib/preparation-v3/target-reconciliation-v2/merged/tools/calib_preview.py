# Copyright (c) Alibaba, Inc. and its affiliates.
"""Build a local fingerprinted camera-only preview; never deploy, connect, or actuate.

Templates deliberately contain null operator inputs and are not live configuration.
The wheel contains the entire current source tree, including unrelated dirty changes.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
IGNORES = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
REQUIRED_SKILLS = {"calib.collect", "calib.assess_motion", "calib.collect_poses"}


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files = [root / name for name in ("pyproject.toml", "README.md", "LICENSE", "uv.lock")]
    files += [
        path
        for path in (root / "src").rglob("*")
        if path.is_file()
        and not any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)
        and path.suffix != ".pyc"
    ]
    files += list((root / "tools").glob("calib*.py"))
    files += [root / "tools/piper/piper.py"] if (root / "tools/piper/piper.py").is_file() else []
    files += list((root / "tools/piper/piper_ops").glob("*.py"))
    return {
        path.relative_to(root).as_posix(): fingerprint(path.read_bytes()) for path in sorted(files)
    }


def camera_template() -> dict[str, Any]:
    """Draft partial workcell fields; null measurements must never reach a live reload."""
    settled_names = (
        "max_feedback_age_s max_coherence_s feedback_uncertainty_s exposure_uncertainty_s "
        "receipt_uncertainty_s max_joint_drift_rad max_tcp_drift_mm max_tcp_rotation_deg "
        "joint_uncertainty_rad tcp_uncertainty_mm rotation_uncertainty_deg poll_s timeout_s"
    ).split()
    return {
        "calibration_acquisition": {
            "camera_role": None,
            "camera_method": "wrist",
            "allowed_modes": None,
            "settled": {**dict.fromkeys(settled_names), "min_samples": 3, "span_s": 0.3},
            "ttl_s": None,
            "quota_bytes": None,
            "max_frames": 16,
        },
        "calibration_motion": {
            "reference_acquisition": None,
            "mount": {
                "mount_id": None,
                "parent_frame": "flange",
                "source": None,
                "seated_at_s": None,
                "notes": "Operator-confirmed unchanged wrist mount",
            },
            "geometry": {
                "link_radii_mm": None,
                "whole_link_workspace": {"x": [None, None], "y": [None, None], "z": [None, None]},
                "keepouts": None,
                "provenance_digest": None,
                "uncertainty_mm": None,
                "sensing_mm": None,
                "timing_mm": None,
                "arrival_mm": None,
                "minimum_clearance_mm": 30,
                "boundary_margin_mm": 20,
                "path_step_deg": 0.5,
                "max_path_samples": 4096,
            },
            "joint_rate_bounds_rad_s": None,
            "endpoint_rate_bound_mm_s": None,
            "rate_provenance_digest": None,
            "speed_pct_max": 10,
            "campaign_translation_mm": 10,
            "campaign_yaw_deg": 3,
            "campaign_joint_deg": 5,
            "max_moves": 8,
            "max_travel_mm": 40,
            "max_joint_l1_deg": 12,
            "max_duration_s": 120,
        },
    }


def trial_template() -> dict[str, Any]:
    """Args-only JSON for piper calibration run --args-file; no command wrapper."""
    return {"pairs": None, "training": None, "mount": None, "prior": None,
            "prior_uncertainty": None, "tracking": {}, "solve": {}, "evaluation": {}}


def git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def check_wheel(wheel: Path, project: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        entries = configparser.ConfigParser()
        entries.read_string(
            archive.read(
                next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
            ).decode()
        )
        assert entries["leapflow.hardware.skills"]["builtin_governed_calibration"] == (
            "leapflow.hardware.skills.builtin.governed_calibration:skills"
        )
        assert entries["console_scripts"]["leap-calib"] == "leapcalib.cli:main"
        source_hashes = {}
        for source in (project / "src").rglob("*"):
            if not source.is_file():
                continue
            name = source.relative_to(project / "src").as_posix()
            if name not in names:
                if source.suffix == ".py":
                    raise ValueError(f"wheel omitted Python source: {name}")
                continue
            if archive.read(name) != source.read_bytes():
                raise ValueError(f"wheel source drift: {name}")
            source_hashes[name] = fingerprint(archive.read(name))
        metadata = Parser().parsestr(
            archive.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            ).decode()
        )
        for dependency in metadata.get_all("Requires-Dist", []):
            if dependency.lower().startswith(("numpy", "opencv", "torch", "scipy")):
                assert "extra ==" in dependency, dependency
    code = """
import importlib.metadata, sys
sys.path.insert(0, sys.argv[1])
import leapflow
assert leapflow.__file__.startswith(sys.argv[1])
from leapflow.hardware.skills.builtin.governed_calibration import skills
found = {s.spec.skill_id: s.spec for s in skills()}
assert {'calib.collect', 'calib.assess_motion', 'calib.collect_poses'} <= found.keys()
assert found['calib.assess_motion'].effect == 'read'
assert found['calib.assess_motion'].runs_leased and not found['calib.assess_motion'].uses_motion
assert found['calib.collect_poses'].uses_motion
assert all(found[k].consent == 'each_run' for k in ('calib.assess_motion', 'calib.collect_poses'))
assert not {'numpy', 'cv2', 'pyrealsense2', 'rclpy', 'torch'} & sys.modules.keys()
assert not any(n.startswith('leapflow.engine') for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheel)],
        cwd=project,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return {
        "version": metadata["Version"],
        "packaged_source_sha256": source_hashes,
        "required_skills": sorted(REQUIRED_SKILLS),
        "isolated_discovery": "passed",
    }


def verification(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        with path.open("rb") as source:
            raw = source.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("verification report exceeds 16MiB")
        text = raw.decode("utf-8-sig")
        if "\x00" in text or any(token in text.upper() for token in ("<!DOCTYPE", "<!ENTITY")):
            raise ValueError("verification report must be UTF-8 XML without DTD or entities")
        suites = list(ElementTree.fromstring(text).iter("testsuite"))
        totals = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        if totals["failures"] or totals["errors"] or not totals["tests"]:
            raise ValueError(f"nonpassing or empty verification report: {path}")
        reports.append({"file": path.name, "sha256": fingerprint(raw), **totals})
    return reports


def preparation_artifacts(path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Verify an offline preparation manifest without promoting it to live readiness."""
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("preparation report exceeds 4MiB")
    report = json.loads(raw)
    if report.get("schema") != "leapflow.camera_preparation.v1" or any(
        report.get(key) is not False for key in ("physical_qualification", "live_configuration_written", "motion_performed")
    ):
        raise ValueError("offline unqualified preparation report required")
    artifacts = [path]
    hashes = report.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("fingerprinted preparation artifacts required")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("preparation artifacts must be sibling filenames")
        artifact = path.parent / name
        if artifact.is_symlink() or not artifact.is_file() or fingerprint(artifact.read_bytes()) != expected:
            raise ValueError("preparation artifact missing or changed")
        artifacts.append(artifact)
    return {"file": "preparation/" + path.name, "sha256": fingerprint(raw),
            "state": report["state"], "unresolved": report["unresolved"],
            "readiness_blockers": report.get("readiness_blockers", []),
            "pose_batches_generated": len(report["pose_batches"]), "physical_qualification": False}, artifacts


def build_preview(output: Path, reports: list[Path], preparation: Path | None = None) -> dict[str, Any]:
    """Snapshot, build and verify locally; refuse overwritten or drifting previews."""
    import yaml

    verified = verification(reports)
    prepared, artifacts = preparation_artifacts(preparation) if preparation else (None, [])
    before = inventory(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    project = output / "snapshot"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "uv.lock"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(ROOT / "src", project / "src", ignore=IGNORES)
    for name in before:
        if name.startswith("tools/"):
            destination = project / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
    if inventory(project) != before:
        raise ValueError("source changed during snapshot; preview is not valid")
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
        timeout=180,
    )
    (output / "build.log").write_text(result.stdout + result.stderr)
    wheels = list((project / "wheelhouse").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("build must produce exactly one wheel")
    wheel = output / wheels[0].name
    shutil.copy2(wheels[0], wheel)
    checked = check_wheel(wheel, project)
    if inventory(ROOT) != before:
        raise ValueError("source changed during build; preview is not valid")
    (output / "camera-workcell.template.yaml").write_text(
        "# DRAFT ONLY: nulls are unresolved inputs, not necessarily human measurements.\n"
        "# Stage A: omit calibration_motion, approve stationary capture, retain indexed anchor.\n"
        "# Stage B: fill calibration_motion, review reload impact, approve each bounded pose run.\n"
        + yaml.safe_dump(camera_template(), sort_keys=False)
    )
    (output / "camera-trial.template.json").write_text(
        json.dumps(trial_template(), indent=2) + "\n"
    )
    for path in reports:
        shutil.copy2(path, output / path.name)
    if artifacts:
        target = output / "preparation"
        target.mkdir()
        for path in artifacts:
            shutil.copy2(path, target / path.name)
        copied, _ = preparation_artifacts(target / preparation.name)
        if copied != prepared:
            raise ValueError("preparation changed during preview snapshot")
    preview = {
        "schema": "leapflow.camera_deployment_preview.v1",
        "state": "awaiting_operator_inputs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--porcelain=v1"]),
        "source_sha256": before,
        "source_inventory_sha256": fingerprint(json.dumps(before, sort_keys=True).encode()),
        "wheel": {"file": wheel.name, "sha256": fingerprint(wheel.read_bytes()), **checked},
        "build_environment": {"python": sys.version, "platform": platform.platform()},
        "verification": verified,
        "preparation": prepared,
        "contains_all_current_source_changes": True,
        "operational_tools_packaged_in_wheel": False,
        "operational_tools_snapshot": "snapshot/tools/piper",
        "prerequisite_tools_snapshot": "snapshot/tools",
        "preserve_remote_dirty_files_before_deployment": True,
        "wheel_install_is_not_an_editable_source_sync": True,
        "activation": "none",
        "physical_admission": "not_established",
        "auto_return": False,
        "remote_changes_performed": False,
        "physical_motion_performed": False,
        "approval_required": [
            "exact deployment commands and impact",
            "config application/reload",
            "any owner restart",
            "stationary capture",
            "each pose run",
            "trial",
        ],
        "prerequisite_inputs": [
            "Target existing owner, service identity, interpreter and current package fingerprint",
            "Existing workcell/device declarations, short private profile root, rollback package",
            "Unchanged wrist camera mount, metric RGB-D profile, timestamp uncertainty bounds",
            "Settled thresholds, nominal FK and feedback uncertainty, retention TTL and quota",
            "Independent whole-link/tool/camera capsule bounds, workspace, keepouts and provenance",
            "Independent joint/endpoint rate bounds and provenance, speed at or below 10 percent",
            "Original indexed stationary anchor in current owner session; approved raw joint targets",
            "Args-only trial JSON; freeze disjoint split before tracking and use the exact wrist SI prior",
            "Non-contact static textured scene, clear route, emergency-stop operator and abort rules",
            "SI camera prior and uncertainty; source-disjoint training/withheld split",
        ],
        "sequence": [
            "Read-only target/dependency/owner/fingerprint preflight",
            "Present exact package, config, reload/restart and rollback commands with impacts",
            "Receive explicit deployment approval; deploy only through existing owner",
            "Complete operator setup; approve stationary acquisition with motion policy absent",
            "Fill indexed original anchor and verified motion policy; preview and approve skill.reload without restart",
            "Plan from fresh feedback; explicitly approve each bounded pose run; no automatic return",
            "Freeze disjoint split; approve leased READ shadow trial before raw expiry",
            "Report withheld verdict; do not activate corrected geometry or camera candidates",
        ],
        "deployment_commands": None,
        "excluded_scope": [
            "activation",
            "online alignment",
            "raw-response adaptation",
            "gripper motion",
            "firmware zeros",
            "corrected IK",
            "trust/key changes",
        ],
    }
    preview["artifact_sha256"] = {
        path.name: fingerprint(path.read_bytes()) for path in output.iterdir() if path.is_file()
    }
    (output / "deployment-preview.json").write_text(json.dumps(preview, indent=2) + "\n")
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new local preview directory")
    parser.add_argument(
        "--verification",
        type=Path,
        action="append",
        required=True,
        help="passing JUnit report; repeat for independently executed tests",
    )
    parser.add_argument("--preparation", type=Path, help="fingerprinted offline preparation.json to include")
    args = parser.parse_args()
    preview = build_preview(args.output.resolve(), args.verification, args.preparation)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "wheel": preview["wheel"]["file"],
                "sha256": preview["wheel"]["sha256"],
                "state": preview["state"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
