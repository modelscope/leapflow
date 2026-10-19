# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline workcell admission (`leap hw validate --workcells`).

The shipped examples must admit against each other, and a broken workcell must
be refused with its reason, without hardware being enabled or opened.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import make_settings

from leapflow.hardware.workcell_validation import validate_workcells

EXAMPLES = Path(__file__).resolve().parents[1] / "src" / "leapflow" / "hardware" / "examples"
WORKCELLS = EXAMPLES / "workcells"


def _copy(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    devices = tmp_path / "devices"
    workcells = tmp_path / "workcells"
    shutil.copytree(EXAMPLES, devices, ignore=shutil.ignore_patterns("workcells"))
    workcells.mkdir()
    raw = yaml.safe_load((WORKCELLS / "piper_cell.yaml").read_text(encoding="utf-8"))
    return devices, workcells, raw


def _write(workcells: Path, raw: dict[str, Any]) -> None:
    (workcells / "piper_cell.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_shipped_examples_admit_offline() -> None:
    report = validate_workcells(WORKCELLS, EXAMPLES)

    assert report.ok, report.to_dict()
    assert report.workcells.admitted == ("piper_cell",)
    for device_id in ("robot_piper_arm", "sensor_realsense_435if", "perception_locator"):
        assert device_id in report.devices.admitted
    assert "policy_smolvla" in report.devices.admitted


def _rejection(report: Any) -> str:
    assert not report.ok
    assert not report.workcells.admitted
    (note,) = report.workcells.rejected
    return note.detail


def test_unknown_field_is_refused(tmp_path: Path) -> None:
    devices, workcells, raw = _copy(tmp_path)
    raw["speed_limit"] = 50
    _write(workcells, raw)

    assert "speed_limit" in _rejection(validate_workcells(workcells, devices))


def test_role_naming_an_undeclared_device_is_refused(tmp_path: Path) -> None:
    devices, workcells, raw = _copy(tmp_path)
    raw["roles"]["camera.wrist"] = "sensor_wrist_cam"
    _write(workcells, raw)

    detail = _rejection(validate_workcells(workcells, devices))
    assert "camera.wrist" in detail and "sensor_wrist_cam" in detail


def test_model_device_missing_is_refused(tmp_path: Path) -> None:
    devices, workcells, raw = _copy(tmp_path)
    (devices / "policy_smolvla.yaml").unlink()
    _write(workcells, raw)

    assert "policy_smolvla" in _rejection(validate_workcells(workcells, devices))


def test_policy_binding_an_undeclared_channel_is_refused(tmp_path: Path) -> None:
    devices, workcells, raw = _copy(tmp_path)
    raw["policies"]["pick_cube"]["actions"]["arm"][5] = "joint.j7.position"
    _write(workcells, raw)

    assert "joint.j7.position" in _rejection(validate_workcells(workcells, devices))


def test_empty_directory_is_not_ok(tmp_path: Path) -> None:
    report = validate_workcells(tmp_path / "missing", EXAMPLES)

    assert not report.ok
    assert report.workcells.admitted == () and report.workcells.rejected == ()


@pytest.mark.parametrize("broken", [False, True])
def test_cli_runs_without_hardware_enabled(monkeypatch, tmp_path, capsys, broken) -> None:
    import leapflow.cli.commands.hardware as hardware_module
    from leapflow.cli.commands.hardware import cmd_hardware

    devices, workcells, raw = _copy(tmp_path)
    if broken:
        raw["roles"]["arm"] = "robot_missing"
    _write(workcells, raw)
    monkeypatch.setattr(hardware_module, "load_config", lambda: make_settings(str(tmp_path)))

    def _no_registry(settings: Any) -> Any:
        raise AssertionError("offline validation must not build the node registry")

    monkeypatch.setattr(hardware_module, "_build_local_registry", _no_registry)
    args = argparse.Namespace(
        hw_action="validate",
        json=True,
        file_or_device="",
        workcells=str(workcells),
        devices=str(devices),
    )

    assert cmd_hardware(args) == (1 if broken else 0)
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is (not broken)
    assert payload["workcells"]["admitted"] == ([] if broken else ["piper_cell"])
