# Copyright (c) Alibaba, Inc. and its affiliates.
"""Print geometry, scale observability, and the offline CLI's failure boundaries."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
import yaml

from leapflow.cli.commands import hardware_board
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.protocol import TargetObservation
from leapflow.hardware.skills.vision import CharucoTarget, _charuco_board, charuco_svg

BOARD = {"squares_x": 5, "squares_y": 4, "square_mm": 30, "marker_mm": 22}


@pytest.mark.parametrize("legacy", [False, True])
def test_svg_has_exact_board_dimensions_and_same_detector_geometry(legacy: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    target = CharucoTarget.from_mapping({**BOARD, "legacy": legacy})
    svg = charuco_svg(target)
    root = ET.fromstring(svg)
    image = root.find("{http://www.w3.org/2000/svg}image")
    assert image.attrib["width"] == "150" and image.attrib["height"] == "120"
    assert root.attrib["width"] == "170mm"  # 10 mm quiet margins
    assert "100 mm ruler" in svg and "h100" in svg
    encoded = base64.b64decode(image.attrib["href"].split(",", 1)[1])
    pixels = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(_charuco_board(target)).detectBoard(pixels)
    assert corners is not None and len(ids) == 12


def test_depth_scale_is_coarse_observable_not_a_certified_error_bound() -> None:
    observation = TargetObservation(
        Transform.from_pose(Pose(0, 0, 0, 0, 0, 0)),
        (10, 20, 500),
        12,
        0.1,
        {},
        1,
        depth_point=(13.22, 26.44, 661),
    )
    assert observation.depth_scale == pytest.approx(1.322)
    assert replace(observation, depth_point=None).depth_scale is None
    assert replace(observation, depth_point=(float("nan"), 0, 0)).depth_scale is None
    assert replace(observation, center=(0, 0, 0)).depth_scale is None


def _args(tmp_path: Path, selection: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        workcell=selection, out=str(tmp_path / "board.svg"), json=True, hw_action="board"
    )


def _declare(tmp_path: Path, name: str = "bench") -> None:
    directory = tmp_path / "hardware" / "workcells"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yaml").write_text(
        yaml.safe_dump({"workcell": name, "calibration_target": BOARD})
    )


@pytest.mark.parametrize("selection", ["", "bench", "path"])
def test_cli_exports_offline_by_id_path_or_unique_default(
    tmp_path: Path, capsys, selection: str
) -> None:
    pytest.importorskip("cv2")
    _declare(tmp_path)
    if selection == "path":
        selection = str(tmp_path / "hardware/workcells/bench.yaml")
    assert (
        hardware_board.run_board(
            _args(tmp_path, selection), SimpleNamespace(profile_dir=tmp_path), True
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and "100%" in result["message"]
    assert Path(result["result"]["path"]).exists()
    assert (
        hardware_board.run_board(
            _args(tmp_path, selection), SimpleNamespace(profile_dir=tmp_path), True
        )
        == 1
    )
    assert "new --out" in json.loads(capsys.readouterr().out)["message"]


def test_cli_dispatch_never_constructs_registry(tmp_path: Path, monkeypatch, capsys) -> None:
    from leapflow.cli.commands import hardware

    _declare(tmp_path)
    monkeypatch.setattr(hardware, "load_config", lambda: SimpleNamespace(profile_dir=tmp_path))
    monkeypatch.setattr(
        hardware, "_build_local_registry", lambda *a: pytest.fail("must stay offline")
    )
    monkeypatch.setattr(hardware_board, "charuco_svg", lambda target: "<svg/>")
    assert hardware.cmd_hardware(_args(tmp_path)) == 0
    assert json.loads(capsys.readouterr().out)["ok"]


@pytest.mark.parametrize("case", ["missing", "ambiguous", "invalid", "dependency"])
def test_cli_refusals_are_actionable(tmp_path: Path, capsys, monkeypatch, case: str) -> None:
    if case != "missing":
        _declare(tmp_path)
    if case == "ambiguous":
        _declare(tmp_path, "other")
    if case == "invalid":
        (tmp_path / "hardware/workcells/bench.yaml").write_text("[]")
    if case == "dependency":

        def absent(target):
            raise ImportError("no cv2")

        monkeypatch.setattr(hardware_board, "charuco_svg", absent)
    assert (
        hardware_board.run_board(_args(tmp_path), SimpleNamespace(profile_dir=tmp_path), True) == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and result["side_effect_state"] == "none" and result["message"]
    assert not (tmp_path / "board.svg").exists()
