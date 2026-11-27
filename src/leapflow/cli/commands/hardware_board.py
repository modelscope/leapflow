# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline printable ChArUco export; never constructs a hardware registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT,
    FAILURE_INVALID_ARGS,
    FAILURE_SKILL_UNAVAILABLE,
    SkillOutcome,
)
from leapflow.hardware.skills.vision import CharucoTarget, charuco_svg
from leapflow.layout import HardwareLayout


def _target(directory: Path, selection: str) -> CharucoTarget:
    path = Path(selection).expanduser() if selection else None
    explicit = path is not None and (
        path.is_file() or path.suffix in {".yaml", ".yml"} or "/" in selection
    )
    paths = [path] if explicit else sorted(directory.glob("*.yaml"))
    matches: list[Mapping[str, Any]] = []
    for source in paths:
        assert source is not None
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"{source}: workcell must be a mapping")
        if explicit or not selection or data.get("workcell") == selection:
            matches.append(data)
    if len(matches) != 1:
        raise ValueError(
            f"found {len(matches)} matching workcells; select one with --workcell ID or a YAML path"
        )
    target = matches[0].get("calibration_target")
    if not isinstance(target, Mapping):
        raise ValueError("the workcell must declare calibration_target with board dimensions")
    return CharucoTarget.from_mapping(target)


def run_board(args: argparse.Namespace, settings: Any, json_mode: bool) -> int:
    """Export one board, refusing ambiguous declarations and existing output files."""
    try:
        layout = HardwareLayout(settings.profile_dir / "hardware")
        target = _target(layout.workcells_dir, str(args.workcell))
        output = Path(args.out).expanduser()
        svg = charuco_svg(target)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(svg)
        outcome = SkillOutcome.succeeded(
            {"path": str(output.resolve()), "target": target.to_dict()},
            message=(
                f"Saved {output}. Print at 100% / actual size, not fit-to-page. "
                f"Measure the 100 mm ruler and a {target.square_mm:g} mm square; "
                "mount the sheet flat on a rigid backing."
            ),
        )
    except ImportError as exc:
        outcome = SkillOutcome.failed(
            FAILURE_SKILL_UNAVAILABLE,
            f"Board export needs OpenCV with cv2.aruco: {exc}. Install the existing vision extra.",
        )
    except (ValueError, yaml.YAMLError) as exc:
        outcome = SkillOutcome.failed(FAILURE_INVALID_ARGS, str(exc))
    except OSError as exc:
        outcome = SkillOutcome.failed(
            FAILURE_DEVICE_FAULT,
            f"Cannot read the workcell or write the board: {exc}. Check paths and use a new --out file.",
        )
    if json_mode:
        print(json.dumps({"ok": outcome.ok, **outcome.to_dict()}, ensure_ascii=False))
    else:
        print(outcome.message)
    return 0 if outcome.ok else 1
