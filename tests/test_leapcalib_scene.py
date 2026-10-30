# Copyright (c) Alibaba, Inc. and its affiliates.
"""Disjoint scene comparison on rendered images, never independent physical proof."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.tracking import SceneConfig, TrackingConfig
from leapcalib.estimation.tracking import compare_scene_supports, track_scene
from leapcalib.geometry.se3 import Transform
from leapcalib.runtime.protocol import WorkerBudgets, WorkerRequest, validate_payload
from leapflow.hardware.calibration.worker import CalibrationWorker
from tests.test_leapcalib_tracking import pair

pytestmark = pytest.mark.component


def scene_pair(*, task_shift: int = 0, common_shift: int = 0):
    import cv2

    first, second = pair()
    profile = replace(
        first.profile, width=320, height=240, fx_px=300, fy_px=300, ppx_px=160, ppy_px=120
    )
    rng = np.random.default_rng(190)
    # Multi-scale texture supports the actual coarse-to-fine LK search. Pure
    # pixel noise can match forward yet lose backward support at the coarsest level.
    coarse = cv2.resize(rng.integers(0, 256, (30, 40), dtype=np.uint8), (320, 240))
    fine = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
    image = cv2.addWeighted(coarse, 0.7, fine, 0.3, 0)
    changed = image.copy()
    if task_shift:
        changed[:, :144] = cv2.warpAffine(
            image[:, :144],
            np.float32([[1, 0, task_shift], [0, 1, 0]]),
            (144, 240),
            borderMode=cv2.BORDER_REFLECT_101,
        )
    if common_shift:
        changed = cv2.warpAffine(
            changed,
            np.float32([[1, 0, common_shift], [0, 1, 0]]),
            (320, 240),
            borderMode=cv2.BORDER_REFLECT_101,
        )
    first = replace(
        first,
        profile=profile,
        width=320,
        height=240,
        color_bgr8=np.repeat(image[:, :, None], 3, axis=2).tobytes(),
        depth_m_f32le=np.full((240, 320), 0.5, dtype="<f4").tobytes(),
    )
    second = replace(
        second,
        profile=profile,
        width=320,
        height=240,
        color_bgr8=np.repeat(changed[:, :, None], 3, axis=2).tobytes(),
        depth_m_f32le=first.depth_m_f32le,
    )

    def support(name, columns):
        tracks = []
        for i, x in enumerate(columns):
            for j, y in enumerate((48, 96, 144, 192)):
                mask = np.zeros(image.shape, dtype=np.uint8)
                mask[y - 8 : y + 9, x - 8 : x + 9] = 255
                corners = cv2.goodFeaturesToTrack(
                    image,
                    maxCorners=1,
                    qualityLevel=0.02,
                    minDistance=8,
                    mask=mask,
                    blockSize=7,
                )
                assert corners is not None, "rendered fixture lacks actual corner support"
                tracks.append(
                    {
                        "feature_id": f"{name}:{i}:{j}",
                        "frame_id": first.frame_id,
                        "pixel": corners.reshape(2).tolist(),
                    }
                )
        return tracks

    # Discover only in the source image, then freeze distinct identified supports.
    task = support("task", (24, 56, 88, 120))
    background = support("background", (200, 232, 264, 296))
    return first, second, task, background


def numerical_policy():
    # Precision floors are explicitly synthetic values for algorithm testing.
    return TrackingConfig(max_tracks=64, fb_gate_px=0.005, depth_floor_m=0.000001)


@pytest.mark.parametrize(
    "task_shift,common_shift,verdict",
    [
        (0, 0, "consistent"),
        (0, 2, "consistent"),
        (4, 0, "relative_motion"),
        (4, 2, "relative_motion"),
    ],
)
def test_scene_comparison_cancels_common_camera_motion_not_relative_target_motion(
    task_shift, common_shift, verdict
):
    first, second, task, background = scene_pair(task_shift=task_shift, common_shift=common_shift)
    result = track_scene(
        first, second, task_tracks=task, background_tracks=background, config=numerical_policy()
    )
    report = result["comparison"]
    assert report["verdict"] == verdict, report
    assert report["coordinates"] == "relative_at_task_centroid;full_SO3_rad"
    assert report["provenance"] == "synthetic"
    assert all(v > 0 for v in report["uncertainty_m_rad"])
    assert (
        report["physical_admission"] == report["independent_static_background"] == "not_established"
    )
    assert report["witness_valid"] is report["executed"] is False
    assert report["task_tracking_digest"] == digest_of(result["task_tracking"])
    assert report["background_tracking_digest"] == digest_of(result["background_tracking"])
    assert len(result["task_tracking"]["tracks"]) == len(task)
    assert len(result["background_tracking"]["tracks"]) == len(background)
    # Known renderer displacement independently bounds these synthetic pixels;
    # the tiny test floor is never applied to actual camera metrology.
    for group, shift in (
        ("task_tracking", task_shift + common_shift),
        ("background_tracking", common_shift),
    ):
        for row in result[group]["tracks"]:
            delta = np.asarray(row["second_pixel"]) - row["first_pixel"]
            assert np.linalg.norm(delta - [shift, 0]) < numerical_policy().fb_gate_px


def test_scene_comparison_checks_full_orientation_not_only_favorable_yaw():
    first, second, task, background = scene_pair()
    result = track_scene(
        first, second, task_tracks=task, background_tracks=background, config=numerical_policy()
    )
    angle = 0.04
    c, s = np.cos(angle), np.sin(angle)
    rotation = Transform(((1, 0, 0), (0, c, -s), (0, s, c)), (0, 0, 0))
    group = result["background_tracking"]
    group["registration"]["motion"] = rotation.to_dict()
    for row in group["tracks"]:
        row["second_m"] = rotation.apply(row["first_m"])
        x, y, z = row["second_m"]
        row["second_pixel"] = [
            x * first.profile.fx_px / z + first.profile.ppx_px,
            y * first.profile.fy_px / z + first.profile.ppy_px,
        ]
    report = compare_scene_supports(
        first,
        result["task_tracking"],
        group,
        task_tracks=task,
        background_tracks=background,
        config=numerical_policy(),
        scene_config=SceneConfig(),
    )
    assert group["delta_m_rad"][3] == pytest.approx(0)
    assert report["error_m_rad"][1] == pytest.approx(angle)
    assert report["verdict"] == "relative_motion"
    assert report["witness_valid"] is False


def test_default_depth_and_pixel_floors_do_not_turn_zero_residual_into_a_witness():
    first, second, task, background = scene_pair()
    result = track_scene(first, second, task_tracks=task, background_tracks=background)
    assert result["comparison"]["verdict"] == "ambiguous"
    assert result["comparison"]["uncertainty_m_rad"][0] > 0.003


@pytest.mark.parametrize(
    "defect",
    [
        "alias",
        "overlap",
        "lost-task",
        "lost-background",
        "wrong-frame",
        "distortion",
        "budget",
        "occluded",
    ],
)
def test_scene_support_cannot_be_shared_reacquired_or_silently_dropped(defect):
    first, second, task, background = scene_pair()
    config = numerical_policy()
    if defect == "alias":
        background[0] = {**background[0], "feature_id": task[0]["feature_id"]}
    elif defect == "overlap":
        background[0] = {**background[0], "pixel": task[0]["pixel"]}
    elif defect == "lost-task":
        task = task[:11]
    elif defect == "lost-background":
        background = background[:11]
    elif defect == "wrong-frame":
        task[0] = {**task[0], "frame_id": "foreign"}
    elif defect == "distortion":
        first = replace(first, profile=replace(first.profile, distortion_model="brown_conrady"))
    elif defect == "budget":
        config = replace(config, max_tracks=24)
    else:
        second = replace(second, color_bgr8=bytes(len(second.color_bgr8)))
    with pytest.raises((ValueError, CalibrationFailure)):
        track_scene(first, second, task_tracks=task, background_tracks=background, config=config)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("max_translation_m", 0.004),
        ("max_rotation_rad", 0.04),
        ("confidence_multiplier", 0.5),
        ("min_separation_px", 20),
        ("max_translation_m", True),
    ],
)
def test_scene_policy_cannot_widen_error_limits_or_erase_shared_floors(setting, value):
    with pytest.raises(ValueError):
        SceneConfig(**{setting: value})


def payload():
    first, second, task, background = scene_pair(common_shift=2)
    return {
        "first": first.to_dict(),
        "second": second.to_dict(),
        "task_tracks": task,
        "background_tracks": background,
        "config": vars(numerical_policy()),
        "scene_config": vars(SceneConfig()),
    }


@pytest.mark.parametrize("defect", ["seed", "total-budget", "roi-budget", "config", "field"])
def test_scene_worker_payload_budgets_are_checked_before_image_decoding(defect):
    data = payload()
    if defect == "seed":
        data["task_tracks"] = []
    elif defect == "total-budget":
        data["background_tracks"] *= 100
    elif defect == "roi-budget":
        data["first"]["width"] = 641
    elif defect == "config":
        data["scene_config"]["max_translation_m"] = 0.004
    else:
        data["witness_valid"] = True
    request = WorkerRequest(
        "r", "j", "cell", "run", "model", "track_scene", time.monotonic() + 2, data, digest_of(data)
    )
    with pytest.raises(ValueError):
        validate_payload(request, WorkerBudgets())


async def test_disjoint_scene_tracking_uses_real_bounded_owner_worker(tmp_path: Path):
    worker = CalibrationWorker(
        tmp_path.resolve().parent / ("lc-scene-" + uuid.uuid4().hex[:6]) / "s", enabled=True
    )
    try:
        data = payload()
        result = await worker.submit(
            "track_scene", data, workcell_id="cell", run_id="synthetic-scene", model_digest="model"
        )
        scene = result["scene"]
        assert scene["schema"] == "leapcalib.scene_consistency.v1"
        assert scene["comparison"]["verdict"] == "consistent"
        assert scene["comparison"]["witness_valid"] is False
        assert worker.last_metrics["peak_rss_bytes"] <= worker.budgets.rss_bytes
        # Shared cancellation/reaping machinery is used; no new camera or permit API.
        assert "permit" not in json.dumps(result)
    finally:
        await worker.close()
    assert worker._process is None


def test_scene_contract_and_comparison_modules_remain_import_light():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import leapcalib.contracts.tracking; import leapcalib.estimation.tracking; assert 'numpy' not in sys.modules; assert 'cv2' not in sys.modules; assert 'leapflow' not in sys.modules",
        ],
        check=True,
    )
