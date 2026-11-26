# Copyright (c) Alibaba, Inc. and its affiliates.
"""Passive live/episode clips and shared bounded media validation."""
from __future__ import annotations

import base64
import json
import sys
from types import SimpleNamespace

import pytest

from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skill_results import share_frame
from leapflow.hardware.skills.builtin.clip import Clip
from leapflow.hardware.skills.clips import WorkcellClips, _encode
from leapflow.hardware.skills.protocol import EFFECT_READ, SIDE_EFFECT_NONE, SkillError, SkillSpec
from leapflow.hardware.skills.workcell import WorkcellSpec
from leapflow.hardware.snapshot_evidence import EvidenceError, SnapshotEvidenceSettings, _decode_video


def spec():
    return WorkcellSpec("cell", {"camera": "cam"}, share_frames=True)


class Clock:
    now = 10.0
    def __call__(self):
        return self.now
    async def sleep(self, delta):
        self.now += delta


def frame(stamp, sequence):
    return RgbdFrame("cam", 64, 48, bytes([10, 20, 30]) * (64 * 48), bytes(64 * 48 * 2),
                     0.001, CameraIntrinsics(64, 48, 50, 50, 32, 24),
                     1000 + stamp, stamp, sequence=sequence)


async def test_live_clip_uses_fresh_frames_and_remains_passive():
    pytest.importorskip("cv2")
    clock, bounds = Clock(), []
    async def capture(camera, bound):
        assert camera == "camera"
        bounds.append(bound)
        clock.now = max(clock.now, bound) + 0.01
        return frame(clock.now, len(bounds))
    port = WorkcellClips(spec(), capture, clock=clock, sleep=clock.sleep)
    skill = Clip()
    ctx = SimpleNamespace(clips=port)
    args = {"camera": "camera", "source": "live", "run_id": "", "seconds": 2, "max_side": 320}
    outcome = await skill.execute(ctx, await skill.plan(ctx, args))
    assert outcome.side_effect_state == SIDE_EFFECT_NONE and skill.spec.effect == EFFECT_READ
    result = outcome.result
    assert result["sampled_frames"] == 4 and len(set(bounds)) == 4
    assert not result["historical"] and result["validity_timebase"] == "node_remaining_duration"
    data, mime, _, duration = _decode_video(result["video"], SnapshotEvidenceSettings())
    assert data and mime == "video/mp4" and duration == pytest.approx(result["duration_s"], abs=0.01)
    shared = {"ok": True, "result": dict(result)}
    assert not share_frame(shared, consent=True)
    assert "video" not in shared["result"] and "_attachments" not in shared


async def test_repeated_frame_refused():
    pytest.importorskip("cv2")
    clock = Clock()
    async def capture(camera, bound):
        return frame(bound + 0.01, 1)
    with pytest.raises(SkillError, match="repeated"):
        await WorkcellClips(spec(), capture, clock=clock, sleep=clock.sleep).capture(
            "camera", source="live", run_id="", seconds=2, max_side=320)


def episode(root, *, workcell="cell", escape=False):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    (root / "meta").mkdir()
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_id": "run", "workcell": workcell}) + "\n")
    directory = root / "videos/episode_run/cam"
    directory.mkdir(parents=True)
    refs = []
    for index in range(20):
        path = directory / f"{index:06d}.jpg"
        cv2.imwrite(str(path), np.full((48, 64, 3), index * 10, dtype=np.uint8))
        refs.append({"timestamp": 100 + index / 4, "frame_ref": str(path.relative_to(root))})
    if escape:
        refs[-1]["frame_ref"] = "../outside.jpg"
    (directory.parent / "cam_refs.json").write_text(json.dumps(refs))


async def test_episode_preserves_duration_and_never_reads_live_camera(tmp_path):
    episode(tmp_path)
    async def capture(*args):
        pytest.fail("Historical clip must not access the camera")
    result = await WorkcellClips(spec(), capture, directory=tmp_path).capture(
        "camera", source="episode", run_id="run", seconds=3, max_side=320)
    assert result["historical"] and result["run_id"] == "run"
    assert result["duration_s"] == 3 and result["observed_at"] == 104.75
    assert result["capture_timestamps"][0] == 101.75
    assert len(base64.b64decode(result["video"]["data"])) <= 700 * 1024


@pytest.mark.parametrize("fault", ["owner", "escape", "unknown"])
async def test_episode_owner_and_path_boundaries(tmp_path, fault):
    episode(tmp_path, workcell="other" if fault == "owner" else "cell", escape=fault == "escape")
    with pytest.raises(SkillError):
        await WorkcellClips(spec(), None, directory=tmp_path).capture(
            "camera", source="episode", run_id="unknown" if fault == "unknown" else "run", seconds=3, max_side=320)


@pytest.mark.parametrize("args", [
    {"source": "episode", "run_id": "../run"}, {"seconds": 11}, {"max_side": 449},
    {"seconds": True}, {"source": "live", "run_id": "run"}, {"camera": "arm"},
])
async def test_invalid_arguments_refuse_before_capture(args):
    values = {"source": "live", "run_id": "", "seconds": 2, "max_side": 320, "camera": "camera", **args}
    with pytest.raises(SkillError):
        await WorkcellClips(spec(), None).capture(**values)


def test_optional_encoder_missing_and_video_validation(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(SkillError, match="OpenCV"):
        _encode([], 2, 320)
    video = {"mime": "video/mp4", "data": base64.b64encode(b"\x00\x00\x00\x18ftyp" + bytes(40)).decode()}
    with pytest.raises(EvidenceError, match="OpenCV"):
        _decode_video(video, SnapshotEvidenceSettings())


def test_evidence_spec_rejects_actuating_and_leased_exports():
    for values in ({}, {"effect": EFFECT_READ, "leased": True}, {"effect": EFFECT_READ, "evidence_media": "audio"}):
        with pytest.raises(ValueError):
            SkillSpec("test", "1", "test", {}, evidence_media=values.pop("evidence_media", "video"), **values)


async def test_live_duration_uses_monotonic_time_across_wall_clock_jump():
    pytest.importorskip("cv2")
    clock, calls = Clock(), []
    async def capture(camera, bound):
        clock.now = max(clock.now, bound) + 0.01
        calls.append(bound)
        value = frame(clock.now, len(calls))
        from dataclasses import replace
        return replace(value, observed_at=1000 + clock.now - 100 * len(calls))
    result = await WorkcellClips(spec(), capture, clock=clock, sleep=clock.sleep).capture(
        "camera", source="live", run_id="", seconds=2, max_side=320)
    assert result["duration_s"] == pytest.approx(1.5)
    assert result["capture_timestamps"][-1] < result["capture_timestamps"][0]


async def test_slow_resize_keeps_node_event_loop_responsive(monkeypatch):
    import asyncio
    import time
    import leapflow.hardware.skills.clips as clips
    pytest.importorskip("cv2")
    original = clips._resize
    started, finished, ticks = [], [], []
    def slow_resize(image, side):
        started.append(time.monotonic())
        time.sleep(0.04)
        value = original(image, side)
        finished.append(time.monotonic())
        return value
    monkeypatch.setattr(clips, "_resize", slow_resize)
    clock = Clock()
    async def capture(camera, bound):
        clock.now = max(clock.now, bound) + 0.01
        return frame(clock.now, len(started) + 1)
    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.005)
    pulse = asyncio.create_task(heartbeat())
    try:
        await WorkcellClips(spec(), capture, clock=clock, sleep=clock.sleep)._live("camera", 2, 320)
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)
    assert len(started) == len(finished) == 4
    assert all(any(a < tick < b for tick in ticks) for a, b in zip(started, finished))


def test_opencv_encoder_error_is_typed(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    def broken_writer(*args):
        raise cv2.error("encoder failed")
    monkeypatch.setattr(cv2, "VideoWriter", broken_writer)
    with pytest.raises(SkillError, match="encoder"):
        _encode([np.zeros((48, 64, 3), dtype=np.uint8)] * 4, 2, 64)


def test_encoder_reencodes_once_then_refuses(monkeypatch):
    import leapflow.hardware.skills.clips as clips
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    attempts = []
    class Writer:
        def __init__(self, path, *args):
            self.path = path
            attempts.append(args[-1])
        def isOpened(self):
            return True
        def write(self, image):
            pass
        def release(self):
            from pathlib import Path
            Path(self.path).write_bytes(bytes(100 if len(attempts) == 2 else 701 * 1024))
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    images = [np.zeros((240, 320, 3), dtype=np.uint8)] * 4
    assert len(clips._encode(images, 2, 320)) == 100
    assert attempts == [(320, 240), (160, 120)]
    attempts.clear()
    monkeypatch.setattr(clips, "MAX_CLIP_BYTES", 50)
    with pytest.raises(SkillError, match="re-encode"):
        clips._encode(images, 2, 320)
    assert len(attempts) == 2
