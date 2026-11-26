# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-side bounded temporal evidence over existing camera and episode seams.

No camera is opened here, no actuator is read or commanded, and encoding never
persists an episode. Temporary files are local to the encoder and always removed.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS, FAILURE_OBSERVATION_INSUFFICIENT,
    FAILURE_SKILL_UNAVAILABLE, SkillError,
)
from leapflow.hardware.skills.workcell import WorkcellSpec

MAX_CLIP_BYTES = 700 * 1024
MAX_FRAMES = 32


def _libraries() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "Clip encoding needs optional OpenCV and numpy") from exc
    return cv2, np


def _resize(image: Any, side: int) -> Any:
    cv2, _ = _libraries()
    height, width = image.shape[:2]
    scale = min(1.0, side / max(height, width))
    size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _encode(images: list[Any], duration: float, side: int) -> bytes:
    cv2, _ = _libraries()
    try:
        return _encode_frames(images, duration, side)
    except (cv2.error, OSError) as exc:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Clip encoder could not produce valid evidence") from exc


def _encode_frames(images: list[Any], duration: float, side: int) -> bytes:
    cv2, _ = _libraries()
    if not 4 <= len(images) <= MAX_FRAMES or not 0 < duration <= 10:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Clip needs 4–32 frames spanning at most 10 seconds")
    with tempfile.TemporaryDirectory(prefix="leap-clip-") as directory:
        for attempt, bound in enumerate((side, max(64, side // 2))):
            resized = [_resize(image, bound) for image in images]
            height, width = resized[0].shape[:2]
            if any(image.shape != resized[0].shape for image in resized):
                raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Camera dimensions changed within the clip")
            path = Path(directory) / f"clip-{attempt}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), len(images) / duration, (width, height))
            try:
                if not writer.isOpened():
                    raise SkillError(FAILURE_SKILL_UNAVAILABLE, "OpenCV mp4v encoder is unavailable")
                for image in resized:
                    writer.write(image)
            finally:
                writer.release()
            size = path.stat().st_size
            if 0 < size <= MAX_CLIP_BYTES:
                return path.read_bytes()
    raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Clip exceeds 700 KiB after one smaller re-encode")


def _inside(root: Path, relative: str, limit: int) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > limit:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Episode reference is missing, oversized, or outside its owning directory")
    return path


def _episode(directory: Path | None, workcell: str, device: str, run_id: str,
             seconds: float, side: int) -> tuple[list[Any], list[float], float]:
    if directory is None:
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "No episode directory is configured")
    root = directory.resolve()
    manifest = _inside(root, "meta/episodes.jsonl", 64 * 1024 * 1024)
    owned = False
    with manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("episode_id") == run_id:
                owned = record.get("workcell") == workcell
                break
    if not owned:
        raise SkillError(FAILURE_INVALID_ARGS, "run_id is unknown or belongs to another workcell")
    camera = re.sub(r"[^a-zA-Z0-9_.-]", "_", device)
    relative = f"videos/episode_{run_id}"
    episode_root = (root / relative).resolve()
    if not episode_root.is_relative_to(root):
        raise SkillError(FAILURE_INVALID_ARGS, "Episode directory escapes its owner")
    refs = json.loads(_inside(root, f"{relative}/{camera}_refs.json", 8 * 1024 * 1024).read_text())
    if not isinstance(refs, list) or len(refs) < 4:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Episode has fewer than four camera frames")
    stamps = [float(ref["timestamp"]) for ref in refs]
    if not all(math.isfinite(t) for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Episode capture timestamps are invalid or repeated")
    # Use a bounded trailing window; historical wall-clock timestamps remain explicit.
    start = next(i for i, stamp in enumerate(stamps) if stamp >= stamps[-1] - seconds)
    window = refs[start:]
    if len(window) < 4:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Requested episode window has fewer than four frames")
    duration = stamps[-1] - stamps[start]
    if duration <= 0:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Episode has no temporal span")
    count = min(MAX_FRAMES, len(window), max(4, math.ceil(duration * 2)))
    indices = [round(i * (len(window) - 1) / (count - 1)) for i in range(count)]
    cv2, _ = _libraries()
    images, selected = [], []
    camera_root = (episode_root / camera).resolve()
    for index in indices:
        ref = window[index]
        path = _inside(root, str(ref["frame_ref"]), 4 * 1024 * 1024)
        if not path.is_relative_to(camera_root):
            raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "JPEG reference belongs to another episode or camera")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] * image.shape[1] > 4096 * 4096:
            raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Episode JPEG is invalid or oversized")
        images.append(_resize(image, side))
        selected.append(float(ref["timestamp"]))
    return images, selected, duration


class WorkcellClips:
    """A read context's clip port; capture callback reuses the owner's FrameHub."""

    def __init__(self, spec: WorkcellSpec, capture: Callable[[str, float], Awaitable[RgbdFrame]], *,
                 directory: Path | None = None, deadline: float = float("inf"),
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        self._spec, self._capture, self._directory = spec, capture, directory
        self._deadline, self._clock, self._sleep = deadline, clock, sleep

    async def _live(self, camera: str, seconds: float, side: int) -> tuple[list[Any], list[float], float]:
        _, np = await asyncio.to_thread(_libraries)
        images, stamps, monotonic_stamps = [], [], []
        start, previous = self._clock(), float("-inf")
        sequence = -1
        count = math.ceil(seconds * 2)
        for index in range(count):
            tick = start + index / 2
            await self._sleep(max(0, tick - self._clock()))
            frame = await asyncio.wait_for(self._capture(camera, max(tick, previous)),
                                          min(2.0, max(0, self._deadline - self._clock())))
            if frame.monotonic_at <= max(tick, previous) or frame.sequence <= sequence:
                raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Camera returned stale or repeated clip frames")
            previous, sequence = frame.monotonic_at, frame.sequence
            image = np.frombuffer(frame.color, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            images.append(await asyncio.to_thread(_resize, image, side))
            stamps.append(frame.observed_at)
            monotonic_stamps.append(frame.monotonic_at)
        duration = monotonic_stamps[-1] - monotonic_stamps[0]
        if not 0 < duration <= 10:
            raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, "Camera sampling exceeded the 10-second envelope")
        return images, stamps, duration

    async def capture(self, camera: str, *, source: str, run_id: str,
                      seconds: float, max_side: int) -> dict[str, Any]:
        if (source not in {"live", "episode"} or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or not 2 <= seconds <= 10
                or type(max_side) is not int or not 64 <= max_side <= 448
                or not re.fullmatch(r"camera(?:\.[a-z0-9_]+)*", camera)
                or (source == "episode") != bool(run_id)
                or (run_id and not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", run_id))):
            raise SkillError(FAILURE_INVALID_ARGS, "Invalid clip source, run_id, duration, camera, or size")
        device = self._spec.role(camera)
        if not device or not self._spec.share_frames:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "Clip requires the exact camera binding and workcell frame sharing")
        cv2, _ = await asyncio.to_thread(_libraries)
        try:
            if source == "live":
                images, stamps, duration = await self._live(camera, float(seconds), max_side)
            else:
                images, stamps, duration = await asyncio.wait_for(asyncio.to_thread(
                    _episode, self._directory, self._spec.workcell_id, device, run_id, float(seconds), max_side),
                    max(0, self._deadline - self._clock()))
            content = await asyncio.wait_for(asyncio.to_thread(_encode, images, duration, max_side),
                                            max(0, self._deadline - self._clock()))
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT, f"Clip evidence unavailable: {exc}") from exc
        identity = uuid.uuid4().hex
        return {"frame": f"clip-{identity}", "capture_id": identity, "camera": camera,
                "observed_at": stamps[-1], "capture_timestamps": stamps,
                "duration_s": duration, "sampled_frames": len(images), "source": source,
                "run_id": run_id, "historical": source == "episode", "liftable": False,
                "valid_for_s": 60.0, "validity_timebase": "node_remaining_duration",
                "video": {"mime": "video/mp4", "data": base64.b64encode(content).decode("ascii")}}
