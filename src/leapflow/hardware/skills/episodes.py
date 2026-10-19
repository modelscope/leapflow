# Copyright (c) Alibaba, Inc. and its affiliates.
"""Episode recording: every executed skill run leaves a replayable, trainable record.

``SkillRuntime`` gives each actuating run one ``EpisodeRecorder``. While the run
executes, a sampler reads the arm and grabs one JPEG per workcell camera at a
fixed rate, and every governed batch that reached a transport is tapped from
``GovernedMotion``; the policy stream adds its model identity and one event per
adopted chunk. When the run ends -- after any halt -- the episode is written
next to the outcome, whatever the verdict: a refused shield action or a missed
grasp is training signal too.

The on-disk layout is ``EpisodeExporter``'s JSON flavour, numpy-free so the node
writes it as-is; a training machine converts it (LeRobot never becomes a
LeapFlow dependency)::

    <episodes_dir>/
      meta/info.json                            # format and row features
      meta/episodes.jsonl                       # one line per episode
      meta/labels.jsonl                         # verified success labels, append-only
      data/episode_<id>.json                    # sampled rows
      data/episode_<id>.commands.json           # every command and stream event
      videos/episode_<id>/<camera>/<seq>.jpg    # frames, paths relative to the root
      videos/episode_<id>/<camera>_refs.json    # frame index per camera

Recording is telemetry: it never fails, delays, or alters a run. A full disk,
a camera that stops answering, or a write error leaves a shorter episode and a
note in the manifest, never a different verdict.

The manifest carries the run's own verdict (``success``) and the sub-goal S2
was pursuing (``intent``, its language annotation). Whether the goal was really
reached is S2's call after it looks again: ``label_episode`` appends that
verdict to ``labels.jsonl``, never rewriting a manifest; ``latest_labels``
reads the newest label per episode back for a converter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence, runtime_checkable

from leapflow.hardware.context import as_numeric
from leapflow.hardware.rgbd import FrameSource, RgbdFrame
from leapflow.hardware.skills.protocol import (
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.workcell import (
    GRIPPER_POSITION,
    POSE_CHANNELS,
    ROLE_ARM,
    ROLE_GRIPPER,
    WorkcellSpec,
    joint_channels,
)
from leapflow.hardware.transport import TransportError

logger = logging.getLogger(__name__)

EPISODE_FORMAT = "leapflow.skill_episode"
EPISODE_FORMAT_VERSION = 1
MAX_EPISODE_FPS = 60.0
MAX_INTENT_CHARS = 500
MAX_LABEL_NOTE_CHARS = 500
# A missing image encoder (ImportError) drops frames like a camera fault would.
_SAMPLE_ERRORS = (TransportError, OSError, RuntimeError, ValueError, ImportError)
_MANIFEST_LOCK = threading.Lock()  # runs on different workcells finish concurrently
_WRITING: set[asyncio.Future[Any]] = set()
_FEATURES = {
    "sequence": "int32",
    "timestamp": "float64 (epoch seconds)",
    "t": "float64 (seconds since the run started)",
    "action_source": "string (the skill id)",
    "joint_positions": "map channel -> float",
    "ee_pose": "map channel -> float (flange, base frame)",
    "gripper_state": "float64 mm, or null",
    "action": "map channel -> float (the latest commanded target)",
    "camera_frames": "map camera -> jpeg path relative to the dataset root",
}

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class EpisodeSettings:
    """Where and how densely runs are recorded; no directory or fps 0 disables it."""

    directory: Path | None = None
    fps: float = 30.0
    min_free_mb: float = 2048.0
    jpeg_quality: int = 90

    @property
    def enabled(self) -> bool:
        return self.directory is not None and self.fps > 0


@runtime_checkable
class EpisodeTap(Protocol):
    """What a run reports to its recorder; every call is cheap and never raises."""

    def command(self, device_id: str, commands: Sequence[tuple[str, Any]]) -> None: ...

    def annotate(self, key: str, value: Any) -> None: ...

    def event(self, kind: str, **data: Any) -> None: ...

    @property
    def recording(self) -> bool:
        """Whether what is tapped is being kept (false when not recorded or skipped)."""
        ...


class NullTap:
    """The tap of a run that is not recorded (plans, READ skills, recording off)."""

    @property
    def recording(self) -> bool:
        return False

    def command(self, device_id: str, commands: Sequence[tuple[str, Any]]) -> None:
        return None

    def annotate(self, key: str, value: Any) -> None:
        return None

    def event(self, kind: str, **data: Any) -> None:
        return None


NULL_TAP = NullTap()


class EpisodeRecorder:
    """One run's episode: sampler, command tap, and the writer. Satisfies ``EpisodeTap``."""

    def __init__(
        self,
        *,
        settings: EpisodeSettings,
        registry: Any,
        spec: WorkcellSpec,
        plan: SkillPlan,
        run_id: str,
        intent: str = "",
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if settings.directory is None:
            raise ValueError("an episode recorder needs a directory")
        self._settings = settings
        self._registry = registry
        self._spec = spec
        self._plan = plan
        self._intent = intent[:MAX_INTENT_CHARS]
        self._clock = clock
        self._sleep = sleep
        self._root = Path(settings.directory)
        self.episode_id = run_id
        self._frames_rel = Path("videos") / f"episode_{run_id}"
        self._cameras: dict[str, str] = {}  # device id -> directory name
        self._rows: list[dict[str, Any]] = []
        self._commands: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._annotations: dict[str, Any] = {}
        self._action: dict[str, float] = {}
        self._dropped: dict[str, int] = {}
        self._skipped = ""
        self._t0 = clock()
        self._started_at = time.time()
        self._sampler: asyncio.Task[None] | None = None

    # ── Tap ──

    @property
    def recording(self) -> bool:
        return not self._skipped

    def command(self, device_id: str, commands: Sequence[tuple[str, Any]]) -> None:
        values = {str(cid): value for cid, value in commands}
        self._commands.append({"t": self._elapsed(), "device": device_id, "commands": values})
        for channel_id, value in values.items():
            number = as_numeric(value)
            if number is not None:
                self._action[channel_id] = float(number)

    def annotate(self, key: str, value: Any) -> None:
        self._annotations[str(key)] = value

    def event(self, kind: str, **data: Any) -> None:
        self._events.append({"t": self._elapsed(), "kind": str(kind), **data})

    # ── Lifecycle ──

    async def start(self) -> None:
        """Prepare the directories, find the cameras, and start sampling. Never raises."""
        try:
            free_mb = await asyncio.to_thread(self._prepare)
            if free_mb < self._settings.min_free_mb:
                self._skip(f"only {free_mb:.0f} MB free (< {self._settings.min_free_mb:g} MB)")
                return
            await self._find_cameras()
        except Exception as exc:  # noqa: BLE001 - recording must never fail a run
            logger.error("episode %s could not start", self.episode_id[:8], exc_info=True)
            self._skip(f"recording could not start: {type(exc).__name__}: {exc}")
            return
        self._t0, self._started_at = self._clock(), time.time()
        self._sampler = asyncio.get_running_loop().create_task(
            self._sample_loop(), name=f"episode-{self.episode_id[:8]}"
        )

    async def finish(self, outcome: SkillOutcome) -> dict[str, Any]:
        """Stop sampling, take the final sample, write the episode; returns its reference.

        Never raises. A stop that cancels the run meanwhile cannot lose the episode:
        writing is shielded and completes on its own, the cancellation is absorbed,
        and the run still reaches its verdict.
        """
        if self._skipped:
            return {"episode_id": self.episode_id, "skipped": self._skipped}
        writing = asyncio.ensure_future(self._finalize(outcome))
        _WRITING.add(writing)  # held until done, even when the run is cancelled
        writing.add_done_callback(_WRITING.discard)
        try:
            return await asyncio.shield(writing)
        except asyncio.CancelledError:
            logger.warning("episode %s: still writing when the run was cancelled", self.episode_id)
            return {"episode_id": self.episode_id, "path": str(self._root), "pending": True}

    async def _finalize(self, outcome: SkillOutcome) -> dict[str, Any]:
        try:
            await self._stop_sampler()
            await self._sample()
            await asyncio.to_thread(self._write, outcome, time.time())
        except Exception as exc:  # noqa: BLE001 - recording must never fail a run
            logger.error("episode %s was not written", self.episode_id[:8], exc_info=True)
            return {"episode_id": self.episode_id, "skipped": f"write failed: {exc}"}
        return {
            "episode_id": self.episode_id,
            "path": str(self._root),
            "steps": len(self._rows),
            "commands": len(self._commands),
            "cameras": sorted(self._cameras.values()),
        }

    # ── Sampling ──

    async def _sample_loop(self) -> None:
        period = 1.0 / min(self._settings.fps, MAX_EPISODE_FPS)
        next_tick = self._clock()
        while True:
            await self._sample()
            next_tick += period
            now = self._clock()
            if next_tick <= now:
                next_tick = now + period  # behind: skip samples, never burst
            await self._sleep(next_tick - now)

    async def _sample(self) -> None:
        sequence = len(self._rows)
        timestamp, elapsed = time.time(), self._elapsed()
        try:
            state = await self._read_arm()
        except _SAMPLE_ERRORS as exc:
            self._drop("arm", exc)
            return
        frames: dict[str, str] = {}
        for device_id, name in self._cameras.items():
            path = await self._capture(device_id, name, sequence)
            if path:
                frames[name] = path
        joints = joint_channels(self._registry.context(self._spec.role(ROLE_ARM)))
        self._rows.append(
            {
                "sequence": sequence,
                "timestamp": timestamp,
                "t": elapsed,
                "action_source": self._plan.skill_id,
                "joint_positions": {cid: state[cid] for cid in joints if cid in state},
                "ee_pose": {cid: state[cid] for cid in POSE_CHANNELS if cid in state},
                "gripper_state": state.get(GRIPPER_POSITION),
                "action": dict(self._action),
                "camera_frames": frames,
            }
        )

    async def _read_arm(self) -> dict[str, float]:
        arm_id, gripper_id = self._spec.role(ROLE_ARM), self._spec.role(ROLE_GRIPPER)
        arm = self._registry.context(arm_id)
        channels = [*POSE_CHANNELS, *joint_channels(arm)]
        if gripper_id == arm_id:
            channels.append(GRIPPER_POSITION)
        state = await self._read(arm_id, channels)
        if gripper_id and gripper_id != arm_id:
            state.update(await self._read(gripper_id, (GRIPPER_POSITION,)))
        return state

    async def _read(self, device_id: str, channel_ids: Sequence[str]) -> dict[str, float]:
        context = self._registry.context(device_id)
        readable = tuple(cid for cid in channel_ids if context.channel(cid) is not None)
        batch = await self._registry.read_batch(device_id, readable)
        values: dict[str, float] = {}
        for reading in batch.readings:
            number = as_numeric(reading.value)
            if number is not None and getattr(reading, "is_trustworthy", True):
                values[reading.channel_id] = float(number)
        return values

    async def _capture(self, device_id: str, name: str, sequence: int) -> str:
        relative = self._frames_rel / name / f"{sequence:06d}.jpg"
        try:
            source = await self._registry.transport(device_id)
            async with self._registry.device_io(device_id):
                frame = await source.capture_rgbd()
            await asyncio.to_thread(
                _write_jpeg, frame, self._root / relative, self._settings.jpeg_quality
            )
        except _SAMPLE_ERRORS as exc:
            self._drop(name, exc)
            return ""
        return relative.as_posix()

    async def _find_cameras(self) -> None:
        for device_id in self._spec.device_ids:
            try:
                transport = await self._registry.transport(device_id)
            except TransportError:
                continue
            if isinstance(transport, FrameSource):
                self._cameras[device_id] = _safe_name(device_id)

    async def _stop_sampler(self) -> None:
        sampler, self._sampler = self._sampler, None
        if sampler is None:
            return
        sampler.cancel()
        await asyncio.wait({sampler})
        if not sampler.cancelled() and sampler.exception() is not None:
            logger.error(
                "episode %s sampler failed", self.episode_id[:8], exc_info=sampler.exception()
            )
            self._dropped["sampler_failed"] = 1

    # ── Writing ──

    def _prepare(self) -> float:
        for relative in ("meta", "data", self._frames_rel):
            (self._root / relative).mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self._root).free / 1e6

    def _write(self, outcome: SkillOutcome, ended_at: float) -> None:
        stem = f"episode_{self.episode_id}"
        data = self._root / "data"
        _dump(data / f"{stem}.json", self._rows)
        _dump(data / f"{stem}.commands.json", {"commands": self._commands, "events": self._events})
        for name in self._cameras.values():
            refs = [
                {"sequence": row["sequence"], "timestamp": row["timestamp"], "frame_ref": ref}
                for row in self._rows
                if (ref := row["camera_frames"].get(name))
            ]
            _dump(self._root / self._frames_rel / f"{name}_refs.json", refs)
        line = json.dumps(self._manifest(outcome, ended_at), ensure_ascii=False, default=str)
        with _MANIFEST_LOCK:
            with (self._root / "meta" / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            _dump(self._root / "meta" / "info.json", self._info())

    def _manifest(self, outcome: SkillOutcome, ended_at: float) -> dict[str, Any]:
        plan = self._plan
        stem = f"data/episode_{self.episode_id}"
        return {
            "episode_id": self.episode_id,
            "task": str(plan.args.get("task") or self._intent or plan.summary),
            "intent": self._intent,
            "success": outcome.ok,
            "commanded": bool(self._commands),
            "started_at": self._started_at,
            "ended_at": ended_at,
            "step_count": len(self._rows),
            "fps": min(self._settings.fps, MAX_EPISODE_FPS),
            "workcell": plan.workcell,
            "skill_id": plan.skill_id,
            "skill_version": plan.version,
            "plan_digest": plan.digest,
            "args": dict(plan.args),
            "summary": plan.summary,
            "status": outcome.status,
            "failure_code": outcome.failure_code,
            "side_effect_state": outcome.side_effect_state,
            "message": outcome.message,
            "calibration": dict(plan.basis.get("calibration") or {}),
            "basis": dict(plan.basis),
            "annotations": dict(self._annotations),
            "cameras": sorted(self._cameras.values()),
            "dropped": dict(self._dropped),
            "data": f"{stem}.json",
            "commands": f"{stem}.commands.json",
        }

    def _info(self) -> dict[str, Any]:
        return {
            "format": EPISODE_FORMAT,
            "format_version": EPISODE_FORMAT_VERSION,
            "fps": min(self._settings.fps, MAX_EPISODE_FPS),
            "data_format": "json",
            "video_format": "jpeg_frames",
            "features": dict(_FEATURES),
        }

    # ── Helpers ──

    def _elapsed(self) -> float:
        return round(self._clock() - self._t0, 4)

    def _drop(self, what: str, exc: BaseException) -> None:
        count = self._dropped.get(what, 0)
        if count == 0:
            logger.warning("episode %s: %s sample failed: %s", self.episode_id[:8], what, exc)
        self._dropped[what] = count + 1

    def _skip(self, reason: str) -> None:
        logger.warning("episode %s not recorded: %s", self.episode_id[:8], reason)
        self._skipped = reason


def label_episode(
    directory: Path | None, episode_id: str, *, success: bool, note: str = "", source: str = ""
) -> dict[str, Any]:
    """Append a verified success label for a written episode; returns the label.

    ``ValueError`` when recording is off, the id is malformed, or no such
    episode was written (still running, skipped, or unknown).
    """
    if directory is None:
        raise ValueError("this node does not record episodes")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", episode_id):
        raise ValueError(f"{episode_id!r} is not an episode id")
    root = Path(directory)
    if not (root / "data" / f"episode_{episode_id}.json").is_file():
        raise ValueError(
            f"no episode {episode_id!r} was written (it is still running, was not recorded, "
            "or is unknown)"
        )
    label = {
        "episode_id": episode_id,
        "success": bool(success),
        "note": note[:MAX_LABEL_NOTE_CHARS],
        "source": source,
        "labeled_at": time.time(),
    }
    line = json.dumps(label, ensure_ascii=False)
    with _MANIFEST_LOCK:
        with (root / "meta" / "labels.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return label


def latest_labels(directory: Path) -> dict[str, dict[str, Any]]:
    """The newest label per episode id in *directory*'s ``labels.jsonl`` (may be empty)."""
    path = Path(directory) / "meta" / "labels.jsonl"
    labels: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            label = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a crash; earlier labels stand
        if isinstance(label, dict) and isinstance(label.get("episode_id"), str):
            labels[label["episode_id"]] = label
    return labels


def _write_jpeg(frame: RgbdFrame, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frame.color_jpeg(quality=quality))


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _safe_name(device_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", device_id) or "camera"


__all__ = [
    "EPISODE_FORMAT",
    "EPISODE_FORMAT_VERSION",
    "MAX_INTENT_CHARS",
    "MAX_LABEL_NOTE_CHARS",
    "NULL_TAP",
    "EpisodeRecorder",
    "EpisodeSettings",
    "EpisodeTap",
    "NullTap",
    "label_episode",
    "latest_labels",
]
