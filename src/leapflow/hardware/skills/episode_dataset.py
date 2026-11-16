# Copyright (c) Alibaba, Inc. and its affiliates.
"""Recorded skill episodes read back as training frames (the dataset side of ``episodes``).

A training machine copies a node's episode directory and turns it into a
dataset; this module is everything that conversion needs short of the dataset
writer itself, so no converter re-implements the format, the labels or the
adapter; dataset-specific exporters only own the writer:

* ``read_episodes`` selects episodes by their *effective* verdict -- the newest
  ``labels.jsonl`` label (S2's verification) when one exists, else the run's
  own -- and can insist on a verified success;
* ``resample`` puts an episode on a fixed-rate grid by zero-order hold; each
  frame's *target* is what was commanded (the latest target per channel,
  holding the observed value of a channel nothing commanded) or, for a
  demonstration nothing commanded (``episode.record``), the next frame's
  observed state;
* a ``TrainingSpace`` turns a frame into state and action vectors:
  ``ChannelSpace`` keeps raw channel values (flange pose or joints, then the
  gripper); ``ModelSpace`` runs the recorded policy binding's adapter in the
  inverse direction (``adapt.encode``), so the vectors are in the model's own
  normalised space, exactly what it consumes and emits on this embodiment.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.hardware.skills.adapt.encode import EpisodeEncoder
from leapflow.hardware.skills.adapt.spec import ImageSpec
from leapflow.hardware.skills.episodes import (
    EPISODE_FORMAT,
    EPISODE_FORMAT_VERSION,
    latest_labels,
)
from leapflow.hardware.skills.workcell import GRIPPER_POSITION, POSE_CHANNELS

SPACE_EE = "ee"
SPACE_JOINT = "joint"
SPACE_MODEL = "model"
SPACES = (SPACE_EE, SPACE_JOINT, SPACE_MODEL)
_GRID_EPS = 1e-9

Sample = Mapping[str, float]


@dataclass(frozen=True)
class EpisodeFilter:
    """Which episodes a dataset takes."""

    skills: tuple[str, ...] = ()  # skill id globs; empty takes every skill
    workcell: str = ""
    min_rows: int = 1
    include_failures: bool = False
    verified_only: bool = False  # only episodes S2 (or an operator) labelled a success


@dataclass(frozen=True)
class RecordedEpisode:
    entry: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    success: bool  # the effective verdict: the newest label, else the run's own
    verified: bool | None = None  # the newest label's verdict, None when unlabelled

    @property
    def episode_id(self) -> str:
        return str(self.entry["episode_id"])

    @property
    def commanded(self) -> bool:
        return bool(self.entry.get("commanded", True))

    @property
    def task(self) -> str:
        return str(self.entry.get("task") or self.entry.get("summary") or "")

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(self.entry.get("cameras") or ())

    @property
    def encoding(self) -> Mapping[str, Any] | None:
        """The policy binding the run was adapted under, when it ran one."""
        policy = (self.entry.get("annotations") or {}).get("policy") or {}
        return policy.get("encoding")

    @property
    def policy_cameras(self) -> Mapping[str, str]:
        """The policy's camera names to the device ids that fed them, when it ran one."""
        policy = (self.entry.get("annotations") or {}).get("policy") or {}
        return dict(policy.get("cameras") or {})


@dataclass(frozen=True)
class Frame:
    """One grid tick: the observed sample, the action target and each camera's frame path."""

    sample: Sample
    target: Sample
    images: Mapping[str, str] = field(default_factory=dict)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_episodes(root: Path, filters: EpisodeFilter) -> list[RecordedEpisode]:
    """The episodes under *root* that *filters* takes; ``ValueError`` for another format."""
    info = _load(root / "meta" / "info.json")
    if info.get("format") != EPISODE_FORMAT or info.get("format_version") != (
        EPISODE_FORMAT_VERSION
    ):
        raise ValueError(f"{root} is not a {EPISODE_FORMAT} v{EPISODE_FORMAT_VERSION} dataset")
    labels = latest_labels(root)
    episodes: list[RecordedEpisode] = []
    for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        label = labels.get(str(entry.get("episode_id")))
        verified = None if label is None else bool(label.get("success"))
        success = bool(entry.get("success")) if verified is None else verified
        if filters.verified_only and verified is not True:
            continue
        if not success and not filters.include_failures:
            continue
        if filters.skills and not any(
            fnmatch.fnmatch(str(entry.get("skill_id")), pattern) for pattern in filters.skills
        ):
            continue
        if filters.workcell and entry.get("workcell") != filters.workcell:
            continue
        rows = tuple(_load(root / entry["data"]))
        if len(rows) >= filters.min_rows:
            episodes.append(RecordedEpisode(entry, rows, success, verified))
    return episodes


def sample_of(row: Mapping[str, Any]) -> dict[str, float]:
    """A row's observed channel values: joints, the flange pose and the gripper."""
    sample = {str(k): float(v) for k, v in (row.get("joint_positions") or {}).items()}
    sample.update({str(k): float(v) for k, v in (row.get("ee_pose") or {}).items()})
    if row.get("gripper_state") is not None:
        sample[GRIPPER_POSITION] = float(row["gripper_state"])
    return sample


def resample(
    episode: RecordedEpisode, channels: Sequence[str], cameras: Sequence[str], fps: float
) -> list[Frame]:
    """*episode* on a *fps* grid from its first row; ticks before every channel and
    camera has a value are dropped, later gaps hold the previous value."""
    rows = sorted(episode.rows, key=lambda row: float(row["t"]))
    if not rows:
        return []
    sample: dict[str, float] = {}
    action: dict[str, float] = {}
    images: dict[str, str] = {}
    ticks: list[tuple[dict[str, float], dict[str, float], dict[str, str]]] = []
    index, tick = 0, 0
    start, end = float(rows[0]["t"]), float(rows[-1]["t"])
    while start + tick / fps <= end + _GRID_EPS:
        at = start + tick / fps
        tick += 1
        while index < len(rows) and float(rows[index]["t"]) <= at + _GRID_EPS:
            row = rows[index]
            sample.update(sample_of(row))
            action.update({str(k): float(v) for k, v in (row.get("action") or {}).items()})
            images.update(row.get("camera_frames") or {})
            index += 1
        if any(c not in sample for c in channels) or any(c not in images for c in cameras):
            continue
        ticks.append((dict(sample), dict(action), {c: images[c] for c in cameras}))
    frames: list[Frame] = []
    for at, (observed, commanded, shot) in enumerate(ticks):
        if episode.commanded:
            target = {**observed, **commanded}
        else:
            target = ticks[min(at + 1, len(ticks) - 1)][0]
        frames.append(Frame(observed, target, shot))
    return frames


def camera_names(episodes: Sequence[RecordedEpisode]) -> tuple[str, ...]:
    return tuple(sorted({name for episode in episodes for name in episode.cameras}))


def camera_keys(episodes: Sequence[RecordedEpisode], *, model: bool) -> dict[str, str]:
    """Recorded camera device id -> dataset image key.

    A ``model`` dataset keeps the cameras the policy observed, under the names
    it observed them by; otherwise (or with no policy recorded) every camera
    keeps its device id.
    """
    if model:
        keys = {device: name for e in episodes for name, device in e.policy_cameras.items()}
        if keys:
            return dict(sorted(keys.items()))
    return {name: name for name in camera_names(episodes)}


# ── Training spaces ──────────────────────────────────────────────────────────


@runtime_checkable
class TrainingSpace(Protocol):
    """How a frame becomes the dataset's state and action vectors."""

    @property
    def channels(self) -> tuple[str, ...]:
        """The channels every frame must hold."""
        ...

    @property
    def state_names(self) -> tuple[str, ...]: ...

    @property
    def action_names(self) -> tuple[str, ...]: ...

    @property
    def fps(self) -> float | None:
        """The rate the space was declared at (``None``: any)."""
        ...

    @property
    def image_specs(self) -> Mapping[str, ImageSpec]:
        """Model camera name to its inference-time fitting declaration."""
        ...

    def state(self, frame: Frame) -> tuple[float, ...]: ...

    def action(self, frame: Frame) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ChannelSpace:
    """Raw channel values (mm, deg): the same channels for state and action."""

    channels: tuple[str, ...]

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(channel.removeprefix("ee.") for channel in self.channels)

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.state_names

    @property
    def fps(self) -> float | None:
        return None

    @property
    def image_specs(self) -> Mapping[str, ImageSpec]:
        return {}

    def state(self, frame: Frame) -> tuple[float, ...]:
        return tuple(frame.sample[c] for c in self.channels)

    def action(self, frame: Frame) -> tuple[float, ...]:
        return tuple(frame.target[c] for c in self.channels)


@dataclass(frozen=True)
class ModelSpace:
    """The recorded policy binding's native space, encoded by its own adapter."""

    encoder: EpisodeEncoder

    @property
    def channels(self) -> tuple[str, ...]:
        return self.encoder.channels

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.encoder.state_names

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.encoder.action_names

    @property
    def fps(self) -> float | None:
        return self.encoder.spaces.action.fps

    @property
    def image_specs(self) -> Mapping[str, ImageSpec]:
        return self.encoder.spaces.observation.images

    def state(self, frame: Frame) -> tuple[float, ...]:
        return self.encoder.encode_state(frame.sample)

    def action(self, frame: Frame) -> tuple[float, ...]:
        return self.encoder.encode_action(frame.target, frame.sample)


def training_space(
    kind: str,
    episodes: Sequence[RecordedEpisode],
    *,
    encoding: Mapping[str, Any] | None = None,
) -> TrainingSpace:
    """The space *kind* names for *episodes*; raises ``ValueError`` when they cannot share it.

    ``model`` takes *encoding* when given (a hand-written record for
    demonstrations, or to re-encode every episode under one binding), else each
    episode's recorded binding -- which must then be one and the same.
    """
    if kind == SPACE_EE:
        return ChannelSpace((*POSE_CHANNELS, GRIPPER_POSITION))
    if kind == SPACE_JOINT:
        joints = sorted({c for e in episodes for r in e.rows for c in r.get("joint_positions", {})})
        if not joints:
            raise ValueError("no episode recorded joint positions")
        return ChannelSpace((*joints, GRIPPER_POSITION))
    if kind != SPACE_MODEL:
        raise ValueError(f"the training space must be one of {', '.join(SPACES)}, not {kind!r}")
    if encoding is None:
        missing = [e.episode_id for e in episodes if e.encoding is None]
        if missing:
            raise ValueError(
                f"episodes {missing} ran no policy, so they pin no encoding; pass one explicitly"
            )
        distinct = {json.dumps(e.encoding, sort_keys=True) for e in episodes}
        if len(distinct) > 1:
            raise ValueError(
                "the episodes ran under different policy bindings; convert them separately or "
                "pass one encoding for all"
            )
        encoding = episodes[0].encoding if episodes else None
    if encoding is None:
        raise ValueError("no episode to take the encoding from")
    return ModelSpace(EpisodeEncoder.from_record(encoding))


__all__ = [
    "SPACES",
    "SPACE_EE",
    "SPACE_JOINT",
    "SPACE_MODEL",
    "ChannelSpace",
    "EpisodeFilter",
    "Frame",
    "ModelSpace",
    "RecordedEpisode",
    "TrainingSpace",
    "camera_keys",
    "camera_names",
    "read_episodes",
    "resample",
    "sample_of",
    "training_space",
]
