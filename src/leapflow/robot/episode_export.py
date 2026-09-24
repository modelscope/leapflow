# Copyright (c) Alibaba, Inc. and its affiliates.
"""Episode export: physical trajectories to standard training formats.

Exports recorded PhysicalTrajectory / PhysicalEpisode data to the standard
Parquet + MP4 layout used by offline imitation learning pipelines (the same
layout LeapRobot's absorbed policy loader expects).  Training itself is out
of scope — LeapFlow records and exports; a separate offline pipeline trains.

pyarrow and opencv are *optional* dependencies.  When unavailable, export
falls back to JSON + individual frame files and logs a degradation warning.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------

_PYARROW_AVAILABLE = False
try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401

    _PYARROW_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

_CV2_AVAILABLE = False
try:
    import cv2  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]

_NUMPY_AVAILABLE = False
try:
    import numpy as np  # noqa: F401

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# EpisodeExporter
# ---------------------------------------------------------------------------


class EpisodeExporter:
    """Exports physical episodes to Parquet (state/action) + MP4 (frames).

    Directory layout::

        export_dir/
            meta/info.json              # schema, fps, features
            meta/episodes.parquet       # per-episode metadata (or .json)
            data/episode_NNN.parquet    # per-step state/action rows (or .json)
            videos/episode_NNN/{cam}.mp4 # camera frames (or raw frames/)

    When ``pyarrow`` is not installed, ``.parquet`` files are replaced with
    ``.json`` and a warning is logged.  When ``opencv-python`` is not
    installed, MP4 encoding is skipped and raw frame references are
    preserved.
    """

    def __init__(self, export_dir: str | Path, *, fps: float = 30.0) -> None:
        self._export_dir = Path(export_dir)
        self._fps = max(1.0, float(fps))
        self._episodes_exported = 0

    @property
    def export_dir(self) -> Path:
        return self._export_dir

    @property
    def episodes_exported(self) -> int:
        return self._episodes_exported

    # -- Public API -----------------------------------------------------------

    async def export_episode(self, episode: Any) -> dict[str, Any]:
        """Export one :class:`PhysicalEpisode`.

        Returns a summary dict: ``{episode_id, files_written, format, ...}``.
        """
        episode_id: str = getattr(episode, "episode_id", "unknown")
        trajectory = getattr(episode, "trajectory", None)
        task: str = getattr(episode, "task", "")
        success: bool = getattr(episode, "success", False)
        started_at: float = getattr(episode, "started_at", 0.0)
        ended_at: float = getattr(episode, "ended_at", 0.0)

        steps: tuple[Any, ...] = ()
        if trajectory is not None:
            steps = getattr(trajectory, "steps", ())

        files_written: list[str] = []

        # 1. State/action Parquet (or JSON).
        data_dir = self._export_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if _PYARROW_AVAILABLE:
            parquet_path = data_dir / f"episode_{episode_id}.parquet"
            self._write_state_action_parquet(steps, parquet_path)
            files_written.append(str(parquet_path))
        else:
            json_path = data_dir / f"episode_{episode_id}.json"
            self._write_state_action_json(steps, json_path)
            files_written.append(str(json_path))
            logger.warning(
                "pyarrow not available; exported episode %s as JSON instead of Parquet.",
                episode_id,
            )

        # 2. Camera frames → MP4 (or raw frame references).
        camera_ids = self._collect_camera_ids(steps)
        for camera_id in camera_ids:
            video_dir = self._export_dir / "videos" / f"episode_{episode_id}"
            video_dir.mkdir(parents=True, exist_ok=True)
            if _CV2_AVAILABLE and _NUMPY_AVAILABLE:
                mp4_path = video_dir / f"{camera_id}.mp4"
                self._write_frames_mp4(steps, camera_id, mp4_path)
                files_written.append(str(mp4_path))
            else:
                refs_path = video_dir / f"{camera_id}_refs.json"
                self._write_frame_refs_json(steps, camera_id, refs_path)
                files_written.append(str(refs_path))
                logger.warning(
                    "opencv/numpy not available; wrote frame references for %s/%s.",
                    episode_id, camera_id,
                )

        # 3. Episode metadata → episodes manifest.
        meta_dir = self._export_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        ep_meta = {
            "episode_id": episode_id,
            "task": task,
            "success": success,
            "started_at": started_at,
            "ended_at": ended_at,
            "step_count": len(steps),
            "fps": self._fps,
        }
        self._append_episode_manifest(meta_dir, ep_meta)

        # 4. Info JSON.
        self._write_info_json(
            meta_dir,
            episode_id=episode_id,
            step_count=len(steps),
            camera_ids=camera_ids,
        )

        self._episodes_exported += 1
        return {
            "episode_id": episode_id,
            "files_written": files_written,
            "format": "parquet" if _PYARROW_AVAILABLE else "json",
            "step_count": len(steps),
        }

    async def export_trajectory(
        self,
        trajectory: Any,
        *,
        task: str = "",
    ) -> dict[str, Any]:
        """Export a raw :class:`PhysicalTrajectory` (demonstration without outcome).

        Wraps the trajectory in a minimal episode-like structure and delegates
        to :meth:`export_episode`.
        """
        trajectory_id: str = getattr(trajectory, "trajectory_id", "unknown")

        class _MinimalEpisode:
            """Lightweight shim so export_episode can consume a raw trajectory."""

            def __init__(self, traj: Any, task_: str) -> None:
                self.episode_id = trajectory_id
                self.trajectory = traj
                self.task = task_
                self.success = False
                self.started_at = getattr(traj, "started_at", 0.0)
                self.ended_at = getattr(traj, "ended_at", 0.0)

        shim = _MinimalEpisode(trajectory, task)
        return await self.export_episode(shim)

    # -- Internal: Parquet / JSON writers -------------------------------------

    def _write_state_action_parquet(
        self,
        steps: tuple[Any, ...] | list[Any],
        path: Path,
    ) -> None:
        """Write per-step joint positions/velocities/gripper/actions to Parquet."""
        if not _PYARROW_AVAILABLE or not steps:
            return

        timestamps: list[float] = []
        sequences: list[int] = []
        action_sources: list[str] = []
        gripper_states: list[float | None] = []
        joint_positions_list: list[str] = []
        joint_velocities_list: list[str] = []

        for step in steps:
            timestamps.append(float(getattr(step, "timestamp", 0.0)))
            sequences.append(int(getattr(step, "sequence", 0)))
            action_sources.append(str(getattr(step, "action_source", "")))
            gripper_states.append(getattr(step, "gripper_state", None))
            positions = dict(getattr(step, "joint_positions", {}))
            velocities = dict(getattr(step, "joint_velocities", {}))
            joint_positions_list.append(json.dumps(positions, ensure_ascii=False))
            joint_velocities_list.append(json.dumps(velocities, ensure_ascii=False))

        table = pa.table({
            "timestamp": pa.array(timestamps, type=pa.float64()),
            "sequence": pa.array(sequences, type=pa.int32()),
            "action_source": pa.array(action_sources, type=pa.string()),
            "gripper_state": pa.array(
                [g if g is not None else float("nan") for g in gripper_states],
                type=pa.float64(),
            ),
            "joint_positions": pa.array(joint_positions_list, type=pa.string()),
            "joint_velocities": pa.array(joint_velocities_list, type=pa.string()),
        })
        pq.write_table(table, str(path))

    def _write_state_action_json(
        self,
        steps: tuple[Any, ...] | list[Any],
        path: Path,
    ) -> None:
        """Fallback: write step data as JSON when pyarrow is unavailable."""
        rows: list[dict[str, Any]] = []
        for step in steps:
            to_dict_fn = getattr(step, "to_dict", None)
            rows.append(to_dict_fn() if callable(to_dict_fn) else {"raw": str(step)})
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_frames_mp4(
        self,
        steps: tuple[Any, ...] | list[Any],
        camera_id: str,
        path: Path,
    ) -> None:
        """Encode a camera's frame sequence to MP4 (best-effort).

        Frames are expected to be file paths (as per PhysicalTrajectoryStep
        convention: camera_frames stores references, not inline bytes).
        Reads each frame from disk, encodes to MP4.
        """
        if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
            return

        writer = None
        try:
            for step in steps:
                camera_frames = dict(getattr(step, "camera_frames", {}))
                frame_ref = camera_frames.get(camera_id)
                if not frame_ref:
                    continue
                frame_path = Path(frame_ref)
                if not frame_path.exists():
                    continue

                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                if writer is None:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))

                writer.write(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MP4 encoding failed for %s: %s", camera_id, exc)
        finally:
            if writer is not None:
                writer.release()

    def _write_frame_refs_json(
        self,
        steps: tuple[Any, ...] | list[Any],
        camera_id: str,
        path: Path,
    ) -> None:
        """Fallback: write frame references as JSON."""
        refs: list[dict[str, Any]] = []
        for idx, step in enumerate(steps):
            camera_frames = dict(getattr(step, "camera_frames", {}))
            frame_ref = camera_frames.get(camera_id)
            if frame_ref:
                refs.append({
                    "sequence": idx,
                    "timestamp": float(getattr(step, "timestamp", 0.0)),
                    "frame_ref": str(frame_ref),
                })
        path.write_text(
            json.dumps(refs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_info_json(
        self,
        meta_dir: Path,
        *,
        episode_id: str,
        step_count: int,
        camera_ids: list[str],
    ) -> None:
        """Write the info.json schema descriptor."""
        info = {
            "fps": self._fps,
            "format_version": 1,
            "last_episode_id": episode_id,
            "last_step_count": step_count,
            "camera_ids": camera_ids,
            "features": {
                "timestamp": "float64",
                "sequence": "int32",
                "action_source": "string",
                "gripper_state": "float64",
                "joint_positions": "json_string",
                "joint_velocities": "json_string",
            },
            "data_format": "parquet" if _PYARROW_AVAILABLE else "json",
            "video_format": "mp4" if _CV2_AVAILABLE else "frame_refs",
        }
        info_path = meta_dir / "info.json"
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_episode_manifest(
        self,
        meta_dir: Path,
        ep_meta: dict[str, Any],
    ) -> None:
        """Append an episode entry to the manifest file.

        Uses JSON lines for simplicity and append-friendliness.  A Parquet
        manifest can be built as a post-processing step.
        """
        manifest_path = meta_dir / "episodes.jsonl"
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep_meta, ensure_ascii=False) + "\n")

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _collect_camera_ids(steps: tuple[Any, ...] | list[Any]) -> list[str]:
        """Collect unique camera ids across all steps."""
        seen: set[str] = set()
        ordered: list[str] = []
        for step in steps:
            camera_frames = getattr(step, "camera_frames", {})
            for cam_id in camera_frames:
                if cam_id not in seen:
                    seen.add(cam_id)
                    ordered.append(cam_id)
        return ordered


__all__ = [
    "EpisodeExporter",
]
