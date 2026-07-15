"""Trajectory recording lifecycle manager.

Wraps CuaDriver's trajectory recording (start_recording/stop_recording)
via the HostRpc protocol. Captures structured per-action data (screenshots,
AX tree snapshots, cursor path) plus optional full-display video.

Frame extraction uses local ffmpeg subprocess — no running host process
needed for post-hoc analysis.

Backward-compatible: exposes the same public interface as the former
VideoRecorder so existing consumers (ImitationPipeline, VideoAnalyzer)
work without changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from leapflow.perception.types import VideoSegment
from leapflow.cache.manager import CacheManager, CacheScope

if TYPE_CHECKING:
    from leapflow.platform.protocol import HostRpc

logger = logging.getLogger(__name__)

_FFMPEG_EXTRACT_TIMEOUT_S = 10.0


class TrajectoryRecorder:
    """Manages trajectory + video recording via CuaDriver MCP.

    Replaces the former VideoRecorder by using CuaDriver's
    ``start_recording`` / ``stop_recording`` MCP tools, which capture:
    - Per-action PNG screenshots
    - Per-action AX tree snapshots (``app_state.json``)
    - Cursor path at 30 Hz (``cursor.jsonl``)
    - Optional full-display MP4 (``recording.mp4``)

    Frame extraction is done locally via ffmpeg subprocess.
    """

    def __init__(
        self,
        rpc: "HostRpc",
        cache_dir: Path,
        *,
        fps: int = 5,
        resolution_scale: float = 0.5,
        codec: str = "h264",
        max_segment_s: int = 600,
        record_video: bool = True,
        cache_manager: CacheManager | None = None,
        workspace_id: str = "",
    ) -> None:
        self._rpc = rpc
        self._cache_dir = cache_dir
        self._fps = fps
        self._resolution_scale = resolution_scale
        self._codec = codec
        self._max_segment_s = max_segment_s
        self._record_video = record_video
        self._cache_manager = cache_manager
        self._workspace_id = workspace_id
        self._session_id: Optional[str] = None
        self._output_dir: Optional[Path] = None
        self._active = False
        self._paused = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    async def start(self, session_id: str) -> None:
        """Begin trajectory recording for *session_id*."""
        if self._cache_manager is not None and self._workspace_id:
            output_dir = self._cache_manager.path(
                scope=CacheScope.SESSION,
                category="video",
                workspace_id=self._workspace_id,
                session_id=session_id,
                source="recording",
            )
        else:
            output_dir = self._cache_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        self._output_dir = output_dir
        try:
            await self._rpc.call("recording.start", {
                "output_dir": str(output_dir),
                "record_video": self._record_video,
            })
            self._active = True
            logger.info(
                "trajectory_recorder.started session=%s video=%s",
                session_id, self._record_video,
            )
        except Exception as exc:
            self._active = False
            self._session_id = None
            self._output_dir = None
            logger.warning(
                "Trajectory recording unavailable: %s — continuing without recording", exc,
            )

    async def stop(self) -> List[VideoSegment]:
        """Stop recording and return segment metadata."""
        if not self._active:
            return []
        session_id = self._session_id
        output_dir = self._output_dir
        try:
            result = await self._rpc.call("recording.stop")
        except Exception as exc:
            logger.warning("recording.stop RPC failed: %s", exc)
            self._reset_state()
            return []
        self._reset_state()
        try:
            segments = self._collect_segments(result, output_dir, session_id=session_id)
            if session_id and output_dir is not None and self._cache_manager is not None and self._workspace_id:
                self._cache_manager.register_directory(
                    root=output_dir,
                    scope=CacheScope.SESSION,
                    category="video",
                    source="recording",
                    workspace_id=self._workspace_id,
                    session_id=session_id,
                    expires_at=None,
                    sensitive=True,
                    syncable=False,
                    owner_component="perception.video",
                    suffixes=(".mp4", ".mkv", ".webm", ".avi"),
                )
        except Exception as exc:
            logger.warning("Failed to collect trajectory segments: %s", exc)
            return []
        logger.info("trajectory_recorder.stopped segments=%d", len(segments))
        return segments

    async def pause(self) -> None:
        """Pause recording. No-op if not active or already paused."""
        if not self._active or self._paused:
            return
        self._paused = True
        logger.debug("trajectory_recorder.paused session=%s", self._session_id)

    async def resume(self) -> None:
        """Resume recording. No-op if not paused."""
        if not self._paused:
            return
        self._paused = False
        logger.debug("trajectory_recorder.resumed session=%s", self._session_id)

    async def extract(
        self, video_path: str, timestamp_s: float, *, max_size: int = 1024,
    ) -> Optional[bytes]:
        """FrameExtractor Protocol — graceful error handling wrapper."""
        try:
            return await self.extract_frame(video_path, timestamp_s, max_size=max_size)
        except Exception as exc:
            logger.warning("Frame extraction failed at %.1fs: %s", timestamp_s, exc)
            return None

    async def extract_frame(
        self, video_path: str, timestamp_s: float, *, max_size: int = 1024,
    ) -> bytes:
        """Extract a single JPEG frame from a video file via local ffmpeg."""
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found — install ffmpeg for frame extraction")

        cmd = [
            str(ffmpeg),
            "-ss", f"{timestamp_s:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "pipe:1",
        ]

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, timeout=_FFMPEG_EXTRACT_TIMEOUT_S,
            ),
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:200]
            raise RuntimeError(f"ffmpeg frame extraction failed (rc={result.returncode}): {stderr}")
        if not result.stdout:
            raise RuntimeError("ffmpeg returned empty frame data")
        return result.stdout

    def load_trajectory_actions(self) -> List[Dict[str, Any]]:
        """Load per-turn action data from the trajectory directory.

        Returns a list of action dicts from the turn-NNNNN/action.json files,
        ordered by turn number. Each dict may contain tool name, arguments,
        timestamps, and click points.
        """
        if self._output_dir is None:
            return []
        actions: List[Dict[str, Any]] = []
        turn_dirs = sorted(
            d for d in self._output_dir.iterdir()
            if d.is_dir() and d.name.startswith("turn-")
        )
        for td in turn_dirs:
            action_file = td / "action.json"
            if action_file.exists():
                try:
                    data = json.loads(action_file.read_text(encoding="utf-8"))
                    data["_turn_dir"] = str(td)
                    screenshot = td / "screenshot.png"
                    if screenshot.exists():
                        data["_screenshot_path"] = str(screenshot)
                    app_state = td / "app_state.json"
                    if app_state.exists():
                        data["_app_state_path"] = str(app_state)
                    actions.append(data)
                except (json.JSONDecodeError, OSError):
                    logger.debug("Skipping malformed turn dir: %s", td)
        return actions

    def _reset_state(self) -> None:
        self._active = False
        self._paused = False
        self._session_id = None
        self._output_dir = None

    def _collect_segments(
        self,
        rpc_result: Any,
        output_dir: Optional[Path],
        *,
        session_id: Optional[str] = None,
    ) -> List[VideoSegment]:
        """Build VideoSegment list from trajectory directory and RPC result."""
        sid = session_id or ""
        segments: List[VideoSegment] = []

        if isinstance(rpc_result, dict):
            raw = rpc_result.get("segments", [])
            for i, seg in enumerate(raw):
                start_time = float(seg.get("start_time", 0))
                duration_s = float(seg.get("duration_s", 0))
                end_time = float(seg.get("end_time", start_time + duration_s))
                segments.append(VideoSegment(
                    segment_id=f"{sid}_seg{i:03d}",
                    session_id=sid,
                    file_path=Path(seg.get("path", "")),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration_s,
                    fps=float(seg.get("fps", self._fps)),
                    resolution=tuple(seg.get("resolution", (0, 0))),
                    codec=seg.get("codec", self._codec),
                    file_size_bytes=int(seg.get("file_size", 0)),
                ))

        if not segments and output_dir is not None:
            mp4 = output_dir / "recording.mp4"
            if mp4.exists():
                size = mp4.stat().st_size
                segments.append(VideoSegment(
                    segment_id=f"{sid}_seg000",
                    session_id=sid,
                    file_path=mp4,
                    start_time=0.0,
                    end_time=0.0,
                    duration=0.0,
                    fps=float(self._fps),
                    resolution=(0, 0),
                    codec=self._codec,
                    file_size_bytes=size,
                ))
        return segments


def _find_ffmpeg() -> Optional[Path]:
    """Locate ffmpeg binary on PATH."""
    path = shutil.which("ffmpeg")
    return Path(path) if path else None


# Backward-compatible alias
VideoRecorder = TrajectoryRecorder
