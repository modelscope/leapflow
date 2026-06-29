"""Video recording lifecycle manager.

Wraps Host-side video capture (SCStream + AVAssetWriter) via RPC,
providing a zero-decision recording interface.  All intelligence
about *what to analyze* lives in the offline analysis phase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from leapflow.perception.types import VideoSegment

if TYPE_CHECKING:
    from leapflow.platform.protocol import HostRpc

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Manages continuous video recording via Host RPC."""

    def __init__(
        self,
        rpc: "HostRpc",
        cache_dir: Path,
        *,
        fps: int = 5,
        resolution_scale: float = 0.5,
        codec: str = "h264",
        max_segment_s: int = 600,
    ) -> None:
        self._rpc = rpc
        self._cache_dir = cache_dir
        self._fps = fps
        self._resolution_scale = resolution_scale
        self._codec = codec
        self._max_segment_s = max_segment_s
        self._session_id: Optional[str] = None
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
        """Begin video recording for *session_id*."""
        output_dir = self._cache_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        try:
            await self._rpc.call("video.start", {
                "trajectory_id": session_id,
                "output_dir": str(output_dir),
                "fps": self._fps,
                "resolution_scale": self._resolution_scale,
                "codec": self._codec,
                "max_segment_s": self._max_segment_s,
            })
            self._active = True
            logger.info("video_recorder.started session=%s fps=%d", session_id, self._fps)
        except Exception as exc:
            self._active = False
            self._session_id = None
            logger.warning("Video recording unavailable: %s \u2014 continuing without video", exc)

    async def stop(self) -> List[VideoSegment]:
        """Stop recording and return segment metadata."""
        if not self._active:
            return []
        # Preserve session_id for segment parsing before clearing state
        session_id = self._session_id
        try:
            result = await self._rpc.call("video.stop")
        except Exception as exc:
            logger.warning("video.stop RPC failed: %s", exc)
            self._active = False
            self._paused = False
            self._session_id = None
            return []
        self._active = False
        self._paused = False
        self._session_id = None
        try:
            segments = self._parse_segments(result, session_id=session_id)
        except Exception as exc:
            logger.warning("Failed to parse video segments: %s", exc)
            return []
        logger.info("video_recorder.stopped segments=%d", len(segments))
        return segments

    async def pause(self) -> None:
        """Pause video recording. No-op if not active or already paused."""
        if not self._active or self._paused:
            return
        try:
            await self._rpc.call("video.pause")
            self._paused = True
            logger.debug("video_recorder.paused session=%s", self._session_id)
        except Exception as exc:
            logger.warning("video.pause RPC failed: %s", exc)

    async def resume(self) -> None:
        """Resume video recording. No-op if not paused."""
        if not self._paused:
            return
        try:
            await self._rpc.call("video.resume")
            self._paused = False
            logger.debug("video_recorder.resumed session=%s", self._session_id)
        except Exception as exc:
            logger.warning("video.resume RPC failed: %s", exc)

    async def extract(self, video_path: str, timestamp_s: float, *, max_size: int = 1024) -> Optional[bytes]:
        """FrameExtractor Protocol implementation with graceful error handling."""
        try:
            return await self.extract_frame(video_path, timestamp_s, max_size=max_size)
        except Exception as exc:
            logger.warning("Frame extraction failed at %.1fs: %s", timestamp_s, exc)
            return None

    async def extract_frame(self, video_path: str, timestamp_s: float, *, max_size: int = 1024) -> bytes:
        """Extract a single frame from a video file at *timestamp_s*."""
        import base64
        result = await self._rpc.call("video.extract_frame", {
            "video_path": video_path,
            "timestamp_s": timestamp_s,
            "max_size": max_size,
        })
        return base64.b64decode(result.get("frame_base64", ""))

    def _parse_segments(self, rpc_result: Any, *, session_id: Optional[str] = None) -> List[VideoSegment]:
        raw_segments = rpc_result.get("segments", []) if isinstance(rpc_result, dict) else []
        segments: List[VideoSegment] = []
        sid = session_id or self._session_id or ""
        for i, seg in enumerate(raw_segments):
            start_time = float(seg.get("start_time", 0))
            duration_s = float(seg.get("duration_s", 0))
            # Prefer RPC-provided end_time; fall back to start_time + duration_s
            if "end_time" in seg:
                end_time = float(seg["end_time"])
            else:
                end_time = start_time + duration_s
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
        return segments
