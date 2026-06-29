"""Frame storage abstraction and local filesystem implementation.

Migrated from leapflow.recording.frame_store with extended metadata
sidecar support for the perception subsystem.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class FrameStore(ABC):
    """Abstract frame storage interface."""

    @abstractmethod
    async def save_frame(
        self,
        session_id: str,
        frame_data: bytes,
        *,
        fmt: str = "jpeg",
        trigger: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a frame and return its unique reference string."""
        ...

    @abstractmethod
    async def load_frame(self, frame_ref: str) -> bytes:
        """Load frame data by reference."""
        ...

    @abstractmethod
    async def list_frames(self, session_id: str) -> List[Dict[str, Any]]:
        """List frame metadata for a session."""
        ...

    @abstractmethod
    async def cleanup(self, session_id: str) -> int:
        """Remove all frames for a session. Return deleted count."""
        ...


class LocalFrameStore(FrameStore):
    """Local filesystem frame storage with metadata sidecars.

    Storage layout:
        {cache_dir}/{session_id}/
            000_{timestamp}.jpeg       (frame data)
            000_{timestamp}.json       (metadata sidecar)
            manifest.json              (frame index)
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir.expanduser().resolve()
        self._counters: Dict[str, int] = {}

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    async def save_frame(
        self,
        session_id: str,
        frame_data: bytes,
        *,
        fmt: str = "jpeg",
        trigger: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        session_dir = self._cache_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        idx = self._counters.get(session_id, 0)
        ts = time.time()
        ts_int = int(ts)
        filename = f"{idx:03d}_{ts_int}.{fmt}"
        filepath = session_dir / filename

        filepath.write_bytes(frame_data)
        self._counters[session_id] = idx + 1

        frame_ref = f"{session_id}/{filename}"
        entry = {
            "idx": idx,
            "filename": filename,
            "timestamp": ts,
            "size": len(frame_data),
            "format": fmt,
            "trigger": trigger,
            "ref": frame_ref,
        }

        # Write metadata sidecar
        if metadata:
            entry["metadata"] = metadata
            sidecar_path = session_dir / f"{idx:03d}_{ts_int}.json"
            sidecar_path.write_text(
                json.dumps(metadata, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        self._update_manifest(session_dir, entry)
        return frame_ref

    async def load_frame(self, frame_ref: str) -> bytes:
        filepath = self._cache_dir / frame_ref
        if not filepath.exists():
            raise FileNotFoundError(f"Frame not found: {frame_ref}")
        return filepath.read_bytes()

    async def list_frames(self, session_id: str) -> List[Dict[str, Any]]:
        manifest_path = self._cache_dir / session_id / "manifest.json"
        if not manifest_path.exists():
            return []
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data.get("frames", [])

    async def cleanup(self, session_id: str) -> int:
        session_dir = self._cache_dir / session_id
        if not session_dir.exists():
            return 0
        count = 0
        for f in session_dir.iterdir():
            f.unlink()
            count += 1
        session_dir.rmdir()
        self._counters.pop(session_id, None)
        return count

    def _update_manifest(self, session_dir: Path, entry: Dict[str, Any]) -> None:
        manifest_path = session_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"frames": []}
        manifest["frames"].append(entry)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
