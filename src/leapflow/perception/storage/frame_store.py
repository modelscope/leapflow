# Copyright (c) Alibaba, Inc. and its affiliates.
"""Frame storage abstraction and local filesystem implementation.

Migrated from leapflow.recording.frame_store with extended metadata
sidecar support for the perception subsystem.

The FrameStore protocol uses typing.Protocol (runtime_checkable) instead of
ABC to enable duck-typing conformance checks without inheritance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class FrameStore(Protocol):
    """Protocol for frame storage backends.

    Implementations provide async frame save/load/list/cleanup operations.
    Conformance is checked via duck-typing (no inheritance required).
    """

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

    async def load_frame(self, frame_ref: str) -> bytes:
        """Load frame data by reference."""
        ...

    async def list_frames(self, session_id: str) -> List[Dict[str, Any]]:
        """List frame metadata for a session."""
        ...

    async def cleanup(self, session_id: str) -> int:
        """Remove all frames for a session. Return deleted count."""
        ...


class FrameStoreRegistry:
    """Registry for frame storage backends.

    Manages backend factories by id and instantiates them on demand.
    """

    def __init__(self) -> None:
        self._backends: Dict[str, type] = {}

    def register(self, backend_id: str, factory: type) -> None:
        """Register a backend factory by id.

        Parameters
        ----------
        backend_id : str
            Unique identifier for the backend (e.g., 'local', 's3').
        factory : type
            Class or callable that creates FrameStore instances.
        """
        if backend_id in self._backends:
            raise ValueError(f"Duplicate backend_id: {backend_id!r}")
        self._backends[backend_id] = factory

    def create(self, backend_id: str, **kwargs: Any) -> FrameStore:
        """Instantiate a backend by id.

        Raises
        ------
        KeyError
            If no backend with the given id is registered.
        """
        factory = self._backends.get(backend_id)
        if factory is None:
            raise KeyError(f"No FrameStore backend registered with id: {backend_id!r}")
        return factory(**kwargs)

    def list_available(self) -> List[str]:
        """List all registered backend ids."""
        return list(self._backends.keys())


class LocalFrameStore:
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
