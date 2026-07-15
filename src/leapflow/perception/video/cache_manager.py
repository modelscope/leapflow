"""Video cache lifecycle management."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

from leapflow.cache.manager import CacheManager, CacheScope

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = frozenset((".mp4", ".mkv", ".webm", ".avi"))


class VideoCacheManager:
    """Video cache policy wrapper backed by the unified CacheManager index."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_age_days: int = 7,
        max_size_gb: float = 5.0,
        cache_manager: CacheManager | None = None,
        workspace_id: str = "",
    ) -> None:
        self._cache_dir = cache_dir
        self._max_age_s = max_age_days * 86400
        self._max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._cache_manager = cache_manager
        self._workspace_id = workspace_id

    def register_session(self, session_id: str, root: Path) -> int:
        """Register session video artifacts in the unified cache index."""
        if self._cache_manager is None:
            return 0
        entries = self._cache_manager.register_directory(
            root=root,
            scope=CacheScope.SESSION,
            category="video",
            source="recording",
            workspace_id=self._workspace_id,
            session_id=session_id,
            expires_at=time.time() + self._max_age_s,
            sensitive=True,
            syncable=False,
            owner_component="perception.video",
            suffixes=_VIDEO_EXTENSIONS,
        )
        return len(entries)

    def cleanup(self) -> int:
        """Run cleanup policies. Returns number of files removed."""
        removed = 0
        if self._cache_manager is not None:
            removed += self._cache_manager.cleanup_expired()
            if self._workspace_id:
                removed += self._cache_manager.cleanup_quota(
                    scope=CacheScope.SESSION.value,
                    category="video",
                    workspace_id=self._workspace_id,
                    max_bytes=self._max_size_bytes,
                )
        if not self._cache_dir.exists():
            return removed

        now = time.time()

        # Fallback guard for unregistered files under the managed video root.
        for f in self._iter_video_files():
            age = now - f.stat().st_mtime
            if age > self._max_age_s:
                f.unlink(missing_ok=True)
                removed += 1
                logger.debug(
                    "cache_cleanup: removed aged file %s (%.1f days)",
                    f.name,
                    age / 86400,
                )

        files = sorted(self._iter_video_files(), key=lambda p: p.stat().st_mtime)
        total_size = sum(f.stat().st_size for f in files)

        while total_size > self._max_size_bytes and files:
            oldest = files.pop(0)
            total_size -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            removed += 1
            logger.debug("cache_cleanup: removed oversized file %s", oldest.name)

        if removed:
            logger.info(
                "cache_cleanup: removed %d files from %s", removed, self._cache_dir
            )

        return removed

    def _iter_video_files(self) -> Iterator[Path]:
        """Iterate video files in cache directory."""
        return (
            f
            for f in self._cache_dir.rglob("*")
            if f.is_file() and f.suffix in _VIDEO_EXTENSIONS
        )
