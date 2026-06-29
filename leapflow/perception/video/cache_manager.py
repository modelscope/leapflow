"""Video cache lifecycle management."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = frozenset((".mp4", ".mkv", ".webm", ".avi"))


class VideoCacheManager:
    """Manages video cache storage with configurable retention policies.

    Policies (applied in order):
    1. Age-based: remove files older than max_age_days
    2. Size-based: if total size exceeds max_size_gb, remove oldest files first
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_age_days: int = 7,
        max_size_gb: float = 5.0,
    ) -> None:
        self._cache_dir = cache_dir
        self._max_age_s = max_age_days * 86400
        self._max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

    def cleanup(self) -> int:
        """Run cleanup policies. Returns number of files removed."""
        if not self._cache_dir.exists():
            return 0

        removed = 0
        now = time.time()

        # Phase 1: Age-based removal
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

        # Phase 2: Size-based removal (oldest first)
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
