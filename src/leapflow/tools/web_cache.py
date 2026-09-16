# Copyright (c) Alibaba, Inc. and its affiliates.
"""Session-scoped body cache for ``web_fetch``.

Split out of the tool because storage is a separate responsibility from transport
and gating: this module owns where a fetched body lives, how long it stays valid,
and how it is indexed — nothing about how it was retrieved.

Every path comes from ``CacheLayout`` rather than being assembled here, and every
failure degrades to "no cache" instead of failing the fetch: caching is an
optimization, never a requirement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_CATEGORY = "web_fetch"


@dataclass(frozen=True)
class CachedBody:
    """A previously fetched body and the response facts needed to replay it."""

    body: bytes
    status: int
    final_url: str
    content_type: str
    truncated: bool
    transport: str
    elapsed_ms: int
    path: Path


@dataclass(frozen=True)
class _Slot:
    body_path: Path
    workspace_id: str
    session_id: str


def _ttl(settings: Any) -> float:
    try:
        return max(0.0, float(getattr(settings, "web_cache_ttl_s", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _slot(url: str, settings: Any) -> _Slot | None:
    """Return the layout-owned location for ``url``, or ``None`` if unavailable.

    Session scope also requires a workspace id, which keeps one TUI's fetches out
    of another's cache — the same isolation the rest of the cache tree follows.
    """
    from leapflow.tools.execution_context import current_tool_context

    ctx = current_tool_context()
    session_id = getattr(ctx, "session_id", "") or "default"
    try:
        from leapflow.cache.manager import CacheManager, CacheScope
        from leapflow.layout import workspace_id_for_path

        workspace_root = getattr(ctx, "workspace_root", None) or settings.workspace_root
        workspace_id = workspace_id_for_path(Path(workspace_root))
        manager = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
        directory = manager.path(
            scope=CacheScope.SESSION,
            category=CACHE_CATEGORY,
            workspace_id=workspace_id,
            session_id=session_id,
            source="body",
        )
    except Exception:  # noqa: BLE001 - caching must never break a fetch
        logger.debug("web cache unavailable", exc_info=True)
        return None
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return _Slot(directory / f"{digest}.bin", workspace_id, session_id)


def _meta_path(body_path: Path) -> Path:
    return body_path.with_suffix(".meta.json")


def cached_path(url: str, settings: Any) -> Path | None:
    """Return where ``url``'s body is stored, without reading it."""
    slot = _slot(url, settings)
    return slot.body_path if slot else None


def read(url: str, settings: Any) -> CachedBody | None:
    """Return a fresh cached body for ``url``, or ``None``."""
    if _ttl(settings) <= 0:
        return None
    slot = _slot(url, settings)
    if slot is None:
        return None
    meta_path = _meta_path(slot.body_path)
    try:
        if not (slot.body_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if time.time() - float(meta.get("fetched_at", 0)) > _ttl(settings):
            return None
        body = slot.body_path.read_bytes()
    except (OSError, ValueError, json.JSONDecodeError):
        logger.debug("web cache entry unreadable for %s", url, exc_info=True)
        return None
    return CachedBody(
        body=body,
        status=int(meta.get("status", 200)),
        final_url=str(meta.get("final_url") or url),
        content_type=str(meta.get("content_type") or ""),
        truncated=bool(meta.get("truncated", False)),
        transport=str(meta.get("transport") or "cache"),
        elapsed_ms=int(meta.get("elapsed_ms", 0)),
        path=slot.body_path,
    )


def write(
    url: str,
    body: bytes,
    *,
    status: int,
    final_url: str,
    content_type: str,
    truncated: bool,
    transport: str,
    elapsed_ms: int,
    origin: str,
    sensitive: bool,
    settings: Any,
) -> Path | None:
    """Persist a body, index it, and return where it was stored."""
    ttl = _ttl(settings)
    if ttl <= 0:
        return None
    slot = _slot(url, settings)
    if slot is None:
        return None
    try:
        slot.body_path.write_bytes(body)
        _meta_path(slot.body_path).write_text(
            json.dumps(
                {
                    "url": url,
                    "final_url": final_url,
                    "status": status,
                    "content_type": content_type,
                    "truncated": truncated,
                    "transport": transport,
                    "elapsed_ms": elapsed_ms,
                    "fetched_at": time.time(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("web cache write failed for %s", url, exc_info=True)
        return None
    try:
        from leapflow.cache.manager import CacheManager, CacheScope

        manager = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
        manager.register(
            path=slot.body_path,
            scope=CacheScope.SESSION,
            category=CACHE_CATEGORY,
            source="body",
            workspace_id=slot.workspace_id,
            session_id=slot.session_id,
            content_hash=hashlib.sha256(body).hexdigest(),
            expires_at=time.time() + ttl,
            # Fetched content is cheap to re-acquire and the URL itself can be the
            # secret, so it never leaves the machine.
            sensitive=sensitive,
            syncable=False,
            owner_component="web_fetch",
            metadata={"url": url, "status": status, "origin": origin},
        )
    except Exception:  # noqa: BLE001 - an unindexed cache file is still usable
        logger.debug("web cache index update failed", exc_info=True)
    return slot.body_path


__all__ = ["CACHE_CATEGORY", "CachedBody", "cached_path", "read", "write"]
