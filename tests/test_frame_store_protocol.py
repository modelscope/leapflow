# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the FrameStore Protocol and its consumers (Fix D4).

Covers:
- A fake in-memory FrameStore that satisfies the runtime_checkable Protocol.
- PerceptionSession honoring an injected store (and defaulting to LocalFrameStore).
- A LocalFrameStore save/load/list/cleanup smoke test in a tmp dir.

Hermetic: no network, no LLM. File I/O confined to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from leapflow.perception.config import PerceptionConfig
from leapflow.perception.session import PerceptionSession
from leapflow.perception.storage.frame_store import (
    FrameStore,
    FrameStoreRegistry,
    LocalFrameStore,
)


class FakeInMemoryFrameStore:
    """A minimal in-memory FrameStore implementing the Protocol.

    Stores frames in nested dicts keyed by session id; used to prove that the
    Protocol can be satisfied by a non-filesystem backend and injected cleanly.
    """

    def __init__(self) -> None:
        self._frames: Dict[str, List[Dict[str, Any]]] = {}
        self._blobs: Dict[str, bytes] = {}
        self._counter: Dict[str, int] = {}

    async def save_frame(
        self,
        session_id: str,
        frame_data: bytes,
        *,
        fmt: str = "jpeg",
        trigger: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        idx = self._counter.get(session_id, 0)
        ref = f"{session_id}/{idx:03d}.{fmt}"
        self._blobs[ref] = frame_data
        self._frames.setdefault(session_id, []).append(
            {"idx": idx, "ref": ref, "trigger": trigger, "metadata": metadata or {}}
        )
        self._counter[session_id] = idx + 1
        return ref

    async def load_frame(self, frame_ref: str) -> bytes:
        if frame_ref not in self._blobs:
            raise FileNotFoundError(frame_ref)
        return self._blobs[frame_ref]

    async def list_frames(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._frames.get(session_id, []))

    async def cleanup(self, session_id: str) -> int:
        entries = self._frames.pop(session_id, [])
        for entry in entries:
            self._blobs.pop(entry["ref"], None)
        self._counter.pop(session_id, None)
        return len(entries)


class TestProtocolConformance:
    """The fake and the real backend both satisfy the Protocol."""

    def test_fake_conforms(self) -> None:
        assert isinstance(FakeInMemoryFrameStore(), FrameStore)

    def test_local_store_conforms(self, tmp_path: Path) -> None:
        assert isinstance(LocalFrameStore(tmp_path), FrameStore)

    def test_plain_object_does_not_conform(self) -> None:
        assert not isinstance(object(), FrameStore)


class TestFakeStoreBehavior:
    """The fake store round-trips frames consistently."""

    async def test_save_load_list_cleanup(self) -> None:
        store = FakeInMemoryFrameStore()
        ref = await store.save_frame("s1", b"pixels", fmt="png", trigger="test")
        assert await store.load_frame(ref) == b"pixels"

        frames = await store.list_frames("s1")
        assert len(frames) == 1
        assert frames[0]["ref"] == ref

        removed = await store.cleanup("s1")
        assert removed == 1
        assert await store.list_frames("s1") == []
        with pytest.raises(FileNotFoundError):
            await store.load_frame(ref)


class TestPerceptionSessionInjection:
    """PerceptionSession must use an injected store and default sensibly."""

    def test_injected_store_is_used(self, tmp_path: Path) -> None:
        config = PerceptionConfig(frame_cache_dir=tmp_path)
        fake = FakeInMemoryFrameStore()
        session = PerceptionSession(config, rpc=object(), frame_store=fake)
        assert session._frame_store is fake

    def test_defaults_to_local_store(self, tmp_path: Path) -> None:
        config = PerceptionConfig(frame_cache_dir=tmp_path)
        session = PerceptionSession(config, rpc=object())
        assert isinstance(session._frame_store, LocalFrameStore)
        assert session._frame_store.cache_dir == tmp_path.expanduser().resolve()


class TestLocalFrameStoreSmoke:
    """LocalFrameStore basic filesystem round-trip in a tmp dir."""

    async def test_save_load_list_cleanup(self, tmp_path: Path) -> None:
        store = LocalFrameStore(tmp_path / "frames")
        ref = await store.save_frame(
            "sess", b"\x89PNGdata", fmt="png", trigger="unit", metadata={"k": "v"}
        )
        assert await store.load_frame(ref) == b"\x89PNGdata"

        frames = await store.list_frames("sess")
        assert len(frames) == 1
        assert frames[0]["ref"] == ref
        assert frames[0]["trigger"] == "unit"

        removed = await store.cleanup("sess")
        assert removed >= 1
        assert await store.list_frames("sess") == []

    async def test_load_missing_frame_raises(self, tmp_path: Path) -> None:
        store = LocalFrameStore(tmp_path / "frames")
        with pytest.raises(FileNotFoundError):
            await store.load_frame("sess/missing.png")


class TestFrameStoreRegistry:
    """The backend registry instantiates and rejects duplicates / unknowns."""

    def test_register_and_create(self, tmp_path: Path) -> None:
        registry = FrameStoreRegistry()
        registry.register("local", LocalFrameStore)
        store = registry.create("local", cache_dir=tmp_path)
        assert isinstance(store, LocalFrameStore)
        assert "local" in registry.list_available()

    def test_duplicate_registration_raises(self) -> None:
        registry = FrameStoreRegistry()
        registry.register("local", LocalFrameStore)
        with pytest.raises(ValueError):
            registry.register("local", LocalFrameStore)

    def test_unknown_backend_raises(self) -> None:
        registry = FrameStoreRegistry()
        with pytest.raises(KeyError):
            registry.create("s3")
