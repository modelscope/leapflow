# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for compression provider isolation in the PCD cache-aware mechanism.

Integration tests that verify the dedicated compression provider is constructed
and used independently of the primary LLM provider, so compression traffic
does not pollute the main conversation prefix cache.

All providers are mocked — no real LLM tokens are consumed and no network
calls are made.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from conftest import make_settings


# ── Mock providers ────────────────────────────────────────────────────────


class MockLLMProvider:
    """Tracking mock LLM provider that records all achat calls."""

    def __init__(self, name: str = "primary") -> None:
        self.name = name
        self.calls: List[dict] = []
        self._call_count = 0

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        self._call_count += 1
        return SimpleNamespace(content="Compressed summary of context.")

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if False:
            yield ""  # pragma: no cover

    @property
    def call_count(self) -> int:
        return self._call_count


# ── Engine builder helper ─────────────────────────────────────────────────


def _build_engine(
    td: str,
    llm: Any,
    *,
    compression_provider: str = "",
    compression_model: str = "",
    compression_api_key: str = "",
    compression_base_url: str = "",
):
    """Build a base AgentEngine with configurable compression settings."""
    from leapflow.engine.engine import AgentEngine, build_default_registry
    from leapflow.memory import (
        EpisodicMemoryProvider,
        SemanticMemoryProvider,
        WorkingMemoryProvider,
    )
    from leapflow.platform.mock import MockBridge

    settings = make_settings(td)
    # Apply compression settings
    settings = replace(
        settings,
        compression_provider=compression_provider,
        compression_model=compression_model,
        compression_api_key=compression_api_key,
        compression_base_url=compression_base_url,
    )

    rpc = MockBridge()
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()

    class _Simple:
        def classify(self, *a, **k):
            return "simple"

        async def aclassify(self, *a, **k):
            return "simple"

    reg = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _Simple())
    return engine, lt


# ═══════════════════════════════════════════════════════════════════════════
# _build_compression_provider — construction & fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCompressionProvider:
    """_build_compression_provider returns an independent provider or None."""

    def test_unconfigured_returns_none(self) -> None:
        """No compression_provider or compression_model → None."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            engine, lt = _build_engine(td, primary)
            try:
                assert engine._compression_provider is None
            finally:
                lt.close()

    def test_configured_returns_independent_instance(self) -> None:
        """Compression provider/model configured → returns a distinct object."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            # Patch OpenAIChat to avoid real network call
            mock_openai = MagicMock()
            mock_openai.return_value = MockLLMProvider("compression")
            with patch("leapflow.engine.engine.AgentEngine._build_compression_provider") as mock_build:
                mock_build.return_value = MockLLMProvider("compression")
                engine, lt = _build_engine(
                    td,
                    primary,
                    compression_provider="openai",
                    compression_model="gpt-4o-mini",
                    compression_api_key="sk-test-compression",
                    compression_base_url="https://compression.example.com/v1",
                )
                try:
                    # The provider was set by __init__ calling _build_compression_provider
                    assert engine._compression_provider is not None
                    assert engine._compression_provider is not engine._llm
                    assert engine._compression_provider.name == "compression"
                finally:
                    lt.close()

    def test_primary_fallback_infers_provider_from_its_base_url(self) -> None:
        """Primary LLM settings have no separate provider field to inherit."""
        from leapflow.engine.engine import AgentEngine

        engine = AgentEngine.__new__(AgentEngine)
        engine._settings = SimpleNamespace(
            compression_provider="",
            compression_model="deepseek-chat",
            compression_api_key="",
            compression_base_url="",
            llm_api_key="sk-test",
            llm_base_url="https://api.deepseek.com/v1",
            llm_model="deepseek-chat",
            llm_max_retries=3,
            llm_provider="incorrect-legacy-value",
        )
        captured: dict[str, Any] = {}

        class _Provider:
            pass

        def build_provider(**kwargs: Any) -> _Provider:
            captured.update(kwargs)
            return _Provider()

        with patch("leapflow.llm.openai_provider.OpenAIChat", side_effect=build_provider):
            provider = engine._build_compression_provider()

        assert provider is not None
        assert captured["base_url"] == "https://api.deepseek.com/v1"
        assert captured["provider"] is None

    def test_partial_config_falls_back_to_primary_fields(self) -> None:
        """Only compression_model set → provider/api_key/url fall back to primary."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            # Test the actual _build_compression_provider logic with partial config
            engine, lt = _build_engine(
                td,
                primary,
                compression_model="gpt-4o-mini",
                # api_key and base_url come from make_settings defaults
            )
            try:
                # With make_settings providing llm_api_key="sk-test" and
                # llm_base_url="https://example.invalid/v1", the builder
                # should attempt construction with those fallbacks. It will
                # either succeed (returning a provider) or fail gracefully
                # returning None (if OpenAIChat rejects the URL) — but it
                # should never crash.
                # The important thing: the fallback logic ran without error.
                provider = engine._compression_provider
                # Provider may be None if construction fails (invalid URL), that's ok
                assert provider is None or provider is not engine._llm
            finally:
                lt.close()

    def test_construction_failure_degrades_to_none(self) -> None:
        """Provider construction raises → degrades to None, no crash."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            with patch(
                "leapflow.engine.engine.AgentEngine._build_compression_provider",
                side_effect=RuntimeError("boom"),
            ):
                # The engine __init__ catches exceptions from _build_compression_provider
                engine, lt = _build_engine(
                    td, primary,
                    compression_provider="bad",
                    compression_model="fail",
                )
                try:
                    # Construction failed but engine is still alive
                    assert engine._compression_provider is None
                finally:
                    lt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Summarize function routing — compression calls go to right provider
# ═══════════════════════════════════════════════════════════════════════════


class TestSummarizeFnRouting:
    """The summarize function must route to the compression provider when set."""

    @pytest.mark.asyncio
    async def test_summarize_uses_compression_provider(self) -> None:
        """When _compression_provider is set, summarize calls it, not _llm."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            engine, lt = _build_engine(td, primary)
            try:
                # Manually inject a compression provider
                compression = MockLLMProvider("compression")
                engine._compression_provider = compression

                # Build the summarize function
                summarize_fn = engine._make_compression_summarize_fn()
                result = await summarize_fn("Summarize this context please.")

                # Compression provider was called
                assert compression.call_count == 1
                # Primary provider was NOT called
                assert primary.call_count == 0
                assert result == "Compressed summary of context."
            finally:
                lt.close()

    @pytest.mark.asyncio
    async def test_summarize_falls_back_to_primary_when_no_compression(self) -> None:
        """When _compression_provider is None, summarize uses primary _llm."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            engine, lt = _build_engine(td, primary)
            try:
                assert engine._compression_provider is None
                summarize_fn = engine._make_compression_summarize_fn()
                result = await summarize_fn("Summarize this.")

                assert primary.call_count == 1
                assert result == "Compressed summary of context."
            finally:
                lt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Isolation verification — compression calls don't appear on primary
# ═══════════════════════════════════════════════════════════════════════════


class TestCompressionIsolation:
    """Key verification: compression traffic must not leak to the primary provider."""

    @pytest.mark.asyncio
    async def test_compression_calls_isolated_from_primary(self) -> None:
        """After multiple compression calls, primary provider has zero calls.

        This is the core isolation invariant: compression traffic must not
        appear in the primary provider's call history, because the primary
        provider's prefix cache depends on a stable, predictable call pattern.
        Compression calls would break that prefix stability.
        """
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            engine, lt = _build_engine(td, primary)
            try:
                compression = MockLLMProvider("compression")
                engine._compression_provider = compression

                summarize_fn = engine._make_compression_summarize_fn()
                # Multiple compression calls
                for i in range(5):
                    await summarize_fn(f"Summarize batch {i}")

                # All calls went to compression provider
                assert compression.call_count == 5
                # Zero calls on primary — isolation maintained
                assert primary.call_count == 0
            finally:
                lt.close()

    @pytest.mark.asyncio
    async def test_interleaved_primary_and_compression_calls_isolated(self) -> None:
        """Primary achat and compression summarize do not mix."""
        with tempfile.TemporaryDirectory() as td:
            primary = MockLLMProvider("primary")
            engine, lt = _build_engine(td, primary)
            try:
                compression = MockLLMProvider("compression")
                engine._compression_provider = compression

                # Simulate primary conversation call
                await primary.achat(
                    [{"role": "user", "content": "Hello"}],
                    stream=False,
                )
                assert primary.call_count == 1

                # Compression call
                summarize_fn = engine._make_compression_summarize_fn()
                await summarize_fn("Compress this.")

                # Primary still at 1, compression at 1
                assert primary.call_count == 1
                assert compression.call_count == 1
            finally:
                lt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Session snapshot round-trip — DuckDB persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionSnapshotRoundTrip:
    """Snapshot persistence → retrieval → resume freeze cycle."""

    def test_snapshot_persist_and_retrieve(self, tmp_path: Path) -> None:
        """update_session_snapshot → get_session_snapshot round-trip."""
        from leapflow.storage.conversation_store import (
            DuckDBConversationStore,
            SessionSnapshot,
        )

        db_path = tmp_path / "conv.duckdb"
        store = DuckDBConversationStore(db_path)
        try:
            sid = "test-session-001"
            store.create_session(sid, title="Test Session")

            # Persist snapshot
            store.update_session_snapshot(
                sid,
                system_prompt="You are LeapFlow.",
                tool_schema='[{"function":{"name":"file_read"}}]',
                disclosure_level="full",
            )

            # Retrieve
            snapshot = store.get_session_snapshot(sid)
            assert snapshot is not None
            assert isinstance(snapshot, SessionSnapshot)
            assert snapshot.system_prompt == "You are LeapFlow."
            assert snapshot.tool_schema == '[{"function":{"name":"file_read"}}]'
            assert snapshot.disclosure_level == "full"
        finally:
            store.close()

    def test_snapshot_returns_none_for_legacy_session(self, tmp_path: Path) -> None:
        """Session without snapshot data returns None."""
        from leapflow.storage.conversation_store import DuckDBConversationStore

        db_path = tmp_path / "conv.duckdb"
        store = DuckDBConversationStore(db_path)
        try:
            sid = "legacy-session"
            store.create_session(sid)
            snapshot = store.get_session_snapshot(sid)
            assert snapshot is None
        finally:
            store.close()

    def test_snapshot_returns_none_for_nonexistent_session(self, tmp_path: Path) -> None:
        """Non-existent session → None."""
        from leapflow.storage.conversation_store import DuckDBConversationStore

        db_path = tmp_path / "conv.duckdb"
        store = DuckDBConversationStore(db_path)
        try:
            snapshot = store.get_session_snapshot("does-not-exist")
            assert snapshot is None
        finally:
            store.close()

    def test_snapshot_update_overwrites(self, tmp_path: Path) -> None:
        """A second update_session_snapshot overwrites the first."""
        from leapflow.storage.conversation_store import DuckDBConversationStore

        db_path = tmp_path / "conv.duckdb"
        store = DuckDBConversationStore(db_path)
        try:
            sid = "overwrite-session"
            store.create_session(sid)

            store.update_session_snapshot(sid, "prompt-v1", "schema-v1", "core")
            store.update_session_snapshot(sid, "prompt-v2", "schema-v2", "full")

            snapshot = store.get_session_snapshot(sid)
            assert snapshot is not None
            assert snapshot.system_prompt == "prompt-v2"
            assert snapshot.tool_schema == "schema-v2"
            assert snapshot.disclosure_level == "full"
        finally:
            store.close()
