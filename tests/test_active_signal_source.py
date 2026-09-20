# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for ActiveSignalSource protocol, ActiveSourceManager, and FileWatchSignalSource.

Verifies lifecycle management, signal flow, failure isolation, backpressure,
channel gating, and PerceptionSession integration for active signal sources.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from leapflow.perception.active_signal_source import (
    ActiveSignalSource,
    ActiveSourceManager,
    EmitCallback,
)
from leapflow.perception.signals import SignalBuffer
from leapflow.perception.types import InteractionSignal


# ═══════════════════════════════════════════════════════════════════
# Test Infrastructure
# ═══════════════════════════════════════════════════════════════════


class FakePipeline:
    """Records fuse() calls for verification."""

    def __init__(self, fuse_delay: float = 0.0) -> None:
        self.fuse_calls: List[Any] = []
        self.fuse_delay = fuse_delay
        self._in_fuse = False
        self.overlaps_detected = 0

    def fuse(self, signals: Any, graph: Any) -> None:
        if self._in_fuse:
            self.overlaps_detected += 1
        self._in_fuse = True
        try:
            self.fuse_calls.append(list(signals))
            if self.fuse_delay:
                time.sleep(self.fuse_delay)
        finally:
            self._in_fuse = False


class FakeGraph:
    pass


class RecordingSource:
    """Test source that emits a scripted number of signals then completes."""

    def __init__(
        self,
        source_id: str,
        channel_id: str,
        signals_to_emit: int = 0,
        *,
        start_delay: float = 0.0,
    ) -> None:
        self._source_id = source_id
        self._channel_id = channel_id
        self._signals_to_emit = signals_to_emit
        self._start_delay = start_delay
        self.start_called = False
        self.stop_called = False

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def channel_id(self) -> str:
        return self._channel_id

    async def start(self, emit: EmitCallback) -> None:
        self.start_called = True
        if self._start_delay:
            await asyncio.sleep(self._start_delay)
        for i in range(self._signals_to_emit):
            emit(InteractionSignal(timestamp=float(i), signal_type=self._channel_id))

    async def stop(self) -> None:
        self.stop_called = True


class FailingStartSource:
    """Source that raises in start()."""

    def __init__(self, source_id: str, channel_id: str = "fail") -> None:
        self._source_id = source_id
        self._channel_id = channel_id
        self.stop_called = False

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def channel_id(self) -> str:
        return self._channel_id

    async def start(self, emit: EmitCallback) -> None:
        raise RuntimeError(f"Source {self._source_id} start failed")

    async def stop(self) -> None:
        self.stop_called = True


class FailingStopSource:
    """Source that raises in stop()."""

    def __init__(self, source_id: str, channel_id: str = "fail") -> None:
        self._source_id = source_id
        self._channel_id = channel_id
        self.start_called = False

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def channel_id(self) -> str:
        return self._channel_id

    async def start(self, emit: EmitCallback) -> None:
        self.start_called = True

    async def stop(self) -> None:
        raise RuntimeError(f"Source {self._source_id} stop failed")


class HangingStopSource:
    """Source whose stop() hangs indefinitely."""

    def __init__(self, source_id: str, channel_id: str = "hang") -> None:
        self._source_id = source_id
        self._channel_id = channel_id
        self.start_called = False

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def channel_id(self) -> str:
        return self._channel_id

    async def start(self, emit: EmitCallback) -> None:
        self.start_called = True

    async def stop(self) -> None:
        await asyncio.sleep(60)  # Hang forever


class NotASource:
    """Object that does NOT satisfy ActiveSignalSource protocol."""

    def hello(self) -> str:
        return "I am not a source"


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def buffer() -> SignalBuffer:
    return SignalBuffer()


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def manager(buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph) -> ActiveSourceManager:
    return ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)


# ═══════════════════════════════════════════════════════════════════
# TestActiveSourceManagerLifecycle
# ═══════════════════════════════════════════════════════════════════


class TestActiveSourceManagerLifecycle:
    """Lifecycle: register, start_all, dispose semantics."""

    async def test_register_source_before_start(self, manager: ActiveSourceManager) -> None:
        """Register succeeds before start_all."""
        source = RecordingSource("s1", "ch1")
        manager.register(source)
        assert manager.source_count == 1

    async def test_register_after_start_raises(self, manager: ActiveSourceManager) -> None:
        """RuntimeError when registering after start_all()."""
        await manager.start_all()
        try:
            with pytest.raises(RuntimeError, match="Cannot register source after start_all"):
                manager.register(RecordingSource("late", "ch"))
        finally:
            await manager.dispose()

    async def test_register_duplicate_source_id_raises(self, manager: ActiveSourceManager) -> None:
        """ValueError on duplicate source_id."""
        source = RecordingSource("dup", "ch1")
        manager.register(source)
        with pytest.raises(ValueError, match="Duplicate source_id"):
            manager.register(RecordingSource("dup", "ch2"))

    async def test_register_non_protocol_raises(self, manager: ActiveSourceManager) -> None:
        """TypeError when arg doesn't satisfy ActiveSignalSource."""
        with pytest.raises(TypeError, match="Not an ActiveSignalSource"):
            manager.register(NotASource())  # type: ignore[arg-type]

    async def test_start_all_spawns_source_tasks(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """After start_all, source tasks exist."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        mgr.register(RecordingSource("a", "ch1"))
        mgr.register(RecordingSource("b", "ch2"))
        await mgr.start_all()
        try:
            assert len(mgr._source_tasks) == 2
            assert mgr._consumer_task is not None
        finally:
            await mgr.dispose()

    async def test_start_all_idempotent(self, manager: ActiveSourceManager) -> None:
        """Calling start_all twice is safe (no-op on second call)."""
        source = RecordingSource("s1", "ch1")
        manager.register(source)
        await manager.start_all()
        # Second call should be no-op
        await manager.start_all()
        try:
            assert len(manager._source_tasks) == 1
        finally:
            await manager.dispose()

    async def test_dispose_cancels_all_tasks(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """After dispose, all source + consumer tasks are done."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        # Use a source that holds start() open (simulating a long-running source)
        mgr.register(RecordingSource("long", "ch", start_delay=10.0))
        await mgr.start_all()
        consumer_task = mgr._consumer_task
        source_tasks = list(mgr._source_tasks.values())

        await mgr.dispose()

        assert consumer_task is not None and consumer_task.done()
        for task in source_tasks:
            assert task.done()

    async def test_dispose_calls_source_stop(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """Each source.stop() is invoked during dispose."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        s1 = RecordingSource("s1", "ch1")
        s2 = RecordingSource("s2", "ch2")
        mgr.register(s1)
        mgr.register(s2)
        await mgr.start_all()
        await asyncio.sleep(0.05)  # Let sources finish start()
        await mgr.dispose()
        assert s1.stop_called
        assert s2.stop_called

    async def test_dispose_idempotent(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """Second dispose is a no-op."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        mgr.register(RecordingSource("x", "ch"))
        await mgr.start_all()
        await mgr.dispose()
        # Second dispose should not raise
        await mgr.dispose()


# ═══════════════════════════════════════════════════════════════════
# TestSignalFlow
# ═══════════════════════════════════════════════════════════════════


class TestSignalFlow:
    """Signal emission flows from source through queue to downstream sinks."""

    async def test_emit_flows_to_signal_buffer(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """Signals reach SignalBuffer.record."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        mgr.register(RecordingSource("emitter", "ch", signals_to_emit=3))
        await mgr.start_all()
        await asyncio.sleep(0.2)  # Let consumer drain
        await mgr.dispose()

        signals = buffer.drain()
        assert len(signals) == 3
        assert all(s.signal_type == "ch" for s in signals)

    async def test_emit_flows_to_causal_pipeline(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """fuse() is called with the signal."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        mgr.register(RecordingSource("emitter", "ch", signals_to_emit=2))
        await mgr.start_all()
        await asyncio.sleep(0.2)
        await mgr.dispose()

        assert len(pipeline.fuse_calls) == 2
        # Each call has exactly one signal
        for call in pipeline.fuse_calls:
            assert len(call) == 1
            assert call[0].signal_type == "ch"

    async def test_consumer_serializes_fuse_calls(
        self, buffer: SignalBuffer, graph: FakeGraph
    ) -> None:
        """Two sources emit simultaneously → fuse called serially (no overlap)."""
        slow_pipeline = FakePipeline(fuse_delay=0.05)
        mgr = ActiveSourceManager(buffer, slow_pipeline, graph, queue_capacity=64)
        # Two sources each emitting 3 signals
        mgr.register(RecordingSource("a", "ch_a", signals_to_emit=3))
        mgr.register(RecordingSource("b", "ch_b", signals_to_emit=3))
        await mgr.start_all()
        await asyncio.sleep(0.5)  # Give consumer time to process all with delays
        await mgr.dispose()

        assert len(slow_pipeline.fuse_calls) == 6
        assert slow_pipeline.overlaps_detected == 0


# ═══════════════════════════════════════════════════════════════════
# TestFailureIsolation
# ═══════════════════════════════════════════════════════════════════


class TestFailureIsolation:
    """Source failures are isolated — no cascading effects."""

    async def test_source_start_exception_isolated(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """One source raising in start() doesn't stop others."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        good_source = RecordingSource("good", "ch", signals_to_emit=2)
        bad_source = FailingStartSource("bad")
        mgr.register(bad_source)
        mgr.register(good_source)
        await mgr.start_all()
        await asyncio.sleep(0.2)
        await mgr.dispose()

        assert good_source.start_called
        # Good source's signals still flow
        signals = buffer.drain()
        assert len(signals) == 2

    async def test_source_stop_exception_isolated(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """One source raising in stop() doesn't prevent others' cleanup."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        bad = FailingStopSource("bad_stop")
        good = RecordingSource("good", "ch")
        mgr.register(bad)
        mgr.register(good)
        await mgr.start_all()
        await asyncio.sleep(0.05)
        # dispose should not raise despite bad source's stop() error
        await mgr.dispose()
        assert good.stop_called

    async def test_source_stop_timeout_isolated(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """A hung source.stop() is killed by timeout without blocking teardown."""
        mgr = ActiveSourceManager(
            buffer, pipeline, graph,
            queue_capacity=64,
            shutdown_timeout_s=0.2,
        )
        hanging = HangingStopSource("hang")
        good = RecordingSource("good", "ch")
        mgr.register(hanging)
        mgr.register(good)
        await mgr.start_all()
        await asyncio.sleep(0.05)

        start = time.monotonic()
        await mgr.dispose()
        elapsed = time.monotonic() - start

        # Should complete well under 2s (the hanging source's 60s sleep)
        assert elapsed < 2.0
        assert good.stop_called

    async def test_consumer_continues_after_fuse_error(
        self, buffer: SignalBuffer, graph: FakeGraph
    ) -> None:
        """When fuse() raises, consumer keeps draining subsequent signals."""

        class FailingPipeline:
            def __init__(self) -> None:
                self.call_count = 0

            def fuse(self, signals: Any, graph: Any) -> None:
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("fuse error")

        failing_pipe = FailingPipeline()
        mgr = ActiveSourceManager(buffer, failing_pipe, graph, queue_capacity=64)
        mgr.register(RecordingSource("src", "ch", signals_to_emit=3))
        await mgr.start_all()
        await asyncio.sleep(0.3)
        await mgr.dispose()

        # Consumer should have processed all 3 despite first fuse() raising
        assert failing_pipe.call_count == 3
        signals = buffer.drain()
        assert len(signals) == 3


# ═══════════════════════════════════════════════════════════════════
# TestBackpressure
# ═══════════════════════════════════════════════════════════════════


class TestBackpressure:
    """Queue backpressure and signal dropping."""

    async def test_queue_full_drops_signal(
        self, buffer: SignalBuffer, graph: FakeGraph
    ) -> None:
        """Emit with capacity=2: 3rd signal is dropped."""
        slow_pipeline = FakePipeline(fuse_delay=0.1)
        mgr = ActiveSourceManager(buffer, slow_pipeline, graph, queue_capacity=2)
        # Source that emits 5 signals synchronously
        mgr.register(RecordingSource("fast", "ch", signals_to_emit=5))
        await mgr.start_all()
        await asyncio.sleep(0.8)  # Let consumer drain what it can
        await mgr.dispose()

        # Some signals should have been dropped
        assert mgr.dropped_count > 0

    async def test_dropped_count_reflects_drops(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """dropped_count increments correctly."""
        # Queue capacity=1, emit 5 synchronously: consumer can't drain fast enough
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=1)
        mgr.register(RecordingSource("burst", "ch", signals_to_emit=5))
        await mgr.start_all()
        await asyncio.sleep(0.2)
        await mgr.dispose()

        # At least some drops happened (1 in queue + consumer may get 1 more)
        assert mgr.dropped_count >= 3


# ═══════════════════════════════════════════════════════════════════
# TestChannelGating
# ═══════════════════════════════════════════════════════════════════


class TestChannelGating:
    """Channel-based source filtering."""

    async def test_disabled_channel_source_not_started(
        self, buffer: SignalBuffer, pipeline: FakePipeline, graph: FakeGraph
    ) -> None:
        """Source with channel_id not in enabled_channels doesn't get a task."""
        mgr = ActiveSourceManager(buffer, pipeline, graph, queue_capacity=64)
        enabled_source = RecordingSource("enabled", "active_ch", signals_to_emit=1)
        disabled_source = RecordingSource("disabled", "inactive_ch", signals_to_emit=1)
        mgr.register(enabled_source)
        mgr.register(disabled_source)
        await mgr.start_all(enabled_channels=frozenset({"active_ch"}))
        await asyncio.sleep(0.1)
        await mgr.dispose()

        assert enabled_source.start_called
        assert not disabled_source.start_called
        # Only enabled source task created
        assert "enabled" in mgr._source_tasks
        assert "disabled" not in mgr._source_tasks


# ═══════════════════════════════════════════════════════════════════
# TestFileWatchSignalSource
# ═══════════════════════════════════════════════════════════════════


class TestFileWatchSignalSource:
    """FileWatchSignalSource — watchdog-based filesystem monitoring."""

    def test_filewatch_source_id_and_channel_id(self, tmp_path: Any) -> None:
        """Protocol conformance: source_id and channel_id."""
        from leapflow.perception.active_sources_builtin import FileWatchSignalSource

        source = FileWatchSignalSource([tmp_path], source_id="fw1")
        assert source.source_id == "fw1"
        assert source.channel_id == "file_watch"

    def test_filewatch_protocol_check(self, tmp_path: Any) -> None:
        """isinstance(source, ActiveSignalSource) is True."""
        from leapflow.perception.active_sources_builtin import FileWatchSignalSource

        source = FileWatchSignalSource([tmp_path])
        assert isinstance(source, ActiveSignalSource)

    async def test_filewatch_emits_on_file_create(self, tmp_path: Any) -> None:
        """Create file → signal emitted with signal_type='file_change' and detail containing path."""
        from leapflow.perception.active_sources_builtin import FileWatchSignalSource, _WATCHDOG_AVAILABLE

        if not _WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed")

        source = FileWatchSignalSource([tmp_path], source_id="fw_test")
        emitted: List[InteractionSignal] = []

        def capture(signal: InteractionSignal) -> None:
            emitted.append(signal)

        await source.start(capture)
        await asyncio.sleep(0.3)  # Let observer settle

        # Create a file
        test_file = tmp_path / "test_create.txt"
        test_file.write_text("hello")
        await asyncio.sleep(0.5)  # Wait for watchdog to fire

        await source.stop()

        # At least one signal should have been emitted
        assert len(emitted) > 0
        assert any(s.signal_type == "file_change" for s in emitted)
        # At least one signal should reference the created file
        assert any("test_create" in s.detail for s in emitted)

    async def test_filewatch_stop_terminates_observer(self, tmp_path: Any) -> None:
        """After stop, subsequent file changes produce no signals."""
        from leapflow.perception.active_sources_builtin import FileWatchSignalSource, _WATCHDOG_AVAILABLE

        if not _WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed")

        source = FileWatchSignalSource([tmp_path], source_id="fw_stop")
        emitted: List[InteractionSignal] = []

        def capture(signal: InteractionSignal) -> None:
            emitted.append(signal)

        await source.start(capture)
        await asyncio.sleep(0.2)
        await source.stop()

        count_before = len(emitted)
        # Create file after stop
        (tmp_path / "after_stop.txt").write_text("should not trigger")
        await asyncio.sleep(0.5)

        # No new signals after stop
        assert len(emitted) == count_before


# ═══════════════════════════════════════════════════════════════════
# TestPerceptionSessionIntegration
# ═══════════════════════════════════════════════════════════════════


class TestPerceptionSessionIntegration:
    """PerceptionSession start/stop integration with ActiveSourceManager."""

    def _make_session(self, active_source_manager=None):
        """Build a minimal PerceptionSession for testing active source hooks."""
        from leapflow.perception.config import PerceptionConfig
        from leapflow.perception.session import PerceptionSession

        config = PerceptionConfig(
            signal_channels=frozenset({"click", "file_watch"}),
        )
        rpc = MagicMock()
        session = PerceptionSession(
            config=config,
            rpc=rpc,
            active_source_manager=active_source_manager,
        )
        return session

    async def test_session_starts_active_sources_on_start(self) -> None:
        """Session.start() triggers manager.start_all()."""
        mock_manager = AsyncMock()
        mock_manager.start_all = AsyncMock()
        mock_manager.dispose = AsyncMock()
        session = self._make_session(active_source_manager=mock_manager)

        await session.start("test-session-1")
        mock_manager.start_all.assert_called_once()
        # Verify enabled_channels passed
        call_kwargs = mock_manager.start_all.call_args[1]
        assert "enabled_channels" in call_kwargs

        await session.stop()

    async def test_session_stops_active_sources_on_stop(self) -> None:
        """Session.stop() triggers manager.dispose()."""
        mock_manager = AsyncMock()
        mock_manager.start_all = AsyncMock()
        mock_manager.dispose = AsyncMock()
        session = self._make_session(active_source_manager=mock_manager)

        await session.start("test-session-2")
        await session.stop()
        mock_manager.dispose.assert_called_once()

    async def test_session_without_manager_unchanged(self) -> None:
        """When active_source_manager=None, start/stop behave exactly as before."""
        session = self._make_session(active_source_manager=None)
        # Should not raise
        await session.start("test-session-3")
        assert session.active
        await session.stop()
        assert not session.active

    async def test_session_teardown_respects_shutdown_timeout(
        self,
    ) -> None:
        """PerceptionSession.stop() completes bounded by shutdown_timeout_s even with hung sources."""
        from leapflow.perception.config import PerceptionConfig
        from leapflow.perception.session import PerceptionSession
        from leapflow.domain.trajectory import RecordingMode
        import time as _time

        class HangingStopSourceLocal:
            source_id = "hang"
            channel_id = "hang"

            async def start(self, emit: EmitCallback) -> None:
                pass

            async def stop(self) -> None:
                # Simulate a source whose stop hangs
                await asyncio.sleep(30)

        buffer = SignalBuffer()
        pipeline = FakePipeline()
        graph = FakeGraph()

        manager = ActiveSourceManager(
            buffer, pipeline, graph,
            queue_capacity=64,
            shutdown_timeout_s=0.2,  # tight timeout for test
        )
        manager.register(HangingStopSourceLocal())

        config = PerceptionConfig(signal_channels=frozenset({"hang"}))
        rpc = MagicMock()
        session = PerceptionSession(
            config=config,
            rpc=rpc,
            active_source_manager=manager,
        )
        session._recording_mode = RecordingMode.VISION_ONLY

        await session.start("teardown-test")
        await asyncio.sleep(0.05)

        start = _time.monotonic()
        await session.stop()
        elapsed = _time.monotonic() - start

        # shutdown_timeout_s (0.2) + drain timeout (0.2) + task cancels (~2s max)
        # In practice should be around 0.4-2.5s, well under 5s
        assert elapsed < 5.0, f"Session teardown took too long: {elapsed:.2f}s"
