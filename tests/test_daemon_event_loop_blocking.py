"""Regression tests for the daemon event-loop permanent-blocking fix.

Root cause chain (observed as a 43-minute daemon freeze where every RPC
timed out):

1. ``SemanticMemoryProvider._init_schema()`` created an index on
   ``session_id`` *before* the ALTER TABLE migration that adds the column,
   so legacy databases failed deferred init with a Binder Error.
2. The failure made ``engine_chat`` re-enter ``_ensure_deferred()`` which
   waited *unboundedly* on ``_deferred_lock``.
3. ``initialize_deferred()`` ran heavy synchronous DuckDB operations
   directly on the event-loop thread, freezing heartbeats and all RPCs.

Validates:
- legacy-schema DuckDB files migrate cleanly and end up with the session
  index in place
- ``engine_chat`` degrades to critical-only mode after a bounded wait
  instead of hanging until deferred init completes
- a foreground timeout on ``_ensure_deferred()`` never cancels the
  background runner task (shielded wait), and the init is not re-executed
- deferred DB work runs on the dedicated executor so concurrent tasks keep
  running with sub-100ms scheduling gaps
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import asyncio
import time
from pathlib import Path
from typing import Any, List, Optional

import duckdb
import pytest

from leapflow.cli.context import Context
from leapflow.daemon.service import RuntimeLeapService
from leapflow.memory.providers.semantic import SemanticMemoryProvider


# ═══════════════════════════════════════════════════════════════════════════
# 1. Semantic schema migration order (defect trigger)
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticSchemaMigrationOrder:
    """Indexes on migrated columns must be created AFTER the migrations."""

    @staticmethod
    def _create_legacy_db(db_path: Path) -> None:
        """Build a pre-migration leap_memory table (no domain/path/session_id)."""
        con = duckdb.connect(str(db_path))
        con.execute(
            """
            CREATE TABLE leap_memory (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at DOUBLE NOT NULL,
                accessed_at DOUBLE NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        con.execute(
            "INSERT INTO leap_memory (id, kind, content, created_at, accessed_at) "
            "VALUES ('m1', 'note', 'legacy row', 1.0, 1.0)"
        )
        con.close()

    @pytest.mark.asyncio
    async def test_initialize_on_legacy_schema_succeeds(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.duckdb"
        self._create_legacy_db(db_path)

        provider = SemanticMemoryProvider(source=db_path)
        # Before the fix this raised a Binder Error: CREATE INDEX on
        # session_id ran before ALTER TABLE ADD COLUMN session_id.
        await provider.initialize()
        try:
            con = provider._connection()
            cols = {
                row[0]
                for row in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'leap_memory'"
                ).fetchall()
            }
            assert {"domain", "path", "session_id"} <= cols

            indexes = {
                row[0]
                for row in con.execute(
                    "SELECT index_name FROM duckdb_indexes() "
                    "WHERE table_name = 'leap_memory'"
                ).fetchall()
            }
            assert "idx_lm_session" in indexes
            assert "idx_lm_domain" in indexes

            # Legacy data survives the migration
            count = con.execute("SELECT COUNT(*) FROM leap_memory").fetchone()[0]
            assert count == 1
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_on_fresh_db_still_works(self, tmp_path: Path) -> None:
        provider = SemanticMemoryProvider(source=tmp_path / "fresh.duckdb")
        await provider.initialize()
        try:
            indexes = {
                row[0]
                for row in provider._connection().execute(
                    "SELECT index_name FROM duckdb_indexes() "
                    "WHERE table_name = 'leap_memory'"
                ).fetchall()
            }
            assert "idx_lm_session" in indexes
        finally:
            await provider.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# 2. engine_chat bounded wait + critical-only degradation
# ═══════════════════════════════════════════════════════════════════════════


class _NeverReadyContext:
    """Context stand-in whose deferred init never completes."""

    def __init__(self) -> None:
        self._deferred_initialized = False
        self.ensure_calls = 0

    async def _ensure_deferred(self) -> None:
        self.ensure_calls += 1
        await asyncio.Event().wait()  # blocks forever until cancelled


class TestEngineChatDegradesOnDeferredTimeout:
    """engine_chat must not hang when deferred init never finishes."""

    @pytest.mark.asyncio
    async def test_degrades_to_critical_only_after_timeout(self) -> None:
        service = RuntimeLeapService.__new__(RuntimeLeapService)
        fake_ctx = _NeverReadyContext()
        service._ctx = fake_ctx
        service._DEFERRED_WAIT_TIMEOUT_S = 0.1  # type: ignore[misc]
        service._turn_admission = type(
            "_Admission", (), {"locked": staticmethod(lambda: False)}
        )()

        stream = service.engine_chat("hello", request_id="req-degrade-1")
        first_chunk: Any = await stream.__anext__()
        assert first_chunk.event_type == "status"
        assert "warming up" in first_chunk.content.lower()

        # The next step must complete within the bounded wait (plus margin):
        # a hang here means the degrade path was not taken.
        try:
            await asyncio.wait_for(stream.__anext__(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "engine_chat hung on _ensure_deferred(); expected bounded "
                "wait followed by critical-only degradation"
            )
        except (StopAsyncIteration, Exception):
            # Reaching the deeper pipeline (which fails on the bare fake) is
            # fine — the point is we got PAST the deferred wait quickly.
            pass
        finally:
            await stream.aclose()

        assert fake_ctx.ensure_calls == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. _ensure_deferred: shielded wait, cancel-safe, no duplicate init
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsureDeferredCancelSafety:
    """A foreground timeout must not cancel or re-run the background init."""

    @staticmethod
    def _bare_context() -> Context:
        ctx = Context.__new__(Context)
        ctx._deferred_initialized = False
        ctx._deferred_lock = asyncio.Lock()
        ctx._deferred_attempts = 0
        ctx._deferred_task = None
        return ctx

    @pytest.mark.asyncio
    async def test_timeout_does_not_cancel_background_runner(self) -> None:
        ctx = self._bare_context()
        release = asyncio.Event()
        init_calls = 0

        async def _slow_init() -> None:
            nonlocal init_calls
            init_calls += 1
            await release.wait()

        ctx.initialize_deferred = _slow_init  # type: ignore[method-assign]

        # Background waiter (daemon's _run_deferred_init analog)
        background = asyncio.create_task(ctx._ensure_deferred())
        await asyncio.sleep(0.01)  # let the runner task start
        runner: Optional[asyncio.Task[None]] = ctx._deferred_task
        assert runner is not None and not runner.done()

        # Foreground bounded wait times out (engine_chat analog)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ctx._ensure_deferred(), timeout=0.05)

        # The runner must survive the foreground timeout untouched
        assert not runner.cancelled()
        assert not runner.done()
        assert init_calls == 1  # no duplicate initialization

        # Completing the init unblocks the background waiter
        release.set()
        await asyncio.wait_for(background, timeout=1.0)
        assert ctx._deferred_initialized is True
        assert init_calls == 1

        # Subsequent calls are cheap no-ops
        await ctx._ensure_deferred()
        assert init_calls == 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. Deferred DB work runs off the event loop (dedicated executor)
# ═══════════════════════════════════════════════════════════════════════════


class TestDeferredDbExecutor:
    """Blocking DuckDB work must not freeze the event loop.

    Asserted behaviorally below: scanning ``initialize_deferred``'s source for
    ``_run_deferred_db`` occurrences would pin the current call-site layout
    rather than the contract — that a blocking DB call keeps the loop
    responsive and that ``status()`` still answers while the DB worker is busy.
    """

    @pytest.mark.asyncio
    async def test_event_loop_stays_responsive_during_blocking_db_op(self) -> None:
        ctx = Context.__new__(Context)
        ctx._deferred_db_executor = None

        ticks: List[float] = []
        stop = asyncio.Event()

        async def _ticker() -> None:
            while not stop.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(_ticker())
        try:
            start = time.monotonic()
            result = await ctx._run_deferred_db(lambda: (time.sleep(0.3), 42)[1])
            elapsed = time.monotonic() - start
            assert result == 42
            # Windows' monotonic clock quantizes to ~15.6ms, so the measured
            # interval can read a few ms short of the blocking sleep itself.
            assert elapsed >= 0.3 - 0.03
        finally:
            stop.set()
            await ticker_task
            executor = ctx._deferred_db_executor
            if executor is not None:
                executor.shutdown(wait=True)

        # The loop must have kept scheduling the ticker while the blocking
        # sleep ran in the executor thread: gaps stay well under 100ms.
        in_window = [t for t in ticks if start <= t <= start + 0.3]
        assert len(in_window) >= 3, (
            f"event loop starved during blocking DB op: only {len(in_window)} "
            "ticker iterations observed in the 300ms window"
        )
        gaps = [b - a for a, b in zip(in_window, in_window[1:])]
        assert gaps and max(gaps) < 0.1, (
            f"event loop stalled for {max(gaps) if gaps else 0:.3f}s during "
            "the blocking DB operation"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Daemon control-plane responsiveness during deferred init
# ═══════════════════════════════════════════════════════════════════════════


class TestDaemonControlPlaneResponsiveness:
    """Status/recovery RPCs must get scheduling priority over background init."""

    @pytest.mark.asyncio
    async def test_service_deferred_runner_yields_before_initialization(self) -> None:
        order: list[str] = []

        class FakeContext:
            async def _ensure_deferred(self) -> None:
                order.append("deferred")

        async def _peer_control_plane_task() -> None:
            order.append("peer")

        service = RuntimeLeapService.__new__(RuntimeLeapService)
        deferred = asyncio.create_task(service._run_deferred_init(FakeContext()))
        peer = asyncio.create_task(_peer_control_plane_task())

        await asyncio.gather(deferred, peer)

        assert order == ["peer", "deferred"]

    @pytest.mark.asyncio
    async def test_status_returns_while_deferred_db_worker_is_busy(self, tmp_path: Path) -> None:
        from conftest import make_settings
        from types import SimpleNamespace

        settings = make_settings(str(tmp_path))
        service = RuntimeLeapService(settings, mock_host=True)
        ctx = Context.__new__(Context)
        ctx.settings = settings
        ctx.engine = None
        ctx._db_holder = SimpleNamespace(db_path=settings.duckdb_path)
        ctx._deferred_initialized = False
        ctx._deferred_lock = asyncio.Lock()
        ctx._deferred_attempts = 0
        ctx._deferred_task = None
        ctx._deferred_db_executor = None

        async def _slow_init() -> None:
            await ctx._run_deferred_db(lambda: time.sleep(0.25))

        ctx.initialize_deferred = _slow_init  # type: ignore[method-assign]
        service._ctx = ctx
        service._deferred_init_task = asyncio.create_task(service._run_deferred_init(ctx))
        try:
            for _ in range(50):
                runner = getattr(ctx, "_deferred_task", None)
                if runner is not None and not runner.done():
                    break
                await asyncio.sleep(0.01)
            assert ctx._deferred_task is not None and not ctx._deferred_task.done()

            status = await asyncio.wait_for(service.status(), timeout=0.1)

            assert status["deferred_init"]["running"] is True
            assert status["deferred_init"]["initialized"] is False
        finally:
            await service._deferred_init_task
            executor = getattr(ctx, "_deferred_db_executor", None)
            if executor is not None:
                executor.shutdown(wait=True)


# ════════════════════════════════════════════════════════════════
# status() must not read the watch store on the loop
# ════════════════════════════════════════════════════════════════


async def test_the_watch_summary_does_not_block_the_loop() -> None:
    """status() is the most-polled RPC there is; it cannot hold the loop.

    The watch store is DuckDB, so reading it takes as long as the query takes and
    grows with the number of armed watches. Read inline, one status poll stalled every
    other RPC for that whole time -- the shape that makes a busy daemon look hung.

    Measured by driving the summary against a store whose read sleeps, and checking
    that an unrelated coroutine still gets scheduled while it is in flight.
    """
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    read_started = asyncio.Event()
    ticks = 0

    class _SlowStore:
        def list_watches(self) -> list[Any]:
            # Blocking on purpose: this is what DuckDB does, and it is the reason the
            # read must not happen on the loop thread.
            read_started.set()
            time.sleep(0.25)
            return []

    coordinator = MonitorCoordinator()
    coordinator._monitors = _SlowStore()
    coordinator._off_loop = _serialized_off_loop()

    async def _tick() -> None:
        nonlocal ticks
        await read_started.wait()
        for _ in range(5):
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(_tick())
    summary = await coordinator.get_summary()
    # Sampled the instant the read returns. Counting after awaiting the ticker
    # measures nothing: the task runs to completion either way, so the first
    # version of this assertion passed against the blocking implementation too.
    ticks_during_read = ticks
    await ticker

    assert summary["total"] == 0
    assert ticks_during_read > 0, (
        "no other coroutine ran while the watch store was being read, so the read "
        "happened on the loop thread; status() would stall every other RPC"
    )


async def test_the_summary_still_answers_without_an_off_loop_channel() -> None:
    """A missing channel must degrade to a slow answer, never to no answer."""
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    class _Store:
        def list_watches(self) -> list[Any]:
            return []

    coordinator = MonitorCoordinator()
    coordinator._monitors = _Store()
    coordinator._off_loop = None

    summary = await coordinator.get_summary()
    assert summary["total"] == 0


def _serialized_off_loop() -> Any:
    """The runtime's channel shape: a single worker, awaited by the caller."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-deferred-db")

    async def _run(fn: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(executor, fn)

    return _run
