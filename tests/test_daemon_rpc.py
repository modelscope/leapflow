from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from leapflow.daemon.client import DaemonClient, DaemonUnavailableError, ensure_daemon_client
from leapflow.daemon.lifecycle import DaemonInfo
from leapflow.daemon.protocol import StreamChunk
from leapflow.daemon.server import UnixRpcServer


class _FakeService:
    def __init__(self) -> None:
        self._client_count = lambda: 0

    def set_client_count_provider(self, provider) -> None:
        self._client_count = provider

    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            request_id="",
            content=f"hello {message}",
            event_type="chunk",
            metadata={"session_id": "sess-1"},
        )
        yield StreamChunk(request_id="", content="done", event_type="final")

    async def status(self) -> dict[str, Any]:
        return {"pid": 123, "active_clients": self._client_count(), "profile": "test"}


class _FailingStreamService(_FakeService):
    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(request_id="", content="partial", event_type="chunk")
        raise RuntimeError("stream exploded")


class _SlowFirstChunkService(_FakeService):
    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        await asyncio.sleep(0.08)
        yield StreamChunk(request_id="", content=f"slow {message}", event_type="chunk")
        yield StreamChunk(request_id="", content="done", event_type="final")


async def _start_server(run_dir: Path, service=None, *, stream_heartbeat_s: float | None = None):
    server = UnixRpcServer(
        service or _FakeService(),
        sock_path=run_dir / "leapd.sock",
        run_dir=run_dir,
        stream_heartbeat_s=stream_heartbeat_s,
    )
    task = asyncio.create_task(server.serve_forever())
    for _ in range(50):
        if (run_dir / "leapd.sock").exists():
            return server, task, run_dir
        await asyncio.sleep(0.02)
    task.cancel()
    raise AssertionError("server did not start")


@pytest.mark.asyncio
async def test_daemon_client_receives_stream_events() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(Path(root) / "run")
        client = DaemonClient(run_dir / "leapd.sock")

        try:
            events = [event async for event in client.engine_chat("world")]
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert [(event.type, event.content) for event in events] == [
        ("chunk", "hello world"),
        ("final", "done"),
    ]
    assert events[0].metadata == {"session_id": "sess-1"}


@pytest.mark.asyncio
async def test_daemon_client_stream_heartbeat_prevents_idle_timeout() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(
            Path(root) / "run",
            service=_SlowFirstChunkService(),
            stream_heartbeat_s=0.01,
        )
        client = DaemonClient(run_dir / "leapd.sock", timeout_s=0.03)

        try:
            events = [event async for event in client.engine_chat("world")]
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    heartbeat_events = [event for event in events if event.metadata == {"heartbeat": True}]
    assert heartbeat_events
    assert [(event.type, event.content) for event in events[-2:]] == [
        ("chunk", "slow world"),
        ("final", "done"),
    ]


@pytest.mark.asyncio
async def test_client_lease_blocks_daemon_idle_shutdown(tmp_path) -> None:
    from leapflow.daemon.lease import ClientLease, read_active_client_leases
    from leapflow.daemon.server import _watch_idle_shutdown

    class IdleServer:
        active_connections = 0

        def __init__(self, run_dir: Path) -> None:
            self._run_dir = run_dir

        @property
        def run_dir(self) -> Path:
            return self._run_dir

    run_dir = tmp_path / "run"
    lease = ClientLease(run_dir, kind="tui", session_id="sess-live", touch_interval_s=1.0)
    await lease.start()
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _watch_idle_shutdown(
            IdleServer(run_dir),
            stop_event,
            idle_timeout_s=0.03,
            lease_ttl_s=1.0,
            poll_interval_s=0.01,
        )
    )
    try:
        await asyncio.sleep(0.08)
        assert not stop_event.is_set()
        snapshots = read_active_client_leases(run_dir, ttl_s=1.0)
        assert [(item.kind, item.session_id, item.state) for item in snapshots] == [
            ("tui", "sess-live", "idle")
        ]

        await lease.stop()
        await asyncio.wait_for(stop_event.wait(), timeout=0.2)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_runtime_service_streams_pending_approval_and_resolves(tmp_path) -> None:
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService
    from leapflow.engine import StreamEvent
    from leapflow.security.actions import ActionDescriptor
    from leapflow.security.approval import ApprovalRequest

    service = RuntimeLeapService(make_settings(str(tmp_path)), mock_host=True)

    class ApprovalEngine:
        context_token_count = 0
        _current_session_id = "sess-approval"

        async def run_stream(self, message: str, *, enable_thinking: bool = False):
            request = ApprovalRequest(
                category="shell.command",
                detail="python << 'EOF'\nprint('ok')\nEOF",
                action=ActionDescriptor.shell("python << 'EOF'\nprint('ok')\nEOF"),
            )
            decision = await service._request_approval(request)
            yield StreamEvent(type="final", content=f"decision={decision}")

    class FakeContext:
        def __init__(self) -> None:
            self.engine = ApprovalEngine()

        def reload_runtime_config_if_changed(self) -> bool:
            return False

    service._ctx = FakeContext()
    stream = service.engine_chat("needs approval")
    try:
        approval_event = await anext(stream)
        payload = (approval_event.metadata or {}).get("approval")

        assert approval_event.event_type == "approval_request"
        assert isinstance(payload, dict)
        assert payload["pending_id"]
        status = await service.approval_status()
        assert len(status["pending"]) == 1

        resolved = await service.approval_resolve(payload["pending_id"], "definitely_not_valid")
        final = await anext(stream)
    finally:
        await stream.aclose()

    assert resolved == {"ok": True, "pending_id": payload["pending_id"], "decision": "deny"}
    assert final.event_type == "final"
    assert final.content == "decision=deny"
    assert final.metadata["session_id"] == "sess-approval"
    assert await service.approval_status() == {"pending": []}


@pytest.mark.asyncio
async def test_daemon_shutdown_rpc_triggers_server_stop() -> None:
    class ShutdownService(_FakeService):
        def __init__(self) -> None:
            super().__init__()
            self.shutdown_called = False

        async def shutdown(self) -> None:
            self.shutdown_called = True

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        run_dir = Path(root) / "run"
        service = ShutdownService()
        shutdown_event = asyncio.Event()
        server = UnixRpcServer(
            service,
            sock_path=run_dir / "leapd.sock",
            run_dir=run_dir,
            on_shutdown=shutdown_event.set,
        )
        task = asyncio.create_task(server.serve_forever())
        for _ in range(50):
            if (run_dir / "leapd.sock").exists():
                break
            await asyncio.sleep(0.02)
        else:
            task.cancel()
            raise AssertionError("server did not start")
        client = DaemonClient(run_dir / "leapd.sock")
        try:
            await client.shutdown()
            await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert service.shutdown_called is True


@pytest.mark.asyncio
async def test_runtime_service_status_reports_host_backend() -> None:
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService

    class FakeRpc:
        def status_snapshot(self) -> dict[str, Any]:
            return {"backend": "cua-driver", "started": True, "tools_count": 2}

    class FakeContext:
        rpc = FakeRpc()
        engine = None
        storage_volatile = False

        def __init__(self, settings) -> None:
            self.settings = settings
            self._db_holder = None

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        settings = make_settings(root)
        service = RuntimeLeapService(settings, mock_host=True)
        service._ctx = FakeContext(settings)
        status = await service.status()

    assert status["host_backend"] == {
        "backend": "cua-driver",
        "started": True,
        "tools_count": 2,
    }


@pytest.mark.asyncio
async def test_daemon_client_host_lifecycle_rpc() -> None:
    class HostRpcService(_FakeService):
        async def host_status(self) -> dict[str, Any]:
            return {"backend": "mock", "started": False}

        async def host_start(self) -> dict[str, Any]:
            return {"ok": True, "backend": "cua-driver", "started": True}

        async def host_stop(self) -> dict[str, Any]:
            return {"ok": True, "backend": "mock", "started": False}

        async def host_restart(self) -> dict[str, Any]:
            return {"ok": True, "backend": "cua-driver", "started": True, "changed": True}

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(Path(root) / "run", service=HostRpcService())
        client = DaemonClient(run_dir / "leapd.sock")
        try:
            status = await client.host_status()
            started = await client.host_start()
            stopped = await client.host_stop()
            restarted = await client.host_restart()
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert status == {"backend": "mock", "started": False}
    assert started["started"] is True
    assert stopped["backend"] == "mock"
    assert restarted["changed"] is True


@pytest.mark.asyncio
async def test_daemon_client_slash_metadata_rpc() -> None:
    class SlashMetadataService(_FakeService):
        async def tools_list(self) -> dict[str, Any]:
            return {"ok": True, "groups": {"core": ["chat"]}, "total": 1, "mcp_count": 0}

        async def usage_summary(self) -> dict[str, Any]:
            return {
                "ok": True,
                "model": "test-model",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "turn_count": 2,
                "context_used": 15,
                "context_length": 100,
            }

        async def model_info(self, model_name: str = "") -> dict[str, Any]:
            return {
                "ok": True,
                "model": "test-model",
                "context_length": 100,
                "requested_model": model_name,
                "switch_supported": False,
                "env_var": "LEAPFLOW_LLM_MODEL",
            }

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(Path(root) / "run", service=SlashMetadataService())
        client = DaemonClient(run_dir / "leapd.sock")
        try:
            tools = await client.tools_list()
            usage = await client.usage_summary()
            model = await client.model_info("next-model")
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert tools["groups"] == {"core": ["chat"]}
    assert usage["total_tokens"] == 15
    assert model["requested_model"] == "next-model"


@pytest.mark.asyncio
async def test_runtime_service_slash_metadata_payloads() -> None:
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService

    class FakeSummary:
        prompt_tokens = 12
        completion_tokens = 8
        total_tokens = 20

    class FakeUsageTracker:
        def summary(self) -> FakeSummary:
            return FakeSummary()

    class FakeCapabilities:
        context_length = 4096

    class FakeCapabilityRegistry:
        def resolve(self, model: str) -> FakeCapabilities:
            return FakeCapabilities()

    class FakeEngine:
        usage_tracker = FakeUsageTracker()
        model_capabilities = FakeCapabilityRegistry()
        context_token_count = 20
        turn_count = 3

    class FakeRpc:
        connected = False

    class FakeContext:
        def __init__(self, settings) -> None:
            self.settings = settings
            self.engine = FakeEngine()
            self.rpc = FakeRpc()
            self.platform_tools: list[Any] = []

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        settings = make_settings(root)
        service = RuntimeLeapService(settings, mock_host=True)
        service._ctx = FakeContext(settings)
        tools = await service.tools_list()
        usage = await service.usage_summary()
        model = await service.model_info("other-model")

    assert tools["ok"] is True
    assert tools["total"] > 0
    assert usage["total_tokens"] == 20
    assert usage["context_length"] == 4096
    assert model["model"] == settings.llm_model
    assert model["requested_model"] == "other-model"


@pytest.mark.asyncio
async def test_runtime_service_host_lifecycle_delegates_to_context() -> None:
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService

    class FakeContext:
        def __init__(self, settings) -> None:
            self.settings = settings
            self.calls: list[str] = []
            self.rpc = object()

        async def host_backend_status(self) -> dict[str, Any]:
            self.calls.append("status")
            return {"backend": "mock", "started": False}

        async def host_backend_start(self) -> dict[str, Any]:
            self.calls.append("start")
            return {"ok": True, "backend": "cua-driver", "started": True}

        async def host_backend_stop(self) -> dict[str, Any]:
            self.calls.append("stop")
            return {"ok": True, "backend": "mock", "started": False}

        async def host_backend_restart(self) -> dict[str, Any]:
            self.calls.append("restart")
            return {"ok": True, "backend": "cua-driver", "started": True}

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        settings = make_settings(root)
        service = RuntimeLeapService(settings, mock_host=True)
        ctx = FakeContext(settings)
        service._ctx = ctx
        status = await service.host_status()
        started = await service.host_start()
        stopped = await service.host_stop()
        restarted = await service.host_restart()

    assert status["backend"] == "mock"
    assert started["started"] is True
    assert stopped["started"] is False
    assert restarted["backend"] == "cua-driver"
    assert ctx.calls == ["status", "start", "stop", "restart"]


@pytest.mark.asyncio
async def test_daemon_client_approval_resolve_rpc() -> None:
    class ApprovalRpcService(_FakeService):
        async def approval_resolve(self, pending_id: str, decision: str, reason: str = "") -> dict[str, Any]:
            return {"ok": True, "pending_id": pending_id, "decision": decision, "reason": reason}

        async def approval_status(self) -> dict[str, Any]:
            return {"pending": []}

        async def approval_cancel(self, pending_id: str, reason: str = "cancelled") -> dict[str, Any]:
            return {"ok": True, "pending_id": pending_id, "decision": "deny", "reason": reason}

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(Path(root) / "run", service=ApprovalRpcService())
        client = DaemonClient(run_dir / "leapd.sock")
        try:
            result = await client.approval_resolve("p1", "allow_once", reason="user")
            status = await client.approval_status()
            cancelled = await client.approval_cancel("p2")
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert result == {"ok": True, "pending_id": "p1", "decision": "allow_once", "reason": "user"}
    assert status == {"pending": []}
    assert cancelled["decision"] == "deny"


@pytest.mark.asyncio
async def test_daemon_client_reports_unknown_method() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(Path(root) / "run")
        client = DaemonClient(run_dir / "leapd.sock")

        try:
            with pytest.raises(DaemonUnavailableError, match="Unknown method"):
                await client.request("missing.method")
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_ensure_daemon_client_reuses_healthy_daemon() -> None:
    from conftest import make_settings

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        root_path = Path(root)
        settings = make_settings(str(root_path))
        server, task, run_dir = await _start_server(settings.profile_dir / "run")

        try:
            client = await ensure_daemon_client(settings)
            assert client.sock_path == run_dir / "leapd.sock"
            status = await client.status()
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert status["profile"] == "test"


@pytest.mark.asyncio
async def test_daemon_client_surfaces_stream_errors() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, run_dir = await _start_server(
            Path(root) / "run",
            service=_FailingStreamService(),
        )
        client = DaemonClient(run_dir / "leapd.sock")
        events = []

        try:
            with pytest.raises(DaemonUnavailableError, match="Daemon stream failed"):
                async for event in client.engine_chat("world"):
                    events.append(event)
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert [(event.type, event.content) for event in events] == [("chunk", "partial")]


@pytest.mark.asyncio
async def test_runtime_service_hot_reloads_config_before_daemon_chat(
    monkeypatch,
    tmp_path,
) -> None:
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService

    data_dir = tmp_path / "leap-home"
    data_dir.mkdir()
    env_path = data_dir / ".env"
    env_path.write_text("LEAPFLOW_LLM_API_KEY=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)

    settings = make_settings(str(tmp_path))
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "data_dir": data_dir,
            "llm_api_key": "",
            "llm_context_length": 128_000,
            "vlm_api_key": "",
            "visual_track_enabled": False,
        }
    )
    service = RuntimeLeapService(settings, mock_host=True)
    await service.start()
    stream = None
    try:
        env_path.write_text(
            "LEAPFLOW_LLM_API_KEY=sk-daemon-hot-reload\n"
            "LEAPFLOW_LLM_CONTEXT_LENGTH=700000\n",
            encoding="utf-8",
        )
        stream = service.engine_chat("hello")
        first = await anext(stream)

        assert first.event_type == "status"
        assert "Configuration reloaded" in first.content
        assert first.metadata == {
            "llm_model": service.context.settings.llm_model,
            "llm_context_length": 700_000,
            "context_used": 0,
        }
        assert service.context.settings.llm_api_key == "sk-daemon-hot-reload"
        assert service.context.settings.llm_context_length == 700_000
        assert service.context.engine._settings.llm_api_key == "sk-daemon-hot-reload"
        assert service.context.engine._settings.llm_context_length == 700_000
        status = await service.status()
        assert status["config_path"] == str(data_dir / ".env")
        assert status["project_env_path"] == str(tmp_path / ".env")
        assert status["llm_context_length"] == 700_000
        assert status["context_used"] == 0
        assert status["runtime_source"].endswith("leapflow/__init__.py")
        assert status["runtime_executable"]
        assert status["runtime_version"]
    finally:
        if stream is not None:
            await stream.aclose()
        await service.shutdown()


@pytest.mark.asyncio
async def test_runtime_service_serializes_engine_chat_streams(tmp_path) -> None:
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService
    from leapflow.engine import StreamEvent

    class SlowEngine:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.context_token_count = 2_048
            self._current_session_id = "sess-daemon"

        async def run_stream(self, message: str, *, enable_thinking: bool = False):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                yield StreamEvent(type="chunk", content=f"start:{message}")
                await asyncio.sleep(0.05)
                yield StreamEvent(type="final", content=f"done:{message}")
            finally:
                self.active -= 1

    class FakeContext:
        def __init__(self) -> None:
            self.engine = SlowEngine()

        def reload_runtime_config_if_changed(self) -> bool:
            return False

    settings = make_settings(str(tmp_path))
    service = RuntimeLeapService(settings, mock_host=True)
    context = FakeContext()
    service._ctx = context

    async def collect(message: str) -> list:
        return [event async for event in service.engine_chat(message)]

    first, second = await asyncio.gather(collect("one"), collect("two"))
    status = await service.status()

    assert [event.content for event in first] == ["start:one", "done:one"]
    assert [event.content for event in second] == ["start:two", "done:two"]
    assert first[0].metadata["context_used"] == 2_048
    assert first[0].metadata["session_id"] == "sess-daemon"
    assert second[0].metadata["context_used"] == 2_048
    assert status["context_used"] == 2_048
    assert context.engine.max_active == 1


@pytest.mark.asyncio
async def test_ensure_daemon_client_does_not_spawn_when_daemon_unhealthy(
    tmp_path,
    monkeypatch,
) -> None:
    from conftest import make_settings
    import leapflow.daemon.client as client_module

    settings = make_settings(str(tmp_path))
    unhealthy = DaemonInfo(
        pid=4321,
        sock_path=settings.profile_dir / "run" / "leapd.sock",
        start_time=1.0,
        is_running=True,
        is_healthy=False,
    )

    monkeypatch.setattr(client_module.DaemonInfo, "discover", lambda run_dir: unhealthy)

    def fail_spawn(*args, **kwargs):
        raise AssertionError("spawn_daemon must not be called for a running unhealthy daemon")

    monkeypatch.setattr(client_module, "spawn_daemon", fail_spawn)

    with pytest.raises(DaemonUnavailableError, match="running but unhealthy"):
        await ensure_daemon_client(settings)


def test_daemon_runtime_status_prints_diagnostics(capsys) -> None:
    from leapflow.cli.commands.daemon import _print_runtime_status

    _print_runtime_status({
        "profile": "default",
        "active_clients": 1,
        "connected_clients": 2,
        "volatile": False,
        "model": "qwen3.7-plus",
        "context_used": 256,
        "llm_context_length": 256_000,
        "session_id": "sess-1",
        "runtime_version": "0.0.test",
        "runtime_source": "/repo/src/leapflow/__init__.py",
        "runtime_executable": "/venv/bin/python",
        "config_path": "/home/.leapflow/.env",
        "project_env_path": "/repo/.env",
        "db_path": "/home/.leapflow/db/leap.duckdb",
        "host_backend": {
            "backend": "cua-driver",
            "started": True,
            "pid": None,
            "command": "/tmp/cua-driver",
            "args": ["mcp"],
            "capability_version": "test-cap",
        },
    })

    output = capsys.readouterr().out
    assert "runtime: profile=default clients=1 connected=2 volatile=False" in output
    assert "model: qwen3.7-plus context=256/256000" in output
    assert "version: 0.0.test" in output
    assert "source: /repo/src/leapflow/__init__.py" in output
    assert "python: /venv/bin/python" in output
    assert "host: backend=cua-driver started=True pid=None" in output
    assert "host_command: /tmp/cua-driver mcp" in output
    assert "host_capability: test-cap" in output


def test_daemon_restart_stops_waits_and_starts(monkeypatch, tmp_path) -> None:
    from leapflow.cli.commands import daemon as daemon_module

    calls: list[str] = []

    class Settings:
        profile_dir = tmp_path

    monkeypatch.setattr(daemon_module, "_stop", lambda run_dir: calls.append("stop") or 0)
    monkeypatch.setattr(daemon_module, "_wait_stopped", lambda run_dir: calls.append("wait") or True)
    monkeypatch.setattr(daemon_module, "_start", lambda settings, mock_host: calls.append("start") or 0)

    assert daemon_module._restart(Settings(), mock_host=True) == 0
    assert calls == ["stop", "wait", "start"]


def test_stream_chunk_notification_preserves_event_shape() -> None:
    notification = StreamChunk(
        request_id="req1",
        content="payload",
        event_type="thinking",
        metadata={"session_id": "s1"},
    ).to_notification()

    assert notification.method == "stream.chunk"
    assert notification.params == {
        "id": "req1",
        "content": "payload",
        "done": False,
        "event_type": "thinking",
        "metadata": {"session_id": "s1"},
    }
