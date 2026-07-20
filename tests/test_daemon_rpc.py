from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from leapflow.daemon.client import DaemonClient, DaemonUnavailableError, ensure_daemon_client
from leapflow.daemon.lifecycle import DaemonInfo
from leapflow.daemon.protocol import StreamChunk
from leapflow.daemon.server import UnixRpcServer


def test_settings_runtime_dir_defaults_to_profile_runtime(tmp_path) -> None:
    from conftest import make_settings

    settings = make_settings(str(tmp_path))
    assert settings.runtime_dir == tmp_path / "profiles" / "default" / "runtime"

    override = tmp_path / "custom-runtime"
    overridden = replace(settings, runtime_dir=override)
    assert overridden.profile_dir == settings.profile_dir
    assert overridden.runtime_dir == override


class _FakeService:
    def __init__(self) -> None:
        self._client_count = lambda: 0
        self.cancelled = False

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

    async def engine_cancel(self) -> bool:
        self.cancelled = True
        return True


class _FailingStreamService(_FakeService):
    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(request_id="", content="partial", event_type="chunk")
        raise RuntimeError("stream exploded")


class _SlowFirstChunkService(_FakeService):
    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        await asyncio.sleep(0.08)
        yield StreamChunk(request_id="", content=f"slow {message}", event_type="chunk")
        yield StreamChunk(request_id="", content="done", event_type="final")


class _RequestIdCaptureService(_FakeService):
    def __init__(self) -> None:
        super().__init__()
        self.request_ids: list[str] = []

    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        request_id = str(kwargs.get("request_id") or "")
        self.request_ids.append(request_id)
        yield StreamChunk(
            request_id=request_id,
            content=f"request {message}",
            event_type="chunk",
            metadata={"request_id": request_id},
        )
        yield StreamChunk(request_id=request_id, content="done", event_type="final")


async def _start_server(runtime_dir: Path, service=None, *, stream_heartbeat_s: float | None = None):
    server = UnixRpcServer(
        service or _FakeService(),
        sock_path=runtime_dir / "leapd.sock",
        runtime_dir=runtime_dir,
        stream_heartbeat_s=stream_heartbeat_s,
    )
    task = asyncio.create_task(server.serve_forever())
    for _ in range(50):
        if (runtime_dir / "leapd.sock").exists():
            return server, task, runtime_dir
        await asyncio.sleep(0.02)
    task.cancel()
    raise AssertionError("server did not start")


@pytest.mark.asyncio
async def test_daemon_client_receives_stream_events() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, runtime_dir = await _start_server(Path(root) / "runtime")
        client = DaemonClient(runtime_dir / "leapd.sock")

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
async def test_daemon_server_injects_rpc_request_id_into_engine_chat() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        service = _RequestIdCaptureService()
        server, task, runtime_dir = await _start_server(Path(root) / "runtime", service=service)
        client = DaemonClient(runtime_dir / "leapd.sock")

        try:
            events = [event async for event in client.engine_chat("world")]
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert len(service.request_ids) == 1
    assert service.request_ids[0]
    assert events[0].metadata == {"request_id": service.request_ids[0]}


@pytest.mark.asyncio
async def test_runtime_service_replays_duplicate_engine_request_without_rerun() -> None:
    from types import SimpleNamespace

    from leapflow.daemon.service import RuntimeLeapService
    from leapflow.engine import StreamEvent

    class FakeEngine:
        def __init__(self) -> None:
            self.calls = 0
            self._current_session_id = "session-1"

        async def run_stream(self, message: str, *, enable_thinking: bool = False, request_id: str = ""):
            self.calls += 1
            yield StreamEvent(
                type="chunk",
                content=f"{request_id}:{message}",
                metadata={"seen_request_id": request_id},
            )
            yield StreamEvent(type="final", content="done")

    class FakeContext:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(llm_context_length=100)
            self.engine = FakeEngine()

        def reload_runtime_config_if_changed(self) -> bool:
            return False

    service = RuntimeLeapService(SimpleNamespace(llm_context_length=100))
    ctx = FakeContext()
    service._ctx = ctx

    first = [chunk async for chunk in service.engine_chat("hello", request_id="req-1")]
    second = [chunk async for chunk in service.engine_chat("hello", request_id="req-1")]

    assert ctx.engine.calls == 1
    assert [chunk.content for chunk in first] == ["req-1:hello", "done"]
    assert [chunk.content for chunk in second] == ["req-1:hello", "done"]
    assert second[0].metadata["replayed_request"] is True


@pytest.mark.asyncio
async def test_runtime_service_prunes_completed_engine_request_replay_records() -> None:
    from types import SimpleNamespace

    from leapflow.daemon.service import RuntimeLeapService

    settings = SimpleNamespace(
        llm_context_length=100,
        daemon_request_ledger_ttl_s=1_000_000_000_000.0,
        daemon_request_ledger_max_entries=2,
    )
    service = RuntimeLeapService(settings)
    service._engine_request_ledger = {
        "old-1": {"status": "completed", "chunks": [], "created_at": 1.0, "completed_at": 1.0},
        "old-2": {"status": "failed", "chunks": [], "created_at": 2.0, "completed_at": 2.0},
        "new-1": {"status": "completed", "chunks": [], "created_at": 3.0, "completed_at": 3.0},
        "running": {"status": "running", "chunks": [], "created_at": 0.0},
    }

    service._prune_engine_request_ledger()

    assert "running" in service._engine_request_ledger
    assert "new-1" in service._engine_request_ledger
    assert len(service._engine_request_ledger) <= 2


@pytest.mark.asyncio
async def test_daemon_client_can_cancel_engine_turn() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        service = _FakeService()
        server, task, runtime_dir = await _start_server(Path(root) / "runtime", service=service)
        client = DaemonClient(runtime_dir / "leapd.sock")

        try:
            cancelled = await client.engine_cancel()
        finally:
            task.cancel()
            await server.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert cancelled is True
    assert service.cancelled is True


@pytest.mark.asyncio
async def test_daemon_client_stream_heartbeat_prevents_idle_timeout() -> None:
    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, runtime_dir = await _start_server(
            Path(root) / "runtime",
            service=_SlowFirstChunkService(),
            stream_heartbeat_s=0.01,
        )
        client = DaemonClient(runtime_dir / "leapd.sock", timeout_s=0.03)

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

        def __init__(self, runtime_dir: Path) -> None:
            self.runtime_dir = runtime_dir

    runtime_dir = tmp_path / "runtime"
    lease = ClientLease(runtime_dir, kind="tui", session_id="sess-live", touch_interval_s=1.0)
    await lease.start()
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _watch_idle_shutdown(
            IdleServer(runtime_dir),
            stop_event,
            idle_timeout_s=0.03,
            lease_ttl_s=1.0,
            poll_interval_s=0.01,
        )
    )
    try:
        await asyncio.sleep(0.08)
        assert not stop_event.is_set()
        snapshots = read_active_client_leases(runtime_dir, ttl_s=1.0)
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
        assert (approval_event.metadata or {}).get("request_id")
        assert payload.get("request_id") == (approval_event.metadata or {}).get("request_id")
        status = await service.approval_status()
        assert len(status["pending"]) == 1

        resolved = await service.approval_resolve(payload["pending_id"], "definitely_not_valid")
        final = await anext(stream)
    finally:
        await stream.aclose()

    assert resolved["ok"] is True
    assert resolved["pending_id"] == payload["pending_id"]
    assert resolved["decision"] == "deny"
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
        runtime_dir = Path(root) / "runtime"
        service = ShutdownService()
        shutdown_event = asyncio.Event()
        server = UnixRpcServer(
            service,
            sock_path=runtime_dir / "leapd.sock",
            runtime_dir=runtime_dir,
            on_shutdown=shutdown_event.set,
        )
        task = asyncio.create_task(server.serve_forever())
        for _ in range(50):
            if (runtime_dir / "leapd.sock").exists():
                break
            await asyncio.sleep(0.02)
        else:
            task.cancel()
            raise AssertionError("server did not start")
        client = DaemonClient(runtime_dir / "leapd.sock")
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
        server, task, runtime_dir = await _start_server(Path(root) / "runtime", service=HostRpcService())
        client = DaemonClient(runtime_dir / "leapd.sock")
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

        async def command_execute(self, name: str, args: str = "") -> dict[str, Any]:
            return {
                "ok": True,
                "view": name,
                "model": "test-model",
                "context_length": 100,
                "requested_model": args,
            }

        async def app_command(self, args: str = "") -> dict[str, Any]:
            return {
                "ok": True,
                "view": "list",
                "args": args,
                "result": {"platforms": [{"id": "feishu", "name": "Feishu", "state": "available"}]},
            }

    with tempfile.TemporaryDirectory(prefix="lfd-", dir="/tmp") as root:
        server, task, runtime_dir = await _start_server(Path(root) / "runtime", service=SlashMetadataService())
        client = DaemonClient(runtime_dir / "leapd.sock")
        try:
            tools = await client.tools_list()
            usage = await client.usage_summary()
            model = await client.command_execute("model", "next-model")
            app_payload = await client.app_command("list")
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
    assert app_payload["view"] == "list"
    assert app_payload["args"] == "list"


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
        from leapflow.gateway.server import GatewayServer

        gateway_server = GatewayServer(settings.profile_dir)
        gateway_server.discover_manifests()
        service = RuntimeLeapService(settings, mock_host=True)
        service._ctx = FakeContext(settings)
        service._ctx.gateway_server = gateway_server
        try:
            tools = await service.tools_list()
            usage = await service.usage_summary()
            model = await service.command_execute("model", "")
            app_list = await service.app_command("list")
            app_status = await service.app_command("status feishu")
        finally:
            await gateway_server.stop()

    assert tools["ok"] is True
    assert tools["total"] > 0
    assert usage["total_tokens"] == 20
    assert usage["context_length"] == 4096
    assert model["model"] == settings.llm_model
    assert model["requested_model"] == ""
    assert app_list["ok"] is True
    assert app_list["view"] == "list"
    assert {entry["id"] for entry in app_list["result"]["platforms"]} >= {"feishu", "telegram", "dingtalk"}
    assert app_status["ok"] is True
    assert app_status["view"] == "status"
    assert app_status["result"]["platform"] == "feishu"


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
        server, task, runtime_dir = await _start_server(Path(root) / "runtime", service=ApprovalRpcService())
        client = DaemonClient(runtime_dir / "leapd.sock")
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
        server, task, runtime_dir = await _start_server(Path(root) / "runtime")
        client = DaemonClient(runtime_dir / "leapd.sock")

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
        server, task, runtime_dir = await _start_server(settings.runtime_dir)

        try:
            client = await ensure_daemon_client(settings)
            assert client.sock_path == runtime_dir / "leapd.sock"
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
        server, task, runtime_dir = await _start_server(
            Path(root) / "runtime",
            service=_FailingStreamService(),
        )
        client = DaemonClient(runtime_dir / "leapd.sock")
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(data_dir))
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)

    settings = make_settings(str(data_dir))
    settings = settings.__class__(
        **{
            **settings.__dict__,
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
        settings.profile_layout.llm_config_path.write_text(
            "llm:\n"
            "  api_key: sk-daemon-hot-reload\n"
            "  context_length: 700000\n",
            encoding="utf-8",
        )
        stream = service.engine_chat("hello")
        first = await anext(stream)

        assert first.event_type == "status"
        assert "Configuration reloaded" in first.content
        metadata = dict(first.metadata or {})
        request_id = metadata.pop("request_id", "")
        assert request_id
        assert metadata == {
            "llm_model": service.context.settings.llm_model,
            "llm_context_length": 700_000,
            "context_used": 0,
        }
        assert service.context.settings.llm_api_key == "sk-daemon-hot-reload"
        assert service.context.settings.llm_context_length == 700_000
        assert service.context.engine._settings.llm_api_key == "sk-daemon-hot-reload"
        assert service.context.engine._settings.llm_context_length == 700_000
        status = await service.status()
        assert status["profile_manifest_path"] == str(settings.profile_layout.manifest_path)
        assert status["profile_config_dir"] == str(settings.profile_layout.config_dir)
        assert status["user_config_path"] == str(settings.layout.user_config_path)
        assert status["workspace_config_path"] == str(tmp_path / ".leapflow" / "config.yaml")
        assert str(settings.profile_layout.llm_config_path) in status["config_sources"]
        assert status["runtime_dir"] == str(settings.runtime_dir)
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
            self.context_budget_snapshot = {
                "message_tokens": 1_800,
                "tool_schema_tokens": 248,
                "total_tokens": 2_048,
                "context_length": 16_000,
                "ratio": 0.128,
                "compressed": False,
                "forced_final_answer": False,
                "context_posture": "research",
                "context_signal": "multi-source",
                "context_guidance": "maintain research ledger and synthesize findings",
                "compression_reason": "threshold-triggered",
                "compression_savings_ratio": 0.25,
            }
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
    assert first[0].metadata["context_budget_snapshot"]["total_tokens"] == 2_048
    assert first[0].metadata["context_budget_snapshot"]["tool_schema_tokens"] == 248
    assert first[0].metadata["llm_context_length"] == 16_000
    assert first[0].metadata["context_posture"] == "research"
    assert first[0].metadata["context_signal"] == "multi-source"
    assert first[0].metadata["context_guidance"] == "maintain research ledger and synthesize findings"
    assert first[0].metadata["compression_reason"] == "threshold-triggered"
    assert first[0].metadata["compression_savings_ratio"] == 0.25
    assert first[0].metadata["session_id"] == "sess-daemon"
    assert second[0].metadata["context_used"] == 2_048
    assert status["context_used"] == 2_048
    assert status["context_budget_snapshot"]["total_tokens"] == 2_048
    assert status["context_posture"] == "research"
    assert status["context_guidance"] == "maintain research ledger and synthesize findings"
    assert status["compression_reason"] == "threshold-triggered"
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
        sock_path=settings.runtime_dir / "leapd.sock",
        start_time=1.0,
        is_running=True,
        is_healthy=False,
    )

    monkeypatch.setattr(client_module.DaemonInfo, "discover", lambda runtime_dir: unhealthy)

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
        "llm_context_length": 1_000_000,
        "session_id": "sess-1",
        "runtime_version": "0.0.test",
        "runtime_source": "/repo/src/leapflow/__init__.py",
        "runtime_executable": "/venv/bin/python",
        "profile_manifest_path": "/home/.leapflow/profiles/default/profile.yaml",
        "profile_config_dir": "/home/.leapflow/profiles/default/config",
        "user_config_path": "/home/.leapflow/config/user.yaml",
        "workspace_config_path": "/repo/.leapflow/config.yaml",
        "runtime_dir": "/home/.leapflow/profiles/default/runtime",
        "db_path": "/home/.leapflow/profiles/default/db/leap.duckdb",
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
    assert "model: qwen3.7-plus context=256/1000000" in output
    assert "version: 0.0.test" in output
    assert "source: /repo/src/leapflow/__init__.py" in output
    assert "python: /venv/bin/python" in output
    assert "profile_config: /home/.leapflow/profiles/default/config" in output
    assert "user_config: /home/.leapflow/config/user.yaml" in output
    assert "workspace_config: /repo/.leapflow/config.yaml" in output
    assert "runtime_dir: /home/.leapflow/profiles/default/runtime" in output
    assert "host: backend=cua-driver started=True pid=None" in output
    assert "host_command: /tmp/cua-driver mcp" in output
    assert "host_capability: test-cap" in output


def test_stop_daemon_sends_sigterm_waits_and_cleans(monkeypatch, tmp_path) -> None:
    import leapflow.daemon.lifecycle as lifecycle_module

    running = lifecycle_module.DaemonInfo(
        pid=1234,
        sock_path=tmp_path / "runtime" / "leapd.sock",
        start_time=None,
        is_running=True,
        is_healthy=True,
    )
    stopped = lifecycle_module.DaemonInfo(
        pid=1234,
        sock_path=None,
        start_time=None,
        is_running=False,
        is_healthy=False,
    )
    states = [running, stopped]
    signals: list[int] = []

    def discover(runtime_dir):
        return states.pop(0) if states else stopped

    def send_signal(runtime_dir, sig):
        signals.append(sig)
        return True

    monkeypatch.setattr(lifecycle_module.DaemonInfo, "discover", staticmethod(discover))
    monkeypatch.setattr(lifecycle_module, "send_signal", send_signal)
    monkeypatch.setattr(lifecycle_module, "cleanup_stale", lambda runtime_dir: True)

    result = lifecycle_module.stop_daemon(tmp_path / "runtime", timeout_s=1.0)

    assert result.stopped is True
    assert result.signal_sent is True
    assert result.stale_cleaned is True
    assert signals == [lifecycle_module.signal.SIGTERM]


def test_stop_daemon_force_escalates_after_timeout(monkeypatch, tmp_path) -> None:
    import leapflow.daemon.lifecycle as lifecycle_module

    running = lifecycle_module.DaemonInfo(
        pid=1234,
        sock_path=tmp_path / "runtime" / "leapd.sock",
        start_time=None,
        is_running=True,
        is_healthy=False,
    )
    signals: list[int] = []

    monkeypatch.setattr(lifecycle_module.DaemonInfo, "discover", staticmethod(lambda runtime_dir: running))
    monkeypatch.setattr(lifecycle_module, "send_signal", lambda runtime_dir, sig: signals.append(sig) or True)

    result = lifecycle_module.stop_daemon(
        tmp_path / "runtime",
        timeout_s=0.05,
        force=True,
        force_timeout_s=0.05,
        poll_interval_s=0.01,
    )

    assert result.stopped is False
    assert result.timed_out is True
    assert result.forced is True
    assert signals == [lifecycle_module.signal.SIGTERM, lifecycle_module.signal.SIGKILL]


def test_stop_daemon_narrates_progress(monkeypatch, tmp_path) -> None:
    import leapflow.daemon.lifecycle as lifecycle_module

    running = lifecycle_module.DaemonInfo(
        pid=4885,
        sock_path=tmp_path / "runtime" / "leapd.sock",
        start_time=None,
        is_running=True,
        is_healthy=False,
    )
    monkeypatch.setattr(lifecycle_module.DaemonInfo, "discover", staticmethod(lambda runtime_dir: running))
    monkeypatch.setattr(lifecycle_module, "send_signal", lambda runtime_dir, sig: True)

    messages: list[str] = []
    result = lifecycle_module.stop_daemon(
        tmp_path / "runtime",
        timeout_s=0.05,
        force=True,
        grace_timeout_s=0.05,
        force_timeout_s=0.05,
        poll_interval_s=0.01,
        on_progress=messages.append,
    )

    assert result.timed_out is True
    # Every escalation step is narrated so the wait is transparent, not silent.
    assert any("graceful" in m.lower() for m in messages)
    assert any("SIGTERM" in m for m in messages)
    assert any("SIGKILL" in m for m in messages)


def test_process_alive_reaps_exited_child_as_dead() -> None:
    import subprocess
    import sys
    import time

    import leapflow.daemon.lifecycle as lifecycle_module

    # A child spawned by this process becomes an unreaped zombie once it exits;
    # os.kill(pid, 0) still succeeds for zombies. _process_alive must reap it and
    # report it as dead so a SIGKILL'd daemon is not seen as "still running".
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 - trusted argv
    deadline = time.time() + 5.0
    alive = True
    while time.time() < deadline:
        alive = lifecycle_module._process_alive(proc.pid)
        if not alive:
            break
        time.sleep(0.05)
    proc.returncode = 0  # already reaped by _process_alive; silence Popen.__del__

    assert alive is False


def test_process_alive_false_for_reaped_pid() -> None:
    import subprocess
    import sys

    import leapflow.daemon.lifecycle as lifecycle_module

    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 - trusted argv
    proc.wait()  # fully reaped; pid no longer exists

    assert lifecycle_module._process_alive(proc.pid) is False


def test_daemon_restart_stops_and_starts(monkeypatch, tmp_path) -> None:
    from leapflow.cli.commands import daemon as daemon_module

    calls: list[str] = []

    class Settings:
        profile_dir = tmp_path
        runtime_dir = tmp_path / "runtime"

    monkeypatch.setattr(
        daemon_module,
        "_stop",
        lambda runtime_dir, **kwargs: calls.append(f"stop:{kwargs.get('force')}") or 0,
    )
    monkeypatch.setattr(daemon_module, "_start", lambda settings, mock_host: calls.append("start") or 0)

    assert daemon_module._restart(Settings(), mock_host=True, force=True) == 0
    assert calls == ["stop:True", "start"]


def test_daemon_restart_aborts_when_stop_fails(monkeypatch, tmp_path) -> None:
    from leapflow.cli.commands import daemon as daemon_module

    calls: list[str] = []

    class Settings:
        profile_dir = tmp_path
        runtime_dir = tmp_path / "runtime"

    monkeypatch.setattr(
        daemon_module,
        "_stop",
        lambda runtime_dir, **kwargs: calls.append("stop") or 1,
    )
    monkeypatch.setattr(daemon_module, "_start", lambda settings, mock_host: calls.append("start") or 0)

    assert daemon_module._restart(Settings(), mock_host=True) == 1
    assert calls == ["stop"]


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
