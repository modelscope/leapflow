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


async def _start_server(run_dir: Path, service=None):
    server = UnixRpcServer(
        service or _FakeService(),
        sock_path=run_dir / "leapd.sock",
        run_dir=run_dir,
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

    settings = make_settings(str(tmp_path))
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "data_dir": data_dir,
            "llm_api_key": "",
            "vlm_api_key": "",
            "visual_track_enabled": False,
        }
    )
    service = RuntimeLeapService(settings, mock_host=True)
    await service.start()
    stream = None
    try:
        env_path.write_text("LEAPFLOW_LLM_API_KEY=sk-daemon-hot-reload\n", encoding="utf-8")
        stream = service.engine_chat("hello")
        first = await anext(stream)

        assert first.event_type == "status"
        assert "Configuration reloaded" in first.content
        assert service.context.settings.llm_api_key == "sk-daemon-hot-reload"
        assert service.context.engine._settings.llm_api_key == "sk-daemon-hot-reload"
        status = await service.status()
        assert status["config_path"] == str(data_dir / ".env")
        assert status["project_env_path"] == str(tmp_path / ".env")
    finally:
        if stream is not None:
            await stream.aclose()
        await service.shutdown()


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
