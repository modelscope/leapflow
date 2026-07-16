"""Hermetic tests for the watch RPC surface and the /board command handler.

No network, no LLM, no full Context: the daemon service and slash handler are
exercised with an in-memory MonitorManager and a fake producer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from leapflow.cli.commands.slash_handlers import command_execute
from leapflow.daemon.service import RuntimeLeapService
from leapflow.monitor import Finding, MonitorManager, ProducerRegistry, Severity
from leapflow.monitor.types import ProducerContext
from leapflow.storage.connection import LocalConnectionHolder


class _DemoProducer:
    @property
    def domain(self) -> str:
        return "demo"

    async def observe(self, ctx: ProducerContext) -> list[Finding]:
        return [Finding(watch_id="", domain="demo", title="hit", severity=Severity.ALERT, dedup_key="d1")]


def _manager(tmp_path: Path, emit=None) -> MonitorManager:
    producers = ProducerRegistry()
    producers.register(_DemoProducer())
    return MonitorManager(
        holder=LocalConnectionHolder(tmp_path / "leap.duckdb"),
        producers=producers,
        emit=emit,
    )


async def test_service_watch_rpc_roundtrip(tmp_path: Path) -> None:
    emitted: list[tuple[str, dict]] = []
    service = RuntimeLeapService(SimpleNamespace())
    service._monitors = _manager(tmp_path, emit=lambda et, p: emitted.append((et, p)))

    view = await service.watch_arm({"name": "D", "domain": "demo", "sensitivity": "notable"})
    watch_id = view["watch_id"]
    assert service.has_active_watches() is True

    assert len(await service.watch_list()) == 1
    assert (await service.watch_get(watch_id))["domain"] == "demo"
    assert (await service.watch_get(watch_id))["client_coupled"] is False

    result = await service.watch_refresh(watch_id)
    assert result["ok"] is True and result["findings"] == 1
    assert any(et == "monitor.finding" for et, _ in emitted)
    assert len(await service.watch_findings(watch_id)) == 1

    assert (await service.watch_pause(watch_id))["state"] == "suspended"
    assert service.has_active_watches() is False
    assert (await service.watch_resume(watch_id))["state"] == "armed"
    assert (await service.watch_mute(watch_id, muted=True))["muted"] is True
    assert (await service.watch_stop(watch_id))["state"] == "done"


async def test_service_watch_summary_separates_keepalive_watches(tmp_path: Path) -> None:
    service = RuntimeLeapService(SimpleNamespace())
    service._monitors = _manager(tmp_path)

    await service.watch_arm({"name": "Session", "domain": "demo", "client_coupled": True})
    await service.watch_arm({"name": "Market", "domain": "demo"})
    summary = service._watch_runtime_summary()

    assert summary["active"] == 2
    assert summary["client_coupled_active"] == 1
    assert summary["standalone_active"] == 1
    assert summary["active_samples"][0]["state"] == "armed"


async def test_service_watch_unavailable_is_graceful() -> None:
    service = RuntimeLeapService(SimpleNamespace())
    # No monitor runtime attached (scheduler disabled).
    assert service.has_active_watches() is False
    assert await service.watch_list() == []
    assert await service.watch_findings() == []


async def test_dashboard_command_execute_flow(tmp_path: Path) -> None:
    ctx = SimpleNamespace(monitors=_manager(tmp_path))

    armed = await command_execute(ctx, "board new", "demo --name Market --trigger 5m")
    assert armed["ok"] is True and armed["mode"] == "armed"
    watch_id = armed["watch"]["watch_id"]
    assert armed["watch"]["trigger"] == "every 5m"

    listed = await command_execute(ctx, "board list", "")
    assert listed["mode"] == "list" and len(listed["watches"]) == 1

    status_payload = await command_execute(ctx, "board status", "")
    assert status_payload["mode"] == "status" and status_payload["count"] == 1

    # Short-prefix id resolution + manual refresh.
    refreshed = await command_execute(ctx, "board refresh", watch_id[:8])
    assert refreshed["mode"] == "refresh" and refreshed["ok"] is True

    finds = await command_execute(ctx, "board findings", "")
    assert finds["mode"] == "findings" and len(finds["findings"]) >= 1

    paused = await command_execute(ctx, "board pause", watch_id[:8])
    assert paused["watch"]["state"] == "suspended"

    unknown = await command_execute(ctx, "board", "bogus")
    assert unknown["ok"] is False


async def test_dashboard_command_scheduler_disabled(tmp_path: Path) -> None:
    ctx = SimpleNamespace(monitors=None)
    disabled = await command_execute(ctx, "board list", "")
    assert disabled["ok"] is False
    assert "unavailable" in disabled["message"].lower()


async def test_dashboard_web_view_actions_work_without_monitor_runtime() -> None:
    """open/home/session must open the web board even with no local monitor
    runtime (the in-process fallback REPL), never leak back to chat."""
    ctx = SimpleNamespace(monitors=None, settings=None, engine=None)
    for command in ("board session", "board open", "board home"):
        payload = await command_execute(ctx, command, "")
        assert payload["ok"] is True, command
        assert payload["view"] == "dashboard"
        assert payload["mode"] == "open"
    session_payload = await command_execute(ctx, "board session", "")
    assert session_payload["action"] == "session"
