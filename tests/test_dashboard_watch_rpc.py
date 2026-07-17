"""Hermetic tests for the watch RPC surface and the /board command handler.

No network, no LLM, no full Context: the daemon service and slash handler are
exercised with an in-memory MonitorManager and a fake producer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from leapflow.cli.commands.slash_handlers import command_execute
from leapflow.daemon.service import RuntimeLeapService
from leapflow.monitor import Finding, MonitorManager, ProducerRegistry, Severity, WatchSpec
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


async def test_schedule_watch_once_runs_in_background(tmp_path: Path) -> None:
    """schedule_watch_once must not block the caller yet still produce findings."""
    manager = _manager(tmp_path)
    view = await manager.arm_watch(WatchSpec(name="D", domain="demo", sensitivity="notable"))

    manager.schedule_watch_once(view.watch_id, force=True)

    tasks = list(manager._background_tasks)
    assert len(tasks) == 1  # scheduled, not awaited inline
    assert manager.finding_store.count(watch_id=view.watch_id) == 0  # not run yet
    await asyncio.gather(*tasks)
    assert manager.finding_store.count(watch_id=view.watch_id) >= 1


async def test_board_session_returns_watch_id_without_blocking(tmp_path: Path) -> None:
    """/board session arms the session watch, returns its id + open payload, and
    schedules the (LLM-backed) analysis in the background instead of awaiting it."""
    manager = _manager(tmp_path)
    ctx = SimpleNamespace(monitors=manager, settings=None, engine=None)

    payload = await command_execute(ctx, "board session", "")

    assert payload["view"] == "dashboard" and payload["mode"] == "open"
    assert payload["action"] == "session"
    assert payload.get("watch_id")  # id surfaced to the user
    assert len(manager._background_tasks) == 1  # analysis deferred, RPC returns fast
    await asyncio.gather(*list(manager._background_tasks))


async def test_board_new_opens_web_view_for_created_watch(tmp_path: Path) -> None:
    """/board new must both confirm the armed watch (text) and open its web view
    (open_web), focused on the freshly created watch."""
    ctx = SimpleNamespace(monitors=_manager(tmp_path), settings=None, engine=None)

    payload = await command_execute(ctx, "board new", "demo --name Market --trigger 5m")

    assert payload["mode"] == "armed"  # still renders the confirmation text
    assert payload["watch"]["name"] == "Market"
    assert payload["open_web"] is True  # and launches the browser
    assert payload["action"] == "watch"
    assert payload["target"] == payload["watch"]["watch_id"]


async def test_board_open_targets_the_requested_watch(tmp_path: Path) -> None:
    """/board open <id> must carry the target so the web lands on the watch
    detail, not the overview."""
    manager = _manager(tmp_path)
    view = await manager.arm_watch(WatchSpec(name="D", domain="demo"))
    ctx = SimpleNamespace(monitors=manager, settings=None, engine=None)

    payload = await command_execute(ctx, "board open", view.watch_id)

    assert payload["mode"] == "open"
    assert payload["action"] == "open"
    assert payload["target"] == view.watch_id


def test_build_view_url_propagates_action_and_target() -> None:
    from leapflow.dashboard import launcher

    assert (
        launcher.build_view_url("127.0.0.1", 8765, "tok", action="watch", target="abc123")
        == "http://127.0.0.1:8765/?token=tok&action=watch&target=abc123"
    )
    # Home is the default view and needs no action/target query params.
    assert launcher.build_view_url("127.0.0.1", 8765, "tok") == "http://127.0.0.1:8765/?token=tok"
    assert (
        launcher.build_view_url("127.0.0.1", 8765, "tok", action="session", target="session")
        == "http://127.0.0.1:8765/?token=tok&action=session&target=session"
    )


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
