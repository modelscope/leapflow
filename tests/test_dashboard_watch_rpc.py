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


# -- Service-level watch RPCs (transport for the monitor subsystem) -----------


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


# -- /board command: one target (current session), template = view lens -------


async def test_board_bare_opens_session_with_default_template(tmp_path: Path) -> None:
    """Bare /board analyzes the current session with the generic default and
    schedules the (LLM-backed) analysis in the background."""
    manager = _manager(tmp_path)
    ctx = SimpleNamespace(monitors=manager, settings=None, engine=None)

    payload = await command_execute(ctx, "board", "")

    assert payload["view"] == "dashboard" and payload["mode"] == "open"
    assert payload["template"] == "generic"
    assert payload.get("watch_id")  # session watch armed on demand
    assert len(manager._background_tasks) == 1  # analysis deferred, RPC returns fast
    await asyncio.gather(*list(manager._background_tasks))


async def test_board_named_template_opens_session_with_lens(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    ctx = SimpleNamespace(monitors=manager, settings=None, engine=None)

    payload = await command_execute(ctx, "board", "finance")

    assert payload["mode"] == "open"
    assert payload["template"] == "finance"
    assert "note" not in payload  # finance is a real builtin lens
    await asyncio.gather(*list(manager._background_tasks))


async def test_board_unknown_template_falls_back_with_note(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    ctx = SimpleNamespace(monitors=manager, settings=None, engine=None)

    payload = await command_execute(ctx, "board", "nope")

    assert payload["mode"] == "open"
    assert payload["template"] == "generic"  # degraded to default
    assert "note" in payload and "nope" in payload["note"]
    await asyncio.gather(*list(manager._background_tasks))


async def test_board_status_and_templates_are_discoverable(tmp_path: Path) -> None:
    ctx = SimpleNamespace(monitors=_manager(tmp_path), settings=None, engine=None)

    status_payload = await command_execute(ctx, "board status", "")
    assert status_payload["mode"] == "status"
    assert "generic" in status_payload["templates"]
    assert status_payload["default"] == "generic"

    templates = await command_execute(ctx, "board templates", "")
    assert templates["mode"] == "templates"
    names = {t["name"] for t in templates["templates"]}
    assert {"generic", "finance", "sentiment", "research"}.issubset(names)


async def test_board_refresh_reanalyzes_session(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    ctx = SimpleNamespace(monitors=manager, settings=None, engine=None)

    refreshed = await command_execute(ctx, "board refresh", "")
    assert refreshed["mode"] == "control"
    assert refreshed["action"] == "refresh" and refreshed["ok"] is True
    await asyncio.gather(*list(manager._background_tasks))


async def test_board_control_requires_monitor_runtime() -> None:
    ctx = SimpleNamespace(monitors=None, settings=None, engine=None)
    disabled = await command_execute(ctx, "board refresh", "")
    assert disabled["ok"] is False
    assert "unavailable" in disabled["message"].lower()


async def test_board_opens_without_monitor_runtime() -> None:
    """Bare /board and /board <template> open the web board even with no local
    monitor runtime (the in-process fallback), never leaking back to chat."""
    ctx = SimpleNamespace(monitors=None, settings=None, engine=None)
    for args in ("", "finance"):
        payload = await command_execute(ctx, "board", args)
        assert payload["ok"] is True, args
        assert payload["view"] == "dashboard"
        assert payload["mode"] == "open"


def test_build_view_url_carries_template() -> None:
    from leapflow.dashboard import launcher

    assert (
        launcher.build_view_url("127.0.0.1", 8765, "tok", template="finance")
        == "http://127.0.0.1:8765/?token=tok&template=finance"
    )
    # Default view needs no template query param.
    assert launcher.build_view_url("127.0.0.1", 8765, "tok") == "http://127.0.0.1:8765/?token=tok"


def _templates_ctx(templates_dir: Path) -> SimpleNamespace:
    layout = SimpleNamespace(dashboard=SimpleNamespace(templates_dir=templates_dir))
    return SimpleNamespace(monitors=None, engine=None,
                           settings=SimpleNamespace(profile_layout=layout))


_VALID_TEMPLATE = (
    "template: crypto\n"
    "title: '{{ title }}'\n"
    "layout:\n"
    "  - type: Page\n"
    "    props:\n"
    "      title: '{{ title }}'\n"
    "    children:\n"
    "      - type: StoryPanel\n"
    "        props:\n"
    "          title: Narrative\n"
    "          text: '{{ analysis.story }}'\n"
)


async def test_board_templates_add_list_remove_flow(tmp_path: Path) -> None:
    """A user points at any local YAML; it validates, installs into the profile
    templates dir, becomes discoverable, and is removable."""
    tpl_dir = tmp_path / "tpl"
    ctx = _templates_ctx(tpl_dir)
    src = tmp_path / "crypto.yaml"
    src.write_text(_VALID_TEMPLATE, encoding="utf-8")

    added = await command_execute(ctx, "board templates", f"add {src}")
    assert added["ok"] is True and added["template"] == "crypto"
    assert (tpl_dir / "crypto.yaml").exists()  # installed into the managed dir

    listed = await command_execute(ctx, "board templates", "")
    entry = next(t for t in listed["templates"] if t["name"] == "crypto")
    assert entry["source"] == "user"

    # The freshly added lens is immediately openable.
    opened = await command_execute(ctx, "board", "crypto")
    assert opened["mode"] == "open" and opened["template"] == "crypto" and "note" not in opened

    removed = await command_execute(ctx, "board templates", "remove crypto")
    assert removed["ok"] is True
    assert not (tpl_dir / "crypto.yaml").exists()


async def test_board_templates_add_rejects_invalid_and_reserved(tmp_path: Path) -> None:
    ctx = _templates_ctx(tmp_path / "tpl")

    bad = tmp_path / "bad.yaml"
    bad.write_text("just a string, not a template mapping", encoding="utf-8")
    invalid = await command_execute(ctx, "board templates", f"add {bad}")
    assert invalid["ok"] is False

    reserved = await command_execute(ctx, "board templates", f"add {bad} --name refresh")
    assert reserved["ok"] is False and "reserved" in reserved["message"].lower()


async def test_board_templates_cannot_remove_builtin(tmp_path: Path) -> None:
    ctx = _templates_ctx(tmp_path / "tpl")
    result = await command_execute(ctx, "board templates", "remove generic")
    assert result["ok"] is False and "builtin" in result["message"].lower()
