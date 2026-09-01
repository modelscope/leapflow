"""`leap hw` — CLI subcommands and pause/resume RPC (Phase 1.4).

Two planes are exercised separately, because they route differently:

* **Read / estop** run in-process against a real ``HardwareRegistry`` built with
  the production ``HardwareTools`` handlers. The registry is forced non-persistent
  and non-streaming so a one-shot command never opens the single-writer reading
  store nor starts a sampling loop it does not own (Phase 0.5).

* **pause / resume** steer the daemon-owned sampling loop. Without a healthy
  daemon the command fails closed; with one it routes through the
  ``hardware.pause`` / ``hardware.resume`` RPCs. The daemon-side core is unit
  tested directly against a lightweight registry, and the service methods are
  driven through their fail-closed branches.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import make_settings

from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    TransportRef,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings


# ── Fixtures / helpers ───────────────────────────────────────────────────────


def _rig_context() -> HardwareContext:
    """A single-device bench with one readable channel over a mock transport."""
    return HardwareContext(
        device_id="rig",
        hc_version=HC_VERSION,
        display_name="Bench rig",
        location="bench-1",
        halt_supported=True,
        transport=TransportRef(
            kind="mock",
            config={"values": {"temp": 21.5}, "halt_supported": True},
        ),
        channels=(
            Channel(
                channel_id="temp",
                direction=Direction.READ.value,
                quantity="temperature.ambient",
                unit="celsius",
                envelope=Envelope(declared=True),
            ),
        ),
        provenance=ContextProvenance(verified_by="jason"),
    )


def _loaded_registry() -> HardwareRegistry:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, require_describe_before_write=False),
        providers=[_StaticProvider(_rig_context())],
    )
    registry.load()
    return registry


class _StaticProvider:
    """Hands a fixed set of declarations to the registry, no discovery I/O."""

    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


def _ns(action: str, **kwargs: Any) -> argparse.Namespace:
    kwargs.setdefault("json", True)
    return argparse.Namespace(hw_action=action, **kwargs)


def _install_local_registry(monkeypatch, tmp_path, registry: Any) -> None:
    """Point the read plane at *registry* without touching real config or disk."""
    import leapflow.cli.commands.hardware as hardware_module

    monkeypatch.setattr(hardware_module, "load_config", lambda: make_settings(str(tmp_path)))
    monkeypatch.setattr(hardware_module, "_build_local_registry", lambda settings: registry)


def _json_out(capsys) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


# ════════════════════════════════════════════════════════════════
# Read plane: dispatch + output through the production handlers
# ════════════════════════════════════════════════════════════════


def test_hw_list_reports_admitted_devices(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    assert cmd_hardware(_ns("list")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["devices"][0]["device_id"] == "rig"


def test_hw_describe_returns_full_reference(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    assert cmd_hardware(_ns("describe", device="rig")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["device_id"] == "rig"
    assert any(ch["channel_id"] == "temp" for ch in payload["channels"])


def test_hw_read_reads_one_channel_on_demand(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    assert cmd_hardware(_ns("read", device="rig", channel="temp")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["reading"]["value"] == 21.5


def test_hw_status_rolls_up_when_device_omitted(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    # Empty device string is the "roll up every device" form, so the CLI reports
    # a list rather than one device's status.
    assert cmd_hardware(_ns("status", device="")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["devices"][0]["device_id"] == "rig"


def test_hw_status_targets_one_device(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    assert cmd_hardware(_ns("status", device="rig")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["device_id"] == "rig"


def test_hw_estop_halts_the_device(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    assert cmd_hardware(_ns("estop", device="rig")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["halted"] is True


def test_hw_read_unknown_device_is_nonzero(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    _install_local_registry(monkeypatch, tmp_path, _loaded_registry())

    # An unknown device is refused by the handler; the CLI mirrors that as a
    # non-zero exit so scripts can branch on it.
    assert cmd_hardware(_ns("read", device="ghost", channel="temp")) == 1
    payload = _json_out(capsys)
    assert payload["ok"] is False


def test_hw_disabled_hardware_fails_closed(monkeypatch, tmp_path, capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    # A None registry stands in for "hardware disabled for this profile".
    _install_local_registry(monkeypatch, tmp_path, None)

    assert cmd_hardware(_ns("list")) == 1
    payload = _json_out(capsys)
    assert payload["ok"] is False
    assert payload["code"] == "hardware_disabled"


def test_hw_no_action_prints_usage(capsys) -> None:
    from leapflow.cli.commands.hardware import cmd_hardware

    assert cmd_hardware(argparse.Namespace(hw_action=None)) == 1
    out = capsys.readouterr().out
    assert "Usage: leap hw" in out


# ── Phase 0.5 invariant: the in-process registry never persists or samples ──


def test_build_local_registry_forces_persistence_and_streaming_off(monkeypatch) -> None:
    import leapflow.hardware.registry as registry_module
    from leapflow.cli.commands.hardware import _build_local_registry

    captured: dict[str, Any] = {}

    class _CapturingRegistry:
        def __init__(self, policy: Any, *args: Any, **kwargs: Any) -> None:
            captured["policy"] = policy

        def load(self) -> None:
            captured["loaded"] = True

    # from_settings would normally read the profile; pin it to an enabled policy
    # that *does* persist and stream, so the override is what we observe.
    monkeypatch.setattr(
        registry_module.HardwareSettings,
        "from_settings",
        classmethod(lambda cls, settings: cls(enabled=True, stream_enabled=True, persist_readings=True)),
    )
    monkeypatch.setattr(registry_module, "HardwareRegistry", _CapturingRegistry)

    registry = _build_local_registry(object())

    assert registry is not None
    assert captured["loaded"] is True
    assert captured["policy"].persist_readings is False
    assert captured["policy"].stream_enabled is False


def test_build_local_registry_returns_none_when_disabled(monkeypatch) -> None:
    import leapflow.hardware.registry as registry_module
    from leapflow.cli.commands.hardware import _build_local_registry

    monkeypatch.setattr(
        registry_module.HardwareSettings,
        "from_settings",
        classmethod(lambda cls, settings: cls(enabled=False)),
    )

    assert _build_local_registry(object()) is None


# ════════════════════════════════════════════════════════════════
# Sampling-control plane: pause / resume route over RPC
# ════════════════════════════════════════════════════════════════


class _FakeClient:
    """Stands in for DaemonClient, recording the RPC the CLI dispatched."""

    last: dict[str, Any] = {}

    def __init__(self, sock_path: Any) -> None:
        _FakeClient.last = {"sock_path": sock_path}

    async def hardware_pause(self, device: str) -> dict[str, Any]:
        _FakeClient.last["method"] = "hardware.pause"
        _FakeClient.last["device"] = device
        return {"ok": True, "device": device, "paused": True, "channels": ["hw:rig:temp"], "scope": "daemon"}

    async def hardware_resume(self, device: str) -> dict[str, Any]:
        _FakeClient.last["method"] = "hardware.resume"
        _FakeClient.last["device"] = device
        return {"ok": True, "device": device, "paused": False, "channels": ["hw:rig:temp"], "failed": [], "scope": "daemon"}


def test_hw_pause_routes_to_daemon_rpc(monkeypatch, tmp_path, capsys) -> None:
    import leapflow.cli.commands.hardware as hardware_module
    from leapflow.cli.commands.hardware import cmd_hardware

    monkeypatch.setattr(hardware_module, "load_config", lambda: make_settings(str(tmp_path)))
    monkeypatch.setattr(hardware_module, "_discover_daemon", lambda settings: SimpleNamespace(sock_path="/tmp/leapd.sock"))
    monkeypatch.setattr(hardware_module, "DaemonClient", _FakeClient)

    assert cmd_hardware(_ns("pause", device="rig")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["paused"] is True
    assert _FakeClient.last["method"] == "hardware.pause"
    assert _FakeClient.last["device"] == "rig"


def test_hw_resume_routes_to_daemon_rpc(monkeypatch, tmp_path, capsys) -> None:
    import leapflow.cli.commands.hardware as hardware_module
    from leapflow.cli.commands.hardware import cmd_hardware

    monkeypatch.setattr(hardware_module, "load_config", lambda: make_settings(str(tmp_path)))
    monkeypatch.setattr(hardware_module, "_discover_daemon", lambda settings: SimpleNamespace(sock_path="/tmp/leapd.sock"))
    monkeypatch.setattr(hardware_module, "DaemonClient", _FakeClient)

    assert cmd_hardware(_ns("resume", device="rig")) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["paused"] is False
    assert _FakeClient.last["method"] == "hardware.resume"


def test_hw_pause_without_daemon_fails_closed(monkeypatch, tmp_path, capsys) -> None:
    import leapflow.cli.commands.hardware as hardware_module
    from leapflow.cli.commands.hardware import cmd_hardware

    monkeypatch.setattr(hardware_module, "load_config", lambda: make_settings(str(tmp_path)))
    monkeypatch.setattr(hardware_module, "_discover_daemon", lambda settings: None)

    assert cmd_hardware(_ns("pause", device="rig")) == 1
    payload = _json_out(capsys)
    assert payload["ok"] is False
    assert payload["code"] == "daemon_required"
    assert "leap daemon start" in payload["error"]


def test_hw_pause_daemon_error_is_reported(monkeypatch, tmp_path, capsys) -> None:
    import leapflow.cli.commands.hardware as hardware_module
    from leapflow.cli.commands.hardware import cmd_hardware
    from leapflow.daemon.client import DaemonUnavailableError

    class _BrokenClient:
        def __init__(self, sock_path: Any) -> None:
            pass

        async def hardware_pause(self, device: str) -> dict[str, Any]:
            raise DaemonUnavailableError("socket gone")

    monkeypatch.setattr(hardware_module, "load_config", lambda: make_settings(str(tmp_path)))
    monkeypatch.setattr(hardware_module, "_discover_daemon", lambda settings: SimpleNamespace(sock_path="/tmp/leapd.sock"))
    monkeypatch.setattr(hardware_module, "DaemonClient", _BrokenClient)

    assert cmd_hardware(_ns("pause", device="rig")) == 1
    payload = _json_out(capsys)
    assert payload["ok"] is False
    assert payload["code"] == "daemon_error"


# ════════════════════════════════════════════════════════════════
# Daemon side: pause/resume core + service fail-closed branches
# ════════════════════════════════════════════════════════════════


class _FakeSource:
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self.stopped = False
        self.started_with: Any = "unset"

    async def stop(self) -> None:
        self.stopped = True

    async def start(self, emit: Any) -> None:
        self.started_with = emit


class _FakeRegistry:
    """A registry surface with just what the sampling-control helpers touch."""

    def __init__(self, device_ids: list[str], sources: list[_FakeSource]) -> None:
        self._ctxs = {d: SimpleNamespace(device_id=d) for d in device_ids}
        self._sources = sources
        self._event_emitter = object()

    def context(self, device_id: str) -> Any:
        return self._ctxs.get(device_id)

    def contexts(self) -> tuple[Any, ...]:
        return tuple(self._ctxs.values())

    def stream_sources(self) -> tuple[_FakeSource, ...]:
        return tuple(self._sources)


@pytest.mark.asyncio
async def test_pause_stops_only_matching_device_sources() -> None:
    from leapflow.daemon.service import pause_hardware_sampling

    rig = _FakeSource("hw:rig:temp")
    other = _FakeSource("hw:other:flow")
    registry = _FakeRegistry(["rig", "other"], [rig, other])

    result = await pause_hardware_sampling(registry, "rig")

    assert result["ok"] is True
    assert result["paused"] is True
    assert result["channels"] == ["hw:rig:temp"]
    assert rig.stopped is True
    # A device prefix must isolate one bench; its neighbour keeps sampling.
    assert other.stopped is False


@pytest.mark.asyncio
async def test_resume_restarts_with_shared_emitter() -> None:
    from leapflow.daemon.service import resume_hardware_sampling

    rig = _FakeSource("hw:rig:temp")
    registry = _FakeRegistry(["rig"], [rig])
    emit = object()

    result = await resume_hardware_sampling(registry, "rig", emit=emit)

    assert result["ok"] is True
    assert result["paused"] is False
    assert result["channels"] == ["hw:rig:temp"]
    assert result["failed"] == []
    assert rig.started_with is emit


@pytest.mark.asyncio
async def test_resume_isolates_a_failing_source() -> None:
    from leapflow.daemon.service import resume_hardware_sampling

    class _FlakySource(_FakeSource):
        async def start(self, emit: Any) -> None:
            raise RuntimeError("port busy")

    good = _FakeSource("hw:rig:temp")
    bad = _FlakySource("hw:rig:pressure")
    registry = _FakeRegistry(["rig"], [good, bad])

    result = await resume_hardware_sampling(registry, "rig", emit=object())

    assert result["ok"] is True
    assert result["channels"] == ["hw:rig:temp"]
    assert result["failed"] == ["hw:rig:pressure"]


@pytest.mark.asyncio
async def test_pause_missing_device_fails_closed() -> None:
    from leapflow.daemon.service import pause_hardware_sampling

    registry = _FakeRegistry(["rig"], [])

    result = await pause_hardware_sampling(registry, "")

    assert result["ok"] is False
    assert result["code"] == "missing_device"


@pytest.mark.asyncio
async def test_pause_unknown_device_lists_admitted() -> None:
    from leapflow.daemon.service import pause_hardware_sampling

    registry = _FakeRegistry(["rig"], [])

    result = await pause_hardware_sampling(registry, "ghost")

    assert result["ok"] is False
    assert result["code"] == "unknown_device"
    assert result["admitted"] == ["rig"]


# ── Service methods: borrow the unbound methods; they only read self._ctx ──


def _service_with_ctx(ctx: Any) -> Any:
    from leapflow.daemon.service import RuntimeLeapService

    return SimpleNamespace(
        _ctx=ctx,
        hardware_pause=RuntimeLeapService.hardware_pause.__get__(SimpleNamespace(_ctx=ctx)),
        hardware_resume=RuntimeLeapService.hardware_resume.__get__(SimpleNamespace(_ctx=ctx)),
    )


@pytest.mark.asyncio
async def test_service_pause_without_runtime_fails_closed() -> None:
    svc = _service_with_ctx(None)

    result = await svc.hardware_pause("rig")

    assert result["ok"] is False
    assert result["code"] == "runtime_unavailable"


@pytest.mark.asyncio
async def test_service_pause_without_hardware_fails_closed() -> None:
    svc = _service_with_ctx(SimpleNamespace(_hardware_registry=None))

    result = await svc.hardware_pause("rig")

    assert result["ok"] is False
    assert result["code"] == "hardware_disabled"


@pytest.mark.asyncio
async def test_service_pause_delegates_to_registry() -> None:
    rig = _FakeSource("hw:rig:temp")
    registry = _FakeRegistry(["rig"], [rig])
    svc = _service_with_ctx(SimpleNamespace(_hardware_registry=registry))

    result = await svc.hardware_pause("rig")

    assert result["ok"] is True
    assert rig.stopped is True


@pytest.mark.asyncio
async def test_service_resume_uses_registry_emitter() -> None:
    rig = _FakeSource("hw:rig:temp")
    registry = _FakeRegistry(["rig"], [rig])
    svc = _service_with_ctx(SimpleNamespace(_hardware_registry=registry))

    result = await svc.hardware_resume("rig")

    assert result["ok"] is True
    # Resume reuses the emitter the daemon installed at startup.
    assert rig.started_with is registry._event_emitter


# ════════════════════════════════════════════════════════════════
# Wiring: RPC registry + client wrappers
# ════════════════════════════════════════════════════════════════


def test_hardware_rpc_methods_are_registered() -> None:
    from leapflow.daemon.protocol import METHOD_REGISTRY

    assert METHOD_REGISTRY["hardware.pause"] == "hardware_pause"
    assert METHOD_REGISTRY["hardware.resume"] == "hardware_resume"


@pytest.mark.asyncio
async def test_client_wrappers_pass_device_param() -> None:
    from leapflow.daemon.client import DaemonClient

    calls: list[tuple[str, dict[str, Any]]] = []

    client = DaemonClient.__new__(DaemonClient)

    async def fake_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append((method, params or {}))
        return {"ok": True}

    client.request = fake_request  # type: ignore[assignment]

    await client.hardware_pause("rig")
    await client.hardware_resume("rig")

    assert calls == [
        ("hardware.pause", {"device": "rig"}),
        ("hardware.resume", {"device": "rig"}),
    ]
