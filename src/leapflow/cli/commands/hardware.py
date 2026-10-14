# Copyright (c) Alibaba, Inc. and its affiliates.
"""`leap hw` — inspect hardware and intervene in it directly (Phase 1.4).

Two planes share one command group:

* **Read / estop** (``list``/``describe``/``read``/``status``/``estop``) reuse the
  production ``HardwareTools`` handlers against a registry built in this process.
  That registry deliberately runs with persistence and streaming *off*: leapd is
  the single writer of the session DuckDB reading store (Phase 0.5), so a one-shot
  CLI command must never open it for writing, and it must not start a sampling
  loop of its own. Live reads still work — they open a transport on demand — while
  sampled history stays with leapd.

* **pause / resume** control the sampling lifecycle, which only ever runs inside
  leapd. When a healthy daemon is present the command routes through the
  ``hardware.pause`` / ``hardware.resume`` RPCs; without one it fails closed with an
  actionable hint rather than pretending to pause a loop that an in-process command
  never starts.

Every subcommand accepts ``--json`` for machine-readable output and returns a
non-zero exit code whenever the structured result reports ``ok`` is false.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import replace
from typing import Any

from leapflow.config import load_config
from leapflow.daemon.client import DaemonClient, DaemonUnavailableError

logger = logging.getLogger(__name__)

# ── ANSI colors (mirrors the palette used by `leap host`) ────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[1;36m"

# Subcommands that read the device on demand or halt it, served in-process.
_LOCAL_ACTIONS = frozenset({"list", "describe", "read", "status", "estop"})
# Subcommands that steer the daemon-owned sampling loop.
_SAMPLING_ACTIONS = frozenset({"pause", "resume"})
# Subcommands that operate on recorded data, no registry required.
_OFFLINE_ACTIONS = frozenset({"replay"})
# Verification subcommands (in-process, write verified.json).
_VERIFICATION_ACTIONS = frozenset({"verify", "validate"})
# Onboarding subcommands (in-process, template/probe/commission).
_ONBOARDING_ACTIONS = frozenset({"init", "probe", "commission"})


def _ok(msg: str) -> None:
    print(f"  {_GREEN}\u2713{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}\u2717{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}!{_RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")


# ── Registry / daemon discovery (module-level so tests can substitute them) ──


def _build_local_registry(settings: Any) -> Any:
    """Return a loaded registry for in-process reads, or None when hardware is off.

    Persistence and streaming are forced off: this process is not leapd, so it must
    neither open the single-writer reading store nor start a sampling loop (Phase
    0.5). On-demand reads and estop do not depend on either.
    """
    from leapflow.hardware.registry import HardwareRegistry, HardwareSettings

    policy = HardwareSettings.from_settings(settings)
    if not policy.enabled:
        return None
    policy = replace(policy, persist_readings=False, stream_enabled=False)
    registry = HardwareRegistry(policy)
    registry.load()
    return registry


def _discover_daemon(settings: Any) -> Any:
    """Return healthy leapd discovery info with a usable socket, else None."""
    from leapflow.daemon.lifecycle import DaemonInfo

    info = DaemonInfo.discover(settings.runtime_dir)
    if getattr(info, "is_healthy", False) and getattr(info, "sock_path", None) is not None:
        return info
    return None


# ── Entry point ──────────────────────────────────────────────────────────────


def cmd_hardware(args: argparse.Namespace) -> int:
    """Route ``leap hw`` subcommands. Synchronous shell around an async worker."""
    action = getattr(args, "hw_action", None)
    if action is None:
        _print_usage()
        return 1
    try:
        return asyncio.run(_dispatch(args, action))
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        sys.stderr.write("\n\033[2m\u2192 Interrupted\033[0m\n")
        return 130


async def _dispatch(args: argparse.Namespace, action: str) -> int:
    json_mode = bool(getattr(args, "json", False))
    if action in _LOCAL_ACTIONS:
        return await _run_local(action, args, json_mode)
    if action in _SAMPLING_ACTIONS:
        return await _run_sampling_control(action, args, json_mode)
    if action in _OFFLINE_ACTIONS:
        return _run_offline(action, args, json_mode)
    if action == "serve":
        return await _run_serve(args)
    if action in _VERIFICATION_ACTIONS:
        return await _run_verification(action, args, json_mode)
    if action == "permit-key":
        return _run_permit_key(args, json_mode)
    if action in _ONBOARDING_ACTIONS:
        return await _run_onboarding(action, args, json_mode)
    _fail(f"Unknown hw action: {action}")
    return 1


# ── Edge runtime (node side) ─────────────────────────────────────────────


async def _run_serve(args: argparse.Namespace) -> int:
    """Serve this machine's devices as an LHP node, until interrupted."""
    from leapflow.hardware.lhp.protocol import is_valid_device_id
    from leapflow.hardware.lhp.runtime import resolve_endpoint_args, serve_node

    node_id, token = resolve_endpoint_args(
        str(getattr(args, "node_id", "") or ""), str(getattr(args, "token", "") or "")
    )
    if not is_valid_device_id(node_id):
        _fail(
            "serve requires a node id matching [a-z0-9_]+ "
            "(--node-id, or LEAPFLOW_LHP_NODE_ID)"
        )
        return 1
    return await serve_node(
        node_id=node_id,
        host=str(getattr(args, "host", "") or "127.0.0.1"),
        port=int(getattr(args, "port", 0)),
        unix_path=str(getattr(args, "unix_path", "") or ""),
        token=token,
    )


# ── Read / estop plane ─────────────────────────────────────────────────────


async def _register_fleet_read_plane(registry: Any) -> None:
    """Reconcile configured fleet nodes into the in-process read registry.

    ``leap hw list/read`` serves from a registry built in this process rather
    than leapd's, so the configured nodes have to be registered here for
    ``<node>__<device>`` ids to resolve.  Best-effort: an unreachable board
    leaves the host's own devices fully usable.  No heartbeat -- the process
    is short-lived and reads are on-demand.
    """
    from leapflow.hardware.fleet import FleetManager
    from leapflow.hardware.fleet_config import (
        configured_nodes_from_settings,
        register_nodes_safely,
    )

    # Same source the daemon plugin reads: typed Settings (``hardware.fleet_nodes``)
    # as carried on the registry, with the legacy env fallback handled inside
    # fleet_config.  Reading the env directly here left the settings-configured
    # nodes invisible to ``leap hw list`` while leapd saw them.
    fleet_nodes_raw = str(getattr(getattr(registry, "settings", None), "fleet_nodes", "") or "")
    nodes = configured_nodes_from_settings(fleet_nodes_raw)
    if not nodes:
        return
    registered = await register_nodes_safely(FleetManager(registry), nodes)
    if registered:
        logger.info(
            "Fleet: %d configured node(s) registered for in-process reads",
            len(registered),
        )


async def _run_local(action: str, args: argparse.Namespace, json_mode: bool) -> int:
    settings = load_config()
    registry = _build_local_registry(settings)
    if registry is None:
        return _emit_result(
            action,
            {
                "ok": False,
                "code": "hardware_disabled",
                "error": (
                    "Hardware is disabled for this profile. Enable it "
                    "(`leap config set hardware.enabled true`) and declare a device, "
                    "then retry."
                ),
            },
            json_mode,
        )

    # Reconcile configured fleet nodes before any action runs: reads are served
    # in this process, so ``<node>__<device>`` ids must be admitted here too.
    await _register_fleet_read_plane(registry)

    from leapflow.hardware.tools import HardwareTools

    # session_id is intentionally empty: reads need no identity and hw_estop is
    # ungated and identity-agnostic, so nothing here should adopt a session that
    # belongs to another client.
    tools = HardwareTools(registry)
    try:
        if action == "list":
            result = await tools.hw_list()
        elif action == "describe":
            result = await tools.hw_describe(device_id=str(args.device))
        elif action == "read":
            result = await tools.hw_read(
                device_id=str(args.device), channel_id=str(args.channel)
            )
        elif action == "status":
            result = await _collect_status(tools, registry, str(getattr(args, "device", "") or ""))
        elif action == "estop":
            result = await tools.hw_estop(device_id=str(args.device))
        else:  # pragma: no cover - guarded by _dispatch
            result = {"ok": False, "code": "unknown_action", "error": action}
    finally:
        # This process owns the registry it just built, and answering a read claims the
        # device -- a camera or microphone included. Nothing else will release it: the
        # plugin path registers ``close_all`` on its EffectScope, but this registry has no
        # scope, so without this the command could exit leaving a capture process holding
        # the device and its indicator light on. ``close_all`` never raises.
        await registry.close_all()
    return _emit_result(action, result, json_mode)


async def _collect_status(tools: Any, registry: Any, device: str) -> dict[str, Any]:
    """Return one device's status, or a roll-up across every admitted device."""
    if device:
        return await tools.hw_status(device_id=device)
    reports = [await tools.hw_status(device_id=c.device_id) for c in registry.contexts()]
    return {"ok": True, "devices": reports, "count": len(reports)}


# ── Sampling-control plane (pause / resume) ─────────────────────────────────


async def _run_sampling_control(
    action: str, args: argparse.Namespace, json_mode: bool
) -> int:
    settings = load_config()
    device = str(args.device)
    info = _discover_daemon(settings)
    if info is None:
        # In-process mode never samples (Phase 0.5), so there is no loop to steer
        # here. Fail closed with a concrete next step rather than a silent no-op.
        return _emit_result(
            action,
            {
                "ok": False,
                "code": "daemon_required",
                "device": device,
                "error": (
                    "Hardware sampling runs only inside leapd: an in-process command "
                    "never samples, so there is nothing to pause or resume here. Start "
                    "the daemon with `leap daemon start`, then re-run "
                    f"`leap hw {action} {device}`."
                ),
            },
            json_mode,
        )

    client = DaemonClient(info.sock_path)
    try:
        if action == "pause":
            result = await client.hardware_pause(device)
        else:
            result = await client.hardware_resume(device)
    except DaemonUnavailableError as exc:
        result = {
            "ok": False,
            "code": "daemon_error",
            "device": device,
            "error": f"leapd request failed: {exc}",
        }
    return _emit_result(action, result, json_mode)


# ── Output ─────────────────────────────────────────────────────────────────


def _emit_result(action: str, result: dict[str, Any], json_mode: bool) -> int:
    if json_mode:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        _render(action, result)
    return 0 if result.get("ok") else 1


def _render(action: str, result: dict[str, Any]) -> None:
    print(f"{_CYAN}LEAP Hardware \u2014 {action}{_RESET}")
    if not result.get("ok"):
        _fail(str(result.get("error") or result.get("code") or "command failed"))
        return
    renderer = {
        "list": _render_list,
        "describe": _render_describe,
        "read": _render_read,
        "status": _render_status,
        "estop": _render_estop,
        "pause": _render_sampling,
        "resume": _render_sampling,
        "replay": _render_replay,
    }.get(action)
    if renderer is not None:
        renderer(result)


def _render_list(result: dict[str, Any]) -> None:
    devices = result.get("devices") or []
    if not devices:
        _info("No hardware devices admitted.")
        return
    for dev in devices:
        quantities = ", ".join(dev.get("quantities") or []) or "-"
        _ok(f"{dev.get('device_id')} \u2014 {dev.get('display_name') or ''}".rstrip())
        _info(
            f"channels={dev.get('channels', 0)} writable={dev.get('writable', 0)} "
            f"streaming={dev.get('streaming', 0)} verified={dev.get('verified')} "
            f"quantities=[{quantities}]"
        )


def _render_describe(result: dict[str, Any]) -> None:
    _ok(f"{result.get('device_id')} \u2014 {result.get('display_name') or ''}".rstrip())
    _info(f"location={result.get('location')} halt_supported={result.get('halt_supported')}")
    _info(f"writable={result.get('writable_channels')} streaming={result.get('streaming_channels')}")
    for channel in result.get("channels") or []:
        _info(
            f"  \u2022 {channel.get('channel_id')} "
            f"[{channel.get('direction')}] {channel.get('quantity')} "
            f"{channel.get('unit') or ''}".rstrip()
        )


def _render_read(result: dict[str, Any]) -> None:
    reading = result.get("reading") or {}
    _ok(
        f"{reading.get('channel_id', '')}={reading.get('value')} "
        f"{reading.get('unit') or ''}".rstrip()
    )
    if result.get("history"):
        _info(f"history: {result['history']}")


def _render_status(result: dict[str, Any]) -> None:
    if "devices" in result:
        for report in result.get("devices") or []:
            _render_one_status(report)
        return
    _render_one_status(result)


def _render_one_status(report: dict[str, Any]) -> None:
    device = report.get("device_id", "")
    if not report.get("ok"):
        _fail(f"{device}: {report.get('error') or 'status unavailable'}")
        return
    status = report.get("status") or {}
    _ok(f"{device}: connected={status.get('connected')} detail={status.get('detail') or ''}".rstrip())
    for event in report.get("recent_events") or []:
        _info(f"  \u2022 {event.get('kind')} {event.get('channel_id') or ''} {event.get('detail') or ''}".rstrip())


def _render_estop(result: dict[str, Any]) -> None:
    _ok(f"{result.get('device_id')}: halted={result.get('halted')}")


def _render_sampling(result: dict[str, Any]) -> None:
    device = result.get("device", "")
    verb = "paused" if result.get("paused") else "resumed"
    channels = result.get("channels") or []
    _ok(f"{device}: {verb} {len(channels)} channel(s) [scope={result.get('scope', 'daemon')}]")
    for channel in channels:
        _info(f"  \u2022 {channel}")
    for channel in result.get("failed") or []:
        _warn(f"  \u2022 {channel} (failed to resume)")


def _render_replay(result: dict[str, Any]) -> None:
    events = result.get("events") or []
    _ok(f"Replayed {result.get('readings', 0)} readings, produced {len(events)} event(s)")
    for event in events:
        _info(f"  \u2022 {event}")


# ── Offline (recorded-data) plane ─────────────────────────────────────────


def _run_offline(action: str, args: argparse.Namespace, json_mode: bool) -> int:
    if action == "replay":
        return _run_replay(args, json_mode)
    _fail(f"Unknown offline action: {action}")
    return 1


def _run_replay(args: argparse.Namespace, json_mode: bool) -> int:
    from pathlib import Path

    from leapflow.hardware.replay import run_replay

    segment_path = Path(str(args.segment_path))
    if not segment_path.exists():
        return _emit_result(
            "replay",
            {
                "ok": False,
                "code": "file_not_found",
                "error": f"Segment file not found: {segment_path}",
            },
            json_mode,
        )
    events = run_replay(segment_path)
    # Count readings from the file for the summary.
    try:
        reading_count = sum(1 for line in segment_path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        reading_count = 0
    result: dict[str, Any] = {
        "ok": True,
        "segment": str(segment_path),
        "readings": reading_count,
        "events": [event.to_detail() for event in events],
    }
    return _emit_result("replay", result, json_mode)


# ── Verification commands ──────────────────────────────────────────────────────────


async def _run_verification(action: str, args: argparse.Namespace, json_mode: bool) -> int:
    """Handle verify/validate subcommands."""
    import time as _time

    import yaml

    from leapflow.hardware.providers.yaml_provider import declaration_digest

    settings = load_config()
    registry = _build_local_registry(settings)
    if registry is None:
        return _emit_result(action, {"ok": False, "error": "hardware is not enabled"}, json_mode)

    from leapflow.layout import HardwareLayout

    layout = HardwareLayout(settings.profile_dir)
    verified_path = layout.verified_path
    device_id = str(getattr(args, "device_id", "") or "")

    if action == "verify":
        if not device_id:
            return _emit_result("verify", {"ok": False, "error": "device_id is required"}, json_mode)
        context = registry.context(device_id)
        if context is None:
            return _emit_result("verify", {"ok": False, "error": f"unknown device {device_id!r}"}, json_mode)

        digest = declaration_digest(context)
        # Read existing verified.json
        existing: dict[str, Any] = {}
        if verified_path.is_file():
            try:
                raw = yaml.safe_load(verified_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
            except (OSError, yaml.YAMLError):
                pass

        import getpass

        verifier = getpass.getuser()
        existing[device_id] = {
            "verifier": verifier,
            "declaration_digest": digest,
            "verified_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }
        verified_path.parent.mkdir(parents=True, exist_ok=True)
        verified_path.write_text(
            yaml.dump(existing, default_flow_style=False, sort_keys=True),
            encoding="utf-8",
        )
        result = {
            "ok": True,
            "device_id": device_id,
            "verifier": verifier,
            "declaration_digest": digest,
            "verified_path": str(verified_path),
        }
        if not json_mode:
            _ok(f"Verified {device_id} (digest={digest[:12]}...) by {verifier}")
            _info(f"Written to {verified_path}")
        return _emit_result("verify", result, json_mode) if json_mode else 0

    if action == "validate":
        contexts = registry.contexts()
        # Load verified.json
        verified: dict[str, Any] = {}
        if verified_path.is_file():
            try:
                raw = yaml.safe_load(verified_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    verified = raw
            except (OSError, yaml.YAMLError):
                pass

        results: list[dict[str, Any]] = []
        for ctx in contexts:
            if device_id and ctx.device_id != device_id:
                continue
            entry = verified.get(ctx.device_id)
            status = "unverified"
            if entry is None:
                status = "unverified"
            elif isinstance(entry, str):
                status = "legacy_unbound"
            elif isinstance(entry, dict):
                stored = str(entry.get("declaration_digest") or "")
                if not stored:
                    status = "unbound"
                else:
                    current = declaration_digest(ctx)
                    status = "valid" if current == stored else "digest_mismatch"

            item = {
                "device_id": ctx.device_id,
                "status": status,
                "is_verified": ctx.provenance.is_verified,
            }
            results.append(item)
            if not json_mode:
                icon = _ok if status == "valid" else (_warn if status in ("legacy_unbound", "unbound") else _fail)
                icon(f"{ctx.device_id}: {status}")

        all_ok = all(r["status"] == "valid" for r in results)
        return _emit_result(
            "validate",
            {"ok": all_ok, "devices": results},
            json_mode,
        )

    return 1


def _run_permit_key(args: argparse.Namespace, json_mode: bool) -> int:
    """Show the permit issuer's public key and key_id."""
    settings = load_config()
    from leapflow.layout import HardwareLayout

    layout = HardwareLayout(settings.profile_dir)
    key_path = layout.root / "lhp_permit_issuer.key"
    if not key_path.is_file():
        result = {
            "ok": False,
            "error": f"No permit issuer key found at {key_path}. "
            "Start a node with --write-mode permit_required to generate one.",
        }
        if not json_mode:
            _fail(result["error"])
        return 1

    try:
        from leapflow.hardware.lhp.permit import (
            key_id_from_public,
            load_private_key,
            public_key_pem,
        )

        private_key = load_private_key(key_path)
        pub = private_key.public_key()
        kid = key_id_from_public(pub)
        pem = public_key_pem(pub)
        result = {
            "ok": True,
            "key_id": kid,
            "public_key_pem": pem,
            "key_path": str(key_path),
        }
        if not json_mode:
            _ok(f"key_id: {kid}")
            print(pem)
            _info(f"Key file: {key_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        if not json_mode:
            _fail(f"Could not load permit key: {exc}")
        return 1


# ── Onboarding commands (init / probe / commission) ───────────────────────────────


_DEVICE_TEMPLATE = """\
# LeapFlow hardware device declaration (hc.v1)
# Documentation: https://leapflow.dev/hardware/declarations
schema: hc.v1

device_id: "{device_id}"
display_name: "{display_name}"
device_class: sensor           # sensor | robot_arm | mobile_robot | actuator
location: "lab bench"          # physical placement description

transport:
  type: serial                 # serial | realsense | lhp | simulated
  # port: /dev/ttyUSB0
  # baud_rate: 115200

channels:
  - channel_id: sensor.temperature
    quantity: temperature
    unit: degC
    direction: read
    representation: scalar
    # min_value: -40.0
    # max_value: 125.0

  # - channel_id: actuator.motor
  #   quantity: position
  #   unit: rad
  #   direction: write
  #   effect: actuate
  #   min_value: -3.14
  #   max_value: 3.14
  #   max_rate: 1.0
  #   tolerance: 0.01
  #   settling_s: 0.5
  #   reversible: true

# interlocks:
#   - name: emergency_stop
#     condition: "sensor.estop == 1"

# degradation:
#   timeout_s: 5.0
#   policy: halt
"""


async def _run_onboarding(action: str, args: argparse.Namespace, json_mode: bool) -> int:
    """Handle init/probe/commission subcommands."""
    if action == "init":
        return _run_init(args, json_mode)
    if action == "probe":
        return await _run_probe(args, json_mode)
    if action == "commission":
        return await _run_commission(args, json_mode)
    return 1


def _run_init(args: argparse.Namespace, json_mode: bool) -> int:
    """Generate an annotated hc.v1 device declaration template."""
    from pathlib import Path

    device_id = str(getattr(args, "device_id", "") or "my_device")
    display_name = device_id.replace("_", " ").replace(".", " ").title()

    template = _DEVICE_TEMPLATE.format(
        device_id=device_id,
        display_name=display_name,
    )

    settings = load_config()
    profile_layout = getattr(settings, "profile_layout", None)
    if profile_layout is not None:
        devices_dir = profile_layout.hardware.devices_dir
    else:
        devices_dir = Path(".") / "devices"

    out_path = devices_dir / f"{device_id}.yaml"
    if out_path.exists():
        if json_mode:
            return _emit_result("init", {"ok": False, "error": f"{out_path} already exists"}, True)
        _fail(f"{out_path} already exists; edit it directly or remove it first")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template, encoding="utf-8")

    result = {
        "ok": True,
        "device_id": device_id,
        "path": str(out_path),
        "hint": "Edit the template, then run `leap hw probe <device>` to verify connectivity.",
    }
    if not json_mode:
        _ok(f"Generated declaration template: {out_path}")
        _info(f"Edit it, then: leap hw probe {device_id}")
        return 0
    return _emit_result("init", result, True)


async def _run_probe(args: argparse.Namespace, json_mode: bool) -> int:
    """Read-only connectivity and channel readability check."""
    settings = load_config()
    registry = _build_local_registry(settings)
    if registry is None:
        return _emit_result("probe", {"ok": False, "error": "hardware is not enabled"}, json_mode)

    await _register_fleet_read_plane(registry)

    device_id = str(getattr(args, "device_id", "") or "")
    if not device_id:
        return _emit_result("probe", {"ok": False, "error": "device_id is required"}, json_mode)

    context = registry.context(device_id)
    if context is None:
        await registry.close_all()
        return _emit_result(
            "probe", {"ok": False, "error": f"unknown device {device_id!r}"}, json_mode
        )

    # Try to open the transport.
    probe_results: list[dict[str, Any]] = []
    transport_ok = False
    try:
        transport = await registry.transport(device_id)
        status = await transport.status()
        transport_ok = getattr(status, "connected", False)
    except Exception as exc:  # noqa: BLE001
        probe_results.append({"check": "transport", "ok": False, "error": str(exc)})
    else:
        probe_results.append({
            "check": "transport",
            "ok": transport_ok,
            "detail": getattr(status, "detail", ""),
            "halt_supported": getattr(status, "halt_supported", False),
        })

    # Try reading each readable channel.
    for ch in context.channels:
        if not ch.is_readable:
            continue
        try:
            reading = await registry.read(device_id, ch.channel_id)
            probe_results.append({
                "check": f"read:{ch.channel_id}",
                "ok": True,
                "value": getattr(reading, "value", None),
                "unit": ch.unit,
            })
        except Exception as exc:  # noqa: BLE001
            probe_results.append({
                "check": f"read:{ch.channel_id}",
                "ok": False,
                "error": str(exc),
            })

    # Check halt availability.
    if transport_ok:
        try:
            transport = await registry.transport(device_id)
            halt_status = await transport.status()
            probe_results.append({
                "check": "halt_available",
                "ok": getattr(halt_status, "halt_supported", False),
            })
        except Exception:  # noqa: BLE001
            probe_results.append({"check": "halt_available", "ok": False})

    await registry.close_all()

    all_ok = all(r.get("ok") for r in probe_results)
    result = {
        "ok": all_ok,
        "device_id": device_id,
        "checks": probe_results,
        "hint": "All checks passed. Run `leap hw verify` then `leap hw commission` to complete onboarding."
        if all_ok else "Fix the failing checks and re-run `leap hw probe`.",
    }
    if not json_mode:
        for check in probe_results:
            icon = _ok if check.get("ok") else _fail
            detail = check.get("detail") or check.get("value") or check.get("error") or ""
            icon(f"{check['check']}: {detail}")
    return _emit_result("probe", result, json_mode)


async def _run_commission(args: argparse.Namespace, json_mode: bool) -> int:
    """Controlled first-write: dry-run → real write → verify → restore."""
    settings = load_config()
    registry = _build_local_registry(settings)
    if registry is None:
        return _emit_result("commission", {"ok": False, "error": "hardware is not enabled"}, json_mode)

    await _register_fleet_read_plane(registry)

    device_id = str(getattr(args, "device_id", "") or "")
    if not device_id:
        await registry.close_all()
        return _emit_result("commission", {"ok": False, "error": "device_id is required"}, json_mode)

    context = registry.context(device_id)
    if context is None:
        await registry.close_all()
        return _emit_result(
            "commission", {"ok": False, "error": f"unknown device {device_id!r}"}, json_mode
        )

    # Find the first writable actuate channel for the commissioning write.
    target_ch = None
    for ch in context.channels:
        if ch.is_writable and getattr(ch, "effect", "") == "actuate":
            target_ch = ch
            break
    if target_ch is None:
        await registry.close_all()
        return _emit_result(
            "commission",
            {"ok": False, "error": f"device {device_id!r} has no writable actuate channel"},
            json_mode,
        )

    # Use the channel's min_value as a safe commissioning target.
    envelope = getattr(target_ch, "envelope", None)
    safe_value = getattr(envelope, "min_value", None)
    if safe_value is None:
        safe_value = 0.0

    steps: list[dict[str, Any]] = []
    from leapflow.hardware.tools import HardwareTools

    tools = HardwareTools(registry)

    # Step 1: Read baseline.
    try:
        baseline = await registry.read(device_id, target_ch.channel_id)
        baseline_value = getattr(baseline, "value", None)
        steps.append({"step": "baseline_read", "ok": True, "value": baseline_value})
    except Exception as exc:  # noqa: BLE001
        baseline_value = None
        steps.append({"step": "baseline_read", "ok": False, "error": str(exc)})

    # Step 2: Dry-run write.
    dry_result = await tools.hw_write(
        device_id=device_id,
        channel_id=target_ch.channel_id,
        value=safe_value,
        dry_run=True,
    )
    steps.append({"step": "dry_run", "ok": dry_result.get("ok", False), "detail": dry_result})

    if not dry_result.get("ok"):
        await registry.close_all()
        return _emit_result(
            "commission",
            {"ok": False, "device_id": device_id, "steps": steps,
             "error": "Dry-run failed; aborting commissioning."},
            json_mode,
        )

    # Step 3: Real write (minimum-step command).
    write_result = await tools.hw_write(
        device_id=device_id,
        channel_id=target_ch.channel_id,
        value=safe_value,
    )
    steps.append({"step": "write", "ok": write_result.get("ok", False), "detail": write_result})

    # Step 4: Read-back verification.
    try:
        readback = await registry.read(device_id, target_ch.channel_id)
        readback_value = getattr(readback, "value", None)
        steps.append({"step": "readback", "ok": True, "value": readback_value})
    except Exception as exc:  # noqa: BLE001
        readback_value = None
        steps.append({"step": "readback", "ok": False, "error": str(exc)})

    # Step 5: Restore baseline if we have one.
    if baseline_value is not None and write_result.get("ok"):
        restore_result = await tools.hw_write(
            device_id=device_id,
            channel_id=target_ch.channel_id,
            value=baseline_value,
        )
        steps.append({"step": "restore", "ok": restore_result.get("ok", False)})

    await registry.close_all()

    all_ok = write_result.get("ok", False)
    result = {
        "ok": all_ok,
        "device_id": device_id,
        "channel_id": target_ch.channel_id,
        "steps": steps,
        "hint": (
            f"Commission complete. Run `leap hw verify {device_id}` to record trust evidence."
            if all_ok else "Commission failed. Review the steps above."
        ),
    }
    if not json_mode:
        for step in steps:
            icon = _ok if step.get("ok") else _fail
            icon(f"{step['step']}: {step.get('value') or step.get('error') or 'done'}")
    return _emit_result("commission", result, json_mode)


def _print_usage() -> None:
    print("Usage: leap hw {list|describe|read|status|estop|pause|resume|replay|serve|verify|validate|permit-key|init|probe|commission} [--json]")
    print()
    print("Inspect hardware and intervene in it directly.")
    print()
    print("Commands:")
    print("  list                 List admitted hardware devices")
    print("  describe <device>    Show the full reference for one device")
    print("  read <device> <ch>   Read one channel")
    print("  status [device]      Show transport health and recent events")
    print("  estop <device>       Emergency-stop a device (no approval required)")
    print("  pause <device>       Pause daemon sampling for a device")
    print("  resume <device>      Resume daemon sampling for a device")
    print("  replay <path>        Replay a raw NDJSON segment through the event detector")
    print("  serve                Serve this machine as an LHP fabric node (edge runtime)")
    print("  verify <device>      Write a digest-bound verification entry for a device")
    print("  validate [device]    Validate declarations against their verification entries")
    print("  permit-key           Show the permit issuer public key and key_id")
    print("  init [device]        Generate an annotated device declaration template")
    print("  probe <device>       Read-only connectivity and channel check")
    print("  commission <device>  Controlled first-write with dry-run, verify, and restore")
