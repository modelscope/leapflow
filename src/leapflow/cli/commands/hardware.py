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
    _fail(f"Unknown hw action: {action}")
    return 1


# ── Read / estop plane ─────────────────────────────────────────────────────


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

    from leapflow.hardware.tools import HardwareTools

    # session_id is intentionally empty: reads need no identity and hw_estop is
    # ungated and identity-agnostic, so nothing here should adopt a session that
    # belongs to another client.
    tools = HardwareTools(registry)
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


def _print_usage() -> None:
    print("Usage: leap hw {list|describe|read|status|estop|pause|resume|replay} [--json]")
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
