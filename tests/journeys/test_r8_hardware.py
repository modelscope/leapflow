"""R8 — physical bench end-to-end through a real daemon.

Phases: a simulated device is discovered and described, the sampling loop lands
downsampled windows in the daemon-owned DuckDB, a channel read returns a live
reading, the IC-1 readiness hard-gate is exercised (unready → init/homing →
preview → ready), a setpoint command is put through the real approval chain, and
the LeapBoard hardware panel renders from a wall-clock digest.

The init/calibration phases (IC-4) verify IC-1's HCP readiness gate end-to-end:
the ``setpoint`` channel declares ``requires_interlocks: [device_homed]`` and a
``homed`` configure channel starts ``false``. A write attempt while unhomed is
refused fail-closed (``not_ready``, no approval sought, deterministic repair
instruction); a configure write sets ``homed=true``; a dry-run preview confirms
feasibility without writing; the real actuate then passes through the readiness
gate and the daemon approval chain.

The actuate phase asserts the approval chain is *load-bearing*: the daemon's
``install_gate`` re-binds the hardware plugin's approval gate to the
stream-routed daemon orchestrator (#24 fix), so a hardware write emits an
approval request that ``_drive_with_auto_approval`` resolves.  Asserting both
a successful completion and the presence of an ``approval_request`` event proves
that the write cannot reach the device without passing through the gate.

Replay determinism is the load-bearing design constraint, and two facts shape it:

- Only tool results that are fed back into a *later* provider request enter a
  cassette fingerprint. Anything a streaming channel produces -- the per-sample
  ``sequence``, a window's ``samples`` count -- is a small integer the scrubber
  leaves intact and that differs every run, so it must never reach the model.
- The device declaration, ``hw_list``/``hw_describe`` and a single read of a
  *non-streaming* channel are pure functions of the YAML (``sequence`` is 1,
  ``observed_at`` is scrubbed as an epoch), so those are safe to script.

So the model only ever touches the static ``setpoint`` and ``homed`` channels.
Persistence of the streaming channel's durable windows is verified through the
daemon's board digest (``storage.windows_written``, ``counts.series``), which
reads ``channel_history()`` — the identical data path that populates
hw_read's ``stored_windows`` field.  This avoids cross-process direct-reads of
the daemon-exclusive DuckDB file (2.1 LocalConnectionHolder migration).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests._harness.cassette_proxy import answer, scripted, tool_call
from tests._harness.journey import JourneyFactory
from tests._harness.leapd import await_for

SUBJECT_PATHS = (
    "src/leapflow/hardware/",
    "src/leapflow/dashboard/",
    "src/leapflow/monitor/",
)

# The journey exercises the hardware data/approval/observation wiring across a
# real daemon, not model quality: every model turn is scripted.
LIVE_SIGNAL = False

SESSION = "r8-hardware"
DEVICE_ID = "bench_r8"
SENSOR = "sensor"
SETPOINT = "setpoint"
HOMED = "homed"

# A simulated device: one streaming read channel (drives the persisted windows
# and the board panel), one static read/write actuator (the setpoint), and a
# boolean configure channel (``homed``) that models the readiness gate. The
# setpoint declares ``requires_interlocks: [device_homed]``, so the IC-1 HCP
# readiness hard-gate fires until ``homed`` is set to ``true`` via
# ``hw_configure``. ``verified_by`` is declared so the default
# deny-unverified-writes policy admits writes; a constant sensor value sits
# well inside its envelope, so the sampling loop emits no events that could
# otherwise perturb a cassette.
DEVICE_YAML = f"""\
hc_version: hc.v0
device_id: {DEVICE_ID}
display_name: "R8 Simulated Bench"
vendor: "LeapFlow"
model: "SimBench"
location: "journey-lab"
halt_supported: true
provenance:
  source: declared
  verified_by: "r8-journey"
transport:
  kind: simulated
  config:
    latency_ms: 0.0
    values:
      {SETPOINT}: 50.0
      {HOMED}: false
    waveforms:
      {SENSOR}:
        kind: constant
        value: 21.0
interlocks:
  - interlock_id: device_homed
    channel_id: {HOMED}
    operator: eq
    value: true
    description: "Device must be homed before commanding actuators."
channels:
  - channel_id: {SENSOR}
    direction: read
    quantity: temperature
    unit: C
    effect: read
    sample_rate_hz: 5.0
    envelope:
      declared: true
      min_value: 0.0
      max_value: 100.0
      notes: "ambient sensor"
  - channel_id: {HOMED}
    direction: readwrite
    quantity: state.homed
    unit: bool
    effect: configure
    verify_after_write: false
    envelope:
      declared: true
  - channel_id: {SETPOINT}
    direction: readwrite
    quantity: temperature
    unit: C
    effect: actuate
    verify_after_write: false
    envelope:
      declared: true
      min_value: 0.0
      max_value: 100.0
      max_rate: 1000.0
      settling_time_s: 0.0
      reversible: true
      requires_interlocks:
        - device_homed
"""


async def _drive_with_auto_approval(
    client: Any,
    message: str,
    *,
    session_id: str,
    workspace: str,
) -> list[Any]:
    events: list[Any] = []
    async for event in client.engine_chat(message, session_id=session_id, workspace_root=workspace):
        events.append(event)
        if event.type == "approval_request":
            approval = (event.metadata or {}).get("approval") or {}
            pending_id = str(approval.get("pending_id") or "")
            assert pending_id, f"approval event lacked pending_id: {event.metadata}"
            await client.approval_resolve(pending_id, "allow_once", reason="r8 hardware bench")
    return events


def _completed_ok(events: list[Any], tool_name: str) -> bool:
    """True when *tool_name* finished and reported ``ok`` -- so an ``unknown_device``
    or a refused write, which still emit ``tool_complete``, do not pass."""
    return any(
        event.type == "tool_complete"
        and event.content == tool_name
        and bool((event.metadata or {}).get("ok"))
        for event in events
    )


def _completed(events: list[Any], tool_name: str) -> bool:
    """True when *tool_name* emitted a completion, whatever its verdict."""
    return any(
        event.type == "tool_complete" and event.content == tool_name for event in events
    )


def _failure_code(events: list[Any], tool_name: str) -> str:
    """The ``failure_code`` carried on *tool_name*'s completion, or ``""``."""
    for event in events:
        if event.type == "tool_complete" and event.content == tool_name:
            return str((event.metadata or {}).get("failure_code") or "")
    return ""


def _detail(events: list[Any], tool_name: str) -> Any:
    """The completion metadata for *tool_name* if present, else the event trail."""
    for event in events:
        if event.type == "tool_complete" and event.content == tool_name:
            return event.metadata
    return [event.type for event in events]



@pytest.mark.asyncio
async def test_r8_hardware_bench(journeys: JourneyFactory) -> None:
    devices_dir = Path(tempfile.mkdtemp(prefix="lfj-r8dev-"))
    (devices_dir / f"{DEVICE_ID}.yaml").write_text(DEVICE_YAML, encoding="utf-8")
    try:
        journey = journeys(
            "r8_hardware",
            script=scripted(
                # Phase 1: discover
                tool_call("hw_list"),
                tool_call("hw_describe", device_id=DEVICE_ID),
                answer("Discovered the simulated bench and read its channel limits."),
                # Phase 3: read
                tool_call("hw_read", device_id=DEVICE_ID, channel_id=SETPOINT),
                answer("Read the setpoint channel."),
                # Phase 4a: readiness hard stop (homed=false)
                # The engine hard-stops the turn on a permission hard-stop payload,
                # so the model is never asked for a follow-up answer after this.
                tool_call("hw_actuate", device_id=DEVICE_ID, channel_id=SETPOINT, value=60.0),
                # Phase 4b: init / homing
                tool_call("hw_configure", device_id=DEVICE_ID, channel_id=HOMED, value=True),
                answer("Homed the device; the readiness gate is now satisfied."),
                # Phase 4c: dry-run preview after homing
                tool_call("hw_actuate", device_id=DEVICE_ID, channel_id=SETPOINT, value=55.0, dry_run=True),
                answer("Dry run confirmed the setpoint command is feasible after homing."),
                # Phase 5: actuate with approval (homed=true)
                tool_call("hw_actuate", device_id=DEVICE_ID, channel_id=SETPOINT, value=65.0),
                answer("Commanded the setpoint through the approval path."),
            ),
            deadline_s=90.0,
            max_llm_calls=15,
            max_llm_tokens=300_000,
            requires_scripted_responses=True,
            extra_env={
                "LEAPFLOW_HARDWARE_ENABLED": "1",
                "LEAPFLOW_HARDWARE_DEVICES_DIR": str(devices_dir),
                # Declarations only. The default provider set also enumerates the host
                # running the journey, whose channel count differs per machine -- and
                # hw_list output is part of the recorded conversation, so an ambient
                # second device would invalidate the cassette on every new runner.
                "LEAPFLOW_HARDWARE_PROVIDERS": "yaml",
                # Memory prefetch injects a "## Recent Context" block into the
                # system prompt from the profile's signal store, and that store
                # picks up ambient desktop events (a focus switch to loginwindow)
                # and conversation echoes whose top-k ordering is timing- and
                # environment-dependent. That block reaches the model, so it
                # enters the cassette fingerprint and makes replay miss when the
                # ambient signal differs from the seed run. Disabled here for the
                # same reason copilot is: this journey cassettes the foreground
                # hardware contract, not the memory layer.
                "LEAPFLOW_MEMORY_INTEGRATION_ENABLED": "0",
                # The write path is gated by the real approval orchestrator here;
                # describe-before-write would be a second, unrelated gate, so it is
                # taken out of the way to keep phase 4 about approval alone.
                "LEAPFLOW_HARDWARE_REQUIRE_DESCRIBE": "0",
                # A one-second window keeps the persisted-history phases inside the
                # journey's wall-clock budget.
                "LEAPFLOW_HARDWARE_DOWNSAMPLE_INTERVAL_S": "1",
            },
        )
        workspace = journey.workspace("bench")
        client = journey.client(timeout_s=120.0)

        with journey.phase("discover: list and describe the simulated bench"):
            events = await _drive_with_auto_approval(
                client,
                "List the connected hardware devices, then describe bench_r8 in full.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            assert _completed_ok(events, "hw_list"), _detail(events, "hw_list")
            # A successful describe proves the device and its channels were admitted:
            # an unknown device would complete with ok=False instead.
            assert _completed_ok(events, "hw_describe"), _detail(events, "hw_describe")

        with journey.phase("sample: wait for the sampling loop to land windows"):
            # After 2.1 (ReadingStore → LocalConnectionHolder) the daemon holds an
            # exclusive connection to instrument.duckdb, so the test process can no
            # longer open it directly.  Window persistence is verified later in the
            # board phase through the digest, which reads channel_history() — the
            # identical ReadingStore query that hw_read uses for stored_windows.
            # Sleep here to let the sampling loop (5 Hz, 1 s downsample) accumulate
            # at least one window before the read phase runs.
            await asyncio.sleep(2.0)

        with journey.phase("read: setpoint value confirms tool pipeline"):
            events = await _drive_with_auto_approval(
                client,
                "Read the current value of the setpoint channel on bench_r8.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            assert _completed_ok(events, "hw_read"), _detail(events, "hw_read")
            # Durable window persistence for the streaming sensor was already
            # verified in the sample phase via board digest (storage.windows_written
            # and counts.series).  A streaming read cannot be scripted because its
            # per-sample sequence/sample counts are not replay-stable.

        # ── IC-4: init / calibration readiness gate ──────────────────────

        with journey.phase("readiness: unready write is hard-stopped before approval"):
            events = await _drive_with_auto_approval(
                client,
                "Actuate the bench_r8 setpoint to 60.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            # The write path checks the device_homed interlock first. With
            # homed=false, hw_actuate is refused fail-closed before consent is
            # sought — no approval request is emitted, nothing reaches the device.
            assert _completed(events, "hw_actuate"), [e.type for e in events]
            assert _failure_code(events, "hw_actuate") == "not_ready", _detail(
                events, "hw_actuate"
            )
            detail = _detail(events, "hw_actuate")
            # The readiness hard-stop carries the IC-1 failure class and blocks
            # approval so the engine terminates the turn.
            assert detail.get("failure_class") == "device_not_ready", detail
            assert detail.get("blocks_approval") is True, detail
            # No approval was sought — the hard stop fires before consent.
            assert not any(e.type == "approval_request" for e in events), (
                "readiness hard stop must block before approval is sought"
            )

        with journey.phase("init: homing satisfies the readiness gate"):
            events = await _drive_with_auto_approval(
                client,
                f"Configure the {DEVICE_ID} {HOMED} channel to true to complete homing.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            assert _completed_ok(events, "hw_configure"), _detail(events, "hw_configure")

        with journey.phase("preview: dry run confirms feasibility without writing"):
            events = await _drive_with_auto_approval(
                client,
                f"Preview an actuate of {DEVICE_ID} setpoint to 55, dry run only.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            # The dry run returns ok=True because the homed interlock now holds,
            # but never reaches the approval gate or the device: ok with no
            # approval_request proves the dry-run short-circuit.
            assert _completed_ok(events, "hw_actuate"), _detail(events, "hw_actuate")
            assert not any(e.type == "approval_request" for e in events), (
                "a dry run must return before consent is sought"
            )

        with journey.phase("actuate: setpoint write is routed through daemon approval"):
            events = await _drive_with_auto_approval(
                client,
                "Actuate the bench_r8 setpoint to 65.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            # With #24 fixed, the daemon's install_gate re-binds the hardware
            # plugin's approval gate to the stream-routed daemon orchestrator.
            # _drive_with_auto_approval auto-approves the resulting consent
            # prompt, so the write reaches the simulated device and succeeds.
            # A value that bypassed the gate entirely would also complete ok but
            # would NOT emit an approval_request event; asserting both proves
            # the chain is load-bearing.
            assert _completed_ok(events, "hw_actuate"), _detail(events, "hw_actuate")
            approval_events = [
                e for e in events if e.type == "approval_request"
            ]
            assert approval_events, (
                "hw_actuate must route through the daemon approval gate, "
                "which emits an approval_request event"
            )

        with journey.phase("board: hardware panel renders from a wall-clock digest"):
            watches = await client.watch_list()
            hardware_watches = [w for w in watches if w.get("domain") == "hardware"]
            assert hardware_watches, [w.get("domain") for w in watches]
            watch_id = str(hardware_watches[0].get("watch_id") or "")
            assert watch_id, hardware_watches[0]

            async def _hardware_finding() -> dict[str, Any] | None:
                await client.watch_refresh(watch_id)
                findings = await client.watch_findings(watch_id=watch_id, limit=10)
                return next((f for f in findings if f.get("domain") == "hardware"), None)

            finding = await await_for(
                _hardware_finding, timeout_s=15.0, interval_s=0.5, what="hardware board finding"
            )
            payload = finding.get("payload") or {}
            # The board renderer contract: the digest declares its clock, and every
            # timestamp in it is wall-clock. The panel would misplot on any other.
            assert payload.get("clock") == "wall", payload
            assert int((payload.get("counts") or {}).get("devices") or 0) >= 1, payload
            # Persistence verification: storage.windows_written > 0 proves the
            # daemon-owned ReadingStore wrote windows to DuckDB.  counts.series > 0
            # proves channel_history() returned data — the identical query that
            # hw_read uses for its stored_windows return field.  Together these
            # replace the former cross-process DuckDB direct-read in sample/read.
            storage = payload.get("storage") or {}
            assert int(storage.get("windows_written") or 0) > 0, storage
            counts = payload.get("counts") or {}
            assert int(counts.get("series") or 0) >= 1, counts

        journey.finish()
    finally:
        shutil.rmtree(devices_dir, ignore_errors=True)
