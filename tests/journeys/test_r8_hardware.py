# Copyright (c) Alibaba, Inc. and its affiliates.
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

The model touches static ``setpoint``/``homed`` channels and one simulated
multi-candidate sequence. Its selected digest-bound plan is also checked directly
on the node, independently of the model-facing result budget.
Persistence of the streaming channel's durable windows is verified through the
daemon's board digest (``storage.windows_written``, ``counts.series``), which
reads ``channel_history()`` — the identical data path that populates
hw_read's ``stored_windows`` field.  This avoids cross-process direct-reads of
the daemon-exclusive DuckDB file (2.1 LocalConnectionHolder migration).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import socket
import time
from contextlib import AsyncExitStack
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests._harness.cassette_proxy import ScriptedTurn, answer, scripted, tool_call
from tests._harness.journey import Journey, JourneyFactory
from tests._harness.leapd import await_for
from tests._harness.planning import placement
from tests._harness.skill_stubs import ARM_ID, SEQUENCE, WORKCELL_ID, move_by
from leapflow.hardware.lhp.server import LhpNodeServer, WRITE_MODE_TRUSTED_V1

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


async def _qualified_provider(journey: Journey, stack: AsyncExitStack, root: Path) -> Any:
    """An offline deployment receipt, real isolated provider and owning node registry.

    The receipt represents synthetic qualification evidence, not live/native parity.
    All provider traffic shares the journey proxy's unchanged aggregate budget.
    """
    from leapflow.config import Settings
    from leapflow.hardware.context import Channel, ContextProvenance, HardwareContext, TransportRef
    from leapflow.hardware.model_contract import ModelInfo
    from leapflow.hardware.model_qualification import deployment_action, qualification_binding
    from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
    from leapflow.hardware.verification import EvidenceBundle
    from leapflow.layout import build_layout
    from leapflow.llm.services.composition import model_transport_factories
    from leapflow.llm.services.config import parse_services
    from leapflow.llm.services.runtime import ModelServiceRuntime
    from leapflow.security.approval import ApprovalDecision
    from leapflow.security.grants import ApprovalAuditLog
    from leapflow.security.orchestrator import ApprovalOrchestrator

    info = ModelInfo("grounding", "r8-fixture", "1", "sha256:r8-fixture", "r8-instance")
    spec = parse_services({"native": {
        "schema_version": 1, "provider": "openai", "base_url": journey.proxy.base_url,
        "model": "fixture-native", "task": "grounding", "auth": {"mode": "none"},
        "processing_location": "local", "identity": {"mode": "server_reported"},
        "model_info": info.to_dict(), "deployment": {
            "engine_version": "offline-fixture-1", "adapter_digest": "sha256:r8-adapter",
            "qualification_ref": "r8-qualification"},
    }})[0]
    layout = build_layout(root)
    profile = layout.ensure(profile_id="default")
    settings = Settings("", "", "", 0, True, root / "signals.duckdb", "INFO",
                        llm_services=(spec,), layout=layout, profile_layout=profile,
                        profile_manifest=profile.load_manifest(), workspace_root=root)
    owner = ModelServiceRuntime.from_settings(settings)
    registry = HardwareRegistry(HardwareSettings(
        enabled=True, stream_enabled=False, instrument_db_path=str(root / "instrument.duckdb")),
        providers=(), transport_factories=model_transport_factories(owner, settings, lambda: registry.evidence_store))
    registry.scope.async_effect(owner.aclose)
    stack.push_async_callback(registry.close_all)
    context = HardwareContext(
        device_id="native_model", display_name="R8 Qualified Fixture", halt_supported=False,
        transport=TransportRef(kind="model_service", config={
            "backend": "provider", "service": "native", "task": "grounding"}),
        channels=(Channel(channel_id="model.info", direction="read", quantity="model_metadata"),),
        provenance=ContextProvenance(verified_by="offline-qualification-fixture"))
    registry.register_external_contexts("offline_model", (context,))
    assert registry.load().admitted == ("native_model",)
    audit = ApprovalAuditLog(profile.approval.audit_path)

    class Operator:
        async def request_approval(self, request: Any) -> ApprovalDecision:
            assert request.action.kind == "device.configure"
            return ApprovalDecision.ALLOW_ONCE

    action = deployment_action(qualification_binding(spec, info))
    assert (await ApprovalOrchestrator(Operator(), audit=audit).evaluate(action)).user_consent
    await registry.evidence_store.record(EvidenceBundle(
        "r8-qualification", "model_qualification", "native", time.time(), None, None, (),
        metadata={"model_qualification": {
            "schema_version": 1, "binding": qualification_binding(spec, info),
            "expires_at": time.time() + 120, "revoked": False, "identity_insertion": "runtime",
            "native_parity": True, "approval_action_id": action.action_id}}))
    transport = await registry.transport("native_model")
    result = await transport.infer({"image": base64.b64encode(b"\xff\xd8r8-fixture").decode(), "queries": ["cup"]})
    assert result.detections == () and result.model == info.identity
    assert transport.model_info.details["service_basis"]["generation"] == 1
    return registry


@pytest.mark.asyncio
async def test_r8_hardware_bench(journeys: JourneyFactory) -> None:
    devices_dir = Path(tempfile.mkdtemp(prefix="lfj-r8dev-"))
    (devices_dir / f"{DEVICE_ID}.yaml").write_text(DEVICE_YAML, encoding="utf-8")
    stack = AsyncExitStack()
    # Reserve a not-yet-listening loopback endpoint. The daemon keeps it as a
    # pending fleet node until the final phase; no specialist traffic escapes
    # the journey's time/call/token accounting during setup.
    reserved = socket.socket()
    reserved.bind(("127.0.0.1", 0))
    native_port = reserved.getsockname()[1]
    stack.callback(reserved.close)
    try:
        registry, runtime = await stack.enter_async_context(placement(devices_dir / "planning"))
        # Build the declaration-only advisory before the journey's fixed transport budget.
        # Cold catalog transport/lifecycle is covered by test_production_planning.
        await runtime.list_async(WORKCELL_ID)
        node = LhpNodeServer(registry, node_id="r8_planning", skill_runtime=runtime,
                             write_mode=WRITE_MODE_TRUSTED_V1)
        host, port = await node.start(host="127.0.0.1", port=0)
        stack.push_async_callback(node.stop)
        # Startup fetches the actual socket declaration; keep its owner loop responsive.
        journey = await asyncio.to_thread(lambda: journeys(
            "r8_hardware",
            script=scripted(
                # Phase 1: discover
                ScriptedTurn(tool_calls=(
                    {"name": "hw_list", "arguments": {}},
                    {"name": "hw_describe", "arguments": {"device_id": DEVICE_ID}},
                    {"name": "hw_read", "arguments": {"device_id": DEVICE_ID, "channel_id": SETPOINT}},
                )),
                answer("Discovered the bench, its channel limits and current setpoint."),
                # Phase 4a: readiness hard stop (homed=false)
                # The engine hard-stops the turn on a permission hard-stop payload,
                # so the model is never asked for a follow-up answer after this.
                tool_call("hw_actuate", device_id=DEVICE_ID, channel_id=SETPOINT, value=60.0),
                # Phase 4b: init / homing
                tool_call("hw_configure", device_id=DEVICE_ID, channel_id=HOMED, value=True),
                answer("Homed the device; the readiness gate is now satisfied."),
                # Phase 4c uses the existing board/daemon dry-run RPC without an LLM round.
                # Phase 5: actuate with approval (homed=true)
                tool_call("hw_actuate", device_id=DEVICE_ID, channel_id=SETPOINT, value=65.0),
                answer("Commanded the setpoint through the approval path."),
                # One response supplies task-equivalent programs ending 5 mm above start.
                tool_call("hw_skill_run", workcell=f"r8_planning/{WORKCELL_ID}", skill=SEQUENCE,
                          candidates=[
                              {"id": "detour", "args": {"steps": [
                                  {"op": "move", "relative": {"dz": 10}},
                                  {"op": "move", "relative": {"dz": -5}}]}},
                              {"id": "direct", "args": move_by(5)},
                              {"id": "outside", "args": {"steps": [
                                  {"op": "move", "relative": {"dz": 1000}},
                                  {"op": "move", "relative": {"dz": -995}}]}},
                          ], selection={"order": ["tcp_distance_mm"]}),
                answer("Executed exactly the selected simulated sequence through approval."),
                # Qualified specialist admission (three describes), then describe + infer.
                *(answer(json.dumps({"model_info": {
                    "task": "grounding", "name": "r8-fixture", "version": "1",
                    "checksum": "sha256:r8-fixture", "instance": "r8-instance"}})) for _ in range(4)),
                answer(json.dumps({"detections": [], "infer_ms": 1, "model": {
                    "name": "r8-fixture", "version": "1", "checksum": "sha256:r8-fixture",
                    "instance": "r8-instance"}})),
                # A concurrent fleet health probe can consume the one remaining call.
                answer(json.dumps({"model_info": {
                    "task": "grounding", "name": "r8-fixture", "version": "1",
                    "checksum": "sha256:r8-fixture", "instance": "r8-instance"}})),
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
                "LEAPFLOW_HARDWARE_FLEET_NODES": json.dumps([
                    {"node_id": "r8_planning", "endpoint": f"tcp://{host}:{port}"},
                    {"node_id": "r8_native", "endpoint": f"tcp://127.0.0.1:{native_port}"}
                ]),
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
        ))
        workspace = journey.workspace("bench")
        client = journey.client(timeout_s=120.0)

        with journey.phase("discover: list and describe the simulated bench"):
            events = await _drive_with_auto_approval(
                client,
                "List the connected hardware devices, describe bench_r8, and read its setpoint channel.",
                session_id=SESSION,
                workspace=str(workspace),
            )
            assert _completed_ok(events, "hw_list"), _detail(events, "hw_list")
            # A successful describe proves the device and its channels were admitted:
            # an unknown device would complete with ok=False instead.
            assert _completed_ok(events, "hw_describe"), _detail(events, "hw_describe")
            assert _completed_ok(events, "hw_read"), _detail(events, "hw_read")

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
            reading = await client.hardware_read(DEVICE_ID, SETPOINT)
            assert reading["ok"] and reading["value"] == 50.0, reading
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
            preview = await client.hardware_write_request(DEVICE_ID, SETPOINT, 55.0)
            assert preview["ok"], preview
            assert not preview.get("approval_pending"), preview
            # The dry run returns ok=True because the homed interlock now holds,
            # but never reaches the approval gate or the device: ok with no
            # approval_request proves the dry-run short-circuit.
            # The next actual write still emits its own routed approval request.

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

        with journey.phase("batch: one planning round selects and approves one simulated program"):
            before = journey.proxy.stats.call_count
            transport = registry.get_open_transport(ARM_ID)
            assert not transport.write_log
            events = await _drive_with_auto_approval(
                client,
                "On r8_planning/piper_cell, supply task-equivalent complete alternatives that "
                "end 5 mm above the current TCP. Select by TCP distance and execute one.",
                session_id=SESSION, workspace=str(workspace),
            )
            detail = _detail(events, "hw_skill_run")
            assert _completed_ok(events, "hw_skill_run"), {
                key: detail.get(key) for key in ("failure_code", "error_preview", "side_effect_state")
            }
            assert sum(e.type == "tool_complete" and e.content == "hw_skill_run" for e in events) == 1
            assert journey.proxy.stats.call_count - before == 2  # alternatives + final answer
            assert sum(e.type == "approval_request" for e in events) == 1
            assert sum(name == "ee.z" for name, _ in transport.write_log) == 1
            (run,) = runtime.status()
            report = run.plan.basis["batch"]
            assert report["selected_id"] == "direct"
            assert [c["status"] for c in report["candidates"]] == ["feasible", "feasible", "refused"]
            assert report["evidence"]["state_reads"] == 1 and report["evidence"]["llm_calls"] == 0
            assert run.outcome.ok and not runtime._plans

        with journey.phase("provider: qualified inference and metadata cross the existing fleet read RPC"):
            before = journey.proxy.stats.call_count
            native_registry = await _qualified_provider(journey, stack, devices_dir / "native")
            assert journey.proxy.stats.call_count - before == 5
            reserved.close()
            native_node = LhpNodeServer(native_registry, node_id="r8_native")
            await native_node.start(host="127.0.0.1", port=native_port)
            stack.push_async_callback(native_node.stop)
            assert (await client.hardware_fleet_reconnect("r8_native"))["ok"]
            reading = await client.hardware_read("r8_native__native_model", "model.info")
            assert reading["ok"], reading
            assert reading["device_id"] == "r8_native__native_model"
            assert reading["value"]["instance"] == "r8-instance"
            assert reading["value"]["service_basis"]["qualification_ref"] == "r8-qualification"
            # No later primary prompt sees the endpoint-bound receipt digest.
            native_calls = [r for r in journey.proxy.stats.requests if r["model"] == "fixture-native"]
            assert 5 <= len(native_calls) <= 6
            assert all(not r.get("stream", False) for r in native_calls)
            await native_node.stop()
            await native_registry.close_all()

        journey.finish()
    finally:
        await stack.aclose()
        shutil.rmtree(devices_dir, ignore_errors=True)
