"""Offline receive/settlement replay; no live SDK imports or CAN sockets."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import piper_replay as replay

from leapflow.hardware.drivers._piper_feedback import (
    CAN_GROUPS, ReceiveLedger, ReceiveSource, attach_receive_ledger, receipt_problem,
)


@pytest.fixture(autouse=True)
def historical_imports(monkeypatch):
    with replay.imports(monkeypatch):
        yield


def load_core():
    return replay.load_script("cl_board/cl_arm_core")


def test_historical_fixture_inventory_and_hashes():
    record = json.loads((replay.ROOT / "manifest.json").read_text())
    assert record["purpose"] == "offline historical replay only"
    rows = record["files"]
    assert rows and len({row["snapshot"] for row in rows}) == len(rows)
    assert set(replay.SEEDS) <= {row["source"] for row in rows}
    assert not list(replay.ROOT.rglob("*.py"))
    for row in rows:
        body = (replay.ROOT / row["snapshot"]).read_bytes()
        assert len(body) == row["bytes"]
        assert hashlib.sha256(body).hexdigest() == row["sha256"]
        compile(body, row["snapshot"], "exec")


def test_historical_import_scope_restores_after_failure(monkeypatch):
    sentinel = ModuleType("cl")
    monkeypatch.setitem(sys.modules, "cl", sentinel)
    old_path, old_finders = list(sys.path), list(sys.meta_path)
    with pytest.raises(RuntimeError, match="offline failure"):
        with replay.imports(monkeypatch):
            config = importlib.import_module("cl.config")
            assert str(replay.ROOT) in config.__file__
            assert replay.ReplayFinder().find_spec("unrelated_module") is None
            for vendor in ("piper_sdk", "pyAgxArm", "pyrealsense2", "can"):
                with pytest.raises(ImportError, match="forbidden"):
                    replay.ReplayFinder().find_spec(vendor)
            raise RuntimeError("offline failure")
    assert sys.modules["cl"] is sentinel
    assert "cl.config" not in sys.modules
    assert sys.path == old_path and sys.meta_path == old_finders


@pytest.fixture
def core():
    return load_core()


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.000001)


TARGET = [285.36, 65.88, 123.52, -180.0, 0.0, -153.26]
FAILED_FINAL = [277.80, 63.54, 116.33, -179.65, -3.90, -153.45]


def snapshot(clock, pose=TARGET, *, source_at=None):
    stamp = clock.now if source_at is None else source_at
    sequence = int(stamp * 200)
    return dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose),
                mono=clock.now, receive_coherent=True,
                receive={g: {"valid": True, "received_mono": stamp,
                             "sequence": sequence} for g in CAN_GROUPS},
                voltages=[24.0] * 6, flags=[[] for _ in range(6)],
                joints=[0.0, 30.0, -30.0, 0.0, 0.0, 0.0],
                temps=[26.0] * 6, driver_temps=[32.0] * 6,
                efforts=[0.0] * 6, currents=[0.0] * 6, speeds_rad_s=[0.0] * 6,
                enabled=True, ctrl_mode=1, arm_status=0,
                gripper_enabled=True, gripper_faults=[], gripper_mm=70.0, gripper_nm=0.0)


class Sampler:
    def __init__(self, read):
        self.read = read

    def latest(self):
        return self.read()


class Arm:
    def __init__(self):
        self.poses, self.grips, self.modes = [], [], []

    def EndPoseCtrl(self, *pose):
        self.poses.append(pose)

    def MotionCtrl_2(self, **mode):
        self.modes.append(mode)

    def GripperCtrl(self, *grip):
        self.grips.append(grip)


@pytest.fixture
def clock(core, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(core.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(core.time, "sleep", clock.sleep)
    return clock


def test_receive_ledger_source_age_and_skew():
    ledger = ReceiveLedger()
    ledger.record("a", mono=10.0, wall=20.0)
    ledger.record("a", mono=10.1, wall=20.1)
    ledger.record("b", mono=10.08, wall=20.08)
    rows = {g: r.to_dict() for g, r in ledger.receipts().items()}
    assert rows["a"]["sequence"] == 2
    assert rows["a"]["interval_s"] == pytest.approx(0.1)
    assert not receipt_problem(rows, ("a", "b"), now=10.11, max_age_s=.12, max_skew_s=.06)
    assert receipt_problem(rows, ("a", "b"), now=11, max_age_s=.12, max_skew_s=.06)
    assert receipt_problem(rows, ("a", "b"), now=10.11, max_age_s=.12, max_skew_s=.01)
    with pytest.raises(ValueError):
        ledger.record("a", mono=10.1, wall=21)


def test_packet_callback_records_after_parser_and_never_breaks_receive_thread():
    callbacks = []
    cache = []
    callbacks.append(cache.append)  # Vendor parser registered before our receipt callback.
    robot = SimpleNamespace(get_context=lambda: SimpleNamespace(
        register_parser_packet_fun=callbacks.append))
    ledger = attach_receive_ledger(robot)
    assert isinstance(ledger, ReceiveSource)
    def receive(message):
        for callback in callbacks:
            callback(message)
    receive(SimpleNamespace(arbitration_id=0x2A1, dlc=8, is_remote_frame=False))
    assert len(cache) == 1 and ledger.receipts()["0x2a1"].valid
    receive(SimpleNamespace(arbitration_id=0x2A1, dlc=2, is_remote_frame=False))
    assert not ledger.receipts()["0x2a1"].valid
    receive(SimpleNamespace(arbitration_id=0x2A1, dlc=8, is_remote_frame=True))
    assert not ledger.receipts()["0x2a1"].valid
    receive(SimpleNamespace(arbitration_id="invalid", dlc=8))  # Must not throw.
    assert ledger.receipts()["0x2a1"].sequence == 3
    assert len(cache) == 4


def test_success_requires_source_stability_and_checked_final_pose(core, clock):
    arm = Arm()
    result = core.move(arm, Sampler(lambda: snapshot(clock)), core.Guard(), TARGET, stable_s=.03)
    assert result["why"] == "arrived" and result["attempts"] == 1
    assert result["duration_s"] >= .3


@pytest.mark.parametrize("deviation_at", [.045, .17])
def test_failed_pick_and_transient_hold_jump_block_closure(core, clock, deviation_at):
    arm = Arm()
    def read():
        elapsed = clock.now - 100
        return snapshot(clock, FAILED_FINAL if deviation_at <= elapsed <= deviation_at + .012 else TARGET)
    with pytest.raises(core.Trip, match="settlement_deviation"):
        core.move(arm, Sampler(read), core.Guard(), TARGET, stable_s=.03)
        core.grip(arm, Sampler(read), 47.6, guard=core.Guard())
    assert arm.grips == []


def test_restamped_frozen_cache_never_arrives(core, clock):
    arm = Arm()
    with pytest.raises(core.Trip, match="stale_feedback"):
        core.move(arm, Sampler(lambda: snapshot(clock, source_at=100.0)), core.Guard(), TARGET, stable_s=.03)
    assert len(arm.poses) == 1


@pytest.mark.parametrize("change,reason", [
    ({"enabled": False}, "lost_enablement"),
    ({"flags": [["driver_error_status"]] + [[] for _ in range(5)]}, "driver_fault"),
    ({"receive": {}}, "stale_feedback"),
    ({"efforts": [float("nan")] * 6}, "telemetry_missing"),
])
def test_guard_failure_before_send(core, clock, change, reason):
    snap = snapshot(clock)
    snap.update(change)
    arm = Arm()
    with pytest.raises(core.Trip, match=reason):
        core.move(arm, Sampler(lambda: snap), core.Guard(), TARGET)
    assert arm.poses == [] and arm.modes == []


def test_current_and_temperature_count_distinct_updates(core, clock):
    guard = core.Guard()
    snap = snapshot(clock)
    for _ in range(40):
        guard.check(snap, armed=True)
    assert len(guard.history) == 1 and len(guard._temp_at) == 1
    clock.sleep(.01)
    guard.check(snapshot(clock), armed=True)
    assert len(guard.history) == 2 and len(guard._temp_at) == 2


def test_negative_closing_effort_is_guarded_candidate_not_verified(core, clock):
    guard, arm = core.Guard(), Arm()
    guard.gripper_convention = {"closing_sign": -1, "unloaded_nm": 0,
                                "verified": True, "evidence_ref": "unloaded-characterization"}
    def read():
        row = snapshot(clock)
        if clock.now > 100:
            row.update(gripper_mm=56.9, gripper_nm=-.5)
        return row
    result = core.grip(arm, Sampler(read), 47.6, guard=guard, stable_s=.03)
    assert result["why"] == "effort_limit"
    assert result["grasp_state"] == "unknown" and not result["contact_proven"]
    assert len(arm.grips) == 1


def test_closing_without_unloaded_evidence_is_refused(core, clock):
    arm = Arm()
    with pytest.raises(core.Refused, match="verified closing sign"):
        core.grip(arm, Sampler(lambda: snapshot(clock)), 47.6, guard=core.Guard())
    assert arm.grips == []


def test_gripper_fault_stops_without_resend(core, clock):
    arm, guard = Arm(), core.Guard()
    def read():
        row = snapshot(clock)
        row["gripper_faults"] = ["driver_error_status"] if clock.now > 100 else []
        return row
    with pytest.raises(core.Trip, match="gripper_fault"):
        core.grip(arm, Sampler(read), 75, guard=guard)
    assert len(arm.grips) == 1


def test_false_stall_never_becomes_contact(core, clock):
    arm = Arm()
    with pytest.raises(core.Trip, match="gripper_stalled"):
        core.grip(arm, Sampler(lambda: snapshot(clock)), 75, guard=core.Guard())
    assert len(arm.grips) == 1


@pytest.fixture
def mission_module():
    return importlib.import_module("offline_piper_cl.mission")


@pytest.fixture
def mission(mission_module, monkeypatch, tmp_path):
    module = mission_module
    cfg = SimpleNamespace(get_path=lambda path, default=None: default,
                          bridge=SimpleNamespace(host="fake", port=1, timeout_s=.01))
    monkeypatch.setattr(module.config, "load_profile", lambda: cfg)
    monkeypatch.setattr(module.vision, "open_eye", lambda config: SimpleNamespace())
    real_run = module.journal.Run
    monkeypatch.setattr(module.journal, "Run", lambda tag: real_run(tag, root=tmp_path))
    return module.Mission(module.build_parser().parse_args(["--stage", "pick", "--yes"]))


def test_contact_stage_and_unbound_hold_refuse_before_any_connection(mission_module, mission):
    for stage, code in ((mission.stage_measure, "contact_excluded"),
                        (mission._observe_hold, "grasp_unavailable"),
                        (mission._identity, "calibration_unqualified")):
        with pytest.raises(mission_module.MissionError) as error:
            stage()
        assert error.value.code == code
    assert mission.arm is None


def test_auto_flag_cannot_bypass_stage_token(mission_module, mission):
    payload = {"lines": ["fake offline target"], "target_mm": [200, 0, 180]}
    with pytest.raises(mission_module.MissionError, match="stale_plan"):
        mission._approve("pick", payload, auto=True)
    mission.args.token = mission_module.approval_token("pick", payload)
    assert mission._approve("pick", payload, auto=True) == mission.args.token
    with pytest.raises(mission_module.MissionError, match="stale_plan"):
        mission._approve("flip", payload, auto=True)


def test_read_only_failure_persists_none_without_stop(mission_module, mission):
    result = mission.failure(mission_module.MissionError("calibration_unqualified", "no certificate"))
    assert result["side_effect_state"] == "none"
    stored = json.loads((mission.run.dir / "verdict.json").read_text())
    assert stored["envelope"]["failure_code"] == "calibration_unqualified"
    assert mission.arm is None


def test_returned_failure_is_classified_before_close_and_not_overwritten(mission_module, mission, monkeypatch):
    calls = []
    class FakeBridge:
        effects_attempted = True
        hello = {"session_id": "offline-session", "version": "fake", "sampler_hz": 200}
        def __init__(self, *args):
            pass
        def connect(self):
            return self
        def stop(self):
            calls.append("stop")
        def latest(self):
            return {}
        def close(self):
            calls.append("close")
    monkeypatch.setattr(mission_module.bridge_mod, "ArmBridge", FakeBridge)
    mission.connect()
    def stage():
        mission._retreat(why="unverified")
        return {"ok": False, "reason": "place_unverified", "released": {"visible": False}}
    monkeypatch.setattr(mission, "run_stage", stage)
    monkeypatch.setattr(mission_module, "Mission", lambda args: mission)
    assert mission_module.main(["--stage", "place"]) == 6
    assert calls == ["stop", "close"]
    stored = json.loads((mission.run.dir / "verdict.json").read_text())
    assert stored["reason"] == "place_unverified"
    assert stored["envelope"]["side_effect_state"] == "unknown"
    assert stored["detail"]["released"]["visible"] is False
    events = [json.loads(row) for row in mission.run.events.read_text().splitlines()]
    assert len([row for row in events if row["kind"] == "verdict"]) == 1


@pytest.mark.parametrize("command,effect", [("state", "none"), ("move", "unknown")])
@pytest.mark.parametrize("failure", ["send", "timeout", "refused"])
def test_bridge_rpc_failure_effects_and_pending_cleanup(mission_module, command, effect, failure):
    bridge = mission_module.bridge_mod.ArmBridge("fake", timeout_s=0)
    bridge.hello = {"lease_contract": "connection-owned/v1"}
    class FakeSocket:
        def sendall(self, data):
            request = json.loads(data)
            if failure == "send":
                raise OSError("fake send failure")
            if failure == "refused":
                event, slot = bridge._pending[request["id"]]
                slot["reply"] = {"ok": False, "error": {"code": "refused", "message": "fake refusal",
                                                       "side_effect_state": "none"}}
                event.set()
    bridge._sock = FakeSocket()
    with pytest.raises(mission_module.bridge_mod.BridgeError) as error:
        bridge.call(command)
    assert error.value.side_effect_state == ("none" if failure == "refused" else effect)
    assert bridge._pending == {}


@pytest.mark.parametrize("fault", ["duplicate", "sampler_freeze", "receive_freeze", "regression", "session"])
def test_bridge_does_not_restamp_duplicate_or_frozen_source(mission_module, monkeypatch, fault):
    clock = Clock()
    bridge_module = mission_module.bridge_mod
    monkeypatch.setattr(bridge_module.time, "monotonic", clock.monotonic)
    bridge = bridge_module.ArmBridge("fake")
    bridge.hello = {"session_id": "offline-session"}
    row = {"session_id": "offline-session", "source_mono": 10, "sent_mono": 10,
           "receive_coherent": True, "receive": {
               group: {"valid": True, "received_mono": 10, "sequence": 1} for group in CAN_GROUPS}}
    bridge._accept_snapshot(row)
    assert bridge.latest()["source_mono"] == 10
    clock.now += .2
    if fault != "duplicate":
        row["sent_mono"] += .2
    if fault == "regression":
        row["sent_mono"] = 9
    if fault == "receive_freeze":
        row["source_mono"] = row["sent_mono"]
    if fault == "session":
        row["session_id"] = "rebooted"
    bridge._accept_snapshot(row)
    with pytest.raises(bridge_module.BridgeError) as error:
        bridge.latest()
    assert error.value.code in ("stale_feedback", "stale_session")
    assert error.value.side_effect_state == "none"


def test_daemon_wire_json_nulls_unavailable_diagnostics():
    module = replay.load_script("cl_board/cl_armd")
    from leapflow.hardware.grasp_evidence import GraspState
    raw = module.wire_json({"samples": [float("nan"), float("inf"), -float("inf"), 1],
                            "state": GraspState.UNKNOWN})
    assert json.loads(raw) == {"samples": [None, None, None, 1], "state": "unknown"}
    assert "NaN" not in raw and "Infinity" not in raw


@pytest.fixture
def commissioning():
    from piper_ops import health
    return health


def health_registry(*, status=0, enabled=True, qualified=True, motors=None):
    from leapflow.hardware.drivers._piper_feedback import _FAULT_FLAGS
    telemetry = {"receive_qualified": qualified, "receive": {
        g: {"valid": True, "sequence": 1, "received_mono": 1., "received_wall": 1.} for g in CAN_GROUPS},
        "motors": motors if motors is not None else [
        {"motor": i, "enabled": enabled, "voltage_v": 24., "motor_temperature_c": 26.,
         "driver_temperature_c": 30., "faults": dict.fromkeys(_FAULT_FLAGS, False)} for i in range(1, 7)]} 
    values = {"arm.status": {"arm_status": status, "ctrl_mode": 1, "motion_status": 0},
              "arm.enabled": enabled, "arm.telemetry": telemetry}

    async def read_batch(device_id, channels):
        return SimpleNamespace(readings=tuple(SimpleNamespace(
            device_id=device_id, channel_id=c, value=values.get(c, 0.0), quality="ok") for c in channels))

    return SimpleNamespace(read_batch=read_batch)


@pytest.mark.asyncio
async def test_commissioning_health_accepts_qualified_six_motors(commissioning):
    from piper_ops.config import Settings
    result = await commissioning.read_snapshot(health_registry(), Settings(), ready=True)
    assert result.enabled is True
    assert result.joints == (0.0,) * 6


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs, message", [
    ({"status": 5}, "controller fault"),
    ({"enabled": False}, "all-six enable"),
    ({"qualified": False}, "receive qualification"),
    ({"motors": []}, "motor records"),
    ({"motors": [{"motor": 1}] * 6}, "motor records"),
    ({"motors": [{"motor": i, "faults": {"stall_status": i == 3}}
                 for i in range(1, 7)]}, "unknown motor"),
])
async def test_commissioning_unhealthy_feedback_refuses(commissioning, kwargs, message):
    from piper_ops.config import Settings
    from piper_ops.evidence import Refusal
    with pytest.raises(Refusal, match=message):
        await commissioning.read_snapshot(health_registry(**kwargs), Settings(), ready=True)


@pytest.mark.asyncio
async def test_commissioning_wrong_health_channels_refuse(commissioning):
    async def read_batch(device_id, channels):
        return SimpleNamespace(readings=tuple(SimpleNamespace(
            channel_id=c, quality="ok") for c in ("arm.status", "arm.enabled", "wrong")))

    from piper_ops.config import Settings
    from piper_ops.evidence import Refusal
    with pytest.raises(Refusal, match="unqualified controller"):
        await commissioning.read_snapshot(SimpleNamespace(read_batch=read_batch), Settings())


@pytest.mark.asyncio
async def test_commissioning_requires_consecutive_arrival(tmp_path):
    from dataclasses import replace
    from piper_ops.config import Settings
    from piper_ops.evidence import Evidence
    from piper_ops.motion import step, wait_arrived
    from test_piper_ops import snapshot as operational_snapshot
    arrived = iter([True, True, False, True, True, True])
    calls = []
    async def sample(**kwargs):
        calls.append(kwargs)
        return operational_snapshot(seq=len(calls) + 1, arrived=next(arrived))
    before = operational_snapshot()
    session = SimpleNamespace(snapshot=sample, settings=replace(Settings(), poll_s=.001),
                              evidence=Evidence(tmp_path / "arrived.json"))
    result = await wait_arrived(session, step("test", (), joints=before.joints), before,
                               {"joint": .5, "position": 2., "orientation": 1., "gripper": .5})
    assert result.receipts[0] == 7
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_commissioning_fault_is_not_replayed(tmp_path):
    from piper_ops.config import Settings
    from piper_ops.evidence import Evidence, Refusal
    from piper_ops.motion import wait_arrived
    from test_piper_ops import snapshot as operational_snapshot
    calls = []
    async def sample(**kwargs):
        calls.append(kwargs)
        raise Refusal("controller fault")
    session = SimpleNamespace(snapshot=sample, settings=Settings(), evidence=Evidence(tmp_path / "fault.json"))
    with pytest.raises(Refusal, match="controller fault"):
        await wait_arrived(session, {}, operational_snapshot(), {})
    assert len(calls) == 1


def test_commissioning_preserves_existing_evidence(tmp_path):
    from piper_ops.cli import main
    evidence = tmp_path / "attempt.json"
    evidence.write_text('{"historical": true}')
    assert main(["--json", str(evidence), "arm", "status"]) == 1
    assert evidence.read_text() == '{"historical": true}'


@pytest.fixture
def layout_tool():
    from piper_ops import layout
    return layout


def test_archive_detects_active_imports(layout_tool, tmp_path):
    (tmp_path / "arm_read.py").write_text("legacy = True\n")
    (tmp_path / "active.py").write_text("from arm_read import legacy\n")
    assert layout_tool.active_references(tmp_path, {"arm_read.py"}) == [
        {"consumer": "active.py", "source": "arm_read.py"}]


def test_archive_apply_preserves_dependencies_and_evidence(layout_tool, tmp_path, monkeypatch, capsys):
    source = tmp_path / "arm_read.py"
    source.write_text("legacy = True\n")
    for name in ("cl", "cl_board", "data", "tools"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "keep.txt").write_text("retained")
    from piper_ops.evidence import Evidence, Refusal
    saved = layout_tool.backup(tmp_path, tmp_path.parent / (tmp_path.name + "-backup"))
    manifest = {"root": str(tmp_path), "backup_manifest": saved["manifest"], "actions": [
        {"source": "arm_read.py", "target": "legacy/arm_read.py", "action": "move",
         "sha256": layout_tool.sha256(source)}]}
    evidence = Evidence(tmp_path / "data/cleanup.json")
    report = layout_tool.apply(tmp_path, manifest, execute=True, evidence=evidence)
    assert report["completed"] == ["arm_read.py"]
    assert not source.exists()
    assert (tmp_path / "legacy/arm_read.py").read_text() == "legacy = True\n"
    for name in ("cl", "cl_board", "data", "tools"):
        assert (tmp_path / name / "keep.txt").read_text() == "retained"
    with pytest.raises(Refusal, match="source changed"):
        layout_tool.apply(tmp_path, manifest)


def test_archive_apply_blocked_before_moves(layout_tool, tmp_path, monkeypatch):
    source = tmp_path / "arm_read.py"
    source.write_text("legacy = True\n")
    (tmp_path / "active.py").write_text("import arm_read\n")
    from piper_ops.evidence import Refusal
    saved = layout_tool.backup(tmp_path, tmp_path.parent / (tmp_path.name + "-backup"))
    manifest = {"root": str(tmp_path), "backup_manifest": saved["manifest"], "actions": [
        {"source": "arm_read.py", "target": "legacy/arm_read.py", "action": "move",
         "sha256": layout_tool.sha256(source)}]}
    with pytest.raises(Refusal, match="dependencies block"):
        layout_tool.apply(tmp_path, manifest)
    assert source.exists()
    assert not (tmp_path / "archive").exists()
