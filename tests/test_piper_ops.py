"""Tracked operational contracts, offline only: no CAN sockets or live credentials."""
from __future__ import annotations

import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from leapflow.engine.recovery.failure_envelope import FailureSource, Recoverability, SideEffectState
from leapflow.hardware.context import HardwareContext, TransportRef
from leapflow.hardware.lhp.client import DirectLhpClient
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.transports import lhp as lhp_transport
from leapflow.hardware.drivers._piper_feedback import CAN_GROUPS, _FAULT_FLAGS
from leapflow.hardware.drivers._piper_motion import CTRL_MODE_CAN, CTRL_MODE_TEACH
from leapflow.hardware.arm_status import ARM_STATUS_EMERGENCY_STOP
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.security.actions import ActionDescriptor
from leapflow.security.approval import ApprovalDecision, ApprovalRequest
from piper_ops import calibration, cli, health, layout, lifecycle, motion, node, session as session_module
from piper_ops.config import Settings
from piper_ops.evidence import Evidence, Refusal, encode
from piper_ops.session import (POSTURE_HELD, POSTURE_UNHELD, POSTURE_UNVERIFIED, Posture,
                               ScopedConsent, Session, unverified_digest)
from leapflow.tools.execution_context import ToolExecutionContext, current_tool_context, set_tool_context

pytestmark = pytest.mark.component


def context():
    path = Path(__file__).parents[1] / "src/leapflow/hardware/examples/piper_arm.yaml"
    return HardwareContext.from_mapping(yaml.safe_load(path.read_text()))


def snapshot(q=(0., 80., -90., 20., -40., -100.), *, seq=1, arrived=True):
    chain = DhChain.from_mapping(context().kinematics.dh)
    motors = [{"motor": i, "enabled": True, "faults": dict.fromkeys(_FAULT_FLAGS, False),
               "voltage_v": 24., "motor_temperature_c": 26., "driver_temperature_c": 30.}
              for i in range(1, 7)]
    telemetry = {"motors": motors, "receive_qualified": True, "receive": {
        g: {"sequence": seq, "received_mono": float(seq), "received_wall": float(seq), "valid": True}
        for g in CAN_GROUPS}}
    return health.Snapshot(q, chain.forward(q), 59.5, True,
                           {"arm_status": 0, "ctrl_mode": 1, "motion_status": 0 if arrived else 1}, telemetry)


def feedback_values(state):
    return {"arm.status": state.status, "arm.telemetry": state.telemetry,
            "arm.enabled": state.enabled, "gripper.position": state.gripper_mm,
            **dict(zip(health.JOINTS, state.joints)), **dict(zip(health.POSE, state.pose.as_tuple()))}


def registry(state):
    values = feedback_values(state)
    async def read_batch(device_id, channels):
        return SimpleNamespace(readings=tuple(SimpleNamespace(device_id=device_id, channel_id=c,
                               value=values[c], quality="ok") for c in channels))
    return SimpleNamespace(read_batch=read_batch)


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), root=tmp_path, gripper_max_mm=74., gripper_spec_max_mm=100.,
                   gripper_usable_max_mm=74., gripper_max_evidence="offline fixture only")


@pytest.mark.parametrize("name,value", [("speed_pct", 11), ("speed_pct", True),
    ("waypoint_mm", 11), ("settled_samples", 2), ("settled_samples", 3.5),
    ("position_mm", 3), ("orientation_deg", 2), ("grip_mm", 1), ("poll_s", "fast")])
def test_config_refuses_invalid(name, value):
    with pytest.raises(ValueError):
        Settings.load(**{name: value})


def test_unknown_config_and_max_evidence():
    with pytest.raises(ValueError, match="unknown"):
        Settings.load(not_a_setting=1)
    with pytest.raises(ValueError, match="evidence"):
        Settings.load(gripper_max_mm=74)


@pytest.mark.parametrize("module", ["cli", "calibration", "node", "lifecycle", "motion", "layout", "session", "health"])
def test_modules_import(module):
    assert importlib.import_module("piper_ops." + module)


def test_cli_dispatch_and_frame():
    args = cli.parser().parse_args(["verify", "motion", "--return", "--finish", "enabled"])
    assert (args.dz_mm, args.dy_mm, args.execute) == (100, 50, False)
    cycle = cli.parser().parse_args(["arm", "cycle", "--frame", "base"])
    assert (cycle.dz_mm, cycle.cycles, cycle.dwell_s) == (200, 5, 5)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["arm", "move", "--dz-mm", "100", "--frame", "viewpoint"])
    for command in (["arm", "status"], ["node", "status"], ["layout", "inspect"], ["calibration", "inspect"]):
        assert cli.parser().parse_args(command).operation in ("status", "inspect")


async def test_healthy_six_motor_feedback(settings):
    result = await health.read_snapshot(registry(snapshot()), settings, ready=True)
    assert result.enabled and len(result.receipts) == len(CAN_GROUPS)


class _CalibrationSession:
    """The session fields calibration.run touches; no node, no credentials."""

    def __init__(self, settings: Settings, evidence: Evidence) -> None:
        self.settings = settings
        self.evidence = evidence
        self.touched = False
        self.success = False


@pytest.fixture
def skill_owner(monkeypatch):
    """A stand-in node owner that records each request and answers success."""
    requests: list[dict[str, object]] = []

    class Client:
        async def hw_skill_run(self, **request: object) -> dict[str, object]:
            requests.append(request)
            return {"ok": True, "side_effect_state": "none"}

    @asynccontextmanager
    async def service(session, approved=None, *, share_frames=False):
        yield Client()

    monkeypatch.setattr(calibration, "service", service)
    return requests


async def test_calibration_execution_binds_the_declared_request_not_the_plan_defaults(
    settings, tmp_path, skill_owner
):
    declared = {"joints": ["j1", "j4", "j5"], "stand": {"j2": 74.0}, "excursion_deg": 15}
    approved = {
        "node_id": settings.node_id, "workcell": "piper_cell", "skill_id": "calib.screws",
        "request_args": declared, "plan_digest": "digest", "skill_plan": {"skill_id": "calib.screws"},
        # what the node echoes back is the plan, with every schema default filled in
        "args": {**declared, "excursion_deg": 15.0, "near_mm": 200.0, "far_mm": 750.0,
                 "camera": "camera", "frames": 3.0, "settle_s": 0.5, "speed_pct": 10.0},
    }
    run = _CalibrationSession(settings, Evidence(tmp_path / "run.json"))
    result = await calibration.run(run, workcell="piper_cell", skill="calib.screws",
                                   args=declared, approved=approved)
    assert result["ok"] is True and run.success
    assert skill_owner[-1]["dry_run"] is False
    assert skill_owner[-1]["approved_plan"] == approved["skill_plan"]
    assert skill_owner[-1]["args"] == approved["args"], (
        "the node re-plans what it receives, so the approved defaults travel with the request")

    with pytest.raises(Refusal, match="differs from approved plan"):
        await calibration.run(run, workcell="piper_cell", skill="calib.screws",
                              args={**declared, "excursion_deg": 20}, approved=approved)
    assert len(skill_owner) == 1, "a request the operator did not approve never reaches the node"


@pytest.mark.parametrize("fault", ["nonboolean", "fault", "missing", "duplicate", "id_list", "enable", "thermal", "voltage", "receipt"])
async def test_fail_closed_feedback(settings, fault):
    state = snapshot()
    motor = state.telemetry["motors"][0]
    if fault == "nonboolean":
        motor["faults"][_FAULT_FLAGS[0]] = 0
    elif fault == "fault":
        motor["faults"][_FAULT_FLAGS[0]] = True
    elif fault == "missing":
        motor["faults"] = {}
    elif fault == "duplicate":
        motor["motor"] = 2
    elif fault == "id_list":
        motor["motor"] = []
    elif fault == "enable":
        motor["enabled"] = False
    elif fault == "thermal":
        motor["motor_temperature_c"] = 60
    elif fault == "voltage":
        motor["voltage_v"] = 20
    elif fault == "receipt":
        state.telemetry["receive"] = {}
    gated = replace(settings, thermal_gate=True, voltage_gate=True, advisory_fault_gate=True)
    with pytest.raises(Refusal):
        await health.read_snapshot(registry(state), gated, ready=True)


@pytest.mark.parametrize("fault,gate,key", [
    ("thermal", "thermal_gate", "motor 1 thermal"),
    ("voltage", "voltage_gate", "motor 1 supply"),
    ("motor_overheating", "advisory_fault_gate", "motor 1 flags"),
    ("voltage_too_low", "advisory_fault_gate", "motor 1 flags"),
])
async def test_firmware_duplicate_gates_default_to_advisories(settings, fault, gate, key):
    """The drivers protect themselves; by default a breach is reported, not refused."""
    state = snapshot()
    motor = state.telemetry["motors"][0]
    if fault == "thermal":
        motor["motor_temperature_c"] = 60
    elif fault == "voltage":
        motor["voltage_v"] = 20
    else:
        motor["faults"][fault] = True
    current = await health.read_snapshot(registry(state), settings, ready=True, preflight=True)
    assert [note.split(":")[0] for note in current.advisories] == [key]
    with pytest.raises(Refusal):
        await health.read_snapshot(registry(state), replace(settings, **{gate: True}), ready=True)


@pytest.mark.parametrize("flag", ["driver_overcurrent", "driver_error_status", "collision_status",
                                  "stall_status"])
async def test_hard_faults_always_refuse(settings, flag):
    state = snapshot()
    state.telemetry["motors"][2]["faults"][flag] = True
    with pytest.raises(Refusal, match=flag):
        await health.read_snapshot(registry(state), settings, ready=True)


async def test_advisory_evidence_is_recorded_on_change_only(settings, tmp_path):
    hot = snapshot()
    hot.telemetry["motors"][1]["driver_temperature_c"] = 61.
    run = Session(settings, Evidence(tmp_path / "advisory.json"))
    run.registry = registry(hot)
    for _ in range(3):
        await run.snapshot(ready=True)
    hot.telemetry["motors"][1]["driver_temperature_c"] = 62.
    run.registry = registry(hot)
    await run.snapshot(ready=True)
    run.registry = registry(snapshot())
    await run.snapshot(ready=True)
    events = [e["event"] for e in run.evidence.events if e["event"].startswith("advisory")]
    assert events == ["advisory", "advisory_cleared"]


async def test_a_hot_arm_may_release_torque_but_not_energize(settings, tmp_path):
    """Holding a pose heats the drives; an enabled thermal gate must not block the way to cool."""
    settings = replace(settings, thermal_gate=True, voltage_gate=True)
    hot = snapshot()
    hot.telemetry["motors"][3]["driver_temperature_c"] = settings.preflight_temp_c + 3
    run = Session(settings, Evidence(tmp_path / "hot.json"))
    run.context, run.registry = context(), registry(hot)
    with pytest.raises(Refusal, match=r"motor 4 motor/driver 26/48 C, limit 45 C.*arm disable"):
        await lifecycle.preview(run, "enable")
    plan = await lifecycle.preview(run, "disable", torque_release=True)
    assert plan["steps"][0]["commands"] == (("arm.enable", 0),)
    # Faults and supply voltage still gate a release.
    hot.telemetry["motors"][3]["voltage_v"] = 20
    run.registry = registry(hot)
    with pytest.raises(Refusal, match="electrical safety gate"):
        await lifecycle.preview(run, "disable", torque_release=True)


def test_initialization_and_manual_intersection(settings):
    state = snapshot((-2.647, -2.149, -.993, 45.754, -7.723, -167.004))
    candidate = motion.initialization(context(), state, settings)
    assert candidate[1:3] == (5., -5.)
    assert candidate[5] == pytest.approx(-164.)
    assert max(abs(a-b) for a,b in zip(candidate, state.joints)) <= settings.init_max_deg
    plan = motion.preview(context(), state, settings, operation="initialize")
    assert plan["steps"][-1]["commands"] == tuple(zip(health.JOINTS, candidate))
    assert plan["initialization_sweep"]


@pytest.mark.parametrize("changes", [{"gripper_max_mm": None}, {"gripper_usable_max_mm": 101},
    {"gripper_max_mm": 100}, {"gripper_spec_max_mm": 68}, {"gripper_max_evidence": ""}])
def test_gripper_maximum_conflicts(settings, changes):
    with pytest.raises(Refusal):
        motion.opening_max(context(), replace(settings, **changes))


def test_declared_fk_and_cartesian_path(settings):
    start = snapshot()
    plan = motion.preview(context(), start, settings, operation="move", dz=100)
    targets = [row for row in plan["steps"] if row["pose"]]
    assert len(targets) >= 10
    assert all(motion.Pose.from_mapping(a["pose"]).distance_mm(motion.Pose.from_mapping(b["pose"])) <= 10
               for a, b in zip(targets, targets[1:]))
    assert targets[-1]["pose"]["z"] == pytest.approx(start.pose.z + 100)
    assert all(row["pose"]["rx"] == start.pose.rx for row in targets)
    with pytest.raises(Refusal, match="FK"):
        motion.preview(context(), replace(start, pose=start.pose.translated(dz=10)), settings,
                       operation="move", dz=100)


def test_full_return_is_preflighted(settings):
    # A nearby but non-singular origin; initialization budget is explicitly larger for this offline case.
    configured = replace(settings, init_max_deg=90.)
    plan = motion.preview(context(), snapshot(), configured, operation="verify", dy=50., dz=100.)
    steps = plan["steps"]
    assert any(row["label"].startswith("return_y") for row in steps)
    assert any(row["label"].startswith("return_z") for row in steps)
    assert steps[-1]["pose"] == pytest.approx(plan["origin_pose"])
    origin = motion.Pose.from_mapping(plan["origin_pose"])
    lifted = [r for r in steps if r["label"].startswith("lift")][-1]
    left = [r for r in steps if r["label"].startswith("left")][-1]
    assert lifted["pose"]["z"] == pytest.approx(origin.z + 100)
    assert left["pose"]["y"] == pytest.approx(origin.y + 50)
    assert left["pose"]["z"] == pytest.approx(origin.z + 100)
    assert all(motion.Pose.from_mapping(r["pose"]).rotation_to_deg(origin) < 1e-5
               for r in steps if r["pose"]) 
    assert plan["finish"] == "enabled"


def test_cycle_repeats_vertical_round_trips(settings):
    start = snapshot()
    plan = motion.preview(context(), start, settings, operation="cycle", dz=50., cycles=3, dwell_s=2.)
    up = [r for r in plan["steps"] if ".up." in r["label"]]
    down = [r for r in plan["steps"] if ".down." in r["label"]]
    dwells = [r for r in plan["steps"] if r["label"].endswith(".dwell")]
    assert len(up) == len(down) > 0 and len(dwells) == 3
    assert (plan["cycles"], plan["dwell_s"]) == (3, 2.)
    assert all(r["dwell_s"] == 2. and r["pose"] == pytest.approx(plan["origin_pose"]) for r in dwells)
    assert max(r["pose"]["z"] for r in up) == pytest.approx(start.pose.z + 50)
    assert plan["steps"][-1]["pose"] == pytest.approx(plan["origin_pose"])


@pytest.mark.parametrize("changes, match", [
    ({"dz": 0.}, "positive Z"),
    ({"dz": 50., "dx": 1.}, "positive Z"),
    ({"dz": 50., "cycles": 0}, "cycle count"),
    ({"dz": 50., "cycles": 21}, "cycle count"),
    ({"dz": 50., "cycles": 1, "dwell_s": -1.}, "dwell"),
    ({"dz": 50., "cycles": 1, "dwell_s": 61.}, "dwell"),
])
def test_cycle_refuses_unsafe_requests(settings, changes, match):
    with pytest.raises(Refusal, match=match):
        motion.preview(context(), snapshot(), settings, operation="cycle", **changes)


def test_joint_space_return_is_stepped_and_swept(settings):
    start = snapshot()
    goal = (5., 70., -70., 25., -40., -100.)
    plan = motion.preview(context(), start, settings, operation="joints", target=goal)
    rows = [r for r in plan["steps"] if r.get("joint_space")]
    assert rows and rows[-1]["joints"] == pytest.approx(goal)
    assert rows[-1]["commands"] == tuple(zip(health.JOINTS, rows[-1]["joints"]))
    previous = start.joints
    for row in rows:
        assert max(abs(a - b) for a, b in zip(previous, row["joints"])) <= settings.joint_step_deg
        previous = row["joints"]
    assert len(plan["initialization_sweep"]) >= 20, "sweep samples at most one degree apart"
    assert plan["target_joints"] == goal
    with pytest.raises(Refusal, match="conservative"):
        motion.preview(context(), start, settings, operation="joints", target=(0., 80., 5., 20., -40., -100.))
    with pytest.raises(Refusal, match="equals"):
        motion.preview(context(), start, settings, operation="joints", target=start.joints)
    args = cli.parser().parse_args(["arm", "joints", "--target", "5,70,-70,25,-40,-100"])
    assert args.target == goal
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["arm", "joints", "--target", "1,2,3"])


def test_bind_steps_skips_dwell(settings, tmp_path):
    plan = motion.preview(context(), snapshot(), settings, operation="cycle", dz=20., cycles=2, dwell_s=1.)
    run = Session(settings, Evidence(tmp_path / "bind-cycle.json"))
    run.bind_steps(plan["steps"])
    assert len(run._approved_steps) == len([r for r in plan["steps"] if r.get("dwell_s") is None])


FLANGE_TCP = {"x": 0., "y": 0., "z": 126.5, "rx": 0., "ry": 0., "rz": 0.}


def workspace(z=(-1000., 2000.), x=(-1000., 1000.), y=(-1000., 1000.), workcell="piper_cell"):
    """A ``calibration.workspace_guard`` reading with the declared 126.5 mm TCP."""
    return {"workcell": workcell, "workspace": {"x": list(x), "y": list(y), "z": list(z)},
            "tcp_calibrated": False, "flange_tcp": dict(FLANGE_TCP)}


def tcp_z(pose):
    return motion.WorkspaceGuard.from_mapping(workspace()).tcp(pose).z


def test_move_is_held_above_the_calibrated_floor(settings):
    start = snapshot()
    floor = tcp_z(start.pose) - 5.
    guard = workspace(z=(floor, 2000.))
    with pytest.raises(Refusal, match="calibrated workspace: TCP z="):
        motion.preview(context(), start, settings, operation="move", dz=-10., workspace=guard)
    plan = motion.preview(context(), start, settings, operation="move", dz=-4., workspace=guard)
    assert plan["workspace_guard"] == guard
    assert motion.preview(context(), start, settings, operation="move", dz=10., workspace=guard)
    # the declared flange envelope alone would have admitted the descent
    assert motion.preview(context(), start, settings, operation="move", dz=-10.)["workspace_guard"] is None
    unbounded = motion.preview(context(), start, settings, operation="move", dz=-10.,
                               workspace={**guard, "workspace": None})
    assert unbounded["workspace_guard"]["workspace"] is None


def test_a_start_below_the_floor_may_rise_but_not_sink(settings):
    start = snapshot()
    guard = workspace(z=(tcp_z(start.pose) + 5., 2000.))
    assert motion.preview(context(), start, settings, operation="move", dz=10., workspace=guard)
    assert motion.preview(context(), start, settings, operation="move", dx=5., workspace=guard)
    with pytest.raises(Refusal, match="calibrated workspace"):
        motion.preview(context(), start, settings, operation="move", dz=-1., workspace=guard)


def test_cycle_and_joint_sweep_are_held_to_the_workspace(settings):
    start = snapshot()
    z = tcp_z(start.pose)
    with pytest.raises(Refusal, match="calibrated workspace"):
        motion.preview(context(), start, settings, operation="cycle", dz=50., cycles=1,
                       workspace=workspace(z=(z - 1., z + 20.)))
    tcp = motion.WorkspaceGuard.from_mapping(workspace()).tcp(start.pose)
    tight = workspace(x=(tcp.x - .5, tcp.x + .5), y=(tcp.y - .5, tcp.y + .5), z=(z - .5, z + .5))
    with pytest.raises(Refusal, match="calibrated workspace"):
        motion.preview(context(), start, settings, operation="joints",
                       target=(5., 70., -70., 25., -40., -100.), workspace=tight)


@pytest.mark.parametrize("operation", ["initialize", "verify"])
def test_bring_up_operations_ignore_the_workspace(settings, operation):
    start = snapshot()
    guard = workspace(z=(tcp_z(start.pose) + 500., 2000.))
    plan = motion.preview(context(), start, replace(settings, init_max_deg=90.), operation=operation,
                          dy=50., dz=100., workspace=guard)
    assert plan["workspace_guard"] is None


def test_execution_rechecks_the_workspace_as_calibrated_now(settings):
    start = snapshot()
    floor = tcp_z(start.pose) - 20.
    plan = motion.preview(context(), start, settings, operation="move", dz=-10.,
                          workspace=workspace(z=(floor, 2000.)))
    plan = json.loads(json.dumps(plan, default=motion.json_default))
    motion.check_workspace(plan, workspace(z=(floor, 2000.)))
    with pytest.raises(Refusal, match="calibrated workspace"):
        motion.check_workspace(plan, workspace(z=(floor + 15., 2000.)))
    with pytest.raises(Refusal, match="disappeared"):
        motion.check_workspace(plan, {**workspace(), "workspace": None})
    with pytest.raises(Refusal, match="workcell"):
        motion.check_workspace(plan, workspace(z=(floor, 2000.), workcell="other_cell"))
    # a plan previewed before any calibration is still held to one that appeared since
    open_plan = motion.preview(context(), start, settings, operation="move", dz=-10.,
                               workspace={**workspace(), "workspace": None})
    motion.check_workspace(open_plan, {**workspace(), "workspace": None})
    with pytest.raises(Refusal, match="calibrated workspace"):
        motion.check_workspace(open_plan, workspace(z=(floor + 15., 2000.)))


async def test_workspace_guard_reads_box_and_tcp_offset(settings, tmp_path, monkeypatch):
    flange = {"x": 190.771, "y": 199.908, "z": 189.772, "rx": 180., "ry": 0., "rz": 90.}
    tcp = motion.WorkspaceGuard.from_mapping(workspace()).tcp(motion.Pose.from_mapping(flange)).to_dict()
    box = {"x": [110., 420.], "y": [-170., 220.], "z": [11.6899, 330.]}
    requests = []
    answer = {"ok": True, "side_effect_state": "none",
              "result": {"flange": flange, "tcp": tcp, "workspace": box, "tcp_calibrated": False}}

    class Client:
        async def hw_skill_run(self, **request):
            requests.append(request)
            return answer

    @asynccontextmanager
    async def service(session, approved=None, *, share_frames=False):
        yield Client()

    monkeypatch.setattr(calibration, "service", service)
    run = _CalibrationSession(settings, Evidence(tmp_path / "guard.json"))
    guard = await calibration.workspace_guard(run, "piper_cell")
    assert requests[-1]["skill"] == "arm.state" and requests[-1]["dry_run"] is True
    assert guard["workspace"] == box and guard["workcell"] == "piper_cell"
    assert motion.Pose.from_mapping(guard["flange_tcp"]).distance_mm(motion.Pose.from_mapping(FLANGE_TCP)) < 1e-3
    answer["ok"] = False
    with pytest.raises(Refusal, match="arm.state"):
        await calibration.workspace_guard(run, "piper_cell")
    for command in (["arm", "move", "--dz-mm", "5", "--frame", "base"], ["arm", "cycle", "--frame", "base"],
                    ["arm", "joints", "--target", "0,0,0,0,0,0"]):
        assert cli.parser().parse_args(command).workcell == "piper_cell"


def test_evidence_never_overwrites(tmp_path):
    path = tmp_path / "run.json"
    first = Evidence(path)
    first.record("original")
    with pytest.raises(Refusal):
        Evidence(path)
    assert "original" in first.event_path.read_text()


@pytest.mark.parametrize("states,initial,expected", [
    ([], "none", "none"),
    (["none"], "none", "none"),
    (["committed", "none"], "none", "committed"),
    (["committed", "partial", "committed"], "none", "partial"),
    (["unknown", "committed"], "none", "unknown"),
    (["committed"], "unknown", "unknown"),
    (["committed"], "partial", "partial"),
    ([None, "committed"], "none", "unknown"),
    (["invalid", "none"], "none", "unknown"),
    ([SideEffectState.COMMITTED], "none", "committed"),
])
def test_failure_preserves_recorded_effects(tmp_path, states, initial, expected):
    evidence = Evidence(tmp_path / "effects.json")
    for state in states:
        evidence.record("qualification_result", result={"side_effect_state": state})
    error = Refusal("original refusal", state=SideEffectState(initial))
    report = evidence.failure(error)
    failure = report["failure"]
    assert failure["side_effect_state"] == SideEffectState(expected)
    assert failure["envelope_id"] == error.envelope.envelope_id
    assert failure["timestamp"] == error.envelope.timestamp
    assert failure["message"] == "original refusal"
    assert json.loads(evidence.path.read_text())["failure"]["side_effect_state"] == expected
    assert [json.loads(line) for line in evidence.event_path.read_text().splitlines()] == json.loads(encode(report["events"]))


def test_preview_and_consent_do_not_imply_physical_effects(tmp_path):
    evidence = Evidence(tmp_path / "no-dispatch.json")
    evidence.record("approved_plan", plan={"steps": [{"side_effect_state": "committed"}]})
    evidence.record("operator_consent", approved=True)
    evidence.record("preview", result={"dry_run": True, "side_effect_state": "none"})
    assert evidence.failure(Refusal("preflight refused"))["failure"]["side_effect_state"] == SideEffectState.NONE


def test_backup_hidden_symlink_and_safe_manifest(tmp_path):
    root = tmp_path / "piper"
    root.mkdir()
    (root / ".config").write_text("hidden")
    (root / "old.py").write_text("pass\n")
    (root / "external").symlink_to(tmp_path / "absent", target_is_directory=True)
    saved = layout.backup(root, tmp_path / "backup")
    assert Path(saved["archive"]).stat().st_mode & 0o777 == 0o600
    record = json.loads(Path(saved["manifest"]).read_text())
    assert any(m["path"] == "piper/.config" for m in record["members"])
    assert any(m["path"] == "piper/external" and m["type"] == "2" for m in record["members"])
    manifest = {"root": str(root), "backup_manifest": saved["manifest"], "actions": [
        {"action": "move", "source": "old.py", "target": "legacy/old.py", "sha256": layout.sha256(root / "old.py")}]}
    assert layout.apply(root, manifest)["executed"] is False
    assert (root / "old.py").exists()
    evidence = Evidence(tmp_path / "move.json")
    assert layout.apply(root, manifest, execute=True, evidence=evidence)["completed"] == ["old.py"]
    assert (root / "legacy/old.py").read_text() == "pass\n"
    assert "cleanup_completed" in evidence.event_path.read_text()


def test_safe_deletion_is_required_before_any_move(tmp_path):
    root = tmp_path / "piper"
    root.mkdir()
    source = root / "old.py"
    source.write_text("pass")
    saved = layout.backup(root, tmp_path / "backup")
    manifest = {"root": str(root), "backup_manifest": saved["manifest"], "actions": [
        {"action": "remove", "source": "old.py", "sha256": layout.sha256(source)}]}
    with pytest.raises(Refusal, match="safe remote deletion"):
        layout.apply(root, manifest)
    assert source.exists()


def test_preview_exact_binding_and_single_use(settings, tmp_path):
    plan = motion.preview(context(), snapshot(), settings, operation="move", dz=10.)
    path = tmp_path / "preview.json"
    path.write_text(json.dumps(cli.preview_result(plan), default=str))
    args = cli.parser().parse_args(["arm", "move", "--dz-mm", "10", "--frame", "base", "--execute",
        "--preview-file", str(path), "--approved-plan-hash", motion.digest_plan(plan),
        "--approval-note", "offline consent", "--approved-digest", unverified_digest(context()), "--clearance-confirmed"])
    assert cli.approved(args, settings, "move")["operation"] == "move"
    with pytest.raises(Refusal, match="consumed"):
        cli.approved(args, settings, "move")


async def test_fresh_consecutive_arrival(settings, tmp_path):
    before = snapshot(seq=1)
    sequence = iter([(2, True), (3, True), (3, True), (4, False), (5, True), (6, True), (7, True)])
    calls = []
    async def sample(**kwargs):
        seq, arrived = next(sequence)
        calls.append(seq)
        return snapshot(seq=seq, arrived=arrived)
    session = SimpleNamespace(settings=replace(settings, poll_s=.001), snapshot=sample, evidence=Evidence(tmp_path / "wait.json"))
    row = motion.step("move", tuple(zip(health.JOINTS, before.joints)), joints=before.joints, pose=before.pose)
    result = await motion.wait_arrived(session, row, before, {"joint": .5, "position": 2., "orientation": 1., "gripper": .5})
    assert result.receipts[0] == 7 and len(calls) == 7


async def test_arrival_accepts_pose_converged_servo_residual(settings, tmp_path):
    """A base-yaw following error must not read as a missed waypoint."""
    before = snapshot(seq=1)
    prediction = tuple(q + .46 for q in before.joints)
    sequence = iter([(2, True), (3, True), (4, True)])
    async def sample(**kwargs):
        seq, arrived = next(sequence)
        return snapshot(seq=seq, arrived=arrived)
    session = SimpleNamespace(settings=replace(settings, poll_s=.001), snapshot=sample,
                              evidence=Evidence(tmp_path / "residual.json"))
    row = motion.step("left.009", (), joints=prediction, pose=before.pose)
    result = await motion.wait_arrived(session, row, before,
                                       {"joint": .5, "position": 2., "orientation": 1., "gripper": .5})
    assert result.receipts[0] == 4


def test_return_check_uses_pose_and_envelope(settings):
    """The return is judged by the flange pose, with the envelope bounding branch departure."""
    origin = snapshot()
    plan = {"arrival": {"joint": .5, "position": 2., "orientation": 1., "gripper": .5}}
    motion.check_return(plan, origin, replace(origin, joints=tuple(q + 1.2 for q in origin.joints)), settings)
    with pytest.raises(Refusal, match="departed"):
        motion.check_return(plan, origin, replace(origin, pose=origin.pose.translated(dz=3.)), settings)
    with pytest.raises(Refusal, match="departed"):
        motion.check_return(plan, origin, replace(origin, joints=tuple(q + 4. for q in origin.joints)), settings)


async def test_snapshot_requalifies_a_momentary_receive_gap(settings, tmp_path):
    """A stale CAN sample is re-read, never reported as an arm state."""
    state = snapshot()
    values, calls = feedback_values(state), []
    async def read_batch(device_id, channels):
        calls.append(1)
        quality = "stale" if len(calls) <= 2 else "ok"
        return SimpleNamespace(readings=tuple(SimpleNamespace(device_id=device_id, channel_id=c,
                               value=values[c], quality=quality) for c in channels))
    run = Session(replace(settings, poll_s=.001), Evidence(tmp_path / "requalify.json"))
    run.registry = SimpleNamespace(read_batch=read_batch)
    with pytest.raises(Refusal, match="unqualified controller channels"):
        await run.snapshot()
    current = await run.snapshot(requalify=True)
    assert len(calls) == 3 and current.joints == state.joints
    assert [e["event"] for e in run.evidence.events] == ["stale_feedback", "feedback_requalified"]


async def test_requalification_deadline_is_a_hard_stop(settings, tmp_path):
    """Feedback that never requalifies stops the run instead of spinning."""
    values = feedback_values(snapshot())
    async def read_batch(device_id, channels):
        return SimpleNamespace(readings=tuple(SimpleNamespace(device_id=device_id, channel_id=c,
                               value=values[c], quality="stale") for c in channels))
    run = Session(replace(settings, poll_s=.001, request_timeout_s=.001), Evidence(tmp_path / "never.json"))
    run.registry = SimpleNamespace(read_batch=read_batch)
    with pytest.raises(Refusal, match="did not requalify") as caught:
        await run.snapshot(requalify=True)
    assert caught.value.envelope.side_effect_state == SideEffectState.UNKNOWN


def posture(state=POSTURE_HELD, **fields):
    """One posture verdict; the arm is only ever described by two qualified samples."""
    base = dict(detail="it holds its posture", joints=(0.,) * 6, ctrl_mode=CTRL_MODE_CAN,
                enabled=True, arm_status=0)
    return Posture(state, **{**base, **fields})


async def test_session_stops_only_when_it_cannot_say_what_it_did(settings, tmp_path, monkeypatch):
    """A failed verdict is a result; a stop is owed only for an effect this run cannot vouch for."""
    settings = replace(settings, failure_policy="stop")
    for case, stopped in (("settled", False), ("unknown", True), ("halted", True)):
        run = Session(settings, Evidence(tmp_path / f"exit-{case}.json"), execute=True)
        run.touched, run.success = True, case == "settled"
        run.stop_required = case == "halted"
        calls = []
        async def stop():
            calls.append("stop")
        async def close():
            calls.append("close")
        async def sample():
            return posture(POSTURE_UNHELD, detail="unqualified fixture", enabled=False)
        monkeypatch.setattr(run, "stop", stop)
        monkeypatch.setattr(run, "_close", close)
        monkeypatch.setattr(run, "posture", sample)
        error = Refusal("run failed", state=SideEffectState.UNKNOWN) if case == "unknown" else None
        if error is None:
            with pytest.raises(Refusal, match="cleanup unconfirmed"):
                await run.__aexit__(None, None, None)
        else:
            await run.__aexit__(type(error), error, None)
        assert calls == (["stop", "close"] if stopped else ["close"])
        assert run.evidence.events[0]["event"] == "posture_check"
        assert run.evidence.events[0]["own_stop"] is stopped


@pytest.mark.parametrize("case", ["unknown", "halted", "cancelled"])
async def test_hold_policy_never_stops_automatically(settings, tmp_path, monkeypatch, case):
    """Default policy: an owed stop is withheld and recorded; the arm keeps its posture."""
    assert settings.failure_policy == "hold"
    run = Session(settings, Evidence(tmp_path / f"hold-{case}.json"), execute=True)
    run.touched = True
    run.stop_required = case == "halted"
    calls = []
    async def stop():
        calls.append("stop")
    async def close():
        calls.append("close")
    async def sample():
        return posture()
    monkeypatch.setattr(run, "stop", stop)
    monkeypatch.setattr(run, "_close", close)
    monkeypatch.setattr(run, "posture", sample)
    error = {"unknown": Refusal("arrival deadline expired", state=SideEffectState.UNKNOWN),
             "cancelled": asyncio.CancelledError(), "halted": None}[case]
    await run.__aexit__(type(error) if error else None, error, None)
    assert calls == ["close"]
    names = [row["event"] for row in run.evidence.events]
    assert names == ["stop_withheld", "posture_check"]
    assert run.evidence.events[1]["own_stop"] is False and run.evidence.events[1]["state"] == "held"


async def test_hold_policy_waits_for_the_last_segment_to_settle(settings, tmp_path, monkeypatch):
    """A still-moving healthy arm is waited out by reads, never written to or stopped."""
    run = Session(replace(settings, poll_s=.001), Evidence(tmp_path / "rest.json"), execute=True)
    run.touched = True
    states = [posture(POSTURE_UNHELD, detail="it is still moving", moved_deg=.4),
              posture(POSTURE_UNHELD, detail="it is still moving", moved_deg=.1), posture()]
    calls = []
    async def stop():
        calls.append("stop")
    async def sample():
        return states.pop(0)
    monkeypatch.setattr(run, "stop", stop)
    monkeypatch.setattr(run, "_close", lambda: asyncio.sleep(0))
    monkeypatch.setattr(run, "posture", sample)
    error = Refusal("observed joint deviation", state=SideEffectState.UNKNOWN)
    await run.__aexit__(Refusal, error, None)
    assert calls == [] and states == []
    check = next(row for row in run.evidence.events if row["event"] == "posture_check")
    assert check["state"] == "held"


async def test_hold_policy_close_failure_does_not_halt(settings, tmp_path, monkeypatch):
    run = Session(settings, Evidence(tmp_path / "hold-close.json"), execute=True)
    run.touched = True
    calls = []
    async def sample():
        return posture(POSTURE_UNVERIFIED, detail="feedback timed out")
    async def failed_close():
        raise OSError("close failed")
    async def halt():
        calls.append("halt")
    monkeypatch.setattr(run, "posture", sample)
    monkeypatch.setattr(run, "_close", failed_close)
    monkeypatch.setattr(run, "_halt_with_fresh_client", halt)
    with pytest.raises(Refusal, match="cleanup unconfirmed"):
        await run.__aexit__(None, None, None)
    assert calls == []


def test_grasp_preview_binds_force_before_stroke(settings):
    start = snapshot()
    plan = motion.preview(context(), start, settings, operation="grasp", width=40., force=2.5)
    row = plan["steps"][-1]
    assert row["commands"] == (("gripper.effort", 2.5), ("gripper.position", 40.))
    assert row["stall_settles"] is True and row["grip"] == 40. and plan["force_n"] == 2.5
    for width, force, message in ((70., 2., "narrower"), (40., 0., "grip force"),
                                  (40., 4., "grip force"), (40., None, "explicit")):
        with pytest.raises(Refusal, match=message):
            motion.preview(context(), start, settings, operation="grasp", width=width, force=force)


def test_full_opening_settles_at_the_end_stop(settings):
    """A release to the fitted maximum ends at the mechanical stop, not on its nominal value."""
    start = snapshot()
    full = motion.preview(context(), start, settings, operation="gripper")["steps"][-1]
    assert full["stall_settles"] is True and full["grip"] > start.gripper_mm
    explicit = motion.preview(context(), start, settings, operation="gripper", width=30.)["steps"][-1]
    assert explicit["stall_settles"] is False


async def test_grasp_settles_on_a_stalled_jaw_and_reports_blocked(settings, tmp_path):
    """Jaws that stop short on an object settle as a grasp instead of timing out."""
    widths = iter([50., 45.] + [41.2] * 500)
    seq = [1]
    async def sample(**kwargs):
        seq[0] += 1
        return replace(snapshot(seq=seq[0]), gripper_mm=next(widths))
    evidence = Evidence(tmp_path / "stall.json")
    session = SimpleNamespace(snapshot=sample, evidence=evidence,
                              settings=replace(settings, poll_s=.001, grasp_settle_s=.002))
    before = snapshot()
    row = motion.step("grasp", (), joints=before.joints, pose=before.pose, grip=30.)
    row["stall_settles"] = True
    result = await motion.wait_arrived(session, row, before,
                                       {"joint": .5, "position": 2., "orientation": 1., "gripper": .5})
    assert result.gripper_mm == 41.2
    settled = next(e for e in evidence.events if e["event"] == "grasp.settled")
    assert settled["blocked"] is True and settled["grasp_target_mm"] == 30.


async def test_plain_gripper_move_still_requires_the_target(settings, tmp_path):
    async def sample(**kwargs):
        sample.seq += 1
        return replace(snapshot(seq=sample.seq), gripper_mm=41.2)
    sample.seq = 1
    session = SimpleNamespace(snapshot=sample, evidence=Evidence(tmp_path / "no-stall.json"),
                              settings=replace(settings, poll_s=.001, move_timeout_s=.05))
    before = snapshot()
    with pytest.raises(Refusal, match="arrival deadline"):
        await motion.wait_arrived(session, motion.step("gripper", (), joints=before.joints,
                                  pose=before.pose, grip=30.), before,
                                  {"joint": .5, "position": 2., "orientation": 1., "gripper": .5})


async def test_a_settled_failure_is_never_stopped(settings, tmp_path, monkeypatch):
    """Every write settled and the arm stands still: a stop here is what dropped it 177 mm."""
    run = Session(settings, Evidence(tmp_path / "settled-failure.json"), execute=True)
    run.touched = True
    calls = []
    async def stop():
        calls.append("stop")
    async def close():
        calls.append("close")
    async def sample():
        return posture()
    monkeypatch.setattr(run, "stop", stop)
    monkeypatch.setattr(run, "_close", close)
    monkeypatch.setattr(run, "posture", sample)
    await run.__aexit__(None, None, None)
    assert calls == ["close"]
    assert [row["event"] for row in run.evidence.events] == ["posture_check"]


async def test_a_released_arm_is_not_re_energized_without_separate_approval(settings, tmp_path, monkeypatch):
    """An uncertain stop never manufactures authority to restart a gravity-loaded arm."""
    run = Session(replace(settings, failure_policy="stop"), Evidence(tmp_path / "arrest.json"), execute=True)
    run.touched = True
    states = [posture(POSTURE_UNHELD, detail="not every drive reports enabled", enabled=False,
                      moved_deg=1.5), posture()]
    calls, writes = [], []
    async def stop():
        calls.append("stop")
    async def close():
        calls.append("close")
    async def sample():
        return states.pop(0)
    async def configure(**params):
        writes.append(params)
        return {"ok": True, "side_effect_state": "committed"}
    run.tools = SimpleNamespace(hw_configure=configure)
    run.consent = SimpleNamespace(expected=(), fail_safe=False)
    monkeypatch.setattr(run, "stop", stop)
    monkeypatch.setattr(run, "_close", close)
    monkeypatch.setattr(run, "posture", sample)
    error = Refusal("dispatch outcome unknown", state=SideEffectState.UNKNOWN)
    await run.__aexit__(Refusal, error, None)
    assert calls == ["stop", "close"]
    assert writes == []
    assert run.consent.expected == ()
    assert run.consent.fail_safe is False
    assert [row["event"] for row in run.evidence.events] == ["posture_check", "posture_unresolved"]
    assert run.evidence.events[-1]["operational_complete"] is False
    assert error.envelope.side_effect_state == SideEffectState.UNKNOWN


async def test_unreadable_holding_is_unsuccessful_unknown_and_never_enables(settings, tmp_path, monkeypatch):
    run = Session(settings, Evidence(tmp_path / "unreadable-holding.json"), execute=True)
    run.touched, run.success = True, True
    writes = []

    async def configure(**params):
        writes.append(params)

    async def sample():
        return posture(POSTURE_UNVERIFIED, detail="feedback timed out")

    run.tools = SimpleNamespace(hw_configure=configure)
    monkeypatch.setattr(run, "_close", lambda: asyncio.sleep(0))
    monkeypatch.setattr(run, "posture", sample)
    with pytest.raises(Refusal, match="cleanup unconfirmed") as error:
        await run.__aexit__(None, None, None)
    assert not run.success and not writes
    assert error.value.envelope.side_effect_state == SideEffectState.UNKNOWN
    unresolved = next(row for row in run.evidence.events if row["event"] == "posture_unresolved")
    assert unresolved["operational_complete"] is False


@pytest.mark.parametrize("original_failure", [False, True])
async def test_posture_reader_exception_keeps_failure_and_marks_unsuccessful(
        settings, tmp_path, monkeypatch, original_failure):
    run = Session(settings, Evidence(tmp_path / "posture-error.json"), execute=True)
    run.touched = run.success = True
    writes = []

    async def sample():
        raise OSError("feedback unavailable")

    monkeypatch.setattr(run, "posture", sample)
    monkeypatch.setattr(run, "_close", lambda: asyncio.sleep(0))
    monkeypatch.setattr(run, "stop", lambda: asyncio.sleep(0))
    run.tools = SimpleNamespace(hw_configure=lambda **params: writes.append(params))
    original = Refusal("original failure", state=SideEffectState.UNKNOWN) if original_failure else None
    if original is None:
        with pytest.raises(Refusal, match="cleanup unconfirmed") as failure:
            await run.__aexit__(None, None, None)
        assert failure.value.envelope.side_effect_state == SideEffectState.UNKNOWN
    else:
        await run.__aexit__(Refusal, original, None)
        assert str(original) == "original failure"
        assert original.envelope.side_effect_state == SideEffectState.UNKNOWN
    assert run.success is False and not writes
    assert any(row["event"] == "posture_unresolved" and not row["operational_complete"]
               for row in run.evidence.events)


async def test_teach_mode_is_left_in_the_operators_hands(settings, tmp_path, monkeypatch):
    """An arm an operator is posturing by hand is not re-energized by a session exit."""
    run = Session(settings, Evidence(tmp_path / "teach.json"), execute=True)
    run.touched, run.success = True, True
    writes = []
    async def configure(**params):
        writes.append(params)
        return {"ok": True}
    async def sample():
        return posture(POSTURE_UNHELD, detail="it is not in CAN control (teach)",
                       ctrl_mode=CTRL_MODE_TEACH, enabled=False)
    run.tools = SimpleNamespace(hw_configure=configure)
    run.consent = SimpleNamespace(expected=(), fail_safe=False)
    monkeypatch.setattr(run, "_close", lambda: asyncio.sleep(0))
    monkeypatch.setattr(run, "posture", sample)
    with pytest.raises(Refusal, match="cleanup unconfirmed"):
        await run.__aexit__(None, None, None)
    assert writes == []
    assert run.success is False
    unresolved = next(row for row in run.evidence.events if row["event"] == "posture_unresolved")
    assert unresolved["operational_complete"] is False
    assert "operator" in unresolved["next_step"]


async def test_a_latch_this_session_did_not_engage_is_never_reset(settings, tmp_path, monkeypatch):
    run = Session(settings, Evidence(tmp_path / "foreign-latch.json"), execute=True)
    run.touched, run.success = True, True
    writes = []
    async def configure(**params):
        writes.append(params)
        return {"ok": True}
    async def sample():
        return posture(POSTURE_UNHELD, detail="its status word is not normal (1)",
                       arm_status=ARM_STATUS_EMERGENCY_STOP)
    run.tools = SimpleNamespace(hw_configure=configure)
    run.consent = SimpleNamespace(expected=(), fail_safe=False)
    monkeypatch.setattr(run, "_close", lambda: asyncio.sleep(0))
    monkeypatch.setattr(run, "posture", sample)
    with pytest.raises(Refusal, match="cleanup unconfirmed"):
        await run.__aexit__(None, None, None)
    assert writes == []
    unresolved = next(row for row in run.evidence.events if row["event"] == "posture_unresolved")
    assert "did not engage" in unresolved["next_step"]


async def test_an_operator_release_is_not_re_held_at_exit(settings, tmp_path, monkeypatch):
    """`arm stop` asks for a released arm; the exit must not undo the request."""
    run = Session(settings, Evidence(tmp_path / "release.json"), execute=True)
    run.context = context()
    async def estop(**kwargs):
        return {"ok": True}
    run.tools = SimpleNamespace(hw_estop=estop)
    await lifecycle.execute(run, await lifecycle.preview(run, "stop"))
    run.touched = True
    assert run.leave_unheld is True
    calls = []
    async def close():
        calls.append("close")
    async def sample():
        raise AssertionError("a released arm is not judged or re-held")
    monkeypatch.setattr(run, "_close", close)
    monkeypatch.setattr(run, "posture", sample)
    await run.__aexit__(None, None, None)
    assert calls == ["close"]


async def test_cleanup_does_not_replace_original_failure(settings, tmp_path, monkeypatch):
    run = Session(settings, Evidence(tmp_path / "failure.json"), execute=True)
    run.touched = True
    async def stop():
        raise OSError("offline stop failure")
    async def close():
        pass
    monkeypatch.setattr(run, "stop", stop)
    monkeypatch.setattr(run, "_close", close)
    error = Refusal("original", state=SideEffectState.UNKNOWN)
    await run.__aexit__(Refusal, error, None)
    assert str(error) == "original"


@pytest.mark.parametrize("original_failure", [False, True])
async def test_cleanup_audit_failure_preserves_original_and_closes(
        settings, tmp_path, monkeypatch, original_failure):
    run = Session(settings, Evidence(tmp_path / "audit-failure.json"), execute=True)
    run.touched = run.success = True
    calls = []

    async def sample():
        return posture(POSTURE_HELD)

    async def close():
        calls.append("close")

    def unavailable(*args, **kwargs):
        raise OSError("offline audit sink unavailable")

    monkeypatch.setattr(run, "posture", sample)
    monkeypatch.setattr(run, "_close", close)
    monkeypatch.setattr(run, "stop", lambda: asyncio.sleep(0))
    monkeypatch.setattr(run.evidence, "record", unavailable)
    original = Refusal("original failure", state=SideEffectState.UNKNOWN) if original_failure else None
    if original is None:
        with pytest.raises(Refusal, match="cleanup unconfirmed") as failure:
            await run.__aexit__(None, None, None)
        assert failure.value.envelope.side_effect_state == SideEffectState.UNKNOWN
    else:
        await run.__aexit__(Refusal, original, None)
        assert str(original) == "original failure"
        assert original.envelope.side_effect_state == SideEffectState.UNKNOWN
    assert calls == ["close"] and not run.success


async def test_exact_request_consent_is_consumed(settings, tmp_path):
    declaration = context()
    gate = ScopedConsent(declaration, Evidence(tmp_path / "consent.json"), True)
    # Exercise a real action descriptor against the human surface, never a fake orchestrator.
    gate.expected = (("arm.speed_pct", 10.),)
    action = ActionDescriptor.device(kind="device.configure", device_id=declaration.device_id,
        channel_id="arm.speed_pct", value=10., unit="percent", location=declaration.location)
    request = ApprovalRequest(category="hardware", detail=action.detail, action=action)
    assert await gate.request_approval(request) == ApprovalDecision.ALLOW_ONCE
    assert await gate.request_approval(request) == ApprovalDecision.DENY


async def test_unknown_command_effect_never_replays(settings, tmp_path, monkeypatch):
    run = Session(settings, Evidence(tmp_path / "unknown.json"), execute=True)
    run.context = context()
    run.digest = unverified_digest(run.context)
    run.consent = ScopedConsent(run.context, run.evidence, True)
    calls = []
    async def request(method):
        return {"node_id": settings.node_id, "devices": [run.context.to_dict()]}
    async def configure(**kwargs):
        calls.append(kwargs)
        return {"ok": False, "side_effect_state": "unknown"}
    async def sample(**kwargs):
        return snapshot()
    # The declaration provenance is made verified in the offline fixture only.
    run.context = replace(run.context, provenance=replace(run.context.provenance, verified_by="fixture", verified_at=1.))
    run.client = SimpleNamespace(request=request)
    run.tools = SimpleNamespace(hw_configure=configure)
    monkeypatch.setattr(run, "snapshot", sample)
    row = motion.step("speed", (("arm.speed_pct", 10.),), configure=True)
    run.bind_steps([row])
    with pytest.raises(Refusal) as error:
        await run.command("speed", (("arm.speed_pct", 10.),), configure=True)
    assert error.value.envelope.side_effect_state == SideEffectState.UNKNOWN
    with pytest.raises(Refusal, match="complete approved"):
        await run.command("speed", (("arm.speed_pct", 10.),), configure=True)
    assert len(calls) == 1


async def test_stop_is_independent_of_faulty_feedback(settings, tmp_path):
    run = Session(settings, Evidence(tmp_path / "fault-stop.json"), execute=True)
    run.context = context()
    calls = []
    async def failing_read(*args, **kwargs):
        raise Refusal("unqualified motor feedback")
    async def estop(**kwargs):
        calls.append(kwargs)
        return {"ok": True}
    run.registry = SimpleNamespace(read_batch=failing_read)
    run.tools = SimpleNamespace(hw_estop=estop)
    plan = await lifecycle.preview(run, "stop")
    result = await lifecycle.execute(run, plan)
    assert result["halt_delivered"] is True and result["feedback_verified"] is False
    assert calls == [{"device_id": settings.device_id}]
    assert run.success


async def test_enter_cleanup_preserves_original(settings, tmp_path, monkeypatch):
    run = Session(settings, Evidence(tmp_path / "enter-failure.json"))
    async def connect():
        raise Refusal("original connection refusal")
    async def close():
        raise OSError("secondary close failure")
    monkeypatch.setattr(run, "_open", connect)
    run.client = SimpleNamespace(close=close)
    with pytest.raises(Refusal, match="original connection"):
        await run.__aenter__()
    assert run.evidence.events[-1]["event"] == "cleanup_failure"


async def test_repeated_cancel_finishes_halt_before_close(settings, tmp_path):
    run = Session(replace(settings, failure_policy="stop"), Evidence(tmp_path / "cancel.json"), execute=True)
    run.touched = True
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def estop(**kwargs):
        entered.set()
        await release.wait()
        calls.append("halt_finished")
        return {"ok": True}
    async def close():
        calls.append("close")
    run.tools = SimpleNamespace(hw_estop=estop)
    run.client = SimpleNamespace(close=close)
    original_context = current_tool_context()
    async def cleanup():
        run._context_token = set_tool_context(ToolExecutionContext.from_strings(
            workspace_root=str(settings.root), session_id="cancel-fixture"))
        await run.__aexit__(asyncio.CancelledError, asyncio.CancelledError(), None)
        assert current_tool_context() == original_context
    task = asyncio.create_task(cleanup())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert calls == []
    release.set()
    await asyncio.wait_for(task, 1.)
    assert calls == ["halt_finished", "close"]
    assert run._context_token is None


@pytest.mark.parametrize("failure_at", ["connect", "read", "drift"])
async def test_post_exit_failure_attempts_halt(settings, tmp_path, monkeypatch, failure_at):
    settings = replace(settings, failure_policy="stop")
    calls = []
    final = snapshot()
    async def open_boundary(self):
        calls.append("connect")
        if failure_at == "connect" and len(calls) == 1:
            raise Refusal("readback connection failed")
        async def estop(**kwargs):
            calls.append("halt")
            return {"ok": True}
        self.tools = SimpleNamespace(hw_estop=estop)
        self.registry = registry(final if failure_at != "drift" else replace(final, gripper_mm=1.))
        if failure_at == "read":
            async def bad_read(*args, **kwargs):
                raise Refusal("readback failed")
            self.registry.read_batch = bad_read
        async def close():
            calls.append("close")
        self.registry.close_all = close
    monkeypatch.setattr(Session, "_open", open_boundary)
    with pytest.raises(Refusal):
        await cli.verify_after_exit(settings, Evidence(tmp_path / "post-exit.json"),
                                    {"arrival": {"joint": .5, "position": 2., "orientation": 1., "gripper": .5}}, final)
    assert calls.count("halt") == 1


async def test_post_exit_success_does_not_halt(settings, tmp_path, monkeypatch):
    final = snapshot()
    async def open_boundary(self):
        self.registry = registry(final)
        async def close():
            pass
        self.registry.close_all = close
    monkeypatch.setattr(Session, "_open", open_boundary)
    result = await cli.verify_after_exit(settings, Evidence(tmp_path / "holding.json"),
                                       {"arrival": {"joint": .5, "position": 2., "orientation": 1., "gripper": .5}}, final)
    assert result["finish"] == "enabled"


def test_internal_defect_is_nonrecoverable():
    failure = Refusal("implementation failure", code="internal_defect").envelope
    assert failure.source == FailureSource.SYSTEM
    assert failure.category == "internal_defect"
    assert failure.recoverability == Recoverability.NON_RECOVERABLE


@pytest.mark.parametrize("relative", [None, [], 3, "", ".", "./old.py", "a//b", "../outside", "/absolute", "bad\x00name"])
def test_manifest_path_schema_is_fail_closed(tmp_path, relative):
    with pytest.raises(Refusal):
        layout.checked_path(tmp_path, relative)


@pytest.mark.parametrize("source", ["from . import old", "from pkg import old", "import old"])
def test_relative_and_package_imports_block_cleanup(tmp_path, source):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/consumer.py").write_text(source)
    assert layout.active_references(tmp_path, {"pkg/old.py"})


def test_node_lifecycle_is_serialized(settings):
    with node.lifecycle_lock(settings):
        with pytest.raises(Refusal, match="another node lifecycle"):
            with node.lifecycle_lock(settings):
                pytest.fail("concurrent lifecycle must not acquire the same inode")
    with node.lifecycle_lock(settings):
        pass


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_supervisor_absent_is_unsupervised_but_a_failing_query_is_not(settings, monkeypatch):
    """Only a missing ``systemctl`` may be read as "nobody supervises this".

    A failing query on a host that has systemd is exactly the state in which
    spawning a rival to a supervised node looks safe: the supervisor would then
    restart its child into a second CAN owner and a second video owner.
    """
    def missing(*args, **kwargs):
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(node.subprocess, "run", missing)
    assert node.supervisor(settings) == ""

    def failing(*args, **kwargs):
        raise OSError("bus unavailable")

    monkeypatch.setattr(node.subprocess, "run", failing)
    with pytest.raises(Refusal, match="supervisor state could not be inspected"):
        node.supervisor(settings)


@pytest.mark.parametrize("stdout,expected", [("active\n", node.SYSTEMD_UNIT), ("inactive\n", ""), ("", "")])
def test_supervisor_names_the_unit_that_owns_the_process(settings, monkeypatch, stdout, expected):
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        return _completed(stdout=stdout)

    monkeypatch.setattr(node.subprocess, "run", run)
    assert node.supervisor(settings) == expected
    assert seen == [["systemctl", "--user", "is-active", node.SYSTEMD_UNIT]]


def test_preview_binds_supervisor_state_into_the_approved_plan(settings, monkeypatch):
    """An approval granted under one owner must not execute under the other.

    Start means "spawn" when unsupervised and "ask systemd" when supervised, so
    the plan digest has to move with that fact or a stale approval would silently
    cross the boundary.
    """
    monkeypatch.setattr(node, "processes", lambda settings: [])
    monkeypatch.setattr(node, "supervisor", lambda settings: node.SYSTEMD_UNIT)
    supervised = node.preview(settings, "status")
    assert supervised["supervisor"] == node.SYSTEMD_UNIT

    monkeypatch.setattr(node, "supervisor", lambda settings: "")
    unsupervised = node.preview(settings, "status")
    assert unsupervised["supervisor"] == ""
    assert node.digest_plan(unsupervised) != node.digest_plan(supervised)


def test_supervised_start_reports_running_without_asking_systemd(settings, tmp_path, monkeypatch):
    """An already healthy node needs no service action at all."""
    evidence = Evidence(tmp_path / "supervised-start.json")
    calls = []
    monkeypatch.setattr(node, "processes", lambda settings: [4242])
    monkeypatch.setattr(node, "ready", lambda settings: True)
    monkeypatch.setattr(node.subprocess, "run", lambda argv, **kwargs: calls.append(argv))

    result = node._execute_supervised(settings, "start", node.SYSTEMD_UNIT,
                                      {"log": str(tmp_path / "serve.log")}, evidence)

    assert result == {"ok": True, "status": "already_running", "pids": [4242],
                      "supervisor": node.SYSTEMD_UNIT, "lhp_ready": True}
    assert calls == []
    assert all(event["event"] != "node_supervised_action" for event in evidence.events)


def test_supervised_start_delegates_and_waits_for_readiness(settings, tmp_path, monkeypatch):
    evidence = Evidence(tmp_path / "supervised-delegate.json")
    calls = []
    running = []

    def run(argv, **kwargs):
        calls.append(argv)
        running.append(5150)
        return _completed()

    monkeypatch.setattr(node, "processes", lambda settings: list(running))
    monkeypatch.setattr(node, "ready", lambda settings: bool(running))
    monkeypatch.setattr(node.subprocess, "run", run)

    result = node._execute_supervised(settings, "restart", node.SYSTEMD_UNIT,
                                      {"log": str(tmp_path / "serve.log")}, evidence)

    assert calls == [["systemctl", "--user", "restart", node.SYSTEMD_UNIT]]
    assert result["status"] == "running"
    assert result["pids"] == [5150]
    assert result["supervisor"] == node.SYSTEMD_UNIT
    assert [event["event"] for event in evidence.events] == ["node_supervised_action", "node_ready"]


def test_supervised_start_fails_closed_on_a_refused_service_action(settings, tmp_path, monkeypatch):
    evidence = Evidence(tmp_path / "supervised-refused.json")
    monkeypatch.setattr(node, "processes", lambda settings: [])
    monkeypatch.setattr(node.subprocess, "run",
                        lambda argv, **kwargs: _completed(stderr="Unit not found.", returncode=5))

    with pytest.raises(Refusal, match="systemd could not start"):
        node._execute_supervised(settings, "start", node.SYSTEMD_UNIT,
                                 {"log": str(tmp_path / "serve.log")}, evidence)


def test_supervised_start_refuses_a_second_owner(settings, tmp_path, monkeypatch):
    """A supervised start that yields two owners is the race, not a slow start."""
    evidence = Evidence(tmp_path / "supervised-race.json")
    owners = []

    def run(argv, **kwargs):
        owners.extend([5150, 5151])
        return _completed()

    monkeypatch.setattr(node, "processes", lambda settings: list(owners))
    monkeypatch.setattr(node, "ready", lambda settings: True)
    monkeypatch.setattr(node.subprocess, "run", run)

    with pytest.raises(Refusal, match="multiple owners detected"):
        node._execute_supervised(settings, "start", node.SYSTEMD_UNIT,
                                 {"log": str(tmp_path / "serve.log")}, evidence)


@pytest.mark.parametrize("value", [None, 2, [1] * 6, [None] * 6])
def test_manual_intervals_have_typed_refusal(value):
    with pytest.raises(ValueError):
        replace(Settings(), manual_joint_limits_deg=value).validate()


@pytest.mark.parametrize("failed_resource", ["registry", "client"])
async def test_successful_run_close_failure_inherits_halt_duty(settings, tmp_path, monkeypatch, failed_resource):
    settings = replace(settings, failure_policy="stop")
    run = Session(settings, Evidence(tmp_path / "close-failure.json"), execute=True)
    run.touched = run.success = True
    calls = []
    async def failed_close():
        calls.append("failed_close")
        raise OSError("close failed after successful dispatch")
    async def good_close():
        calls.append("good_close")
    run.registry = SimpleNamespace(close_all=failed_close if failed_resource == "registry" else good_close)
    run.client = SimpleNamespace(close=failed_close if failed_resource == "client" else good_close)
    async def open_boundary(self):
        assert self.execute is False  # No private issuer or motion authority on reconnect.
        calls.append("safety_connect")
        async def estop(**kwargs):
            calls.append("halt")
            return {"ok": True}
        self.tools = SimpleNamespace(hw_estop=estop)
        self.client = SimpleNamespace(close=good_close)
    monkeypatch.setattr(Session, "_open", open_boundary)
    with pytest.raises(Refusal, match="cleanup unconfirmed") as failure:
        await run.__aexit__(None, None, None)
    assert failure.value.envelope.side_effect_state == SideEffectState.UNKNOWN
    assert calls.count("halt") == 1
    assert calls.index("failed_close") < calls.index("safety_connect") < calls.index("halt")


@pytest.fixture
async def governed_node(settings, tmp_path, monkeypatch, request):
    """Production gate/permit/node wiring; only socket I/O and key loading are replaced."""
    private, _ = generate_keypair()  # Ephemeral fixture authority, never persisted or deployed.
    issuer = PermitIssuer(private)
    declaration = context()
    effect = getattr(request, "param", None)
    config = {"values": feedback_values(snapshot())}
    if effect is not None:
        config["failures"] = [{"channel_id": "arm.speed_pct", "on_call": 1,
                               "side_effect_state": effect, "failure_code": "offline_device_failure"}]
    declaration = replace(declaration,
        provenance=replace(declaration.provenance, verified_by="offline fixture", verified_at=1.),
        transport=TransportRef(kind="mock", config=config))
    provider = SimpleNamespace(kind="offline_fixture", discover=lambda: (declaration,))
    owner = HardwareRegistry(HardwareSettings(enabled=True, persist_readings=False,
        stream_enabled=False), providers=[provider])
    report = owner.load()
    assert report.admitted == (settings.device_id,)
    verifier = PermitVerifier({issuer.key_id: issuer.public_key}, node_id=settings.node_id)
    server = LhpNodeServer(owner, node_id=settings.node_id, write_mode="permit_required",
                           permit_verifier=verifier)
    requests = []
    class ObservedClient(DirectLhpClient):
        async def request(self, method, params=None):
            if method in ("write", "write_batch", "halt"):
                requests.append((method, params))
            return await super().request(method, params)
    def socket_boundary(*args, **kwargs):
        return ObservedClient(server)
    def existing_key_boundary(path, *, auto_generate):
        assert auto_generate is False
        return issuer
    monkeypatch.setattr(session_module, "SocketLhpClient", socket_boundary)
    monkeypatch.setattr(lhp_transport, "SocketLhpClient", socket_boundary)
    monkeypatch.setattr(PermitIssuer, "from_key_file", existing_key_boundary)
    token = tmp_path / "fixture-token"
    token.write_text("offline-only")
    configured = replace(settings, token_file=token)
    transport = await owner.transport(settings.device_id)
    try:
        yield SimpleNamespace(settings=configured, declaration=declaration, owner=owner,
                              transport=transport, server=server, requests=requests, effect=effect)
    finally:
        await owner.close_all()


async def test_operational_session_uses_real_governor_and_single_use_node_permit(governed_node, tmp_path):
    fixture = governed_node
    run = Session(fixture.settings, Evidence(tmp_path / "governed.json"), execute=True,
                  digest=unverified_digest(fixture.declaration), approval_note="offline exact consent")
    original_context = current_tool_context()
    async with run:
        run.bind_steps([motion.step("speed", (("arm.speed_pct", 10.),), configure=True)])
        with pytest.raises(Refusal, match="complete approved"):
            await run.command("speed", (("arm.speed_pct", 11.),), configure=True)
        await run.command("speed", (("arm.speed_pct", 10.),), configure=True)
        assert fixture.transport.write_log == (("arm.speed_pct", 10.),)
        with pytest.raises(Refusal, match="complete approved"):
            await run.command("speed", (("arm.speed_pct", 10.),), configure=True)
        # An unapproved request through the same real tools cannot obtain a permit.
        denied = await run.tools.hw_configure(device_id=fixture.settings.device_id,
                                              channel_id="arm.speed_pct", value=11., dry_run=False)
        assert denied["ok"] is False
        assert fixture.transport.write_attempts("arm.speed_pct") == 1
        run.success = True
    assert current_tool_context() == original_context
    writes = [params for method, params in fixture.requests if method == "write"]
    assert len(writes) == 1 and writes[0]["permit"]["action_digest"]
    client = DirectLhpClient(fixture.server)
    await client.connect()
    try:
        params = {"device_id": fixture.settings.device_id, "channel_id": "arm.speed_pct", "value": 10.}
        absent = await client.request("write", params)
        assert absent["failure_code"] == "permit_required"
        altered = await client.request("write", {**writes[0], "value": 11.})
        assert altered["failure_code"] == "permit_digest_mismatch"
        replay = await client.request("write", writes[0])
        assert replay["failure_code"] == "permit_replayed"
        assert fixture.transport.write_attempts("arm.speed_pct") == 1
    finally:
        await client.close()
    assert not any(method == "halt" for method, _ in fixture.requests)


@pytest.mark.parametrize("governed_node", ["none", "partial", "unknown", "committed"], indirect=True)
async def test_real_governed_transport_failure_preserves_effect_and_halts(governed_node, tmp_path):
    fixture = governed_node
    run = Session(replace(fixture.settings, failure_policy="stop"), Evidence(tmp_path / "effect.json"), execute=True,
                  digest=unverified_digest(fixture.declaration), approval_note="offline exact consent")
    with pytest.raises(Refusal) as failure:
        async with run:
            run.bind_steps([motion.step("speed", (("arm.speed_pct", 10.),), configure=True)])
            await run.command("speed", (("arm.speed_pct", 10.),), configure=True)
    command = next(row for row in run.evidence.events if row["event"] == "speed")
    assert command["result"]["side_effect_state"] == fixture.effect
    # A write that never landed leaves the arm standing: no stop is owed, and the exit
    # verifies the posture instead of releasing it. Every other verdict halts.
    stopped = fixture.effect != "none"
    assert failure.value.envelope.side_effect_state.value == fixture.effect
    assert fixture.transport.write_attempts("arm.speed_pct") == 1
    assert sum(method == "halt" for method, _ in fixture.requests) == int(stopped)
    assert next(row for row in run.evidence.events if row["event"] == "posture_check")["own_stop"] is stopped


@pytest.mark.parametrize("summary,report_write_fails", [(False, False), (True, False), (True, True)])
def test_cli_mode_loss_after_committed_writes_never_reports_none(
        governed_node, tmp_path, monkeypatch, capsys, summary, report_write_fails):
    """Drive the real CLI/governor/node; inject mode loss at the transport I/O boundary."""
    fixture = governed_node
    configured = fixture.settings
    config_path, preview_path, report_path = (tmp_path / name for name in
                                              ("config.json", "preview.json", "mode-loss.json"))
    config_path.write_text(encode(asdict(configured)))
    start = snapshot((0., -2., 0., 20., -40., -100.))
    for channel_id, value in feedback_values(start).items():
        fixture.transport.set_value(channel_id, value)
    plan = motion.preview(fixture.declaration, start, configured, operation="initialize")
    preview_path.write_text(encode(cli.preview_result(plan)))
    real_read = fixture.transport.read
    sequence = 1

    async def read(channel_id):
        nonlocal sequence
        if channel_id == "arm.telemetry":
            sequence += 1
            fixture.transport.set_value(channel_id, snapshot(seq=sequence).telemetry)
        if channel_id == "arm.status" and fixture.transport.write_attempts(health.JOINTS[-1]):
            fixture.transport.set_value(channel_id, {**snapshot().status, "ctrl_mode": 0})
        return await real_read(channel_id)

    monkeypatch.setattr(fixture.transport, "read", read)
    if report_write_fails:
        real_open = Path.open

        def open_boundary(path, mode="r", *args, **kwargs):
            if path == report_path and mode == "w":
                raise OSError("offline report storage unavailable")
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_boundary)
    argv = ["--config", str(config_path), "--json", str(report_path)]
    if summary:
        argv.append("--summary")
    argv += ["arm", "initialize", "--execute", "--preview-file", str(preview_path),
             "--approved-plan-hash", motion.digest_plan(plan), "--approval-note", "offline consent",
             "--approved-digest", unverified_digest(fixture.declaration), "--clearance-confirmed"]
    assert cli.main(argv) == 1
    output = capsys.readouterr()
    failure = json.loads(output.out)["failure"]
    assert failure["side_effect_state"] == "committed"
    assert failure["failure_code"] == "piper_operations_refused"
    assert "ctrl_mode=0" in failure["message"]
    expected = (("arm.speed_pct", configured.speed_pct), ("arm.enable", 1),
                *tuple(zip(health.JOINTS, plan["origin_joints"])))
    # A lost mode does not authorize an unpreviewed enable at session exit.
    assert fixture.transport.write_log == expected
    assert fixture.transport.halt_calls == 0
    events = [json.loads(line) for line in report_path.with_suffix(".events.ndjson").read_text().splitlines()]
    assert [row["result"]["side_effect_state"] for row in events
            if row["event"] in ("speed", "enable", "initialize")] == ["committed"] * 3
    names = [row["event"] for row in events]
    assert "damped_stop" not in names
    assert names.index("posture_check") < names.index("posture_unresolved")
    check = next(row for row in events if row["event"] == "posture_check")
    assert check["own_stop"] is False and check["state"] == "unheld"
    assert check["ctrl_mode"] == 0
    unresolved = next(row for row in events if row["event"] == "posture_unresolved")
    assert unresolved["operational_complete"] is False
    assert "approve a separate" in unresolved["next_step"]
    assert "not holding its posture" in output.err
    assert preview_path.with_suffix(".claimed").exists()
    saved = json.loads(report_path.read_text())
    if report_write_fails:
        assert saved["status"] == "in_progress" and "Evidence write failed" in output.err
    else:
        assert saved["failure"] == failure and saved["events"] == events


async def test_refusal_before_governed_dispatch_has_no_effect(governed_node, tmp_path):
    fixture = governed_node
    evidence = Evidence(tmp_path / "preflight.json")
    run = Session(fixture.settings, evidence, execute=True,
                  digest=unverified_digest(fixture.declaration), approval_note="offline exact consent")
    with pytest.raises(Refusal) as failure:
        async with run:
            run.bind_steps([motion.step("speed", (("arm.speed_pct", 10.),), configure=True)])
            await run.command("speed", (("arm.speed_pct", 11.),), configure=True)
    assert evidence.failure(failure.value)["failure"]["side_effect_state"] == SideEffectState.NONE
    assert fixture.transport.write_log == () and fixture.transport.halt_calls == 0


async def test_uncertain_dispatch_is_not_erased_by_successful_halt(governed_node, tmp_path, monkeypatch):
    fixture = governed_node
    evidence = Evidence(tmp_path / "cancelled-write.json")
    entered = asyncio.Event()

    async def write(channel_id, value):
        entered.set()
        await asyncio.Future()  # External device never acknowledges this attempted dispatch.

    monkeypatch.setattr(fixture.transport, "write", write)
    run = Session(replace(fixture.settings, failure_policy="stop"), evidence, execute=True,
                  digest=unverified_digest(fixture.declaration), approval_note="offline exact consent")

    async def execute():
        async with run:
            run.bind_steps([motion.step("speed", (("arm.speed_pct", 10.),), configure=True)])
            await run.command("speed", (("arm.speed_pct", 10.),), configure=True)

    task = asyncio.create_task(execute())
    await asyncio.wait_for(entered.wait(), 1.)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1.)
    report = evidence.failure(Refusal("run cancelled"))
    assert report["failure"]["side_effect_state"] == SideEffectState.UNKNOWN
    assert fixture.transport.halt_calls == 1 and fixture.transport.write_log == ()
    check = next(row for row in evidence.events if row["event"] == "posture_check")
    assert check["own_stop"] is True and check["state"] == "held"


async def test_real_governed_batch_binds_every_ordered_target(governed_node, tmp_path):
    fixture = governed_node
    commands = (("gripper.position", 60.), ("gripper.effort", 1.))
    run = Session(fixture.settings, Evidence(tmp_path / "batch.json"), execute=True,
                  digest=unverified_digest(fixture.declaration), approval_note="offline exact batch")
    async with run:
        run.bind_steps([motion.step("jaws", commands)])
        with pytest.raises(Refusal, match="complete approved"):
            await run.command("jaws", tuple(reversed(commands)))
        await run.command("jaws", commands)
        assert fixture.transport.write_log == commands
        run.success = True
    batches = [params for method, params in fixture.requests if method == "write_batch"]
    assert len(batches) == 1 and batches[0]["permit"]["action_digest"]
    client = DirectLhpClient(fixture.server)
    await client.connect()
    try:
        reordered = await client.request("write_batch", {**batches[0],
            "commands": list(reversed(batches[0]["commands"]))})
        assert reordered["ok"] is False and reordered["side_effect_state"] == "none"
        assert len(reordered["outcomes"]) == len(commands)
        assert all(row["failure_code"] == "permit_digest_mismatch" for row in reordered["outcomes"])
        assert fixture.transport.write_log == commands
    finally:
        await client.close()


@pytest.mark.parametrize("changed", [False, True])
def test_node_signal_uses_creation_bound_kernel_handle(settings, tmp_path, monkeypatch, changed):
    calls = []
    def pgrep(args, **kwargs):
        assert args[:2] == ["pgrep", "-f"]
        return SimpleNamespace(returncode=0, stdout="42\n")
    def proc_stat(path, *args, **kwargs):
        assert str(path) == "/proc/42/stat"
        return "42 (node) " + " ".join(["S"] + ["0"] * 18 + ["101" if changed else "100"])
    monkeypatch.setattr(node.subprocess, "run", pgrep)
    monkeypatch.setattr(Path, "read_text", proc_stat)
    monkeypatch.setattr(node.os, "pidfd_open", lambda pid, flags: 7, raising=False)
    monkeypatch.setattr(node.signal, "pidfd_send_signal", lambda fd, sig: calls.append((fd, sig)), raising=False)
    monkeypatch.setattr(node.os, "close", lambda fd: calls.append(("closed", fd)))
    evidence = Evidence(tmp_path / "pid.json")
    if changed:
        with pytest.raises(Refusal, match="creation identity changed"):
            node.terminate_exact(settings, 42, "100", evidence)
        assert calls == [("closed", 7)]
        assert evidence.events == []
    else:
        node.terminate_exact(settings, 42, "100", evidence)
        assert calls == [(7, node.signal.SIGTERM), ("closed", 7)]
        assert evidence.events[0]["process_start"] == "100"


def test_node_control_interpreter_configuration(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"node_control_python": "/usr/bin/python3"}))
    assert Settings.load(path).node_control_python == Path("/usr/bin/python3")
    path.write_text(json.dumps({"node_control_python": None}))
    assert Settings.load(path).node_control_python is None
    with pytest.raises(ValueError, match="node_control_python"):
        replace(Settings(), node_control_python="python3").validate()


@pytest.mark.parametrize("result,state", [
    ((0, '{"ok":true,"signal_attempted":true}'), None),
    ((1, '{"ok":false,"signal_attempted":false}'), SideEffectState.NONE),
    ((1, '{"ok":false,"signal_attempted":true}'), SideEffectState.UNKNOWN),
    ((0, 'null'), SideEffectState.UNKNOWN),
    ((0, 'invalid'), SideEffectState.UNKNOWN),
    (None, SideEffectState.UNKNOWN),
])
def test_opt_in_node_control_uses_isolated_pidfd_helper(settings, tmp_path, monkeypatch, result, state):
    configured = replace(settings, node_control_python=Path("/usr/bin/python3"))
    evidence = Evidence(tmp_path / "helper.json")
    monkeypatch.delattr(node.os, "pidfd_open", raising=False)
    monkeypatch.setattr(node, "processes", lambda s: [42])
    monkeypatch.setattr(node, "process_start", lambda pid: "100")
    real_stat, real_resolve = Path.stat, Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda p, **kw: p if p == configured.node_control_python else real_resolve(p, **kw))
    monkeypatch.setattr(Path, "stat", lambda p, **kw: SimpleNamespace(st_uid=0, st_mode=0o100755)
                        if p == configured.node_control_python else real_stat(p, **kw))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        assert args[:3] == ["/usr/bin/python3", "-I", "-c"]
        assert args[3] == node._PIDFD_TERMINATE
        assert args[4:6] == ["42", "100"]
        assert json.loads(args[6]) == node.argv(configured)
        assert kwargs["timeout"] == settings.request_timeout_s
        if result is None:
            raise node.subprocess.TimeoutExpired(args, 10)
        return SimpleNamespace(returncode=result[0], stdout=result[1])
    monkeypatch.setattr(node.subprocess, "run", run)
    if state is None:
        node.terminate_exact(configured, 42, "100", evidence)
    else:
        with pytest.raises(Refusal) as failure:
            node.terminate_exact(configured, 42, "100", evidence)
        assert failure.value.envelope.side_effect_state == state
    assert len(calls) == 1
    assert evidence.events[0]["event"] == "node_term_attempt"


def test_missing_pidfd_does_not_infer_interpreter(settings, tmp_path, monkeypatch):
    evidence = Evidence(tmp_path / "no-helper.json")
    monkeypatch.delattr(node.os, "pidfd_open", raising=False)
    monkeypatch.setattr(node.subprocess, "run", lambda *a, **k: pytest.fail("must not signal"))
    with pytest.raises(Refusal, match="node_control_python"):
        node.terminate_exact(settings, 42, "100", evidence)
    assert not evidence.events


@pytest.mark.parametrize("unsafe_mode,unsafe_owner", [(0o100777, 0), (0o100755, 501), (0o40755, 0)])
def test_node_control_rejects_untrusted_interpreter(settings, tmp_path, monkeypatch, unsafe_mode, unsafe_owner):
    configured = replace(settings, node_control_python=Path("/usr/bin/python3"))
    evidence = Evidence(tmp_path / "unsafe-helper.json")
    monkeypatch.setattr(Path, "resolve", lambda p, **kw: p)
    monkeypatch.setattr(Path, "stat", lambda p: SimpleNamespace(st_uid=unsafe_owner, st_mode=unsafe_mode))
    with pytest.raises(Refusal, match="root-owned"):
        node.terminate_with_interpreter(configured, 42, "100", evidence)
    assert not evidence.events


@pytest.mark.parametrize("changed", [None, "start", "argv"])
def test_isolated_helper_binds_creation_and_argv(monkeypatch, capsys, changed):
    import os
    import signal
    import sys
    calls = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid, flags: 7, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig: calls.append((fd, sig)), raising=False)
    monkeypatch.setattr(os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(sys, "argv", ["helper", "42", "100", '["python", "leap"]'])
    monkeypatch.setattr(Path, "read_text", lambda p: "42 (node) " +
                        " ".join(["S"] + ["0"] * 18 + ["101" if changed == "start" else "100"]))
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"python\0other\0" if changed == "argv" else b"python\0leap\0")
    with pytest.raises(SystemExit) as exit_info:
        exec(compile(node._PIDFD_TERMINATE, "<pidfd-helper>", "exec"), {})
    report = json.loads(capsys.readouterr().out)
    assert report == {"ok": changed is None, "signal_attempted": changed is None}
    assert exit_info.value.code == (0 if changed is None else 1)
    assert calls == ([(7, signal.SIGTERM), ("close", 7)] if changed is None else [("close", 7)])


async def test_service_readiness_is_hello_list_only(governed_node, monkeypatch):
    fixture = governed_node
    requests = []
    class ReadinessClient(DirectLhpClient):
        async def request(self, method, params=None):
            requests.append(method)
            return await super().request(method, params)
    monkeypatch.setattr(node, "SocketLhpClient", lambda *args, **kwargs: ReadinessClient(fixture.server))
    assert await node._ready(fixture.settings) is True
    assert requests == ["list"]
    assert fixture.transport.write_log == ()


@pytest.fixture
def reference_scope(tmp_path):
    root, consumers = tmp_path / "piper", tmp_path / "consumers"
    root.mkdir()
    consumers.mkdir()
    (root / "old.py").write_text("pass\n")
    (root / "tools").mkdir()
    (root / "tools/helper.py").write_text("pass\n")
    return root, consumers


def test_external_references_are_scoped_candidates_without_matching_text(reference_scope, tmp_path):
    root, consumers = reference_scope
    script = consumers / "consumer.sh"
    script.write_text(f"PRIVATE_LITERAL python {root}/old.py\npython 'tools/helper.py'\n")
    (tmp_path / "unrequested.sh").write_text("old.py")
    result = layout.reference_report(root, [consumers])
    assert result["scope_complete"] is True and result["cleanup_authorized"] is False
    assert result["inspected"] == [{"path": str(script), "sha256": layout.sha256(script)}]
    assert {row["source"] for row in result["matches"]} == {".", "old.py", "tools/helper.py"}
    assert "PRIVATE_LITERAL" not in json.dumps(result)
    assert all(row["consumer"] == str(script) for row in result["matches"])
    script.write_text("older.py old.py.extra notold.py\n")
    assert layout.reference_report(root, [script])["matches"] == []


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "100", None])
def test_reference_budget_is_typed(reference_scope, budget):
    root, consumers = reference_scope
    with pytest.raises(Refusal, match="positive integer"):
        layout.reference_report(root, [consumers], max_bytes=budget)


@pytest.mark.parametrize("kind", ["missing", "oversized", "binary", "unselected", "sensitive", "excluded",
                                  "link", "ancestor_link", "child_link", "unreadable"])
def test_reference_inspection_gaps_never_imply_complete_scope(reference_scope, tmp_path, monkeypatch, kind):
    root, consumers = reference_scope
    scope = consumers
    target = consumers / "input.sh"
    target.write_text("old.py")
    if kind == "missing":
        scope = tmp_path / "missing"
    elif kind == "oversized":
        target.write_bytes(b"x" * 21)
    elif kind == "binary":
        target.write_bytes(b"\xff\xfe")
    elif kind == "unselected":
        target = target.rename(consumers / "input.bin")
    elif kind == "sensitive":
        target = target.rename(consumers / ".env.production")
    elif kind == "excluded":
        scope = consumers / "vault"
        scope.mkdir()
        (scope / "secret.sh").write_text("NEVER_PRINT old.py")
    elif kind in ("link", "ancestor_link"):
        link = tmp_path / "alias"
        link.symlink_to(consumers, target_is_directory=True)
        scope = link if kind == "link" else link / target.name
    elif kind == "child_link":
        (consumers / "child").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "unreadable":
        real_open = layout.os.open
        def unreadable(path, flags, *args, **kwargs):
            if path == "input.sh":
                raise PermissionError("NEVER_PRINT")
            return real_open(path, flags, *args, **kwargs)
        monkeypatch.setattr(layout.os, "open", unreadable)
    result = layout.reference_report(root, [scope], max_bytes=20)
    assert result["gaps"] and result["scope_complete"] is False
    assert result["cleanup_authorized"] is False
    assert "NEVER_PRINT" not in json.dumps(result)
    if kind not in ("child_link",):
        assert result["inspected"] == []


@pytest.mark.parametrize("component", ["file", "parent"])
def test_reference_open_refuses_link_replacement(reference_scope, tmp_path, monkeypatch, component):
    root, consumers = reference_scope
    target = consumers / "input.sh"
    target.write_text("old.py")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / target.name).write_text("MUST_NOT_READ old.py")
    real_open = layout.os.open
    changed = False
    def replace_before_open(path, flags, *args, **kwargs):
        nonlocal changed
        expected = target.name if component == "file" else consumers.name
        if path == expected and not changed:
            changed = True
            if component == "file":
                target.rename(consumers / "saved.sh")
                target.symlink_to(outside / target.name)
            else:
                consumers.rename(tmp_path / "saved-consumers")
                consumers.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(layout.os, "open", replace_before_open)
    report = layout.reference_report(root, [target])
    assert changed and report["gaps"] and report["inspected"] == []
    assert "MUST_NOT_READ" not in json.dumps(report)


def test_reference_growth_is_not_accepted(reference_scope, monkeypatch):
    root, consumers = reference_scope
    target = consumers / "input.sh"
    target.write_text("old.py")
    original = layout.os.fstat
    calls = 0
    def grow(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_text("old.py plus more content")
        return original(descriptor)
    monkeypatch.setattr(layout.os, "fstat", grow)
    report = layout.reference_report(root, [target], max_bytes=10)
    assert report["inspected"] == [] and report["gaps"]


@pytest.fixture
def process_scope(tmp_path):
    root, proc = tmp_path / "piper", tmp_path / "proc"
    root.mkdir()
    proc.mkdir()
    process = proc / "42"
    process.mkdir()
    (process / "stat").write_text("42 (worker) " + " ".join(["S"] + ["0"] * 18 + ["100"]))
    (process / "cwd").symlink_to(root, target_is_directory=True)
    (process / "cmdline").write_bytes(b"python\x00" + str(root / "old.py").encode() + b"\x00PRIVATE_ARG\x00")
    (process / "fd").mkdir()
    (process / "fd/3").symlink_to(root / "old.py")
    return root, proc, process


def test_process_snapshot_redacts_argv_and_targets(process_scope):
    root, proc, _ = process_scope
    report = layout.process_references(root, proc=proc)
    assert report["matches"] == [{"pid": 42, "process_start": "100",
                                   "references": ["argv_root", "cwd", "open_file"]}]
    assert report["gaps"] == [] and report["cleanup_authorized"] is False
    assert "PRIVATE_ARG" not in json.dumps(report) and "old.py" not in json.dumps(report)


def test_process_inspection_unsupported_and_other_user(process_scope, tmp_path, monkeypatch):
    root, proc, _ = process_scope
    assert layout.process_references(root, proc=tmp_path / "missing")["supported"] is False
    monkeypatch.setattr(layout.os, "getuid", lambda: -1, raising=False)
    assert layout.process_references(root, proc=proc)["matches"] == []


@pytest.mark.parametrize("failure", ["identity", "malformed", "cwd", "argv", "fd", "overflow", "disappeared"])
def test_process_snapshot_reports_gaps_and_never_accepts_changed_identity(process_scope, monkeypatch, failure):
    root, proc, process = process_scope
    original_text, original_link, original_open = Path.read_text, layout.os.readlink, Path.open
    reads = 0
    def stat_text(path, *args, **kwargs):
        nonlocal reads
        if path == process / "stat":
            reads += 1
            if failure == "disappeared" and reads == 2:
                raise FileNotFoundError("PRIVATE_ERROR")
            if failure == "malformed":
                return "malformed stat"
            if failure == "identity" and reads == 2:
                return original_text(path).replace("100", "101")
        return original_text(path, *args, **kwargs)
    def link_text(path, *args, **kwargs):
        if (failure == "cwd" and path == process / "cwd" or
                failure == "fd" and path.parent == process / "fd"):
            raise FileNotFoundError("PRIVATE_ERROR")
        return original_link(path, *args, **kwargs)
    def open_file(path, *args, **kwargs):
        if failure == "argv" and path == process / "cmdline":
            raise PermissionError("PRIVATE_ERROR")
        return original_open(path, *args, **kwargs)
    if failure == "overflow":
        (process / "cmdline").write_bytes(b"x" * 64_001)
    monkeypatch.setattr(Path, "read_text", stat_text)
    monkeypatch.setattr(layout.os, "readlink", link_text)
    monkeypatch.setattr(Path, "open", open_file)
    report = layout.process_references(root, proc=proc)
    assert report["gaps"] and report["cleanup_authorized"] is False
    assert "PRIVATE_ERROR" not in json.dumps(report)
    if failure in ("identity", "malformed", "disappeared"):
        assert report["matches"] == []
    else:
        assert report["matches"]  # A disappearing field must not erase other observed references.


def test_layout_inspect_optional_dispatch_and_previous_namespace(reference_scope, tmp_path, monkeypatch):
    root, consumers = reference_scope
    configured = replace(Settings(), root=root)
    evidence = Evidence(tmp_path / "inspect.json")
    previous = cli.local(SimpleNamespace(group="layout", operation="inspect"), configured, evidence)
    assert previous["ok"] is True and "references" not in previous and "processes" not in previous
    (consumers / "consumer.sh").write_text("old.py")
    monkeypatch.setattr(layout, "process_references", lambda path: {"supported": False, "matches": []})
    args = cli.parser().parse_args(["layout", "inspect", "--reference-root", str(consumers),
        "--reference-root", str(consumers / "consumer.sh"), "--reference-max-bytes", "100", "--processes"])
    report = cli.local(args, configured, evidence)
    assert report["references"]["max_bytes"] == 100 and len(report["references"]["locations"]) == 2
    assert report["references"]["matches"] and report["processes"]["supported"] is False
