"""Piper transport contracts against fake pyAgxArm; no CAN hardware or vendor import.

The fake exposes SDK-native rad/m/N and feeds the production packet callback.
Tests pin model/firmware selection, read-only bring-up, governed energize,
held vectors, freshness, batch ordering and conservative side-effect verdicts.
"""
from __future__ import annotations

import asyncio
import importlib
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.drivers._piper_feedback import CAN_GROUPS
from leapflow.hardware.drivers._piper_motion import (
    SPACE_JOINT, PiperMotion, firmware_version, joints_to_sdk, pose_to_sdk,
)
from leapflow.hardware.drivers.piper import PiperArmTransport
from leapflow.hardware.registry import _DEVICE_ID_ALLOWED
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED, SIDE_EFFECT_NONE, SIDE_EFFECT_UNKNOWN, TransportError,
)

_EXAMPLE = Path(__file__).parents[1] / "src/leapflow/hardware/examples/piper_arm.yaml"
_BASE_CONFIG = {"can_name": "can0", "robot": "piper_x", "firmware_profile": "v189",
                "enable_timeout_s": .12, "enable_settle_s": .02, "mode_switch_timeout_s": .12}
_SEED_JOINTS = [10., 20., -30., 40., 50., -160.]
_SEED_POSE = [200., 0., 300., 179., 0., 90.]


def _context() -> HardwareContext:
    return HardwareContext.from_mapping(yaml.safe_load(_EXAMPLE.read_text()))


def _message(body: Any) -> Any:
    return SimpleNamespace(msg=body) if body is not None else None


class _FakeGripper:
    def __init__(self, arm: _FakeArm) -> None:
        self.arm = arm
        self.stroke = .02
        self.force = .5
        self.down = False
        self.fail = False
        self.calls: list[tuple[float, float]] = []
        self.modes: list[int] = []

    def get_gripper_status(self) -> Any:
        return None if self.down else _message(SimpleNamespace(value=self.stroke, force=self.force))

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.calls.append((value, force))
        self.modes.append(self.arm.ctrl_mode)
        self.arm.events.append("gripper")
        if self.fail:
            raise OSError("simulated CAN failure")
        self.stroke, self.force = value, force


class _FakeArm:
    """Public pyAgxArm API stand-in; receives independently of getters."""

    def __init__(self, config: dict, *, silent: bool = False, firmware: str = "S-V1.9-0") -> None:
        self.config = config
        self.silent = silent
        self.firmware = firmware
        self.connected = False
        self.ok = True
        self.comm_error = False
        self.connect_calls = self.disconnect_calls = 0
        self.limits_enabled = False
        self.joints: Any = joints_to_sdk(_SEED_JOINTS)
        self.pose: Any = pose_to_sdk(_SEED_POSE)
        self.ctrl_mode = self.arm_status = 0
        self.motion_status = 1
        self.mode_feedback = 1
        self.err_code = 0
        # A controller that accepts CAN control and then faults: after one clean
        # status read it answers with this fault, then clears itself and leaves the
        # arm in standby with the drives released -- what a bad enable looks like.
        self.fault_on_can = False
        self.fault_delay = 0
        self.fault_reads = 0
        self.fault_code = 0x05
        self.fault_err_code = 0x16
        self.flags: Any = [False] * 6
        self.refuse_enable = self.refuse_reset = self.ignore_mode = False
        self.ignore_move_mode = False
        self.drop_mode_frames = 0
        # pyAgxArm discards a firmware answer that beats its own fps estimate.
        self.drop_firmware_replies = 0
        self.fail_joint = self.fail_pose = False
        self.joint_calls: list[list[float]] = []
        self.pose_calls: list[tuple[str, list[float]]] = []
        self.mode_calls: list[str] = []
        self.speed_calls: list[int] = []
        self.enable_calls = self.disable_calls = self.reset_calls = self.estop_calls = 0
        self.events: list[str] = []
        self.callbacks: list[Any] = []
        self.gripper = _FakeGripper(self)

    def init_effector(self, name: str) -> _FakeGripper:
        assert name == "agx_gripper"
        self.events.append("effector")
        return self.gripper

    def get_context(self) -> Any:
        return SimpleNamespace(register_parser_packet_fun=self.callbacks.append)

    def receive(self, groups: tuple[str, ...] = CAN_GROUPS, *, dlc: int = 8) -> None:
        for group in groups:
            frame = SimpleNamespace(arbitration_id=int(group, 16), dlc=dlc, is_remote_frame=False)
            for callback in self.callbacks:
                callback(frame)

    def set_joint_limits_enabled(self, value: bool) -> None:
        self.events.append("limits")
        self.limits_enabled = value

    def connect(self) -> None:
        self.connect_calls += 1
        self.connected = True
        self.events.append("connect")
        if self.silent:
            self.joints = None

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def is_ok(self) -> bool:
        return self.ok

    def has_comm_error(self) -> bool:
        return self.comm_error

    def get_fps(self) -> float:
        return 250.

    def get_firmware(self, *, timeout: float, min_interval: float) -> Any:
        self.events.append("firmware")
        if self.drop_firmware_replies:
            self.drop_firmware_replies -= 1
            return None
        return {"software_version": self.firmware}

    def get_joint_angles(self) -> Any:
        return _message(self.joints)

    def get_flange_pose(self) -> Any:
        return _message(self.pose)

    def get_arm_status(self) -> Any:
        if self.fault_delay:
            self.fault_delay -= 1
        elif self.fault_reads:
            self.fault_reads -= 1
            if self.fault_reads:
                return _message(SimpleNamespace(ctrl_mode=0, arm_status=self.fault_code,
                                                mode_feedback=self.mode_feedback,
                                                motion_status=self.motion_status,
                                                err_code=self.fault_err_code))
            # The controller cleared it by itself and stayed in standby.
            self.arm_status = self.ctrl_mode = 0
        return _message(SimpleNamespace(ctrl_mode=self.ctrl_mode, arm_status=self.arm_status,
                                        mode_feedback=self.mode_feedback,
                                        motion_status=self.motion_status,
                                        err_code=self.err_code))

    def get_driver_states(self, index: int) -> Any:
        if self.flags is None or len(self.flags) < index:
            return None
        return _message(SimpleNamespace(
            foc_status=SimpleNamespace(driver_enable_status=self.flags[index - 1],
                                      driver_error_status=False),
            vol=24., motor_temp=26., foc_temp=32.))

    def get_motor_states(self, index: int) -> Any:
        return _message(SimpleNamespace(current=.1, torque=.2, velocity=.3))

    def enable(self) -> bool:
        self.events.append("enable")
        self.enable_calls += 1
        if not self.refuse_enable:
            self.flags = [True] * 6
        return all(self.flags or [False])

    def disable(self) -> bool:
        self.disable_calls += 1
        self.flags = [False] * 6
        return True

    def electronic_emergency_stop(self) -> None:
        self.estop_calls += 1
        self.arm_status = 1

    def reset(self) -> None:
        self.events.append("reset")
        self.reset_calls += 1
        if not self.refuse_reset:
            self.arm_status = self.ctrl_mode = 0
            self.flags = [False] * 6

    def set_motion_mode(self, mode: str) -> None:
        self.events.append(f"mode:{mode}")
        self.mode_calls.append(mode)
        if self.drop_mode_frames:
            self.drop_mode_frames -= 1
        elif not (self.ignore_mode or self.ignore_move_mode):
            self.mode_feedback = {"p": 0, "j": 1, "l": 2}[mode]
            self.ctrl_mode = 1

    def set_speed_percent(self, speed: int) -> None:
        self.events.append("speed")
        self.speed_calls.append(speed)
        if self.drop_mode_frames:
            self.drop_mode_frames -= 1
        elif not self.ignore_mode:
            self.ctrl_mode = 1
            if self.fault_on_can:
                self.fault_on_can = False
                self.fault_delay, self.fault_reads = 1, 2

    def move_j(self, joints: list[float]) -> None:
        self.events.append("joint")
        self.joint_calls.append(joints)
        if self.fail_joint:
            raise OSError("simulated CAN failure")

    def _pose_move(self, path: str, pose: list[float]) -> None:
        self.events.append(path)
        self.pose_calls.append((path, pose))
        if self.fail_pose:
            raise OSError("simulated CAN failure")

    def move_l(self, pose: list[float]) -> None:
        self._pose_move("linear", pose)

    def move_p(self, pose: list[float]) -> None:
        self._pose_move("point", pose)


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, *, silent: bool = False,
                      firmware: str = "S-V1.9-0", drop_firmware: int = 0) -> list[_FakeArm]:
    created: list[_FakeArm] = []
    real_import = importlib.import_module

    def create(config: dict) -> _FakeArm:
        arm = _FakeArm(config, silent=silent, firmware=firmware)
        arm.drop_firmware_replies = drop_firmware
        created.append(arm)
        return arm

    def resolve(robot: str, version: str) -> str:
        assert robot in ("piper_x", "piper", "piper_h", "piper_l")
        if version not in ("S-V1.9-0", "S-V1.8-8"):
            raise ValueError("invalid firmware")
        return "v188" if version == "S-V1.8-8" else "v189"

    sdk = SimpleNamespace(create_agx_arm_config=lambda **kwargs: kwargs,
                          AgxArmFactory=SimpleNamespace(create_arm=create),
                          resolve_firmware_profile=resolve)
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a, **kw: sdk if name == "pyAgxArm" else real_import(name, *a, **kw))
    return created


def _open_transport(monkeypatch: pytest.MonkeyPatch, overrides: dict | None = None, *,
                    receive: bool = True) -> tuple[PiperArmTransport, _FakeArm, Any]:
    created = _install_fake_sdk(monkeypatch)
    transport = PiperArmTransport({**_BASE_CONFIG, **(overrides or {})})
    status = asyncio.run(transport.open(_context()))
    arm = created[0]
    if receive:
        arm.receive()
    return transport, arm, status


def _write(transport: PiperArmTransport, channel: str, value: Any) -> Any:
    return asyncio.run(transport.write(channel, value))


def test_example_yaml_pins_model_limits_units_and_dh() -> None:
    context = _context()
    assert context.device_id == "robot_piper_arm"
    assert all(c in _DEVICE_ID_ALLOWED for c in context.device_id)
    assert context.transport.kind == "piper" and context.halt_supported
    assert context.transport.config["robot"] == "piper_x"
    assert context.transport.config["firmware_profile"] == "v189"
    assert context.kinematics.dof == 6
    # j6 is the one axis where the manual (±165) is narrower than the pyAgxArm
    # preset (±170); the declaration must follow the manual.
    for i, bound in ((4, 89), (5, 89), (6, 165)):
        channel = context.channel(f"joint.j{i}.position")
        assert channel.envelope.min_value == -bound and channel.envelope.max_value == bound
        assert channel.envelope.tolerance == .5
        assert context.kinematics.joint_limits[i - 1] == pytest.approx(
            [-math.radians(bound), math.radians(bound)], abs=.0004)
    assert context.channel("gripper.effort").unit == "N"
    assert context.channel("gripper.effort").quantity == "force"
    assert context.channel("ee.ry").envelope.max_value == 90
    raw = yaml.safe_load(_EXAMPLE.read_text())
    assert [link["theta_offset"] for link in raw["kinematics"]["dh"]["links"]] == [180, 7.78, 162.61, 4.62, 90, 0]
    assert not {"sdk_interface", "mode_switch", "default_effort_n_m"} & context.transport.config.keys()


def test_missing_sdk_is_a_typed_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        assert name == "pyAgxArm"
        raise ImportError("not installed")
    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(TransportError) as error:
        asyncio.run(PiperArmTransport().open(_context()))
    assert error.value.failure_code == "pyagxarm_missing"


def test_open_is_read_only_model_aware_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, status = _open_transport(monkeypatch, {"enable_on_open": True})
    assert status.connected and status.halt_supported
    assert arm.config == {"robot": "piper_x", "comm": "can", "firmeware_version": "v189", "channel": "can0"}
    assert arm.events == ["effector", "limits", "connect", "firmware"]
    assert arm.limits_enabled is True
    assert not arm.enable_calls and not arm.reset_calls and not arm.speed_calls
    assert status.metadata["firmware_version"] == "S-V1.9-0"
    assert status.metadata["enabled_motors"] == [False] * 6
    assert asyncio.run(transport.open(_context())).connected
    assert arm.connect_calls == 1


@pytest.mark.parametrize("firmware,code", [("S-V1.8-8", "piper_firmware_mismatch"),
                                          ("bad", "piper_firmware_mismatch"),
                                          ("", "piper_firmware_unavailable")])
def test_wrong_firmware_disconnects(monkeypatch: pytest.MonkeyPatch, firmware: str, code: str) -> None:
    created = _install_fake_sdk(monkeypatch, firmware=firmware)
    with pytest.raises(TransportError) as error:
        asyncio.run(PiperArmTransport(_BASE_CONFIG).open(_context()))
    assert error.value.failure_code == code
    assert created[0].disconnect_calls == 1 and not created[0].enable_calls


def test_open_survives_the_first_firmware_reply_being_dropped(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A discarded first answer must not read as an unverifiable arm.

    pyAgxArm's readiness predicate throws away a reply that arrives while its
    own fps estimate is still zero and then waits out the window instead of
    asking again, so the very first request after connect is routinely lost.
    """
    created = _install_fake_sdk(monkeypatch, drop_firmware=1)
    status = asyncio.run(PiperArmTransport(_BASE_CONFIG).open(_context()))
    assert status.connected and status.metadata["firmware_version"] == "S-V1.9-0"
    assert created[0].events.count("firmware") == 2


def test_firmware_version_spends_its_budget_then_reports_unanswered() -> None:
    """Re-asking is bounded by the configured budget and never spins."""

    class _Mute:
        def __init__(self) -> None:
            self.calls = 0

        def get_firmware(self, *, timeout: float, min_interval: float) -> Any:
            self.calls += 1
            time.sleep(timeout)  # the vendor waits out the window it was given
            return None

    robot = _Mute()
    assert firmware_version(robot, 0.0) == "" and robot.calls == 0
    started = time.monotonic()
    assert firmware_version(robot, 0.6) == ""
    assert robot.calls >= 2 and time.monotonic() - started < 1.5


def test_warmup_timeout_disconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_sdk(monkeypatch, silent=True)
    transport = PiperArmTransport({**_BASE_CONFIG, "warmup_timeout_s": .1})
    with pytest.raises(TransportError) as error:
        asyncio.run(transport.open(_context()))
    assert error.value.failure_code == "piper_warmup_timeout"
    assert created[0].disconnect_calls == 1
    with pytest.raises(TransportError, match="not open"):
        asyncio.run(transport.read("ee.x"))


@pytest.mark.parametrize("config", [{"sdk_interface": "v2"}, {"mode_switch": "direct"},
    {"default_effort_n_m": .5}, {"crash_protection_levels": [1] * 6}, {"robot": "nero"},
    {"pose_move_mode": "circular"}, {"stale_after_s": float("nan")}, {"warmup_timeout_s": 0}])
def test_invalid_or_retired_config_fails_closed(config: dict) -> None:
    with pytest.raises(TransportError) as error:
        PiperArmTransport(config)
    assert error.value.failure_code == "piper_invalid_config"


def test_governed_enable_reset_enable_can_order_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    asyncio.run(transport.halt())
    arm.events.clear()
    result = _write(transport, "arm.enable", 1)
    assert result.ok and result.side_effect_state == SIDE_EFFECT_COMMITTED
    assert arm.events == ["reset", "enable", "speed"]
    assert arm.ctrl_mode == 1 and arm.reset_calls == 1 and arm.enable_calls == 1
    assert asyncio.run(transport.read("arm.enabled")).value is True
    assert _write(transport, "arm.enable", 0).ok
    assert arm.disable_calls == 1
    assert asyncio.run(transport.read("arm.enabled")).value is False
    assert _write(transport, "arm.enable", 5).failure_code == "invalid_value"


def test_idle_enable_never_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    assert _write(transport, "arm.enable", 1).ok
    assert arm.reset_calls == 0 and arm.enable_calls == 1


@pytest.mark.parametrize("fault,code", [("refuse_enable", "piper_enable_refused"),
    ("refuse_reset", "piper_resume_refused"), ("ignore_mode", "piper_mode_switch_refused")])
def test_enable_confirmation_failure_is_committed(monkeypatch: pytest.MonkeyPatch, fault: str, code: str) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    if fault == "refuse_reset":
        asyncio.run(transport.halt())
    setattr(arm, fault, True)
    result = _write(transport, "arm.enable", 1)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_COMMITTED
    assert result.failure_code == code


@pytest.mark.parametrize("flags", [None, [], [True], [True] * 5, ["true"] * 6])
def test_incomplete_enable_confirmation_cannot_succeed(monkeypatch: pytest.MonkeyPatch, flags: Any) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.refuse_enable = True
    arm.flags = flags
    result = _write(transport, "arm.enable", 1)
    assert result.failure_code == "piper_enable_refused"
    assert not arm.disable_calls


def test_can_mode_retry_survives_dropped_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch, {"mode_switch_timeout_s": .4})
    arm.drop_mode_frames = 1
    assert _write(transport, "arm.enable", 1).ok
    assert len(arm.speed_calls) == 2


def test_arm_status_names_the_joints_the_fault_word_implicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A joint communication fault must name the joints, not just a number.

    0x16 is the firmware's own answer to *which* motors went silent; decoding it
    here is what turns ``arm_status`` 0x05 into an instruction (check those motor
    cables) instead of a hex code. The high byte is the angle-over-limit mask.
    """
    transport, arm, _ = _open_transport(monkeypatch)
    arm.arm_status, arm.err_code = 0x05, 0x16
    status = asyncio.run(transport.read("arm.status")).value
    assert status["error_code"] == 0x16
    assert status["joint_communication_errors"] == [2, 3, 5]
    assert status["joint_limit_errors"] == []
    arm.err_code = 0x0100
    assert asyncio.run(transport.read("arm.status")).value["joint_limit_errors"] == [1]


def test_an_enable_that_drops_is_retried_within_the_same_budget(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A controller that faults after entering CAN is re-enabled, not believed.

    The motor flags keep reading enabled across a fault, so the first enable
    alone reports success; the arm then drops to standby and sags. Observing the
    settle window catches it, and one retry inside the same budget recovers.
    """
    transport, arm, _ = _open_transport(monkeypatch,
                                        {"enable_timeout_s": .6, "enable_settle_s": .05})
    arm.fault_on_can = True
    result = _write(transport, "arm.enable", 1)
    assert result.ok and result.side_effect_state == SIDE_EFFECT_COMMITTED
    assert arm.enable_calls == 2
    assert arm.ctrl_mode == 1


def test_an_enable_that_keeps_dropping_is_refused_with_the_fault_named(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """When every attempt faults, the caller is refused with the fault and joints.

    Succeeding here would hand back an arm in standby with its drives released --
    the exact state that lets a later move collapse and be reported as stale.
    """
    transport, arm, _ = _open_transport(monkeypatch,
                                        {"enable_timeout_s": .3, "enable_settle_s": .05})
    arm.fault_reads = 10 ** 9
    result = _write(transport, "arm.enable", 1)
    assert not result.ok and result.failure_code == "piper_enable_unstable"
    assert result.side_effect_state == SIDE_EFFECT_COMMITTED
    assert "joint communication error" in result.error
    assert "joints 2, 3, 5" in result.error
    assert arm.enable_calls > 1


def test_a_target_the_firmware_refuses_is_refused_not_committed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The arm's own answer outranks delivery: a planning refusal moved nothing.

    Observed on a real cell: the arm stood a couple of degrees outside two joint
    limits, every target was refused, and each write still reported committed --
    so a caller built on a posture the arm had never taken.
    """
    transport, arm, _ = _open_transport(monkeypatch)
    arm.arm_status, arm.err_code = 0x04, 0x0600  # joints 2 and 3 past their limit
    result = _write(transport, "joint.j1.position", 12.5)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_NONE
    assert result.failure_code == "piper_target_refused"
    assert "joints 2, 3 out of limit" in (result.error or "")


def test_a_refused_target_is_never_held_as_a_command(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused posture is not where the arm is, so it may not fill a later hold."""
    transport, arm, _ = _open_transport(monkeypatch)
    assert _write(transport, "joint.j1.position", 12.5).ok
    arm.arm_status = 0x04
    assert not _write(transport, "joint.j1.position", 15).ok
    arm.arm_status = 0
    arm.joints = joints_to_sdk([11., 21., -31., 41., 51., -161.])
    assert _write(transport, "joint.j3.position", -35).ok
    assert arm.joint_calls[-1] == pytest.approx(
        joints_to_sdk([11., 21., -35., 41., 51., -161.])), (
        "the unnamed axes must follow feedback, never the refused vector")


def test_a_fault_after_the_target_is_unknown_not_committed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A fault is not a refusal: the arm may have moved before it stopped."""
    transport, arm, _ = _open_transport(monkeypatch)
    arm.arm_status = 0x07
    result = _write(transport, "ee.z", 400)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert result.failure_code == "piper_arm_fault"
    assert "collision" in (result.error or "")


def test_joint_units_and_hold_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    result = _write(transport, "joint.j1.position", 12.5)
    assert result.ok and result.readback.value == pytest.approx(10)
    assert arm.joint_calls[0] == pytest.approx(joints_to_sdk([12.5, 20, -30, 40, 50, -160]))
    assert _write(transport, "joint.j3.position", -35).ok
    assert arm.joint_calls[-1] == pytest.approx(joints_to_sdk([12.5, 20, -35, 40, 50, -160]))


def test_a_disowned_hold_is_never_replayed() -> None:
    """A hold is a commitment, and ending ownership voids it.

    Only a vector this driver sent may fill an unnamed axis. Once an enable,
    stop or disable has ended that ownership the arm may be anywhere, so the
    next command holds from feedback rather than from the old target.
    """
    motion = PiperMotion(speed_pct=10, pose_path="linear")
    arm = _FakeArm({})
    sent = [10., 20., -30., 40., 50., -160.]
    assert motion.send(arm, SPACE_JOINT, sent, [(0, 12.5)]) == pytest.approx(
        [12.5, *sent[1:]])
    later = [3., 2.5, -.4, 44., 2., 120.]
    motion.disown("an emergency stop")
    assert motion.send(arm, SPACE_JOINT, later, [(5, 100.)]) == pytest.approx([*later[:5], 100.])


def test_enable_never_turns_a_transitional_frame_into_a_held_axis(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Energising republishes postures the arm was never in; none may be held.

    The joint frame read while the arm is being energized is a measurement of a
    transition, not a command. Holding it filled an unnamed j2 with 0.0 while
    the arm had settled at 2.2 degrees, so a one-axis step moved an axis nobody
    named and still verified as reached.
    """
    transport, arm, _ = _open_transport(monkeypatch)
    arm.joints = joints_to_sdk([0., 0., 0., 0., 0., 0.])
    assert _write(transport, "arm.enable", 1).ok
    settled = [.7, 2.2, -.5, 44.6, 2.6, 120.6]
    arm.joints = joints_to_sdk(settled)
    assert _write(transport, "joint.j1.position", 2.7).ok
    assert arm.joint_calls[-1] == pytest.approx(joints_to_sdk([2.7, *settled[1:]])), (
        "the unnamed axes must follow feedback, not the enable-time frame")


def test_a_held_axis_outside_its_envelope_refuses_the_whole_vector(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The vendor clamps silently, so a held axis outside the envelope must be refused.

    The SDK rewrites an out-of-limit joint to the limit before it reaches the
    arm, so holding j2 where it stands would travel an axis the caller never
    named while the write reported success. Nothing may be sent instead.
    """
    transport, arm, _ = _open_transport(monkeypatch)
    arm.joints = joints_to_sdk([10., -2.2, -30., 40., 50., -160.])
    before = list(arm.joint_calls)
    result = _write(transport, "joint.j1.position", 12.5)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_NONE
    assert result.failure_code == "piper_hold_outside_envelope"
    assert arm.joint_calls == before
    assert "joint.j2.position" in (result.error or ""), "the refusal must name the axis"


def test_a_held_axis_inside_the_declared_tolerance_is_clamped_by_the_driver(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Within measurement precision the driver clamps itself, so sent equals executed."""
    transport, arm, _ = _open_transport(monkeypatch)
    arm.joints = joints_to_sdk([10., -0.2, -30., 40., 50., -160.])
    assert _write(transport, "joint.j1.position", 12.5).ok
    assert arm.joint_calls[-1] == pytest.approx(
        joints_to_sdk([12.5, 0., -30., 40., 50., -160.])), (
        "a hold a hair outside the limit is clamped to it, never handed to the vendor")


@pytest.mark.parametrize("channel,value", [("joint.j4.position", 90), ("joint.j5.position", -90),
    ("joint.j6.position", 171), ("ee.z", 5000), ("ee.ry", 91), ("arm.speed_pct", 150),
    ("gripper.effort", -1), ("joint.j1.position", float("nan")), ("ee.x", True)])
def test_invalid_targets_send_nothing(monkeypatch: pytest.MonkeyPatch, channel: str, value: Any) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    before = list(arm.events)
    result = _write(transport, channel, value)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_NONE
    assert arm.events == before


@pytest.mark.parametrize("path", ["linear", "point"])
def test_pose_units_path_and_holds(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    transport, arm, _ = _open_transport(monkeypatch, {"pose_move_mode": path})
    assert _write(transport, "ee.z", 400).ok
    assert arm.pose_calls == [(path, pytest.approx(pose_to_sdk([200, 0, 400, 179, 0, 90])))]
    assert not arm.joint_calls
    assert _write(transport, "ee.x", 250).ok
    assert arm.pose_calls[-1][1] == pytest.approx(pose_to_sdk([250, 0, 400, 179, 0, 90]))


def test_switching_spaces_invalidates_other_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    _write(transport, "joint.j1.position", 15)
    _write(transport, "ee.z", 400)
    arm.joints = joints_to_sdk([11, 21, -31, 41, 51, -161])
    assert _write(transport, "joint.j3.position", -35).ok
    assert arm.joint_calls[-1] == pytest.approx(joints_to_sdk([11, 21, -35, 41, 51, -161]))


@pytest.mark.parametrize("path", ["joint", "pose"])
def test_uncertain_send_drops_both_hold_caches(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    _write(transport, "joint.j1.position", 15)
    setattr(arm, f"fail_{path}", True)
    result = _write(transport, "joint.j1.position" if path == "joint" else "ee.z", 20 if path == "joint" else 400)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert result.failure_code == "piper_send_failed"
    setattr(arm, f"fail_{path}", False)
    _write(transport, "joint.j3.position", -35)
    assert arm.joint_calls[-1] == pytest.approx(joints_to_sdk([10, 20, -35, 40, 50, -160]))


def test_missing_pose_feedback_refuses_without_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.pose = None
    before = list(arm.events)
    result = _write(transport, "ee.z", 400)
    assert result.failure_code == "piper_no_feedback" and result.side_effect_state == SIDE_EFFECT_NONE
    assert arm.events == before


def test_settings_are_staged_and_applied_before_batch_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    before = list(arm.events)
    assert _write(transport, "arm.speed_pct", 5).ok
    assert _write(transport, "arm.pose_path", "point").ok
    assert arm.events == before
    assert asyncio.run(transport.read("arm.speed_pct")).value == 5
    assert _write(transport, "arm.pose_path", "bad").failure_code == "invalid_value"
    batch = asyncio.run(transport.write_batch((("ee.z", 350), ("arm.speed_pct", 10))))
    assert batch.ok and arm.speed_calls[-1] == 10
    assert arm.pose_calls[-1][0] == "point"


def test_mixed_or_unknown_batch_is_wholly_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    for commands, code in [((("arm.enable", 1), ("ee.z", 400), ("joint.j1.position", 15)), "mixed_motion_space"),
                           ((("arm.enable", 1), ("unknown", 3)), "not_writable")]:
        before = list(arm.events)
        batch = asyncio.run(transport.write_batch(commands))
        assert not batch.ok and batch.side_effect_state == SIDE_EFFECT_NONE
        assert all(o.failure_code == code for o in batch.outcomes)
        assert arm.events == before


def test_batch_order_and_atomic_motion_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    batch = asyncio.run(transport.write_batch((("gripper.position", 30), ("joint.j1.position", 15),
        ("arm.enable", 1), ("arm.speed_pct", 5), ("joint.j3.position", -35))))
    assert batch.ok and len(batch.outcomes) == 5
    assert arm.joint_calls == [pytest.approx(joints_to_sdk([15, 20, -35, 40, 50, -160]))]
    assert arm.events[-4:] == ["speed", "mode:j", "joint", "gripper"]
    assert arm.gripper.calls == [(pytest.approx(.03), 1)]


def test_batch_failure_aborts_remaining_gripper(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.fail_joint = True
    batch = asyncio.run(transport.write_batch((("joint.j1.position", 15), ("gripper.position", 30))))
    assert not batch.ok and batch.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert batch.outcomes[1].failure_code == "piper_batch_aborted"
    assert not arm.gripper.calls
    arm.fail_joint = False
    batch = asyncio.run(transport.write_batch((("joint.j1.position", 15), ("joint.j3.position", 999))))
    assert batch.side_effect_state == SIDE_EFFECT_NONE
    assert all(o.failure_code == "value_out_of_range" for o in batch.outcomes)


def test_halt_and_close_never_disable_or_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    assert "resume" in asyncio.run(transport.halt()).detail
    assert arm.estop_calls == 1 and not arm.disable_calls and not arm.reset_calls
    assert not asyncio.run(transport.close()).connected
    assert not asyncio.run(transport.close()).connected
    assert arm.disconnect_calls == 1
    assert not asyncio.run(transport.halt()).connected
    assert _write(transport, "ee.x", 200).failure_code == "transport_not_open"


def test_cached_reads_units_status_telemetry_and_unknown_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    batch = asyncio.run(transport.read_batch(("ee.z", "joint.j6.position", "gripper.effort", "arm.telemetry")))
    assert [r.value for r in batch.readings[:3]] == pytest.approx([300, -160, .5])
    assert [r.unit for r in batch.readings[:3]] == ["mm", "deg", "N"]
    assert all(r.quality == "ok" for r in batch.readings)
    telemetry = batch.readings[-1].value
    assert telemetry["receive_qualified"] and telemetry["effort_source"] == "sdk_current_estimate"
    assert telemetry["motors"][0]["estimated_effort_nm"] == .2
    assert asyncio.run(transport.read("arm.ctrl_mode")).value == "standby"
    arm.ctrl_mode = 2
    assert asyncio.run(transport.read("arm.ctrl_mode")).value == "teach"
    status = asyncio.run(transport.read("arm.status"))
    assert status.value == {"ctrl_mode": 2, "arm_status": 0, "mode_feedback": 1, "motion_status": 1,
                            "error_code": 0, "joint_limit_errors": [],
                            "joint_communication_errors": []}
    assert asyncio.run(transport.read("arm.motion")).value is False
    arm.motion_status = 0
    assert asyncio.run(transport.read("arm.motion")).value is True
    with pytest.raises(TransportError) as error:
        asyncio.run(transport.read_batch(("ee.x", "unknown")))
    assert error.value.failure_code == "unknown_channel"
    assert _write(transport, "arm.enabled", True).failure_code == "not_writable"


@pytest.mark.parametrize("channel,value", [("ee.x", 210), ("joint.j1.position", 15),
                                          ("gripper.position", 30), ("arm.enable", 1)])
def test_writes_refuse_missing_receipts_before_any_send(monkeypatch: pytest.MonkeyPatch, channel: str, value: int) -> None:
    transport, arm, _ = _open_transport(monkeypatch, receive=False)
    before = list(arm.events)
    result = _write(transport, channel, value)
    assert result.failure_code == "piper_unqualified_feedback"
    assert result.side_effect_state == SIDE_EFFECT_NONE and arm.events == before


def test_getters_cannot_refresh_frozen_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch, {"stale_after_s": .02})
    first = asyncio.run(transport.read("joint.j1.position"))
    assert first.quality == "ok"
    time.sleep(.03)
    assert asyncio.run(transport.read("joint.j1.position")).quality == "stale"
    arm.receive(tuple(f"0x{i:03x}" for i in range(0x2a2, 0x2a5)))
    assert asyncio.run(transport.read("ee.x")).quality == "ok"
    assert asyncio.run(transport.read("joint.j1.position")).quality == "stale"
    arm.receive()
    assert asyncio.run(transport.read("joint.j1.position")).quality == "ok"


def test_invalid_frames_degrade_read_and_block_write(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.receive(("0x2a5",), dlc=2)
    assert asyncio.run(transport.read("joint.j1.position")).quality == "stale"
    assert _write(transport, "ee.x", 210).side_effect_state == SIDE_EFFECT_NONE
    arm.joints = None
    assert asyncio.run(transport.read_batch(("joint.j1.position",))).readings[0].quality == "stale"
    with pytest.raises(TransportError) as error:
        asyncio.run(transport.read("joint.j1.position"))
    assert error.value.failure_code == "piper_no_feedback"


@pytest.mark.parametrize("channel,value", [("ee.x", 210), ("gripper.position", 30), ("arm.enable", 1)])
def test_receive_loss_during_mode_selection_prevents_target(monkeypatch: pytest.MonkeyPatch, channel: str, value: int) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    original = arm.set_speed_percent
    def lose(speed: int) -> None:
        original(speed)
        arm.receive(dlc=0)
    monkeypatch.setattr(arm, "set_speed_percent", lose)
    result = _write(transport, channel, value)
    assert not result.ok and result.side_effect_state in (SIDE_EFFECT_COMMITTED, SIDE_EFFECT_UNKNOWN)
    assert not arm.pose_calls and not arm.gripper.calls


def test_gripper_stroke_force_pair_and_mode_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    assert _write(transport, "gripper.effort", 2).ok
    assert _write(transport, "gripper.position", 40).ok
    assert arm.gripper.calls == [(pytest.approx(.02), 2), (pytest.approx(.04), 2)]
    assert arm.gripper.modes == [1, 1]
    assert len(arm.speed_calls) == 1


@pytest.mark.parametrize("failure", ["missing", "send"])
def test_gripper_failed_pair_does_not_advance_cache(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.gripper.down = failure == "missing"
    arm.gripper.fail = failure == "send"
    result = _write(transport, "gripper.effort", 2)
    assert not result.ok
    assert result.side_effect_state == (SIDE_EFFECT_NONE if failure == "missing" else SIDE_EFFECT_UNKNOWN)
    arm.gripper.down = arm.gripper.fail = False
    assert _write(transport, "gripper.position", 40).ok
    assert arm.gripper.calls[-1] == (pytest.approx(.04), 1)


def test_a_joint_move_asserts_its_move_mode_after_the_speed_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The targets ride a mode the firmware acknowledged, not one it may adopt.

    The speed frame parks the firmware's move mode at UNKNOWN and restores the
    previous mode only in the SDK's own cache, so the first target after a speed
    change would be evaluated in a mode the firmware cannot execute -- and
    dropped without an error. The mode frame carries the mode those targets
    belong to and is confirmed from feedback before any target goes out.
    """
    transport, arm, _ = _open_transport(monkeypatch)
    arm.mode_feedback = 0xFF  # parked by the frame that also selected CAN control
    assert _write(transport, "joint.j1.position", 12.5).ok
    assert arm.mode_calls == ["j"] and arm.mode_feedback == 1
    assert arm.events.index("speed") < arm.events.index("mode:j") < arm.events.index("joint")


def test_a_speed_change_is_applied_before_the_move_mode_is_reasserted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    assert _write(transport, "joint.j1.position", 12.5).ok
    arm.events.clear()
    batch = asyncio.run(transport.write_batch((("arm.speed_pct", 20), ("joint.j2.position", 12))))
    assert batch.ok and arm.speed_calls[-1] == 20
    assert arm.events == ["speed", "mode:j", "joint"]


def test_a_firmware_that_never_latches_the_mode_sends_no_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode the firmware never adopts is a typed refusal, not a settle timeout.

    No target frame goes out, so the arm was never asked to move and the caller
    learns why instead of watching a deadline expire on an arm that could not
    have arrived.
    """
    transport, arm, _ = _open_transport(monkeypatch)
    arm.ignore_move_mode = True
    arm.mode_feedback = 0xFF
    result = _write(transport, "joint.j1.position", 12.5)
    assert not result.ok and result.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert result.failure_code == "piper_move_mode_refused"
    assert "unknown(0xFF)" in (result.error or "") and "no target was sent" in (result.error or "")
    assert not arm.joint_calls


@pytest.mark.parametrize("path,mode", [("linear", "l"), ("point", "p")])
def test_a_pose_move_asserts_the_mode_of_its_own_path(
    monkeypatch: pytest.MonkeyPatch, path: str, mode: str,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    assert _write(transport, "arm.pose_path", path).ok
    assert _write(transport, "ee.x", 210).ok
    assert arm.mode_calls == [mode]


def test_slow_vendor_send_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    original = arm.move_j
    def slow(joints: list[float]) -> None:
        time.sleep(.1)
        original(joints)
    monkeypatch.setattr(arm, "move_j", slow)
    async def run() -> None:
        task = asyncio.create_task(transport.write("joint.j1.position", 15))
        await asyncio.sleep(.04)
        assert not task.done()
        halt = await transport.halt()
        assert halt.halt_supported and arm.estop_calls == 1
        await task
    asyncio.run(run())
