"""Piper transport contract tests against a fake piper_sdk (no CAN hardware).

Mirrors test_realsense_transport.py: the vendor SDK is injected by patching
``importlib.import_module``, so every runtime path of the driver -- bring-up,
unit conversion, hold-position joint vectors, side-effect verdicts, the halt
mapping, and staleness -- is exercised without a physical arm.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.drivers.piper import PiperArmTransport
from leapflow.hardware.registry import _DEVICE_ID_ALLOWED
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    TransportError,
)

_EXAMPLE = (
    Path(__file__).parents[1]
    / "src"
    / "leapflow"
    / "hardware"
    / "examples"
    / "piper_arm.yaml"
)

_BASE_CONFIG: dict[str, Any] = {"can_name": "can0", "sdk_interface": "v2"}

_SEED_JOINTS_RAW = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000]
"""Feedback the fake arm starts reporting after ConnectPort: 10..60 deg."""


def _context() -> HardwareContext:
    payload = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    return HardwareContext.from_mapping(payload)


class _FakeArm:
    """Minimal piper_sdk interface stand-in that records every command frame."""

    def __init__(
        self,
        can_name: str,
        judge_flag: bool = False,
        can_auto_init: bool = True,
        dh_is_offset: int = 0,
        start_sdk_joint_limit: bool = False,
        start_sdk_gripper_limit: bool = False,
    ) -> None:
        self.can_name = can_name
        self.constructor_kwargs = {
            "judge_flag": judge_flag,
            "can_auto_init": can_auto_init,
            "dh_is_offset": dh_is_offset,
            "start_sdk_joint_limit": start_sdk_joint_limit,
            "start_sdk_gripper_limit": start_sdk_gripper_limit,
        }
        self.connected = False
        self.ok = True
        self.fps = 0.0
        self.joints: list[int] | None = None
        self.stroke_raw = 20_000
        self.effort_raw = 500
        self.ctrl_mode = 0x00
        self.arm_status = 0x00
        self.motion_status = 0x01
        self.enable_flags: list[bool] | None = None
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.joint_ctrl_calls: list[tuple[int, ...]] = []
        self.gripper_calls: list[tuple[int, int, int, int]] = []
        self.enable_calls: list[tuple[int, int]] = []
        self.disable_calls: list[tuple[int, int]] = []
        self.reset_calls = 0
        self.estop_calls: list[int] = []
        self.mode_calls: list[dict[str, int]] = []
        self.crash_calls: list[tuple[int, ...]] = []
        self.fail_joint_ctrl = False
        self.fail_gripper_ctrl = False
        self.joint_feedback_down = False
        self.gripper_feedback_down = False
        self.joint_stamp: float | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def ConnectPort(
        self, can_init: bool = False, piper_init: bool = True, start_thread: bool = True
    ) -> None:
        self.connect_calls += 1
        self.connected = True
        if not self.joint_feedback_down:
            self.joints = list(_SEED_JOINTS_RAW)
            self.fps = 250.0

    def DisconnectPort(self, thread_timeout: float = 0.1) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def get_connect_status(self) -> bool:
        return self.connected

    def isOk(self) -> bool:
        return self.ok

    def GetCanFps(self) -> float:
        return self.fps

    def GetPiperFirmwareVersion(self) -> str:
        return "V1.5-2"

    def GetCurrentInterfaceVersion(self) -> str:
        return "v2.0"

    # ── Feedback ──────────────────────────────────────────────────────────────

    def GetArmJointMsgs(self) -> SimpleNamespace:
        if self.joint_feedback_down or self.joints is None:
            return SimpleNamespace(joint_state=None)
        state = SimpleNamespace(**{f"joint_{i}": self.joints[i - 1] for i in range(1, 7)})
        return SimpleNamespace(joint_state=state, time_stamp=self.joint_stamp)

    def GetArmGripperMsgs(self) -> SimpleNamespace:
        if self.gripper_feedback_down:
            return SimpleNamespace(gripper_state=None)
        state = SimpleNamespace(grippers_angle=self.stroke_raw, grippers_effort=self.effort_raw)
        return SimpleNamespace(gripper_state=state)

    def GetArmStatus(self) -> SimpleNamespace:
        feedback = SimpleNamespace(
            ctrl_mode=self.ctrl_mode,
            arm_status=self.arm_status,
            motion_status=self.motion_status,
        )
        return SimpleNamespace(arm_status=feedback)

    def GetArmEnableStatus(self) -> list[bool] | None:
        return list(self.enable_flags) if self.enable_flags is not None else None

    # ── Commands ──────────────────────────────────────────────────────────────

    def JointCtrl(self, *joints: int) -> None:
        self.joint_ctrl_calls.append(tuple(int(j) for j in joints))
        if self.fail_joint_ctrl:
            raise RuntimeError("simulated CAN write failure")

    def GripperCtrl(
        self, gripper_angle: int = 0, gripper_effort: int = 0,
        gripper_code: int = 0, set_zero: int = 0,
    ) -> None:
        self.gripper_calls.append(
            (int(gripper_angle), int(gripper_effort), int(gripper_code), int(set_zero))
        )
        if self.fail_gripper_ctrl:
            raise RuntimeError("simulated CAN write failure")
        self.stroke_raw = int(gripper_angle)

    def EnableArm(self, motor_num: int = 7, enable_flag: int = 0x02) -> None:
        self.enable_calls.append((int(motor_num), int(enable_flag)))
        self.enable_flags = [True] * 7

    def DisableArm(self, motor_num: int = 7, enable_flag: int = 0x01) -> None:
        self.disable_calls.append((int(motor_num), int(enable_flag)))
        self.enable_flags = [False] * 7

    def EmergencyStop(self, emergency_stop: int = 0) -> None:
        self.estop_calls.append(int(emergency_stop))
        if emergency_stop == 0x01:
            self.arm_status = 0x01

    def ModeCtrl(
        self, ctrl_mode: int = 0x01, move_mode: int = 0x01,
        move_spd_rate_ctrl: int = 50, is_mit_mode: int = 0x00,
    ) -> None:
        self.mode_calls.append({
            "ctrl_mode": ctrl_mode,
            "move_mode": move_mode,
            "move_spd_rate_ctrl": move_spd_rate_ctrl,
        })
        self.ctrl_mode = ctrl_mode

    def CrashProtectionConfig(self, *levels: int) -> None:
        self.crash_calls.append(tuple(int(level) for level in levels))

    def ResetPiper(self) -> None:
        """Recorded only so tests can assert it is never reached."""
        self.reset_calls += 1


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, silent: bool = False) -> list[_FakeArm]:
    """Patch ``importlib.import_module`` so ``piper_sdk`` resolves to the fake.

    Returns the arm instances the driver constructs, in creation order. With
    ``silent`` the arm connects but never produces feedback, which drives the
    bring-up warm-up loop to its typed timeout.
    """
    created: list[_FakeArm] = []
    real_import = importlib.import_module

    class _Recording(_FakeArm):
        def __init__(self, can_name: str, **kwargs: Any) -> None:
            super().__init__(can_name, **kwargs)
            self.joint_feedback_down = silent
            created.append(self)

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "piper_sdk":
            return SimpleNamespace(C_PiperInterface_V2=_Recording)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return created


def _open_transport(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any] | None = None
) -> tuple[PiperArmTransport, _FakeArm, Any]:
    created = _install_fake_sdk(monkeypatch)
    transport = PiperArmTransport({**_BASE_CONFIG, **(overrides or {})})
    status = asyncio.run(transport.open(_context()))
    return transport, created[0], status


# ── Declaration ───────────────────────────────────────────────────────────────


def test_example_yaml_is_a_valid_declarative_context() -> None:
    context = _context()

    assert context.device_id == "robot_piper_arm"
    # Registry admission (V2) only admits [a-z0-9_]+ device ids; pin the
    # declaration to that contract so a dotted id can never ship again.
    assert all(char in _DEVICE_ID_ALLOWED for char in context.device_id)
    assert context.transport.kind == "piper"
    assert context.transport.config["can_name"] == "can0"
    assert context.halt_supported is True
    assert context.kinematics is not None
    assert context.kinematics.dof == 6
    assert len(context.kinematics.joint_limits) == 6
    assert context.safety is not None
    assert context.safety.emergency_decel_s > 0.0

    expected = {
        *(f"joint.j{i}.position" for i in range(1, 7)),
        "gripper.position",
        "gripper.effort",
        "arm.enable",
        "arm.enabled",
        "arm.status",
        "arm.motion",
    }
    assert {channel.channel_id for channel in context.channels} == expected

    j2 = context.channel("joint.j2.position")
    assert j2 is not None
    assert j2.envelope.min_value == 0.0
    assert j2.envelope.max_value == 180.0
    # Write verification defaults to an exact match; the explicit tolerance is
    # what makes verify-after-write actually passable on a servo readback.
    assert j2.envelope.tolerance == 0.5
    j3 = context.channel("joint.j3.position")
    assert j3 is not None
    assert j3.envelope.min_value == -170.0
    assert j3.envelope.max_value == 0.0

    enable = context.channel("arm.enable")
    assert enable is not None
    assert enable.envelope.allowed_values == (0, 1)
    assert context.channel("joint.j1.position").is_writable
    assert context.channel("arm.enabled").is_readable


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def test_missing_sdk_is_a_typed_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name == "piper_sdk"
        raise ImportError("not installed")

    monkeypatch.setattr(importlib, "import_module", missing_sdk)
    transport = PiperArmTransport(dict(_BASE_CONFIG))

    with pytest.raises(TransportError, match="piper_sdk") as error:
        asyncio.run(transport.open(_context()))

    assert error.value.failure_code == "piper_sdk_missing"


def test_open_follows_the_canonical_bring_up_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    _, arm, status = _open_transport(monkeypatch)

    assert status.connected is True
    assert status.halt_supported is True
    assert arm.can_name == "can0"
    assert arm.constructor_kwargs["judge_flag"] is False
    assert arm.constructor_kwargs["start_sdk_joint_limit"] is True
    assert arm.mode_calls == [{"ctrl_mode": 1, "move_mode": 1, "move_spd_rate_ctrl": 20}]
    assert arm.enable_calls == []  # enable_on_open stays off by default
    metadata = status.metadata
    assert metadata["firmware_version"] == "V1.5-2"
    assert metadata["can_fps"] == 250.0
    assert metadata["ctrl_mode"] == 1
    assert metadata["enabled_motors"] is None


def test_open_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_sdk(monkeypatch)
    transport = PiperArmTransport(dict(_BASE_CONFIG))
    first = asyncio.run(transport.open(_context()))
    second = asyncio.run(transport.open(_context()))

    assert len(created) == 1
    assert created[0].connect_calls == 1
    assert first.connected is True
    assert second.connected is True


def test_warmup_timeout_is_typed_and_leaves_the_transport_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_sdk(monkeypatch, silent=True)
    transport = PiperArmTransport({**_BASE_CONFIG, "warmup_timeout_s": 0.12})

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.open(_context()))

    assert error.value.failure_code == "piper_warmup_timeout"
    # The half-open CAN link was disconnected during bring-up cleanup.
    assert created[0].disconnect_calls == 1
    with pytest.raises(TransportError) as closed:
        asyncio.run(transport.read("joint.j1.position"))
    assert closed.value.failure_code == "transport_not_open"


def test_motors_are_never_energized_during_bring_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open must stay side-effect free: a read-only open may not energize.

    The stale ``enable_on_open`` key is passed deliberately: an unknown config
    key must be inert, never a way back to auto-enabling in the driver.
    """
    _, arm, status = _open_transport(monkeypatch, {"enable_on_open": True})

    assert arm.enable_calls == []
    assert arm.estop_calls == []
    assert arm.disable_calls == []
    assert status.metadata.get("enable_on_open") is None


def test_governed_enable_resumes_the_emergency_stop_then_energizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    asyncio.run(transport.halt())
    assert arm.estop_calls == [1]

    resumed = asyncio.run(transport.write("arm.enable", 1))

    assert resumed.ok is True
    assert resumed.side_effect_state == SIDE_EFFECT_COMMITTED
    # 0x02 resume is the only vendor path out of a latched halt; it rides the
    # same governed write as the energize, never the halt path.
    assert arm.estop_calls == [1, 2]
    assert arm.enable_calls == [(7, 2)]
    assert arm.disable_calls == []


def test_crash_protection_levels_are_configured_on_bring_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, status = _open_transport(
        monkeypatch, {"crash_protection_levels": [1, 2, 3, 1, 1, 1]}
    )

    assert arm.crash_calls == [(1, 2, 3, 1, 1, 1)]
    assert status.metadata["crash_protection_levels"] == [1, 2, 3, 1, 1, 1]


def test_invalid_sdk_interface_is_rejected_at_construction() -> None:
    with pytest.raises(TransportError) as error:
        PiperArmTransport({"sdk_interface": "v9"})

    assert error.value.failure_code == "piper_invalid_config"


def test_invalid_crash_protection_levels_are_rejected_at_construction() -> None:
    with pytest.raises(TransportError) as error:
        PiperArmTransport({"crash_protection_levels": [1, 2, 3]})

    assert error.value.failure_code == "piper_invalid_config"


# ── Joint writes ──────────────────────────────────────────────────────────────


def test_joint_write_converts_degrees_and_holds_the_uncommanded_joints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    outcome = asyncio.run(transport.write("joint.j1.position", 12.5))

    assert arm.joint_ctrl_calls == [(12_500, 20_000, 30_000, 40_000, 50_000, 60_000)]
    assert outcome.ok is True
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert outcome.readback is not None
    assert outcome.readback.value == pytest.approx(10.0)  # fake feedback is unchanged


def test_next_write_holds_previously_commanded_joints(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    asyncio.run(transport.write("joint.j1.position", 15.0))
    # j3's declared envelope is [-170, 0], so the command must be negative.
    asyncio.run(transport.write("joint.j3.position", -35.0))

    assert arm.joint_ctrl_calls[1] == (15_000, 20_000, -35_000, 40_000, 50_000, 60_000)


def test_envelope_violation_refuses_before_any_send(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    outcome = asyncio.run(transport.write("joint.j2.position", 999.0))

    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == "value_out_of_range"
    assert arm.joint_ctrl_calls == []


def test_send_failure_reports_an_unknown_side_effect_and_persists_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.fail_joint_ctrl = True

    outcome = asyncio.run(transport.write("joint.j1.position", 15.0))

    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert outcome.failure_code == "piper_send_failed"

    arm.fail_joint_ctrl = False
    asyncio.run(transport.write("joint.j3.position", -35.0))
    # The failed command never became the hold vector: j1 stays at feedback.
    assert arm.joint_ctrl_calls[-1] == (10_000, 20_000, -35_000, 40_000, 50_000, 60_000)


# ── Batch writes ──────────────────────────────────────────────────────────────


def test_write_batch_merges_joint_commands_into_one_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    batch = asyncio.run(
        transport.write_batch(
            (
                ("arm.enable", 1),
                ("joint.j1.position", 15.0),
                ("joint.j3.position", -35.0),
                ("gripper.position", 30.0),
            )
        )
    )

    assert arm.joint_ctrl_calls == [(15_000, 20_000, -35_000, 40_000, 50_000, 60_000)]
    assert arm.enable_calls == [(7, 2)]
    assert arm.gripper_calls == [(30_000, 500, 1, 0)]
    assert batch.ok is True
    assert batch.side_effect_state == SIDE_EFFECT_COMMITTED
    assert len(batch.outcomes) == 4
    assert all(outcome.ok for outcome in batch.outcomes)


def test_write_batch_refuses_the_whole_joint_vector_when_one_channel_is_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JointCtrl is atomic: one refused joint must not send the other five."""
    transport, arm, _ = _open_transport(monkeypatch)

    batch = asyncio.run(
        transport.write_batch((("joint.j1.position", 15.0), ("joint.j3.position", 999.0)))
    )

    assert batch.ok is False
    assert batch.side_effect_state == SIDE_EFFECT_NONE
    for outcome in batch.outcomes:
        assert outcome.ok is False
        assert outcome.side_effect_state == SIDE_EFFECT_NONE
        assert outcome.failure_code == "value_out_of_range"
    assert arm.joint_ctrl_calls == []


def test_write_batch_reports_the_worst_side_effect_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.fail_joint_ctrl = True

    batch = asyncio.run(
        transport.write_batch((("joint.j1.position", 15.0), ("gripper.position", 30.0)))
    )

    assert batch.ok is False
    assert batch.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert batch.outcomes[0].side_effect_state == SIDE_EFFECT_UNKNOWN
    assert batch.outcomes[1].side_effect_state == SIDE_EFFECT_COMMITTED


# ── Halt ──────────────────────────────────────────────────────────────────────


def test_halt_maps_to_the_vendor_emergency_stop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    status = asyncio.run(transport.halt())

    assert arm.estop_calls == [1]  # 0x01: engage
    assert arm.reset_calls == 0  # ResetPiper drops the arm
    assert arm.disable_calls == []  # DisableArm drops the arm
    assert status.halt_supported is True
    assert "resume" in status.detail


def test_halt_is_safe_on_a_closed_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch)
    transport = PiperArmTransport(dict(_BASE_CONFIG))

    status = asyncio.run(transport.halt())  # must not raise

    assert status.connected is False


# ── Reads ─────────────────────────────────────────────────────────────────────


def test_reads_convert_sdk_units(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    j2 = asyncio.run(transport.read("joint.j2.position"))
    stroke = asyncio.run(transport.read("gripper.position"))
    effort = asyncio.run(transport.read("gripper.effort"))

    assert j2.value == pytest.approx(20.0)
    assert j2.unit == "deg"
    assert j2.quantity == "angular_position"
    assert j2.quality == "ok"
    assert stroke.value == pytest.approx(20.0)
    assert stroke.unit == "mm"
    assert effort.value == pytest.approx(0.5)
    assert effort.unit == "N_m"


def test_enable_write_round_trip_and_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    with pytest.raises(TransportError) as unavailable:
        asyncio.run(transport.read("arm.enabled"))
    assert unavailable.value.failure_code == "piper_no_feedback"

    enabled = asyncio.run(transport.write("arm.enable", 1))
    assert enabled.ok is True
    assert enabled.side_effect_state == SIDE_EFFECT_COMMITTED
    assert arm.enable_calls == [(7, 2)]
    assert arm.estop_calls == [2]  # enable carries the resume frame
    assert asyncio.run(transport.read("arm.enabled")).value is True

    disabled = asyncio.run(transport.write("arm.enable", 0))
    assert disabled.ok is True
    assert arm.disable_calls == [(7, 1)]
    assert asyncio.run(transport.read("arm.enabled")).value is False

    bogus = asyncio.run(transport.write("arm.enable", 5))
    assert bogus.ok is False
    assert bogus.side_effect_state == SIDE_EFFECT_NONE
    assert bogus.failure_code == "invalid_value"

    readonly = asyncio.run(transport.write("arm.enabled", True))
    assert readonly.ok is False
    assert readonly.side_effect_state == SIDE_EFFECT_NONE
    assert readonly.failure_code == "not_writable"


def test_status_and_motion_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    status = asyncio.run(transport.read("arm.status"))
    moving = asyncio.run(transport.read("arm.motion"))

    assert status.value == {"ctrl_mode": 1, "arm_status": 0, "motion_status": 1}
    assert moving.value is False  # fake reports motion_status 0x01 (not arrived)

    arm.motion_status = 0x00
    assert asyncio.run(transport.read("arm.motion")).value is True


def test_read_batch_shares_one_timebase_and_degrades_to_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    asyncio.run(transport.write("arm.enable", 1))

    batch = asyncio.run(
        transport.read_batch(
            ("joint.j1.position", "gripper.position", "arm.status", "arm.enabled")
        )
    )

    assert batch.device_id == "robot_piper_arm"
    values = {reading.channel_id: reading for reading in batch.readings}
    assert values["joint.j1.position"].value == pytest.approx(10.0)
    assert values["arm.enabled"].value is True
    assert len({reading.observed_at for reading in batch.readings}) == 1

    arm.joint_feedback_down = True
    stale = asyncio.run(transport.read_batch(("joint.j1.position",)))
    assert stale.readings[0].value == 0.0
    assert stale.readings[0].quality == "stale"


def test_freshness_tracks_feedback_advancement_not_poll_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.joint_stamp = 100.0
    assert asyncio.run(transport.read("joint.j1.position")).quality == "ok"

    # The receive thread stalls: the SDK keeps returning the same cached frame,
    # so parse success must not keep relabeling it fresh.
    transport._last_feedback_mono = time.monotonic() - 5.0
    assert asyncio.run(transport.read("joint.j1.position")).quality == "stale"

    # A new frame arrives: freshness recovers without reopening.
    arm.joint_stamp = 100.2
    assert asyncio.run(transport.read("joint.j1.position")).quality == "ok"


def test_single_read_raises_typed_error_when_feedback_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.joint_feedback_down = True

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.read("joint.j1.position"))

    assert error.value.failure_code == "piper_no_feedback"


def test_unknown_channels_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.read("gripper.opening"))
    assert error.value.failure_code == "unknown_channel"

    with pytest.raises(TransportError) as batch_error:
        asyncio.run(transport.read_batch(("joint.j1.position", "nope")))
    assert batch_error.value.failure_code == "unknown_channel"
    assert arm.joint_ctrl_calls == []


# ── Close ─────────────────────────────────────────────────────────────────────


def test_close_is_idempotent_and_never_powers_down(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    closed = asyncio.run(transport.close())
    again = asyncio.run(transport.close())

    assert arm.disconnect_calls == 1
    assert closed.connected is False
    assert again.connected is False
    assert arm.disable_calls == []  # close never releases torque

    outcome = asyncio.run(transport.write("joint.j1.position", 15.0))
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == "transport_not_open"

    halted = asyncio.run(transport.halt())
    assert halted.connected is False


# ── Gripper ───────────────────────────────────────────────────────────────────


def test_gripper_writes_pair_stroke_and_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, arm, _ = _open_transport(monkeypatch)

    effort = asyncio.run(transport.write("gripper.effort", 1.0))
    stroke = asyncio.run(transport.write("gripper.position", 40.0))

    # An effort-only command pairs with the stroke feedback; the next stroke
    # write retains the commanded effort.
    assert arm.gripper_calls == [(20_000, 1_000, 1, 0), (40_000, 1_000, 1, 0)]
    assert effort.ok is True
    assert stroke.ok is True
    assert stroke.readback is not None
    assert stroke.readback.value == pytest.approx(40.0)


def test_refused_effort_write_does_not_leak_into_the_next_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.gripper_feedback_down = True

    refused = asyncio.run(transport.write("gripper.effort", 2.0))

    assert refused.ok is False
    assert refused.side_effect_state == SIDE_EFFECT_NONE
    assert refused.failure_code == "piper_no_feedback"

    arm.gripper_feedback_down = False
    asyncio.run(transport.write("gripper.position", 40.0))
    # The refused effort never reached the pair: the next stroke write pairs
    # with the previously accepted default (500), not the refused 2000.
    assert arm.gripper_calls == [(40_000, 500, 1, 0)]


def test_failed_gripper_send_does_not_advance_the_commanded_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, arm, _ = _open_transport(monkeypatch)
    arm.fail_gripper_ctrl = True

    failed = asyncio.run(transport.write("gripper.effort", 2.0))

    assert failed.ok is False
    assert failed.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert failed.failure_code == "piper_send_failed"

    arm.fail_gripper_ctrl = False
    asyncio.run(transport.write("gripper.position", 40.0))
    # Same rule after an uncertain send: the commanded pair only advances on
    # success, so the retry-friendly stroke write keeps the old effort. The
    # first entry is the failed attempt itself, which the fake records before
    # raising; the pin is the second call's effort, not the first's.
    assert arm.gripper_calls == [(20_000, 2_000, 1, 0), (40_000, 500, 1, 0)]
