# Copyright (c) Alibaba, Inc. and its affiliates.
"""The arm-status word the firmware answers a commanded target with.

The status word is the controller's own verdict on the posture it was given, and
two layers act on it: the transport, which may only report a write as committed
once that verdict is known, and the motion skill, which decides during a long
move whether to keep waiting or to fail. Both must read the same code the same
way -- a code one layer calls a refusal and the other a slow arrival ends a run
with the drives released -- so the vocabulary lives here, next to neither.

The codes fall into three groups, and the distinction is what makes a write
honest rather than merely delivered:

- ``NORMAL``: the firmware is carrying the target out.
- a *refusal*: the firmware planned the target and declined it. Nothing moved,
  so the arm stands exactly where the command found it.
- a *fault*: the arm stopped executing for a reason that has nothing to do with
  the target. It may have moved before it stopped.
"""

from __future__ import annotations

ARM_STATUS_NORMAL = 0x00
ARM_STATUS_EMERGENCY_STOP = 0x01

ARM_STATUS_NAMES: dict[int, str] = {
    ARM_STATUS_NORMAL: "normal",
    ARM_STATUS_EMERGENCY_STOP: "an engaged emergency stop",
    0x02: "no inverse-kinematics solution",
    0x03: "a singularity",
    0x04: "a target past a joint limit",
    0x05: "a joint communication error",
    0x06: "a joint brake that did not release",
    0x07: "a collision",
    0x08: "overspeed while dragging",
    0x09: "a joint status error",
    0x0A: "a controller fault",
    0x0E: "main-controller over-temperature",
    0x0F: "release-resistor over-temperature",
}

TARGET_REFUSALS: dict[int, str] = {
    0x02: "no inverse-kinematics solution for the target",
    0x03: "the target lies on a singularity",
    0x04: "the target exceeds a joint limit",
}
"""Codes that mean the firmware planned the target and refused it: nothing moved."""

ARM_FAULTS: dict[int, str] = {
    0x01: "the emergency stop is engaged",
    0x05: "a joint lost communication with the controller",
    0x06: "a joint brake did not release",
    0x07: "a collision was detected",
    0x08: "overspeed was detected while dragging",
    0x09: "a joint reported a status error",
    0x0A: "the controller reported a fault",
    0x0E: "the controller is over temperature",
    0x0F: "the release resistor is over temperature",
}
"""Codes that mean the arm stopped carrying commands out: it may have moved."""

# Byte 6 of the fault word flags joints pushed past their angle limit; byte 7
# flags joints whose motor stopped answering on the controller's internal bus.
LIMIT_FAULT_SHIFT = 8
COMM_FAULT_SHIFT = 0
_JOINT_NUMBERS = tuple(range(1, 7))


def status_name(code: int) -> str:
    """The declared name of one raw status code, naming an unassigned code explicitly."""
    return ARM_STATUS_NAMES.get(code, f"fault 0x{code:02X}")


def joint_faults(code: int | None, shift: int) -> list[int] | None:
    """The joint numbers set in one half of the firmware's fault word."""
    if code is None:
        return None
    mask = (code >> shift) & 0xFF
    return [joint for joint in _JOINT_NUMBERS if mask >> (joint - 1) & 1]
