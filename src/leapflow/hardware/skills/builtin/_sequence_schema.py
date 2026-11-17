# Copyright (c) Alibaba, Inc. and its affiliates.
"""The vocabulary of a ``motion.sequence`` program: ops, guards, the step schema, the card.

Pure declarations, so the planner and the runner in ``sequence`` read one
definition of what a step may say, and the model reads the same one in the card.
"""

from __future__ import annotations

from typing import Any, Mapping

OP_MOVE = "move"
OP_MOVE_LINEAR = "move_linear"
OP_MOVE_JOINTS = "move_joints"
OP_GRIPPER = "gripper"
OP_WAIT = "wait"
OPS = (OP_MOVE, OP_MOVE_LINEAR, OP_MOVE_JOINTS, OP_GRIPPER, OP_WAIT)

EXPECT_HELD = "held"
EXPECT_RELEASED = "released"
EXPECT_REACHED = "reached"
EXPECT_NO_CONTACT = "no_contact"
ON_FAIL_STOP = "stop"
ON_FAIL_HOLD = "hold"

POSE_OPS = frozenset({OP_MOVE, OP_MOVE_LINEAR})
COMMON_FIELDS = frozenset({"op", "expect", "on_fail"})
FIELDS: Mapping[str, frozenset[str]] = {
    OP_MOVE: frozenset({"pose", "relative", "ref", "offset", "speed_pct", "path"}),
    OP_MOVE_LINEAR: frozenset({"pose", "relative", "ref", "offset", "speed_pct"}),
    OP_MOVE_JOINTS: frozenset({"joints", "speed_pct"}),
    OP_GRIPPER: frozenset({"width_mm", "grip", "ref", "effort"}),
    OP_WAIT: frozenset({"seconds"}),
}
EXPECTS: Mapping[str, frozenset[str]] = {
    OP_MOVE: frozenset({EXPECT_HELD, EXPECT_REACHED, EXPECT_NO_CONTACT}),
    OP_MOVE_LINEAR: frozenset({EXPECT_HELD, EXPECT_REACHED, EXPECT_NO_CONTACT}),
    OP_MOVE_JOINTS: frozenset({EXPECT_HELD, EXPECT_REACHED, EXPECT_NO_CONTACT}),
    OP_GRIPPER: frozenset({EXPECT_HELD, EXPECT_RELEASED, EXPECT_REACHED}),
    OP_WAIT: frozenset({EXPECT_HELD, EXPECT_REACHED}),
}
OBJECT_POINTS = ("", "top", "bottom")
GRASP_POINTS = ("pregrasp", "grasp", "retreat")

SPEED_ARG = {
    "type": "number",
    "minimum": 1,
    "maximum": 100,
    "description": "Speed, percent of the arm maximum; capped by the run's limit.",
}
_ANGLE = {"type": "number", "minimum": -180, "maximum": 180}
_OFFSET = {"type": "number", "minimum": -1000, "maximum": 1000, "default": 0.0}
STEP_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": list(OPS)},
        "pose": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "rx": _ANGLE,
                "ry": _ANGLE,
                "rz": _ANGLE,
                "frame": {"type": "string", "enum": ["base", "table"], "default": "base"},
            },
            "required": ["x", "y", "z"],
        },
        "relative": {
            "type": "object",
            "properties": {
                "dx": _OFFSET,
                "dy": _OFFSET,
                "dz": _OFFSET,
                "frame": {"type": "string", "enum": ["base", "tool"], "default": "base"},
            },
        },
        "ref": {"type": "string", "minLength": 3, "maxLength": 40},
        # A path name is checked against the arm's declared arm.pose_path at plan;
        # the schema only has to let the key through.
        "path": {"type": "string", "minLength": 1},
        "offset": {
            "type": "object",
            "properties": {"dx": _OFFSET, "dy": _OFFSET, "dz": _OFFSET},
        },
        "joints": {"type": "object", "additionalProperties": True},
        "width_mm": {"type": "number", "minimum": 0},
        "grip": {"type": "string", "enum": ["open", "close"]},
        "effort": {"type": "number", "minimum": 0},
        "seconds": {"type": "number", "minimum": 0},
        "speed_pct": SPEED_ARG,
        "expect": {
            "type": "string",
            "enum": [EXPECT_HELD, EXPECT_RELEASED, EXPECT_REACHED, EXPECT_NO_CONTACT],
        },
        "on_fail": {"type": "string", "enum": [ON_FAIL_STOP, ON_FAIL_HOLD], "default": ON_FAIL_HOLD},
    },
    "required": ["op"],
}

CARD = (
    "Run a motion program on the arm and gripper, approved once as a whole. Each step "
    "has an op: move (a free transit: point-to-point when the arm declares it, else the "
    "arm's default path) or move_linear (straight line) to a TCP "
    "target; move_joints {joint: deg}; gripper (width_mm, or grip open|close; "
    "ref grasp:gN opens to that grasp's open_mm and records the object as held when "
    "the jaws close on it); wait (seconds). A move target is one of: pose {x, y, z, "
    "rx?, ry?, rz?} in mm/deg (frame base, or table: z above the table surface); "
    "relative {dx, dy, dz, frame base|tool} from the previous target; or ref "
    "object:oN[.top|.bottom] / grasp:gN.pregrasp|grasp|retreat plus an optional "
    "offset {dx, dy, dz} in the base frame. Omitted angles keep the previous target's "
    "orientation. A move may name path: linear|point to ride that path when the arm "
    "declares it (move_linear always rides linear); approach and retreat close to an "
    "object with move_linear, and transit between them with move (a straight line out "
    "of a wrist singularity is infeasible: move off it first). A step may declare "
    "expect: held (the jaws hold something), "
    "released (the jaws opened freely), reached (the TCP is at the last target), or "
    "no_contact (the move arrived and the jaws were not pushed); a failed guard leaves "
    "the arm holding in place (on_fail hold, the default) or also halts and latches it "
    "(stop). A released guard after a hold needs an opening the jaws can confirm, "
    "wider than the held width by twice the gripper tolerance; plan refuses one that "
    "cannot pass. Plan uses the arm's declared FK to sample every "
    "joint path and refuses any tool target outside the workspace, and solves every move "
    "with IK inside the joint limits: an unreachable one is refused with the farthest "
    "reachable target, so rewrite the program to it; execute repeats the workspace check "
    "from fresh joint state and refuses when the tool or a referenced object moved."
)

__all__ = [
    "CARD",
    "COMMON_FIELDS",
    "EXPECTS",
    "EXPECT_HELD",
    "EXPECT_NO_CONTACT",
    "EXPECT_REACHED",
    "EXPECT_RELEASED",
    "FIELDS",
    "GRASP_POINTS",
    "OBJECT_POINTS",
    "ON_FAIL_HOLD",
    "ON_FAIL_STOP",
    "OPS",
    "OP_GRIPPER",
    "OP_MOVE",
    "OP_MOVE_JOINTS",
    "OP_MOVE_LINEAR",
    "OP_WAIT",
    "POSE_OPS",
    "SPEED_ARG",
    "STEP_SCHEMA",
]
