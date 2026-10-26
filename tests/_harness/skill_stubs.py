# Copyright (c) Alibaba, Inc. and its affiliates.
"""A simulated Piper-shaped workcell for node-skill tests.

``SimArmTransport`` behaves like the real arm at the level skills observe it:
an ``ee.*`` or ``joint.*`` batch sets a target, ``arm.motion`` reads False
until the arm has "travelled" for a few polls, and the gripper closes until it
reaches its target or an object (``object_width_mm``). ``SkillNodeRegistry``
serves it under the shipped ``piper_arm.yaml`` declaration, so envelopes, rate
limits, and tolerances are the production ones, with a real ``CalibrationStore``.

``SimRgbdCamera`` renders boxes standing on a table for a camera looking
straight down, and ``SimLocator`` answers ``grounding`` with the masks (or
boxes, or centre points) that camera last rendered -- enough to drive
perception end to end.
``SimPolicy`` serves ``action_chunk`` under a declared action space: each chunk
steps the native space by a fixed amount per action, so a stream's commands
are predictable; ``action_space`` / ``embodiment`` build the two declarations.
"""

from __future__ import annotations

import asyncio
import math
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import yaml

from leapflow.hardware.calibration_store import CalibrationRecord, CalibrationStore
from leapflow.hardware.context import Channel, HardwareContext
from leapflow.hardware.model_contract import (
    TASK_ACTION_CHUNK,
    TASK_GROUNDING,
    ActionChunkResult,
    Detection,
    GroundingResult,
    MaskRle,
    ModelIdentity,
    ModelInfo,
)
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills.adapt.rotations import ARM_EULER, rotation_width
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    BatchReading,
    BatchWriteOutcome,
    Reading,
    TransportError,
    WriteOutcome,
)

PIPER_YAML = (
    Path(__file__).parents[2] / "src" / "leapflow" / "hardware" / "examples" / "piper_arm.yaml"
)
ARM_ID = "robot_piper_arm"
WORKCELL_ID = "piper_cell"
TOOL_LENGTH_MM = 120.0
# Flange pointing straight down at (250, 0, 370) puts the TCP at (250, 0, 250).
START_FLANGE = (250.0, 0.0, 250.0 + TOOL_LENGTH_MM, 180.0, 0.0, 0.0)
SAFE_BOX = {"x_min": 100, "x_max": 450, "y_min": -250, "y_max": 250, "z_min": 80, "z_max": 450}
JOINTS = tuple(f"joint.j{i}.position" for i in range(1, 7))
_AXES = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")

CAMERA_ID = "realsense_cam"
LOCATOR_ID = "perception_locator"
POLICY_ID = "policy_server"
TABLE_Z = 60.0
# A camera 540 mm above the table looking straight down (camera +Z = base -Z).
CAMERA_POSE = {"x": 300.0, "y": 0.0, "z": 600.0, "rx": 180.0, "ry": 0.0, "rz": 0.0}
SIM_INTRINSICS = CameraIntrinsics(width=200, height=150, fx=250.0, fy=250.0, ppx=100.0, ppy=75.0)
# The target mount a hand-eye record must carry: which frame the board was fixed
# to when the mount was measured (``calib.handeye`` writes the real one).
IDENTITY_MATRIX = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def piper_context() -> HardwareContext:
    return HardwareContext.from_mapping(yaml.safe_load(PIPER_YAML.read_text(encoding="utf-8")))


class SimArmTransport:
    """Piper-shaped arm + gripper that settles after ``travel_reads`` polls."""

    def __init__(
        self,
        *,
        flange: tuple[float, ...] = START_FLANGE,
        gripper_mm: float = 60.0,
        enabled: bool = True,
        travel_reads: int = 2,
        stall: bool = False,
        object_width_mm: float | None = None,
        refuse_writes: bool = False,
        joints: Sequence[float] | None = None,
        forward: Callable[[Sequence[float]], Pose] | None = None,
        arm_status: int | None = None,
        joint_communication_errors: Sequence[int] = (),
    ) -> None:
        self.joints = dict(zip(JOINTS, joints or (0.0,) * len(JOINTS)))
        self.forward = forward  # when set, joint moves drive the flange pose
        self.pose = list(flange if forward is None else self._forward_pose())
        self.gripper_mm = gripper_mm
        self.gripper_target = gripper_mm
        self.enabled = enabled
        self.travel_reads = travel_reads
        self.stall = stall
        self.object_width_mm = object_width_mm
        self.refuse_writes = refuse_writes
        self.arm_status = arm_status  # vendor verdict on the last target, 0x04 = refused
        self.joint_communication_errors = tuple(joint_communication_errors)
        self.halts = 0
        self.batches: list[tuple[tuple[str, Any], ...]] = []
        self._pending: dict[str, float] = {}
        self._remaining = 0

    # ── Transport surface ──

    async def open(self, context: HardwareContext) -> None:
        return None

    async def close(self) -> None:
        return None

    async def halt(self) -> None:
        self.halts += 1
        self._pending.clear()
        self.gripper_target = self.gripper_mm

    async def read(self, channel_id: str) -> Reading:
        return (await self.read_batch((channel_id,))).readings[0]

    async def read_batch(self, channel_ids: tuple[str, ...]) -> BatchReading:
        self._advance()
        readings = tuple(
            Reading(device_id=ARM_ID, channel_id=cid, value=self._value(cid)) for cid in channel_ids
        )
        return BatchReading(device_id=ARM_ID, readings=readings)

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        batch = await self.write_batch(((channel_id, value),))
        return batch.outcomes[0]

    async def write_batch(self, commands: tuple[tuple[str, Any], ...]) -> BatchWriteOutcome:
        if self.refuse_writes:
            raise TransportError("simulated bus failure", failure_code="transport_error")
        self.batches.append(tuple(commands))
        for channel_id, value in commands:
            if channel_id.startswith(("ee.", "joint.")):
                self._pending[channel_id] = float(value)
                self._remaining = self.travel_reads
            elif channel_id == "gripper.position":
                self.gripper_target = float(value)
        outcomes = tuple(
            WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED) for _ in commands
        )
        return BatchWriteOutcome(
            ok=True, outcomes=outcomes, side_effect_state=SIDE_EFFECT_COMMITTED
        )

    # ── Simulation ──

    @property
    def flange(self) -> Pose:
        return Pose(*self.pose)

    @property
    def moving(self) -> bool:
        return bool(self._pending)

    def _advance(self) -> None:
        if self._pending and not self.stall:
            self._remaining -= 1
            if self._remaining <= 0:
                for channel_id, value in self._pending.items():
                    if channel_id in _AXES:
                        self.pose[_AXES.index(channel_id)] = value
                    else:
                        self.joints[channel_id] = value
                if self.forward is not None:
                    self.pose = list(self._forward_pose())
                self._pending.clear()
        target = self.gripper_target
        held = self.object_width_mm
        if held is not None and target < held <= self.gripper_mm:
            target = held  # closing on an object stops at the object
        step = max(-10.0, min(10.0, target - self.gripper_mm))
        self.gripper_mm += step

    def _forward_pose(self) -> tuple[float, ...]:
        assert self.forward is not None
        return self.forward([self.joints[cid] for cid in JOINTS]).as_tuple()

    def _value(self, channel_id: str) -> Any:
        if channel_id in _AXES:
            return self.pose[_AXES.index(channel_id)]
        if channel_id in self.joints:
            return self.joints[channel_id]
        if channel_id == "gripper.position":
            return self.gripper_mm
        if channel_id == "arm.motion":
            return not self._pending
        if channel_id == "arm.enabled":
            return self.enabled
        if channel_id == "arm.status":
            if self.arm_status is None:
                return None
            return {"ctrl_mode": 0x01, "arm_status": self.arm_status, "motion_status": 0x00,
                    "joint_communication_errors": list(self.joint_communication_errors)}
        return None


class SkillNodeRegistry:
    """The node registry surface skills, the gate, and the runtime use.

    *devices* adds transports by device id (a camera, a model service) next to
    the arm; each is "declared" for admission.
    """

    def __init__(
        self,
        transport: SimArmTransport | None = None,
        *,
        store_path: Path | None = None,
        safety_deny_reason: str = "",
        devices: Mapping[str, Any] | None = None,
    ) -> None:
        self.sim = transport or SimArmTransport()
        self.devices = dict(devices or {})
        self._context = piper_context()
        self._io_lock = asyncio.Lock()
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        self._store = CalibrationStore(db_path=store_path) if store_path is not None else None
        self._deny = safety_deny_reason

    def contexts(self) -> tuple[HardwareContext, ...]:
        return (self._context,)

    def context(self, device_id: str) -> Any:
        if device_id == ARM_ID:
            return self._context
        return SimpleNamespace(device_id=device_id) if device_id in self.devices else None

    def device_io(self, device_id: str) -> asyncio.Lock:
        return self._io_lock

    async def transport(self, device_id: str) -> Any:
        if device_id == ARM_ID:
            return self.sim
        if device_id in self.devices:
            return self.devices[device_id]
        raise TransportError(f"unknown device {device_id!r}", failure_code="unknown_device")

    def get_open_transport(self, device_id: str) -> Any:
        """The live link only; never opens one (matches ``HardwareRegistry``).

        This harness opens nothing lazily: a declared device is live from the
        moment the registry is built, which is what makes a halt reach it.
        """
        if device_id == ARM_ID:
            return self.sim
        if device_id in self.devices:
            return self.devices[device_id]
        raise TransportError(
            f"No open transport for device {device_id!r}.",
            failure_code="transport_not_open",
        )

    async def halt(self, device_id: str) -> Any:
        """Stop a declared device, matching ``HardwareRegistry.halt``.

        This harness caches nothing lazily -- a declared device is live from the
        moment the registry is built -- so there is always a link to stop over.
        """
        return await (await self.transport(device_id)).halt()

    async def read_batch(self, device_id: str, channel_ids: tuple[str, ...]) -> BatchReading:
        transport = await self.transport(device_id)
        async with self._io_lock:
            return await transport.read_batch(tuple(channel_ids))

    def check_safety_policy(
        self, context: HardwareContext, channel: Channel, value: Any, *, rate: float | None = None
    ) -> tuple[bool, str]:
        return (False, self._deny) if self._deny else (True, "")

    def record_command(self, device_id: str, channel_id: str, value: float) -> None:
        self._last[(device_id, channel_id)] = (float(value), time.monotonic())

    def last_command(self, device_id: str, channel_id: str) -> tuple[float, float] | None:
        return self._last.get((device_id, channel_id))

    def forget_rate_baselines(self) -> None:
        """Let consecutive test moves skip slew spacing (time 'passed')."""
        self._last.clear()

    @property
    def calibration_store(self) -> CalibrationStore | None:
        return self._store

    def calibrate(
        self,
        procedure_id: str,
        *,
        device_id: str = WORKCELL_ID,
        parameters: Mapping[str, Any] | None = None,
        matrix: Sequence[Sequence[float]] | None = None,
        pose: Mapping[str, Any] | None = None,
        recorded_at: float | None = None,
    ) -> float:
        assert self._store is not None, "build the registry with store_path to calibrate"
        at = time.time() if recorded_at is None else recorded_at
        assert self._store.record(
            CalibrationRecord(
                device_id=device_id,
                procedure_id=procedure_id,
                recorded_at=at,
                parameters=dict(parameters or {}),
                matrix=None if matrix is None else [list(row) for row in matrix],
                pose=None if pose is None else dict(pose),
            )
        )
        return at

    def calibrate_cell(self, box: Mapping[str, Any] | None = SAFE_BOX) -> None:
        """TCP offset straight out of the flange plus the safe box.

        *box* replaces the measured workspace; ``None`` leaves the workspace
        unmeasured, so skills have to plan in the cell's bootstrap box.
        """
        self.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
        if box is not None:
            self.calibrate("workspace.safe_box", parameters=dict(box))

    def calibrate_perception(
        self,
        *,
        table_z: float | None = TABLE_Z,
        intrinsics: CameraIntrinsics | None = SIM_INTRINSICS,
        camera_pose: Mapping[str, float] = CAMERA_POSE,
    ) -> None:
        """The cell plus a table-referenced workspace and a base-mounted camera."""
        self.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
        box: dict[str, Any] = dict(SAFE_BOX)
        if table_z is not None:
            box["floor"] = {"method": "table_z", "table_z": table_z}
        self.calibrate("workspace.safe_box", parameters=box)
        self.calibrate(
            "camera.handeye",
            pose=camera_pose,
            parameters={
                "target_mount": IDENTITY_MATRIX,
                **({"intrinsics": intrinsics.to_dict()} if intrinsics is not None else {}),
            },
        )


WORKCELL_YAML: dict[str, Any] = {
    "workcell": WORKCELL_ID,
    "roles": {"arm": ARM_ID, "gripper": ARM_ID},
    "frames": {"tcp": {"parent": "flange", "procedure": "tcp.offset"}},
    "workspace": {
        "procedure": "workspace.safe_box",
        "tool_cone_deg": 30,
        "bootstrap_box": {"x": [150, 400], "y": [-200, 200], "z": [120, 400]},
    },
    "home": {"joint.j1.position": 0.0, "joint.j2.position": 10.0},
    "gripper_spec": {"max_width_mm": 70, "min_hold_width_mm": 3},
    "limits": {"speed_pct_max": 30, "default_speed_pct": 20, "move_timeout_s": 5},
}

PERCEPTION_WORKCELL_YAML: dict[str, Any] = {
    **WORKCELL_YAML,
    "roles": {**WORKCELL_YAML["roles"], "camera": CAMERA_ID},
    "frames": {
        **WORKCELL_YAML["frames"],
        "camera": {"parent": "base", "procedure": "camera.handeye"},
    },
    "models": {"locator": LOCATOR_ID},
}


# -- Perception simulation --


@dataclass(frozen=True)
class SimBox:
    """A box standing on the table: footprint centre (mm), size, yaw of its length."""

    label: str
    center: tuple[float, float]
    length: float
    width: float
    height: float
    yaw_deg: float = 0.0
    score: float = 0.9

    def covers(self, x: float, y: float) -> bool:
        yaw = math.radians(self.yaw_deg)
        dx, dy = x - self.center[0], y - self.center[1]
        along = dx * math.cos(yaw) + dy * math.sin(yaw)
        across = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return abs(along) <= self.length / 2.0 and abs(across) <= self.width / 2.0


class SimRgbdCamera:
    """A ``FrameSource`` over *boxes*, seen from ``CAMERA_POSE`` looking straight down."""

    def __init__(
        self,
        boxes: Sequence[SimBox] = (),
        *,
        intrinsics: CameraIntrinsics = SIM_INTRINSICS,
        table_z: float = TABLE_Z,
        fail: bool = False,
    ) -> None:
        self.boxes = list(boxes)
        self.intrinsics = intrinsics
        self.table_z = table_z
        self.fail = fail
        self.captures = 0
        self.halts = 0
        self.masks: list[tuple[SimBox, MaskRle]] = []

    async def halt(self) -> None:
        self.halts += 1

    async def capture_rgbd(self) -> RgbdFrame:
        if self.fail:
            raise TransportError("simulated USB reset", failure_code="realsense_timeout")
        self.captures += 1
        k = self.intrinsics
        w, h = k.width, k.height
        cx, cy, cz = CAMERA_POSE["x"], CAMERA_POSE["y"], CAMERA_POSE["z"]
        depth = array("H", [0]) * (w * h)
        hits: list[list[bool]] = [[False] * (w * h) for _ in self.boxes]
        for u in range(w):
            for v in range(h):
                rx, ry = (u - k.ppx) / k.fx, (v - k.ppy) / k.fy
                best, owner = cz - self.table_z, -1
                for index, box in enumerate(self.boxes):
                    d = cz - (self.table_z + box.height)
                    if d < best and box.covers(cx + rx * d, cy - ry * d):
                        best, owner = d, index
                depth[v * w + u] = round(best)
                if owner >= 0:
                    hits[owner][u * h + v] = True  # column-major, like COCO RLE
        self.masks = [(box, _rle(hit, h, w)) for box, hit in zip(self.boxes, hits)]
        return RgbdFrame(
            device_id=CAMERA_ID,
            width=w,
            height=h,
            color=bytes(w * h * 3),
            depth=depth.tobytes(),
            depth_scale_m=0.001,
            intrinsics=k,
            observed_at=time.time(),
            monotonic_at=time.monotonic(),
            sequence=self.captures,
        )


def _rle(mask: Sequence[bool], height: int, width: int) -> MaskRle:
    counts: list[int] = []
    current, run = False, 0
    for value in mask:
        if value != current:
            counts.append(run)
            current, run = value, 0
        run += 1
    counts.append(run)
    return MaskRle(height=height, width=width, counts=tuple(counts))


class SimLocator:
    """A ``ModelService`` grounding queries in what *camera* last rendered.

    A box matches a query its label contains; the detection carries the query
    as its label, as an open-vocabulary model does. *geometry* picks what each
    detection carries: the rendered ``mask``, its pixel ``box``, or the box's
    centre ``point`` (a pointing VLM).
    """

    def __init__(
        self,
        camera: SimRgbdCamera,
        *,
        task: str = TASK_GROUNDING,
        fail: bool = False,
        geometry: str = "mask",
    ) -> None:
        self.camera = camera
        self.task = task
        self.fail = fail
        self.geometry = geometry
        self.requests: list[Mapping[str, Any]] = []

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task=self.task, name="sim-locator", version="1")

    async def refresh(self) -> ModelInfo:
        return self.model_info

    async def halt(self) -> None:
        return None

    async def infer(self, request: Mapping[str, Any]) -> GroundingResult:
        self.requests.append(request)
        if self.fail:
            raise TransportError("model server unreachable", failure_code="model_unreachable")
        detections = [
            self._detection(query, box.score, mask)
            for query in request["queries"]
            for box, mask in self.camera.masks
            if query.casefold() in box.label.casefold() and mask.area
        ]
        return GroundingResult(tuple(detections), 12.0, self.model_info.identity)

    def _detection(self, label: str, score: float, mask: MaskRle) -> Detection:
        if self.geometry == "mask":
            return Detection(label, score, mask=mask)
        pixels = list(mask.foreground())
        us, vs = [u for u, _ in pixels], [v for _, v in pixels]
        box = (float(min(us)), float(min(vs)), float(max(us) + 1), float(max(vs) + 1))
        if self.geometry == "box":
            return Detection(label, score, box=box)
        return Detection(label, score, point=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


GRIPPER_CHANNEL = "gripper.position"
TOP_IMAGE = {"top": {"width": 16, "height": 12}}


def action_space(
    kind: str = "ee_pose",
    *,
    fps: float = 20.0,
    gripper: Mapping[str, Any] | None = None,
    **arm: Any,
) -> dict[str, Any]:
    """A model ``action_space``: one ``arm`` group of *kind* plus an optional gripper.

    ``arm`` keys go into the arm group (unit, rotation, reference, frame, ...);
    the arm group's width follows its kind and rotation.
    """
    width = 6 if kind == "joints" else 3 + rotation_width(arm.get("rotation", ARM_EULER))
    groups: list[dict[str, Any]] = [
        {"name": "arm", "kind": kind, "dims": list(range(width)), **arm}
    ]
    if gripper is not None:
        groups.append({"name": "gripper", "kind": "gripper", "dims": [width], **gripper})
        width += 1
    return {"fps": fps, "width": width, "groups": groups}


def embodiment(space: Mapping[str, Any], **decl: Any) -> dict[str, Any]:
    """``policies.<name>`` binding every group of *space* to the Piper's channels."""
    channels = {"joints": list(JOINTS), "ee_pose": list(_AXES), "gripper": [GRIPPER_CHANNEL]}
    actions = {group["name"]: channels[group["kind"]] for group in space["groups"]}
    return {"actions": actions, **decl}


def _mirrored_state(space: Mapping[str, Any]) -> dict[str, Any]:
    """The action space read as an absolute base-frame state, dimension for dimension."""
    groups = []
    for group in space["groups"]:
        state = {**group, "reference": "absolute"}
        if state.get("frame") == "tool":
            state["frame"] = "base"
        groups.append(state)
    return {k: v for k, v in space.items() if k not in ("fps", "pending")} | {"groups": groups}


class SimPolicy:
    """An ``action_chunk`` ``ModelService`` (contract v1.1) stepping its native space.

    ``model_info`` declares *space* as its ``action_space``; the observation
    space defaults to the same layout read as an absolute state (so the state
    the request carries lines up with the action dims) plus one ``top`` image.
    Row ``t`` of a chunk is, per dimension and by its group's reference:
    absolute ``state + (t + 1) * step``, relative ``(t + 1) * step``, delta
    ``step``. ``rows`` replaces the computed chunk verbatim (malformed answers).

    ``done_after`` reports ``done`` from that many answered chunks on,
    ``progress`` is echoed with every chunk, and ``swap_after`` answers as a
    different model from that many chunks on. ``versions`` is the version each
    ``refresh`` reports in turn (the last one repeats), counted in
    ``refreshes``. ``details`` overrides any
    declared ``model_info`` field (``None`` removes it).
    """

    def __init__(
        self,
        space: Mapping[str, Any],
        step: Sequence[float],
        *,
        observation: Mapping[str, Any] | None = None,
        horizon: int = 10,
        latency_s: float = 0.0,
        fail: bool = False,
        rows: Sequence[Sequence[float]] | None = None,
        version: str = "1",
        done_after: int | None = None,
        progress: float | None = None,
        swap_after: int | None = None,
        versions: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.space = dict(space)
        self.step = tuple(float(v) for v in step)
        self.horizon = horizon
        self.latency_s = latency_s
        self.fail = fail
        self.rows = None if rows is None else tuple(tuple(row) for row in rows)
        self.version = version
        self.done_after = done_after
        self.progress = progress
        self.swap_after = swap_after
        self.versions = tuple(versions)
        self.refreshes = 0
        if observation is None:
            observation = {"state": _mirrored_state(space), "images": TOP_IMAGE}
        self.details: dict[str, Any] = {
            "action_space": self.space,
            "observation_space": dict(observation),
            "tasks": ["pick up the cube"],
            **dict(details or {}),
        }
        self.requests: list[Mapping[str, Any]] = []
        self.responses: list[tuple[tuple[float, ...], ...]] = []
        self.answered = 0

    @property
    def model_info(self) -> ModelInfo:
        details = {k: v for k, v in self.details.items() if v is not None}
        return ModelInfo(
            task=TASK_ACTION_CHUNK, name="sim-policy", version=self.version, details=details
        )

    async def refresh(self) -> ModelInfo:
        if self.versions:
            self.version = self.versions[min(self.refreshes, len(self.versions) - 1)]
        self.refreshes += 1
        return self.model_info

    async def halt(self) -> None:
        return None

    def _references(self) -> list[str]:
        references = ["absolute"] * len(self.step)
        for group in self.space["groups"]:
            for dim in group["dims"]:
                references[dim] = group.get("reference", "absolute")
        return references

    def _chunk(self, state: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        references = self._references()
        rows = []
        for t in range(self.horizon):
            row = []
            for dim, (reference, delta) in enumerate(zip(references, self.step)):
                if reference == "delta":
                    row.append(delta)
                elif reference == "relative":
                    row.append((t + 1) * delta)
                else:
                    row.append(state[dim] + (t + 1) * delta)
            rows.append(tuple(row))
        return tuple(rows)

    async def infer(self, request: Mapping[str, Any]) -> ActionChunkResult:
        self.requests.append(request)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.fail:
            raise TransportError("policy server unreachable", failure_code="model_unreachable")
        rows = self.rows if self.rows is not None else self._chunk(request["state"])
        self.responses.append(rows)
        self.answered += 1
        identity = self.model_info.identity
        if self.swap_after is not None and self.answered > self.swap_after:
            identity = ModelIdentity(identity.name, f"{identity.version}-swapped")
        done = self.done_after is not None and self.answered >= self.done_after
        return ActionChunkResult(rows, 5.0, identity, done=done, progress=self.progress)


def perception_registry(
    store_path: Path,
    boxes: Sequence[SimBox] = (),
    *,
    arm: SimArmTransport | None = None,
    camera: SimRgbdCamera | None = None,
) -> SkillNodeRegistry:
    """An arm, a camera over *boxes*, and a locator, for ``PERCEPTION_WORKCELL_YAML``."""
    camera = camera or SimRgbdCamera(boxes)
    return SkillNodeRegistry(
        arm,
        store_path=store_path,
        devices={CAMERA_ID: camera, LOCATOR_ID: SimLocator(camera)},
    )


SEQUENCE = "motion.sequence"


def move_by(
    dz: float = 0.0,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    frame: str = "base",
    speed_pct: float | None = None,
    **step: Any,
) -> dict[str, Any]:
    """``motion.sequence`` args for one relative move from the current TCP."""
    move = {"op": "move", "relative": {"dx": dx, "dy": dy, "dz": dz, "frame": frame}, **step}
    args: dict[str, Any] = {"steps": [move]}
    if speed_pct is not None:
        args["speed_pct"] = speed_pct
    return args


def grip(action: str, **step: Any) -> dict[str, Any]:
    """``motion.sequence`` args for one gripper step (``open`` or ``close``)."""
    return {"steps": [{"op": "gripper", "grip": action, **step}]}


def write_workcell(directory: Path, data: Mapping[str, Any] | None = None, name: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(WORKCELL_YAML if data is None else data)
    path = directory / f"{name or payload.get('workcell') or 'cell'}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


__all__ = [
    "ARM_ID",
    "CAMERA_ID",
    "CAMERA_POSE",
    "GRIPPER_CHANNEL",
    "JOINTS",
    "LOCATOR_ID",
    "PERCEPTION_WORKCELL_YAML",
    "POLICY_ID",
    "SAFE_BOX",
    "SEQUENCE",
    "SIM_INTRINSICS",
    "START_FLANGE",
    "TABLE_Z",
    "TOOL_LENGTH_MM",
    "TOP_IMAGE",
    "WORKCELL_ID",
    "WORKCELL_YAML",
    "SimArmTransport",
    "SimBox",
    "SimLocator",
    "SimPolicy",
    "SimRgbdCamera",
    "SkillNodeRegistry",
    "action_space",
    "embodiment",
    "grip",
    "move_by",
    "perception_registry",
    "piper_context",
    "write_workcell",
]
