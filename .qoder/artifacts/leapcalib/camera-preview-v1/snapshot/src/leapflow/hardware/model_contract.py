# Copyright (c) Alibaba, Inc. and its affiliates.
"""The ``ModelService`` capability and the v1.1 model-serving wire contract.

A served model is a device: declared in YAML, opened by a transport, admitted by
the registry. What a skill needs from it is narrower than any one binding -- the
model's identity, and one typed inference call per task. This module owns both,
plus the parsing that turns a wire payload into a typed result, so an HTTP
binding today and an in-process binding later validate identically.

Three tasks exist, and each is a contract, not a model choice:

``grounding``
    ``{image: <JPEG base64>, queries: [str]}`` ->
    ``{detections: [{label, score, box?, mask_rle?, point?}], infer_ms}``. Each
    detection carries whatever its model grounds with -- a segmenter's mask, a
    detector's ``box`` ``[x0, y0, x1, y1]``, a pointing VLM's ``point``
    ``[u, v]`` -- and at least one of them. 2-D only, in input-image pixels;
    lifting to 3-D happens on the node, which owns depth and calibration.

``action_chunk``
    ``{images: {<camera>: <JPEG base64>}, state: [float], task: str,
    pending?: {actions: [[float]], latency_steps: int}}`` ->
    ``{actions: [[float] * D] * H, infer_ms, done?: bool, progress?: float}``.
    Rows are in the model's *native* space, which ``model_info`` declares as
    ``action_space`` / ``observation_space`` (see ``skills.adapt.spec``) with an
    ``instruction_policy`` (``open``, or ``exact`` over its ``tasks``); the node
    owns every mapping onto the embodiment, so the server only runs the forward
    pass. ``pending`` (sent when the action space declares ``pending: true``)
    carries the not-yet-executed rows of the previous chunk, in the model's
    space, and how many steps the next answer will arrive late -- what
    real-time chunking conditions on. ``done`` says the instruction is
    accomplished (the node finishes the current chunk and stops); ``progress``
    in ``[0, 1]`` is reported, not acted on.

``grasp_proposal``
    ``{image: <JPEG base64>, depth: {data: <uint16 little-endian base64>,
    scale_m}, intrinsics: {...}, object: {label, center: [x, y, z], pixel:
    [u, v], size: {length, width, height}}, gripper: {max_width_mm}, top_k}`` ->
    ``{grasps: [{pose: {x, y, z, rx, ry, rz}, width_mm, score, approach_mm?}],
    infer_ms}``. Object centre and grasp poses are in the camera frame (mm,
    deg, the arm's Euler convention); the node maps them into the base frame.

``mask_rle`` is COCO's uncompressed RLE: ``{size: [h, w], counts: [int]}``,
column-major, alternating background/foreground runs starting with background.

Every ``/v1/infer`` response echoes the serving identity as ``model: {name,
version, checksum, instance}`` (``instance`` names the serving process), and
the binding compares it with the identity it verified: a model swapped or
restarted between calls is caught on the very next response, not at the next
open.

Pure Python, so the node runtime imports it without numpy.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

TASK_GROUNDING = "grounding"
TASK_ACTION_CHUNK = "action_chunk"
TASK_GRASP_PROPOSAL = "grasp_proposal"
CONTRACT_VERSION = "1.1"
MODEL_TASKS = frozenset({TASK_GROUNDING, TASK_ACTION_CHUNK, TASK_GRASP_PROPOSAL})
GEOMETRY_MASK = "mask"
GEOMETRY_BOX = "box"
GEOMETRY_POINT = "point"
BINDING_KEYS = ("name", "version", "checksum")
IDENTITY_KEYS = (*BINDING_KEYS, "instance")
MAX_QUERIES = 16
_POSE_KEYS = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(frozen=True)
class ModelIdentity:
    """Which weights answer, and which serving process: what every response echoes."""

    name: str = ""
    version: str = ""
    checksum: str = ""
    instance: str = ""

    @classmethod
    def from_mapping(cls, data: Any) -> "ModelIdentity":
        if not isinstance(data, Mapping):
            raise ValueError("model must be an object naming the serving identity")
        values = {key: str(data.get(key) or "").strip() for key in IDENTITY_KEYS}
        if not values["name"] or not values["version"]:
            raise ValueError("model identity needs a name and a version")
        return cls(**values)

    def difference(self, other: "ModelIdentity") -> str:
        """How *other* differs from this identity (``""`` when it is the same)."""
        for key in IDENTITY_KEYS:
            mine, theirs = getattr(self, key), getattr(other, key)
            if mine != theirs:
                return f"{key} {theirs!r} != {mine!r}"
        return ""

    def to_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in IDENTITY_KEYS}


@dataclass(frozen=True)
class ModelInfo:
    """What the server says it is serving; ``details`` keeps every other field."""

    task: str
    name: str
    version: str
    checksum: str = ""
    instance: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Any) -> "ModelInfo":
        """Parse ``/v1/health``'s ``model_info``; raises ``ValueError`` when unusable."""
        if not isinstance(data, Mapping):
            raise ValueError("model_info must be an object")
        values = {key: str(data.get(key) or "").strip() for key in ("task", *IDENTITY_KEYS)}
        missing = [key for key in ("task", "name", "version") if not values[key]]
        if missing:
            raise ValueError(f"model_info is missing {', '.join(missing)}")
        details = {k: v for k, v in data.items() if k not in ("task", *IDENTITY_KEYS)}
        return cls(details=details, **values)

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(self.name, self.version, self.checksum, self.instance)

    def binding_problem(self, expect: Mapping[str, Any]) -> str:
        """Why this model is not the one a declaration pinned (``""`` when it is)."""
        for key in BINDING_KEYS:
            wanted = str(expect.get(key) or "").strip()
            actual = getattr(self, key)
            if wanted and wanted != actual:
                return f"{key} {actual!r} != expected {wanted!r}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **dict(self.details),
            "task": self.task,
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "instance": self.instance,
        }


@dataclass(frozen=True)
class MaskRle:
    """A binary mask as column-major run lengths, background first."""

    height: int
    width: int
    counts: tuple[int, ...]

    @classmethod
    def from_mapping(cls, data: Any) -> "MaskRle":
        if not isinstance(data, Mapping):
            raise ValueError("mask_rle must be an object")
        size, counts = data.get("size"), data.get("counts")
        if not (isinstance(size, Sequence) and len(size) == 2):
            raise ValueError("mask_rle.size must be [height, width]")
        height, width = (_int(value, "mask_rle.size") for value in size)
        if height <= 0 or width <= 0:
            raise ValueError("mask_rle.size must be positive")
        if isinstance(counts, (str, bytes)) or not isinstance(counts, Sequence):
            raise ValueError("mask_rle.counts must be an integer list (uncompressed RLE)")
        runs = tuple(_int(value, "mask_rle.counts") for value in counts)
        if any(run < 0 for run in runs):
            raise ValueError("mask_rle.counts must be non-negative")
        if sum(runs) != height * width:
            raise ValueError("mask_rle.counts must cover the whole image")
        return cls(height=height, width=width, counts=runs)

    @property
    def area(self) -> int:
        return sum(self.counts[1::2])

    def foreground(self, *, stride: int = 1) -> Iterator[tuple[int, int]]:
        """Yield ``(u, v)`` foreground pixels; *stride* keeps every n-th of them."""
        step = max(1, int(stride))
        index, kept = 0, 0
        for position, run in enumerate(self.counts):
            if position % 2:
                for offset in range(run):
                    if kept % step == 0:
                        pixel = index + offset
                        yield (pixel // self.height, pixel % self.height)
                    kept += 1
            index += run

    def to_dict(self) -> dict[str, Any]:
        return {"size": [self.height, self.width], "counts": list(self.counts)}


@dataclass(frozen=True)
class Detection:
    """One grounded query: which, how sure, and where in the image (mask, box, or point).

    ``geometry`` names the richest support present -- a mask over a box over a
    point -- which is what the node lifts.
    """

    label: str
    score: float
    box: tuple[float, float, float, float] | None = None
    mask: MaskRle | None = None
    point: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.box is None and self.mask is None and self.point is None:
            raise ValueError("a detection needs a mask_rle, a box, or a point")

    @property
    def geometry(self) -> str:
        if self.mask is not None:
            return GEOMETRY_MASK
        return GEOMETRY_BOX if self.box is not None else GEOMETRY_POINT

    @classmethod
    def from_mapping(cls, data: Any) -> "Detection":
        if not isinstance(data, Mapping):
            raise ValueError("a detection must be an object")
        label = str(data.get("label") or "").strip()
        if not label:
            raise ValueError("a detection needs a label")
        score = _finite(data.get("score"), "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        raw_mask = data.get("mask_rle")
        return cls(
            label,
            score,
            box=None if data.get("box") is None else parse_box(data["box"]),
            mask=None if raw_mask is None else MaskRle.from_mapping(raw_mask),
            point=None if data.get("point") is None else parse_point(data["point"]),
        )


def parse_box(value: Any) -> tuple[float, float, float, float]:
    """``[x0, y0, x1, y1]`` in pixels, ordered; raises ``ValueError``."""
    box = _numbers(value, "box")
    if len(box) != 4 or box[2] < box[0] or box[3] < box[1] or min(box) < 0:
        raise ValueError("box must be [x0, y0, x1, y1] with 0 <= x0 <= x1 and 0 <= y0 <= y1")
    return (box[0], box[1], box[2], box[3])


def parse_point(value: Any) -> tuple[float, float]:
    """``[u, v]`` in pixels; raises ``ValueError``."""
    point = _numbers(value, "point")
    if len(point) != 2 or min(point) < 0:
        raise ValueError("point must be [u, v] with non-negative pixel coordinates")
    return (point[0], point[1])


@dataclass(frozen=True)
class GroundingResult:
    detections: tuple[Detection, ...]
    infer_ms: float
    model: ModelIdentity = field(default_factory=ModelIdentity)


@dataclass(frozen=True)
class ActionChunkResult:
    """``actions[t]`` is the native action for step ``t``; every row has the same width."""

    actions: tuple[tuple[float, ...], ...]
    infer_ms: float
    model: ModelIdentity = field(default_factory=ModelIdentity)
    done: bool = False
    progress: float | None = None

    @property
    def horizon(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class ProposedGrasp:
    """One grasp a model proposes: a camera-frame TCP pose (mm, deg) and its opening."""

    pose: tuple[float, float, float, float, float, float]
    width_mm: float
    score: float
    approach_mm: float | None = None

    @classmethod
    def from_mapping(cls, data: Any) -> "ProposedGrasp":
        if not isinstance(data, Mapping) or not isinstance(data.get("pose"), Mapping):
            raise ValueError("a grasp must be an object with a pose object")
        pose = data["pose"]
        values = tuple(_finite(pose.get(key), f"pose.{key}") for key in _POSE_KEYS)
        width = _finite(data.get("width_mm"), "width_mm")
        score = _finite(data.get("score"), "score")
        if width <= 0 or not 0.0 <= score <= 1.0:
            raise ValueError("a grasp needs a positive width_mm and a score in [0, 1]")
        approach = data.get("approach_mm")
        approach_mm = None if approach is None else _finite(approach, "approach_mm")
        if approach_mm is not None and approach_mm <= 0:
            raise ValueError("approach_mm must be positive")
        return cls(values, width, score, approach_mm)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GraspProposalResult:
    grasps: tuple[ProposedGrasp, ...]
    infer_ms: float
    model: ModelIdentity = field(default_factory=ModelIdentity)


InferenceResult = GroundingResult | ActionChunkResult | GraspProposalResult


def check_request(task: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a request before it leaves the node; raises ``ValueError``."""
    if task == TASK_GRASP_PROPOSAL:
        return _check_grasp_request(request)
    if task == TASK_GROUNDING:
        queries = request.get("queries")
        if not isinstance(request.get("image"), str) or not request["image"]:
            raise ValueError("image must be a base64 JPEG string")
        if isinstance(queries, str) or not isinstance(queries, Sequence) or not queries:
            raise ValueError("queries must be a non-empty list of strings")
        if len(queries) > MAX_QUERIES or not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError(f"queries must be 1..{MAX_QUERIES} non-empty strings")
        return {"image": request["image"], "queries": [q.strip() for q in queries]}
    if task == TASK_ACTION_CHUNK:
        return _check_action_request(request)
    raise ValueError(f"unknown model task {task!r}")


def _check_action_request(request: Mapping[str, Any]) -> dict[str, Any]:
    images = request.get("images")
    if not isinstance(images, Mapping) or not images:
        raise ValueError("images must map camera names to base64 JPEG strings")
    if not all(isinstance(v, str) and v for v in images.values()):
        raise ValueError("every image must be a base64 JPEG string")
    state = _numbers(request.get("state"), "state")
    instruction = str(request.get("task") or "").strip()
    if not instruction:
        raise ValueError("task must be a non-empty instruction")
    checked: dict[str, Any] = {"images": dict(images), "state": list(state), "task": instruction}
    pending = request.get("pending")
    if pending is not None:
        if not isinstance(pending, Mapping):
            raise ValueError("pending must be {actions, latency_steps}")
        raw = pending.get("actions")
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError("pending.actions must be a list of rows")
        rows = [list(_numbers(row, "pending.actions[]")) for row in raw]
        if rows and any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("every pending row must have the same width")
        latency = _int(pending.get("latency_steps"), "pending.latency_steps")
        if latency < 0:
            raise ValueError("pending.latency_steps must be non-negative")
        checked["pending"] = {"actions": rows, "latency_steps": latency}
    return checked


def _check_grasp_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request.get("image"), str) or not request["image"]:
        raise ValueError("image must be a base64 JPEG string")
    depth, obj, gripper = (request.get(key) for key in ("depth", "object", "gripper"))
    if not isinstance(depth, Mapping) or not isinstance(depth.get("data"), str):
        raise ValueError("depth must carry base64 uint16 data")
    if not _finite(depth.get("scale_m"), "depth.scale_m") > 0:
        raise ValueError("depth.scale_m must be positive")
    if not isinstance(request.get("intrinsics"), Mapping):
        raise ValueError("intrinsics must be an object")
    if not isinstance(obj, Mapping) or len(_numbers(obj.get("center"), "object.center")) != 3:
        raise ValueError("object must carry a three-coordinate centre")
    if (
        not isinstance(gripper, Mapping)
        or not _finite(gripper.get("max_width_mm"), "gripper.max_width_mm") > 0
    ):
        raise ValueError("gripper.max_width_mm must be positive")
    top_k = _int(request.get("top_k"), "top_k")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return dict(request)


def parse_response(task: str, payload: Any) -> InferenceResult:
    """Turn a ``/v1/infer`` body into a typed result; raises ``ValueError``."""
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    infer_ms = _finite(payload.get("infer_ms"), "infer_ms")
    if infer_ms < 0:
        raise ValueError("infer_ms must be non-negative")
    model = ModelIdentity.from_mapping(payload.get("model"))
    if task == TASK_GROUNDING:
        raw = payload.get("detections")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("detections must be a list")
        return GroundingResult(tuple(Detection.from_mapping(d) for d in raw), infer_ms, model)
    if task == TASK_ACTION_CHUNK:
        return _parse_actions(payload, infer_ms, model)
    if task == TASK_GRASP_PROPOSAL:
        raw = payload.get("grasps")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("grasps must be a list")
        return GraspProposalResult(
            tuple(ProposedGrasp.from_mapping(g) for g in raw), infer_ms, model
        )
    raise ValueError(f"unknown model task {task!r}")


def _parse_actions(
    payload: Mapping[str, Any], infer_ms: float, model: ModelIdentity
) -> ActionChunkResult:
    raw = payload.get("actions")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ValueError("actions must be a non-empty list of rows")
    rows = tuple(_numbers(row, "actions[]") for row in raw)
    if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("every action row must have the same non-zero width")
    done = payload.get("done", False)
    if not isinstance(done, bool):
        raise ValueError("done must be a boolean")
    progress = payload.get("progress")
    if progress is not None:
        progress = _finite(progress, "progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
    return ActionChunkResult(rows, infer_ms, model, done, progress)


@runtime_checkable
class ModelService(Protocol):
    """A served model a node-local skill may call."""

    @property
    def model_info(self) -> ModelInfo:
        """The identity last verified; raises ``TransportError`` when not open."""
        ...

    async def refresh(self) -> ModelInfo:
        """Re-read and re-verify the serving identity now (a swap shows up here)."""
        ...

    async def infer(self, request: Mapping[str, Any]) -> InferenceResult:
        """One inference for the service's task; raises ``TransportError`` on failure,
        including a response from an identity other than the verified one."""
        ...


async def ground(
    service: ModelService, image_jpeg: bytes, queries: Sequence[str]
) -> GroundingResult:
    """Ask a ``grounding`` service where *queries* appear in an image."""
    result = await service.infer(
        {"image": base64.b64encode(image_jpeg).decode("ascii"), "queries": list(queries)}
    )
    if not isinstance(result, GroundingResult):
        raise TypeError(f"model {service.model_info.name!r} does not ground queries")
    return result


async def act(
    service: ModelService,
    images_jpeg: Mapping[str, bytes],
    state: Sequence[float],
    instruction: str,
    *,
    pending: Sequence[Sequence[float]] | None = None,
    latency_steps: int = 0,
) -> ActionChunkResult:
    """Ask an ``action_chunk`` service for the next chunk of actions.

    *pending* (the previous chunk's not-yet-executed native rows) is sent only
    when given, so a model that never declared ``pending`` never sees the key.
    """
    images = {name: base64.b64encode(data).decode("ascii") for name, data in images_jpeg.items()}
    request: dict[str, Any] = {"images": images, "state": list(state), "task": instruction}
    if pending is not None:
        request["pending"] = {
            "actions": [list(row) for row in pending],
            "latency_steps": max(0, int(latency_steps)),
        }
    result = await service.infer(request)
    if not isinstance(result, ActionChunkResult):
        raise TypeError(f"model {service.model_info.name!r} does not produce actions")
    return result


async def propose_grasps(service: ModelService, request: Mapping[str, Any]) -> GraspProposalResult:
    """Ask a ``grasp_proposal`` service for grasps of one object (request as documented)."""
    result = await service.infer(request)
    if not isinstance(result, GraspProposalResult):
        raise TypeError(f"model {service.model_info.name!r} does not propose grasps")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must hold integers")
    return value


def _numbers(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of numbers")
    return tuple(_finite(item, name) for item in value)


__all__ = [
    "BINDING_KEYS",
    "CONTRACT_VERSION",
    "GEOMETRY_BOX",
    "GEOMETRY_MASK",
    "GEOMETRY_POINT",
    "IDENTITY_KEYS",
    "MAX_QUERIES",
    "MODEL_TASKS",
    "TASK_ACTION_CHUNK",
    "TASK_GRASP_PROPOSAL",
    "TASK_GROUNDING",
    "ActionChunkResult",
    "Detection",
    "GraspProposalResult",
    "GroundingResult",
    "InferenceResult",
    "MaskRle",
    "ModelIdentity",
    "ModelInfo",
    "ModelService",
    "ProposedGrasp",
    "act",
    "check_request",
    "ground",
    "parse_box",
    "parse_point",
    "parse_response",
    "propose_grasps",
]
