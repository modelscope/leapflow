# Copyright (c) Alibaba, Inc. and its affiliates.
"""Private, bounded length-prefixed JSON contracts; no executable/path-bearing requests."""

from __future__ import annotations

import asyncio
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.failures import FAILURE_CODES
from leapcalib.contracts.identity import digest_of

VERSION = "leapcalib.worker.v1"
OPERATIONS = frozenset(
    {
        "diagnostics",
        "solve_motion",
        "solve_raw_response",
        "propose_raw",
        "update_raw",
        "track_metric",
        "track_scene",
    }
)
_JSON_ESCAPES = re.compile(r'[\x00-\x1f"\\\x7f-\U0010ffff]')


@dataclass(frozen=True)
class WorkerBudgets:
    request_bytes: int = 16 * 1024 * 1024
    response_bytes: int = 2 * 1024 * 1024
    queued_jobs: int = 2
    max_keyframes: int = 128
    max_tracks: int = 2048
    roi_width: int = 640
    roi_height: int = 480
    startup_s: float = 10.0
    update_s: float = 2.0
    solve_s: float = 30.0
    rss_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        ceilings = {
            "request_bytes": 16 * 1024 * 1024,
            "response_bytes": 2 * 1024 * 1024,
            "queued_jobs": 2,
            "max_keyframes": 128,
            "max_tracks": 2048,
            "roi_width": 640,
            "roi_height": 480,
            "rss_bytes": 512 * 1024 * 1024,
        }
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} exceeds bounded worker contract")
        for name, maximum in (("startup_s", 10), ("update_s", 2), ("solve_s", 30)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"{name} exceeds worker elapsed contract")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON fields")
        result[key] = value
    return result


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _lexical_bound(data: bytes) -> None:
    # Bound nesting and individual JSON array counts BEFORE json.loads allocates arrays.
    stack: list[list[Any]] = []
    quoted = escaped = False
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
            continue
        if byte in (9, 10, 13, 32):
            continue
        if byte == 44:
            if stack and stack[-1][0] == 91:
                stack[-1][2] = False
            continue
        if byte in (93, 125):
            if not stack:
                raise ValueError("invalid JSON nesting")
            stack.pop()
            continue
        if stack and stack[-1][0] == 91 and not stack[-1][2]:
            stack[-1][1] += 1
            stack[-1][2] = True
            if stack[-1][1] > 65536:
                raise ValueError("JSON array element budget exceeded")
        if byte in (91, 123):
            stack.append([byte, 0, False])
            if len(stack) > 32:
                raise ValueError("JSON nesting budget exceeded")
        elif byte == 34:
            quoted = True


def freeze_json(value: Mapping[str, Any], maximum: int) -> dict[str, Any]:
    """Bound and copy JSON containers before suspension, without serializing buffers.

    Count canonical ASCII JSON bytes before allocating the wire representation.
    Primitive strings are immutable and shared; caller-owned lists/dicts are not.
    Only plain JSON types are accepted, so custom iterators cannot run here.
    """
    if type(maximum) is not int or not 0 < maximum <= 16 * 1024 * 1024 or type(value) is not dict:
        raise ValueError("bounded plain JSON object required")
    remaining = maximum

    def charge(count: int) -> None:
        nonlocal remaining
        remaining -= count
        if remaining < 0:
            raise ValueError("worker byte budget exceeded before serialization")

    def copy(item: Any, depth: int) -> Any:
        if depth > 32:
            raise ValueError("JSON nesting budget exceeded")
        if type(item) is str:
            if len(item) + 2 > remaining:
                raise ValueError("worker string byte budget exceeded")
            size = len(item) + 2
            # The common lossless base64 buffers need only a C-level scan, not
            # millions of Python iterations on the hardware event loop.
            for match in _JSON_ESCAPES.finditer(item):
                char = match.group()
                size += (
                    1
                    if char in ('"', "\\", "\b", "\f", "\n", "\r", "\t")
                    else 11
                    if ord(char) > 0xFFFF
                    else 5
                )
                if size > remaining:
                    raise ValueError("worker string byte budget exceeded")
            charge(size)
            return item
        if item is None or type(item) is bool:
            charge(4 if item is None or item is True else 5)
            return item
        if type(item) in (int, float):
            if (
                type(item) is int
                and item.bit_length() > 1024
                or type(item) is float
                and not math.isfinite(item)
            ):
                raise ValueError("bounded finite JSON number required")
            charge(len(str(item)))
            return item
        if type(item) not in (list, tuple, dict) or len(item) > 65536:
            raise ValueError("bounded plain JSON containers required")
        charge(2 + max(0, len(item) - 1))
        if type(item) is dict:
            result = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                copy(key, depth + 1)
                charge(1)
                result[key] = copy(child, depth + 1)
            return result
        return [copy(child, depth + 1) for child in item]

    return copy(value, 1)


def encode(value: Mapping[str, Any], maximum: int) -> bytes:
    frozen = freeze_json(value, maximum)
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if not 0 < len(encoded) <= maximum:
        raise ValueError("worker byte budget exceeded")
    return struct.pack("!I", len(encoded)) + encoded


async def receive(reader: asyncio.StreamReader, maximum: int) -> dict[str, Any]:
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    if not 0 < size <= maximum:
        raise ValueError("worker byte budget exceeded before allocation")
    data = await reader.readexactly(size)
    return await asyncio.to_thread(decode_json, data, maximum)


def decode_json(data: bytes, maximum: int) -> dict[str, Any]:
    """The same bounded strict decoder for private sockets and owner-local evidence."""
    if not isinstance(data, bytes) or not 0 < len(data) <= maximum:
        raise ValueError("worker byte budget exceeded before decoding")
    _lexical_bound(data)
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_object,
        parse_float=_finite,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("worker envelope must be an object")
    return value


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("worker identity must be bounded and explicit")
    return value


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    job_id: str
    workcell_id: str
    run_id: str
    model_digest: str
    operation: str
    deadline_monotonic_s: float
    payload: Mapping[str, Any]
    input_digest: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "job_id",
            "workcell_id",
            "run_id",
            "model_digest",
            "input_digest",
        ):
            _identity(getattr(self, name))
        if self.operation not in OPERATIONS:
            raise ValueError("unknown compute-only operation")
        deadline = self.deadline_monotonic_s
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or deadline <= 0
        ):
            raise ValueError("deadline must be finite node monotonic seconds")
        if not isinstance(self.payload, Mapping) or digest_of(self.payload) != self.input_digest:
            raise ValueError("worker input digest mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": VERSION, **vars(self), "payload": dict(self.payload)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WorkerRequest:
        if set(value) != {"schema", *cls.__dataclass_fields__} or value["schema"] != VERSION:
            raise ValueError("worker request schema/fields mismatch")
        return cls(**{k: v for k, v in value.items() if k != "schema"})

    def response(
        self, *, result: Mapping[str, Any] | None = None, failure: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if (result is None) == (failure is None):
            raise ValueError("exactly one typed result/failure is required")
        return {
            "schema": VERSION,
            "kind": "result" if failure is None else "failure",
            **{
                k: getattr(self, k)
                for k in (
                    "request_id",
                    "job_id",
                    "workcell_id",
                    "run_id",
                    "model_digest",
                    "input_digest",
                )
            },
            "result": result,
            "failure": failure,
        }

    def validate_response(self, response: Mapping[str, Any]) -> dict[str, Any]:
        if set(response) != set(self.response(result={})) or response["schema"] != VERSION:
            raise ValueError("worker response schema mismatch")
        for key in (
            "request_id",
            "job_id",
            "workcell_id",
            "run_id",
            "model_digest",
            "input_digest",
        ):
            if response[key] != getattr(self, key):
                raise ValueError("late or foreign worker response identity")
        kind = response["kind"]
        if (
            kind not in ("result", "failure")
            or not isinstance(response[kind], Mapping)
            or response["failure" if kind == "result" else "result"] is not None
        ):
            raise ValueError("worker result/failure mismatch")
        if kind == "failure":
            failure = response["failure"]
            if (
                set(failure) != {"code", "message", "retryable"}
                or not isinstance(failure["code"], str)
                or failure["code"] not in FAILURE_CODES
                or not isinstance(failure["message"], str)
                or not 0 < len(failure["message"]) <= 512
                or type(failure["retryable"]) is not bool
            ):
                raise ValueError("typed bounded worker failure required")
        digest_of(response)  # Refuse nonfinite nested output, including a compromised worker.
        return dict(response)


def validate_payload(request: WorkerRequest, budgets: WorkerBudgets) -> None:
    expected = {
        "diagnostics": set(),
        "solve_motion": {"evidence", "options"},
        "solve_raw_response": {"evidence", "options"},
        "propose_raw": {"fit", "current", "target", "now"},
        "update_raw": {"fit", "sample"},
        "track_metric": {"first", "second", "tracks", "config"},
        "track_scene": {
            "first",
            "second",
            "task_tracks",
            "background_tracks",
            "config",
            "scene_config",
        },
    }[request.operation]
    fields = set(request.payload)
    # The optional tightening is additive to update_raw v1. Legacy requests keep
    # the original 0.05 bound; a caller cannot widen it or add unrelated fields.
    if request.operation == "update_raw" and fields == expected | {"max_update_norm"}:
        value = request.payload["max_update_norm"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 0.05:
            raise ValueError("recursive update bound must be positive and at most 0.05")
    elif fields != expected:
        raise ValueError("unknown compute payload fields")
    if request.operation in ("track_metric", "track_scene"):
        for key in ("first", "second"):
            roi = request.payload[key]
            if not isinstance(roi, Mapping):
                raise ValueError("typed bounded metric ROI required")
            width, height = roi.get("width"), roi.get("height")
            if (
                type(width) is not int
                or type(height) is not int
                or not 8 <= width <= budgets.roi_width
                or not 8 <= height <= budgets.roi_height
            ):
                raise ValueError("metric ROI exceeds worker dimensions")
            for buffer, multiplier in (("color_bgr8", 3), ("depth_m_f32le", 4)):
                value = roi.get(buffer)
                if not isinstance(value, str) or len(value) != 4 * (
                    (width * height * multiplier + 2) // 3
                ):
                    raise ValueError("metric ROI buffer exceeds declared byte count")
        names = (
            ("tracks",)
            if request.operation == "track_metric"
            else ("task_tracks", "background_tracks")
        )
        groups, config = [request.payload[name] for name in names], request.payload["config"]
        if (
            any(not isinstance(tracks, list) for tracks in groups)
            or sum(len(tracks) for tracks in groups) > budgets.max_tracks
            or not isinstance(config, Mapping)
        ):
            raise ValueError("tracking job exceeds bounded feature budget")
        from leapcalib.contracts.tracking import SceneConfig, TrackingConfig

        settings = TrackingConfig(**config)
        if (
            settings.max_tracks > budgets.max_tracks
            or sum(len(tracks) for tracks in groups) > settings.max_tracks
        ):
            raise ValueError("tracking configuration exceeds worker feature budget")
        if request.operation == "track_scene":
            if any(len(tracks) < settings.min_tracks for tracks in groups) or not isinstance(
                request.payload["scene_config"], Mapping
            ):
                raise ValueError("scene tracking requires two frozen supported track groups")
            SceneConfig(**request.payload["scene_config"])
    evidence = request.payload.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("content"), Mapping):
            raise ValueError("missing bounded evidence content")
        observations = evidence["content"].get("observations")
        if not isinstance(observations, list) or len(observations) > budgets.max_keyframes * 4:
            raise ValueError("worker observation budget exceeded")
        frames = 0
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError("invalid observation")
            if observation.get("kind") == "rgbd":
                frames += 1
                samples = observation.get("samples", [])
                if not isinstance(samples, list) or len(samples) > budgets.max_tracks:
                    raise ValueError("worker track count exceeded")
            pairs = observation.get("correspondences", [])
            if not isinstance(pairs, list) or len(pairs) > budgets.max_tracks:
                raise ValueError("worker correspondence count exceeded")
        if frames > budgets.max_keyframes:
            raise ValueError("worker keyframe budget exceeded")
