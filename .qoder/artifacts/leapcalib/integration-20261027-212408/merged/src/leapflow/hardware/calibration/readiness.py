# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded metric readiness using a node-owned capture and cache index, never actuation."""

from __future__ import annotations

import asyncio
import math
import sys
import time
import uuid
from typing import Any, Awaitable, Callable

from leapcalib.contracts.units import canonical_number
from leapcalib.validation.result import result_payload, write_result
from leapflow.cache.manager import CacheScope
from leapflow.hardware.calibration.measurements import frame_evidence
from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_SKILL_UNAVAILABLE,
    SkillError,
)

CATEGORY = "calibration_evidence"


async def metric_readiness(
    capture: Callable[..., Awaitable[RgbdFrame]],
    cache: Any,
    *,
    frames: int,
    points: int,
    role: str,
    stamp_uncertainty_s: float,
    ttl_s: float,
    quota_bytes: int,
) -> dict[str, Any]:
    """Persist sampled metric observations only locally; return descriptive summaries."""
    if type(frames) is not int or not 2 <= frames <= 32:
        raise ValueError("readiness requires 2..32 frames")
    if type(points) is not int or not 16 <= points <= 2048:
        raise ValueError("readiness requires 16..2048 sampled pixels")
    if role not in ("fixed", "wrist"):
        raise ValueError("readiness needs an explicit fixed/wrist camera role")
    stamp_uncertainty_s, ttl_s = canonical_number(stamp_uncertainty_s), canonical_number(ttl_s)
    if not 0 <= stamp_uncertainty_s <= 1 or not 1 <= ttl_s <= 86400:
        raise ValueError("readiness timing/retention exceeds bounded limits")
    if type(quota_bytes) is not int or not 1024 * 1024 <= quota_bytes <= 64 * 1024 * 1024:
        raise ValueError("readiness cache quota must be 1..64 MiB")
    if cache is None:
        raise SkillError(
            FAILURE_SKILL_UNAVAILABLE, "metric readiness requires the node's bound cache index"
        )
    summaries, observations, receipts, compute = [], [], [], []
    started = time.perf_counter()
    cpu_started = time.process_time()
    previous = -1
    profile = None
    for _ in range(frames):
        bound = time.monotonic()
        frame = await asyncio.wait_for(capture(after_monotonic=bound), timeout=5.0)
        if frame.sequence <= previous or not bound <= frame.monotonic_at <= time.monotonic():
            raise SkillError(
                FAILURE_CHECK_FAILED, "capture did not provide a fresh increasing receipt sequence"
            )
        previous = frame.sequence
        before = time.perf_counter()
        record = await asyncio.to_thread(
            frame_evidence, frame, limit=points, role=role, stamp_uncertainty_s=stamp_uncertainty_s
        )
        compute.append(time.perf_counter() - before)
        if profile is not None and record["profile_digest"] != profile:
            raise SkillError(FAILURE_CHECK_FAILED, "live metric profile changed during readiness")
        profile = record["profile_digest"]
        observation = record.pop("observation")
        observations.append(observation)
        receipts.append(frame.monotonic_at)
        record.update(
            {
                "intrinsics": frame.intrinsics.to_dict(),
                "depth_scale_m": frame.depth_scale_m,
                "receipt_age_s": max(0, time.monotonic() - frame.monotonic_at),
                "sdk_frame_number": frame.metadata.get("sdk_frame_number"),
                "sdk_timestamp_ms": frame.metadata.get("sdk_timestamp_ms"),
                "active_profile": frame.metadata.get("active_profile", {}),
                "sampling_compute_s": compute[-1],
            }
        )
        summaries.append(record)
    if not any(s["measured_samples"] for s in summaries):
        raise SkillError(FAILURE_CHECK_FAILED, "metric readiness found no valid measured depth")
    resource_usage: dict[str, Any] = {"peak_rss_bytes": None}
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        resource_usage["peak_rss_bytes"] = int(
            usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        )
    except ImportError:
        pass  # Resource counters are optional; capture remains board-neutral.
    summary = {
        "frames": summaries,
        "frame_count": frames,
        "profile_digest": profile,
        "receipt_intervals_s": [b - a for a, b in zip(receipts, receipts[1:])],
        "compute_s": sum(compute),
        "elapsed_s": time.perf_counter() - started,
        "process_cpu_s": time.process_time() - cpu_started,
        "resource_usage": resource_usage,
        "stamp_semantics": "process_receipt",
        "stamp_uncertainty_s": stamp_uncertainty_s,
        "motion_support": False,
        "hardware_accuracy_certified": False,
        "limitations": [
            "camera readiness only; no observed robot motion or mount observability",
            "factory intrinsics/depth scale retained; receipt is not exposure",
            "sampling grid is not tracked correspondence",
        ],
        "raw_exported": False,
    }
    content = {
        "kind": "metric_camera_readiness",
        "observations": observations,
        "summary": summary,
        "retention": {"ttl_s": ttl_s, "syncable": False, "sensitivity": "raw_evidence"},
    }
    payload = result_payload(content)

    def persist() -> str:
        cache.cleanup_expired()
        directory = cache.path(
            scope=CacheScope.PROFILE, category=CATEGORY, source="metric_readiness"
        )
        path = write_result(directory / f"{uuid.uuid4().hex}.json", payload)
        cache.register(
            path=path,
            scope=CacheScope.PROFILE,
            category=CATEGORY,
            source="metric_readiness",
            content_hash=payload["digest"],
            expires_at=time.time() + ttl_s,
            sensitive=True,
            syncable=False,
            owner_component="hardware.calibration",
        )
        cache.cleanup_quota(
            scope=CacheScope.PROFILE.value, category=CATEGORY, max_bytes=quota_bytes
        )
        if not path.exists():
            raise SkillError(
                FAILURE_CHECK_FAILED, "readiness evidence exceeded its configured cache quota"
            )
        return path.name

    summary["local_artifact"] = await asyncio.to_thread(persist)
    summary["evidence_digest"] = payload["digest"]
    summary["syncable"] = False
    if any(not math.isfinite(v) for v in compute):
        raise SkillError(FAILURE_CHECK_FAILED, "nonfinite sampling timing")
    return summary
