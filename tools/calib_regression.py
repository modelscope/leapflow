# Copyright (c) Alibaba, Inc. and its affiliates.
"""Board-neutral candidate/shadow regression and existing-owner metric readiness.

Numerical results are explicitly synthetic, never hardware accuracy certification.
Camera mode invokes a read-only discovered skill; raw metric samples stay node-local.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from dataclasses import replace
from typing import Any

from leapcalib.contracts.identity import (
    DeviceIdentity,
    ModelIdentity,
    StreamProfile,
    TopologyIdentity,
)
from leapcalib.contracts.motion import (
    MODE_FIXED,
    MODE_WRIST,
    MotionObservation,
    PointCorrespondence,
)
from leapcalib.contracts.observations import PROVENANCE_SYNTHETIC, RgbdObservation, Stamp
from leapcalib.contracts.task import TaskDomain, TaskState, TaskTransition
from leapcalib.estimation.handeye import estimate_motion
from leapcalib.estimation.numeric import SolveConfig, numpy, perturbation
from leapcalib.estimation.response import estimate_response, propose_shadow
from leapcalib.geometry.se3 import Transform
from leapcalib.validation.bundle import EvidenceBundle, Retention
from leapcalib.validation.result import result_payload, write_result


def topology(role: str = "fixed") -> TopologyIdentity:
    stream = StreamProfile(
        "synthetic_camera", "depth_aligned", 640, 480, 600, 600, 320, 240, 0.001, role=role
    )
    return TopologyIdentity(
        "synthetic_cell",
        DeviceIdentity("synthetic_arm"),
        ModelIdentity("synthetic_frozen_chain", 6, "synthetic_chain_v1"),
        (stream,),
    )


def motion_study(
    *,
    mode: str = MODE_FIXED,
    seed: int = 41,
    single_axis: bool = False,
    stationary: bool = False,
    outliers: bool = False,
    noise_m: float = 0.0,
) -> tuple[EvidenceBundle, Transform, Transform]:
    """Generate independent source pairs and 20 untouched synthetic holdout motions."""
    np = numpy()
    rng = np.random.default_rng(seed)
    topo = topology("fixed" if mode == MODE_FIXED else "wrist")
    prior = Transform.from_rpy_rad(
        0.1,
        -0.15,
        0.2,
        (0.25, -0.1, 0.4),
        parent="base" if mode == MODE_FIXED else "flange",
        child="camera",
    )
    true = replace(
        prior.compose(perturbation(np.array([0.014, -0.009, 0.018, 0.025, -0.03, 0.02]))),
        parent=prior.parent,
        child=prior.child,
    )
    observations, accepted, heldout = [], [], []
    for i in range(44):
        axis = np.array([0.0, 0.0, 1.0]) if single_axis else rng.normal(size=3)
        angle = 0.0 if stationary else float(rng.uniform(0.08, 0.55))
        robot = Transform.from_axis_angle(
            axis, angle, rng.uniform(-0.03, 0.03, 3) if not stationary else (0, 0, 0)
        )
        camera = true.inverse().compose(robot).compose(true)
        if outliers and i < 4:
            camera = Transform.from_rpy_rad(0.4, -0.3, 0.6, (0.1, -0.1, 0.05))
        mapping = camera if mode == MODE_FIXED else camera.inverse()
        points = rng.uniform((-0.1, -0.1, 0.4), (0.1, 0.1, 0.7), (36, 3))
        pairs = tuple(
            PointCorrespondence(
                str(j), tuple(p), tuple(np.asarray(mapping.apply(p)) + rng.normal(0, noise_m, 3))
            )
            for j, p in enumerate(points)
        )
        first = RgbdObservation(
            f"m{i}:first", topo.streams[0], Stamp(i * 2.0), PROVENANCE_SYNTHETIC
        )
        second = RgbdObservation(
            f"m{i}:second", topo.streams[0], Stamp(i * 2.0 + 1), PROVENANCE_SYNTHETIC
        )
        factor = MotionObservation(
            f"m{i}",
            first.observation_id,
            second.observation_id,
            topo.streams[0],
            topo.model,
            first.stamp,
            second.stamp,
            PROVENANCE_SYNTHETIC,
            mode,
            "synthetic_rigid_scene",
            robot,
            pairs,
            True,
            0.0,
        )
        observations.extend((first, second, factor))
        (accepted if i < 24 else heldout).append(factor.observation_id)
    return (
        EvidenceBundle(
            f"synthetic_motion_{seed}",
            topo,
            tuple(observations),
            0,
            Retention(3600),
            tuple(accepted),
            holdout_ids=tuple(heldout),
            limitations=("synthetic injection, not physical accuracy",),
        ),
        prior,
        true,
    )


def response_study(
    *,
    seed: int = 43,
    columns: tuple[int, ...] = (0, 1, 2, 3),
    stationary: bool = False,
    outliers: bool = False,
) -> tuple[EvidenceBundle, TaskDomain, Any]:
    np = numpy()
    rng = np.random.default_rng(seed)
    topo = topology()
    domain = TaskDomain(
        "synthetic_local",
        "synthetic_feature",
        (-0.1, -0.1, 0.2, -0.8),
        (0.1, 0.1, 0.4, 0.8),
        (0.005, 0.005, 0.005, 0.05),
        (0.005, 0.005, 0.005, 0.05),
        0.025,
        0.13,
        0.0,
        0.0,
        0.5,
    )
    response = np.array(
        [[0.9, 0.08, 0, 0], [-0.05, 1.1, 0, 0], [0, 0, 0.8, 0.02], [0, 0, 0.03, 1.05]]
    )
    observations = []
    for i in range(60):
        start = rng.uniform((-0.07, -0.07, 0.23, -0.5), (0.07, 0.07, 0.37, 0.5))
        delta = np.zeros(4)
        if not stationary:
            delta[list(columns)] = (
                rng.uniform(-1, 1, len(columns)) * np.asarray(domain.max_step_m_rad)[list(columns)]
            )
        feature = start.copy()
        measured = response @ (delta / domain.max_step_m_rad) * domain.tolerance_m_rad
        if outliers and i < 5:
            measured += np.array([0.015, -0.02, 0.02, 0.2])

        def state(name: str, seconds: float, pose: Any, values: Any) -> TaskState:
            return TaskState(
                name,
                Stamp(seconds),
                tuple(pose),
                tuple(values),
                domain.feature_id,
                domain.domain_id,
                True,
                True,
                (0.00001, 0.00001, 0.00001, 0.0001),
            )

        first = state(f"t{i}:first", i * 2.0, start, feature)
        second = state(f"t{i}:second", i * 2.0 + 1, start + delta, feature + measured)
        observations.append(
            TaskTransition(
                f"t{i}", topo.streams[0], topo.model, first, second, PROVENANCE_SYNTHETIC
            )
        )
    return (
        EvidenceBundle(
            f"synthetic_response_{seed}",
            topo,
            tuple(observations),
            0,
            Retention(3600),
            tuple(f"t{i}" for i in range(40)),
            holdout_ids=tuple(f"t{i}" for i in range(40, 60)),
        ),
        domain,
        response,
    )


def numerical_regression() -> dict[str, Any]:
    started = time.perf_counter()
    motion_results = []
    for mode in (MODE_FIXED, MODE_WRIST):
        bundle, prior, true = motion_study(mode=mode, outliers=True, noise_m=0.0001)
        fit = estimate_motion(bundle, prior)
        candidate = Transform.from_mapping(fit.metadata["candidate_transform"])
        translation_error = candidate.distance_m(true)
        rotation_error = candidate.angle_to_rad(true)
        predictions = fit.metadata["holdout_predictions"]
        assert (
            fit.metadata["data_rank"] == 6 and translation_error < 0.001 and rotation_error < 0.005
        )
        assert len(predictions) == 20 and len(fit.accepted_ids) == 20
        motion_results.append(
            {
                "mode": mode,
                "translation_error_m": translation_error,
                "rotation_error_rad": rotation_error,
                "data_rank": 6,
                "holdouts": len(predictions),
                "evidence_digest": bundle.digest,
                "candidate_digest": fit.digest,
            }
        )
    still, prior, _ = motion_study(stationary=True)
    assert not estimate_motion(still, prior).estimates[0].supported_components
    bundle, domain, expected = response_study(outliers=True)
    fit = estimate_response(bundle, domain)
    np = numpy()
    assert np.max(np.abs(np.asarray(fit.estimates[0].values).reshape(4, 4) - expected)) < 1e-8
    current = TaskState(
        "current",
        Stamp(200),
        (0, 0, 0.3, 0),
        (0, 0, 0.3, 0),
        domain.feature_id,
        domain.domain_id,
        True,
        True,
        (0.00001,) * 4,
    )
    proposal = propose_shadow(
        fit,
        current,
        (0.01, -0.008, 0.3, 0.1),
        domain,
        Stamp(200.01),
        topology_digest=bundle.topology.digest,
    )
    assert proposal["accepted"] and proposal["executed"] is False
    return {
        "provenance": "synthetic",
        "hardware_accuracy_certified": False,
        "motion": motion_results,
        "response_rank": fit.metadata["data_rank"],
        "response_holdouts": len(fit.metadata["holdout_predictions"]),
        "shadow": proposal,
        "compute_s": time.perf_counter() - started,
        "config": SolveConfig().to_dict(),
    }


async def camera_regression(
    endpoint: str,
    workcell: str,
    role: str,
    frames: int,
    points: int,
    stamp_uncertainty_s: float,
    token_file: str | None = None,
    *,
    preview_device: str | None = None,
    preview_channel: str | None = None,
) -> dict[str, Any]:
    from leapflow.hardware.lhp.client import SocketLhpClient

    if bool(preview_device) != bool(preview_channel):
        raise ValueError("preview coexistence needs both device and channel")
    token = (await asyncio.to_thread(Path(token_file).read_text)).strip() if token_file else ""
    client = SocketLhpClient(endpoint, timeout_s=180, token=token)
    preview = SocketLhpClient(endpoint, timeout_s=10, token=token) if preview_device else None
    samples: list[dict[str, Any]] = []
    stop = asyncio.Event()
    task = None

    async def preview_sample() -> None:
        assert preview is not None
        before = time.perf_counter()
        frame = await preview.request(
            "read_frame",
            {
                "device_id": preview_device,
                "channel_id": preview_channel,
                "max_width": 320,
                "quality": 50,
                "fps": 4,
                "min_sequence": samples[-1]["sequence"] + 1 if samples else 0,
            },
        )
        # Only an explicitly requested bounded preview crosses the wire; never persist pixels.
        frame.pop("data_b64", None)
        if samples and frame["sequence"] <= samples[-1]["sequence"]:
            raise ValueError("preview did not provide increasing frame sequences")
        samples.append(
            {
                "sequence": frame["sequence"],
                "size_bytes": frame["size_bytes"],
                "latency_s": time.perf_counter() - before,
                "received_s": time.perf_counter(),
            }
        )

    async def poll_preview() -> None:
        # Bounded even if readiness fails; the final sample verifies the retained owner.
        for _ in range(64):
            if stop.is_set():
                break
            await preview_sample()
            await asyncio.sleep(0.05)

    try:
        hello = await client.connect()
        if preview:
            catalog = await client.request("skill.list", {"workcell": workcell})
            cells = catalog.get("workcells", [])
            if not any(c["roles"].get(role) == preview_device for c in cells):
                raise ValueError("preview must use the readiness camera's exact declared device")
            await preview.connect()
            await preview_sample()
            task = asyncio.create_task(poll_preview())
        started = time.perf_counter()
        result = await client.request(
            "skill.plan",
            {
                "workcell": workcell,
                "skill": "calib.metric_readiness",
                "args": {
                    "camera_role": role,
                    "frames": frames,
                    "points": points,
                    "stamp_uncertainty_s": stamp_uncertainty_s,
                    "ttl_s": 3600,
                    "quota_bytes": 16 * 1024 * 1024,
                },
            },
        )
        ended = time.perf_counter()
        stop.set()
        if task:
            await task
            await preview_sample()
        overlap = sum(started <= s["received_s"] <= ended for s in samples)
        return {
            "provenance": "real_camera_readiness",
            "hardware_accuracy_certified": False,
            "node_id": hello["node_id"],
            "result": result,
            "preview_coexistence": {
                "requested": bool(preview),
                "overlap_samples": overlap,
                "verified": bool(overlap),
                "samples": samples,
                "preview_pixels_persisted": False,
            },
        }
    finally:
        stop.set()
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.close()
        if preview:
            await preview.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("numerical", "camera"))
    parser.add_argument("--endpoint")
    parser.add_argument("--workcell")
    parser.add_argument(
        "--token-file", help="existing LHP authentication file; contents are never printed"
    )
    parser.add_argument("--camera-role", choices=("camera", "camera.wrist"), default="camera")
    parser.add_argument("--preview-device", help="opt in to concurrent bounded preview reads")
    parser.add_argument("--preview-channel", help="declared frame channel for the explicit preview")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--stamp-uncertainty-s", type=float, default=0.05)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.mode == "camera" and not (args.endpoint and args.workcell):
        parser.error("camera mode requires --endpoint and --workcell")
    if bool(args.preview_device) != bool(args.preview_channel):
        parser.error("preview coexistence requires --preview-device and --preview-channel")
    result = (
        numerical_regression()
        if args.mode == "numerical"
        else asyncio.run(
            camera_regression(
                args.endpoint,
                args.workcell,
                args.camera_role,
                args.frames,
                args.points,
                args.stamp_uncertainty_s,
                args.token_file,
                preview_device=args.preview_device,
                preview_channel=args.preview_channel,
            )
        )
    )
    payload = result_payload(result)
    if args.output:
        write_result(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if args.mode == "camera":
        success = result["result"].get("outcome", {}).get("status") == "succeeded"
        coexistence = result["preview_coexistence"]
        return 0 if success and (not coexistence["requested"] or coexistence["verified"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
