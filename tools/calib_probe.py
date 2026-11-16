# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only calibration prerequisite export from the existing authenticated LHP owner.

No SDK, camera owner, device writes, control sessions, signing keys, or restart.
Wire wall-clock timestamps are diagnostics, never qualified monotonic bounds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from leapflow.hardware.lhp.client import SocketLhpClient


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {name: ordered[round(q * (len(ordered) - 1))]
            for name, q in (("p50", 0.5), ("p99", 0.99), ("max", 1.0))}


async def probe(client: SocketLhpClient, *, node_id: str, arm_id: str,
                workcell: str, samples: int) -> dict[str, Any]:
    if type(samples) is not int or not 3 <= samples <= 300:
        raise ValueError("probe requires 3..300 bounded read samples")
    hello = await client.connect()
    if hello.get("node_id") != node_id:
        raise ValueError("read-only probe node identity mismatch")
    listing = await client.request("list")
    catalog = await client.request("skill.list", {"workcell": workcell})
    cells = catalog.get("workcells", [])
    if len(cells) != 1 or cells[0].get("workcell") != workcell:
        raise ValueError("exact admitted workcell required")
    cell = cells[0]
    wrist = cell["roles"].get("camera.wrist")
    if cell["roles"].get("arm") != arm_id or not wrist or cell["frames"]["camera.wrist"]["parent"] != "flange":
        raise ValueError("explicit wrist/flange and arm bindings required")
    devices = {d["device_id"]: d for d in listing["devices"]}
    arm, camera = devices[arm_id], devices[wrist]
    # Transport dictionaries can hold authentication. Export only needed nonsecret identity.
    arm = {key: value for key, value in arm.items() if key != "transport"}
    camera = {key: value for key, value in camera.items() if key != "transport"}
    metadata = await client.request("read_batch", {"device_id": wrist, "channel_ids": ["camera.metadata"]})
    joint_ids = [f"joint.j{i}.position" for i in range(1, 7)]
    channels = joint_ids + ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz",
                            "arm.enabled", "arm.motion", "arm.ctrl_mode", "arm.status", "arm.speed_pct"]
    ages, coherence, rtts, joints = [], [], [], []
    last: dict[str, Any] = {}
    for _ in range(samples):
        start = time.monotonic()
        batch = await client.request("read_batch", {"device_id": arm_id, "channel_ids": channels})
        rtts.append(time.monotonic() - start)
        readings = batch["readings"]
        values = {r["channel_id"]: r["value"] for r in readings}
        if len(readings) != len(channels) or set(values) != set(channels):
            raise ValueError("incomplete owner read cohort")
        stamps = [r["observed_at"] for r in readings]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in stamps):
            raise ValueError("invalid wire observation stamps")
        ages.append(batch["observed_at"] - min(stamps))
        coherence.append(max(stamps) - min(stamps))
        joints.append([values[c] for c in joint_ids])
        last = values
        await asyncio.sleep(0.05)
    return {"schema": "leapflow.calibration_preflight.v1", "node_id": node_id,
            "probed_at_s": time.time(), "write_mode": hello.get("write_mode"),
            "arm": arm, "camera": camera, "workcell": cell, "camera_metadata": metadata,
            "joint_ids": joint_ids, "last_feedback": last,
            "timing_diagnostics": {"samples": samples, "wire_wall_age_s": quantiles(ages),
                "wire_wall_coherence_s": quantiles(coherence), "client_rtt_s": quantiles(rtts),
                "qualified_monotonic_bounds": False,
                "stationary_joint_spread_deg": [max(c) - min(c) for c in zip(*joints)]},
            "physical_qualification": False, "remote_changes_performed": False}


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    token = args.token_file.expanduser().read_text().strip()
    client = SocketLhpClient(args.endpoint, token=token, timeout_s=10)
    try:
        return await asyncio.wait_for(probe(client, node_id=args.node_id, arm_id=args.arm_id,
                                            workcell=args.workcell, samples=args.samples), timeout=90)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--workcell", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new probe evidence path")
    result = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "read_only": True}))


if __name__ == "__main__":
    main()
