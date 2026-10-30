"""One-shot diagnostic through the existing owner; no motion or deployment."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from leapcalib.validation.result import result_payload, write_result
from leapflow.config import load_config
from leapflow.hardware.fleet_config import configured_nodes_from_settings
from leapflow.hardware.lhp.client import SocketLhpClient


async def check() -> dict:
    nodes = configured_nodes_from_settings(load_config().hardware_fleet_nodes)
    node = next(n for n in nodes if n.node_id == "yahboom")
    client = SocketLhpClient(node.endpoint, token=node.token, timeout_s=90)
    report = {
        "scope": "camera_only_readiness",
        "motion_commanded": False,
        "activation_performed": False,
        "hardware_accuracy_certified": False,
        "raw_exported": False,
        "blockers": [],
    }
    try:
        hello = await client.connect()
        report["node_id"] = hello["node_id"]
        report["write_mode"] = hello["write_mode"]
        if hello["node_id"] != node.node_id or hello["write_mode"] != "permit_required":
            report["blockers"].append("Unexpected owner identity or write mode")
            return report
        report["before"] = {
            "control": await client.request("control.status", {}),
            "skills": await client.request("skill.status", {}),
        }
        if report["before"]["control"]["sessions"] or any(
            r.get("status") == "running" or r.get("outcome") is None
            for r in report["before"]["skills"]["runs"]
        ):
            report["blockers"].append("Owner has an active lease/run; no capture attempted")
            return report
        catalog = await client.request("skill.list", {})
        cells = [
            c for c in catalog["workcells"]
            if c["roles"].get("camera.wrist") == "sensor_realsense_d405"
        ]
        if len(cells) != 1:
            report["blockers"].append("Wrist camera workcell binding is not unique")
            return report
        cell = cells[0]
        report["workcell"] = cell["workcell"]
        report["roles"] = cell["roles"]
        report["calibration_stale"] = cell.get("calibration_stale", {})
        report["acquisition_declared"] = bool(cell.get("calibration_acquisition"))
        report["reference_declared"] = bool(cell.get("calibration_reference"))
        cards = {s["skill_id"]: s for s in cell["skills"]}
        report["calibration_skills"] = {
            k: {field: v.get(field) for field in ("available", "reason", "effect", "leased")}
            for k, v in cards.items() if k.startswith("calib.")
        }
        card = cards.get("calib.metric_readiness", {})
        if not card.get("available") or card.get("effect") != "read" or card.get("leased"):
            report["blockers"].append("Non-leased READ metric readiness is unavailable")
            return report
        report["readiness"] = await client.request(
            "skill.plan",
            {
                "workcell": cell["workcell"],
                "skill": "calib.metric_readiness",
                "args": {
                    "camera_role": "camera.wrist",
                    "frames": 8,
                    "points": 256,
                    "stamp_uncertainty_s": 0.05,
                    "ttl_s": 3600,
                    "quota_bytes": 16 * 1024 * 1024,
                },
            },
        )
        report["after"] = {
            "control": await client.request("control.status", {}),
            "skills": await client.request("skill.status", {}),
        }
        if report["readiness"].get("outcome", {}).get("status") != "succeeded":
            report["blockers"].append("Live wrist metric readiness did not succeed")
        report["limitations"] = [
            "Receipt uncertainty is a diagnostic input, not measured exposure uncertainty",
            "No multi-pose evidence, transform candidate, or heldout calibration result",
            "Control status does not certify absence of uncertain device state",
        ]
        return report
    finally:
        await client.close()


if __name__ == "__main__":
    output = Path(__file__).with_name("camera_only_readiness_result.json")
    payload = result_payload(asyncio.run(asyncio.wait_for(check(), timeout=120)))
    write_result(output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
