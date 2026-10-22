"""Existing calibration/skill APIs, with offline validation separated from acquisition."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leapflow.hardware.calibration_qualification import admit_package
from leapflow.hardware.fleet import FleetNode, FleetTopology
from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.permit import PermitIssuer
from leapflow.hardware.skill_tools import SkillService
from leapflow.security.approval import ApprovalDecision, ApprovalRequest, SessionAwareGate
from leapflow.security.orchestrator import ApprovalOrchestrator

from .evidence import Evidence, Refusal
from .session import Session


def validate(path: Path, device_id: str) -> dict[str, Any]:
    """Calculate admission from an exported package without changing a store or device."""
    if path.stat().st_size > 1024 * 1024:
        raise Refusal("calibration package exceeds production admission byte budget")
    package = admit_package(json.loads(path.read_text()), device_id=device_id, recorded_at=0.0)
    return {"ok": True, "offline": True, "digest": package.digest,
            "components": [c.component for c in package.certificates], "stored": False}


@dataclass(frozen=True)
class _Fleet:
    node: FleetNode
    permit_issuer: PermitIssuer | None

    def topology(self) -> FleetTopology:
        return FleetTopology(nodes=(self.node,))


class _Consent:
    def __init__(self, plan: dict[str, Any] | None, evidence: Evidence) -> None:
        self.plan, self.evidence = plan, evidence

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        meta = request.action.metadata if request.action else {}
        plan = self.plan
        self.plan = None
        allowed = bool(plan and all(meta.get(key) == plan[key] for key in
                                   ("node_id", "workcell", "skill_id", "plan_digest")))
        self.evidence.record("calibration_consent", approved=allowed)
        return ApprovalDecision.ALLOW_ONCE if allowed else ApprovalDecision.DENY


async def run(session: Session, *, workcell: str = "", skill: str = "", args: dict[str, Any] | None = None,
              approved: dict[str, Any] | None = None) -> dict[str, Any]:
    """Plan or acquire through production SkillService, its governor and leased permits."""
    settings = session.settings
    token = settings.token_file.read_text().strip()
    issuer = PermitIssuer.from_key_file(settings.issuer_key, auto_generate=False) if approved else None
    fleet = _Fleet(FleetNode(settings.node_id, settings.node_id, settings.endpoint, token=token), issuer)
    consent = _Consent(approved, session.evidence)
    gate = ApprovalOrchestrator(SessionAwareGate(consent))

    class ConfirmedClient(SocketLhpClient):
        async def connect(self) -> dict[str, Any]:
            hello = await super().connect()
            if hello.get("node_id") != settings.node_id or hello.get("write_mode") != "permit_required":
                raise Refusal("calibration node identity/authority changed")
            return hello

    service = SkillService(fleet, gate=gate, session_id="piper-calibration",
                           client_factory=lambda node: ConfirmedClient(node.endpoint, token=node.token,
                                                                      timeout_s=settings.request_timeout_s))
    try:
        if not skill:
            return await service.hw_skill_list(node=settings.node_id)
        # Calibration skills only; no policy, contact or zeroing shortcut gets this interface.
        if not skill.startswith("calib."):
            raise Refusal("calibration run requires a production calib.* skill")
        if approved and (approved.get("skill_id") != skill or approved.get("workcell") != workcell or
                         approved.get("args") != args or approved.get("node_id") != settings.node_id):
            raise Refusal("calibration request differs from approved plan")
        result = await service.hw_skill_run(workcell=f"{settings.node_id}/{workcell}", skill=skill,
                                            args=args or {}, dry_run=approved is None,
                                            timeout_s=settings.run_timeout_s)
        session.evidence.record("calibration_result", result=result)
        session.success = True
        return result
    finally:
        await service.close()
