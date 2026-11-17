# Copyright (c) Alibaba, Inc. and its affiliates.
"""One host planning call, one approval, and exact selected execution over LHP."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from leapflow.hardware.lhp.server import WRITE_MODE_PERMIT_REQUIRED
from leapflow.hardware.skills.planning import PlanningSettings
from tests._harness.skill_stubs import SEQUENCE, WORKCELL_ID, move_by
from tests.test_skill_tools import _Gate, _Tampering, rigs as rigs

pytestmark = pytest.mark.component


def batch() -> dict[str, Any]:
    return {"workcell": WORKCELL_ID, "skill": SEQUENCE,
            "candidates": [{"id": "long", "args": move_by(10)},
                           {"id": "short", "args": move_by(5)},
                           {"id": "bad", "args": move_by(1000)}],
            "selection": {"require_verified": False, "order": ["tcp_distance_mm"]}}


def spy(clients: list[Any], **kwargs: Any) -> Any:
    def wrap(client: Any) -> Any:
        clients.append(_Tampering(client, **kwargs))
        return clients[-1]
    return wrap


async def test_batch_selects_one_program_and_issues_one_permit(tmp_path: Any, rigs: Any, monkeypatch: Any) -> None:
    clients: list[Any] = []
    rig = rigs(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED, wrap=spy(clients))
    issued: list[Any] = []
    issue = type(rig.issuer).issue

    def count(self: Any, **kwargs: Any) -> Any:
        issued.append(kwargs)
        return issue(self, **kwargs)

    monkeypatch.setattr(type(rig.issuer), "issue", count)
    result = await rig.service.hw_skill_run(**batch())
    assert result["ok"], result
    assert result["batch"]["selected_id"] == "short"
    assert result["plan"]["args"]["steps"][0]["relative"]["dz"] == 5
    assert result["plan"]["basis"]["batch"] == result["batch"]
    assert len(rig.gate.seen) == len(issued) == 1
    assert clients[0].calls.count("skill.plan") == clients[0].calls.count("skill.execute") == 1
    assert len([b for b in rig.sim.batches if any(n == "ee.z" for n, _ in b)]) == 1
    assert rig.runtime._plans == {}


async def test_batch_preview_reuse_exact_args_and_consumption_never_replans(tmp_path: Any, rigs: Any) -> None:
    clients: list[Any] = []
    rig = rigs(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED, wrap=spy(clients))
    preview = await rig.service.hw_skill_run(**batch(), dry_run=True)
    assert preview["ok"] and rig.sim.batches == [] and rig.gate.seen == []
    exact = {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": preview["plan"]["args"],
             "approved_plan": preview["plan"]}
    altered = await rig.service.hw_skill_run(**{**exact, "args": move_by(20)})
    assert altered["failure_code"] == "plan_unverified"
    tampered = deepcopy(preview["plan"])
    tampered["summary"] = "tampered"
    assert (await rig.service.hw_skill_run(**{**exact, "approved_plan": tampered}))["failure_code"] == "plan_unverified"
    assert rig.gate.seen == []
    result = await rig.service.hw_skill_run(**exact)
    assert result["ok"], result
    commands = len(rig.sim.batches)
    repeated = await rig.service.hw_skill_run(**exact)
    assert repeated["failure_code"] == "plan_unknown", repeated
    assert len(rig.sim.batches) == commands
    assert clients[0].calls.count("skill.plan") == 1


async def test_expired_batch_preview_and_denied_batch_do_not_fall_back(tmp_path: Any, rigs: Any) -> None:
    clients: list[Any] = []
    rig = rigs(tmp_path, wrap=spy(clients))
    preview = await rig.service.hw_skill_run(**batch(), dry_run=True)
    digest = preview["plan"]["digest"]
    plan, _ = rig.runtime._plans[digest]
    rig.runtime._plans[digest] = (plan, rig.runtime._clock() - 10000)
    result = await rig.service.hw_skill_run(workcell=WORKCELL_ID, skill=SEQUENCE,
                                          args=preview["plan"]["args"], approved_plan=preview["plan"])
    assert result["failure_code"] == "plan_unknown"
    assert clients[0].calls.count("skill.plan") == 1 and rig.sim.batches == []
    rig.gate.approve = False
    refused = await rig.service.hw_skill_run(**batch())
    assert refused["failure_code"] == "approval_denied" and refused["batch"]["selected_id"] == "short"
    assert rig.sim.batches == []


@pytest.mark.parametrize("uncertain", [False, True])
async def test_batch_failure_has_no_candidate_failover(tmp_path: Any, rigs: Any, uncertain: bool) -> None:
    clients: list[Any] = []
    rig = rigs(tmp_path, stall=True, wrap=spy(clients, fail="skill.execute" if uncertain else ()))
    result = await rig.service.hw_skill_run(**batch(), timeout_s=0.2)
    assert not result["ok"] and result["effect_uncertain"], result
    assert result["batch"]["selected_id"] == "short"
    assert len(rig.gate.seen) == 1
    assert clients[0].calls.count("skill.plan") == clients[0].calls.count("skill.execute") == 1
    assert len([b for b in rig.sim.batches if any(n == "ee.z" for n, _ in b)]) <= 1


async def test_old_node_capability_and_unverified_policy_refuse_before_approval(tmp_path: Any, rigs: Any) -> None:
    clients: list[Any] = []
    rig = rigs(tmp_path, wrap=spy(clients))
    refused = await rig.service.hw_skill_run(**{**batch(), "selection": {}})
    assert not refused["ok"] and refused["batch"]["selected_id"] is None
    assert len(refused["batch"]["candidates"]) == 3 and rig.gate.seen == []

    def old_node(method: str, reply: Any) -> Any:
        if method == "skill.list":
            for cell in reply["workcells"]:
                for skill in cell["skills"]:
                    skill.pop("batch_planning", None)
        return reply

    clients[0]._on_reply = old_node
    calls = clients[0].calls.count("skill.plan")
    assert (await rig.service.hw_skill_run(**batch()))["failure_code"] == "skill_unsupported"
    assert clients[0].calls.count("skill.plan") == calls and rig.sim.batches == []


@pytest.mark.parametrize("extra", [{"args": {}}, {"approved_plan": {}}, {"selection": {"order": []}}])
async def test_invalid_batch_envelopes_never_plan(tmp_path: Any, rigs: Any, extra: Any) -> None:
    clients: list[Any] = []
    rig = rigs(tmp_path, wrap=spy(clients), gate=_Gate())
    result = await rig.service.hw_skill_run(**{**batch(), **extra})
    assert result["failure_code"] == "invalid_request"
    assert clients[0].calls.count("skill.plan") == 0 and rig.gate.seen == [] and rig.sim.batches == []


def test_public_schema_discloses_batch_and_exact_preview_without_effect_exemption() -> None:
    from leapflow.hardware.skill_tools import SkillService, build_skill_tools
    from tests.test_skill_tools import _Fleet
    tool = next(t for t in build_skill_tools(SkillService(_Fleet([], None))) if t.name == "hw_skill_run")
    assert {"args", "candidates", "selection", "approved_plan"} <= tool.parameters_schema["properties"].keys()
    assert tool.execution_policy == "external_side_effect" and tool.x_leapflow["requires_approval"]
    assert PlanningSettings().timeout_s == 30
