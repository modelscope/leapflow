# Copyright (c) Alibaba, Inc. and its affiliates.
"""Actual LHP reload authority and unchanged owner-session tests; no physical claims."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from leapflow.hardware.lhp.permit import request_digest
from leapflow.hardware.skills.protocol import SkillError, SkillOutcome
from leapflow.hardware.skills.workcell_reload import workcell_digest
from tests import test_leapcalib_collection as collection_tests
from tests._harness.skill_stubs import ARM_ID, WORKCELL_ID, write_workcell
from tests.test_leapcalib_collection import planned, request

pytestmark = pytest.mark.component
owner = collection_tests.owner


def proof(issuer, binding):
    return issuer.issue(audience="node", session_id="software-session", workspace="software-workspace",
                        approval_id="explicit-reload-fixture", digest=request_digest("skill.reload", binding)).to_dict()


async def approved_reload(server, issuer):
    preview = await request(server, "skill.reload", workcell=WORKCELL_ID)
    assert "reload" in preview, preview
    binding = preview["reload"]["request"]
    applied = await request(server, "skill.reload", **binding, permit=proof(issuer, binding))
    assert applied.get("reload", {}).get("applied") is True, applied
    return applied["reload"]


def change_disk(tmp_path: Path, **changes):
    # The fixture writer owns the filename; read the only declaration, not a guessed alias.
    path = next((tmp_path / "workcells").glob("*.yaml"))
    declaration = yaml.safe_load(path.read_text())
    declaration.update(changes)
    write_workcell(tmp_path / "workcells", declaration)
    return path


async def test_reload_preview_permit_commit_and_cache_invalidation(owner, tmp_path):
    registry, _, _, runtime, server, issuer = owner
    old = runtime._workcells[WORKCELL_ID]
    session = runtime._node_session_id
    plan = await planned(server)
    change_disk(tmp_path, share_frames=True)
    preview = (await request(server, "skill.reload", workcell=WORKCELL_ID))["reload"]
    binding = preview["request"]
    assert preview["applied"] is False and runtime._workcells[WORKCELL_ID] is old
    assert preview["permit_digest"] == request_digest("skill.reload", binding)
    refused = await request(server, "skill.reload", **binding)
    assert refused["outcome"]["failure_code"] == "permit_required"
    assert runtime.cached_plan(plan["plan"]["digest"]) is not None
    signed = proof(issuer, binding)
    applied = await request(server, "skill.reload", **binding, permit=signed)
    assert applied["reload"]["invalidated_plans"] == 1
    assert runtime.cached_plan(plan["plan"]["digest"]) is None
    assert runtime._node_session_id == session and applied["reload"]["node_session_unchanged"]
    assert runtime._workcells[WORKCELL_ID].share_frames
    assert (await registry.transport(ARM_ID)).write_log == ()
    # Same-declaration requests still consume exactly one nonce.
    no_op = (await request(server, "skill.reload", workcell=WORKCELL_ID))["reload"]["request"]
    once = proof(issuer, no_op)
    assert "reload" in await request(server, "skill.reload", **no_op, permit=once)
    replay = await request(server, "skill.reload", **no_op, permit=once)
    assert replay["outcome"]["failure_code"] == "permit_replayed"
    with pytest.raises(SkillError, match="skill.reload permit"):
        runtime.reload_workcells()


@pytest.mark.parametrize("field", ["workcell_digest", "previous_digest", "node_session_id"])
async def test_changed_binding_refused_without_burning_permit(owner, tmp_path, field):
    _, _, _, runtime, server, issuer = owner
    binding = (await request(server, "skill.reload", workcell=WORKCELL_ID))["reload"]["request"]
    signed = proof(issuer, binding)
    stale = await request(server, "skill.reload", **{**binding, field: "b" * 64}, permit=signed)
    assert stale["outcome"]["failure_code"] == "plan_stale"
    assert "reload" in await request(server, "skill.reload", **binding, permit=signed)
    assert runtime._node_session_id == binding["node_session_id"]


async def test_reload_rejects_malformed_duplicate_missing_and_busy(owner, tmp_path, monkeypatch):
    _, _, _, runtime, server, _ = owner
    before = workcell_digest(runtime._workcells[WORKCELL_ID])
    path = next((tmp_path / "workcells").glob("*.yaml"))
    original = path.read_text()
    path.write_text("workcell: [broken\n")
    assert "outcome" in await request(server, "skill.reload", workcell=WORKCELL_ID)
    assert workcell_digest(runtime._workcells[WORKCELL_ID]) == before
    path.write_text(original)
    duplicate = path.with_name("duplicate.yaml")
    duplicate.write_text(original)
    assert "outcome" in await request(server, "skill.reload", workcell=WORKCELL_ID)
    # Invalid declaration remains in place; no partial admission is applied.
    duplicate.write_text("workcell: missing\nroles: {}\n")
    monkeypatch.setitem(runtime._held, ARM_ID, "unsettled")
    assert "outcome" in await request(server, "skill.reload", workcell=WORKCELL_ID)
    assert workcell_digest(runtime._workcells[WORKCELL_ID]) == before


async def test_reload_busy_gate_preserves_nonce(owner, monkeypatch):
    _, _, _, runtime, server, issuer = owner
    binding = (await request(server, "skill.reload", workcell=WORKCELL_ID))["reload"]["request"]
    signed = proof(issuer, binding)
    with monkeypatch.context() as patch:
        patch.setattr(server.sessions, "busy_device", lambda devices: ARM_ID)
        refused = await request(server, "skill.reload", **binding, permit=signed)
        assert refused["outcome"]["failure_code"] == "device_busy"
    assert "reload" in await request(server, "skill.reload", **binding, permit=signed)
    assert not runtime._held


async def test_inflight_plan_cannot_seal_after_reload(owner, tmp_path, monkeypatch):
    _, _, _, runtime, server, issuer = owner
    entered, release = asyncio.Event(), asyncio.Event()
    skill = runtime._skills["calib.collect"]
    original = type(skill).plan

    async def delayed(self, ctx, args):
        entered.set()
        await release.wait()
        return await original(self, ctx, args)

    monkeypatch.setattr(type(skill), "plan", delayed)
    pending = asyncio.create_task(runtime.plan(WORKCELL_ID, "calib.collect", {"region": [2, 1, 8, 8]}))
    await asyncio.wait_for(entered.wait(), timeout=2)
    change_disk(tmp_path, share_frames=True)
    await approved_reload(server, issuer)
    release.set()
    outcome = await pending
    assert isinstance(outcome, SkillOutcome) and outcome.failure_code == "plan_stale"
    assert not runtime._plans


async def test_operations_reload_uses_governor_and_refuses_stale_preview(owner, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from piper_ops import calibration, cli
    from piper_ops.config import Settings
    from piper_ops.evidence import Evidence, Refusal

    _, _, _, _, server, issuer = owner
    calls = []

    async def wire(method, params):
        calls.append(method)
        return await request(server, method, **params)

    settings = Settings.load(node_id="node", root=tmp_path)
    session = SimpleNamespace(client=SimpleNamespace(request=wire), settings=settings,
                              evidence=Evidence(tmp_path / "reload.json"), success=False)
    monkeypatch.setattr(calibration.PermitIssuer, "from_key_file", lambda *a, **kw: issuer)
    preview = (await calibration.reload_workcell(session, workcell=WORKCELL_ID))["reload"]
    assert calls == ["skill.reload"]
    approved = {"workcell": WORKCELL_ID, "node_id": "node", "skill_id": "skill.reload",
                "reload_request": preview["request"], "plan_digest": preview["permit_digest"]}
    result = await calibration.reload_workcell(session, workcell=WORKCELL_ID, approved=approved)
    assert result["ok"] and result["reload"]["applied"]
    assert any(event["event"] == "calibration_consent" and event["approved"] for event in session.evidence.events)
    with pytest.raises(Refusal, match="changed"):
        await calibration.reload_workcell(session, workcell=WORKCELL_ID, approved={**approved, "plan_digest": "wrong"})
    parsed = cli.parser().parse_args(["calibration", "reload", "--workcell", WORKCELL_ID])
    assert parsed.execute is False
    with pytest.raises(Refusal):
        cli.approved(parsed, settings, "reload")
