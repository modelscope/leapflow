# Copyright (c) Alibaba, Inc. and its affiliates.
"""Host ``hw_skill_*`` tools against real node servers over simulated workcells.

Covers the governed run path end to end -- plan, host-side plan verification,
approval as one ``device.skill`` action, a plan-bound permit in
``permit_required`` mode, execute, and lease renewal -- plus every refusal
before motion (which must move nothing), the stop paths (explicit stop, host
timeout, cancelled turn, closed service), and the S2-facing result contract.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from leapflow.hardware.fleet import FleetNode
from leapflow.hardware.lhp.client import DirectLhpClient
from leapflow.hardware.lhp.permit import (
    PermitIssuer,
    PermitVerifier,
    generate_keypair,
    save_private_key,
)
from leapflow.hardware.lhp.server import (
    WRITE_MODE_OBSERVE_ONLY,
    WRITE_MODE_PERMIT_REQUIRED,
    WRITE_MODE_TRUSTED_V1,
    LhpNodeServer,
)
from leapflow.hardware.risk import HardwareRiskClassifier
from leapflow.hardware.skill_results import (
    FAILURE_AMBIGUOUS_WORKCELL,
    FAILURE_APPROVAL_DENIED,
    FAILURE_EXECUTE_UNCONFIRMED,
    FAILURE_NODE_UNREACHABLE,
    FAILURE_PERMIT_ISSUER_UNAVAILABLE,
    FAILURE_PLAN_UNVERIFIED,
    FAILURE_STATUS_UNAVAILABLE,
    FAILURE_WRITES_DISABLED,
    next_step,
)
from leapflow.hardware.skill_tools import SkillService, build_skill_tools
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.episodes import EpisodeSettings, latest_labels
from leapflow.hardware.skills.protocol import (
    FAILURE_CALIBRATION_STALE,
    FAILURE_LEASE_LOST,
    FAILURE_STOPPED,
    FAILURE_UNKNOWN_WORKCELL,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    STATUS_SUCCEEDED,
)
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.security.actions import ActionDescriptor, ActionKind
from leapflow.security.risk import RiskLevel
from tests._harness.skill_stubs import (
    SEQUENCE,
    WORKCELL_ID,
    SimArmTransport,
    SkillNodeRegistry,
    move_by,
    write_workcell,
)

NODE_ID = "jetson01"
QUALIFIED = f"{NODE_ID}/{WORKCELL_ID}"
MOVE = {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": move_by(10)}


class _Node:
    """A node server over a simulated, calibrated workcell that trusts *issuer*."""

    def __init__(
        self,
        tmp_path: Path,
        issuer: PermitIssuer,
        *,
        node_id: str = NODE_ID,
        write_mode: str = WRITE_MODE_TRUSTED_V1,
        **sim: Any,
    ) -> None:
        root = tmp_path / node_id
        self.registry = SkillNodeRegistry(SimArmTransport(**sim), store_path=root / "cal.duckdb")
        self.registry.calibrate_cell()
        write_workcell(root / "workcells")
        self.runtime = SkillRuntime(
            self.registry,
            settings=SkillRuntimeSettings(workcells_dir=root / "workcells"),
            skills=(*motion_skills(), *sequence_skills()),
            poll_s=0.005,
        )
        self.server = LhpNodeServer(
            self.registry,
            node_id=node_id,
            write_mode=write_mode,
            permit_verifier=PermitVerifier({issuer.key_id: issuer.public_key}, node_id=node_id),
            skill_runtime=self.runtime,
        )
        self.fleet_node = FleetNode(
            node_id=node_id, display_name=node_id, endpoint=f"tcp://{node_id}:8767"
        )

    @property
    def sim(self) -> SimArmTransport:
        return self.registry.sim


class _Fleet:
    """The FleetManager surface the skill service reads."""

    def __init__(self, nodes: list[_Node], permit_issuer: PermitIssuer | None) -> None:
        self.nodes = nodes
        self._issuer = permit_issuer

    @property
    def permit_issuer(self) -> PermitIssuer | None:
        return self._issuer

    def topology(self) -> SimpleNamespace:
        return SimpleNamespace(nodes=tuple(node.fleet_node for node in self.nodes))


class _Gate:
    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.seen: list[ActionDescriptor] = []

    async def evaluate(self, descriptor: ActionDescriptor) -> SimpleNamespace:
        self.seen.append(descriptor)
        return SimpleNamespace(approved=self.approve, denial_message="the user declined")


class _Audit:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, **fields: Any) -> None:
        self.records.append(fields)


class _Tampering:
    """A client that lets a test rewrite requests or replies, or fail methods."""

    def __init__(
        self,
        inner: DirectLhpClient,
        *,
        on_reply: Any = None,
        on_params: Any = None,
        fail: str | tuple[str, ...] = (),
    ) -> None:
        self._inner = inner
        self._on_reply = on_reply
        self._on_params = on_params
        self._fail = (fail,) if isinstance(fail, str) else tuple(fail)
        self.calls: list[str] = []

    async def connect(self) -> Mapping[str, Any]:
        return await self._inner.connect()

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self.calls.append(method)
        if method in self._fail:
            raise OSError("connection reset by peer")
        if self._on_params is not None:
            params = self._on_params(method, dict(params or {}))
        reply = await self._inner.request(method, params)
        return self._on_reply(method, reply) if self._on_reply is not None else reply

    async def close(self) -> None:
        await self._inner.close()


class _Rig:
    """Nodes, the fleet view, a gate, and the service under test."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        nodes: tuple[str, ...] = (NODE_ID,),
        write_mode: str = WRITE_MODE_TRUSTED_V1,
        host_issuer: bool = True,
        gate: _Gate | None = None,
        no_gate: bool = False,
        lease_s: float = 0.3,
        wrap: Any = None,
        **sim: Any,
    ) -> None:
        self.issuer = PermitIssuer(generate_keypair()[0])
        self.nodes = [
            _Node(tmp_path, self.issuer, node_id=node_id, write_mode=write_mode, **sim)
            for node_id in nodes
        ]
        servers = {node.fleet_node.node_id: node.server for node in self.nodes}
        self.gate = None if no_gate else (gate or _Gate())
        self.audit = _Audit()

        def factory(fleet_node: FleetNode) -> Any:
            client = DirectLhpClient(servers[fleet_node.node_id])
            return wrap(client) if wrap is not None else client

        self.service = SkillService(
            _Fleet(self.nodes, self.issuer if host_issuer else None),
            gate=self.gate,
            audit_log=self.audit,
            session_id="session-1",
            client_factory=factory,
            lease_s=lease_s,
        )

    @property
    def sim(self) -> SimArmTransport:
        return self.nodes[0].sim

    @property
    def runtime(self) -> SkillRuntime:
        return self.nodes[0].runtime


@pytest.fixture
async def rigs() -> Any:
    built: list[_Rig] = []

    def make(tmp_path: Path, **kwargs: Any) -> _Rig:
        rig = _Rig(tmp_path, **kwargs)
        built.append(rig)
        return rig

    yield make
    for rig in built:
        await rig.service.close()
        for node in rig.nodes:
            await node.server.stop()


async def _until(predicate: Any, timeout_s: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.01)


def _run_ended(runtime: SkillRuntime) -> bool:
    runs = runtime.status()
    return bool(runs) and not any(run.running for run in runs)


# ── Catalog ──────────────────────────────────────────────────────────────────


async def test_list_qualifies_workcells_by_node(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path)
    result = await rig.service.hw_skill_list()
    (cell,) = result["workcells"]
    assert result["ok"] and cell["workcell"] == QUALIFIED
    skills = {skill["skill_id"]: skill for skill in cell["skills"]}
    assert skills[SEQUENCE]["available"] is True
    assert skills[SEQUENCE]["args_schema"]


async def test_list_reports_an_unreachable_node_without_failing(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, wrap=lambda client: _Tampering(client, fail="skill.list"))
    result = await rig.service.hw_skill_list()
    assert result["ok"] and result["workcells"] == []
    (entry,) = result["unavailable"]
    assert entry["failure_code"] == FAILURE_NODE_UNREACHABLE


# ── The governed run path ────────────────────────────────────────────────────


async def test_trusted_run_is_approved_as_one_skill_action_and_moves(
    tmp_path: Path, rigs: Any
) -> None:
    rig = rigs(tmp_path)
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["ok"] and result["status"] == STATUS_SUCCEEDED, result
    assert result["workcell"] == QUALIFIED and rig.sim.batches
    (descriptor,) = rig.gate.seen
    assert descriptor.kind == ActionKind.DEVICE_SKILL.value
    assert descriptor.resource.startswith(f"skill:{QUALIFIED}:{SEQUENCE}@")
    assert descriptor.metadata["plan_digest"] == result["plan"]["digest"]
    actions = [record["outcome"] for record in rig.audit.records]
    assert any(text.startswith("started ") for text in actions)


async def test_permit_required_run_carries_a_plan_bound_permit(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED)
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["ok"] and result["status"] == STATUS_SUCCEEDED, result
    assert rig.sim.batches


async def test_permit_required_without_a_host_key_refuses_before_asking(
    tmp_path: Path, rigs: Any
) -> None:
    rig = rigs(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED, host_issuer=False)
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_PERMIT_ISSUER_UNAVAILABLE
    assert "permit-key" in result["next_step"]
    assert rig.gate.seen == [] and rig.sim.batches == []


async def test_observe_only_node_refuses_writes(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, write_mode=WRITE_MODE_OBSERVE_ONLY)
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_WRITES_DISABLED
    assert rig.gate.seen == [] and rig.sim.batches == []


async def test_dry_run_returns_the_plan_and_moves_nothing(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path)
    result = await rig.service.hw_skill_run(**MOVE, dry_run=True)
    assert result["ok"] and result["dry_run"] and result["plan"]["digest"]
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    assert rig.gate.seen == [] and rig.sim.batches == []


async def test_a_denied_approval_moves_nothing(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, gate=_Gate(approve=False))
    result = await rig.service.hw_skill_run(**MOVE)
    assert (result["ok"], result["failure_code"]) == (False, FAILURE_APPROVAL_DENIED)
    assert result["side_effect_state"] == SIDE_EFFECT_NONE and "effect_uncertain" not in result
    assert rig.sim.batches == []


async def test_no_gate_refuses_with_an_interaction_request(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, no_gate=True)
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_APPROVAL_DENIED
    assert result["interaction_request"]["interaction_type"] == "approval_required"
    assert rig.sim.batches == []


async def test_a_plan_that_fails_verification_is_never_shown(tmp_path: Path, rigs: Any) -> None:
    def tamper(method: str, reply: Any) -> Any:
        if method == "skill.plan":
            reply["plan"]["summary"] = "something harmless"
        return reply

    rig = rigs(tmp_path, wrap=lambda client: _Tampering(client, on_reply=tamper))
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_PLAN_UNVERIFIED
    assert rig.gate.seen == [] and rig.sim.batches == []


async def test_a_plan_for_another_skill_is_never_shown(tmp_path: Path, rigs: Any) -> None:
    def swap(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "skill.plan":
            params.update(skill="motion.home", args={})
        return params

    rig = rigs(tmp_path, wrap=lambda client: _Tampering(client, on_params=swap))
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_PLAN_UNVERIFIED
    assert rig.gate.seen == []


async def test_an_unconfirmed_execute_is_uncertain_and_never_replayed(
    tmp_path: Path, rigs: Any
) -> None:
    rig = rigs(tmp_path, wrap=lambda client: _Tampering(client, fail="skill.execute"))
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_EXECUTE_UNCONFIRMED
    assert result["side_effect_state"] == SIDE_EFFECT_UNKNOWN and result["effect_uncertain"]
    assert "Do not run it again" in result["next_step"]


# ── Resolution ───────────────────────────────────────────────────────────────


async def test_unknown_workcell_is_a_typed_refusal(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path)
    result = await rig.service.hw_skill_run(**{**MOVE, "workcell": "nowhere"})
    assert result["failure_code"] == FAILURE_UNKNOWN_WORKCELL and result["next_step"]
    assert rig.sim.batches == []


async def test_a_bare_workcell_on_two_nodes_must_be_qualified(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, nodes=(NODE_ID, "jetson02"))
    ambiguous = await rig.service.hw_skill_run(**MOVE)
    assert ambiguous["failure_code"] == FAILURE_AMBIGUOUS_WORKCELL
    result = await rig.service.hw_skill_run(**{**MOVE, "workcell": f"jetson02/{WORKCELL_ID}"})
    assert result["ok"] and result["workcell"] == f"jetson02/{WORKCELL_ID}"
    assert rig.nodes[1].sim.batches and rig.nodes[0].sim.batches == []


# ── Lease, stop, timeout, cancellation ───────────────────────────────────────


async def test_renewal_keeps_a_run_alive_until_stopped(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, lease_s=0.2, stall=True)
    started = await rig.service.hw_skill_run(**MOVE, wait=False)
    assert started["status"] == "running" and started["side_effect_state"] == SIDE_EFFECT_UNKNOWN
    await asyncio.sleep(0.8)  # four lease periods
    (run,) = (await rig.service.hw_skill_status(run_id=started["run_id"]))["runs"]
    assert run["status"] == "running", run

    stopped = await rig.service.hw_skill_stop(run_id=started["run_id"], reason="test over")
    (run,) = stopped["stopped"]
    assert stopped["ok"] and run["failure_code"] == FAILURE_STOPPED
    assert run["effect_uncertain"] and rig.sim.halts == 1


async def test_closing_the_service_lets_the_lease_lapse(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, lease_s=0.2, stall=True)
    await rig.service.hw_skill_run(**MOVE, wait=False)
    await rig.service.close()
    await _until(lambda: _run_ended(rig.runtime))
    (run,) = rig.runtime.status()
    assert run.outcome is not None and run.outcome.failure_code == FAILURE_LEASE_LOST
    assert rig.sim.halts == 1


async def test_a_host_timeout_stops_the_run(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, stall=True)
    result = await rig.service.hw_skill_run(**MOVE, timeout_s=0.3)
    assert result["timed_out"] and result["failure_code"] == FAILURE_STOPPED
    assert result["effect_uncertain"] and "stopped the run" in result["next_step"]
    assert rig.sim.halts == 1


async def test_a_cancelled_turn_stops_the_run(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path, stall=True)
    task = asyncio.ensure_future(rig.service.hw_skill_run(**MOVE))
    await _until(lambda: rig.sim.batches)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _until(lambda: _run_ended(rig.runtime))
    (run,) = rig.runtime.status()
    assert run.outcome is not None and run.outcome.failure_code == FAILURE_STOPPED


async def test_an_unreadable_status_stops_the_run_and_ends_renewal(
    tmp_path: Path, rigs: Any
) -> None:
    clients: list[_Tampering] = []

    def wrap(client: DirectLhpClient) -> _Tampering:
        clients.append(_Tampering(client, fail="skill.status"))
        return clients[-1]

    rig = rigs(tmp_path, lease_s=0.2, stall=True, wrap=wrap)
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_STATUS_UNAVAILABLE
    assert result["side_effect_state"] == SIDE_EFFECT_UNKNOWN and result["run_id"]
    await _until(lambda: _run_ended(rig.runtime))
    (run,) = rig.runtime.status()
    assert run.outcome is not None and run.outcome.failure_code == FAILURE_STOPPED
    renewals = clients[0].calls.count("control.renew")
    await asyncio.sleep(0.5)  # renewal ended with the stop
    assert clients[0].calls.count("control.renew") == renewals


async def test_an_unobservable_run_falls_to_the_node_deadman(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(
        tmp_path,
        lease_s=0.2,
        stall=True,
        wrap=lambda client: _Tampering(client, fail=("skill.status", "skill.stop")),
    )
    result = await rig.service.hw_skill_run(**MOVE)
    assert result["failure_code"] == FAILURE_STATUS_UNAVAILABLE
    await _until(lambda: _run_ended(rig.runtime))
    (run,) = rig.runtime.status()
    assert run.outcome is not None and run.outcome.failure_code == FAILURE_LEASE_LOST
    assert rig.sim.halts == 1


async def test_invalid_timeout_is_refused(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path)
    result = await rig.service.hw_skill_run(**MOVE, timeout_s=10_000)
    assert result["failure_code"] == "invalid_args" and rig.sim.batches == []


# ── Result contract ──────────────────────────────────────────────────────────


def test_next_step_forbids_replay_only_when_something_may_have_moved() -> None:
    assert "never repeat" not in next_step(FAILURE_APPROVAL_DENIED, SIDE_EFFECT_NONE)
    assert "never repeat" in next_step(FAILURE_STOPPED, SIDE_EFFECT_COMMITTED)
    assert "never repeat" in next_step("an_unmapped_code", SIDE_EFFECT_UNKNOWN)
    assert next_step(FAILURE_CALIBRATION_STALE, SIDE_EFFECT_NONE).startswith("Run the calib.*")
    assert next_step("an_unmapped_code", SIDE_EFFECT_NONE).startswith("Nothing moved")


def test_tool_metadata_gates_run_and_never_gates_stop() -> None:
    tools = {tool.name: tool for tool in build_skill_tools(SkillService(_Fleet([], None)))}
    assert set(tools) == {
        "hw_skill_list",
        "hw_skill_run",
        "hw_skill_status",
        "hw_skill_stop",
        "hw_skill_label",
    }
    run = tools["hw_skill_run"]
    assert run.x_leapflow["requires_approval"] is True
    assert run.execution_policy == "external_side_effect"
    assert tools["hw_skill_stop"].x_leapflow["requires_approval"] is False
    assert tools["hw_skill_status"].x_leapflow["risk_level"] == "read_only"
    assert tools["hw_skill_label"].x_leapflow["requires_approval"] is False


# ── Episode annotations ───────────────────────────────────────────────────────


async def test_intent_and_label_reach_the_node_episode(tmp_path: Path, rigs: Any) -> None:
    rig = rigs(tmp_path)
    episodes = tmp_path / "episodes"
    rig.runtime._settings = SkillRuntimeSettings(  # type: ignore[attr-defined]
        workcells_dir=tmp_path / NODE_ID / "workcells",
        episodes=EpisodeSettings(directory=episodes, min_free_mb=0),
    )
    result = await rig.service.hw_skill_run(**MOVE, intent="clear the fixture")
    assert result["ok"], result
    (entry,) = [json.loads(line) for line in (episodes / "meta" / "episodes.jsonl").open()]
    assert entry["intent"] == "clear the fixture"

    labelled = await rig.service.hw_skill_label(run_id=result["run_id"], success=False)
    assert labelled["ok"] and labelled["label"]["success"] is False, labelled
    assert latest_labels(episodes)[result["run_id"]]["success"] is False
    assert any(record["action"] == "skill_label" for record in rig.audit.records)

    refused = await rig.service.hw_skill_label(run_id=result["run_id"], success="yes")
    assert refused["failure_code"] == "invalid_args"
    unknown = await rig.service.hw_skill_label(run_id="0" * 32, success=True)
    assert unknown["ok"] is False and unknown["failure_code"]


# ── Descriptor and risk ──────────────────────────────────────────────────────


def _skill_descriptor(consent: str = "session", cap: float = 30.0) -> ActionDescriptor:
    return ActionDescriptor.device_skill(
        node_id=NODE_ID,
        workcell=WORKCELL_ID,
        skill_id=SEQUENCE,
        version="1.0.0",
        summary="Move the TCP up 10 mm",
        plan_digest="a" * 64,
        consent=consent,
        limits={"speed_pct_max": cap},
    )


def test_session_grants_cover_one_skill_version_at_no_looser_cap() -> None:
    base = _skill_descriptor()
    assert base.resource == f"skill:{QUALIFIED}:{SEQUENCE}@1.0.0@speed<=30"
    assert _skill_descriptor(cap=50).signature() != base.signature()
    each_run = _skill_descriptor(consent="each_run")
    assert each_run.resource.endswith("#" + "a" * 16)


def test_skill_risk_is_high_and_never_permanent() -> None:
    assessment = HardwareRiskClassifier(registry=None).assess(_skill_descriptor())
    assert assessment.level == RiskLevel.HIGH and assessment.allow_permanent is False
    unbound = ActionDescriptor(
        kind=ActionKind.DEVICE_SKILL.value,
        summary="x",
        detail="x",
        effect="execute",
        resource="skill:x",
    )
    refused = HardwareRiskClassifier(registry=None).assess(unbound)
    assert refused.hardline


# ── Plugin wiring ────────────────────────────────────────────────────────────


def _plugin_registry(key_path: str) -> SimpleNamespace:
    nodes = json.dumps([{"node_id": NODE_ID, "endpoint": "tcp://127.0.0.1:1"}])
    return SimpleNamespace(
        settings=SimpleNamespace(fleet_nodes=nodes, permit_issuer_key_path=key_path)
    )


def test_plugin_loads_the_profile_permit_key_and_keeps_one_skill_service(tmp_path: Path) -> None:
    from leapflow.hardware.plugin import HardwareContextPlugin

    key = tmp_path / "issuer.key"
    save_private_key(generate_keypair()[0], key)
    plugin = HardwareContextPlugin()
    plugin._registry = _plugin_registry(str(key))
    plugin._init_fleet_manager()
    assert plugin._fleet_manager.permit_issuer is not None
    names = {tool.name for tool in plugin._build_skill_tools(None)}
    service = plugin._skill_service
    plugin._build_skill_tools(None)
    assert "hw_skill_run" in names and plugin._skill_service is service


def test_plugin_never_generates_a_permit_key(tmp_path: Path) -> None:
    from leapflow.hardware.plugin import HardwareContextPlugin

    key = tmp_path / "missing.key"
    plugin = HardwareContextPlugin()
    plugin._registry = _plugin_registry(str(key))
    plugin._init_fleet_manager()
    assert plugin._fleet_manager.permit_issuer is None and not key.exists()
