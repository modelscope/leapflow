# Copyright (c) Alibaba, Inc. and its affiliates.
"""LLM-callable node skills: ``hw_skill_list``, ``hw_skill_run``, ``hw_skill_status``,
``hw_skill_stop``, ``hw_skill_label``.

S2 (the LLM) names *what* to do -- a skill on a workcell, with arguments. The
node's ``SkillRuntime`` owns *how* (S1.5 planning, S1 motion) over LHP
``skill.v1``. The host's part is governance, and it never drives a control loop
across the network:

1. ``skill.plan`` -- the node plans and returns the plan, the digest a run
   permit must bind, and the skill's consent scope. A READ skill answers its
   outcome outright.
2. The host verifies that the plan digest covers exactly what it will show
   (``verify_wire_plan``), so the human approves the content the node executes.
3. The plan is approved as one ``device.skill`` action. A ``session`` consent
   grant covers later plans of the same skill version and workcell under no
   looser speed cap; ``each_run`` binds the grant to this plan alone.
4. ``skill.execute`` -- with a single-use permit bound to the plan when the node
   runs ``permit_required``. The run's lease belongs to this host connection and
   the host renews it while the run lives. A lost host, connection, or renewer
   lets the lease lapse, and the node halts the run (``lease_lost``).

The run's ``intent`` (S2's sub-goal) rides on ``skill.execute`` as its episode's
language label, outside the approval; ``hw_skill_label`` records S2's own
verification of a finished run as the episode's success label.

Result shaping (failure codes, next steps, views) lives in ``skill_results.py``; the
tool definitions in ``skill_tool_defs.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from leapflow.hardware.execution_authorization import approved_authorization
from leapflow.hardware.lhp.authority import WRITE_MODE_OBSERVE_ONLY, WRITE_MODE_PERMIT_REQUIRED
from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.permit import PERMIT_PARAM
from leapflow.hardware.lhp.protocol import CAP_SKILL_V1, LhpRemoteError
from leapflow.hardware.skill_results import (
    FAILURE_AMBIGUOUS_WORKCELL,
    FAILURE_APPROVAL_DENIED,
    FAILURE_EXECUTE_UNCONFIRMED,
    FAILURE_NO_SKILL_NODES,
    FAILURE_PERMIT_ISSUER_UNAVAILABLE,
    FAILURE_PLAN_UNVERIFIED,
    FAILURE_SKILL_UNSUPPORTED,
    FAILURE_STATUS_UNAVAILABLE,
    FAILURE_UNKNOWN_NODE,
    FAILURE_WRITES_DISABLED,
    SkillRefusal,
    as_mapping,
    audit_outcome,
    next_step,
    outcome_result,
    plan_mismatch,
    plan_view,
    refusal_result,
    run_view,
    runs_of,
    share_frame,
    unavailable_entry,
    unreachable,
    workcell_view,
)
from leapflow.hardware.skill_tool_defs import (
    DEFAULT_RUN_TIMEOUT_S,
    MAX_RUN_TIMEOUT_S,
    build_skill_tools,
)
from leapflow.hardware.skills.digest import verify_wire_plan
from leapflow.hardware.skills.episodes import MAX_INTENT_CHARS, MAX_LABEL_NOTE_CHARS
from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS,
    FAILURE_STOPPED,
    FAILURE_UNKNOWN_WORKCELL,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
)
from leapflow.hardware.write_governor import WriteGovernanceRequest, WriteGovernor
from leapflow.security.actions import ActionDescriptor

logger = logging.getLogger(__name__)

DEFAULT_LEASE_S = 2.0
"""Run lease the host asks for; renewed every third of it, so two renewals may fail."""

STATUS_WAIT_S = 2.0
"""One bounded ``skill.status`` wait; under the node's 5 s cap and client timeouts."""
STOP_REQUEST_S = 2.0
"""One bounded ``skill.stop``; past it the node's lease deadman halts the run."""

_MAX_TRACKED_RUNS = 32
_WIRE_ERRORS = (LhpRemoteError, OSError, ValueError)


@runtime_checkable
class SkillNodeClient(Protocol):
    """The LHP client surface the skill service uses (socket or in-process)."""

    async def connect(self) -> Mapping[str, Any]: ...

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any: ...

    async def close(self) -> None: ...


ClientFactory = Callable[[Any], SkillNodeClient]
"""Builds the client for one ``FleetNode``."""


def _socket_client(node: Any) -> SkillNodeClient:
    return SocketLhpClient(node.endpoint, token=node.token)


@dataclass
class _Link:
    """One node's client and its latest hello."""

    key: tuple[str, str, str]
    client: SkillNodeClient
    hello: Mapping[str, Any]

    @property
    def write_mode(self) -> str:
        return str(self.hello.get("write_mode") or "")


@dataclass(frozen=True)
class _Target:
    node_id: str
    workcell: str
    link: _Link

    @property
    def qualified(self) -> str:
        return f"{self.node_id}/{self.workcell}"


@dataclass
class _HostRun:
    """Host bookkeeping for a run this host started."""

    run_id: str
    node_id: str
    control_id: str
    lease_s: float
    renewer: asyncio.Task[None] | None = None
    lease_error: str = ""


class SkillService:
    """Governed access to the skill runtimes of every fleet node."""

    def __init__(
        self,
        fleet_manager: Any,
        *,
        gate: Any = None,
        audit_log: Any = None,
        session_id: str = "",
        client_factory: ClientFactory | None = None,
        lease_s: float = DEFAULT_LEASE_S,
        share_frames: bool = False,
    ) -> None:
        self._fleet = fleet_manager
        self._governor = WriteGovernor(gate=gate, audit_log=audit_log)
        self._audit = audit_log
        self._fallback_session_id = session_id
        self._client_factory = client_factory or _socket_client
        self._lease_s = float(lease_s)
        # The host half of camera-frame consent; a node also needs its workcell's.
        self._share_frames = bool(share_frames)
        self._links: dict[str, _Link] = {}
        self._runs: dict[str, _HostRun] = {}
        self._background: set[asyncio.Task[Any]] = set()

    def set_gate(self, gate: Any) -> None:
        """Late-bind the approval gate (daemon ``install_gate`` path)."""
        self._governor.set_gate(gate)

    async def close(self) -> None:
        """Stop renewing and drop every connection; nodes halt runs whose lease lapses."""
        for tracked in self._runs.values():
            _cancel(tracked.renewer)
        self._runs.clear()
        for task in list(self._background):
            task.cancel()
        links, self._links = self._links, {}
        for link in links.values():
            try:
                await link.client.close()
            except (*_WIRE_ERRORS, RuntimeError):
                logger.debug("skill client close failed", exc_info=True)

    # ── Tools ──

    async def hw_skill_list(self, **kwargs: Any) -> dict[str, Any]:
        """Workcells and skill cards across nodes; ids are qualified ``node/workcell``."""
        wanted = str(kwargs.get("node") or "")
        nodes = [node for node in self._nodes() if not wanted or node.node_id == wanted]
        if wanted and not nodes:
            return refusal_result(FAILURE_UNKNOWN_NODE, f"no fleet node {wanted!r}")
        workcells: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        rejected: list[Any] = []
        for node in nodes:
            try:
                link = await self._link(node)
                listed = as_mapping(await self._request(node.node_id, link, "skill.list", {}))
            except SkillRefusal as refusal:
                unavailable.append(unavailable_entry(node.node_id, refusal))
                continue
            workcells.extend(
                workcell_view(node.node_id, cell)
                for cell in listed.get("workcells") or ()
                if isinstance(cell, Mapping)
            )
            rejected.extend(listed.get("rejected") or ())
        result: dict[str, Any] = {"ok": True, "workcells": workcells}
        if unavailable:
            result["unavailable"] = unavailable
        if rejected:
            result["rejected"] = rejected
        return result

    async def hw_skill_run(self, **kwargs: Any) -> dict[str, Any]:
        """Plan, approve, and run one skill; wait for its outcome unless ``wait`` is false."""
        skill = str(kwargs.get("skill") or "").strip()
        try:
            args = kwargs.get("args") or {}
            if not isinstance(args, Mapping):
                raise SkillRefusal(FAILURE_INVALID_ARGS, "args must be an object")
            if not skill:
                raise SkillRefusal(FAILURE_INVALID_ARGS, "skill is required")
            timeout_s = _timeout(kwargs.get("timeout_s"))
            target = await self._resolve(str(kwargs.get("workcell") or "").strip())
            planned = as_mapping(
                await self._request(
                    target.node_id,
                    target.link,
                    "skill.plan",
                    {
                        "workcell": target.workcell,
                        "skill": skill,
                        "args": dict(args),
                        "share_frames": self._share_frames,
                    },
                )
            )
        except SkillRefusal as refusal:
            return refusal_result(refusal.failure_code, refusal.message, skill=skill)
        if "outcome" in planned:  # a READ skill's result, or a refusal to plan
            result = outcome_result(target.qualified, skill, as_mapping(planned["outcome"]))
            shared = share_frame(result, consent=self._share_frames)
            if shared:
                self._audit_record("skill_frames", target.qualified, skill, f"shared {shared}")
            return result
        plan = as_mapping(planned.get("plan"))
        permit_digest = str(planned.get("permit_digest") or "")
        problem = verify_wire_plan(plan, permit_digest) or plan_mismatch(
            plan, target.workcell, skill
        )
        if problem:
            logger.error("skill plan from %s failed verification: %s", target.node_id, problem)
            return refusal_result(
                FAILURE_PLAN_UNVERIFIED, problem, workcell=target.qualified, skill=skill
            )
        base = {
            "workcell": target.qualified,
            "skill": skill,
            "plan": plan_view(plan, str(planned.get("consent") or "")),
        }
        if bool(kwargs.get("dry_run")):
            return {
                "ok": True,
                "dry_run": True,
                **base,
                "side_effect_state": SIDE_EFFECT_NONE,
                "next_step": "Call hw_skill_run again without dry_run to run it (needs approval).",
            }
        gated = await self._approve(target, plan, base)
        if isinstance(gated, dict):
            return gated
        intent = str(kwargs.get("intent") or "").strip()[:MAX_INTENT_CHARS]
        started = await self._start(target, gated, permit_digest, base, intent)
        if isinstance(started, dict):
            return started
        if not bool(kwargs.get("wait", True)):
            return {
                "ok": True,
                **base,
                "run_id": started.run_id,
                "status": "running",
                "side_effect_state": SIDE_EFFECT_UNKNOWN,
                "next_step": "Follow it with hw_skill_status; stop it with hw_skill_stop.",
            }
        return await self._finish(target, started, _skill_ref(gated), base, timeout_s)

    async def hw_skill_status(self, **kwargs: Any) -> dict[str, Any]:
        """Runs on the node that owns ``run_id``, or every run on every node."""
        run_id = str(kwargs.get("run_id") or "")
        runs: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        for node_id, link in await self._links_for(run_id, unavailable):
            try:
                reply = await self._request(node_id, link, "skill.status", {"run_id": run_id})
            except SkillRefusal as refusal:
                unavailable.append(unavailable_entry(node_id, refusal))
                continue
            runs.extend(self._view(node_id, run) for run in runs_of(reply))
        result: dict[str, Any] = {"ok": True, "runs": runs}
        if unavailable:
            result["unavailable"] = unavailable
        return result

    async def hw_skill_stop(self, **kwargs: Any) -> dict[str, Any]:
        """Stop one run, or every run on every node. Never needs approval.

        Renewal stops first: even when a node cannot be reached, its lease then
        lapses and the node halts the run on its own.
        """
        run_id = str(kwargs.get("run_id") or "")
        reason = str(kwargs.get("reason") or "") or "stopped from the host"
        for tracked in self._runs.values():
            if not run_id or tracked.run_id == run_id:
                _cancel(tracked.renewer)
        params: dict[str, Any] = {"reason": reason}
        if run_id:
            params["run_id"] = run_id
        stopped: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        for node_id, link in await self._links_for(run_id, unavailable):
            try:
                reply = await self._request(node_id, link, "skill.stop", params)
            except SkillRefusal as refusal:
                unavailable.append(unavailable_entry(node_id, refusal))
                continue
            stopped.extend(self._view(node_id, run) for run in runs_of(reply))
        self._audit_record("skill_stop", run_id or "*", "", f"stopped={len(stopped)} {reason}")
        result: dict[str, Any] = {"ok": not unavailable, "stopped": stopped}
        if unavailable:
            result["unavailable"] = unavailable
            result["next_step"] = (
                "Unreachable nodes halt their runs when the lease lapses (seconds). Use "
                "hw_estop on the arm if motion must end now."
            )
        return result

    async def hw_skill_label(self, **kwargs: Any) -> dict[str, Any]:
        """Record S2's verdict on a finished run as its episode's success label."""
        run_id = str(kwargs.get("run_id") or "").strip()
        success = kwargs.get("success")
        if not run_id or not isinstance(success, bool):
            return refusal_result(FAILURE_INVALID_ARGS, "run_id and a boolean success are required")
        params = {
            "run_id": run_id,
            "success": success,
            "note": str(kwargs.get("note") or "")[:MAX_LABEL_NOTE_CHARS],
        }
        unavailable: list[dict[str, Any]] = []
        refused: dict[str, Any] = {}
        for node_id, link in await self._links_for(run_id, unavailable):
            try:
                reply = as_mapping(await self._request(node_id, link, "skill.label", params))
            except SkillRefusal as refusal:
                unavailable.append(unavailable_entry(node_id, refusal))
                continue
            if "label" in reply:
                self._audit_record("skill_label", node_id, run_id, f"success={success}")
                return {"ok": True, "node": node_id, "label": as_mapping(reply["label"])}
            refused = as_mapping(reply.get("outcome"))
        result = refusal_result(
            str(refused.get("failure_code") or FAILURE_INVALID_ARGS),
            str(refused.get("message") or f"no node recorded run {run_id!r}"),
            run_id=run_id,
        )
        if unavailable:
            result["unavailable"] = unavailable
        return result

    # ── Run path ──

    async def _approve(
        self, target: _Target, plan: Mapping[str, Any], base: Mapping[str, Any]
    ) -> ActionDescriptor | dict[str, Any]:
        """The approved ``device.skill`` descriptor, or a refusal result."""
        mode = target.link.write_mode
        if mode == WRITE_MODE_OBSERVE_ONLY:
            return refusal_result(
                FAILURE_WRITES_DISABLED, f"node {target.node_id!r} is observe-only", **base
            )
        if mode == WRITE_MODE_PERMIT_REQUIRED and self._fleet.permit_issuer is None:
            return refusal_result(
                FAILURE_PERMIT_ISSUER_UNAVAILABLE,
                f"node {target.node_id!r} requires signed permits and this host has no "
                "permit issuer key",
                **base,
            )
        descriptor = ActionDescriptor.device_skill(
            node_id=target.node_id,
            workcell=target.workcell,
            skill_id=str(plan.get("skill_id") or ""),
            version=str(plan.get("version") or ""),
            summary=str(plan.get("summary") or ""),
            plan_digest=str(plan.get("digest") or ""),
            consent=str(base["plan"]["consent"]),
            limits={str(k): float(v) for k, v in as_mapping(plan.get("limits")).items()},
        )
        decision = await self._governor.evaluate(
            WriteGovernanceRequest(
                device_id=target.qualified,
                channel_ids=(),
                values=(),
                descriptor=descriptor,
                identity=self._session_identity(),
                reversible=False,
            )
        )
        if decision.allowed:
            return descriptor
        self._audit_record(
            "skill_run", target.qualified, _skill_ref(descriptor), f"denied: {decision.denial}"
        )
        result = refusal_result(FAILURE_APPROVAL_DENIED, decision.denial, **base)
        if decision.no_gate:
            from leapflow.hardware.tools import no_gate_interaction

            result["interaction_request"] = no_gate_interaction(target.qualified)
        return result

    async def _start(
        self,
        target: _Target,
        descriptor: ActionDescriptor,
        permit_digest: str,
        base: Mapping[str, Any],
        intent: str = "",
    ) -> _HostRun | dict[str, Any]:
        """Execute the approved plan and start renewing its lease, or a result."""
        skill_ref = _skill_ref(descriptor)
        params: dict[str, Any] = {
            "plan_digest": str(descriptor.metadata["plan_digest"]),
            "lease_s": self._lease_s,
        }
        if intent:
            params["intent"] = intent
        if target.link.write_mode == WRITE_MODE_PERMIT_REQUIRED:
            authorization = approved_authorization(descriptor, session_id=self._session_identity())
            params[PERMIT_PARAM] = self._fleet.permit_issuer.issue(
                audience=str(target.link.hello.get("node_id") or target.node_id),
                session_id=authorization.session_id,
                workspace=authorization.workspace,
                approval_id=authorization.approval_id,
                digest=permit_digest,
            ).to_dict()
        try:
            started = as_mapping(await target.link.client.request("skill.execute", params))
        except _WIRE_ERRORS as exc:
            self._audit_record("skill_run", target.qualified, skill_ref, f"unconfirmed: {exc}")
            return refusal_result(
                FAILURE_EXECUTE_UNCONFIRMED,
                f"skill.execute was sent but not confirmed: {exc}",
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                **base,
            )
        if "outcome" in started:
            outcome = as_mapping(started["outcome"])
            self._audit_record("skill_run", target.qualified, skill_ref, audit_outcome(outcome))
            return {**outcome_result(target.qualified, str(base["skill"]), outcome), **base}
        tracked = self._track(
            target, as_mapping(started.get("run")), as_mapping(started.get("lease"))
        )
        self._audit_record("skill_run", target.qualified, skill_ref, f"started {tracked.run_id}")
        return tracked

    async def _finish(
        self,
        target: _Target,
        tracked: _HostRun,
        skill_ref: str,
        base: Mapping[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Wait for the run's outcome and shape it."""
        run, timed_out = await self._await_run(tracked, target.link, timeout_s)
        if run is None:
            return refusal_result(
                FAILURE_STATUS_UNAVAILABLE,
                "the run started but its state could not be read; the host asked the node "
                "to stop it and ended lease renewal, so the node's deadman halts it",
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                run_id=tracked.run_id,
                **base,
            )
        result = {**self._view(target.node_id, run), "skill": base["skill"], "plan": base["plan"]}
        if run.get("outcome"):
            self._audit_record(
                "skill_run", target.qualified, skill_ref, audit_outcome(as_mapping(run["outcome"]))
            )
        if timed_out:
            result["timed_out"] = True
            result["next_step"] = (
                f"The host stopped waiting after {timeout_s:g}s and stopped the run. "
                + next_step(FAILURE_STOPPED, str(result.get("side_effect_state") or ""))
            )
        return result

    def _track(self, target: _Target, run: Mapping[str, Any], lease: Mapping[str, Any]) -> _HostRun:
        tracked = _HostRun(
            run_id=str(run.get("run_id") or ""),
            node_id=target.node_id,
            control_id=str(lease.get("control_id") or ""),
            lease_s=float(lease.get("lease_s") or self._lease_s),
        )
        tracked.renewer = asyncio.get_running_loop().create_task(
            self._renew(tracked, target.link.client), name=f"skill-lease-{tracked.run_id[:8]}"
        )
        self._prune()
        self._runs[tracked.run_id] = tracked
        return tracked

    async def _renew(self, tracked: _HostRun, client: SkillNodeClient) -> None:
        """Keep the run's lease alive until the node ends it or renewal fails.

        A failure is recorded, never raised: the node's deadman halts the run.
        """
        interval = max(0.05, tracked.lease_s / 3.0)
        while True:
            await asyncio.sleep(interval)
            try:
                reply = as_mapping(
                    await asyncio.wait_for(
                        client.request("control.renew", {"control_id": tracked.control_id}),
                        timeout=tracked.lease_s,
                    )
                )
            except asyncio.TimeoutError:
                tracked.lease_error = "lease renewal timed out"
            except _WIRE_ERRORS as exc:
                tracked.lease_error = f"lease renewal failed: {exc}"
            else:
                if reply.get("ok"):
                    continue
                if reply.get("failure_code") == "session_halted":
                    return  # the run ended; its lease went with it
                tracked.lease_error = str(reply.get("error") or reply.get("failure_code"))
            logger.warning("skill run %s: %s", tracked.run_id[:8], tracked.lease_error)
            return

    async def _await_run(
        self, tracked: _HostRun, link: _Link, timeout_s: float
    ) -> tuple[dict[str, Any] | None, bool]:
        """Poll until the run ends or *timeout_s* passes (then stop it): ``(run, timed_out)``.

        A run whose status cannot be read is stopped too, and its lease renewal
        ends with it: an unobservable run must fall to the node's deadman, never
        keep moving on a lease the host renews blind.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        run: dict[str, Any] | None = None
        try:
            while (remaining := deadline - loop.time()) > 0:
                run = await self._poll(tracked, link, min(STATUS_WAIT_S, remaining))
                if run is None:
                    await self._stop_run(tracked, link, "the host lost the run's status")
                    return None, False
                if run.get("outcome"):
                    return run, False
        except asyncio.CancelledError:
            # The turn was cancelled: never leave the arm running unattended.
            self._spawn(self._stop_run(tracked, link, "the host turn was cancelled"))
            raise
        await self._stop_run(tracked, link, f"host wait timed out after {timeout_s:g}s")
        return (await self._poll(tracked, link, 0.0)) or run, True

    async def _poll(self, tracked: _HostRun, link: _Link, wait_s: float) -> dict[str, Any] | None:
        try:
            reply = await link.client.request(
                "skill.status", {"run_id": tracked.run_id, "wait_s": wait_s}
            )
        except _WIRE_ERRORS as exc:
            logger.warning("skill run %s: status failed: %s", tracked.run_id[:8], exc)
            return None
        found = runs_of(reply)
        return dict(found[0]) if found else None

    async def _stop_run(self, tracked: _HostRun, link: _Link, reason: str) -> None:
        _cancel(tracked.renewer)
        try:
            await asyncio.wait_for(
                link.client.request("skill.stop", {"run_id": tracked.run_id, "reason": reason}),
                timeout=STOP_REQUEST_S,
            )
        except asyncio.TimeoutError:
            logger.warning("skill run %s: stop timed out (lease will lapse)", tracked.run_id)
        except _WIRE_ERRORS as exc:
            logger.warning("skill run %s: stop failed (lease will lapse): %s", tracked.run_id, exc)

    # ── Resolution ──

    def _nodes(self) -> list[Any]:
        return list(self._fleet.topology().nodes)

    async def _link(self, node: Any) -> _Link:
        """The node's client with a fresh hello; refuses nodes without ``skill.v1``."""
        key = (node.node_id, node.endpoint, node.token)
        link = self._links.get(node.node_id)
        if link is not None and link.key != key:
            await link.client.close()
            link = None
        if link is None:
            link = _Link(key=key, client=self._client_factory(node), hello={})
            self._links[node.node_id] = link
        try:
            link.hello = as_mapping(await link.client.connect())
        except _WIRE_ERRORS as exc:
            raise unreachable(node.node_id, exc) from exc
        if CAP_SKILL_V1 not in tuple(link.hello.get("capabilities") or ()):
            raise SkillRefusal(
                FAILURE_SKILL_UNSUPPORTED, f"node {node.node_id!r} does not offer {CAP_SKILL_V1}"
            )
        return link

    async def _resolve(self, workcell: str) -> _Target:
        """``node/cell`` names its node; a bare ``cell`` must match on exactly one node."""
        node_id, _, cell = workcell.rpartition("/")
        if not cell:
            raise SkillRefusal(FAILURE_INVALID_ARGS, "workcell is required (see hw_skill_list)")
        nodes = self._nodes()
        if not nodes:
            raise SkillRefusal(FAILURE_NO_SKILL_NODES, "no hardware node is registered")
        if node_id:
            nodes = [node for node in nodes if node.node_id == node_id]
            if not nodes:
                raise SkillRefusal(FAILURE_UNKNOWN_NODE, f"no fleet node {node_id!r}")
        if len(nodes) == 1:
            return _Target(nodes[0].node_id, cell, await self._link(nodes[0]))
        matches: list[_Target] = []
        for node in nodes:
            try:
                link = await self._link(node)
                listed = await self._request(node.node_id, link, "skill.list", {"workcell": cell})
            except SkillRefusal:
                continue
            if as_mapping(listed).get("workcells"):
                matches.append(_Target(node.node_id, cell, link))
        if not matches:
            raise SkillRefusal(
                FAILURE_UNKNOWN_WORKCELL, f"no reachable node offers workcell {cell!r}"
            )
        if len(matches) > 1:
            names = ", ".join(match.qualified for match in matches)
            raise SkillRefusal(FAILURE_AMBIGUOUS_WORKCELL, f"workcell {cell!r} is on {names}")
        return matches[0]

    async def _links_for(
        self, run_id: str, unavailable: list[dict[str, Any]]
    ) -> list[tuple[str, _Link]]:
        """The owning node of a tracked *run_id*, otherwise every skill node."""
        tracked = self._runs.get(run_id) if run_id else None
        nodes = self._nodes()
        if tracked is not None:
            nodes = [node for node in nodes if node.node_id == tracked.node_id] or nodes
        links: list[tuple[str, _Link]] = []
        for node in nodes:
            try:
                links.append((node.node_id, await self._link(node)))
            except SkillRefusal as refusal:
                if refusal.failure_code != FAILURE_SKILL_UNSUPPORTED:
                    unavailable.append(unavailable_entry(node.node_id, refusal))
        return links

    @staticmethod
    async def _request(node_id: str, link: _Link, method: str, params: Mapping[str, Any]) -> Any:
        """A request whose failure means nothing moved: raised as a NONE refusal."""
        try:
            return await link.client.request(method, params)
        except _WIRE_ERRORS as exc:
            raise unreachable(node_id, exc) from exc

    # ── Bookkeeping ──

    def _view(self, node_id: str, run: Mapping[str, Any]) -> dict[str, Any]:
        """``run_view`` plus host knowledge; an ended run stops being tracked."""
        view = run_view(node_id, run)
        tracked = self._runs.get(str(run.get("run_id") or ""))
        if tracked is None:
            return view
        if tracked.lease_error:
            view["lease_error"] = tracked.lease_error
        if run.get("outcome"):
            _cancel(tracked.renewer)
            self._runs.pop(tracked.run_id, None)
        return view

    def _prune(self) -> None:
        """Bound bookkeeping: forget runs whose renewal already ended."""
        if len(self._runs) < _MAX_TRACKED_RUNS:
            return
        for run_id, tracked in list(self._runs.items()):
            if tracked.renewer is None or tracked.renewer.done():
                self._runs.pop(run_id, None)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _session_identity(self) -> str:
        from leapflow.tools.execution_context import current_tool_context

        context = current_tool_context()
        session_id = getattr(context, "session_id", "") if context is not None else ""
        return str(session_id or self._fallback_session_id)

    def _audit_record(self, action: str, device: str, channel: str, outcome: str) -> None:
        if self._audit is None:
            return
        self._audit.record(
            action=action,
            device=device,
            channel=channel,
            outcome=outcome,
            identity=self._session_identity(),
        )


def _skill_ref(descriptor: ActionDescriptor) -> str:
    return f"{descriptor.metadata['skill_id']}@{descriptor.metadata['version']}"


def _cancel(task: asyncio.Task[Any] | None) -> None:
    if task is not None and not task.done():
        task.cancel()


def _timeout(raw: Any) -> float:
    if raw is None:
        return DEFAULT_RUN_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SkillRefusal(FAILURE_INVALID_ARGS, "timeout_s must be a number") from exc
    if not 0 < value <= MAX_RUN_TIMEOUT_S:
        raise SkillRefusal(FAILURE_INVALID_ARGS, f"timeout_s must be in (0, {MAX_RUN_TIMEOUT_S:g}]")
    return value


__all__ = [
    "DEFAULT_LEASE_S",
    "DEFAULT_RUN_TIMEOUT_S",
    "MAX_RUN_TIMEOUT_S",
    "SkillNodeClient",
    "SkillService",
    "build_skill_tools",
]
