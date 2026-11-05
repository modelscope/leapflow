# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP ``skill.v1``: the wire face of the node's ``SkillRuntime``.

Seven methods, all answered with typed payloads rather than error frames for
anything the runtime can explain (a refusal is an outcome with side effect
NONE; only a malformed request is an error frame):

- ``skill.reload``  -- preview an idle workcell replacement, or apply the exact
  declaration under a one-shot permit bound to the old/new digests and owner
  session. No disk writes, device commands, or node restart.
- ``skill.list``    -- workcells and their skill cards, each available or why not.
- ``skill.plan``    -- plan an actuating skill (returns the plan, the digest a
  run permit must bind, and the skill's consent scope), or run a READ skill
  outright (returns its outcome). ``share_frames`` carries the host's consent
  to receive camera frames; without it a skill that shares frames is refused.
- ``skill.execute`` -- start an approved plan under a lease owned by the calling
  connection. Gated by the node's ``WriteAuthority`` against the *plan's*
  content digest, so one permit authorizes exactly one approved run and every
  motion inside it, and nothing else. An optional ``intent`` (S2's sub-goal)
  annotates the run's episode; it is outside the permit and changes nothing.
- ``skill.status``  -- runs, optionally waiting (bounded) for one to end.
- ``skill.stop``    -- stop one run or all of them. A safety action: any
  authenticated connection may stop, as with ``halt``.
- ``skill.label``   -- record whether a finished run's goal was verified reached,
  the success label its episode is trained on. Writes no device.

The run's lease is an ordinary node session (``control_sessions.NodeSessions``):
the host keeps it alive with ``control.renew``; expiry or a dropped connection
stops the run with ``lease_lost`` and halts whatever it moved.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from leapflow.hardware.lhp.authority import WRITE_MODE_PERMIT_REQUIRED, WriteAuthority
from leapflow.hardware.lhp.control_sessions import NodeSessions
from leapflow.hardware.lhp.params import (
    optional_bool,
    optional_float,
    optional_mapping,
    optional_str,
    require_bool,
    require_str,
)
from leapflow.hardware.lhp.permit import request_digest
from leapflow.hardware.skills.workcell_reload import matches
from leapflow.hardware.skills.episodes import MAX_INTENT_CHARS, MAX_LABEL_NOTE_CHARS
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_BUSY,
    FAILURE_LEASE_LOST,
    FAILURE_PLAN_UNKNOWN,
    FAILURE_WRITE_REFUSED,
    SkillError,
    SkillOutcome,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime

MAX_STATUS_WAIT_S = 5.0
"""Longest a ``skill.status`` call may block; stays well under client request timeouts."""

Handler = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]


class SkillRunBody:
    """A skill run as a session body: its lease ends when the run does."""

    kind = "skill"

    def __init__(self, runtime: SkillRuntime, run: SkillRun) -> None:
        self._runtime = runtime
        self._run = run

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self._run.devices

    @property
    def finished(self) -> bool:
        return not self._run.running

    async def close(self, reason: str) -> dict[str, Any]:
        if self._run.running:
            await self._runtime.stop(
                self._run.run_id, f"the host closed the run's lease ({reason})"
            )
        return {"status": "stopped", "reason": reason, "run": self._run.to_dict()}

    async def lose(self, reason: str) -> None:
        await self._runtime.stop(
            self._run.run_id, f"lease lost ({reason})", failure_code=FAILURE_LEASE_LOST
        )

    def describe(self) -> dict[str, Any]:
        plan = self._run.plan
        return {"run_id": self._run.run_id, "skill_id": plan.skill_id, "workcell": plan.workcell}


class SkillMethods:
    """Handlers for the ``skill.*`` methods of one node."""

    def __init__(
        self,
        runtime: SkillRuntime,
        *,
        authority: WriteAuthority,
        sessions: NodeSessions,
        connection_id: Callable[[], str],
    ) -> None:
        self._runtime = runtime
        self._authority = authority
        self._sessions = sessions
        self._connection_id = connection_id

    def handlers(self) -> dict[str, Handler]:
        return {
            "skill.list": self.list,
            "skill.plan": self.plan,
            "skill.execute": self.execute,
            "skill.status": self.status,
            "skill.stop": self.stop,
            "skill.label": self.label,
            "skill.reload": self.reload,
        }

    async def reload(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workcell = require_str(params, "workcell")
        preview = require_bool(params, "preview") if "preview" in params else True
        prepared = self._runtime.prepare_workcell_reload(workcell)
        if isinstance(prepared, SkillOutcome):
            return _outcome(prepared)
        binding = {"workcell": workcell, "preview": False,
                   "previous_digest": prepared.previous_digest,
                   "workcell_digest": prepared.candidate_digest,
                   "node_session_id": prepared.node_session_id}
        if preview:
            return {"reload": {**prepared.to_dict(), "applied": False,
                               "request": binding, "permit_digest": request_digest("skill.reload", binding)}}
        stale = matches(prepared, previous_digest=require_str(params, "previous_digest"),
                        expected_digest=require_str(params, "workcell_digest"),
                        node_session_id=require_str(params, "node_session_id"))
        if stale is not None:
            return _outcome(stale)
        if self._sessions.busy_device(prepared.devices):
            return _outcome(SkillOutcome.failed(FAILURE_DEVICE_BUSY, "reload devices are held by a session"))
        if self._authority.mode != WRITE_MODE_PERMIT_REQUIRED:
            return _outcome(SkillOutcome.failed(FAILURE_WRITE_REFUSED, "live reload requires permit_required authority"))
        refusal = self._authority.check("skill.reload", params)
        if refusal is not None:
            return _outcome(SkillOutcome.failed(refusal.failure_code or FAILURE_WRITE_REFUSED,
                                                refusal.error or "reload authority refused"))
        # No await between preparation, authority and commit: no stale plan can race.
        return {"reload": self._runtime.commit_workcell_reload(prepared)}

    async def list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return self._runtime.list(optional_str(params, "workcell"))

    async def plan(self, params: Mapping[str, Any]) -> dict[str, Any]:
        planned = await self._runtime.plan(
            require_str(params, "workcell"),
            require_str(params, "skill"),
            optional_mapping(params, "args"),
            share_frames=optional_bool(params, "share_frames"),
        )
        if isinstance(planned, SkillOutcome):
            return _outcome(planned)
        return {
            "plan": planned.to_dict(),
            "permit_digest": self._runtime.permit_digest(planned),
            "consent": self._runtime.consent(planned),
        }

    async def execute(self, params: Mapping[str, Any]) -> dict[str, Any]:
        digest = require_str(params, "plan_digest")
        intent = optional_str(params, "intent", max_len=MAX_INTENT_CHARS)
        runtime = self._runtime
        plan = runtime.cached_plan(digest)
        if plan is None:
            return _outcome(
                SkillOutcome.failed(
                    FAILURE_PLAN_UNKNOWN, "no such plan on this node (expired or already executed)"
                )
            )
        # Snapshot I/O must settle before consuming the one-shot authority.
        model_basis = await runtime.execution_basis(plan)
        if isinstance(model_basis, SkillOutcome):
            return _outcome(model_basis)
        if runtime.cached_plan(digest) is not plan:
            return _outcome(
                SkillOutcome.failed(FAILURE_PLAN_UNKNOWN, "plan expired or was executed during admission")
            )
        # Busy before authority: a refusal for a held device must not burn the permit.
        busy = self._sessions.busy_device(runtime.plan_devices(plan))
        if busy:
            return _outcome(
                SkillOutcome.failed(FAILURE_DEVICE_BUSY, f"device {busy!r} is held by a session")
            )
        refusal, admission = self._authority.admit_run(
            params, expected_digest=runtime.permit_digest(plan)
        )
        if refusal is not None:
            code = refusal.failure_code or FAILURE_WRITE_REFUSED
            return _outcome(SkillOutcome.failed(code, refusal.error or code))
        # The prepared execution branch does not suspend before lease adoption.
        started = await runtime.execute(
            digest, intent=intent, admission=admission, model_basis=model_basis
        )
        if isinstance(started, SkillOutcome):
            return _outcome(started)
        session = self._sessions.adopt(
            SkillRunBody(runtime, started),
            lease_s=params.get("lease_s"),
            connection_id=self._connection_id(),
        )
        return {"run": started.to_dict(), "lease": session.lease()}

    async def status(self, params: Mapping[str, Any]) -> dict[str, Any]:
        run_id = optional_str(params, "run_id")
        wait_s = optional_float(params, "wait_s", upper=MAX_STATUS_WAIT_S)
        if run_id and wait_s:
            await self._runtime.wait(run_id, wait_s)
        return {"runs": [run.to_dict() for run in self._runtime.status(run_id)]}

    async def stop(self, params: Mapping[str, Any]) -> dict[str, Any]:
        run_id = optional_str(params, "run_id")
        reason = optional_str(params, "reason") or "stopped by the host"
        if run_id:
            run = await self._runtime.stop(run_id, reason)
            return {"runs": [] if run is None else [run.to_dict()]}
        active = [run for run in self._runtime.status() if run.running]
        await self._runtime.stop_all(reason)
        return {"runs": [run.to_dict() for run in active]}

    async def label(self, params: Mapping[str, Any]) -> dict[str, Any]:
        try:
            label = self._runtime.label(
                require_str(params, "run_id"),
                success=require_bool(params, "success"),
                note=optional_str(params, "note", max_len=MAX_LABEL_NOTE_CHARS),
            )
        except SkillError as error:
            return _outcome(SkillOutcome.from_error(error))
        return {"label": label}


def _outcome(outcome: SkillOutcome) -> dict[str, Any]:
    return {"outcome": outcome.to_dict()}


__all__ = ["MAX_STATUS_WAIT_S", "SkillMethods", "SkillRunBody"]
