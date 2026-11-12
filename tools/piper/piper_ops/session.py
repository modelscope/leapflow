"""Persistent-owner connection and exact-step production approval composition.

One invariant governs how a session ends: *a session that commanded the arm may
only undo what it did*. Concretely, an exit never leaves the arm worse than the
run left it -- a gravity-loaded arm is either still holding a verified posture or
the evidence says otherwise and why.

That invariant exists because the only stop this arm offers is a damped emergency
stop, and on a Piper X that stop releases holding torque. Measured on this cell:
a calibration run settled at ``z=249.62 mm``, the session then stopped it because
the *verification* had failed, and two minutes later the arm was at
``z=72.556 mm`` with the gripper resting on the table -- 177 mm of uncommanded
descent that also put the tool below the workspace floor, which then refuses
every later plan. So a failed verdict is not a motion hazard: every write settled
and the arm is standing still. A stop is owed only when this run cannot say what
it did to the arm -- an effect that did not land cleanly, an unknown or halted
run, a device fault, a cancellation (``Session._stop_owed``).

Even an owed stop is sent only under ``Settings.failure_policy == "stop"``. The
default ``hold`` policy never stops automatically: the controller keeps its last
commanded target (every waypoint is at most one bounded segment away), the exit
waits for the arm to come to rest and reports the withheld stop, and only the
operator decides on a stop. A released arm is less controllable than a held one.
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Coroutine, Mapping, TypeVar

from leapflow.engine.recovery.failure_envelope import SideEffectState
from leapflow.hardware.arm_status import ARM_STATUS_EMERGENCY_STOP, ARM_STATUS_NORMAL
from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.context import HardwareContext, TransportRef
from leapflow.hardware.drivers._piper_motion import CTRL_MODE_CAN, CTRL_MODE_TEACH, ctrl_mode_name
from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.permit import PermitIssuer
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT,
    FAILURE_INTERNAL_DEFECT,
    SIDE_EFFECT_UNKNOWN,
    STATUS_HALTED,
)
from leapflow.hardware.attestation import declaration_digest
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.risk import build_risk_classifier
from leapflow.hardware.tools import HardwareTools
from leapflow.security.actions import ActionDescriptor
from leapflow.security.approval import ApprovalDecision, ApprovalRequest, SessionAwareGate
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context

from .config import Settings
from .evidence import Evidence, Refusal, StaleFeedback
from .health import Snapshot, read_snapshot


def unverified_digest(context: HardwareContext) -> str:
    return declaration_digest(replace(context, provenance=replace(
        context.provenance, verified_by="", verified_at=0.0)))


class ConfirmedProvider:
    kind = "piper_operations_remote"

    def __init__(self, context: HardwareContext, config: dict[str, Any]) -> None:
        self.context = replace(context, transport=TransportRef(kind="lhp", config=config))

    def discover(self) -> tuple[HardwareContext, ...]:
        return (self.context,)


class ScopedConsent:
    """Consume only the approved current ordered channel/value request."""

    def __init__(self, context: HardwareContext, evidence: Evidence, approved: bool) -> None:
        self.context, self.evidence, self.approved = context, evidence, approved
        self.expected: tuple[tuple[str, Any], ...] = ()

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        action = request.action
        meta = action.metadata if action else {}
        matches = False
        if action and meta.get("device_id") == self.context.device_id and self.expected:
            if meta.get("batch"):
                raw = meta.get("commands", [])
                if isinstance(raw, list) and all(isinstance(c, dict) and
                        {"channel_id", "value"} <= c.keys() for c in raw):
                    actual = tuple((c["channel_id"], c["value"]) for c in raw)
                    matches = actual == self.expected
            elif len(self.expected) == 1:
                channel_id, value = self.expected[0]
                channel = self.context.channel(channel_id)
                if channel is not None:
                    expected = ActionDescriptor.device(kind=action.kind, device_id=self.context.device_id,
                        channel_id=channel_id, value=value, unit=channel.unit, location=self.context.location)
                    matches = meta.get("channel_id") == channel_id and action.detail == expected.detail
        allow = self.approved and matches
        self.evidence.record("operator_consent", approved=allow,
                             action_id=action.action_id if action else "")
        self.expected = ()
        return ApprovalDecision.ALLOW_ONCE if allow else ApprovalDecision.DENY


T = TypeVar("T")


async def complete_cleanup(operation: Coroutine[Any, Any, T]) -> T:
    """Finish bounded cleanup before propagating even repeated cancellation."""
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


POSTURE_HELD = "held"
POSTURE_UNHELD = "unheld"
POSTURE_UNVERIFIED = "unverified"

STOP_FAILURE_CODES = frozenset({FAILURE_DEVICE_FAULT, FAILURE_INTERNAL_DEFECT})
"""Failures that mean the arm, not the goal, went wrong: a stop is owed for these."""


def outcome_requires_stop(result: Mapping[str, Any]) -> bool:
    """Whether a finished operation's own verdict calls for a stop.

    Mirrors the node's own rule (``skills.runs.needs_halt``): only a run that could
    not finish, an unknown effect, or a device fault. A calibration that measured
    out of tolerance is a *result* -- its writes settled and the arm is standing
    still -- so stopping it would only release the posture it was holding.
    """
    if str(result.get("side_effect_state") or SIDE_EFFECT_UNKNOWN) == SIDE_EFFECT_UNKNOWN:
        return True
    if str(result.get("status") or "") == STATUS_HALTED:
        return True
    return result.get("failure_code") in STOP_FAILURE_CODES


@dataclass(frozen=True)
class Posture:
    """Whether the arm holds the posture it is in, from qualified samples.

    ``held`` needs all four at once: every drive enabled, CAN control selected, a
    normal status word, and an unchanged joint vector between samples. Enable flags
    keep reading true across an emergency stop and the status word recovers, so the
    unchanged joint vector is the only part that cannot be reported while the arm
    is drooping.
    """

    state: str
    detail: str
    joints: tuple[float, ...] = ()
    moved_deg: float = 0.0
    ctrl_mode: int = 0
    enabled: bool = False
    arm_status: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "detail": self.detail, "joints": list(self.joints),
                "moved_deg": self.moved_deg, "ctrl_mode": self.ctrl_mode,
                "enabled": self.enabled, "arm_status": self.arm_status}


class Session:
    """One operational run; never creates or closes the node's SDK link."""

    def __init__(self, settings: Settings, evidence: Evidence, *, execute: bool = False,
                 digest: str = "", approval_note: str = "") -> None:
        self.settings, self.evidence, self.execute = settings, evidence, execute
        self.digest, self.approval_note = digest, approval_note
        self.client: SocketLhpClient | None = None
        self.registry: HardwareRegistry | None = None
        self.tools: HardwareTools | None = None
        self.touched = False
        self.success = False
        self.stop_required = False
        """The operation's own verdict demands a halt (unknown effect, halted run, fault)."""
        self.leave_unheld = False
        """The operator asked for a released arm (``arm stop``/``arm disable``)."""
        self._halt_delivered = False
        self.context: HardwareContext | None = None
        self.listing: dict[str, Any] = {}
        self._context_token = None
        self.consent: ScopedConsent | None = None
        self._approved_steps: list[tuple[str, tuple[tuple[str, Any], ...], bool]] = []

    def bind_steps(self, steps: list[dict[str, Any]]) -> None:
        """Bind the complete preview once; execution cannot invent a subsequent step."""
        if self.touched or self._approved_steps:
            raise Refusal("steps are already bound or execution has begun")
        self._approved_steps = [(r["label"], tuple((c, v) for c, v in r["commands"]),
                                 r["configure"]) for r in steps if r.get("dwell_s") is None]

    async def __aenter__(self) -> Session:
        try:
            await self._open()
        except BaseException:
            try:
                await self._close()
            except (Exception, asyncio.CancelledError) as error:
                self._record_cleanup_error(error)
            raise
        return self

    async def _open(self) -> None:
        s = self.settings
        try:
            token = s.token_file.read_text().strip()
        except OSError as exc:
            raise Refusal("existing node token is unavailable; supply --token-file") from exc
        if not token:
            raise Refusal("token file is empty")
        self.client = SocketLhpClient(s.endpoint, token=token, timeout_s=s.request_timeout_s)
        hello = await self.client.connect()
        if hello.get("node_id") != s.node_id or hello.get("write_mode") != "permit_required":
            raise Refusal("unexpected node identity or write authority mode")
        self.listing = await self.client.request("list")
        self.context = self._declaration(self.listing)
        config = {"endpoint": s.endpoint, "token": token, "node_id": s.node_id,
                  "remote_device_id": s.device_id, "timeout_s": s.request_timeout_s}
        if self.execute:
            if not self.digest or not self.approval_note:
                raise Refusal("execute requires declaration digest and recorded operator consent")
            if unverified_digest(self.context) != self.digest or not self.context.provenance.is_verified:
                raise Refusal("declaration is not the exact operator-confirmed version")
            try:
                config["permit_issuer"] = PermitIssuer.from_key_file(s.issuer_key, auto_generate=False)
            except (OSError, ValueError) as exc:
                raise Refusal("trusted signing authority unavailable; execute from the authorized host; "
                              "do not copy a private issuer key to the board") from exc
        self.registry = HardwareRegistry(HardwareSettings(enabled=True, persist_readings=False,
            stream_enabled=False), providers=[ConfirmedProvider(self.context, config)])
        self.registry.load()
        self.consent = ScopedConsent(self.context, self.evidence, self.execute)
        gate = ApprovalOrchestrator(SessionAwareGate(self.consent),
                                    risk_classifier=build_risk_classifier(self.registry))
        self.tools = HardwareTools(self.registry, gate=gate,
                                  audit_log=HardwareAuditLog(self.evidence.audit_path))
        self._context_token = set_tool_context(ToolExecutionContext.from_strings(
            workspace_root=str(s.root), session_id="piper-ops-" + uuid.uuid4().hex))
        described = await self.tools.hw_describe(device_id=s.device_id)
        if described.get("ok") is not True:
            raise Refusal("persistent owner binding failed; inspect node status")
        self.evidence.record("connected", node_id=s.node_id, write_mode=hello["write_mode"],
                             declaration_digest=unverified_digest(self.context), execute=self.execute)

    def _declaration(self, listing: dict[str, Any]) -> HardwareContext:
        if not isinstance(listing, dict) or listing.get("node_id") != self.settings.node_id:
            raise Refusal("node listing identity changed")
        devices = listing.get("devices")
        if not isinstance(devices, list) or any(not isinstance(c, dict) for c in devices):
            raise Refusal("node declarations are malformed")
        raw = [c for c in devices if c.get("device_id") == self.settings.device_id]
        if len(raw) != 1:
            raise Refusal("missing/duplicate device declaration")
        context = HardwareContext.from_mapping(raw[0])
        if context.model != "Piper X":
            raise Refusal("declared device is not Piper X")
        return context

    async def snapshot(self, *, ready: bool = False, preflight: bool = False,
                       requalify: bool = False) -> Snapshot:
        """Read one qualified snapshot; ``requalify`` re-samples only a receive gap.

        A stale CAN sample describes the reception, not the arm, so an observation may
        read again inside the configured request budget while the servo keeps following
        its waypoint; a reception that never requalifies stops the run instead of
        spinning. Every other refusal — fault, thermal, enable, mode — still aborts at
        once, and without ``requalify`` the sample fails closed exactly as before.
        """
        assert self.registry is not None
        deadline, stale = time.monotonic() + self.settings.request_timeout_s, 0
        while True:
            try:
                current = await read_snapshot(self.registry, self.settings,
                                             ready=ready, preflight=preflight)
            except StaleFeedback as exc:
                if not requalify:
                    raise
                if time.monotonic() >= deadline:
                    raise Refusal("controller feedback did not requalify; halt and inspect",
                                  state=SideEffectState.UNKNOWN) from exc
                stale += 1
                if stale == 1:
                    self.evidence.record("stale_feedback", detail=str(exc))
                await asyncio.sleep(self.settings.poll_s)
                continue
            if stale:
                self.evidence.record("feedback_requalified", stale_samples=stale)
            return current

    async def command(self, label: str, commands: tuple[tuple[str, Any], ...], *, configure: bool = False) -> None:
        """Send once through HardwareTools; a failed/uncertain result is never replayed."""
        assert self.tools and self.client and self.consent
        current = self._declaration(await self.client.request("list"))
        if self.execute and (unverified_digest(current) != self.digest or not current.provenance.is_verified):
            raise Refusal("declaration changed during execution")
        await self.snapshot(ready=not configure, preflight=True, requalify=True)
        if self.execute:
            if not self._approved_steps or self._approved_steps[0] != (label, commands, configure):
                raise Refusal("command differs from the complete approved preview")
            self._approved_steps.pop(0)  # consume before dispatch, never replay an uncertain send
        self.consent.expected = commands
        if self.execute:
            self.evidence.record("command_attempt", label=label, commands=commands, configure=configure)
            self.touched = True  # set before dispatch, including cancellation/uncertain send
        try:
            if len(commands) == 1 and configure:
                channel, value = commands[0]
                result = await asyncio.wait_for(self.tools.hw_configure(device_id=self.settings.device_id,
                    channel_id=channel, value=value, dry_run=not self.execute),
                    self.settings.request_timeout_s)
            else:
                result = await asyncio.wait_for(self.tools.batch_actuate({"device_id": self.settings.device_id,
                    "commands": [{"channel_id": c, "value": v} for c, v in commands],
                    "dry_run": not self.execute, "verify": False}), self.settings.request_timeout_s)
        except BaseException:
            if self.execute:
                self.evidence.observe_effect(SideEffectState.UNKNOWN)
            raise
        if self.execute:
            self.evidence.observe_effect(result.get("side_effect_state"))
        self.evidence.record(label, commands=commands, result=result)
        if result.get("ok") is not True:
            raw_state = result.get("side_effect_state")
            try:
                state = SideEffectState(raw_state)
            except (ValueError, TypeError):
                state = SideEffectState.UNKNOWN if self.execute else SideEffectState.NONE
            raise Refusal(f"{label} failed; inspect evidence, do not replay", state=state)

    async def stop(self) -> None:
        """Production safety halt; deliberately independent of motion qualification."""
        if self.tools is None:
            raise Refusal("halt connection unavailable; use the physical emergency stop",
                          state=SideEffectState.UNKNOWN)
        try:
            result = await asyncio.wait_for(self.tools.hw_estop(device_id=self.settings.device_id),
                                            self.settings.request_timeout_s)
        except BaseException:
            self.evidence.observe_effect(SideEffectState.UNKNOWN)
            raise
        state = SideEffectState.COMMITTED if result.get("ok") is True else SideEffectState.UNKNOWN
        self.evidence.record("damped_stop", result=result, side_effect_state=state)
        if result.get("ok") is not True:
            raise Refusal("stop delivery unconfirmed; use the physical emergency stop",
                          state=SideEffectState.UNKNOWN)
        self._halt_delivered = True

    async def posture(self) -> Posture:
        """Judge from two qualified samples whether the arm holds its posture.

        ``held`` needs all four at once: every drive enabled, CAN control selected, a
        normal status word, and an unchanged joint vector between the samples. Enable
        flags keep reading true across an emergency stop and the status word recovers
        by itself, so the joint vector is the part a drooping arm cannot satisfy.
        Unreadable feedback is ``unverified``: never evidence of a hold, and never a
        reason to write.
        """
        try:
            previous = await self.snapshot(requalify=True)
            await asyncio.sleep(self.settings.poll_s)
            later = await self.snapshot(requalify=True)
        except Exception as error:  # noqa: BLE001 - cleanup reports, a read never raises here
            return Posture(POSTURE_UNVERIFIED, f"{type(error).__name__}: {error}")
        moved = max(abs(a - b) for a, b in zip(previous.joints, later.joints))
        ctrl_mode = int(later.status.get("ctrl_mode", -1))
        arm_status = int(later.status.get("arm_status", -1))
        if not later.enabled:
            problem = "not every drive reports enabled"
        elif ctrl_mode != CTRL_MODE_CAN:
            problem = f"it is not in CAN control ({ctrl_mode_name(ctrl_mode) or ctrl_mode})"
        elif arm_status != ARM_STATUS_NORMAL:
            problem = f"its status word is not normal ({arm_status})"
        elif moved > self.settings.posture_still_deg:
            problem = f"it is still moving ({moved:.3f} deg between samples)"
        else:
            problem = ""
        return Posture(POSTURE_UNHELD if problem else POSTURE_HELD, problem or "it holds its posture",
                       joints=tuple(later.joints), moved_deg=round(moved, 4), ctrl_mode=ctrl_mode,
                       enabled=bool(later.enabled), arm_status=arm_status)

    @property
    def auto_stop(self) -> bool:
        """Whether an owed stop is sent automatically (``failure_policy == "stop"``)."""
        return self.settings.failure_policy == "stop"

    def _stop_owed(self, exc: BaseException | None) -> bool:
        """Whether this run's own verdict requires a stop before the session ends.

        A failure that never reached the arm -- a preflight refusal, a preview that
        did not match, a declaration that changed -- leaves the arm exactly where the
        last settled command did, so stopping here would only release the posture the
        arm is holding. That is how a failed *verification* dropped this arm 177 mm.
        """
        if self.stop_required:
            return True
        if exc is None:
            return False
        if isinstance(exc, Refusal):
            return exc.envelope.side_effect_state != SideEffectState.NONE
        return True  # cancellation, a read or a defect: none of them say where the arm is

    async def _settle_posture(self, exc: BaseException | None) -> bool:
        """Leave the arm holding, or say why it is not: the end-of-session invariant.

        Returns whether the arm was left verifiably holding, which is what the exit
        needs to know before it decides a close failure still owes a physical halt.
        """
        owed = self._stop_owed(exc)
        own_stop = owed and self.auto_stop
        if own_stop:
            await self.stop()
        elif owed:
            self.evidence.record("stop_withheld", failure_policy=self.settings.failure_policy,
                                 detail="the run failed; holding the last commanded posture instead "
                                        "of an automatic damped stop")
        before = await self.posture()
        if not own_stop:
            # A run that sent no stop leaves a powered arm: residual motion of the
            # last segment or a sample jitter is waited out, never written to.
            before = await self._await_rest(before)
        self.evidence.record("posture_check", own_stop=own_stop, success=self.success, **before.to_dict())
        if before.state == POSTURE_HELD:
            return True
        if not own_stop and before.state == POSTURE_UNHELD and self._only_moving(before):
            reason = ("the arm did not come to rest and no automatic stop was sent; "
                      "watch it and use the physical emergency stop if it does not settle")
        elif before.state == POSTURE_UNVERIFIED:
            reason = "feedback is unreadable; inspect hardware before any recovery command"
            self.evidence.observe_effect(SideEffectState.UNKNOWN)
        elif before.ctrl_mode == CTRL_MODE_TEACH:
            reason = "teach mode belongs to the operator; do not re-energize automatically"
        elif before.arm_status == ARM_STATUS_EMERGENCY_STOP and not own_stop:
            reason = "an operator stop this session did not engage must not be reset"
        else:
            reason = "inspect the arm, then preview and approve a separate governed arm enable"
        self.success = False
        self.evidence.record("posture_unresolved", before=before.to_dict(),
                             operational_complete=False, next_step=reason)
        print("The arm is not holding its posture after this run; " + reason + ".", file=sys.stderr)
        return False

    def _only_moving(self, posture: Posture) -> bool:
        """Healthy, enabled and in CAN control, merely not yet at rest."""
        return (posture.enabled and posture.ctrl_mode == CTRL_MODE_CAN
                and posture.arm_status == ARM_STATUS_NORMAL
                and posture.moved_deg > self.settings.posture_still_deg)

    async def _await_rest(self, posture: Posture) -> Posture:
        """Let the last bounded segment finish before judging; reads only, never writes."""
        deadline = time.monotonic() + self.settings.move_timeout_s
        while posture.state == POSTURE_UNHELD and self._only_moving(posture) and time.monotonic() < deadline:
            await asyncio.sleep(self.settings.poll_s)
            posture = await self.posture()
        return posture

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        cleanup_error: BaseException | None = None
        commands_arm = self.execute and self.touched and not self.leave_unheld
        settled = False
        try:
            if commands_arm:
                settled = await complete_cleanup(self._settle_posture(exc))
                if not settled and exc is None:
                    cleanup_error = Refusal("holding unconfirmed; inspect hardware before continuing")
        except (Exception, asyncio.CancelledError) as error:
            cleanup_error = error
            self.success = False
            self.evidence.observe_effect(SideEffectState.UNKNOWN)
            self._record_cleanup_event("posture_unresolved", operational_complete=False,
                                       next_step="inspect hardware before any recovery command")
            print("Posture settlement unconfirmed: inspect the arm before continuing.", file=sys.stderr)
        finally:
            try:
                await self._close()
            except (Exception, asyncio.CancelledError) as error:
                cleanup_error = cleanup_error or error
                # Closing may already have disposed the original connection, and a fresh
                # read-only client can request the production safety halt. Only an arm
                # this exit could not confirm as holding still owes that halt: a settled
                # arm that merely failed to close must not be dropped by it.
                if commands_arm and not settled and not self._halt_delivered and self.auto_stop:
                    try:
                        await complete_cleanup(self._halt_with_fresh_client())
                    except (Exception, asyncio.CancelledError) as halt_error:
                        self._record_cleanup_error(halt_error)
                        print("Close-failure halt unconfirmed: use the physical emergency stop and inspect.",
                              file=sys.stderr)
        if cleanup_error is not None:
            self.success = False
            if commands_arm and not isinstance(cleanup_error, Refusal):
                self.evidence.observe_effect(SideEffectState.UNKNOWN)
            self._record_cleanup_error(cleanup_error)
            if exc is None:
                failure = Refusal("cleanup unconfirmed; inspect hardware before continuing")
                failure.envelope = self.evidence.failure_envelope(failure)
                raise failure from cleanup_error
        if isinstance(exc, Refusal):
            exc.envelope = self.evidence.failure_envelope(exc)

    async def _halt_with_fresh_client(self) -> None:
        """Request only a safety halt after disposal failure; never replay motion."""
        async with Session(self.settings, self.evidence) as safety:
            await safety.stop()

    def _record_cleanup_event(self, event: str, **details: Any) -> None:
        """An unavailable audit sink must not replace the original run failure."""
        try:
            self.evidence.record(event, **details)
        except OSError:
            print("Cleanup evidence unavailable; preserve the run and inspect.", file=sys.stderr)

    def _record_cleanup_error(self, error: BaseException) -> None:
        self._record_cleanup_event("cleanup_failure", failure_class=type(error).__name__)

    async def _close(self) -> None:
        try:
            await complete_cleanup(self._dispose())
        finally:
            # ContextVar tokens must be reset in the task that created them.
            if self._context_token is not None:
                reset_tool_context(self._context_token)
                self._context_token = None

    async def _dispose(self) -> None:
        first: BaseException | None = None
        try:
            if self.registry is not None:
                await asyncio.wait_for(self.registry.close_all(), self.settings.request_timeout_s)
        except (Exception, asyncio.CancelledError) as error:
            first = error
        try:
            if self.client is not None:
                await asyncio.wait_for(self.client.close(), self.settings.request_timeout_s)
        except (Exception, asyncio.CancelledError) as error:
            first = first or error
        if first is not None:
            raise first
