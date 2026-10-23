"""Persistent-owner connection and exact-step production approval composition."""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import replace
from typing import Any, Coroutine, TypeVar

from leapflow.engine.recovery.failure_envelope import SideEffectState
from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.context import HardwareContext, TransportRef
from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.permit import PermitIssuer
from leapflow.hardware.providers.yaml_provider import declaration_digest
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

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        cleanup_error: BaseException | None = None
        halt_required = self.execute and self.touched
        halt_attempted = halt_required and (exc is not None or not self.success)
        try:
            if halt_attempted:
                await complete_cleanup(self.stop())
        except (Exception, asyncio.CancelledError) as error:
            cleanup_error = error
            print("Damped halt unconfirmed: use the physical emergency stop and inspect.", file=sys.stderr)
        finally:
            try:
                await self._close()
            except (Exception, asyncio.CancelledError) as error:
                cleanup_error = cleanup_error or error
                if halt_required and not halt_attempted:
                    # Closing may already have disposed the original connection.
                    # A fresh read-only client can request the production safety halt.
                    try:
                        await complete_cleanup(self._halt_with_fresh_client())
                    except (Exception, asyncio.CancelledError) as halt_error:
                        self._record_cleanup_error(halt_error)
                        print("Close-failure halt unconfirmed: use the physical emergency stop and inspect.",
                              file=sys.stderr)
        if cleanup_error is not None:
            if halt_required:
                self.evidence.observe_effect(SideEffectState.UNKNOWN)
            self._record_cleanup_error(cleanup_error)
            if exc is None:
                raise Refusal("cleanup unconfirmed; inspect hardware before continuing",
                              state=SideEffectState.UNKNOWN) from cleanup_error
        if isinstance(exc, Refusal):
            exc.envelope = self.evidence.failure_envelope(exc)

    async def _halt_with_fresh_client(self) -> None:
        """Request only a safety halt after disposal failure; never replay motion."""
        async with Session(self.settings, self.evidence) as safety:
            await safety.stop()

    def _record_cleanup_error(self, error: BaseException) -> None:
        try:
            self.evidence.record("cleanup_failure", failure_class=type(error).__name__)
        except OSError:
            print("Cleanup evidence unavailable; preserve the run and inspect.", file=sys.stderr)

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
