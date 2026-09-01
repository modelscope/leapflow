"""Approval lifecycle coordinator extracted from RuntimeLeapService."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from leapflow.daemon.protocol import StreamChunk
from leapflow.daemon.turn_admission import parked_for_human_decision

logger = logging.getLogger(__name__)


class ApprovalCoordinator:
    """Manages daemon approval lifecycle: pending queue, resolution, cleanup.

    Pending approvals never expire on a timer. They are released by liveness,
    not by age: the owning turn ends, its stream closes, the command finishes, or
    the client's connection drops (a failed heartbeat write cancels the handler,
    which runs those same release paths within one heartbeat interval).
    """

    def __init__(self) -> None:
        self._approval_pending: dict[str, dict[str, Any]] = {}
        # request_ids that currently have an approval route installed. Registered
        # by whoever sets the route ContextVar (engine turns and command.execute)
        # and removed in the same finally, so it tracks live owners rather than
        # elapsed time.
        self._live_routes: set[str] = set()

    def register_route(self, request_id: str) -> None:
        """Mark *request_id* as owning a live approval route."""
        if request_id:
            self._live_routes.add(str(request_id))

    def unregister_route(self, request_id: str) -> None:
        """Drop *request_id* from the live route set."""
        self._live_routes.discard(str(request_id))

    def install_gate(self, ctx: Any, service: Any) -> None:
        """Install the daemon-mode approval gate on ctx.

        *service* is the owning RuntimeLeapService (needed by _DaemonApprovalGate
        to route approval requests back through the coordinator).
        """
        try:
            from leapflow.security.approval import SessionAwareGate
            from leapflow.security.actions import ActionDescriptor
            from leapflow.security.orchestrator import ApprovalOrchestrator
            from leapflow.security.policy import ApprovalPolicyEngine
            from leapflow.plugins import get_registry as _get_tool_registry
            from leapflow.tools.config_tools import set_config_approval_gate
            from leapflow.tools.gateway_tool import set_gateway_approval_gate
            from leapflow.tools.shell_tools import set_approval_gate
            from leapflow.tools.web_fetch import set_web_approval_gate

            existing = getattr(ctx, "_approval_orchestrator", None)
            gate = SessionAwareGate(_DaemonApprovalGate(self))
            # The hardware classifier is composed here too, not only in-process: a gate
            # installed at one site and not the other makes a device behave differently
            # depending on whether leapd happens to be running. The registry is built by
            # LeapContext; reusing it keeps both paths assessing the same declarations.
            from leapflow.hardware.risk import build_risk_classifier

            orchestrator = ApprovalOrchestrator(
                gate,
                risk_classifier=build_risk_classifier(getattr(ctx, "_hardware_registry", None)),
                policy=ApprovalPolicyEngine(bypass=getattr(getattr(ctx, 'settings', None), 'approval_bypass', False)),
                grants=getattr(existing, "grants", None),
                audit=getattr(existing, "audit", None),
            )
            ctx._approval_gate = gate
            ctx._approval_orchestrator = orchestrator
            set_approval_gate(orchestrator)
            set_gateway_approval_gate(orchestrator)
            # Config writes go through the same daemon-side approval path, so a
            # daemon session cannot change settings unattended either.
            set_config_approval_gate(orchestrator)
            # Same for outbound fetches that resolve to internal addresses.
            set_web_approval_gate(orchestrator)
            # Mutating semantic desktop tools (click, type_text, ...) share the
            # same approval path.
            _tool_registry = _get_tool_registry()
            _tool_registry.set_desktop_gate(orchestrator)
            # Inject the plugin self-modification approval gate (Phase 2.4
            # Self-Modification). Reuses the same orchestrator as the desktop
            # gate so plugin management gets identical human-in-the-loop
            # treatment; bind_runtime only reaches plugins that declare the
            # 'plugin_approval_gate' dependency (self_management).
            #
            # Same call also wires the active LLM provider and the opt-in
            # plugin_generation_enabled flag into self_management, so its
            # plugin_generate handler can drive real code synthesis in daemon
            # mode. ``ctx.llm`` is None-safe: if credentials are absent, the
            # handler still reports the missing provider instead of crashing.
            settings = getattr(ctx, "settings", None)
            # Resolve the profile-scoped plugin install directory and (optionally)
            # a marketplace client from settings. Both are injected via the same
            # bind_runtime path; self_management declares them as dependencies.
            plugin_install_dir = self._resolve_plugin_install_dir(settings)
            marketplace_client = self._build_marketplace_client(settings, plugin_install_dir)
            _tool_registry.bind_runtime(
                plugin_approval_gate=orchestrator,
                hardware_approval_gate=orchestrator,
                llm_provider=getattr(ctx, "llm", None),
                plugin_generation_enabled=bool(
                    getattr(settings, "plugin_generation_enabled", False)
                ),
                plugin_install_dir=plugin_install_dir,
                marketplace_client=marketplace_client,
                marketplace_trusted_pubkeys=tuple(
                    getattr(settings, "plugin_marketplace_trusted_pubkeys", ()) or ()
                ),
            )

            class _FileReadGate:
                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(
                    self,
                    path: str,
                    mode: str = "raw",
                    sensitivity_meta: dict | None = None,
                ) -> bool:
                    result = await orchestrator.evaluate(
                        ActionDescriptor.file_read(path, mode=mode, metadata=dict(sensitivity_meta or {}))
                    )
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            class _FileWriteGate:
                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(
                    self,
                    path: str,
                    content: str,
                    mode: str = "overwrite",
                    sensitivity_meta: dict | None = None,
                ) -> bool:
                    result = await orchestrator.evaluate(
                        ActionDescriptor.file_write(path, content, mode=mode, metadata=dict(sensitivity_meta or {}))
                    )
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            _tool_registry.set_file_read_gate(_FileReadGate())
            _tool_registry.set_file_write_gate(_FileWriteGate())
            logger.debug("daemon approval gate installed")
        except (ImportError, AttributeError) as exc:
            logger.debug("daemon approval gate installation skipped: %s", exc, exc_info=True)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch: gate wiring must not crash daemon startup
            logger.error(
                "daemon approval gate installation failed with unexpected error: %s. "
                "Daemon will continue without full approval gating. "
                "This is a serious safety issue that should be investigated.",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _resolve_plugin_install_dir(settings: Any) -> str | None:
        """Resolve the profile-scoped directory for installed plugins.

        Precedence: explicit ``Settings.plugin_install_dir`` -> the active
        ``ProfileLayout.plugins_dir``. Returns ``None`` when neither can be
        determined, in which case self_management falls back to resolving the
        layout itself. Never joins ad-hoc path strings; always defers to the
        layout API for the profile-scoped default.
        """
        configured = getattr(settings, "plugin_install_dir", None)
        if configured:
            return str(configured)
        profile_layout = getattr(settings, "profile_layout", None)
        if profile_layout is not None:
            try:
                return str(profile_layout.plugins_dir)
            except (AttributeError, OSError):
                return None
        return None

    @staticmethod
    def _build_marketplace_client(settings: Any, install_dir: str | None) -> Any:
        """Build a MarketplaceClient from settings, or None when unconfigured.

        A URL source takes precedence over a local directory root. The client
        installs into the resolved profile-scoped plugins directory so that
        marketplace and code installs share one managed location.
        """
        root = getattr(settings, "plugin_marketplace_root", None)
        url = getattr(settings, "plugin_marketplace_url", None)
        if not root and not url:
            return None
        target_dir = install_dir
        if not target_dir:
            profile_layout = getattr(settings, "profile_layout", None)
            if profile_layout is None:
                return None
            target_dir = str(profile_layout.plugins_dir)
        try:
            from pathlib import Path

            from leapflow.plugins.marketplace import (
                HttpMarketplaceSource,
                MarketplaceClient,
            )
            from leapflow.plugins.marketplace.client import LocalDirectorySource

            if url:
                source: Any = HttpMarketplaceSource(str(url))
            else:
                source = LocalDirectorySource(Path(str(root)))
            return MarketplaceClient(source, install_dir=Path(target_dir))
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("marketplace client construction failed: %s", exc, exc_info=True)
            return None

    async def request_approval(self, request: Any, route: "tuple[asyncio.Queue[StreamChunk], str] | None") -> str:
        """Block until the human decides; called from tool execution.

        *route* is the per-turn (queue, request_id) tuple from the ContextVar.

        There is no timeout. The wait ends only when the user answers, or when
        the owning turn/stream/command ends and denies the request through
        ``deny_for_request``/``deny_for_queue``. A deadline here used to auto-deny
        whatever the user had stepped away from, so an action the user never saw
        was refused on their behalf.

        The turn's admission slot is handed back for the duration of the wait:
        human think-time is unbounded, and holding one of N slots would let a few
        unanswered prompts stop every other workspace from starting a turn and
        block exclusive maintenance (config reload, daemon stop).
        """
        if route is None:
            return "deny"
        queue, active_request_id = route
        pending_id = str(getattr(request, "request_id", "") or uuid.uuid4().hex)
        request_id = active_request_id or pending_id
        payload = request.to_dict()
        payload["pending_id"] = pending_id
        payload["request_id"] = request_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._approval_pending[pending_id] = {
            "request": payload,
            "future": future,
            "queue": queue,
            "created_at": time.time(),
        }
        await queue.put(StreamChunk(
            request_id=request_id,
            content="Approval required",
            event_type="approval_request",
            metadata={"approval": payload, "request_id": request_id},
        ))
        try:
            async with parked_for_human_decision():
                result = await future
            return str(result.get("decision") or "deny")
        finally:
            self._approval_pending.pop(pending_id, None)

    async def resolve(self, pending_id: str, decision: str, reason: str = "") -> dict[str, Any]:
        """Resolve a pending approval."""
        pending = self._approval_pending.get(pending_id)
        if pending is None:
            return {"ok": False, "error": f"Unknown approval request: {pending_id}"}
        future = pending.get("future")
        if not isinstance(future, asyncio.Future) or future.done():
            return {"ok": False, "error": f"Approval request is no longer pending: {pending_id}"}
        future.set_result({"decision": self._normalize_decision(decision), "reason": reason})
        return {"ok": True, "pending_id": pending_id, "decision": self._normalize_decision(decision)}

    async def cancel(self, pending_id: str, reason: str = "cancelled") -> dict[str, Any]:
        """Cancel a pending approval."""
        return await self.resolve(pending_id, "deny", reason=reason)

    def get_status(self) -> dict[str, Any]:
        """Return current approval queue status."""
        return {"pending": self._pending_payloads()}

    def pending_count(self) -> int:
        """Return the number of currently pending approvals."""
        return len(self._approval_pending)

    def deny_for_queue(self, queue: "asyncio.Queue[StreamChunk]", reason: str = "stream_closed") -> None:
        """Deny all pending approvals bound to a specific queue."""
        for pending_id, pending in list(self._approval_pending.items()):
            if pending.get("queue") is not queue:
                continue
            future = pending.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result({"decision": "deny", "reason": reason})
            self._approval_pending.pop(pending_id, None)

    def deny_for_request(self, request_id: str, reason: str = "turn_ended") -> None:
        """Deny all pending approvals bound to a specific request.

        Called when a turn ends (normally or exceptionally) to prevent
        orphaned approval futures from leaking memory indefinitely.
        """
        for pending_id, pending in list(self._approval_pending.items()):
            payload = pending.get("request") or {}
            if payload.get("request_id") != request_id:
                continue
            future = pending.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result({"decision": "deny", "reason": reason})
            self._approval_pending.pop(pending_id, None)

    def prune_orphaned(self) -> int:
        """Release pendings whose owning turn/command is gone. Returns count.

        A backstop for the release paths, keyed on liveness rather than age: a
        pending is dropped only when its ``request_id`` no longer has a live
        route. Deliberately conservative — a pending with no request_id is left
        alone, because guessing here would auto-deny a prompt the user is still
        looking at, which is exactly the behaviour this design removes.
        """
        pruned = 0
        for pending_id, pending in list(self._approval_pending.items()):
            payload = pending.get("request") or {}
            request_id = str(payload.get("request_id") or "")
            if not request_id or request_id in self._live_routes:
                continue
            future = pending.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result({"decision": "deny", "reason": "owner_gone"})
            self._approval_pending.pop(pending_id, None)
            pruned += 1
        if pruned:
            logger.info("daemon: released %d approval(s) whose owner is gone", pruned)
        return pruned

    def _pending_payloads(self) -> list[dict[str, Any]]:
        return [dict(item.get("request") or {}) for item in self._approval_pending.values()]

    @staticmethod
    def _normalize_decision(decision: str) -> str:
        allowed = {
            "allow",
            "allow_once",
            "allow_session",
            "allow_all_session",
            "allow_always",
            "deny",
            "deny_always",
            "cancel_workflow",
        }
        value = str(decision or "deny").strip().lower()
        return value if value in allowed else "deny"


class _DaemonApprovalGate:
    """Approval gate that bridges daemon-side actions to thin clients."""

    def __init__(self, coordinator: ApprovalCoordinator) -> None:
        self._coordinator = coordinator

    async def request_approval(self, request: Any) -> Any:
        from leapflow.security.approval import ApprovalDecision

        # Import ContextVar from shared module (avoids circular dep with service).
        from leapflow.daemon.approval_route import approval_route as _approval_route

        route = _approval_route.get()
        decision = await self._coordinator.request_approval(request, route)
        try:
            return ApprovalDecision(decision)
        except ValueError:
            return ApprovalDecision.DENY
