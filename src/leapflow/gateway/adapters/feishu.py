"""Feishu/Lark adapter backed by the official lark-cli."""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from leapflow.gateway.action_packs.feishu import ACTION_SPECS
from leapflow.gateway.adapters.common import AdapterLifecycle, bool_option
from leapflow.gateway.backends.cli_backend import CliBackend
from leapflow.gateway.connectors.action_registry import ActionRegistry
from leapflow.gateway.connectors.event_sources import UnavailableEventSource
from leapflow.gateway.connectors.lark_event_source import BotIdentity, LarkCliEventSource
from leapflow.gateway.connectors.protocol import (
    ActionDiscovery,
    ActionPreview,
    ActionResult,
    ActionSpec,
    BackendEventSource,
    BackendStatus,
    ExecutionBackend,
)
from leapflow.gateway.protocol import OutboundContent, SendResult, SendTarget

logger = logging.getLogger(__name__)


class BackendNotReadyError(RuntimeError):
    """Raised when a connector backend is installed but not ready to use."""

    def __init__(self, detail: str, metadata: Mapping[str, Any]) -> None:
        super().__init__(detail)
        self.metadata = dict(metadata)


class FeishuAdapter(AdapterLifecycle):
    """Feishu adapter implemented through the generic CLI backend."""

    platform_id = "feishu"
    supports_async_delivery = True
    max_message_length = 8000

    def __init__(
        self,
        profile: str = "",
        identity: str = "bot",
        binary: str = "lark-cli",
        max_message_length: int = 8000,
        backend: ExecutionBackend | None = None,
        events_enabled: Any = False,
        event_keys: Sequence[str] | str | None = None,
        event_key: str = "",
        **_: Any,
    ) -> None:
        super().__init__(profile=profile or "default")
        self._profile = profile or ""
        self._identity = identity or "bot"
        self._binary = binary or "lark-cli"
        self.max_message_length = max(1, int(max_message_length or 8000))
        self._backend = backend or CliBackend(
            binary=self._binary,
            profile=self._profile,
            identity=self._identity,
        )
        discovery = self._backend if isinstance(self._backend, ActionDiscovery) else None
        self._registry = ActionRegistry(ACTION_SPECS, discovery=discovery)
        self._bot_id = ""
        self._bot_name = ""

        self._events_enabled = bool_option(events_enabled, default=False)
        if self._events_enabled:
            resolved_keys: Sequence[str]
            if event_keys:
                resolved_keys = [event_keys] if isinstance(event_keys, str) else list(event_keys)
            elif event_key:
                resolved_keys = [event_key]
            else:
                from leapflow.gateway.connectors.lark_event_source import _DEFAULT_EVENT_KEYS
                resolved_keys = list(_DEFAULT_EVENT_KEYS)
            self._event_source: BackendEventSource = LarkCliEventSource(
                event_keys=resolved_keys,
                binary=self._binary,
                profile=self._profile,
                identity=self._identity,
            )
        else:
            self._event_source = UnavailableEventSource(
                platform_id=self.platform_id,
                backend_kind=self._backend.kind,
                detail=(
                    "Feishu inbound events are not enabled. "
                    "Set events_enabled=true in platform options to receive messages."
                ),
                metadata={
                    "available": False,
                    "current_mode": "outbound_actions_only",
                },
            )

    async def connect(self, *, is_reconnect: bool = False) -> None:
        status = await self._backend.status()
        if not status.ok:
            metadata = {**self.status_metadata(), **dict(status.metadata)}
            raise BackendNotReadyError(status.detail or "Feishu CLI backend is not ready", metadata)

        if isinstance(self._event_source, LarkCliEventSource):
            identity = await self._event_source.fetch_bot_identity()
            self._bot_id = identity.open_id
            self._bot_name = identity.app_name
        await super().connect(is_reconnect=is_reconnect)

    @property
    def bot_identity(self) -> BotIdentity:
        """Return the resolved bot identity (open_id + app_name)."""
        return BotIdentity(open_id=self._bot_id, app_name=self._bot_name)

    def status_metadata(self) -> dict[str, Any]:
        """Return non-secret connector diagnostics for status/list UX."""
        binary = getattr(self._backend, "binary", "")
        profile = getattr(self._backend, "profile", self._profile) or "default"
        identity = getattr(self._backend, "identity", self._identity)
        events_available = self._events_enabled
        return {
            "backend_kind": self._backend.kind,
            "binary": binary,
            "profile": profile,
            "identity": identity,
            "actions": sorted(self._registry.all().keys()),
            "event_source": {
                "available": events_available,
                "mode": "bidirectional" if events_available else "outbound_actions_only",
            },
        }

    async def backend_status(self) -> BackendStatus:
        """Return live backend auth diagnostics without exposing credentials."""
        status = await self._backend.status()
        return BackendStatus(
            ok=status.ok,
            backend_kind=status.backend_kind,
            detail=status.detail,
            metadata={**self.status_metadata(), **dict(status.metadata)},
        )

    async def disconnect(self) -> None:
        await super().disconnect()

    async def send(self, target: SendTarget, content: OutboundContent) -> SendResult:
        result = await self.execute_action(
            "im.send_message",
            {
                "chat_id": target.chat_id,
                "thread_id": target.thread_id,
                "text": content.text[:self.max_message_length],
            },
        )
        if not result.ok:
            return SendResult(ok=False, error=result.error)
        return SendResult(ok=True, message_id=result.resource_id)

    def action_specs(self) -> Mapping[str, ActionSpec]:
        """Return Feishu actions exposed by this connector."""
        return self._registry.all()

    def action_spec(self, action: str) -> ActionSpec | None:
        """Return one Feishu action spec."""
        return self._registry.get(action)

    async def preview_action(
        self,
        action: str,
        payload: Mapping[str, Any],
    ) -> ActionPreview:
        """Return a side-effect-free action preview for approval."""
        spec = self._registry.get(action)
        if spec is None:
            return ActionPreview(ok=False, error=f"Unknown Feishu action: {action}")
        validation = self._registry.validate(action, payload)
        if not validation.ok:
            return ActionPreview(ok=False, error=validation.error)
        preview = getattr(self._backend, "preview", None)
        if preview is None:
            return ActionPreview(ok=True, summary=f"Run {self.platform_id}.{action}")
        return await preview(spec, payload)

    def event_source(self) -> BackendEventSource | None:
        """Return the configured inbound event source, if available."""
        return self._event_source

    async def discover_actions(self, *, groups: Sequence[str] = ()) -> int:
        """Discover additional actions via CLI --help introspection.

        Returns the number of newly discovered actions.
        """
        return await self._registry.refresh_discovery(groups=groups)

    async def execute_action(
        self,
        action: str,
        payload: Mapping[str, Any],
    ) -> ActionResult:
        """Execute a Feishu action through the configured backend."""
        spec = self._registry.get(action)
        if spec is None:
            return ActionResult(ok=False, error=f"Unknown Feishu action: {action}")
        validation = self._registry.validate(action, payload)
        if not validation.ok:
            return ActionResult(ok=False, error=validation.error)
        return await self._backend.execute(spec, payload)
