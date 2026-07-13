"""Reusable backend event source implementations."""
from __future__ import annotations

from typing import Any, AsyncIterator, Mapping

from leapflow.gateway.connectors.protocol import BackendEvent, EventSourceStatus


class UnavailableEventSource:
    """Event source that explicitly reports an unsupported or unconfigured inbound path."""

    def __init__(
        self,
        *,
        platform_id: str,
        backend_kind: str,
        detail: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.platform_id = platform_id
        self.backend_kind = backend_kind
        self._detail = detail
        self._metadata = dict(metadata or {})
        self._started = False

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        self._started = False
        return EventSourceStatus(
            ok=False,
            backend_kind=self.backend_kind,
            detail=self._detail,
            checkpoint=checkpoint,
            metadata={**self._metadata, "available": False},
        )

    async def stop(self) -> EventSourceStatus:
        self._started = False
        return await self.status()

    async def events(self) -> AsyncIterator[BackendEvent]:
        if False:
            yield BackendEvent(event_id="", event_type="", platform_id=self.platform_id)
        return

    async def status(self) -> EventSourceStatus:
        return EventSourceStatus(
            ok=False,
            backend_kind=self.backend_kind,
            detail=self._detail,
            metadata={**self._metadata, "available": False, "started": self._started},
        )
