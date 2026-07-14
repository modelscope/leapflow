"""Bridge between gateway events and the platform EventBus.

Subscribes to ``GatewayServer.on_event`` and publishes equivalent
``SystemEvent`` objects into ``EventBus``, making gateway IM signals
visible to the learning pipeline (PatternMiner, Copilot, etc.)
without coupling the gateway to the EventBus directly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)


class GatewayEventBridge:
    """Translates gateway events into EventBus SystemEvents.

    Usage::

        bridge = GatewayEventBridge(event_bus)
        gateway_server = GatewayServer(..., on_event=bridge.on_gateway_event)
    """

    _EVENT_TYPE_MAP = {
        "GatewayMessageReceived": "gateway.message.received",
        "GatewaySessionCreated": "gateway.session.created",
        "GatewaySessionEnded": "gateway.session.ended",
        "BackendEvent": "gateway.signal",
        "InboundCallback": "gateway.callback.received",
    }

    def __init__(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    async def on_gateway_event(self, event: object) -> None:
        """Convert a gateway event and publish it to the EventBus."""
        class_name = type(event).__name__
        event_type = self._EVENT_TYPE_MAP.get(class_name)
        if event_type is None:
            return

        payload = self._extract_payload(event)
        payload["_gateway_class"] = class_name

        try:
            await self._event_bus.handle_event(event_type, payload)
        except Exception:
            logger.debug("Failed to bridge gateway event %s", event_type, exc_info=True)

    @staticmethod
    def _extract_payload(event: object) -> dict[str, Any]:
        """Extract a serializable dict from a gateway event object."""
        try:
            data = asdict(event)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            data = {}
            for attr in ("source", "text", "session_key", "platform_id",
                         "event_type", "event_id", "callback_id"):
                val = getattr(event, attr, None)
                if val is not None:
                    data[attr] = str(val) if not isinstance(val, (dict, list)) else val

        source = getattr(event, "source", None)
        if source is not None:
            platform = getattr(source, "platform", "")
            if platform:
                data["_platform"] = platform

        if "timestamp" not in data:
            data["timestamp"] = time.time()

        return data
