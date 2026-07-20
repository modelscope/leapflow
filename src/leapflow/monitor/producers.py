"""Producer registry: resolve per-domain observation logic by ``domain`` key.

The registry keeps the runtime domain-agnostic. A new scenario registers a
``MonitorProducer`` for its ``domain``; core scheduling, persistence, and push
code never branch on the domain. Unknown domains resolve to ``None`` so the
manager can skip them gracefully instead of failing.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from leapflow.monitor.types import MonitorProducer

logger = logging.getLogger(__name__)


class ProducerRegistry:
    """Mutable registry mapping ``domain`` -> ``MonitorProducer``."""

    def __init__(self) -> None:
        self._producers: Dict[str, MonitorProducer] = {}

    def register(self, producer: MonitorProducer) -> None:
        """Register (or replace) a producer for its declared domain."""
        domain = producer.domain
        if not domain:
            raise ValueError("MonitorProducer.domain must be a non-empty string")
        self._producers[domain] = producer
        logger.debug("monitor: registered producer for domain=%s", domain)

    def resolve(self, domain: str) -> Optional[MonitorProducer]:
        """Return the producer for a domain, or None when unregistered."""
        return self._producers.get(domain)

    def domains(self) -> list[str]:
        """Return all registered domain keys."""
        return sorted(self._producers.keys())

    def __contains__(self, domain: object) -> bool:
        return domain in self._producers


__all__ = ["ProducerRegistry"]
