# Copyright (c) Alibaba, Inc. and its affiliates.
"""Registration supervision for configured fleet nodes that are not yet admitted.

A configured node that cannot be reached at startup (a board still booting, a
cable unplugged) used to be dropped for the life of the daemon: the heartbeat
only probes registered nodes, so nothing ever tried again and the only recovery
was restarting leapd. This module keeps such nodes as *pending* with their last
failure and an exponential backoff schedule, so the heartbeat sweep can retry
them and every status surface can say "unregistered, retrying in N s" instead of
showing an empty fleet.

Kept apart from ``fleet.py`` so the manager only gains the hooks that drive it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leapflow.hardware.fleet import FleetNode

DEFAULT_RETRY_INITIAL_S = 5.0
DEFAULT_RETRY_MAX_S = 60.0


@dataclass(frozen=True)
class RegistrationBackoff:
    """Exponential retry schedule: ``initial * 2**(attempts-1)``, capped at ``maximum``."""

    initial_s: float = DEFAULT_RETRY_INITIAL_S
    maximum_s: float = DEFAULT_RETRY_MAX_S

    def __post_init__(self) -> None:
        import math

        if not all(math.isfinite(v) and v > 0 for v in (self.initial_s, self.maximum_s)):
            raise ValueError("Fleet retry delays must be positive finite seconds")
        maximum = float(self.maximum_s)
        object.__setattr__(self, "initial_s", min(float(self.initial_s), maximum))
        object.__setattr__(self, "maximum_s", maximum)

    def delay(self, attempts: int) -> float:
        """Seconds to wait after the ``attempts``-th consecutive failure."""
        exponent = max(0, int(attempts) - 1)
        # Cap the exponent before shifting so a long outage cannot overflow.
        return min(self.maximum_s, self.initial_s * float(2 ** min(exponent, 32)))


@dataclass(frozen=True)
class PendingNode:
    """A configured node that is not registered, with why and when it is retried."""

    node: "FleetNode"
    last_error: str
    failure_code: str
    attempts: int
    next_retry_at: float  # monotonic

    def to_dict(self, now: float) -> dict[str, Any]:
        """Diagnostic view; never includes the node token."""
        return {
            "node_id": self.node.node_id,
            "endpoint": self.node.endpoint,
            "status": "unregistered",
            "last_error": self.last_error,
            "failure_code": self.failure_code,
            "attempts": self.attempts,
            "next_retry_in_s": round(max(0.0, self.next_retry_at - now), 1),
            "auto_retry": True,
        }


class PendingNodes:
    """The set of configured-but-unregistered nodes and their retry schedule."""

    def __init__(self, backoff: RegistrationBackoff | None = None) -> None:
        self._backoff = backoff or RegistrationBackoff()
        self._pending: dict[str, PendingNode] = {}

    @property
    def backoff(self) -> RegistrationBackoff:
        return self._backoff

    def record_failure(self, node: "FleetNode", error: BaseException, now: float) -> PendingNode:
        """Remember one failed registration and schedule the next attempt."""
        previous = self._pending.get(node.node_id)
        attempts = (previous.attempts if previous is not None else 0) + 1
        entry = PendingNode(
            node=node,
            last_error=str(error) or type(error).__name__,
            failure_code=str(getattr(error, "failure_code", "") or "fleet_registration_error"),
            attempts=attempts,
            next_retry_at=now + self._backoff.delay(attempts),
        )
        self._pending[node.node_id] = entry
        return entry

    def due(self, now: float) -> tuple[PendingNode, ...]:
        """Entries whose retry time has arrived, in node-id order."""
        return tuple(
            entry for _, entry in sorted(self._pending.items()) if entry.next_retry_at <= now
        )

    def expedite(self, node_id: str | None, now: float) -> tuple[PendingNode, ...]:
        """Make one (or every) pending entry due now; return what was expedited."""
        selected = [
            entry
            for nid, entry in sorted(self._pending.items())
            if node_id is None or nid == node_id
        ]
        for entry in selected:
            self._pending[entry.node.node_id] = replace(entry, next_retry_at=now)
        return tuple(self._pending[entry.node.node_id] for entry in selected)

    def discard(self, node_id: str) -> PendingNode | None:
        return self._pending.pop(node_id, None)

    def get(self, node_id: str) -> PendingNode | None:
        return self._pending.get(node_id)

    def entries(self) -> tuple[PendingNode, ...]:
        return tuple(entry for _, entry in sorted(self._pending.items()))

    def clear(self) -> None:
        self._pending.clear()

    def __len__(self) -> int:
        return len(self._pending)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._pending


__all__ = [
    "DEFAULT_RETRY_INITIAL_S",
    "DEFAULT_RETRY_MAX_S",
    "PendingNode",
    "PendingNodes",
    "RegistrationBackoff",
]
