"""One-shot guard — ensures each strategy key fires at most once per lifetime.

Replaces the pattern of 9+ boolean flags (tried_compress, tried_failover, etc.)
with a generic, auditable guard that tracks which strategies have been attempted
and when they were consumed.
"""
from __future__ import annotations

import time


class OneShotGuard:
    """Generalized one-shot guard replacing TurnRecoveryState's boolean flags.

    Each strategy key can fire at most once per guard lifetime.
    Provides an audit trail of which strategies were attempted and when.
    """

    def __init__(self) -> None:
        self._used: dict[str, float] = {}

    def is_available(self, key: str) -> bool:
        """Check whether a strategy key has not been used yet."""
        return key not in self._used

    def mark_used(self, key: str) -> None:
        """Mark a strategy key as consumed. Idempotent — re-marking is a no-op."""
        if key not in self._used:
            self._used[key] = time.time()

    def used_strategies(self) -> list[str]:
        """Return the list of strategy keys that have been used, in usage order."""
        return sorted(self._used.keys(), key=lambda k: self._used[k])

    def usage_history(self) -> dict[str, float]:
        """Return a mapping of strategy key → timestamp when it was consumed."""
        return dict(self._used)

    def reset(self) -> None:
        """Clear all usage records. Intended for testing and turn resets."""
        self._used.clear()

    def __len__(self) -> int:
        """Number of strategy keys that have been consumed."""
        return len(self._used)

    def __contains__(self, key: str) -> bool:
        """Support `key in guard` syntax."""
        return key in self._used
