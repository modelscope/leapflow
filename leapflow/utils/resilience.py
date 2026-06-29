"""Universal async resilience: timeout + retry with exponential backoff.

Provides a composable execution wrapper usable across all execution paths
(registry invoke, ReAct steps, scheduler nodes). Follows OCP: retryable
predicate is injectable, not hard-coded.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _default_retryable(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError))


@dataclass
class ResiliencePolicy:
    """Configuration for timeout and retry behavior."""

    timeout_s: float = 60.0
    max_retries: int = 0
    backoff_base: float = 1.0
    backoff_multiplier: float = 2.0
    retryable: Callable[[Exception], bool] = field(default_factory=lambda: _default_retryable)

    def delay_for_attempt(self, attempt: int) -> float:
        return self.backoff_base * (self.backoff_multiplier ** attempt)


async def execute_with_resilience(
    coro_factory: Callable[[], Awaitable[Any]],
    policy: ResiliencePolicy,
) -> Any:
    """Execute an async operation with timeout and retry.

    Args:
        coro_factory: Zero-arg callable returning a fresh coroutine per attempt.
        policy: Resilience configuration.

    Raises:
        asyncio.CancelledError: Always re-raised immediately (never retried).
        The last exception encountered if all attempts fail.
    """
    last_exc: Exception = RuntimeError("no attempts made")

    for attempt in range(policy.max_retries + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=policy.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < policy.max_retries and policy.retryable(exc):
                delay = policy.delay_for_attempt(attempt)
                logger.debug(
                    "resilience.retry attempt=%d delay=%.2fs error=%s",
                    attempt + 1, delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                raise

    raise last_exc
