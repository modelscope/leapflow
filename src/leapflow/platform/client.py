"""Platform client utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import Coroutine, Optional

logger = logging.getLogger(__name__)


def fire_and_forget(coro: Coroutine) -> Optional[asyncio.Task]:
    """Schedule a coroutine as a fire-and-forget task with error suppression.

    Prevents "Task exception was never retrieved" warnings by catching and
    logging exceptions from unawaited tasks (e.g. best-effort RPC calls).
    """

    async def _wrapper() -> None:
        try:
            await coro
        except Exception as e:
            logger.debug("Fire-and-forget failed: %s", e)

    try:
        return asyncio.get_running_loop().create_task(_wrapper())
    except RuntimeError:
        return None
