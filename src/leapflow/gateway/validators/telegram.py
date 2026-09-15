# Copyright (c) Alibaba, Inc. and its affiliates.
"""Telegram credential validator.

Referenced declaratively by ``manifests/telegram.yaml`` as
``validation.method: telegram_getme``.
"""

from __future__ import annotations

from typing import Dict, Tuple

from leapflow.gateway.validators._http import make_client_timeout


async def getme(credentials: Dict[str, str]) -> Tuple[bool, str]:
    """Validate a Telegram bot token via the ``getMe`` API.

    The token is embedded in the URL path (Telegram API convention), so
    ``trace_configs=[]`` disables aiohttp request tracing to keep the
    token-bearing URL out of debug logs.
    """
    import aiohttp

    token = credentials.get("bot_token", "")
    if not token:
        return False, "Missing bot_token"

    url = f"https://api.telegram.org/bot{token}/getMe"
    async with aiohttp.ClientSession(
        timeout=make_client_timeout(),
        trace_configs=[],
    ) as session:
        async with session.get(url) as resp:
            data = await resp.json()
            # Telegram reports failure as ok=false with a description.
            if data.get("ok"):
                bot_name = data.get("result", {}).get("username", "unknown")
                return True, f"Bot: @{bot_name}"
            return False, data.get("description", "Invalid bot token")


__all__ = ["getme"]
