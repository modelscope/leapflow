# Copyright (c) Alibaba, Inc. and its affiliates.
"""DingTalk credential validator.

Referenced declaratively by ``manifests/dingtalk.yaml`` as
``validation.method: dingtalk_token_check``.
"""

from __future__ import annotations

from typing import Dict, Tuple

from leapflow.gateway.validators._http import make_client_timeout


async def token_check(credentials: Dict[str, str]) -> Tuple[bool, str]:
    """Validate DingTalk credentials by fetching ``access_token``."""
    import aiohttp

    app_key = credentials.get("app_key", "")
    app_secret = credentials.get("app_secret", "")
    if not app_key or not app_secret:
        return False, "Missing app_key or app_secret"

    url = "https://oapi.dingtalk.com/gettoken"
    params = {"appkey": app_key, "appsecret": app_secret}
    async with aiohttp.ClientSession(
        timeout=make_client_timeout(),
        trace_configs=[],
    ) as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            # DingTalk reports failure as errcode/errmsg.
            if data.get("errcode") == 0 and data.get("access_token"):
                return True, ""
            return False, data.get("errmsg", "Unknown error from DingTalk API")


__all__ = ["token_check"]
