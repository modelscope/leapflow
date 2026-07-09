"""Platform credential validation functions.

Each validator is a simple async function:
``(credentials: Dict[str, str]) → (ok: bool, error_or_info: str)``.

Validators are registered by name so YAML manifests can reference them
declaratively.  New platforms add a validator + register call — zero
core changes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ValidatorFn = Callable[[Dict[str, str]], Awaitable[Tuple[bool, str]]]

_registry: Dict[str, ValidatorFn] = {}


# ═══════════════════════════════════════════════════════════════
# Registry API
# ═══════════════════════════════════════════════════════════════

def register_validator(name: str, fn: ValidatorFn) -> None:
    """Register a credential validator by name."""
    _registry[name] = fn


def get_validator(name: str) -> Optional[ValidatorFn]:
    """Retrieve a registered validator (or ``None``)."""
    return _registry.get(name)


async def validate_credentials(
    method_name: str,
    credentials: Dict[str, str],
    *,
    timeout_s: float = 10.0,
) -> Tuple[bool, str]:
    """Run the named validator with a timeout.

    Returns ``(True, "")`` if no validator is registered (safe default).
    Errors are redacted before returning.
    """
    fn = _registry.get(method_name)
    if fn is None:
        return True, ""

    try:
        ok, msg = await asyncio.wait_for(fn(credentials), timeout=timeout_s)
        return ok, msg
    except asyncio.TimeoutError:
        return False, f"Validation timed out after {timeout_s}s"
    except Exception as exc:
        from leapflow.security.redact import redact_sensitive_text

        safe_error = redact_sensitive_text(str(exc), force=True)
        return False, f"Validation error: {safe_error}"


# ═══════════════════════════════════════════════════════════════
# Built-in validators (for bundled manifests)
# ═══════════════════════════════════════════════════════════════

def _make_client_timeout() -> "aiohttp.ClientTimeout":
    """Create a strict per-request timeout for credential validation.

    Imported lazily to avoid top-level ``aiohttp`` dependency.
    """
    import aiohttp

    return aiohttp.ClientTimeout(total=8, connect=5)


async def _feishu_token_check(credentials: Dict[str, str]) -> Tuple[bool, str]:
    """Validate Feishu credentials by fetching ``tenant_access_token``."""
    import aiohttp

    app_id = credentials.get("app_id", "")
    app_secret = credentials.get("app_secret", "")
    if not app_id or not app_secret:
        return False, "Missing app_id or app_secret"

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    async with aiohttp.ClientSession(
        timeout=_make_client_timeout(),
        trace_configs=[],
    ) as session:
        async with session.post(
            url,
            json={"app_id": app_id, "app_secret": app_secret},
        ) as resp:
            data = await resp.json()
            if data.get("code") == 0 and data.get("tenant_access_token"):
                return True, ""
            return False, data.get("msg", "Unknown error from Feishu API")


async def _dingtalk_token_check(credentials: Dict[str, str]) -> Tuple[bool, str]:
    """Validate DingTalk credentials by fetching ``access_token``."""
    import aiohttp

    app_key = credentials.get("app_key", "")
    app_secret = credentials.get("app_secret", "")
    if not app_key or not app_secret:
        return False, "Missing app_key or app_secret"

    url = "https://oapi.dingtalk.com/gettoken"
    params = {"appkey": app_key, "appsecret": app_secret}
    async with aiohttp.ClientSession(
        timeout=_make_client_timeout(),
        trace_configs=[],
    ) as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if data.get("errcode") == 0 and data.get("access_token"):
                return True, ""
            return False, data.get("errmsg", "Unknown error from DingTalk API")


async def _telegram_getme(credentials: Dict[str, str]) -> Tuple[bool, str]:
    """Validate Telegram bot token via ``getMe`` API.

    The token is embedded in the URL path (Telegram API convention).
    ``trace_configs=[]`` disables aiohttp request tracing to prevent
    the token-containing URL from appearing in debug logs.
    """
    import aiohttp

    token = credentials.get("bot_token", "")
    if not token:
        return False, "Missing bot_token"

    url = f"https://api.telegram.org/bot{token}/getMe"
    async with aiohttp.ClientSession(
        timeout=_make_client_timeout(),
        trace_configs=[],
    ) as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if data.get("ok"):
                bot_name = data.get("result", {}).get("username", "unknown")
                return True, f"Bot: @{bot_name}"
            return False, data.get("description", "Invalid bot token")


# ── Auto-register built-in validators ────────────────────────
register_validator("feishu_token_check", _feishu_token_check)
register_validator("dingtalk_token_check", _dingtalk_token_check)
register_validator("telegram_getme", _telegram_getme)
