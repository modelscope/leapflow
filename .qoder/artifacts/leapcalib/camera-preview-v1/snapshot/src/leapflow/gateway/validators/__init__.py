# Copyright (c) Alibaba, Inc. and its affiliates.
"""Platform credential validation: a neutral registry plus per-vendor modules.

Each validator is a simple async function:
``(credentials: Dict[str, str]) → (ok: bool, error_or_info: str)``.

Validators are registered by name so YAML manifests can reference them
declaratively (``validation.method``). A new platform adds a module here plus a
``register_validator`` call — no change to gateway core.

Vendor implementations live in sibling modules (``dingtalk.py``,
``telegram.py``) rather than in this file, keeping platform endpoints and error
shapes out of gateway core, alongside how ``adapters/`` and ``normalizers/`` are
organized. They are imported eagerly at the bottom of this module because
``GatewayServer.configure_platform`` validates credentials *before* it builds
the adapter: a validator registered lazily from an adapter module would not
exist yet at that point.
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


def registered_validators() -> tuple[str, ...]:
    """Return the registered validator names (sorted, for diagnostics)."""
    return tuple(sorted(_registry))


async def validate_credentials(
    method_name: str,
    credentials: Dict[str, str],
    *,
    timeout_s: float = 10.0,
) -> Tuple[bool, str]:
    """Run the named validator with a timeout.

    An empty ``method_name`` means the manifest opts out of validation, which is
    a legitimate configuration and passes. A *named but unregistered* validator
    is a configuration error, and is rejected rather than passed: silently
    treating it as "valid" would store unverified credentials and surface the
    problem much later as an opaque runtime failure.

    Errors are redacted before returning.
    """
    if not method_name:
        return True, ""

    fn = _registry.get(method_name)
    if fn is None:
        logger.warning(
            "gateway: no credential validator registered for %r (registered: %s)",
            method_name, ", ".join(registered_validators()) or "none",
        )
        return False, (
            f"Credential validator {method_name!r} is not registered, so these "
            "credentials cannot be verified."
        )

    try:
        ok, msg = await asyncio.wait_for(fn(credentials), timeout=timeout_s)
    except asyncio.TimeoutError:
        return False, f"Validation timed out after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001 - any vendor error becomes a safe message
        from leapflow.security.redact import redact_sensitive_text

        safe_error = redact_sensitive_text(str(exc), force=True)
        return False, f"Validation error: {safe_error}"

    if not ok and msg:
        # Vendor error strings can echo request parameters back; never assume a
        # third-party API keeps credentials out of its error messages.
        from leapflow.security.redact import redact_sensitive_text

        msg = redact_sensitive_text(msg, force=True)
    return ok, msg


# ── Register built-in validators (eager: see module docstring) ────────
from leapflow.gateway.validators import dingtalk as _dingtalk  # noqa: E402
from leapflow.gateway.validators import telegram as _telegram  # noqa: E402

register_validator("dingtalk_token_check", _dingtalk.token_check)
register_validator("telegram_getme", _telegram.getme)

__all__ = [
    "ValidatorFn",
    "get_validator",
    "register_validator",
    "registered_validators",
    "validate_credentials",
]
