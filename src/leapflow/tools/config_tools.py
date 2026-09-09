"""Config tools: let the model read and change settings without touching paths.

Without these, a request like "switch the model to X" has no legal path: the
model has only ``file_read`` / ``shell_run``, so it guesses at
``~/.leapflow/...`` and the workspace sandbox correctly refuses. Telling it
"do not probe that path" in a description only moves the guess elsewhere — the
goal is unchanged while no capability exists to serve it.

These tools close that gap by delegating to ``ConfigService``, the same control
plane behind ``leap config`` and ``/config``. They take a key, never a path, so
the sandbox is never involved and the layout stays an implementation detail.
Routing writes through the service (rather than letting the model edit YAML)
also keeps type coercion, scope validation, vault-backed secrets, and
hot-reload semantics intact.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# A listing of every writable field is long; keep the default bounded and let the
# model narrow by category (categories come back in the payload either way).
_DEFAULT_LIST_LIMIT = 60

# Set by the CLI/daemon so a write can hot-reload the live session, the same way
# ``/config set`` does. Without it a write lands on disk while the in-process
# Settings singleton keeps the old value, and the model's read-back shows the
# stale value and looks like a failed write.
_context_ref: Any = None

# Approval gate for writes. ``requires_approval`` in the tool's x_leapflow block
# only informs capability disclosure — it does not gate execution — so a write
# must consult this explicitly, the same way shell/file_write do. Several writable
# keys weaken safety machinery (``guardrail.enabled``, ``confirm.default_level``,
# ``codegen.sandbox``), so an unguarded config_set would let the model disable its
# own supervision.
_approval_gate: Any = None


def set_config_context(ctx: Any) -> None:
    """Bind the runtime Context so config writes can reload the live session."""
    global _context_ref
    _context_ref = ctx


def set_config_approval_gate(gate: Any) -> None:
    """Install the approval gate consulted before a config write."""
    global _approval_gate
    _approval_gate = gate


def get_config_approval_gate() -> Any:
    """Return the installed config approval gate (or ``None``)."""
    return _approval_gate


def _active_settings() -> Any:
    """Return the live session's settings, falling back to the global singleton."""
    settings = getattr(_context_ref, "settings", None) if _context_ref is not None else None
    if settings is not None:
        return settings
    from leapflow.config import get_settings

    return get_settings()


def _reload_after_write() -> bool:
    """Apply a persisted change to the running session; return whether it took."""
    reload_fn = getattr(_context_ref, "reload_runtime_config_if_changed", None)
    if reload_fn is None:
        return False
    try:
        return bool(reload_fn(force=True))
    except Exception:  # noqa: BLE001 - a failed reload must not undo a valid write
        logger.debug("config write: session reload failed", exc_info=True)
        return False


def _service() -> Any:
    """Build a ConfigService over the active settings."""
    from leapflow.config_service import ConfigService

    return ConfigService(_active_settings())


def _field_payload(view: Any, *, include_description: bool = True) -> Dict[str, Any]:
    """Render a ConfigFieldView for the model.

    ``hot_reload`` is always included: a ``restart-required`` field that appears
    to change but does not take effect is the most confusing outcome of a config
    edit, so the model must be able to tell the user.
    """
    payload: Dict[str, Any] = {
        "key": view.key,
        "value": view.value,
        "value_type": getattr(view.value_type, "__name__", str(view.value_type)),
        "category": view.category,
        "scopes": list(view.scopes),
        "hot_reload": view.hot_reload,
        "secret": bool(view.secret),
    }
    if include_description and view.description:
        payload["description"] = view.description
    if view.value_hint:
        payload["value_hint"] = view.value_hint
    if view.examples:
        payload["examples"] = list(view.examples)
    return payload


async def config_list_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """List writable config fields, optionally narrowed to one category."""
    category = str(args.get("category") or "").strip() or None
    try:
        limit = max(1, int(args.get("limit") or _DEFAULT_LIST_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT

    try:
        service = _service()
        views = service.list_fields(category)
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool failure, not a crash
        logger.debug("config_list failed", exc_info=True)
        return {"ok": False, "error": f"Could not read the config catalog: {exc}", "retryable": False}

    categories = sorted({view.category for view in service.list_fields(None)})
    if category and not views:
        return {
            "ok": False,
            "error": f"No config fields in category {category!r}.",
            "available_categories": categories,
            "retryable": True,
        }

    truncated = len(views) > limit
    return {
        "ok": True,
        "total": len(views),
        "returned": min(len(views), limit),
        "truncated": truncated,
        "categories": categories,
        "fields": [
            _field_payload(view, include_description=bool(category))
            for view in views[:limit]
        ],
    }


async def config_get_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return one config field with its value and semantics."""
    key = str(args.get("key") or "").strip()
    if not key:
        return {"ok": False, "error": "config_get requires 'key'.", "retryable": True}

    try:
        service = _service()
        view = service.describe(key)
    except ValueError as exc:
        # Unknown key is recoverable in the same turn: hand back near matches so
        # the model can correct itself instead of falling back to file probing.
        return {
            "ok": False,
            "error": str(exc),
            "retryable": True,
            "did_you_mean": _suggest(key),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("config_get failed for %s", key, exc_info=True)
        return {"ok": False, "error": f"Could not read config key {key!r}: {exc}", "retryable": False}

    payload = _field_payload(view)
    payload.update({"ok": True})
    return payload


async def _approve_write(key: str, *, scope: str, secret: bool, hot_reload: str) -> str:
    """Return a denial reason, or ``""`` when the write may proceed.

    Goes through the orchestrator's native ``evaluate(ActionDescriptor)`` rather
    than the shell-oriented ``check()``, so a config change is classified as
    ``runtime.configure`` and picks up risk assessment, policy, existing grants,
    and the audit trail for free.

    Fails closed when no gate is installed: an unguarded path here would let the
    model turn off its own guardrails. The value is never included, so a
    credential cannot reach an approval prompt or the audit log.
    """
    gate = _approval_gate
    if gate is None:
        return (
            "Config changes require an approval gate, which is not available in this "
            "session. Ask the user to run `leap config set` / `/config set` instead."
        )
    try:
        from leapflow.security.actions import ActionDescriptor, ActionEffect, ActionKind

        action = ActionDescriptor(
            kind=ActionKind.RUNTIME_CONFIGURE.value,
            summary=f"Change LeapFlow setting {key} (scope={scope})",
            detail=f"config_set {key} scope={scope}",
            effect=ActionEffect.CONFIGURE.value,
            resource=key,
            metadata={
                "tool": "config_set",
                "config_key": key,
                "scope": scope,
                "secret": secret,
                "hot_reload": hot_reload,
            },
        )
        result = await gate.evaluate(action)
    except Exception:  # noqa: BLE001 - a broken gate must not become an open door
        logger.warning("config_set: approval evaluation failed; denying", exc_info=True)
        return "Config change denied: the approval gate could not be consulted."

    if getattr(result, "approved", False):
        return ""
    return str(
        getattr(result, "denial_message", "")
        or getattr(result, "reason", "")
        or f"Config change denied by approval gate: {key}"
    )


async def config_set_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Write one config field through ConfigService."""
    key = str(args.get("key") or "").strip()
    if not key:
        return {"ok": False, "error": "config_set requires 'key'.", "retryable": True}
    if "value" not in args:
        return {"ok": False, "error": "config_set requires 'value'.", "retryable": True}
    scope = str(args.get("scope") or "profile").strip() or "profile"

    try:
        service = _service()
        before = service.describe(key)
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "retryable": True,
            "did_you_mean": _suggest(key),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("config_set failed to describe %s", key, exc_info=True)
        return {"ok": False, "error": f"Could not read config key {key!r}: {exc}", "retryable": False}

    denial = await _approve_write(
        key, scope=scope, secret=bool(before.secret), hot_reload=before.hot_reload,
    )
    if denial:
        return {
            "ok": False,
            "error": denial,
            "failure_code": "approval_denied",
            "retryable": False,
            "requires_approval": True,
            "blocks_approval": True,
            "llm_instruction": (
                "STOP: The configuration change was not approved. Do NOT retry it or "
                "attempt the same change through another tool. Report the denial to the user."
            ),
        }

    try:
        result = service.set(key, args["value"], scope=scope)  # type: ignore[arg-type]
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "retryable": True}
    except Exception as exc:  # noqa: BLE001
        logger.debug("config_set failed for %s", key, exc_info=True)
        return {"ok": False, "error": f"Could not set config key {key!r}: {exc}", "retryable": False}

    payload: Dict[str, Any] = {
        "ok": bool(result.ok),
        "key": key,
        "scope": scope,
        "message": result.message,
        "changed_keys": list(result.changed_keys),
        "hot_reload": before.hot_reload,
    }
    if result.warnings:
        payload["warnings"] = list(result.warnings)
    # Never echo a credential back into the transcript.
    if not before.secret:
        payload["value"] = args["value"]
    if before.hot_reload == "restart-required":
        payload["restart_required"] = True
        payload["next_step"] = "Run `leap daemon restart` for this change to take effect."
    elif result.ok:
        # Reload so an immediate config_get reflects the new value; otherwise the
        # model sees the stale singleton and concludes the write failed.
        payload["session_reloaded"] = _reload_after_write()
    return payload


def _suggest(key: str, *, limit: int = 5) -> list[str]:
    """Return catalog keys resembling ``key``.

    Substring matching alone is not enough: the realistic mistakes are typos
    (``llm.modle``) and dropped separators (``daemon.loglevel``), which share no
    substring with the real key. Fuzzy matching on both the full key and its last
    segment covers those, so the model can correct itself in the same turn
    instead of falling back to probing files.
    """
    needle = str(key or "").strip().lower()
    if not needle:
        return []
    try:
        candidates = list(_service().writable_keys())
    except Exception:  # noqa: BLE001 - suggestions are best-effort
        return []

    ranked: list[str] = []
    # Substring hits first: an exact fragment is a stronger signal than similarity.
    ranked.extend(c for c in candidates if needle in c.lower())

    compact = needle.replace("_", "").replace("-", "").replace(".", "")
    ranked.extend(
        c for c in candidates
        if c.lower().replace("_", "").replace(".", "") == compact
    )
    ranked.extend(difflib.get_close_matches(needle, candidates, n=limit, cutoff=0.6))

    tail = needle.rsplit(".", 1)[-1]
    if tail and tail != needle:
        tails = {c: c.rsplit(".", 1)[-1] for c in candidates}
        ranked.extend(c for c, t in tails.items() if t == tail)
        close_tails = set(difflib.get_close_matches(tail, list(tails.values()), n=limit, cutoff=0.7))
        ranked.extend(c for c, t in tails.items() if t in close_tails)

    seen: set[str] = set()
    ordered = [c for c in ranked if not (c in seen or seen.add(c))]
    return ordered[:limit]


__all__ = [
    "config_get_handler",
    "config_list_handler",
    "config_set_handler",
    "get_config_approval_gate",
    "set_config_approval_gate",
    "set_config_context",
]
