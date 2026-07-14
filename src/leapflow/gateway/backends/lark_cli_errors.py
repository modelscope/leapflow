"""lark-cli error normalization for App Connector CLI actions.

This module translates lark-cli's Problem JSON and legacy plain-text failures
into the platform-neutral ActionFailure model. It intentionally lives in the
backend layer so gateway core remains free of vendor/CLI wire-shape knowledge.
"""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.gateway.connectors.protocol import ActionFailure, ActionSpec
from leapflow.security.redact import redact_sensitive_text


def classify_lark_cli_failure(
    spec: ActionSpec,
    error: str,
    raw: Mapping[str, Any],
    *,
    binary: str = "",
    profile: str = "",
    identity: str = "",
) -> ActionFailure:
    """Classify lark-cli action failures into platform-neutral ActionFailure."""
    capability = spec.capability or spec.name

    error_obj = raw.get("error")
    if isinstance(error_obj, Mapping):
        typed_error = str(error_obj.get("type") or "")
        if typed_error:
            return _classify_problem_error(
                error_obj,
                spec,
                error,
                binary=binary,
                profile=profile,
                identity=identity or str(error_obj.get("identity") or ""),
            )
        error = str(error_obj.get("message") or error_obj.get("hint") or error)

    err_type = str(raw.get("type") or "")
    if err_type in ("authorization", "authentication", "api"):
        return _classify_problem_error(
            raw,
            spec,
            error,
            binary=binary,
            profile=profile,
            identity=identity,
        )

    lowered = error.lower()
    safe_error = redact_sensitive_text(error, force=True)
    capability_hint = f"Action: {spec.name}. Capability: {capability}."

    if any(kw in lowered for kw in ("access denied", "access_denied", "permission denied")):
        login_cmd = _auth_cmd(binary, profile, "login")
        status_cmd = _auth_cmd(binary, profile, "status")
        return ActionFailure(
            failure_class="authorization",
            failure_code="access_denied",
            message=safe_error,
            recoverability="admin_required",
            retryable=False,
            recovery_hint=(
                "Access denied for this operation. Possible causes: missing scope, "
                "missing user authorization, or tenant policy restriction. "
                "Grant the required permissions in the developer console and reinstall/republish "
                f"the application, then retry. {capability_hint}"
            ),
            next_steps=(
                f"Open the developer console and grant the required scopes for capability '{capability}'.",
                "Publish or reinstall the application after granting scopes.",
                f"Run: {status_cmd}",
                f"Retry after authorization: {login_cmd}",
            ),
            identity=identity,
            capability=capability,
            blocks_approval=True,
        )

    if any(kw in lowered for kw in ("missing scope", "insufficient scope", "scope", "permission")):
        status_cmd = _auth_cmd(binary, profile, "status")
        return ActionFailure(
            failure_class="authorization",
            failure_code="missing_scope",
            message=safe_error,
            recoverability="admin_required",
            retryable=False,
            recovery_hint=(
                f"Missing required scope to execute '{spec.name}'. "
                "Grant the missing scope in the developer console and re-publish the app."
            ),
            next_steps=(
                f"Grant the required scope for '{capability}' in the developer console.",
                "Re-publish or reinstall the application.",
                f"Run: {status_cmd}",
            ),
            identity=identity,
            capability=capability,
            blocks_approval=True,
        )

    if any(kw in lowered for kw in ("unauthorized", "unauthenticated", "invalid token", "token expired", "not logged in")):
        login_cmd = _auth_cmd(binary, profile, "login")
        status_cmd = _auth_cmd(binary, profile, "status")
        return ActionFailure(
            failure_class="authentication",
            failure_code="auth_expired",
            message=safe_error,
            recoverability="user_action",
            retryable=True,
            recovery_hint=(
                "CLI identity is not authenticated or token has expired. "
                f"Run: {login_cmd}"
            ),
            next_steps=(
                f"Authenticate: {login_cmd}",
                f"Verify: {status_cmd}",
            ),
            identity=identity,
            capability=capability,
            blocks_approval=False,
        )

    if any(kw in lowered for kw in ("rate limit", "too many requests", "429")):
        return ActionFailure(
            failure_class="rate_limit",
            failure_code="rate_limited",
            message=safe_error,
            recoverability="retryable",
            retryable=True,
            recovery_hint="Rate limit reached. Wait a moment and retry.",
            identity=identity,
            capability=capability,
            blocks_approval=False,
        )

    if any(kw in lowered for kw in ("timeout", "timed out")):
        return ActionFailure(
            failure_class="timeout",
            failure_code="timeout",
            message=safe_error,
            recoverability="retryable",
            retryable=True,
            recovery_hint="Request timed out. Retry after a moment.",
            identity=identity,
            capability=capability,
            blocks_approval=False,
        )

    return ActionFailure(
        failure_class="unknown",
        failure_code="action_failed",
        message=safe_error,
        recoverability="retryable",
        retryable=True,
        recovery_hint=f"Platform action failed: {safe_error}",
        identity=identity,
        capability=capability,
        blocks_approval=False,
    )


def _classify_problem_error(
    error_obj: Mapping[str, Any],
    spec: ActionSpec,
    fallback_message: str,
    *,
    binary: str,
    profile: str,
    identity: str,
) -> ActionFailure:
    """Classify lark-cli Problem wire errors."""
    capability = spec.capability or spec.name
    err_type = str(error_obj.get("type") or "")
    subtype = str(error_obj.get("subtype") or "")
    message = str(error_obj.get("message") or error_obj.get("hint") or fallback_message or "")
    hint = str(error_obj.get("hint") or "")
    retryable = bool(error_obj.get("retryable", False))
    console_url = str(error_obj.get("console_url") or "")
    err_identity = str(error_obj.get("identity") or identity)
    missing_scopes = tuple(str(s) for s in (error_obj.get("missing_scopes") or []) if s)
    requested_scopes = tuple(str(s) for s in (error_obj.get("requested_scopes") or []) if s)
    granted_scopes = tuple(str(s) for s in (error_obj.get("granted_scopes") or []) if s)
    log_id = str(error_obj.get("log_id") or "")
    safe_message = redact_sensitive_text(message, force=True)

    status_cmd = _auth_cmd(binary, profile, "status")
    login_cmd = _auth_cmd(binary, profile, "login")

    if err_type == "authorization":
        failure_code = subtype or "access_denied"
        is_scope = subtype in ("missing_scope", "insufficient_scope") or bool(missing_scopes)
        recovery_hint_parts = [safe_message]
        if hint:
            recovery_hint_parts.append(hint)
        if missing_scopes:
            recovery_hint_parts.append(
                f"Missing scopes: {', '.join(missing_scopes)}. "
                "Grant these in the developer console, then re-publish/reinstall the app."
            )
        elif is_scope:
            recovery_hint_parts.append(
                "A required permission scope is missing. "
                "Grant the scope in the developer console and re-publish the app."
            )
        if console_url:
            recovery_hint_parts.append(f"Developer console: {console_url}")
        if log_id:
            recovery_hint_parts.append(f"Log ID: {log_id}")
        next_steps: tuple[str, ...] = (
            *(
                (f"Grant missing scopes {list(missing_scopes)} in the developer console.",)
                if missing_scopes else (f"Grant the required scope for '{capability}' in the developer console.",)
            ),
            "Re-publish or reinstall the application after granting scopes.",
            f"Verify authorization: {status_cmd}",
        )
        if console_url:
            next_steps = (f"Open: {console_url}",) + next_steps
        return ActionFailure(
            failure_class="authorization",
            failure_code=failure_code,
            message=safe_message,
            recoverability="admin_required",
            retryable=False,
            recovery_hint=" ".join(recovery_hint_parts),
            next_steps=next_steps,
            missing_scopes=missing_scopes,
            requested_scopes=requested_scopes,
            granted_scopes=granted_scopes,
            identity=err_identity,
            console_url=console_url,
            capability=capability,
            blocks_approval=True,
            raw=dict(error_obj),
        )

    if err_type == "authentication":
        return ActionFailure(
            failure_class="authentication",
            failure_code=subtype or "unauthenticated",
            message=safe_message,
            recoverability="user_action",
            retryable=retryable,
            recovery_hint=(hint or f"Authentication failed. Run: {login_cmd}"),
            next_steps=(
                f"Authenticate: {login_cmd}",
                f"Verify: {status_cmd}",
            ),
            identity=err_identity,
            capability=capability,
            blocks_approval=False,
            raw=dict(error_obj),
        )

    if err_type == "api":
        code = int(error_obj.get("code") or 0)
        if code == 429 or "rate" in safe_message.lower():
            return ActionFailure(
                failure_class="rate_limit",
                failure_code="rate_limited",
                message=safe_message,
                recoverability="retryable",
                retryable=True,
                recovery_hint="Rate limit reached. Wait a moment and retry.",
                identity=err_identity,
                capability=capability,
                blocks_approval=False,
                raw=dict(error_obj),
            )
        return ActionFailure(
            failure_class="api_error",
            failure_code=subtype or f"api_{code}" if code else "api_error",
            message=safe_message,
            recoverability="retryable" if retryable else "user_action",
            retryable=retryable,
            recovery_hint=hint or safe_message,
            identity=err_identity,
            capability=capability,
            blocks_approval=False,
            raw=dict(error_obj),
        )

    return ActionFailure(
        failure_class=err_type or "unknown",
        failure_code=subtype or "typed_error",
        message=safe_message,
        recoverability="retryable",
        retryable=retryable,
        recovery_hint=hint or safe_message,
        identity=err_identity,
        capability=capability,
        blocks_approval=False,
        raw=dict(error_obj),
    )


def _auth_cmd(binary: str, profile: str, subcmd: str) -> str:
    if not binary:
        return f"<cli> auth {subcmd} --json"
    parts = [binary]
    if profile:
        parts.extend(["--profile", profile])
    parts.extend(["auth", subcmd, "--json"])
    return " ".join(parts)
