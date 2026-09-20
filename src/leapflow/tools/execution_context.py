# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-turn tool execution context for workspace-scoped safety.

Daemon-backed turns from different TUI clients may share one Python process but
must not share a project root.  The engine installs this context around each
actual tool execution so path-oriented tools resolve relative paths against the
active turn's workspace and reject accidental cross-workspace absolute paths.
"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolExecutionContext:
    """Workspace boundary for one tool execution turn."""

    workspace_root: Path
    allowed_roots: tuple[Path, ...]
    session_id: str = ""
    task_id: str = ""
    approval_bypass: bool = False
    orchestrator: Any = field(default=None, compare=False, hash=False, repr=False)

    @classmethod
    def from_strings(
        cls,
        *,
        workspace_root: str,
        allowed_roots: tuple[str, ...] = (),
        session_id: str = "",
        task_id: str = "",
        approval_bypass: bool = False,
        orchestrator: Any = None,
    ) -> "ToolExecutionContext":
        root = Path(workspace_root).expanduser().resolve()
        roots = tuple(Path(item).expanduser().resolve() for item in allowed_roots if item)
        return cls(
            workspace_root=root,
            allowed_roots=roots or (root,),
            session_id=session_id,
            task_id=task_id,
            approval_bypass=approval_bypass,
            orchestrator=orchestrator,
        )


_current_context: contextvars.ContextVar[ToolExecutionContext | None] = contextvars.ContextVar(
    "leapflow_tool_execution_context",
    default=None,
)


def current_tool_context() -> ToolExecutionContext | None:
    """Return the active tool context, if the caller installed one."""
    return _current_context.get()


def set_tool_context(ctx: ToolExecutionContext | None) -> contextvars.Token[ToolExecutionContext | None] | None:
    """Install a tool context and return a token for reset."""
    if ctx is None:
        return None
    return _current_context.set(ctx)


def reset_tool_context(token: contextvars.Token[ToolExecutionContext | None] | None) -> None:
    """Reset a previously installed tool context."""
    if token is not None:
        _current_context.reset(token)


def resolve_workspace_path(value: Any, *, default: str = ".") -> Path:
    """Resolve a tool path under the active workspace when it is relative."""
    raw = str(value if value not in (None, "") else default)
    path = Path(raw).expanduser()
    ctx = current_tool_context()
    if ctx is not None and not path.is_absolute():
        path = ctx.workspace_root / path
    return path.resolve()


def is_within_allowed_roots(path: Path, ctx: ToolExecutionContext | None = None) -> bool:
    """Return whether ``path`` is inside the active allowed roots."""
    context = current_tool_context() if ctx is None else ctx
    if context is None:
        return True
    resolved = path.expanduser().resolve()
    for root in context.allowed_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def leapflow_managed_hint(path: Path) -> str:
    """Return a redirect hint when ``path`` is LeapFlow's own managed state.

    A refusal that only says "outside the workspace" leaves the model to guess
    another path, which is how a config change turns into a sequence of blocked
    probes. Classification comes from the layout descriptor rather than string
    matching, so it follows the path tree instead of duplicating it.

    Public because the shell gate refuses the same targets through a different
    entry point; a hint that only the file tools emit would leave the shell path
    telling the model nothing about what to use instead.
    """
    try:
        from leapflow.config import get_settings

        layout = getattr(get_settings(), "layout", None)
        if layout is None:
            return ""
        descriptor = layout.describe_path(path)
    except Exception:  # noqa: BLE001 - a hint must never break the refusal itself
        return ""

    category = str(getattr(descriptor, "category", "") or "")
    if category in {"config", "mcp_config", "workspace_manifest"}:
        return (
            " This is LeapFlow's own configuration: use the config_list / config_get / "
            "config_set tools instead of reading or editing these files."
        )
    if category == "secret_vault":
        return (
            " This is LeapFlow's credential vault and is never readable as a file: "
            "set credentials with config_set (e.g. key='llm.api_key')."
        )
    return ""


def workspace_scope_refusal(path: Path, *, operation: str) -> dict[str, Any] | None:
    """Build the refusal for a path outside the workspace, or None if inside.

    Internal to this module's gate. Callers must use ``require_workspace_access``
    instead: this function only *describes* a refusal, it never asks anyone, and
    eleven of twelve call sites once returned it directly — telling the user
    "Approval is required" while never opening a prompt.
    """
    ctx = current_tool_context()
    if ctx is None or is_within_allowed_roots(path, ctx):
        return None
    hint = leapflow_managed_hint(path)
    return {
        "ok": False,
        "error": (
            f"{operation} path is outside the active workspace. "
            f"Resolved path: {path}; workspace root: {ctx.workspace_root}. "
            "Approval is required to access paths outside the workspace."
            + hint
        ),
        "error_type": "outside_workspace",
        "retryable": False,
        "workspace_root": str(ctx.workspace_root),
        "allowed_roots": [str(root) for root in ctx.allowed_roots],
        "resolved_path": str(path),
        "session_id": ctx.session_id,
    }


def _active_orchestrator() -> Any:
    """Return the approval orchestrator for this turn, if one is reachable.

    The context carries it (the engine copies it in per turn). The module-level
    shell gate is a lazy fallback for contexts built without one; the import is
    deferred because ``shell_tools`` imports this module at load time.
    """
    ctx = current_tool_context()
    orchestrator = getattr(ctx, "orchestrator", None) if ctx is not None else None
    if orchestrator is not None:
        return orchestrator
    try:
        from leapflow.tools.shell_tools import _approval_gate

        return _approval_gate
    except ImportError:  # pragma: no cover - defensive
        return None



async def require_workspace_access(
    path: Path,
    *,
    operation: str,
    effect: str = "read",
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Gate access to *path*. Returns None when permitted, else a refusal dict.

    The whole sequence lives here — boundary check, policy, approval, refusal —
    because it used to be spelled out per call site and only the shell path ever
    got it right. The other eleven returned the refusal directly, so
    ``file_list``/``code_search`` refused in 39ms with a message claiming approval
    was required, and ignored a session-wide "Allow ALL" that the shell honoured.

    *effect* is the caller's real effect (``read`` / ``write`` / ``execute``); it
    drives the risk tier, so a listing is not weighed like a write.

    Fails closed: no orchestrator, a gate that cannot evaluate, or an exception
    all refuse. A broken gate must not become an open door.
    """
    refusal = workspace_scope_refusal(path, operation=operation)
    if refusal is None:
        return None

    orchestrator = _active_orchestrator()
    evaluate = getattr(orchestrator, "evaluate", None)
    if not callable(evaluate):
        logger.debug(
            "workspace escape refused for %s: no approval orchestrator in context", operation,
        )
        return refusal

    from leapflow.security.actions import ActionDescriptor

    action = ActionDescriptor.workspace_escape(
        str(path),
        operation=operation,
        effect=effect,
        detail=detail,
        metadata={
            "workspace_root": str(refusal.get("workspace_root", "")),
            **(metadata or {}),
        },
    )
    try:
        result = await evaluate(action)
    except Exception:  # noqa: BLE001 - a broken gate must not become an open door
        logger.warning(
            "workspace escape approval failed for %s; refusing", operation, exc_info=True,
        )
        return refusal
    if getattr(result, "approved", False):
        return None
    denial = str(getattr(result, "denial_message", "") or "")
    if denial:
        # Surface the gate's own wording: it states that the user did not consent
        # and must not be worked around, which a generic scope error does not.
        return {**refusal, "error": denial}
    return refusal
