# Copyright (c) Alibaba, Inc. and its affiliates.
"""Handler for ``/session`` slash commands.

Provides session lifecycle management: list, archive, pin/unpin, hide/unhide.
Delegates to :class:`DuckDBConversationStore` for persistence.
"""
from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from leapflow.cli.context import Context

logger = logging.getLogger(__name__)


def _get_store(ctx: "Context") -> Any | None:
    """Resolve the conversation store from context."""
    store = getattr(ctx, "_conversation_store", None)
    if store is None:
        engine = getattr(ctx, "engine", None)
        store = getattr(engine, "_conversation_store", None) if engine else None
    return store


def _session_not_found(session_id: str) -> Dict[str, Any]:
    return {"ok": False, "message": f"Session not found: {session_id}"}


def build_session_payload(ctx: "Context", args: str = "") -> Dict[str, Any]:
    """Handle ``/session`` commands and return a serializable result payload.

    Subcommands:
    - ``list [--all|--hidden|--archived]`` — list sessions
    - ``archive <id>`` — archive a session
    - ``pin <id>`` — pin a session
    - ``unpin <id>`` — unpin a session
    - ``hide <id>`` — hide a session
    - ``unhide <id>`` — unhide a session
    """
    store = _get_store(ctx)
    if store is None:
        return {"ok": False, "message": "Conversation store is not available."}

    parts = args.strip().split(None, 1)
    verb = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if verb == "list" or not args.strip():
        return _handle_list(store, rest)
    if verb == "archive":
        return _handle_archive(store, rest)
    if verb == "pin":
        return _handle_pin(store, rest)
    if verb == "unpin":
        return _handle_unpin(store, rest)
    if verb == "hide":
        return _handle_hide(store, rest)
    if verb == "unhide":
        return _handle_unhide(store, rest)

    return {"ok": False, "message": f"Unknown session subcommand: {verb}. Use list, archive, pin, unpin, hide, or unhide."}


def _handle_list(store: Any, args: str) -> Dict[str, Any]:
    """List sessions with optional filters."""
    include_hidden = "--hidden" in args or "--all" in args
    include_archived = "--archived" in args or "--all" in args

    try:
        sessions = store.list_sessions(
            limit=30,
            active_only=not include_archived,
            include_hidden=include_hidden,
            include_archived=include_archived,
        )
    except TypeError:
        # Fallback for stores that don't support the new params yet
        sessions = store.list_sessions(limit=30, active_only=not include_archived)

    if not sessions:
        return {"ok": True, "message": "No sessions found."}

    lines = ["Sessions:"]
    for s in sessions:
        flags: list[str] = []
        if getattr(s, "pinned", False):
            flags.append("\U0001f4cc")  # 📌
        if getattr(s, "hidden", False):
            flags.append("\U0001f441\ufe0f\u200d\U0001f5e8\ufe0f")  # eye-hidden
        if not getattr(s, "is_active", True):
            flags.append("\U0001f4e6")  # 📦 archived
        flag_str = " ".join(flags)
        ts = datetime.datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
        title = s.title or "(untitled)"
        sid_short = s.session_id[:8]
        lines.append(f"  {flag_str} {sid_short}  {ts}  {title}  [{s.message_count} msgs]")

    return {"ok": True, "message": "\n".join(lines)}


def _handle_archive(store: Any, session_id: str) -> Dict[str, Any]:
    """Archive a session."""
    if not session_id:
        return {"ok": False, "message": "Usage: /session archive <session_id>"}
    session = store.get_session(session_id)
    if session is None:
        return _session_not_found(session_id)
    store.archive_session(session_id)
    return {"ok": True, "message": f"Session {session_id[:8]} archived."}


def _handle_pin(store: Any, session_id: str) -> Dict[str, Any]:
    """Pin a session."""
    if not session_id:
        return {"ok": False, "message": "Usage: /session pin <session_id>"}
    session = store.get_session(session_id)
    if session is None:
        return _session_not_found(session_id)
    store.pin_session(session_id)
    return {"ok": True, "message": f"Session {session_id[:8]} pinned."}


def _handle_unpin(store: Any, session_id: str) -> Dict[str, Any]:
    """Unpin a session."""
    if not session_id:
        return {"ok": False, "message": "Usage: /session unpin <session_id>"}
    session = store.get_session(session_id)
    if session is None:
        return _session_not_found(session_id)
    store.unpin_session(session_id)
    return {"ok": True, "message": f"Session {session_id[:8]} unpinned."}


def _handle_hide(store: Any, session_id: str) -> Dict[str, Any]:
    """Hide a session."""
    if not session_id:
        return {"ok": False, "message": "Usage: /session hide <session_id>"}
    session = store.get_session(session_id)
    if session is None:
        return _session_not_found(session_id)
    store.hide_session(session_id)
    return {"ok": True, "message": f"Session {session_id[:8]} hidden."}


def _handle_unhide(store: Any, session_id: str) -> Dict[str, Any]:
    """Unhide a session."""
    if not session_id:
        return {"ok": False, "message": "Usage: /session unhide <session_id>"}
    session = store.get_session(session_id)
    if session is None:
        return _session_not_found(session_id)
    store.unhide_session(session_id)
    return {"ok": True, "message": f"Session {session_id[:8]} unhidden."}
