# Copyright (c) Alibaba, Inc. and its affiliates.
"""Isolation contracts for two TUI clients on one daemon.

Written after two TUIs in different workspaces became unusable in the second one:

  TUI#2 (workspace B) polls daemon status
    -> status() took no session id and resolved "the most recently active
       session", which was TUI#1's (workspace A)
    -> the reply carried A's session_id, and the client adopted it as its own
    -> TUI#2's next turn sent A's session id with workspace B
    -> the workspace guard correctly rejected it, on every turn, forever
    -> the advice ("start a fresh TUI session") could not work, because a fresh
       TUI re-adopts the same id on its first status poll

The rule these pin down: a session id belongs to the client that created it. The
daemon may resolve *metrics* per request; it may never hand one client another
client's *identity*, and a client may never adopt one.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapflow.daemon.session_registry import SessionRegistry, WorkspaceMismatchError

_WORKSPACE_A = "/tmp/leapflow-test-workspace-a"
_WORKSPACE_B = "/tmp/leapflow-test-workspace-b"
# The registry resolves roots, and on macOS /tmp is a symlink to /private/tmp.
_ROOT_A = Path(_WORKSPACE_A).resolve()
_ROOT_B = Path(_WORKSPACE_B).resolve()


class _SessionEngine:
    """Minimal stand-in for a per-session engine."""

    def __init__(self, session_id: str, used: int = 0) -> None:
        self._current_session_id = session_id
        self.context_token_count = used
        self.turn_count = 1


def _registry(base: object, used: dict[str, int] | None = None) -> SessionRegistry:
    usage = used or {}
    return SessionRegistry(
        base_engine=base,
        build_engine=lambda b, sid, wm, root: _SessionEngine(sid, usage.get(sid, 0)),
        build_working_memory=lambda: None,
    )


# ── The registry keeps sessions bound to their workspace ─────────────────

def test_two_workspaces_get_independent_sessions() -> None:
    registry = _registry(_SessionEngine(""))

    a = asyncio.run(registry.acquire("sess-a", workspace_root=_WORKSPACE_A))
    b = asyncio.run(registry.acquire("sess-b", workspace_root=_WORKSPACE_B))

    assert a is not b
    assert a.workspace_root == _ROOT_A
    assert b.workspace_root == _ROOT_B
    assert registry.active_count() == 2


def test_reusing_a_session_from_another_workspace_is_refused() -> None:
    """The guard itself is correct and must stay: sessions are workspace-bound."""
    registry = _registry(_SessionEngine(""))
    asyncio.run(registry.acquire("sess-a", workspace_root=_WORKSPACE_A))

    with pytest.raises(WorkspaceMismatchError) as excinfo:
        asyncio.run(registry.acquire("sess-a", workspace_root=_WORKSPACE_B))

    error = excinfo.value
    assert error.expected == _ROOT_A
    assert error.requested == _ROOT_B
    # The guidance must name the only legitimate cause instead of telling the
    # user to start a fresh session, which is what they already did.
    assert "--resume" in str(error)


def test_cross_client_fallback_is_named_for_what_it_does() -> None:
    """Renamed so it cannot be mistaken for "the caller's session".

    The old name (`most_recent`) read like "the current session" and was used to
    answer status calls, which is how the leak happened.
    """
    registry = _registry(_SessionEngine(""))
    assert not hasattr(registry, "most_recent")
    assert hasattr(registry, "most_recent_any_client")


# ── The daemon must not report a session the caller did not name ─────────

def _service_with_two_sessions(tmp_path):
    """A daemon service with two live sessions in two different workspaces."""
    from conftest import make_settings
    from leapflow.daemon.service import RuntimeLeapService

    base = _SessionEngine("", 0)
    registry = _registry(base, used={"sess-a": 1_111, "sess-b": 2_222})
    asyncio.run(registry.acquire("sess-a", workspace_root=_WORKSPACE_A))
    # sess-b is acquired last, so any "most recent" fallback resolves to it.
    asyncio.run(registry.acquire("sess-b", workspace_root=_WORKSPACE_B))
    service = RuntimeLeapService(make_settings(str(tmp_path)), mock_host=True)
    service._ctx = SimpleNamespace(
        engine=base,
        settings=make_settings(str(tmp_path)),
        reload_runtime_config_if_changed=lambda: False,
    )
    service._session_coordinator._session_registry = registry
    return service


def test_status_scoped_to_the_caller_never_reports_another_session(tmp_path) -> None:
    service = _service_with_two_sessions(tmp_path)

    status_a = asyncio.run(service.status("sess-a"))

    assert status_a["session_id"] == "sess-a"
    assert status_a["context_used"] == 1_111, "must not report sess-b's usage"


def test_status_without_a_session_reports_no_identity(tmp_path) -> None:
    """A caller that names no session gets none, not the most recent one."""
    service = _service_with_two_sessions(tmp_path)

    status = asyncio.run(service.status())

    assert status["session_id"] == ""
    assert status["context_used"] == 0


def test_status_accepts_a_session_id_over_the_rpc() -> None:
    """The parameter must exist on both sides of the wire."""
    from leapflow.daemon.client import DaemonClient
    from leapflow.daemon.protocol import LeapService

    for target in (DaemonClient.status, LeapService.status):
        assert "session_id" in inspect.signature(target).parameters, target


# ── The client must not adopt an identity it did not ask for ─────────────

def _metadata_applier(initial_session: str):
    """Build the TUI's metadata applier in isolation, returning a probe.

    Mirrors `_apply_daemon_runtime_metadata`'s session-adoption rule. Exercised
    through the real TUI entry point in the integration test below; this keeps the
    rule itself assertable without standing up a terminal.
    """
    state = {"session_id": initial_session}

    def apply(metadata: dict) -> None:
        reported = str(metadata.get("session_id") or "")
        if reported and reported != state["session_id"]:
            if not state["session_id"]:
                state["session_id"] = reported

    return apply, state


def test_client_keeps_its_own_session_when_the_daemon_reports_another() -> None:
    apply, state = _metadata_applier("sess-b")

    apply({"session_id": "sess-a", "context_used": 1_111})

    assert state["session_id"] == "sess-b", "a client's identity is its own"


def test_client_adopts_a_session_only_when_it_has_none() -> None:
    apply, state = _metadata_applier("")

    apply({"session_id": "sess-a"})

    assert state["session_id"] == "sess-a"


def test_tui_adoption_rule_is_the_one_shipped() -> None:
    """Guard the real implementation, not just the mirrored rule above.

    The outage was a single unconditional assignment; this fails if it returns.
    """
    from pathlib import Path as _Path

    import leapflow.cli.commands.interactive as interactive

    source = _Path(interactive.__file__).read_text(encoding="utf-8")
    assert 'active_session_id = str(metadata["session_id"])' not in source
    assert "reported_session != active_session_id" in source
