# Copyright (c) Alibaba, Inc. and its affiliates.
"""Live-Qt signal bridge: a real offscreen app's ground truth reaches LeapFlow.

The strongest headless realism available without a hypervisor: a real PyQt6
``BaseLeapApp`` runs offscreen, writes its own ``state.json`` from the live
widget tree (role-aware ``elements``), and LeapFlow's *shipped*
``StateSnapshotService`` perceives it through ``LeapSpaceHostRpc``. This upgrades
the earlier bridge test from a hand-written envelope to one a real app emitted.

Runs in the conda ``leap`` env (PyQt6 present); skips where PyQt6 is absent.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
# Must be set before any QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from leapflow.perception.state_snapshot import (  # noqa: E402
    SnapshotFidelity,
    StateSnapshotService,
)
from leapspace.app_space.apps import _base as base_mod  # noqa: E402
from leapspace.app_space.apps.chat import ChatApp  # noqa: E402
from leapspace.app_space.host_rpc import LeapSpaceHostRpc  # noqa: E402


class _FakeEpisodic:
    def recent(self, limit: int = 5):
        return []

    def search_fragments(self, terms, limit: int = 1):
        return []


@pytest.fixture(scope="session")
def qapp():
    """One process-wide QApplication, mirroring test_base's proven pattern.

    Session-scoped and never torn down: constructing/destroying QApplication per
    test aborts the offscreen platform, and a live app instance is what keeps
    ``QApplication.instance()`` valid across the file when it runs standalone.
    """
    return QApplication.instance() or QApplication([])


def _make_app(state_root: Path, monkeypatch) -> ChatApp:
    """Instantiate a real ChatApp offscreen; __init__ writes state.json."""
    monkeypatch.setattr(
        base_mod,
        "get_sandbox_state_dir",
        lambda in_sandbox=True, system=None: state_root,
    )
    return ChatApp()


def _dispose(win: ChatApp, qapp: QApplication) -> None:
    """Tear one app down cleanly: stop its timer, schedule deletion, flush it.

    ChatApp starts an inbox poll ``QTimer``; leaving it live while a second
    top-level window is created aborts the offscreen platform. Stopping it and
    letting ``deleteLater`` run under ``processEvents`` keeps repeated
    instantiation stable, which is what a standalone run of this file needs.
    """
    timer = getattr(win, "_inbox_timer", None)
    if timer is not None:
        timer.stop()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_real_app_emits_role_aware_elements(qapp, tmp_path, monkeypatch):
    win = _make_app(tmp_path, monkeypatch)
    try:
        state = json.loads((tmp_path / "chat" / "state.json").read_text())
        elements = {e["name"]: e for e in state["elements"]}
        # Roles come from the live widget classes, not a declaration.
        assert elements["send_button"]["role"] == "QPushButton"
        assert elements["message_input"]["role"] == "QLineEdit"
        assert elements["contact_list"]["role"] == "QListWidget"
        # The role-aware channel and the bare name list agree on membership.
        assert set(state["interface"]) == set(elements)
    finally:
        _dispose(win, qapp)


def test_live_app_signal_reaches_leapflow_perception(qapp, tmp_path, monkeypatch):
    win = _make_app(tmp_path, monkeypatch)
    try:
        rpc = LeapSpaceHostRpc(tmp_path)
        svc = StateSnapshotService(
            rpc, _FakeEpisodic(), default_fidelity=SnapshotFidelity.FULL
        )
        svc.update_focus("chat", "LeapChat")
        snap = asyncio.run(svc.capture(SnapshotFidelity.FULL))

        assert snap.app_bundle_id == "chat"
        assert snap.ax_digest, "perception saw an empty environment"
        # The real Qt role and the bound name both flow through the summary.
        assert "QPushButton" in snap.ax_summary
        assert "send_button" in snap.ax_summary
    finally:
        _dispose(win, qapp)
