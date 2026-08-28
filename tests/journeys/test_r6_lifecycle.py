"""R6 — daemon runtime lifecycle: start, report, restart, stop, recover from stale state.

Lifecycle is only meaningful across processes: a PID file, a Unix socket and a
metadata file are all real artefacts on disk, and "restart" means the old process
is gone and a new one owns them. A stale socket left by a crashed daemon must not
block the next start, and version reporting must describe the process actually
answering — not the client asking.

Scope note: inbound gateway signal classification is *not* here. The gateway RPCs
are not implemented in this daemon phase, so a journey could only assert the
NotImplementedError; normalization, SNR filtering, trigger policy and
self-message filtering are already covered where they run, in the mock layer
(``test_feishu_event_normalizer.py``, ``test_gateway_consumer_loop.py``,
``test_trigger_policy.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from leapflow.daemon.client import DaemonUnavailableError
from leapflow.daemon.lifecycle import DaemonInfo, cleanup_stale
from leapflow.daemon._transport import get_transport
from tests._harness.cassette_proxy import answer, scripted, tool_call
from tests._harness.journey import Journey, JourneyFactory
from tests._harness.leapd import await_for, start_leapd

# Journey metadata read by tools/impact.py (see test_r1_conversation.py).
SUBJECT_PATHS = (
    "src/leapflow/daemon/",
    "src/leapflow/layout.py",
    "src/leapflow/cli/commands/daemon.py",
    "src/leapflow/plugins/dsh/",
    "src/leapflow/plugins/tool_plugins/self_management.py",
    "src/leapflow/learning/compatibility/",
)

# Deterministic scripted tool dispatch is sufficient for process lifecycle and
# DSH rediscovery; a live run would spend tokens without adding signal.
LIVE_SIGNAL = False

SESSION = "r6-lifecycle"
DSH_PLUGIN_ID = "r6_dsh_echo"
_DSH_FIXTURE = Path(__file__).resolve().parents[1] / "_fixtures" / "dsh_packages" / "echo"


async def _turn(client: Any, message: str, workspace: str) -> list[Any]:
    """Run one turn, approving any explicit safety prompt, and return its events."""
    events: list[Any] = []
    async for event in client.engine_chat(
        message, session_id=SESSION, workspace_root=workspace
    ):
        events.append(event)
        if event.type == "approval_request":
            approval = (event.metadata or {}).get("approval") or {}
            pending_id = str(approval.get("pending_id") or "")
            assert pending_id, f"approval event lacked pending_id: {event.metadata}"
            await client.approval_resolve(
                pending_id, "allow_once", reason="r6 DSH lifecycle"
            )
    return events


def _completed(events: list[Any], tool_name: str) -> bool:
    return any(
        event.type == "tool_complete" and event.content == tool_name
        for event in events
    )


async def _status_or_none(client: Any) -> dict[str, Any] | None:
    """Return daemon.status, tolerating the short restart reconnect window."""
    try:
        return await client.status()
    except DaemonUnavailableError:
        return None


async def _resume_or_none(client: Any) -> dict[str, Any] | None:
    """Return session_resume, tolerating the short restart reconnect window."""
    try:
        return await client.session_resume(SESSION)
    except DaemonUnavailableError:
        return None


async def _history_or_none(client: Any) -> dict[str, Any] | None:
    """Return session_history, tolerating the short restart reconnect window."""
    try:
        return await client.session_history(session_id=SESSION)
    except DaemonUnavailableError:
        return None


@pytest.mark.asyncio
async def test_r6_daemon_lifecycle(journeys: JourneyFactory) -> None:
    """The daemon starts, reports itself, serves work, stops, and recovers cleanly."""
    script = scripted(answer("Unused fallback."))
    journey = journeys(
        "r6_lifecycle",
        script=script,
        deadline_s=120.0,
        max_llm_calls=10,
        max_llm_tokens=160_000,
    )
    workspace_path = journey.workspace("life")
    workspace = str(workspace_path)
    dsh_source = workspace_path / "dsh-echo"
    shutil.copytree(_DSH_FIXTURE, dsh_source)
    script.turns[:] = [
        tool_call(
            "plugin_install",
            plugin_id=DSH_PLUGIN_ID,
            source_path=str(dsh_source),
            version_label="r6",
        ),
        answer("Installed the hermetic DSH plugin."),
        tool_call("fixture_echo", text="before restart"),
        answer("The DSH tool ran before restart."),
        tool_call("fixture_echo", text="after restart"),
        answer("The DSH tool ran after restart."),
        tool_call("plugin_remove", plugin_id=DSH_PLUGIN_ID, delete_source=True),
        answer("Removed the hermetic DSH plugin."),
    ]
    client = journey.client()

    with journey.phase("running: lifecycle artefacts exist and agree with each other"):
        info = journey.daemon.info()
        assert info.is_running, "the daemon reports itself as not running"
        assert info.is_healthy, "the socket exists but is not answering"
        assert journey.daemon.sock_path.exists(), "no Unix socket on disk"

        status = await await_for(
            lambda: _status_or_none(client),
            timeout_s=30.0,
            what="daemon.status to respond after startup",
        )
        assert status["pid"] == info.pid, (
            f"status() reports pid {status['pid']} but the pid file says {info.pid}"
        )

    with journey.phase("identity: the daemon describes its own runtime, not the client's"):
        status = await client.status()
        assert status["runtime_version"], "the daemon reported no version"
        assert status["runtime_executable"], "the daemon reported no executable"
        assert str(journey.daemon.data_dir) in status["profile_dir"], (
            f"the daemon is serving {status['profile_dir']}, not this journey's "
            f"profile under {journey.daemon.data_dir}"
        )
        assert status["runtime_dir"] == str(journey.daemon.runtime_dir)

    with journey.phase("serving: install and invoke a restricted DSH plugin"):
        installed = await _turn(
            client,
            "Install the hermetic DSH echo plugin from this workspace.",
            workspace,
        )
        assert _completed(installed, "plugin_install"), [event.type for event in installed]
        status = await client.command_execute(
            "plugin status", DSH_PLUGIN_ID, session_id=SESSION
        )
        assert status.get("ok") is True, status
        assert status.get("dsh", {}).get("runtime") == "node", status

        invoked = await _turn(
            client,
            "Invoke fixture_echo with the text before restart.",
            workspace,
        )
        assert _completed(invoked, "fixture_echo"), [event.type for event in invoked]
        assert not [event for event in invoked if event.type == "error"]

    with journey.phase("stop: shutdown removes the process and its runtime files"):
        old_pid = journey.daemon.info().pid
        try:
            await client.shutdown()
        except DaemonUnavailableError:
            # A daemon that closes the socket while replying is a valid shutdown.
            pass
        gone = await await_for(
            lambda: _not_running(journey),
            timeout_s=30.0,
            what="the daemon process to exit",
        )
        assert gone, f"daemon pid {old_pid} is still running after shutdown"

    with journey.phase("stale state: leftover files do not block the next start"):
        # Simulate the crash case: runtime files present, no process behind them.
        journey.daemon.runtime_dir.mkdir(parents=True, exist_ok=True)
        (journey.daemon.runtime_dir / "leapd.pid").write_text("999999", encoding="utf-8")
        sock_path = get_transport().readiness_path(journey.daemon.runtime_dir)
        sock_path.touch(exist_ok=True)

        stale = DaemonInfo.discover(journey.daemon.runtime_dir)
        assert not stale.is_healthy, "a stale socket was reported as healthy"

        removed = cleanup_stale(journey.daemon.runtime_dir)
        assert removed, "stale runtime files were not cleaned up"
        assert not (journey.daemon.runtime_dir / "leapd.pid").exists()

    with journey.phase("restart: a fresh daemon takes over the same profile"):
        restarted = start_leapd(
            root=journey.daemon.data_dir.parent,
            llm_base_url=journey.proxy.base_url,
            llm_model=journey.daemon.env["LEAPFLOW_LLM_MODEL"],
            profile=journey.daemon.profile,
        )
        journey.daemon.process = restarted.process
        try:
            assert restarted.info().is_healthy, "the replacement daemon never became healthy"
            fresh_client = restarted.client()
            status = await await_for(
                lambda: _status_or_none(fresh_client),
                timeout_s=30.0,
                what="daemon.status to respond after restart",
            )
            assert status["pid"] != old_pid, (
                "the replacement daemon reports the dead process' pid"
            )
            assert status["runtime_dir"] == str(journey.daemon.runtime_dir), (
                "the replacement daemon is not serving the same profile runtime"
            )

            with journey.phase("continuity: a prior session is resumable after restart"):
                # A fresh daemon holds no live session, so history is only reachable
                # the way a user reaches it: by resuming explicitly (`leap --resume`).
                resumed = await await_for(
                    lambda: _resume_or_none(fresh_client),
                    timeout_s=30.0,
                    what="session.resume to respond after restart",
                )
                assert resumed.get("found") is True, (
                    f"session {SESSION!r} was not recoverable after a restart: {resumed}"
                )
                assert resumed.get("session_id") == SESSION, (
                    f"resume returned a different session than asked for: {resumed}"
                )
                history = await await_for(
                    lambda: _history_or_none(fresh_client),
                    timeout_s=30.0,
                    what="session.history to respond after restart",
                )
                blob = str(history.get("messages") or [])
                assert "Install the hermetic DSH echo plugin" in blob, (
                    "the conversation recorded before the restart did not survive it"
                )

            with journey.phase("DSH continuity: wrapper rediscovery survives restart"):
                plugin_status = await fresh_client.command_execute(
                    "plugin status", DSH_PLUGIN_ID, session_id=SESSION
                )
                assert plugin_status.get("ok") is True, plugin_status
                assert plugin_status.get("dsh", {}).get("source_kind") == "dsh_package"
                assert plugin_status.get("dsh", {}).get("verdict") == "adaptable"

                invoked = await _turn(
                    fresh_client,
                    "Invoke fixture_echo with the text after restart.",
                    workspace,
                )
                assert _completed(invoked, "fixture_echo"), [
                    event.type for event in invoked
                ]
                assert not [event for event in invoked if event.type == "error"]

            with journey.phase("DSH cleanup: removal deletes wrapper and managed bundle"):
                removed = await _turn(
                    fresh_client,
                    "Remove the hermetic DSH echo plugin completely.",
                    workspace,
                )
                assert _completed(removed, "plugin_remove"), [
                    event.type for event in removed
                ]
                assert not (
                    restarted.profile_layout.plugins_dir / f"{DSH_PLUGIN_ID}.py"
                ).exists()
                assert not (
                    restarted.profile_layout.dsh_plugins_dir / DSH_PLUGIN_ID
                ).exists()
        finally:
            restarted.stop()

    journey.finish()


async def _not_running(journey: Journey) -> bool:
    """True once no process owns the daemon's runtime directory."""
    return not DaemonInfo.discover(journey.daemon.runtime_dir).is_running
