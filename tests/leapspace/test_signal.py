"""Hermetic unit tests for LeapSignal's record_* sentinel protocol."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("cua_sandbox")  # leapspace extra only (signal pulls in app_space.utils)

import leapspace.app_space.signal as signal_module
from leapspace.app_space.signal import (
    RECORD_DONE_FILE,
    RECORD_START_FILE,
    RECORD_STOP_FILE,
    LeapSignal,
)


def test_constructor_creates_missing_signal_dir(tmp_path):
    signal_dir = tmp_path / "nested" / "signal"
    LeapSignal(signal_dir)
    assert signal_dir.is_dir()


def test_await_stop_returns_immediately_when_stop_file_exists(tmp_path):
    (tmp_path / RECORD_STOP_FILE).write_text('{"stop": true}')
    asyncio.run(LeapSignal(tmp_path)._await_stop())


def test_await_stop_polls_until_stop_file_lands(tmp_path, monkeypatch):
    monkeypatch.setattr(signal_module, "SENTINEL_POLL_S", 0.01)
    leap = LeapSignal(tmp_path)

    async def touch_later():
        await asyncio.sleep(0.05)
        (tmp_path / RECORD_STOP_FILE).write_text('{"stop": true}')

    async def scenario():
        # a serial wait would deadlock: _await_stop only returns once the
        # concurrent touch lands, proving the poll loop rechecks the file
        await asyncio.gather(touch_later(), leap._await_stop())

    asyncio.run(scenario())


def test_run_failure_writes_diagnosable_done(tmp_path, monkeypatch):
    async def boom(self):
        raise RuntimeError("daemon exploded")

    monkeypatch.setattr(LeapSignal, "_start", boom)
    rc = asyncio.run(LeapSignal(tmp_path).run())
    assert rc == 1
    done = json.loads((tmp_path / RECORD_DONE_FILE).read_text())
    assert done["ok"] is False
    assert "daemon exploded" in done["error"]


def test_run_success_writes_start_then_done(tmp_path, monkeypatch):
    async def fake_start(self):
        self._write(RECORD_START_FILE, {"trajectory_id": "t-1"})

    async def fake_stop(self):
        return {"trajectory_id": "t-1", "steps": 3, "episodes": 1}

    monkeypatch.setattr(LeapSignal, "_start", fake_start)
    monkeypatch.setattr(LeapSignal, "_stop", fake_stop)
    # stop file pre-created: run() must pass through the real _await_stop
    (tmp_path / RECORD_STOP_FILE).write_text('{"stop": true}')

    rc = asyncio.run(LeapSignal(tmp_path, goal="g").run())

    assert rc == 0
    assert json.loads((tmp_path / RECORD_START_FILE).read_text()) == {
        "trajectory_id": "t-1"
    }
    assert json.loads((tmp_path / RECORD_DONE_FILE).read_text()) == {
        "trajectory_id": "t-1",
        "steps": 3,
        "episodes": 1,
        "ok": True,
    }


def test_fake_intent_inferrer_echoes_action_names():
    episodes = [
        SimpleNamespace(
            semantic_actions=[
                SimpleNamespace(action_name="fs.change"),
                SimpleNamespace(action_name="ui.type"),
            ]
        ),
        SimpleNamespace(semantic_actions=[]),
    ]
    results = asyncio.run(signal_module.FakeIntentInferrer().infer_batch(episodes))
    assert [r.goal for r in results] == ["fake:fs.change | ui.type", "fake:"]
    assert all(r.confidence == 1.0 for r in results)


def test_fake_intent_inferrer_handles_empty_batch():
    assert asyncio.run(signal_module.FakeIntentInferrer().infer_batch([])) == []


def test_fake_intent_inferrer_infers_single_episode():
    episode = SimpleNamespace(
        semantic_actions=[SimpleNamespace(action_name="app.focus_change")]
    )
    result = asyncio.run(signal_module.FakeIntentInferrer().infer(episode))
    assert result.goal == "fake:app.focus_change"
