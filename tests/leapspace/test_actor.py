"""Tests for LeapAppActor's polling helpers (wait_for_window / snapshot_tree)."""

import pytest

pytest.importorskip("cua_sandbox")  # leapspace dependency group only

from leapspace.app_space.actor import ActionResult, LeapAppActor


def make_actor() -> LeapAppActor:
    # __init__ only stores the sandbox; the helpers under test never touch it
    # because list_windows / get_window_state are stubbed per test.
    return LeapAppActor(sandbox=object())


@pytest.mark.asyncio
async def test_wait_for_window_finds_matching_title():
    actor = make_actor()
    target = {"pid": 1, "window_id": 2, "title": "LeapChat (1)"}

    async def list_windows():
        return ActionResult(ok=True, via="sdk", data={"windows": [target]})

    actor.list_windows = list_windows
    window = await actor.wait_for_window("LeapChat")
    assert window is target


@pytest.mark.asyncio
async def test_wait_for_window_retries_until_match():
    actor = make_actor()
    calls = 0

    async def list_windows():
        nonlocal calls
        calls += 1
        if calls < 3:
            return ActionResult(ok=True, via="sdk", data={"windows": []})
        return ActionResult(
            ok=True, via="sdk",
            data={"windows": [{"pid": 1, "window_id": 2, "title": "LeapChat"}]},
        )

    actor.list_windows = list_windows
    window = await actor.wait_for_window("LeapChat", poll_s=0.01)
    assert window["title"] == "LeapChat"
    assert calls == 3


@pytest.mark.asyncio
async def test_wait_for_window_accepts_bare_list_payload():
    actor = make_actor()

    async def list_windows():
        return ActionResult(
            ok=True, via="sdk",
            data=[{"pid": 1, "window_id": 2, "title": "LeapChat"}],
        )

    actor.list_windows = list_windows
    assert (await actor.wait_for_window("Leap"))["pid"] == 1


@pytest.mark.asyncio
async def test_wait_for_window_times_out():
    actor = make_actor()

    async def list_windows():
        return ActionResult(ok=True, via="sdk", data={"windows": []})

    actor.list_windows = list_windows
    with pytest.raises(RuntimeError, match="did not appear"):
        await actor.wait_for_window("Nope", timeout_s=0.05, poll_s=0.01)


@pytest.mark.asyncio
async def test_snapshot_tree_returns_tree_markdown():
    actor = make_actor()

    async def get_window_state(pid, window_id):
        assert (pid, window_id) == (1, 2)
        return ActionResult(ok=True, via="mcp", data={"tree_markdown": "- [0] window"})

    actor.get_window_state = get_window_state
    assert await actor.snapshot_tree(1, 2) == "- [0] window"
