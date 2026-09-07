"""Tests for LeapAppActor's routing policy and polling helpers."""

import pytest

pytest.importorskip("cua_sandbox")  # leapspace extra only

from cua_sandbox.interfaces.shell import CommandResult

from leapspace.app_space.actor import (
    ActionResult,
    LeapAppActor,
    element_center,
    find_ax_element,
)
from leapspace.app_space.utils import get_image_venv_python

VENV_PYTHON = get_image_venv_python("linux")
SYSTEM_PYTHON = "/usr/bin/python3"
STAGE_DIR = "/tmp/leapspace/.actor"


def make_actor() -> LeapAppActor:
    # __init__ only stores the sandbox; the helpers under test never touch it
    # because list_windows / get_window_state are stubbed per test.
    return LeapAppActor(sandbox=object())


class FakeKeyboard:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.pressed: list[object] = []
        self.fail = False

    async def type(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("kb down")
        self.typed.append(text)

    async def keypress(self, key: object) -> None:
        if self.fail:
            raise RuntimeError("kb down")
        self.pressed.append(key)


class FakeMouse:
    def __init__(self) -> None:
        self.clicks: list[tuple] = []
        self.scrolls: list[tuple] = []

    async def click(self, x: int, y: int, button: str = "left") -> None:
        self.clicks.append((x, y, button))

    async def scroll(self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0) -> None:
        self.scrolls.append((x, y, scroll_x, scroll_y))


class FakeSandbox:
    def __init__(self) -> None:
        self.keyboard = FakeKeyboard()
        self.mouse = FakeMouse()


def make_routing_actor() -> tuple[LeapAppActor, FakeSandbox, list[tuple[str, dict]]]:
    actor = LeapAppActor(sandbox=FakeSandbox())
    mcp_calls: list[tuple[str, dict]] = []

    async def call_tool(name: str, args: dict) -> ActionResult:
        mcp_calls.append((name, args))
        return ActionResult(ok=True, via="mcp", data={"echo": name})

    actor._call_mcp_tool = call_tool  # type: ignore[method-assign]
    return actor, actor.sandbox, mcp_calls  # type: ignore[return-value]


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
    assert (await actor.wait_for_window("LeapChat"))["pid"] == 1


@pytest.mark.asyncio
async def test_wait_for_window_rejects_similar_title():
    actor = make_actor()

    async def list_windows():
        return ActionResult(
            ok=True, via="sdk",
            data={"windows": [{"pid": 1, "window_id": 2, "title": "LeapChat Pro"}]},
        )

    actor.list_windows = list_windows
    with pytest.raises(RuntimeError, match="did not appear"):
        await actor.wait_for_window("LeapChat", timeout_s=0.05, poll_s=0.01)


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


# ── routing policy: SDK first, driver fallback, element addressing MCP-only ──


@pytest.mark.asyncio
async def test_press_key_goes_sdk_first():
    actor, sandbox, mcp_calls = make_routing_actor()
    result = await actor.press_key("down")
    assert result.via == "sdk"
    assert sandbox.keyboard.pressed == ["down"]
    assert mcp_calls == []


@pytest.mark.asyncio
async def test_press_key_falls_back_to_mcp_when_sdk_fails():
    actor, sandbox, mcp_calls = make_routing_actor()
    sandbox.keyboard.fail = True
    result = await actor.press_key("down")
    assert result.via == "mcp"
    assert mcp_calls[0][0] == "press_key"
    assert mcp_calls[0][1]["key"] == "down"


@pytest.mark.asyncio
async def test_press_key_raises_when_both_paths_fail():
    actor, sandbox, _ = make_routing_actor()

    async def failing_tool(name: str, args: dict) -> ActionResult:
        return ActionResult(ok=False, via="mcp", error="driver no-op")

    actor._call_mcp_tool = failing_tool  # type: ignore[method-assign]
    sandbox.keyboard.fail = True
    with pytest.raises(RuntimeError, match=r"via SDK .* and MCP \(driver no-op\)"):
        await actor.press_key("down")


@pytest.mark.asyncio
async def test_type_text_goes_sdk_first_without_element_addressing():
    actor, sandbox, mcp_calls = make_routing_actor()
    result = await actor.type_text("Hello.")
    assert result.via == "sdk"
    assert sandbox.keyboard.typed == ["Hello."]
    assert mcp_calls == []


@pytest.mark.asyncio
async def test_type_text_element_addressing_is_driver_only():
    actor, sandbox, mcp_calls = make_routing_actor()
    result = await actor.type_text("hi", element_index=3)
    assert result.via == "mcp"
    assert sandbox.keyboard.typed == []
    assert mcp_calls[0][0] == "type_text"
    assert mcp_calls[0][1]["element_index"] == 3


@pytest.mark.asyncio
async def test_type_text_element_addressing_failure_has_no_sdk_fallback():
    actor, sandbox, _ = make_routing_actor()

    async def failing_tool(name: str, args: dict) -> ActionResult:
        return ActionResult(ok=False, via="mcp", error="boom")

    actor._call_mcp_tool = failing_tool  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no SDK fallback"):
        await actor.type_text("hi", element_index=3)
    assert sandbox.keyboard.typed == []


@pytest.mark.asyncio
async def test_click_element_addressing_is_driver_only():
    actor, sandbox, mcp_calls = make_routing_actor()
    result = await actor.click(1, 2, element_index=5)
    assert result.via == "mcp"
    assert sandbox.mouse.clicks == []
    assert mcp_calls[0][0] == "click"
    assert mcp_calls[0][1]["element_index"] == 5


@pytest.mark.asyncio
async def test_click_pixel_goes_sdk_first_with_translated_coords():
    actor, sandbox, mcp_calls = make_routing_actor()

    async def to_screen(pid, window_id, x, y):
        return x + 100, y + 200

    actor._to_screen = to_screen  # type: ignore[method-assign]
    result = await actor.click(1, 2, x=10, y=20, button="right")
    assert result.via == "sdk"
    assert sandbox.mouse.clicks == [(110, 220, "right")]
    assert mcp_calls == []


@pytest.mark.asyncio
async def test_hotkey_goes_sdk_first():
    actor, sandbox, mcp_calls = make_routing_actor()
    result = await actor.hotkey(["ctrl", "c"])
    assert result.via == "sdk"
    assert sandbox.keyboard.pressed == [["ctrl", "c"]]
    assert mcp_calls == []


@pytest.mark.asyncio
async def test_scroll_keystroke_path_is_driver_only():
    actor, sandbox, mcp_calls = make_routing_actor()
    result = await actor.scroll("down")
    assert result.via == "mcp"
    assert mcp_calls[0][0] == "scroll"
    assert mcp_calls[0][1]["direction"] == "down"


@pytest.mark.asyncio
async def test_scroll_pixel_path_goes_sdk_first():
    actor, sandbox, mcp_calls = make_routing_actor()

    async def to_screen(pid, window_id, x, y):
        return x, y

    actor._to_screen = to_screen  # type: ignore[method-assign]
    result = await actor.scroll("down", pid=1, window_id=2, x=5, y=5, amount=3)
    assert result.via == "sdk"
    assert mcp_calls == []


# ── human-real input: action_utils dispatch, element lookup ────────────────


class ShellFake:
    """Records the shell traffic the action_utils-backed helpers produce."""

    def __init__(self):
        self.commands: list[str] = []

    async def shell_run(self, command, timeout=30, background=False, check=True):
        self.commands.append(command)
        return CommandResult(stdout="{}", stderr="", returncode=0)


@pytest.mark.asyncio
async def test_disable_agent_cursor_disables_config_then_unmaps():
    actor, _, mcp_calls = make_routing_actor()
    shell = ShellFake()
    actor.shell_run = shell.shell_run  # type: ignore[method-assign]
    await actor.disable_agent_cursor()
    assert mcp_calls == [
        ("set_agent_cursor_enabled", {"enabled": False, "session": "default"})
    ]
    assert shell.commands == [
        f"{VENV_PYTHON} -m leapspace.app_space.action_utils unmap_overlay"
    ]


@pytest.mark.asyncio
async def test_disable_agent_cursor_unmaps_even_when_config_disable_fails():
    actor, _, _ = make_routing_actor()

    async def failing_tool(name: str, args: dict) -> ActionResult:
        return ActionResult(ok=False, via="mcp", error="tool not exposed")

    shell = ShellFake()
    actor._call_mcp_tool = failing_tool  # type: ignore[method-assign]
    actor.shell_run = shell.shell_run  # type: ignore[method-assign]
    await actor.disable_agent_cursor()
    assert shell.commands == [
        f"{VENV_PYTHON} -m leapspace.app_space.action_utils unmap_overlay"
    ]


@pytest.mark.asyncio
async def test_ax_elements_parses_the_dump():
    actor = make_actor()
    shell = ShellFake()
    dump = (
        '{"name": "boss", "role": "list item", "x": 15, "y": 80, "w": 226, "h": 14, "value": null}\n'
        '{"name": "message_input", "role": "text", "x": 248, "y": 365, "w": 142, "h": 22, "value": "hi"}\n'
    )

    async def fs_read(path):
        assert path == f"{STAGE_DIR}/ax_elements.jsonl"
        return dump

    actor.shell_run = shell.shell_run  # type: ignore[method-assign]
    actor.fs_read = fs_read  # type: ignore[method-assign]
    elements = await actor.ax_elements()
    assert [element["name"] for element in elements] == ["boss", "message_input"]
    # the AT-SPI walk runs on the OS python (apt pyatspi), imported from the
    # checkout via PYTHONPATH; the dump lands in the staging dir
    assert shell.commands == [
        f"mkdir -p {STAGE_DIR}",
        f"PYTHONPATH=/opt/leapflow/src {SYSTEM_PYTHON}"
        f" -m leapspace.app_space.action_utils dump_elements"
        f" {STAGE_DIR}/ax_elements.jsonl",
    ]


@pytest.mark.asyncio
async def test_type_keys_dispatches_to_action_utils():
    actor = make_actor()
    shell = ShellFake()
    actor.shell_run = shell.shell_run  # type: ignore[method-assign]
    await actor.type_keys("Hi there.")
    # text travels as the typed argument; spaces survive via shell quoting
    assert shell.commands == [
        f"{VENV_PYTHON} -m leapspace.app_space.action_utils type_text 'Hi there.'"
    ]


def test_find_ax_element_resolves_by_name_and_role():
    elements = [
        {"name": "boss", "role": "list item", "x": 1, "y": 2, "w": 3, "h": 4},
        {"name": "boss", "role": "text", "x": 5, "y": 6, "w": 7, "h": 8},
    ]
    assert find_ax_element(elements, "boss", role="list item") is elements[0]
    with pytest.raises(LookupError, match="ambiguous"):
        find_ax_element(elements, "boss")
    with pytest.raises(LookupError, match="no element named 'ghost'"):
        find_ax_element(elements, "ghost")


def test_element_center_computes_screen_center():
    assert element_center({"x": 15, "y": 80, "w": 226, "h": 14}) == (128, 87)
