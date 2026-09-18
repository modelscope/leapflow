# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapAppActor: the OS-signal source of leapspace.

Drives the in-sandbox apps through the cua-driver MCP tool surface plus
direct sandbox handles (sb.shell, sb.clipboard, ...).

Routing policy: input actions the Sandbox SDK can perform go SDK-first and
fall back to the driver when the SDK raises. The driver's input paths are
unreliable in this image (press_key reports success without delivering any
event; type_text lowercases and drops the final character), while SDK input
is real X11 input, visible to the in-sandbox observers. Element addressing
(element_index/element_token) and keystroke-path scroll have no SDK
equivalent and stay driver-only; fs/clipboard actions are SDK-only because
the sandbox MCP exposes no such tools.

A second layer serves reference scripts that must be human-real — every
action observable by the in-box input tap AND effective in the app:
disable_agent_cursor() removes the driver's full-screen overlay that eats
all pixel clicks, ax_elements() provides screen-space element bounds for
coordinate targeting, and type_keys() emits one XTest device event per
character (SDK type_text is observer-invisible XSendEvent). Their
implementations live in the in-box action_utils module, invoked as
`python -m` with the interpreter that owns each dependency stack.

Action signatures mirror the cua-driver MCP tool reference
(https://github.com/trycua/cua/blob/main/docs/content/docs/reference/cua-driver/mcp-tools.mdx):
params the driver marks required (pid on double_click/right_click/
set_value/kill_app/bring_to_front, pid+window_id on get_window_state) are
required here; params the driver genuinely optionalizes (pid on
click/type_text/press_key/hotkey/scroll via scope="desktop") stay optional.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Awaitable, Callable, Literal

from cua_sandbox import Sandbox
from cua_sandbox.interfaces.files import FileEntry
from cua_sandbox.interfaces.shell import CommandResult
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from leapspace.app_space.state import (
    CUA_MCP_PORT,
    LINUX_LEAPFLOW_SRC,
    get_actor_stage_dir,
    get_image_venv_python,
    get_image_system_python,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionResult:
    """Result envelope returned by every dual-path action, MCP or SDK.

    data is the tool payload (structuredContent when present, else parsed
    text); error is None on success and the failure text otherwise.
    """

    ok: bool
    via: Literal["mcp", "sdk"]
    data: Any = None
    images: list[str] = field(default_factory=list)
    error: str | None = None


# Nodes in tree_markdown look like:  - [7] text "Message input" [actions=[...]]
_ELEMENT_RE = re.compile(r'-\s*\[(\d+)\]\s+([a-z ]+?)\s+"(.*?)"')


def find_element(tree_markdown: str, name: str, *, role: str | None = None) -> int:
    """Resolve an accessible name to its element index in a tree snapshot.

    action.py addresses widgets semantically: bound names survive layout
    evolution, pixel coordinates do not. Raises LookupError when the name is
    absent or ambiguous — both are scripting bugs, not runtime conditions.
    Indices expire on the next snapshot; always re-resolve after acting.
    """
    matches = [
        (int(index), node_role)
        for index, node_role, node_name in _ELEMENT_RE.findall(tree_markdown)
        if node_name == name and (role is None or node_role == role)
    ]
    if not matches:
        raise LookupError(f"no element named {name!r} in snapshot")
    if len(matches) > 1:
        raise LookupError(f"ambiguous element name {name!r}: {matches}")
    return matches[0][0]


def find_ax_element(
    elements: list[dict[str, Any]], name: str, *, role: str | None = None
) -> dict[str, Any]:
    """Resolve one element from an ax_elements() dump by accessible name.

    Same contract as find_element: absent or ambiguous names are scripting
    bugs (LookupError), not runtime conditions.
    """
    matches = [
        element
        for element in elements
        if element.get("name") == name and (role is None or element.get("role") == role)
    ]
    if not matches:
        raise LookupError(f"no element named {name!r} in ax dump")
    if len(matches) > 1:
        raise LookupError(f"ambiguous element name {name!r}: {len(matches)} matches")
    return matches[0]


def element_center(element: dict[str, Any]) -> tuple[int, int]:
    """Screen-space center of an ax_elements() dump entry — the pixel a
    human-real click targets."""
    return element["x"] + element["w"] // 2, element["y"] + element["h"] // 2


# In-box helper module the human-real layer invokes as
# `python -m <module> <function> [args...]`; see its docstring for the
# environment gaps it closes.
_ACTION_UTILS = "leapspace.app_space.action_utils"



class LeapAppActor:
    """Human-operation emitter driving a sandbox from the host.

    The MCP connection is lazy: it is established on first use by
    _prepare_cua_mcp() and torn down with the actor. The actual tool list
    is snapshotted at connect time because the sandbox exposes fewer tools
    than the cua-driver docs claim, so action methods must check real
    availability instead of assuming the documented surface.
    """

    def __init__(self, sandbox: Sandbox, *, system: str = "linux") -> None:
        self.sandbox = sandbox
        self.system = system
        self._stage_dir = get_actor_stage_dir(system)
        self._venv_python = get_image_venv_python(system)
        self._system_python = get_image_system_python(system)

        self.exit_stack = AsyncExitStack()
        self.cua_mcp: ClientSession | None = None
        self.cua_mcp_tools: list[str] | None = None

    async def __aenter__(self) -> LeapAppActor:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Safe when MCP was never prepared (aclose on an untouched stack is
        # a no-op) or when preparation failed midway (the half-entered HTTP
        # client is still popped from the stack). The streamable-http client
        # teardown raises cancel-scope/task-affinity RuntimeErrors after a
        # GET-stream reconnect — the transport is being discarded with the
        # sandbox anyway, so that noise is logged instead of masking the
        # exception (if any) the caller is actually unwinding.
        try:
            await self.exit_stack.aclose()
        except Exception:
            logger.warning("cua MCP teardown failed (ignored)", exc_info=True)
        self.cua_mcp = None
        self.cua_mcp_tools = None

    async def _prepare_cua_mcp(self) -> None:
        if self.cua_mcp is not None:
            return
        mcp_port = self.sandbox.exposed_ports[CUA_MCP_PORT]
        mcp_url = f"http://localhost:{mcp_port}/mcp"

        client_ctx = streamable_http_client(mcp_url)
        read_stream, write_stream, _ = await self.exit_stack.enter_async_context(client_ctx)
        session_ctx = ClientSession(read_stream, write_stream)
        self.cua_mcp = await self.exit_stack.enter_async_context(session_ctx)

        await self.cua_mcp.initialize()
        tools = await self.cua_mcp.list_tools()
        self.cua_mcp_tools = [tool.name for tool in tools.tools]

    async def _call_mcp_tool(self, tool_name: str, tool_args: dict[str, Any]) -> ActionResult:
        """Call a cua-driver MCP tool and flatten the result into an ActionResult.

        The MCP branch never raises: an unpreparable connection, an
        unexposed tool, a transport error, or an isError result all come
        back as ok=False with the error text, so action methods pick the
        SDK fallback from the status instead of catching exceptions.
        """
        try:
            if self.cua_mcp is None:
                await self._prepare_cua_mcp()
        except Exception as exc:
            return ActionResult(
                ok=False, via="mcp", error=f"connect: {exc.__class__.__name__}: {exc}"
            )

        if self.cua_mcp_tools is None or tool_name not in self.cua_mcp_tools:
            return ActionResult(
                ok=False, via="mcp", error=f"cua-driver tool '{tool_name}' not exposed"
            )

        # None means "not provided": the driver validates args against the
        # tool schema and rejects JSON null for typed fields, so optional
        # params must be omitted rather than sent as nulls.
        args = {k: v for k, v in tool_args.items() if v is not None}
        try:
            raw = await self.cua_mcp.call_tool(tool_name, args)
        except Exception as exc:
            return ActionResult(ok=False, via="mcp", error=f"{exc.__class__.__name__}: {exc}")

        # call_tool does not raise on tool-level failure — it returns
        # isError=True — so the status is folded into the envelope below
        # and callers route on ok.
        text_parts: list[str] = []
        images: list[str] = []
        for part in getattr(raw, "content", None) or []:
            if getattr(part, "type", None) == "text":
                text_parts.append(getattr(part, "text", "") or "")
            elif getattr(part, "type", None) == "image":
                b64 = getattr(part, "data", None)
                if b64:
                    images.append(b64)

        # Tools serialize their payload as JSON text blocks; a
        # structuredContent block, when present, is the preferred form.
        data: Any = None
        if text_parts:
            joined = "\n".join(t for t in text_parts if t)
            try:
                data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
            except json.JSONDecodeError:
                data = joined
        structured = getattr(raw, "structuredContent", None)
        if structured is not None:
            data = structured

        if bool(getattr(raw, "isError", False)):
            return ActionResult(ok=False, via="mcp", error=str(data))
        return ActionResult(ok=True, via="mcp", data=data, images=images)

    async def _run_sdk_first(
        self,
        sdk_op: Callable[[], Awaitable[None]],
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ActionResult:
        """Run one input action through the SDK, falling back to the driver.

        The SDK delivers real X11 input; the driver's equivalents are broken
        in this image (press_key delivers nothing, type_text mangles text),
        so the SDK goes first wherever it can perform the operation. Only
        when the SDK raises does the driver get the call; both failing is
        the error, and it carries both failure texts.
        """
        try:
            await sdk_op()
        except Exception as exc:
            result = await self._call_mcp_tool(tool_name, tool_args)
            if result.ok:
                return result
            raise RuntimeError(
                f"{tool_name} failed via SDK ({exc}) and MCP ({result.error})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

    async def _mcp_only(self, tool_name: str, tool_args: dict[str, Any]) -> ActionResult:
        """Call a driver-only tool; no SDK equivalent exists to fall back to."""
        result = await self._call_mcp_tool(tool_name, tool_args)
        if not result.ok:
            raise RuntimeError(
                f"{tool_name} failed via MCP and has no SDK fallback: {result.error}"
            )
        return result

    # ── Observation (MCP only) ──────────────────────────────────────────

    async def list_apps(self) -> ActionResult:
        result = await self._call_mcp_tool("list_apps", {})
        if not result.ok:
            raise RuntimeError(f"list_apps failed via MCP: {result.error}")
        return result

    async def list_windows(
        self, pid: int | None = None, *, on_screen_only: bool | None = None
    ) -> ActionResult:
        result = await self._call_mcp_tool(
            "list_windows", {"pid": pid, "on_screen_only": on_screen_only}
        )
        if not result.ok:
            raise RuntimeError(f"list_windows failed via MCP: {result.error}")
        return result

    async def get_window_state(
        self,
        pid: int,
        window_id: int,
        *,
        query: str | None = None,
        include_screenshot: bool | None = None,
        max_elements: int | None = None,
        max_depth: int | None = None,
    ) -> ActionResult:
        """Snapshot the AX tree (+ screenshot) of one window.

        Element indices expire on the next snapshot — re-snapshot every turn
        before any element-indexed action.
        """
        result = await self._call_mcp_tool(
            "get_window_state",
            {
                "pid": pid,
                "window_id": window_id,
                "query": query,
                "include_screenshot": include_screenshot,
                "max_elements": max_elements,
                "max_depth": max_depth,
            },
        )
        if not result.ok:
            raise RuntimeError(f"get_window_state failed via MCP: {result.error}")
        return result

    async def wait_for_window(
        self, app_title: str, *, timeout_s: float = 60, poll_s: float = 2.0
    ) -> dict[str, Any]:
        """Poll list_windows until the app's window appears.

        A window matches when its title equals app_title, optionally with
        one " (...)" status suffix — "LeapChat (3)" matches "LeapChat";
        "LeapChat Pro" does not.
        """
        pattern = re.compile(rf"{re.escape(app_title)}(?: \([^)]*\))?")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = await self.list_windows()
            windows = result.data
            if isinstance(windows, dict):
                windows = windows.get("windows", [])
            for window in windows or []:
                if pattern.fullmatch(window.get("title", "")):
                    return window
            await asyncio.sleep(poll_s)
        raise RuntimeError(
            f"window {app_title!r} did not appear within {timeout_s}s"
        )

    async def snapshot_tree(self, pid: int, window_id: int) -> str:
        """Fresh AX tree for one window, as tree_markdown.

        Element indices expire on the next snapshot — re-snapshot before
        every element-indexed action.
        """
        result = await self.get_window_state(pid, window_id)
        return result.data["tree_markdown"]

    # ── App management ───────────────────────────────────────────────────

    async def launch_app(
        self,
        name: str,
        *,
        bundle_id: str | None = None,
        urls: list[str] | None = None,
    ) -> ActionResult:
        """Launch an app; falls back to a detached shell start."""
        result = await self._call_mcp_tool(
            "launch_app", {"name": name, "bundle_id": bundle_id, "urls": urls}
        )
        if result.ok:
            return result
        try:
            await self.shell_run(f"nohup {shlex.quote(name)} >/dev/null 2>&1 &")
        except Exception as exc:
            raise RuntimeError(
                f"launch_app failed via MCP ({result.error}) and shell ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

    async def bring_to_front(self, pid: int, window_id: int | None = None) -> ActionResult:
        result = await self._call_mcp_tool(
            "bring_to_front", {"pid": pid, "window_id": window_id}
        )
        if not result.ok:
            # Linux declares this unsupported: AT-SPI/X11 already reach
            # backgrounded windows, so the request is satisfied by doing nothing.
            if result.error and "bring_to_front_unsupported_on_platform" in result.error:
                return ActionResult(ok=True, via="mcp", data={"skipped": True})
            raise RuntimeError(f"bring_to_front failed via MCP: {result.error}")
        return result

    async def kill_app(self, pid: int) -> ActionResult:
        """Force-terminate a process; falls back to `kill -9` over shell."""
        result = await self._call_mcp_tool("kill_app", {"pid": pid})
        if result.ok:
            return result
        try:
            await self.shell_run(f"kill -9 {int(pid)}")
        except Exception as exc:
            raise RuntimeError(
                f"kill_app failed via MCP ({result.error}) and shell ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

    # ── UI actions (SDK first, driver fallback; element addressing is
    # driver-only). The SDK delivers real X11 input; the driver is the
    # fallback for pixel targets and the only path for element-addressed
    # targets, which the SDK has no AX knowledge to resolve. A call raises
    # only when both paths have failed, with both errors in the message.

    async def click(
        self,
        pid: int | None = None,
        window_id: int | None = None,
        *,
        element_index: int | None = None,
        element_token: str | None = None,
        snapshot_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
    ) -> ActionResult:
        """Click an element or a pixel point.

        Window scope (pid given): x/y are window-local screenshot pixels.
        Desktop scope (no pid): x/y are true screen pixels.
        """
        args = {
            "pid": pid,
            "window_id": window_id,
            "element_index": element_index,
            "element_token": element_token,
            "snapshot_id": snapshot_id,
            "x": x,
            "y": y,
            "button": button,
            "scope": "desktop" if pid is None else None,
        }
        if x is None or y is None:
            # Element-addressed or targetless click: the SDK cannot
            # synthesize a click without pixel coordinates.
            return await self._mcp_only("click", args)

        async def sdk_op() -> None:
            sx, sy = await self._to_screen(pid, window_id, x, y)
            await self.sandbox.mouse.click(sx, sy, button=button)

        return await self._run_sdk_first(sdk_op, "click", args)

    async def double_click(
        self,
        pid: int,
        *,
        window_id: int | None = None,
        element_index: int | None = None,
        element_token: str | None = None,
        snapshot_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> ActionResult:
        """Double-click an element or a pixel point.

        Unlike click/right_click, the driver's pixel path here takes true
        screen coordinates, so the SDK path needs no translation.
        """
        args = {
            "pid": pid,
            "window_id": window_id,
            "element_index": element_index,
            "element_token": element_token,
            "snapshot_id": snapshot_id,
            "x": x,
            "y": y,
        }
        if x is None or y is None:
            return await self._mcp_only("double_click", args)
        return await self._run_sdk_first(
            lambda: self.sandbox.mouse.double_click(x, y), "double_click", args
        )

    async def right_click(
        self,
        pid: int,
        *,
        window_id: int | None = None,
        element_index: int | None = None,
        element_token: str | None = None,
        snapshot_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> ActionResult:
        """Right-click an element or a window-local pixel point."""
        args = {
            "pid": pid,
            "window_id": window_id,
            "element_index": element_index,
            "element_token": element_token,
            "snapshot_id": snapshot_id,
            "x": x,
            "y": y,
        }
        if x is None or y is None:
            return await self._mcp_only("right_click", args)

        async def sdk_op() -> None:
            sx, sy = await self._to_screen(pid, window_id, x, y)
            await self.sandbox.mouse.right_click(sx, sy)

        return await self._run_sdk_first(sdk_op, "right_click", args)

    async def type_text(
        self,
        text: str,
        *,
        pid: int | None = None,
        window_id: int | None = None,
        element_index: int | None = None,
        element_token: str | None = None,
        snapshot_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> ActionResult:
        """Type text into an element/pixel target, or the focused widget when
        no target is given (desktop scope).

        The SDK path types into whatever has focus (clicking x/y first when
        a pixel target is given) and delivers the text verbatim; the
        driver's type_text mangles it (lowercases, drops the last
        character), so it serves only as the fallback.
        """
        args = {
            "text": text,
            "pid": pid,
            "window_id": window_id,
            "element_index": element_index,
            "element_token": element_token,
            "snapshot_id": snapshot_id,
            "x": x,
            "y": y,
            "scope": "desktop" if pid is None else None,
        }
        if element_index is not None or element_token is not None:
            # Element addressing needs the driver's AX knowledge; the SDK
            # has no way to resolve an element to a focus target.
            return await self._mcp_only("type_text", args)

        async def sdk_op() -> None:
            if x is not None and y is not None:
                sx, sy = await self._to_screen(pid, window_id, x, y)
                await self.sandbox.mouse.click(sx, sy)
            await self.sandbox.keyboard.type(text)

        return await self._run_sdk_first(sdk_op, "type_text", args)

    async def press_key(self, key: str, *, pid: int | None = None) -> ActionResult:
        """Press a single key (return, escape, tab, ...).

        The SDK keypress delivers a real X11 key event to the focused
        widget; the driver's press_key is a silent no-op in this image —
        it reports success without delivering any event — so it serves
        only as the fallback.
        """
        args = {"key": key, "pid": pid, "scope": "desktop" if pid is None else None}
        return await self._run_sdk_first(
            lambda: self.sandbox.keyboard.keypress(key), "press_key", args
        )

    async def hotkey(self, keys: list[str], *, pid: int | None = None) -> ActionResult:
        """Press a key combination, e.g. ["ctrl", "c"]."""
        args = {
            "keys": keys,
            "pid": pid,
            "scope": "desktop" if pid is None else None,
        }
        return await self._run_sdk_first(
            lambda: self.sandbox.keyboard.keypress(keys), "hotkey", args
        )

    async def scroll(
        self,
        direction: str,
        *,
        pid: int | None = None,
        window_id: int | None = None,
        amount: int = 3,
        by: str | None = None,
        element_index: int | None = None,
        element_token: str | None = None,
        snapshot_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> ActionResult:
        """Scroll a window-local point (pixel-wheel path) or the focused
        region (keystroke path, no target).

        The pixel-wheel path is dual (SDK first, driver fallback); the
        keystroke path and element addressing are driver-only — the SDK
        mouse cannot scroll without pixel coordinates.
        """
        args = {
            "direction": direction,
            "pid": pid,
            "window_id": window_id,
            "amount": amount,
            "by": by,
            "element_index": element_index,
            "element_token": element_token,
            "snapshot_id": snapshot_id,
            "x": x,
            "y": y,
            "scope": "desktop" if pid is None else None,
        }
        if x is None or y is None or element_index is not None or element_token is not None:
            return await self._mcp_only("scroll", args)

        scroll_x, scroll_y = 0, 0
        if direction in ("up", "down"):
            scroll_y = amount if direction == "down" else -amount
        else:
            scroll_x = amount if direction == "right" else -amount

        async def sdk_op() -> None:
            sx, sy = await self._to_screen(pid, window_id, x, y)
            await self.sandbox.mouse.scroll(sx, sy, scroll_x=scroll_x, scroll_y=scroll_y)

        return await self._run_sdk_first(sdk_op, "scroll", args)

    async def set_value(
        self,
        pid: int,
        value: str,
        *,
        window_id: int | None = None,
        element_index: int | None = None,
        element_token: str | None = None,
        snapshot_id: str | None = None,
    ) -> ActionResult:
        """Set an element's value directly. MCP-only: the SDK has no equivalent."""
        return await self._mcp_only(
            "set_value",
            {
                "pid": pid,
                "value": value,
                "window_id": window_id,
                "element_index": element_index,
                "element_token": element_token,
                "snapshot_id": snapshot_id,
            },
        )

    # ── human-real input: observable + effective X11 events ────────────
    #
    # click(x, y) / press_key already deliver real device events, but two
    # environment gaps make the rest of a human session invisible or
    # ineffective. These helpers close them through the in-box
    # action_utils module (see its docstring for the mechanism).

    async def disable_agent_cursor(self) -> None:
        """Clear the driver's agent-cursor overlay out of the X input path.

        The overlay is a full-screen window stacked above every app, so all
        pixel input lands on it instead of the app under the pointer. The
        config-level disable stops the driver from showing it again, but
        leaves an already-created window mapped — the unmap is what
        actually frees the input path. Call once, after the apps are up and
        before any pixel input.
        """
        result = await self._call_mcp_tool(
            "set_agent_cursor_enabled", {"enabled": False, "session": "default"}
        )
        if not result.ok:
            # The unmap below still fixes the live window; only the future
            # re-show guard is missing, so this is a warning, not a failure.
            logger.warning(
                "agent cursor config disable failed (unmapping anyway): %s",
                result.error,
            )
        unmap = await self.shell_run(f"{self._venv_python} -m {_ACTION_UTILS} unmap_overlay")
        logger.info("agent cursor overlay: %s", (unmap.stdout or "").strip())

    async def ax_elements(self) -> list[dict[str, Any]]:
        """Dump every named accessible element on the desktop.

        Each entry carries its screen-space bounds (x/y/w/h, XY_SCREEN)
        and its text value when the element exposes one — the coordinate
        source for human-real pixel clicks and the readback path for
        typed-text verification. Runs on the OS python (apt pyatspi,
        reached via PYTHONPATH into the checkout) because the in-box
        driver's get_window_state returns only tree_markdown, no bounds.
        """
        dump_path = f"{self._stage_dir}/ax_elements.jsonl"
        await self.shell_run(f"mkdir -p {shlex.quote(str(self._stage_dir))}")
        await self.shell_run(
            f"PYTHONPATH={LINUX_LEAPFLOW_SRC} {self._system_python}"
            f" -m {_ACTION_UTILS} dump_elements {shlex.quote(dump_path)}"
        )
        raw = await self.fs_read(dump_path)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    async def type_keys(self, text: str) -> None:
        """Type text into the focused widget, one real keystroke per char.

        Unlike type_text (SDK XSendEvent — verbatim but invisible to
        XRecord-based observers; driver — mangles text), every character
        here is an XTest device event: the input tap records it and the
        focused widget receives it, exactly like a physical keyboard.
        Latin-1 characters only; anything else fails the run naming the
        untypeable characters.
        """
        await self.shell_run(
            f"{self._venv_python} -m {_ACTION_UTILS} type_text {shlex.quote(text)}"
        )

    async def _to_screen(
        self, pid: int | None, window_id: int | None, x: int, y: int
    ) -> tuple[int, int]:
        """Translate action coordinates into the screen pixels the SDK wants.

        Desktop-scope calls (no pid) already carry screen pixels; window-scope
        calls carry window-local screenshot pixels and are shifted by the
        window origin.
        """
        if pid is None:
            return x, y
        if window_id is None:
            raise RuntimeError("window_id is required to translate window-local coordinates")
        ox, oy = await self._window_origin(pid, window_id)
        return ox + x, oy + y

    async def _window_origin(self, pid: int, window_id: int) -> tuple[int, int]:
        """Screen-space origin of a window, from the list_windows bounds."""
        result = await self._call_mcp_tool("list_windows", {"pid": pid})
        if not result.ok:
            raise RuntimeError(
                f"list_windows failed during coordinate translation: {result.error}"
            )
        windows = result.data
        if isinstance(windows, dict):
            windows = windows.get("windows", [])
        for window in windows or []:
            if window.get("window_id") == window_id:
                bounds = window.get("bounds") or {}
                return int(bounds.get("x", 0)), int(bounds.get("y", 0))
        raise RuntimeError(f"window {window_id} not found for pid {pid}")

    # ── fs channel (SDK only — the sandbox MCP exposes no fs tools) ──────

    async def fs_create(self, path: str, content: str = "") -> None:
        await self.sandbox.files.write_text(path, content)

    async def fs_write(self, path: str, content: str) -> None:
        await self.sandbox.files.write_text(path, content)

    async def fs_move(self, src: str, dst: str) -> None:
        await self.shell_run(f"mv -- {shlex.quote(src)} {shlex.quote(dst)}")

    async def fs_delete(self, path: str) -> None:
        if await self.sandbox.files.is_dir(path):
            await self.sandbox.files.remove_dir(path)
        else:
            await self.sandbox.files.remove(path)

    async def fs_mkdir(self, path: str) -> None:
        await self.sandbox.files.make_dir(path)

    async def fs_read(self, path: str) -> str:
        """Read a sandbox text file — the ground-truth readback path for
        evaluation assertions."""
        return await self.sandbox.files.read_text(path)

    async def fs_exists(self, path: str) -> bool:
        return await self.sandbox.files.exists(path)

    async def fs_list(self, path: str) -> list[FileEntry]:
        return await self.sandbox.files.list(path)

    async def fs_upload(self, local_path: str, remote_path: str) -> None:
        """Push a host file into the sandbox (fixtures, app code)."""
        await self.sandbox.files.upload(local_path, remote_path)

    async def fs_download(self, remote_path: str, local_path: str) -> None:
        """Pull a sandbox file down to the host (evidence, recordings)."""
        await self.sandbox.files.download(remote_path, local_path)

    # ── desktop state (SDK only — MCP exposes neither a full-desktop
    # screenshot nor a frontmost-window read) ──

    async def screenshot(self) -> bytes:
        """Capture the full desktop as PNG bytes (evidence, desktop-level
        assertions that get_window_state's per-window grabs can't serve)."""
        return await self.sandbox.screen.screenshot()

    async def active_window_title(self) -> str:
        """Title of the currently focused window — focus-channel assertion
        helper."""
        return await self.sandbox.window.get_active_title()

    # ── clipboard channel (SDK only — MCP clipboard tools are not exposed) ──

    async def clip_set(self, text: str) -> None:
        await self.sandbox.clipboard.set(text)

    async def clip_get(self) -> str:
        return await self.sandbox.clipboard.get()

    async def shell_run(
        self,
        command: str,
        timeout: int = 30,
        background: bool = False,
        check: bool = True,
    ) -> CommandResult:
        """Run an in-sandbox shell command; check=True raises on non-zero exit.

        check=False is the verdict channel: expect programs exit non-zero
        on failure while their stdout is the payload, so the caller needs
        the raw result, not an exception. With background=True the command
        returns immediately and stdout carries the spawned pid — the way
        to start long-lived services (demo apps, Flask). Public escape
        hatch for scenario setup (installs, service starts) that has no
        dedicated actor method; also backs the launch/kill fallbacks and
        fs_move.
        """
        result = await self.sandbox.shell.run(command, timeout=timeout, background=background)
        if check and not result.success:
            raise RuntimeError(
                f"in-sandbox command failed ({result.returncode}): {result.stderr.strip()}"
            )
        return result
