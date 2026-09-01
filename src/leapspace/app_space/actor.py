"""LeapAppActor: the OS-signal source of leapspace.

Drives the in-sandbox apps through the cua-driver MCP tool surface plus
direct sandbox handles (sb.shell, sb.clipboard, ...).

Routing policy: evaluation-relevant actions go MCP-first (AX element
addressing, same tool surface as the agent under test) and fall back to
the Sandbox SDK when MCP fails. fs/clipboard actions are SDK-only because
the sandbox MCP exposes no such tools.

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
import re
import shlex
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal

from cua_sandbox import Sandbox
from cua_sandbox.interfaces.files import FileEntry
from cua_sandbox.interfaces.shell import CommandResult
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from leapspace.app_space.utils import CUA_MCP_PORT


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


class LeapAppActor:
    """Human-operation emitter driving a sandbox from the host.

    The MCP connection is lazy: it is established on first use by
    _prepare_cua_mcp() and torn down with the actor. The actual tool list
    is snapshotted at connect time because the sandbox exposes fewer tools
    than the cua-driver docs claim, so action methods must check real
    availability instead of assuming the documented surface.
    """

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

        self.exit_stack = AsyncExitStack()
        self.cua_mcp: ClientSession | None = None
        self.cua_mcp_tools: list[str] | None = None

    @classmethod
    async def attach(cls, name: str) -> LeapAppActor:
        """Attach to a running sandbox by name — action.py's entry point (D6)."""
        return cls(await Sandbox.connect(name, local=True))

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
        # client is still popped from the stack).
        await self.exit_stack.aclose()
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
        self, title_prefix: str, *, timeout_s: float = 60, poll_s: float = 2.0
    ) -> dict[str, Any]:
        """Poll list_windows until a window titled `title_prefix...` appears."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = await self.list_windows()
            windows = result.data
            if isinstance(windows, dict):
                windows = windows.get("windows", [])
            for window in windows or []:
                if window.get("title", "").startswith(title_prefix):
                    return window
            await asyncio.sleep(poll_s)
        raise RuntimeError(
            f"window {title_prefix!r} did not appear within {timeout_s}s"
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
            await self.shell_checked(f"nohup {shlex.quote(name)} >/dev/null 2>&1 &")
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
            await self.shell_checked(f"kill -9 {int(pid)}")
        except Exception as exc:
            raise RuntimeError(
                f"kill_app failed via MCP ({result.error}) and shell ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

    # ── UI actions (MCP first, SDK pixel/focus fallback) ─────────────────
    # Fallbacks are pixel-only: an element-addressed call that fails has no
    # SDK retargeting and raises the MCP error directly. A call raises only
    # when both paths have failed, with both errors in the message.

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
        result = await self._call_mcp_tool(
            "click",
            {
                "pid": pid,
                "window_id": window_id,
                "element_index": element_index,
                "element_token": element_token,
                "snapshot_id": snapshot_id,
                "x": x,
                "y": y,
                "button": button,
                "scope": "desktop" if pid is None else None,
            },
        )
        if result.ok:
            return result
        if x is None or y is None:
            raise RuntimeError(
                f"click failed via MCP and has no pixel target for SDK fallback: "
                f"{result.error}"
            )
        try:
            sx, sy = await self._to_screen(pid, window_id, x, y)
            await self.sandbox.mouse.click(sx, sy, button=button)
        except Exception as exc:
            raise RuntimeError(
                f"click failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

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
        screen coordinates, so the SDK fallback needs no translation.
        """
        result = await self._call_mcp_tool(
            "double_click",
            {
                "pid": pid,
                "window_id": window_id,
                "element_index": element_index,
                "element_token": element_token,
                "snapshot_id": snapshot_id,
                "x": x,
                "y": y,
            },
        )
        if result.ok:
            return result
        if x is None or y is None:
            raise RuntimeError(
                f"double_click failed via MCP and has no pixel target for SDK fallback: "
                f"{result.error}"
            )
        try:
            await self.sandbox.mouse.double_click(x, y)
        except Exception as exc:
            raise RuntimeError(
                f"double_click failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

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
        result = await self._call_mcp_tool(
            "right_click",
            {
                "pid": pid,
                "window_id": window_id,
                "element_index": element_index,
                "element_token": element_token,
                "snapshot_id": snapshot_id,
                "x": x,
                "y": y,
            },
        )
        if result.ok:
            return result
        if x is None or y is None:
            raise RuntimeError(
                f"right_click failed via MCP and has no pixel target for SDK fallback: "
                f"{result.error}"
            )
        try:
            sx, sy = await self._to_screen(pid, window_id, x, y)
            await self.sandbox.mouse.right_click(sx, sy)
        except Exception as exc:
            raise RuntimeError(
                f"right_click failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

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
        """Type text into an element/pixel target, or the frontmost app when
        no target is given (desktop scope)."""
        result = await self._call_mcp_tool(
            "type_text",
            {
                "text": text,
                "pid": pid,
                "window_id": window_id,
                "element_index": element_index,
                "element_token": element_token,
                "snapshot_id": snapshot_id,
                "x": x,
                "y": y,
                "scope": "desktop" if pid is None else None,
            },
        )
        if result.ok:
            return result
        if element_index is not None or element_token is not None:
            raise RuntimeError(
                f"type_text failed via MCP and element addressing has no SDK fallback: "
                f"{result.error}"
            )
        try:
            if x is not None and y is not None:
                sx, sy = await self._to_screen(pid, window_id, x, y)
                await self.sandbox.mouse.click(sx, sy)
            await self.sandbox.keyboard.type(text)
        except Exception as exc:
            raise RuntimeError(
                f"type_text failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

    async def press_key(self, key: str, *, pid: int | None = None) -> ActionResult:
        """Press a single key (return, escape, tab, ...)."""
        result = await self._call_mcp_tool(
            "press_key",
            {"key": key, "pid": pid, "scope": "desktop" if pid is None else None},
        )
        if result.ok:
            return result
        try:
            await self.sandbox.keyboard.keypress(key)
        except Exception as exc:
            raise RuntimeError(
                f"press_key failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

    async def hotkey(self, keys: list[str], *, pid: int | None = None) -> ActionResult:
        """Press a key combination, e.g. ["ctrl", "c"]."""
        result = await self._call_mcp_tool(
            "hotkey",
            {"keys": keys, "pid": pid, "scope": "desktop" if pid is None else None},
        )
        if result.ok:
            return result
        try:
            await self.sandbox.keyboard.keypress(keys)
        except Exception as exc:
            raise RuntimeError(
                f"hotkey failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

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
        region (keystroke path, no target)."""
        result = await self._call_mcp_tool(
            "scroll",
            {
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
            },
        )
        if result.ok:
            return result
        if x is None or y is None:
            raise RuntimeError(
                f"scroll failed via MCP and has no pixel target for SDK fallback: "
                f"{result.error}"
            )
        scroll_x, scroll_y = 0, 0
        if direction in ("up", "down"):
            scroll_y = amount if direction == "down" else -amount
        else:
            scroll_x = amount if direction == "right" else -amount
        try:
            sx, sy = await self._to_screen(pid, window_id, x, y)
            await self.sandbox.mouse.scroll(sx, sy, scroll_x=scroll_x, scroll_y=scroll_y)
        except Exception as exc:
            raise RuntimeError(
                f"scroll failed via MCP ({result.error}) and SDK ({exc})"
            ) from exc
        return ActionResult(ok=True, via="sdk")

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
        result = await self._call_mcp_tool(
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
        if not result.ok:
            raise RuntimeError(f"set_value failed via MCP: {result.error}")
        return result

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
        await self.shell_checked(f"mv -- {shlex.quote(src)} {shlex.quote(dst)}")

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

    async def shell_checked(
        self, command: str, timeout: int = 30, background: bool = False
    ) -> CommandResult:
        """Run an in-sandbox shell command, raising on non-zero exit.

        Public escape hatch for scenario setup (installs, service starts)
        that has no dedicated actor method; also backs the launch/kill
        fallbacks and fs_move. With background=True the command returns
        immediately and stdout carries the spawned pid — no exit code to
        check, so this is the way to start long-lived services (demo apps,
        Flask).
        """
        result = await self.sandbox.shell.run(command, timeout=timeout, background=background)
        if not result.success:
            raise RuntimeError(
                f"in-sandbox command failed ({result.returncode}): {result.stderr.strip()}"
            )
        return result
