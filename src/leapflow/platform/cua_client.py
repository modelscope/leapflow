# Copyright (c) Alibaba, Inc. and its affiliates.
"""CuaDriverClient — MCP stdio bridge to cua-driver for unified OS execution.

Implements the HostRpc Protocol by mapping LeapFlow's Methods constants to
cua-driver MCP tool calls. Designed for LLM-native context pipelines where
the execution layer is fully delegated to cua-driver-rs.

Architecture:
  - _AsyncBridge: daemon thread running an asyncio event loop, bridging
    sync/async boundaries transparently.
  - _McpSession: lifecycle coroutine owning the MCP stdio contexts
    (enter + exit in the SAME task — anyio cancel-scope invariant).
  - CuaDriverClient: public facade implementing HostRpc.call(), dispatching
    Methods → cua-driver MCP tools via a declarative routing table.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from leapflow.platform.protocol import HostRpc, Methods, RpcError

logger = logging.getLogger(__name__)

# ── Configuration (all overridable via env) ──────────────────────────────────

_CUA_DRIVER_CMD = os.environ.get("LEAPFLOW_CUA_DRIVER_CMD", "cua-driver")
_CUA_DRIVER_ARGS_DEFAULT: List[str] = ["mcp"]
_CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"

_SESSION_READY_TIMEOUT_S = float(os.environ.get("LEAPFLOW_CUA_SESSION_TIMEOUT", "15.0"))
_CALL_TIMEOUT_S = float(os.environ.get("LEAPFLOW_CUA_CALL_TIMEOUT", "30.0"))
_KEEPALIVE_INTERVAL_S = float(os.environ.get("LEAPFLOW_CUA_KEEPALIVE_INTERVAL", "20.0"))
_MANIFEST_TIMEOUT_S = float(os.environ.get("LEAPFLOW_CUA_MANIFEST_TIMEOUT", "6.0"))
# Cold app.list on Windows enumerates Start-Menu shortcuts + WinRT packages and
# can exceed a minute. Cutting it short does not free the serial MCP pipe — the
# driver keeps enumerating and every later call queues behind it — so the
# timeout must outlast the worst cold enumeration.
_APP_LIST_TIMEOUT_S = float(os.environ.get("LEAPFLOW_CUA_APP_LIST_TIMEOUT", "120.0"))


# ── Telemetry policy ─────────────────────────────────────────────────────────

def _telemetry_disabled() -> bool:
    """Default: disable cua-driver telemetry unless explicitly opted-in."""
    val = os.environ.get(_CUA_TELEMETRY_ENV_VAR, "")
    if val == "1":
        return False
    return True


def _child_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build environment dict for spawning cua-driver subprocess."""
    env = dict(base if base is not None else os.environ)
    if _telemetry_disabled():
        env[_CUA_TELEMETRY_ENV_VAR] = "0"
    # Suppress async update banner on stderr — it writes directly to the
    # inherited FD (bypassing patch_stdout) and corrupts prompt_toolkit TUI.
    if env.get("CUA_DRIVER_RS_UPDATE_CHECK", "").lower() not in ("1", "true", "yes"):
        env["CUA_DRIVER_RS_UPDATE_CHECK"] = "0"
    return env


# ── Driver discovery ─────────────────────────────────────────────────────────

def _resolve_mcp_invocation(
    driver_cmd: str,
    *,
    timeout: float = _MANIFEST_TIMEOUT_S,
) -> Tuple[str, List[str]]:
    """Discover MCP spawn args via `cua-driver manifest`. Falls back gracefully."""
    try:
        proc = subprocess.run(
            [driver_cmd, "manifest"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        return driver_cmd, list(_CUA_DRIVER_ARGS_DEFAULT)

    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return driver_cmd, list(_CUA_DRIVER_ARGS_DEFAULT)

    try:
        manifest = json.loads(proc.stdout.strip())
    except (ValueError, TypeError):
        return driver_cmd, list(_CUA_DRIVER_ARGS_DEFAULT)

    if not isinstance(manifest, dict):
        return driver_cmd, list(_CUA_DRIVER_ARGS_DEFAULT)

    invocation = manifest.get("mcp_invocation")
    if not isinstance(invocation, dict):
        return driver_cmd, list(_CUA_DRIVER_ARGS_DEFAULT)

    args = invocation.get("args")
    command = invocation.get("command")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return driver_cmd, list(_CUA_DRIVER_ARGS_DEFAULT)
    if not isinstance(command, str) or not command:
        return driver_cmd, args
    return command, args


def cua_driver_available() -> bool:
    """True if cua-driver binary is discoverable on PATH."""
    return bool(shutil.which(_CUA_DRIVER_CMD))


# ── AsyncBridge ──────────────────────────────────────────────────────────────

class _AsyncBridge:
    """Daemon thread running an asyncio event loop. Marshals coroutines
    from any thread into that loop and returns results synchronously."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(
            target=_run, daemon=True, name="cua-driver-bridge"
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("cua-driver asyncio bridge failed to start")

    def run(self, coro: Any, timeout: Optional[float] = _CALL_TIMEOUT_S) -> Any:
        """Schedule a coroutine on the bridge loop and block until result."""
        if not self._loop or not self._thread or not self._thread.is_alive():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("cua-driver bridge not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                f"cua-driver call timed out after {timeout}s"
            ) from None

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None


# ── MCP Session ──────────────────────────────────────────────────────────────

class _McpSession:
    """Manages the MCP stdio connection lifecycle in a single coroutine task.

    The lifecycle coroutine opens stdio_client + ClientSession, populates
    tool capabilities, signals ready, then blocks until shutdown. Tool
    calls run as independent coroutines on the same loop.
    """

    def __init__(self, bridge: _AsyncBridge) -> None:
        self._bridge = bridge
        self._session: Any = None
        self._lock = threading.Lock()
        self._started = False
        self._tools: Dict[str, Set[str]] = {}  # tool_name → capabilities
        self._capability_version: str = ""
        self._ready_event = threading.Event()
        self._shutdown_event: Optional[asyncio.Event] = None
        self._lifecycle_future: Optional[concurrent.futures.Future] = None
        self._setup_error: Optional[BaseException] = None
        self._command: str = _CUA_DRIVER_CMD
        self._args: List[str] = list(_CUA_DRIVER_ARGS_DEFAULT)
        self._last_error: str = ""
        self._restart_count = 0

    @property
    def started(self) -> bool:
        return self._started

    @property
    def available_tools(self) -> Dict[str, Set[str]]:
        return self._tools

    @property
    def capability_version(self) -> str:
        return self._capability_version

    @property
    def command(self) -> str:
        return self._command

    @property
    def args(self) -> List[str]:
        return list(self._args)

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def has_tool(self, name: str) -> bool:
        """True if tools/list advertised this tool name."""
        return name in self._tools

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        """Check if a capability is advertised (optionally scoped to a tool)."""
        if tool is not None:
            return capability in self._tools.get(tool, set())
        return any(capability in caps for caps in self._tools.values())

    async def _lifecycle_coro(self) -> None:
        """Long-lived owner of MCP contexts. Enter and exit happen in the
        SAME asyncio task to preserve anyio cancel-scope invariant."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._shutdown_event = asyncio.Event()

        try:
            if not cua_driver_available():
                raise RuntimeError(
                    "cua-driver not found on PATH. Set LEAPFLOW_CUA_DRIVER_CMD "
                    "or install: https://github.com/trycua/cua"
                )

            command, args = _resolve_mcp_invocation(_CUA_DRIVER_CMD)
            self._command = command
            self._args = list(args)
            self._last_error = ""
            params = StdioServerParameters(
                command=command,
                args=args,
                env=_child_env(),
            )

            # Route subprocess stderr away from the terminal — cua-driver logs
            # and update notices must not corrupt prompt_toolkit rendering.
            with open(os.devnull, "w", encoding="utf-8") as _devnull:
                async with stdio_client(params, errlog=_devnull) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await self._discover_capabilities(session)
                        self._session = session
                        self._ready_event.set()
                        await self._shutdown_event.wait()
        except BaseException as e:
            self._setup_error = e
            self._last_error = str(e)
            self._ready_event.set()
            raise
        finally:
            self._session = None

    async def _discover_capabilities(self, session: Any) -> None:
        """Populate per-tool capability sets from tools/list."""
        try:
            tools_response = await session.list_tools()
            for tool in getattr(tools_response, "tools", []) or []:
                name = getattr(tool, "name", None)
                if not isinstance(name, str):
                    continue
                caps = getattr(tool, "capabilities", None)
                if caps is None:
                    extra = getattr(tool, "model_extra", None) or {}
                    caps = extra.get("capabilities")
                if isinstance(caps, list):
                    self._tools[name] = {c for c in caps if isinstance(c, str)}
                else:
                    self._tools[name] = set()

            cv = getattr(tools_response, "capability_version", None)
            if cv is None:
                extra = getattr(tools_response, "model_extra", None) or {}
                cv = extra.get("capability_version")
            if isinstance(cv, str):
                self._capability_version = cv
        except Exception as e:
            logger.debug("cua-driver capability discovery failed: %s", e)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._bridge.start()
            self._start_lifecycle()
            self._started = True

    def _start_lifecycle(self) -> None:
        """Spawn lifecycle coroutine and wait for ready. Caller must hold lock."""
        self._ready_event = threading.Event()
        self._setup_error = None
        self._last_error = ""
        self._shutdown_event = None
        self._tools = {}
        self._capability_version = ""

        loop = self._bridge.loop
        if loop is None:
            raise RuntimeError("cua-driver bridge loop not available")

        self._lifecycle_future = asyncio.run_coroutine_threadsafe(
            self._lifecycle_coro(), loop
        )
        if not self._ready_event.wait(timeout=_SESSION_READY_TIMEOUT_S):
            self._signal_shutdown()
            raise RuntimeError(
                f"cua-driver session not ready within {_SESSION_READY_TIMEOUT_S}s"
            )
        if self._setup_error is not None:
            raise RuntimeError(
                f"cua-driver session setup failed: {self._setup_error}"
            ) from self._setup_error

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_lifecycle()

    def _stop_lifecycle(self) -> None:
        """Signal shutdown and wait for lifecycle unwind. Caller must hold lock."""
        self._signal_shutdown()
        fut = self._lifecycle_future
        if fut is None:
            return
        try:
            fut.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            logger.warning("cua-driver session shutdown timed out")
        except Exception as e:
            logger.debug("cua-driver shutdown: %s", e)
        finally:
            self._lifecycle_future = None

    def _signal_shutdown(self) -> None:
        loop = self._bridge.loop
        event = self._shutdown_event
        if loop and event and loop.is_running():
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass

    def _restart(self) -> None:
        """Reconnect after session drop. Caller must hold lock."""
        if self._started:
            try:
                self._stop_lifecycle()
            except Exception as e:
                logger.debug("cleanup before reconnect: %s", e)
        self._started = False
        self._restart_count += 1
        self._start_lifecycle()
        self._started = True

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an MCP tool and return extracted result dict."""
        if self._session is None:
            raise RuntimeError("cua-driver session not active")
        result = await self._session.call_tool(name, args)
        return _extract_result(result)

    def call_tool_sync(
        self, name: str, args: Dict[str, Any], timeout: float = _CALL_TIMEOUT_S
    ) -> Dict[str, Any]:
        """Synchronous tool call with reconnect-once on session drop."""
        if not self._started:
            raise RuntimeError("cua-driver session not started")
        try:
            return self._bridge.run(self.call_tool(name, args), timeout=timeout)
        except Exception as e:
            if not _is_closed_session_error(e):
                raise
            logger.warning("cua-driver session closed during %s; reconnecting", name)
            with self._lock:
                self._restart()
            return self._bridge.run(self.call_tool(name, args), timeout=timeout)


# ── Result extraction ────────────────────────────────────────────────────────

def _extract_result(mcp_result: Any) -> Dict[str, Any]:
    """Flatten an MCP CallToolResult into a plain dict."""
    data: Any = None
    images: List[str] = []
    is_error = bool(getattr(mcp_result, "isError", False))
    structured: Optional[Dict] = getattr(mcp_result, "structuredContent", None) or None
    text_parts: List[str] = []

    for part in getattr(mcp_result, "content", []) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            text_parts.append(getattr(part, "text", "") or "")
        elif ptype == "image":
            b64 = getattr(part, "data", None)
            if b64:
                images.append(b64)

    if text_parts:
        joined = "\n".join(t for t in text_parts if t)
        try:
            data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
        except json.JSONDecodeError:
            data = joined

    return {
        "data": data,
        "images": images,
        "structuredContent": structured,
        "isError": is_error,
    }


def _is_closed_session_error(exc: Exception) -> bool:
    """Detect MCP/stdio failures recoverable by reconnecting."""
    name = exc.__class__.__name__
    module = getattr(exc.__class__, "__module__", "")
    return (
        name in {"ClosedResourceError", "BrokenResourceError", "EndOfStream"}
        or (module.startswith("anyio") and "Resource" in name)
        or isinstance(exc, (BrokenPipeError, EOFError))
    )


# ── Local operations (clipboard, file) ──────────────────────────────────────

def _clipboard_get() -> str:
    """Read clipboard via platform-native command."""
    if sys.platform == "win32":
        # -Raw preserves the clipboard text verbatim (the default mode
        # returns a line array and reflows newlines). Force UTF-8 output —
        # the default console codepage mangles CJK.
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5.0)
            text = result.stdout.decode("utf-8", errors="replace")
            # PowerShell appends exactly one trailing newline to stdout;
            # strip only that one so genuine trailing newlines survive.
            if text.endswith("\r\n"):
                return text[:-2]
            if text.endswith("\n"):
                return text[:-1]
            return text
        except Exception as e:
            raise RpcError("clipboard_error", f"Failed to read clipboard: {e}", {})

    cmd = ["pbpaste"] if sys.platform == "darwin" else ["xclip", "-selection", "clipboard", "-o"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            errors="replace", timeout=5.0,
        )
        return result.stdout
    except Exception as e:
        raise RpcError("clipboard_error", f"Failed to read clipboard: {e}", {})


def _clipboard_set(text: str) -> None:
    """Write clipboard via platform-native command."""
    if sys.platform == "win32":
        # PowerShell -Command re-parses the joined argv, stripping the
        # quoting subprocess added — any text with spaces breaks parameter
        # binding. Base64 transport is immune to spaces/quotes/CJK/$.
        encoded = base64.b64encode(text.encode("utf-16-le")).decode("ascii")
        script = (
            "$t=[Text.Encoding]::Unicode.GetString("
            f"[Convert]::FromBase64String('{encoded}'));"
            "if ($t.Length -gt 0) { Set-Clipboard -Value $t }"
            " else { Set-Clipboard -Value $null }"
        )
        cmd = ["powershell", "-NoProfile", "-Command", script]
        try:
            subprocess.run(cmd, capture_output=True, timeout=5.0, check=True)
        except Exception as e:
            raise RpcError("clipboard_error", f"Failed to set clipboard: {e}", {})
        return

    cmd = ["pbcopy"] if sys.platform == "darwin" else ["xclip", "-selection", "clipboard"]
    try:
        subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=5.0, check=True
        )
    except Exception as e:
        raise RpcError("clipboard_error", f"Failed to set clipboard: {e}", {})


def _file_list(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List directory contents via pathlib."""
    directory = Path(params.get("path", "."))
    if not directory.exists():
        raise RpcError("file_not_found", f"Directory not found: {directory}", {})
    entries = []
    for entry in sorted(directory.iterdir()):
        entries.append({
            "name": entry.name,
            "path": str(entry),
            "is_dir": entry.is_dir(),
            "size": entry.stat().st_size if entry.is_file() else 0,
        })
    return entries


def _file_move(params: Dict[str, Any]) -> Dict[str, Any]:
    src = Path(params["source"])
    dst = Path(params["destination"])
    src.rename(dst)
    return {"ok": True, "moved": str(dst)}


def _file_copy(params: Dict[str, Any]) -> Dict[str, Any]:
    import shutil as _shutil
    src = Path(params["source"])
    dst = Path(params["destination"])
    if src.is_dir():
        _shutil.copytree(str(src), str(dst))
    else:
        _shutil.copy2(str(src), str(dst))
    return {"ok": True, "copied": str(dst)}


def _file_delete(params: Dict[str, Any]) -> Dict[str, Any]:
    import shutil as _shutil
    target = Path(params["path"])
    if target.is_dir():
        _shutil.rmtree(str(target))
    else:
        target.unlink()
    return {"ok": True, "deleted": str(target)}


# ── Dispatch helpers ─────────────────────────────────────────────────────────

def _launch_app_key(app: str) -> str:
    """Pick the launch_app schema field for an app identifier.

    cua-driver launch_app accepts ``bundle_id`` (preferred) or
    ``name`` only. AUMIDs (``!``) and reverse-DNS identifiers (at least
    two dots, no path separators or spaces) are bundle ids; everything
    else — display names and executable paths — goes through ``name``.
    """
    if "!" in app:
        return "bundle_id"
    if (
        app.count(".") >= 2
        and "/" not in app
        and "\\" not in app
        and " " not in app
    ):
        return "bundle_id"
    return "name"


# ax.perform action → (cua tool, forced args). Covers the click tool's action
# vocabulary (press/show_menu/pick/confirm/cancel/open), the discrete
# double_click/right_click/type_text/set_value tools, and the legacy AX action
# names emitted by SemanticAdapter. Unknown actions fall back to a plain click.
_AX_ACTION_TABLE: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "click": ("click", {}),
    "press": ("click", {}),
    "AXPress": ("click", {}),
    "AXShowDefaultUI": ("click", {}),
    "open": ("click", {"action": "open"}),
    "pick": ("click", {"action": "pick"}),
    "select": ("click", {"action": "pick"}),
    "confirm": ("click", {"action": "confirm"}),
    "cancel": ("click", {"action": "cancel"}),
    "double_click": ("double_click", {}),
    "AXOpen": ("double_click", {}),
    "right_click": ("right_click", {}),
    "show_menu": ("right_click", {}),
    "AXShowMenu": ("right_click", {}),
    "type": ("type_text", {}),
    "type_text": ("type_text", {}),
    "set_value": ("set_value", {}),
}


def _element_target_args(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract cua-driver element/pixel target args from neutral params.

    ``element_token`` is preferred (it carries pid/window/snapshot); an
    int-like ``node_id`` is treated as an ``element_index``, anything else
    as a token. Pixel coordinates land as separate ``x``/``y`` fields —
    the click schema has no ``coordinates`` parameter.
    """
    args: Dict[str, Any] = {}

    if params.get("element_token"):
        args["element_token"] = params["element_token"]
    elif params.get("element_index") is not None:
        args["element_index"] = params["element_index"]
    else:
        node_id = str(params.get("node_id", "") or "")
        if node_id.isdigit():
            args["element_index"] = int(node_id)
        elif node_id:
            args["element_token"] = node_id

    for key in ("snapshot_id", "pid", "window_id"):
        if key in params:
            args[key] = params[key]

    coords = params.get("coordinates")
    if isinstance(coords, dict) and "x" in coords and "y" in coords:
        args["x"], args["y"] = coords["x"], coords["y"]
    elif isinstance(coords, (list, tuple)) and len(coords) == 2:
        args["x"], args["y"] = coords[0], coords[1]
    for key in ("x", "y"):
        if key in params:
            args[key] = params[key]
    return args


def _resolve_ax_perform_tool(params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Map ax.perform params to the appropriate cua-driver tool + args."""
    action = str(params.get("action", "") or "click")
    tool, forced = _AX_ACTION_TABLE.get(action, ("click", {}))

    args = _element_target_args(params)

    delivery_mode = params.get("delivery_mode", "background")
    if delivery_mode != "background":
        args["delivery_mode"] = delivery_mode

    if tool == "type_text" and "text" in params:
        args["text"] = params["text"]
    if tool == "set_value" and "value" in params:
        args["value"] = params["value"]

    args.update(forced)
    return tool, args


_SHORTCUT_SPLIT_RE = re.compile(r"[+\s]+")


def _normalize_shortcut_keys(keys: Any) -> List[str]:
    """Normalize a shortcut spec ('cmd+c', 'cmd c', or a list) to a key list."""
    if isinstance(keys, (list, tuple)):
        return [str(k).strip() for k in keys if str(k).strip()]
    text = str(keys or "").strip()
    if not text:
        return []
    return [part for part in _SHORTCUT_SPLIT_RE.split(text) if part]


# ── CuaDriverClient ──────────────────────────────────────────────────────────

class CuaDriverClient(HostRpc):
    """MCP stdio bridge to cua-driver, implementing HostRpc Protocol.

    Design principles:
    - AsyncBridge: background thread running asyncio event loop
    - Session management: MCP lifecycle_coro with enter/exit in same task
    - Capability negotiation: tools/list discovery at startup
    - Verify-Then-Escalate: AX background → PX pixel → foreground
    - Element Token: opaque token tracking for staleness detection
    - Heartbeat keepalive: periodic get_screen_size probe, auto-reconnect
    """

    def __init__(
        self,
        *,
        call_timeout: float = _CALL_TIMEOUT_S,
        keepalive_interval: float = _KEEPALIVE_INTERVAL_S,
        timeout_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._bridge = _AsyncBridge()
        self._session = _McpSession(self._bridge)
        self._call_timeout = call_timeout
        self._keepalive_interval = keepalive_interval
        self._keepalive_task: Optional[asyncio.Task] = None
        self._closed = False
        self._last_start_time: Optional[float] = None
        self._last_error = ""
        # Per-method-prefix timeout overrides. Exact method names win over
        # prefixes: app.list enumerates installed + running apps on Windows
        # (can exceed a minute when cold), far beyond any fast-path budget.
        self._timeout_map: Dict[str, float] = {
            "ping": 3.0,
            "ax": 8.0,
            "app": 30.0,
            "app.list": _APP_LIST_TIMEOUT_S,
            "input": 5.0,
            "screen": 10.0,
            "recording": 10.0,
            "clipboard": 3.0,
            "file": 15.0,
            "system": 5.0,
        }
        if timeout_overrides:
            self._timeout_map.update(timeout_overrides)

    def _resolve_timeout(self, method: str) -> float:
        """Resolve timeout by exact method name, then by method prefix."""
        exact = self._timeout_map.get(method)
        if exact is not None:
            return exact
        prefix = method.split(".", 1)[0] if method else ""
        return self._timeout_map.get(prefix, self._call_timeout)

    @property
    def connected(self) -> bool:
        """Return True when the cua-driver MCP session is active."""
        return self._session.started

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialize the bridge and MCP session."""
        self._closed = False
        try:
            self._session.start()
            self._start_keepalive()
        except Exception as exc:
            self._last_error = str(exc)
            self._bridge.stop()
            raise
        self._last_start_time = time.time()
        self._last_error = ""
        logger.info("CuaDriverClient started (tools: %d)", len(self._session.available_tools))

    def stop(self) -> None:
        """Gracefully shut down."""
        self._closed = True
        self._stop_keepalive()
        # Suppress expected "Process group termination failed" from the MCP
        # library during controlled shutdown — the fallback terminate works fine.
        _mcp_logger = logging.getLogger("mcp.os.posix.utilities")
        _prev_level = _mcp_logger.level
        _mcp_logger.setLevel(logging.CRITICAL)
        try:
            self._session.stop()
        finally:
            _mcp_logger.setLevel(_prev_level)
        self._bridge.stop()
        logger.info("CuaDriverClient stopped")

    def status_snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot for daemon and host status surfaces."""
        return {
            "backend": "cua-driver",
            "started": self._session.started,
            "closed": self._closed,
            "command": self._session.command,
            "args": self._session.args,
            "pid": None,
            "pid_source": "unavailable",
            "capability_version": self._session.capability_version,
            "tools_count": len(self._session.available_tools),
            "last_start_time": self._last_start_time,
            "last_error": self._last_error or self._session.last_error,
            "restart_count": self._session.restart_count,
        }

    def _start_keepalive(self) -> None:
        """Start periodic heartbeat on the bridge loop."""
        loop = self._bridge.loop
        if loop is None:
            return

        async def _heartbeat() -> None:
            while not self._closed:
                await asyncio.sleep(self._keepalive_interval)
                if self._closed:
                    break
                try:
                    # get_screen_size is an instant liveness round-trip;
                    # list_apps enumerates the UI tree (~20s on Windows)
                    # and would saturate the serial MCP pipe.
                    await self._session.call_tool("get_screen_size", {})
                except Exception as e:
                    logger.debug("keepalive probe failed: %s", e)
                    break

        self._keepalive_task = asyncio.run_coroutine_threadsafe(
            _heartbeat(), loop
        )

    def _stop_keepalive(self) -> None:
        fut = self._keepalive_task
        if fut is not None:
            fut.cancel()
            self._keepalive_task = None

    # ── HostRpc Protocol implementation ──────────────────────────────────

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch a LeapFlow RPC method to the appropriate handler.

        Routes Methods.* constants to cua-driver MCP tools or local
        implementations. Supports Verify-Then-Escalate on action tools.
        """
        params = params or {}
        timeout = self._resolve_timeout(method)

        # Local-only operations (no cua-driver roundtrip)
        handler = _LOCAL_DISPATCH.get(method)
        if handler is not None:
            return handler(params)

        # cua-driver tool dispatch (may raise _LocalResult for synthesized responses)
        try:
            tool_name, tool_args = self._map_to_cua_tool(method, params)
        except _LocalResult as lr:
            return lr.data

        result = await self._call_cua_tool(tool_name, tool_args, timeout)

        # Verify-Then-Escalate: check if response recommends escalation
        if self._should_escalate(result):
            escalated_args = self._apply_escalation(tool_args, result)
            result = await self._call_cua_tool(tool_name, escalated_args, timeout)

        return self._unwrap_result(result)

    async def _call_cua_tool(
        self, name: str, args: Dict[str, Any], timeout: float
    ) -> Dict[str, Any]:
        """Call a cua-driver MCP tool with reconnect-once semantics."""
        if self._session._session is None:
            raise RpcError("not_connected", "cua-driver session not active", {})
        try:
            return await asyncio.wait_for(
                self._session.call_tool(name, args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise RpcError("timeout", f"cua-driver {name} timed out after {timeout}s", {})
        except Exception as e:
            if not _is_closed_session_error(e):
                raise RpcError(
                    "cua_error",
                    f"cua-driver {name} failed: {e}",
                    {"tool": name, "original": str(e)},
                )
            # Reconnect once
            logger.warning("cua-driver session dropped during %s; reconnecting", name)
            with self._session._lock:
                self._session._restart()
            try:
                return await asyncio.wait_for(
                    self._session.call_tool(name, args),
                    timeout=timeout,
                )
            except Exception as retry_exc:
                raise RpcError(
                    "cua_reconnect_failed",
                    f"cua-driver {name} failed after reconnect: {retry_exc}",
                    {"tool": name},
                ) from retry_exc

    # ── Method → Tool mapping ────────────────────────────────────────────

    def _map_to_cua_tool(self, method: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Translate a LeapFlow Methods constant to (cua_tool_name, args)."""
        if method == Methods.AX_TREE:
            if "pid" not in params or "window_id" not in params:
                raise RpcError(
                    "invalid_params",
                    "ax.tree requires pid and window_id (discover them via ax.list)",
                    {"provided": sorted(params.keys())},
                )
            args: Dict[str, Any] = {
                "pid": params["pid"],
                "window_id": params["window_id"],
            }
            for key in ("query", "include_screenshot", "max_elements", "max_depth"):
                if key in params:
                    args[key] = params[key]
            return "get_window_state", args

        elif method == Methods.AX_LIST:
            return "list_windows", {}

        elif method == Methods.AX_PERFORM:
            return _resolve_ax_perform_tool(params)

        elif method == Methods.AX_SCROLL:
            direction = str(params.get("direction", "") or "")
            if direction not in ("up", "down", "left", "right"):
                raise RpcError(
                    "invalid_params",
                    f"scroll requires direction up/down/left/right, got '{direction}'",
                    {},
                )
            args = _element_target_args(params)
            args["direction"] = direction
            if "amount" in params:
                args["amount"] = int(params["amount"])
            return "scroll", args

        elif method == Methods.APP_LAUNCH:
            app = params.get("app_name") or params.get("name") or params.get("bundle_id", "")
            args: Dict[str, Any] = {}
            if app:
                args[_launch_app_key(app)] = app
            urls = params.get("urls")
            if isinstance(urls, list) and urls:
                # Driver-native: file paths/URLs handed to the app as open targets.
                args["urls"] = [str(u) for u in urls]
            return "launch_app", args

        elif method == Methods.APP_ACTIVATE:
            # launch_app is explicitly backgrounded; foreground activation
            # is bring_to_front, addressed by pid.
            if "pid" not in params:
                raise RpcError(
                    "invalid_params",
                    "app.activate requires pid (from ax.list or launch_app's response)",
                    {"provided": sorted(params.keys())},
                )
            args = {"pid": params["pid"]}
            if "window_id" in params:
                args["window_id"] = params["window_id"]
            return "bring_to_front", args

        elif method == Methods.APP_LIST:
            return "list_apps", {}

        elif method == Methods.INPUT_TYPE_TEXT:
            args = {"text": params.get("text", "")}
            args.update(_element_target_args(params))
            if "pid" not in args:
                # Without a pid/window target, desktop scope is the documented
                # way to type into the frontmost application.
                args["scope"] = "desktop"
            return "type_text", args

        elif method == Methods.INPUT_SHORTCUT:
            keys = params.get("keys", params.get("shortcut", ""))
            parts = _normalize_shortcut_keys(keys)
            if not parts:
                raise RpcError("invalid_params", "shortcut requires at least one key", {})
            if len(parts) == 1:
                # hotkey requires modifiers + one key (>=2 items); a bare key
                # (enter, escape, tab) is a press_key.
                args = {"key": parts[0]}
                tool = "press_key"
            else:
                args = {"keys": parts}
                tool = "hotkey"
            if "pid" in params:
                args["pid"] = params["pid"]
            else:
                args["scope"] = "desktop"
            return tool, args

        elif method == Methods.SCREEN_CAPTURE_FRAME:
            # Capture is window-scoped on cua-driver: get_window_state writes the
            # PNG for a (pid, window_id) pair via screenshot_out_file. The driver
            # exposes no full-display capture tool, so a targetless request is
            # refused with the same contract as ax.tree instead of being mapped
            # onto a tool that does not exist -- the earlier mapping pointed at
            # get_desktop_state, which only existed in an older driver line and
            # came back as "Unknown tool" at runtime.
            if "pid" not in params or "window_id" not in params:
                raise RpcError(
                    "invalid_params",
                    "screen.capture_frame requires pid and window_id (discover "
                    "them via ax.list); cua-driver has no full-display capture",
                    {"provided": sorted(params.keys())},
                )
            args = {
                "pid": params["pid"],
                "window_id": params["window_id"],
            }
            if "screenshot_out_file" in params:
                args["screenshot_out_file"] = params["screenshot_out_file"]
            return "get_window_state", args

        elif method == Methods.RECORDING_START:
            args: Dict[str, Any] = {}
            if "output_dir" in params:
                args["output_dir"] = params["output_dir"]
            if "record_video" in params:
                args["record_video"] = params["record_video"]
            return "start_recording", args

        elif method == Methods.RECORDING_STOP:
            return "stop_recording", {}

        elif method == Methods.PING:
            # Liveness probe only — callers never read the payload, and
            # list_apps' UI enumeration (~20s on Windows) would exceed
            # the 3s ping timeout.
            return "get_screen_size", {}

        elif method == Methods.SYSTEM_INFO:
            return self._build_system_info(params)

        elif method == Methods.SYSTEM_MANIFEST:
            return self._build_system_manifest(params)

        else:
            # Passthrough: treat method as direct tool name
            return method, params

    def _build_system_info(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """system.info is synthesized locally + from tool list."""
        # We return a sentinel that _unwrap_result handles
        raise _LocalResult({
            "platform": sys.platform,
            "arch": platform.machine(),
            "os_version": platform.version(),
            "cua_driver_cmd": _CUA_DRIVER_CMD,
            "capability_version": self._session.capability_version,
            "tools_available": sorted(self._session.available_tools.keys()),
        })

    def _build_system_manifest(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """system.manifest built from tools/list discovery."""
        raise _LocalResult({
            "capability_version": self._session.capability_version,
            "tools": {
                name: sorted(caps) for name, caps in self._session.available_tools.items()
            },
        })

    # ── Verify-Then-Escalate ─────────────────────────────────────────────

    @staticmethod
    def _should_escalate(result: Dict[str, Any]) -> bool:
        """Check if cua-driver recommends escalation to foreground/PX."""
        structured = result.get("structuredContent") or {}
        # Explicit escalation recommendation
        escalation = structured.get("escalation") or {}
        if escalation.get("recommended") == "foreground":
            return True
        # Degraded or suspected noop
        if structured.get("degraded") is True:
            return True
        if structured.get("suspected_noop") is True:
            return True
        return False

    @staticmethod
    def _apply_escalation(
        original_args: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify args for escalated retry (foreground delivery)."""
        args = dict(original_args)
        structured = result.get("structuredContent") or {}
        escalation = structured.get("escalation") or {}

        if escalation.get("recommended") == "foreground":
            args["delivery_mode"] = "foreground"
        elif structured.get("degraded") or structured.get("suspected_noop"):
            # Fall back to pixel coordinates if available
            if "coordinates" in structured:
                args["coordinates"] = structured["coordinates"]
            args["delivery_mode"] = "foreground"
        return args

    # ── Result unwrapping ────────────────────────────────────────────────

    @staticmethod
    def _unwrap_result(result: Dict[str, Any]) -> Any:
        """Unwrap the flattened tool result into caller-friendly form.

        Dict payloads gain ``ok: True`` (errors already raised) so callers'
        envelope checks hold, and MCP image blocks ride along as ``images``
        instead of being dropped when structured content is present.
        """
        if result.get("isError"):
            data = result.get("data", "unknown error")
            raise RpcError("cua_tool_error", str(data), result)

        images = result.get("images") or []

        def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(payload)
            if images and "images" not in out:
                out["images"] = images
            out.setdefault("ok", True)
            return out

        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return _finalize(structured)
        if structured is not None:
            return structured
        data = result.get("data")
        if isinstance(data, dict):
            return _finalize(data)
        if data is not None:
            return data
        if images:
            return {"ok": True, "images": images}
        return None


# ── Local dispatch table ─────────────────────────────────────────────────────

class _LocalResult(Exception):
    """Sentinel: call() intercepts this to return local data without cua-driver."""

    def __init__(self, data: Any) -> None:
        self.data = data


def _local_clipboard_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the PerceptionPort clipboard contract shape.

    The platform command cannot observe change counts, so change_count is 0
    and change_ts is the read time.
    """
    return {
        "ok": True,
        "text": _clipboard_get(),
        "change_count": 0,
        "change_ts": time.time(),
    }


def _local_clipboard_set(params: Dict[str, Any]) -> Dict[str, Any]:
    _clipboard_set(params.get("text", params.get("content", "")))
    return {"ok": True}


def _local_clipboard_last_change(params: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort: the platform command exposes no change counter.
    return {
        "ok": True,
        "text": _clipboard_get(),
        "change_count": 0,
        "change_ts": time.time(),
    }


def _local_fs_subscribe(params: Dict[str, Any]) -> Dict[str, Any]:
    """FS events are handled by Python observers (ObservationDaemon), not cua-driver."""
    return {"subscription_id": "local-observer-fs", "path": params.get("path", "")}


def _local_screen_permission_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Screen permission is managed by the OS; return best-effort status."""
    return {"status": "unknown", "message": "Permission managed by OS (check System Settings)"}


def _local_open_url(params: Dict[str, Any]) -> Dict[str, Any]:
    """Open a URL via the OS default browser.

    cua-driver's launch_app(urls=...) blocks until the browser window
    settles (~80s, and effectively forever when the URL lands in an
    already-running browser), so URL dispatch stays with the OS shell,
    which hands off to the default handler and returns immediately.
    """
    import webbrowser

    url = str(params.get("url", "")).strip()
    if not url:
        return {"ok": False, "error": "url required"}
    try:
        opened = webbrowser.open(url)
    except Exception as exc:
        return {"ok": False, "error": f"open_url failed: {exc}"}
    return {"ok": bool(opened), "url": url}


_LOCAL_DISPATCH: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    Methods.CLIPBOARD_GET: _local_clipboard_get,
    Methods.CLIPBOARD_SET: _local_clipboard_set,
    Methods.CLIPBOARD_LAST_CHANGE: _local_clipboard_last_change,
    Methods.FILE_LIST: _file_list,
    Methods.FILE_MOVE: _file_move,
    Methods.FILE_COPY: _file_copy,
    Methods.FILE_DELETE: _file_delete,
    Methods.FS_SUBSCRIBE: _local_fs_subscribe,
    Methods.OPEN_URL: _local_open_url,
    Methods.SCREEN_PERMISSION_STATUS: _local_screen_permission_status,
}
