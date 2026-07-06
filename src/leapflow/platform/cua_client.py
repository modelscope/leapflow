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
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
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
        return fut.result(timeout=timeout)

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

    @property
    def started(self) -> bool:
        return self._started

    @property
    def available_tools(self) -> Dict[str, Set[str]]:
        return self._tools

    @property
    def capability_version(self) -> str:
        return self._capability_version

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
            params = StdioServerParameters(
                command=command,
                args=args,
                env=_child_env(),
            )

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await self._discover_capabilities(session)
                    self._session = session
                    self._ready_event.set()
                    await self._shutdown_event.wait()
        except BaseException as e:
            self._setup_error = e
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
    if sys.platform == "darwin":
        cmd = ["pbpaste"]
    elif sys.platform == "win32":
        cmd = ["powershell", "-command", "Get-Clipboard"]
    else:
        cmd = ["xclip", "-selection", "clipboard", "-o"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
        return result.stdout
    except Exception as e:
        raise RpcError("clipboard_error", f"Failed to read clipboard: {e}", {})


def _clipboard_set(text: str) -> None:
    """Write clipboard via platform-native command."""
    if sys.platform == "darwin":
        cmd = ["pbcopy"]
    elif sys.platform == "win32":
        cmd = ["powershell", "-command", "Set-Clipboard", "-Value", text]
    else:
        cmd = ["xclip", "-selection", "clipboard"]

    try:
        if sys.platform == "win32":
            subprocess.run(cmd, capture_output=True, timeout=5.0, check=True)
        else:
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


def _file_move(params: Dict[str, Any]) -> Dict[str, str]:
    src = Path(params["source"])
    dst = Path(params["destination"])
    src.rename(dst)
    return {"moved": str(dst)}


def _file_copy(params: Dict[str, Any]) -> Dict[str, str]:
    import shutil as _shutil
    src = Path(params["source"])
    dst = Path(params["destination"])
    if src.is_dir():
        _shutil.copytree(str(src), str(dst))
    else:
        _shutil.copy2(str(src), str(dst))
    return {"copied": str(dst)}


def _file_delete(params: Dict[str, Any]) -> Dict[str, str]:
    import shutil as _shutil
    target = Path(params["path"])
    if target.is_dir():
        _shutil.rmtree(str(target))
    else:
        target.unlink()
    return {"deleted": str(target)}


# ── Dispatch helpers ─────────────────────────────────────────────────────────

def _resolve_ax_perform_tool(params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Map ax.perform params to the appropriate cua-driver tool + args.

    cua-driver splits AX actions into discrete tools: click, type_text,
    set_value. We infer the target from the `action` param.
    """
    action = params.get("action", "")
    element_index = params.get("element_index")
    element_token = params.get("element_token")

    # Common args shared across tools
    base_args: Dict[str, Any] = {}
    if element_index is not None:
        base_args["element_index"] = element_index
    if element_token is not None:
        base_args["element_token"] = element_token

    # Delivery mode for Verify-Then-Escalate
    delivery_mode = params.get("delivery_mode", "background")
    if delivery_mode != "background":
        base_args["delivery_mode"] = delivery_mode

    if action in ("click", "double_click", "right_click"):
        args = {**base_args, "action": action}
        if "coordinates" in params:
            args["coordinates"] = params["coordinates"]
        return "click", args

    elif action in ("type", "type_text"):
        args = {**base_args}
        if "text" in params:
            args["text"] = params["text"]
        return "type_text", args

    elif action == "set_value":
        args = {**base_args}
        if "value" in params:
            args["value"] = params["value"]
        return "set_value", args

    elif action == "select":
        args = {**base_args}
        return "click", args

    else:
        # Fallback: pass action directly as a click variant
        args = {**base_args, "action": action}
        return "click", args


# ── CuaDriverClient ──────────────────────────────────────────────────────────

class CuaDriverClient(HostRpc):
    """MCP stdio bridge to cua-driver, implementing HostRpc Protocol.

    Design principles:
    - AsyncBridge: background thread running asyncio event loop
    - Session management: MCP lifecycle_coro with enter/exit in same task
    - Capability negotiation: tools/list discovery at startup
    - Verify-Then-Escalate: AX background → PX pixel → foreground
    - Element Token: opaque token tracking for staleness detection
    - Heartbeat keepalive: periodic list_apps probe, auto-reconnect
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
        # Per-method-prefix timeout overrides
        self._timeout_map: Dict[str, float] = {
            "ping": 3.0,
            "ax": 8.0,
            "app": 5.0,
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
        """Resolve timeout by method prefix."""
        prefix = method.split(".", 1)[0] if method else ""
        return self._timeout_map.get(prefix, self._call_timeout)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialize the bridge and MCP session."""
        self._session.start()
        self._start_keepalive()
        logger.info("CuaDriverClient started (tools: %d)", len(self._session.available_tools))

    def stop(self) -> None:
        """Gracefully shut down."""
        self._closed = True
        self._stop_keepalive()
        self._session.stop()
        self._bridge.stop()
        logger.info("CuaDriverClient stopped")

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
                    await self._session.call_tool("list_apps", {})
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
            args: Dict[str, Any] = {}
            if "app" in params:
                args["app"] = params["app"]
            if "window_id" in params:
                args["window_id"] = params["window_id"]
            return "get_window_state", args

        elif method == Methods.AX_PERFORM:
            return _resolve_ax_perform_tool(params)

        elif method == Methods.AX_SCROLL:
            args = {}
            for key in ("x", "y", "direction", "amount", "coordinates", "element_index"):
                if key in params:
                    args[key] = params[key]
            return "scroll", args

        elif method == Methods.APP_LAUNCH:
            args = {"app_name": params.get("app_name", params.get("name", ""))}
            return "launch_app", args

        elif method == Methods.APP_ACTIVATE:
            args = {"app_name": params.get("app_name", params.get("name", ""))}
            return "launch_app", args

        elif method == Methods.APP_LIST:
            return "list_apps", {}

        elif method == Methods.INPUT_TYPE_TEXT:
            args = {"text": params.get("text", "")}
            return "type_text", args

        elif method == Methods.INPUT_SHORTCUT:
            # Parse key combo into cua-driver hotkey format
            keys = params.get("keys", params.get("shortcut", ""))
            args = {"keys": keys} if isinstance(keys, list) else {"key": keys}
            return "hotkey", args

        elif method == Methods.SCREEN_CAPTURE_FRAME:
            # Prefer screenshot tool if available, else get_window_state
            if self._session.has_tool("screenshot"):
                args = {}
                if "app" in params:
                    args["app"] = params["app"]
                return "screenshot", args
            else:
                args = {}
                if "app" in params:
                    args["app"] = params["app"]
                return "get_window_state", args

        elif method == Methods.RECORDING_START:
            return "start_recording", params

        elif method == Methods.RECORDING_STOP:
            return "stop_recording", params

        elif method == Methods.PING:
            return "list_apps", {}

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
        """Unwrap the flattened tool result into caller-friendly form."""
        if result.get("isError"):
            data = result.get("data", "unknown error")
            raise RpcError("cua_tool_error", str(data), result)
        # Prefer structuredContent, then data, then images
        structured = result.get("structuredContent")
        if structured is not None:
            return structured
        data = result.get("data")
        if data is not None:
            return data
        images = result.get("images")
        if images:
            return {"images": images}
        return None


# ── Local dispatch table ─────────────────────────────────────────────────────

class _LocalResult(Exception):
    """Sentinel: call() intercepts this to return local data without cua-driver."""

    def __init__(self, data: Any) -> None:
        self.data = data


def _local_clipboard_get(params: Dict[str, Any]) -> str:
    return _clipboard_get()


def _local_clipboard_set(params: Dict[str, Any]) -> None:
    _clipboard_set(params.get("text", params.get("content", "")))


def _local_clipboard_last_change(params: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort: return current content with no timestamp
    return {"content": _clipboard_get(), "timestamp": None}


def _local_fs_subscribe(params: Dict[str, Any]) -> Dict[str, Any]:
    """FS events are handled by Python observers (ObservationDaemon), not cua-driver."""
    return {"subscription_id": "local-observer-fs", "path": params.get("path", "")}


def _local_screen_permission_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Screen permission is managed by the OS; return best-effort status."""
    return {"status": "unknown", "message": "Permission managed by OS (check System Settings)"}


_LOCAL_DISPATCH: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    Methods.CLIPBOARD_GET: _local_clipboard_get,
    Methods.CLIPBOARD_SET: _local_clipboard_set,
    Methods.CLIPBOARD_LAST_CHANGE: _local_clipboard_last_change,
    Methods.FILE_LIST: _file_list,
    Methods.FILE_MOVE: _file_move,
    Methods.FILE_COPY: _file_copy,
    Methods.FILE_DELETE: _file_delete,
    Methods.FS_SUBSCRIBE: _local_fs_subscribe,
    Methods.SCREEN_PERMISSION_STATUS: _local_screen_permission_status,
}
