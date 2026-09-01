"""Transport that drives a device through an MCP server.

The second southbound implementation, and therefore the first real test of the
claim the transport seam was built on: adding a standard is a module plus a lookup
row, with nothing above the seam changing. Whether that held is measurable in the
diff rather than assertable in a docstring.

Everything device-specific is *declared*, never inferred. Tool names, argument
names and the response key carrying the value all come from the declaration:

    transport:
      kind: mcp
      config:
        server: bench-mcp          # informational; the tool name is what routes
        read_tool: bench_read
        write_tool: bench_write
        probe_tool: bench_status   # optional; absent means report local state only
        halt_tool: bench_estop     # optional; absent means halt is unsupported
        channel_arg: channel       # argument carrying the channel id
        value_arg: value           # argument carrying the commanded value
        value_path: value          # response key holding the reading
        extra_args: {rig: "A"}     # merged into every call
        sequence_path: seq         # optional; see the note on drop detection

There is deliberately no name matching, no "try these keys" chain, and no verb
enumeration. A tool this transport was not told about is not called, and a response
shape it was not told about is an error naming the keys that did arrive -- because a
guess that lands on the wrong tool is a physical action nobody authorised.

Governance: a write through this transport is gated once, by the hardware approval
descriptor, which is the only gate that knows the device, channel, value and
envelope. The MCP *tool* gate that fronts model-issued calls is not re-applied here.
Prompting twice for one physical action is not twice the safety -- it is how people
learn to click through prompts, and the second prompt would describe the call in
transport terms the operator cannot evaluate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

logger = logging.getLogger(__name__)

_CLIENT_PROVIDER: Callable[[], Any] | None = None
"""Process-wide resolver for the MCP client, installed by the runtime.

A provider rather than the client itself: MCP servers are reconfigured at runtime
(``leap config`` reloads them), so a captured client would outlive the session it
belongs to. Returns an undo callable for the same reason ``register_transport``
does -- this is a global mutation with a lifetime.
"""


def set_mcp_client_provider(provider: Callable[[], Any] | None) -> Callable[[], None]:
    """Install the resolver used when a declaration does not inject a client.

    Returns the undo callable so a caller can restore the previous provider; the
    table is process-global and a test or a reload must be able to put it back.
    """
    global _CLIENT_PROVIDER
    previous = _CLIENT_PROVIDER
    _CLIENT_PROVIDER = provider

    def _undo() -> None:
        global _CLIENT_PROVIDER
        _CLIENT_PROVIDER = previous

    return _undo


class McpTransport:
    """Six-method transport over one MCP server's declared tools."""

    kind = "mcp"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self._server = str(config.get("server") or "")
        self._read_tool = str(config.get("read_tool") or "")
        self._write_tool = str(config.get("write_tool") or "")
        self._probe_tool = str(config.get("probe_tool") or "")
        self._halt_tool = str(config.get("halt_tool") or "")
        self._channel_arg = str(config.get("channel_arg") or "channel")
        self._value_arg = str(config.get("value_arg") or "value")
        self._value_path = str(config.get("value_path") or "value")
        self._quality_path = str(config.get("quality_path") or "quality")
        self._sequence_path = str(config.get("sequence_path") or "")
        raw_extra = config.get("extra_args")
        self._extra_args: dict[str, Any] = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
        # A declaration may state that the device cannot stop even when a halt tool
        # exists; it may not claim the reverse, because a halt tool that was never
        # named cannot be called.
        self._halt_supported = bool(config.get("halt_supported", True)) and bool(self._halt_tool)
        self._injected_client = config.get("client")
        self._connected = False
        self._context: HardwareContext | None = None
        self._sequence: dict[str, int] = {}

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Bind the declaration and resolve a client. Idempotent.

        Validated here rather than at first use: a declaration missing the tool for a
        channel it exposes is a configuration fault, and surfacing it at admission
        time is the difference between an unusable device and a device that fails
        halfway through an experiment.

        After structural validation, the declared tool names are cross-checked
        against the MCP server's actual capability list.  A mismatch is fail-closed:
        a tool this transport was not told about is not called, and a tool that does
        not exist on the server will fail on every invocation.
        """
        self._context = context
        client = self._require_client()
        if any(channel.is_readable for channel in context.channels) and not self._read_tool:
            raise TransportError(
                "mcp transport exposes readable channels but declares no read_tool",
                failure_code="mcp_read_tool_missing",
            )
        if any(channel.is_writable for channel in context.channels) and not self._write_tool:
            raise TransportError(
                "mcp transport exposes writable channels but declares no write_tool",
                failure_code="mcp_write_tool_missing",
            )
        # Cross-check declared tool names against the server's actual capabilities.
        await self._validate_server_capabilities(client)
        self._connected = True
        return await self.probe()

    async def _validate_server_capabilities(self, client: Any) -> None:
        """Verify that every declared tool exists on the MCP server.

        Fail-closed: a declared tool that the server does not advertise will
        fail on every invocation, so refusing at open() is strictly better than
        failing halfway through an experiment.  The check is skipped when the
        server does not expose a capability list (older servers, non-standard
        implementations).

        ``list_tools`` may be synchronous (returning a list directly) or
        asynchronous (returning a coroutine).  Both shapes are accepted.
        """
        server_tools: set[str] | None = None
        try:
            # MCP clients typically expose server capabilities via list_tools()
            # or a tools property.  We try both common shapes.
            if hasattr(client, "list_tools"):
                raw = client.list_tools()
                tool_list = await raw if asyncio.iscoroutine(raw) else raw
                if isinstance(tool_list, (list, tuple)):
                    server_tools = set()
                    for entry in tool_list:
                        if isinstance(entry, str):
                            server_tools.add(entry)
                        elif hasattr(entry, "name"):
                            server_tools.add(str(entry.name))
                        elif isinstance(entry, dict) and "name" in entry:
                            server_tools.add(str(entry["name"]))
            elif hasattr(client, "tools"):
                raw = client.tools
                if isinstance(raw, (list, tuple)):
                    server_tools = set()
                    for entry in raw:
                        if isinstance(entry, str):
                            server_tools.add(entry)
                        elif hasattr(entry, "name"):
                            server_tools.add(str(entry.name))
                        elif isinstance(entry, dict) and "name" in entry:
                            server_tools.add(str(entry["name"]))
        except Exception as exc:  # noqa: BLE001 - best-effort capability check
            logger.debug(
                "Could not enumerate MCP server tools for %s: %s",
                self._server, exc, exc_info=True,
            )
            return  # Cannot check — degrade gracefully, do not fail.

        if server_tools is None:
            return  # Server does not expose a capability list.

        declared = [
            ("read_tool", self._read_tool),
            ("write_tool", self._write_tool),
            ("probe_tool", self._probe_tool),
            ("halt_tool", self._halt_tool),
        ]
        missing: list[str] = []
        for role, name in declared:
            if name and name not in server_tools:
                missing.append(f"{role}={name!r}")
        if missing:
            raise TransportError(
                f"MCP server {self._server!r} does not advertise the following "
                f"declared tools: {', '.join(missing)}. Available: "
                f"{sorted(server_tools)}",
                failure_code="mcp_capability_mismatch",
            )

    async def close(self) -> TransportStatus:
        # Must never raise: teardown runs on paths where an exception would mask the
        # failure that caused it. The MCP session is owned by the runtime, not by this
        # transport, so there is nothing here to tear down beyond the local flag.
        self._connected = False
        return TransportStatus(
            connected=False, halt_supported=self._halt_supported, detail="closed"
        )

    async def probe(self) -> TransportStatus:
        """Report health, calling the declared probe tool when there is one.

        Side-effect free by declaration: naming a tool here is the operator asserting
        that calling it is free, exactly as naming a read tool is.
        """
        if not self._probe_tool:
            return TransportStatus(
                connected=self._connected,
                halt_supported=self._halt_supported,
                detail=f"mcp transport ({self._server or 'unnamed server'})",
                metadata={"server": self._server, "probe_tool": ""},
            )
        response = await self._call(self._probe_tool, dict(self._extra_args))
        failure = _failure_of(response)
        return TransportStatus(
            connected=self._connected and failure is None,
            halt_supported=self._halt_supported,
            detail=failure or f"mcp transport ({self._server or 'unnamed server'})",
            metadata={"server": self._server, "probe_tool": self._probe_tool},
        )

    async def halt(self) -> TransportStatus:
        """Stop the device, or report that it cannot be stopped.

        Returns ``halt_supported=False`` rather than raising when no halt tool was
        declared. The registry then withdraws every writable channel for the device,
        so "cannot stop" degrades the capability instead of being assumed away.
        """
        if not self._halt_supported:
            return TransportStatus(
                connected=self._connected,
                halt_supported=False,
                detail="no halt_tool declared for this device",
            )
        response = await self._call(self._halt_tool, dict(self._extra_args))
        failure = _failure_of(response)
        return TransportStatus(
            connected=self._connected,
            halt_supported=True,
            detail=failure or "halted",
        )

    # ── Data plane ──

    async def read(self, channel_id: str) -> Reading:
        self._require_channel(channel_id)
        response = await self._call(
            self._read_tool, {self._channel_arg: channel_id, **self._extra_args}
        )
        failure = _failure_of(response)
        if failure is not None:
            raise TransportError(
                f"mcp read of {channel_id!r} failed: {failure}",
                failure_code="mcp_read_failed",
            )
        channel = self._context.channel(channel_id) if self._context is not None else None
        return Reading(
            device_id=self._context.device_id if self._context is not None else "",
            channel_id=channel_id,
            value=self._extract(response, self._value_path, channel_id),
            quantity=channel.quantity if channel is not None else "",
            unit=channel.unit if channel is not None else "",
            sequence=self._next_sequence(channel_id, response),
            quality=str(_dig(response, self._quality_path) or Quality.OK.value),
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Command a channel, reporting whether the effect may have landed.

        The verdict on failure is ``UNKNOWN`` unless the server states otherwise, and
        that is not caution for its own sake: the MCP client turns a timeout into an
        ordinary error reply, so a failure here genuinely cannot distinguish "never
        sent" from "sent, no answer". Reporting ``NONE`` would let the recovery layer
        replay a physical command that already executed.
        """
        self._require_channel(channel_id)
        response = await self._call(
            self._write_tool,
            {self._channel_arg: channel_id, self._value_arg: value, **self._extra_args},
        )
        failure = _failure_of(response)
        if failure is not None:
            declared = str(_dig(response, "side_effect_state") or "").strip().lower()
            state = declared if declared in _SIDE_EFFECT_STATES else SIDE_EFFECT_UNKNOWN
            return WriteOutcome(
                ok=False,
                side_effect_state=state,
                error=failure,
                failure_code=str(_dig(response, "failure_code") or "mcp_write_failed"),
                raw=_as_mapping(response),
            )
        readback = await self.read(channel_id) if self._needs_readback(channel_id) else None
        return WriteOutcome(
            ok=True,
            side_effect_state=SIDE_EFFECT_COMMITTED,
            readback=readback,
            settled=self._settling_time(channel_id) <= 0.0,
            raw=_as_mapping(response),
        )

    # ── Internals ──

    def _require_client(self) -> Any:
        client = self._injected_client
        if client is None and _CLIENT_PROVIDER is not None:
            try:
                client = _CLIENT_PROVIDER()
            except Exception as exc:  # noqa: BLE001 - a broken resolver must fail closed
                raise TransportError(
                    f"mcp client resolver failed: {exc}", failure_code="mcp_client_unavailable"
                ) from exc
        if client is None or not hasattr(client, "call_tool"):
            raise TransportError(
                "no MCP client is available for this device; the runtime installs one "
                "when MCP servers are configured",
                failure_code="mcp_client_unavailable",
            )
        return client

    def _require_channel(self, channel_id: str) -> None:
        if not self._connected:
            raise TransportError(
                f"transport for {channel_id!r} is not open", failure_code="transport_not_open"
            )
        known = self._context.channel(channel_id) if self._context is not None else None
        if known is None:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        client = self._require_client()
        try:
            return await client.call_tool(tool, arguments)
        except TransportError:
            raise
        except Exception as exc:  # noqa: BLE001 - a client fault is an unusable device
            raise TransportError(
                f"mcp call to {tool!r} raised {type(exc).__name__}: {exc}",
                failure_code="mcp_call_raised",
            ) from exc

    def _extract(self, response: Any, path: str, channel_id: str) -> Any:
        """Return the declared value, or fail naming the keys that did arrive.

        No fallback chain. Reading a different key than the one declared is how a
        transport reports one channel's value under another channel's identity, and
        nothing downstream can detect that.
        """
        found = _dig(response, path)
        if found is None:
            keys = sorted(_as_mapping(response).keys())
            raise TransportError(
                f"mcp read of {channel_id!r} returned no {path!r}; response carried {keys}",
                failure_code="mcp_value_path_missing",
            )
        return found

    def _next_sequence(self, channel_id: str, response: Any) -> int:
        """Prefer the server's sequence; fall back to a local counter.

        The fallback is a real loss of information and is documented as such: a local
        counter never gaps, so it cannot show that something was dropped between the
        device and the server. Only a server that numbers its own samples can.
        """
        if self._sequence_path:
            supplied = _dig(response, self._sequence_path)
            if isinstance(supplied, int) and not isinstance(supplied, bool):
                return supplied
        nxt = self._sequence.get(channel_id, 0) + 1
        self._sequence[channel_id] = nxt
        return nxt

    def _needs_readback(self, channel_id: str) -> bool:
        channel = self._context.channel(channel_id) if self._context is not None else None
        return bool(channel is not None and channel.verify_after_write and self._read_tool)

    def _settling_time(self, channel_id: str) -> float:
        channel = self._context.channel(channel_id) if self._context is not None else None
        return channel.envelope.settling_time_s if channel is not None else 0.0


_SIDE_EFFECT_STATES = frozenset({"none", "committed", "partial", "unknown"})


def _as_mapping(response: Any) -> dict[str, Any]:
    return dict(response) if isinstance(response, Mapping) else {}


def _dig(response: Any, path: str) -> Any:
    """Resolve a dotted path in a mapping, returning None when absent."""
    if not path:
        return None
    current: Any = response
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _failure_of(response: Any) -> str | None:
    """Return the failure text when a response reports one, else None.

    ``ok`` is honoured when present because that is the shape ``McpManager`` itself
    produces for a timeout or an unknown tool. A response with neither ``ok`` nor
    ``error`` is treated as success: many MCP tools simply return content, and
    demanding an envelope they never promised would make every such server unusable.
    """
    if not isinstance(response, Mapping):
        return None
    if response.get("ok") is False:
        return str(response.get("error") or "mcp tool reported failure")
    error = response.get("error")
    if error:
        return str(error)
    return None


def build_transport(config: Mapping[str, Any] | None = None) -> McpTransport:
    """Factory registered in the transport table."""
    return McpTransport(config)


__all__ = [
    "SIDE_EFFECT_NONE",
    "McpTransport",
    "build_transport",
    "set_mcp_client_provider",
]
