"""VSI Facade — single entry point exposing platform-agnostic ports to the engine."""

from __future__ import annotations

import logging
from typing import Optional

from leapflow.domain.platform import (
    Capability,
    PlatformID,
    PlatformManifest,
    capability_from_str,
)
from leapflow.platform.protocol import HostRpc

logger = logging.getLogger(__name__)


class VirtualSystemInterface:
    """Mediates between the agent logic layer and the platform-specific host.

    Lifecycle:
        1. Construct with a HostRpc instance.
        2. Call handshake() to discover host capabilities.
        3. Access perception / execution ports based on discovered capabilities.
    """

    def __init__(self, rpc: HostRpc) -> None:
        self._rpc = rpc
        self._manifest: Optional[PlatformManifest] = None

    @property
    def rpc(self) -> HostRpc:
        return self._rpc

    @property
    def manifest(self) -> PlatformManifest:
        if self._manifest is None:
            raise RuntimeError("VSI not initialized; call handshake() first")
        return self._manifest

    @property
    def is_initialized(self) -> bool:
        return self._manifest is not None

    async def handshake(self) -> PlatformManifest:
        """Perform capability handshake with the host process.

        Falls back to a default Darwin manifest if the host does not support
        the system.manifest RPC.

        When the underlying transport is CuaDriverClient, capabilities are
        derived from the tools/list discovery (no system.manifest RPC needed).
        """
        from leapflow.platform.cua_client import CuaDriverClient

        if isinstance(self._rpc, CuaDriverClient):
            self._manifest = _manifest_from_cua_tools(self._rpc)
            logger.info(
                "VSI handshake OK (cua-driver): caps=%d",
                len(self._manifest.capabilities),
            )
            return self._manifest

        try:
            result = await self._rpc.call("system.manifest")
            self._manifest = _parse_manifest(result)
            logger.info(
                "VSI handshake OK: platform=%s caps=%d",
                self._manifest.platform_id.value,
                len(self._manifest.capabilities),
            )
        except Exception as exc:
            logger.warning(
                "VSI handshake failed (%s); using default darwin manifest", exc
            )
            self._manifest = PlatformManifest.default_darwin()
        return self._manifest

    def can(self, cap: Capability) -> bool:
        """Check whether the current host supports a given capability."""
        return self.manifest.supports(cap)

    def can_all(self, *caps: Capability) -> bool:
        return self.manifest.supports_all(*caps)

    def can_any(self, *caps: Capability) -> bool:
        return self.manifest.supports_any(*caps)


def _parse_manifest(raw: dict) -> PlatformManifest:
    """Parse the raw RPC response into a PlatformManifest."""
    platform_str = str(raw.get("platform_id", "unknown"))
    try:
        platform_id = PlatformID(platform_str)
    except ValueError:
        platform_id = PlatformID.UNKNOWN

    os_version = str(raw.get("os_version", "0.0.0"))

    raw_caps = raw.get("capabilities", [])
    caps = frozenset(
        cap for s in raw_caps if (cap := capability_from_str(str(s))) is not None
    )

    metadata = dict(raw.get("metadata") or {})

    return PlatformManifest(
        platform_id=platform_id,
        os_version=os_version,
        capabilities=caps,
        metadata=metadata,
    )


# ── Capability mapping from cua-driver tools to PlatformManifest ─────────────

_CUA_TOOL_TO_CAPABILITIES: dict[str, list[str]] = {
    "get_window_state": ["accessibility", "ax_tree"],
    "click": ["accessibility", "ax_perform"],
    "type_text": ["accessibility", "input"],
    "set_value": ["accessibility", "ax_perform"],
    "scroll": ["accessibility", "input"],
    "hotkey": ["input"],
    "screenshot": ["screen_capture"],
    "launch_app": ["app_management"],
    "list_apps": ["app_management"],
    "start_recording": ["recording"],
    "stop_recording": ["recording"],
}


def _manifest_from_cua_tools(rpc: "CuaDriverClient") -> PlatformManifest:
    """Build a PlatformManifest from CuaDriverClient's discovered tools."""
    import platform as _platform

    from leapflow.platform.cua_client import CuaDriverClient

    session = rpc._session  # noqa: SLF001
    tools = session.available_tools

    # Derive capabilities from discovered tool names
    caps_strs: set[str] = set()
    for tool_name in tools:
        if tool_name in _CUA_TOOL_TO_CAPABILITIES:
            caps_strs.update(_CUA_TOOL_TO_CAPABILITIES[tool_name])

    caps = frozenset(
        cap for s in caps_strs if (cap := capability_from_str(s)) is not None
    )

    pid = PlatformID.resolve()

    return PlatformManifest(
        platform_id=pid,
        os_version=_platform.version(),
        capabilities=caps,
        metadata={
            "driver": "cua-driver",
            "capability_version": session.capability_version,
            "tools": sorted(tools.keys()),
        },
    )
