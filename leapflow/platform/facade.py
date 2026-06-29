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
        the system.manifest RPC (backward compatibility with older OSHost).
        """
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
