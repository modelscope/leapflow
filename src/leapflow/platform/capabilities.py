"""Runtime environment capability detection and caching.

Probes the platform connection, platform manifest, and permission state
to inform precondition checks and execution routing. Results are cached
with a configurable TTL to avoid redundant RPC calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from leapflow.domain.platform import Capability, PlatformManifest, capability_from_str
from leapflow.platform.protocol import HostRpc, Methods

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL = 30.0


@dataclass
class EnvironmentState:
    """Snapshot of the current runtime environment."""

    connected: bool = False
    manifest: PlatformManifest = field(default_factory=PlatformManifest.default_darwin)
    permissions: Dict[str, bool] = field(default_factory=dict)
    probed_at: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.probed_at if self.probed_at else float("inf")


class EnvironmentProbe:
    """Probes and caches runtime environment state.

    Single responsibility: query host capabilities. Depends on HostRpc
    abstraction (DIP) — works with real or mock RPC.
    """

    def __init__(
        self,
        rpc: HostRpc,
        manifest: PlatformManifest,
        *,
        cache_ttl: float = _DEFAULT_CACHE_TTL,
    ) -> None:
        self._rpc = rpc
        self._manifest = manifest
        self._cache_ttl = cache_ttl
        self._cached: Optional[EnvironmentState] = None

    @property
    def last_state(self) -> Optional[EnvironmentState]:
        return self._cached

    async def probe(self) -> EnvironmentState:
        """Query the environment, returning cached result if fresh."""
        if self._cached and self._cached.age_seconds < self._cache_ttl:
            return self._cached

        state = EnvironmentState(
            manifest=self._manifest,
            probed_at=time.time(),
        )

        try:
            await self._rpc.call(Methods.PING, {})
            state.connected = True
        except Exception:
            state.connected = False

        if state.connected:
            state.permissions = await self._probe_permissions()

        self._cached = state
        return state

    def invalidate(self) -> None:
        """Force re-probe on next access."""
        self._cached = None

    def satisfies(self, requirements: List[str]) -> Tuple[bool, List[str]]:
        """Check requirements against last probed state (synchronous).

        Returns (all_satisfied, list_of_missing).
        Unrecognized requirements are treated as satisfied (graceful degradation).
        """
        if self._cached is None:
            return True, []

        state = self._cached
        missing: List[str] = []

        for req in requirements:
            if req == "connected":
                if not state.connected:
                    missing.append("connected")
            elif req.startswith("capability:"):
                cap_str = req.split(":", 1)[1]
                cap = capability_from_str(cap_str)
                if cap and not state.manifest.supports(cap):
                    missing.append(req)
            elif req.startswith("permission:"):
                perm_name = req.split(":", 1)[1]
                if not state.permissions.get(perm_name, True):
                    missing.append(req)
            # Unknown requirement format → pass through (no hard rules)

        return len(missing) == 0, missing

    async def _probe_permissions(self) -> Dict[str, bool]:
        """Best-effort permission probing."""
        permissions: Dict[str, bool] = {}
        try:
            result = await self._rpc.call(Methods.SCREEN_PERMISSION_STATUS, {})
            if isinstance(result, dict):
                permissions["screen_recording"] = bool(result.get("granted", False))
        except Exception:
            pass

        if self._manifest.supports(Capability.AX_TREE_READ):
            permissions["accessibility"] = True

        return permissions
