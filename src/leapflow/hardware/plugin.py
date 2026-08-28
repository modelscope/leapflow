"""Hardware context plugin -- a ToolPlugin, not a sibling subsystem.

Being an ordinary ``ToolPlugin`` is a deliberate structural choice, for two
reasons.

It inherits the whole engineering surface for free: discovery, topological
dependency injection, single-pass assembly, fiber lifecycle, hot reload,
sandboxing, manifest signing, the trust ledger, and usage tracking. None of it is
reimplemented here.

More importantly, it puts hardware on the governed path. Tools registered as
plugin metadata carry ``x_leapflow`` and therefore reach PCD disclosure and the
approval chain; a device exposed by bypassing that would execute physical commands
with no risk classification and no audit record. Reusing the plugin philosophy is
the governance decision, not a convenience.
"""

from __future__ import annotations

import logging
from typing import Any

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)


class HardwareContextPlugin:
    """Exposes admitted hardware devices through the eight generic tools."""

    def __init__(self) -> None:
        self._registry: Any = None
        self._gate: Any = None
        self._scope: Any = None
        self._session_id: str = ""
        self._tools: Any = None

    @property
    def plugin_id(self) -> str:
        return "hardware_context"

    @property
    def category(self) -> str:
        return "hardware"

    @property
    def dependencies(self) -> list[str]:
        return [
            "hardware_registry",
            "hardware_approval_gate",
            "effect_scope",
            "session_id",
        ]

    def bind_runtime(self, **deps: Any) -> None:
        """Receive the registry, the gate, and the scope that owns teardown.

        The registry is optional on purpose: with hardware disabled nothing binds
        it, ``tools`` stays empty, and the tool index is byte-identical to a build
        without this plugin. That property is what keeps the feature default-off and
        reversible, and it is also what keeps journey cassettes valid.
        """
        if "hardware_registry" in deps:
            self._registry = deps.get("hardware_registry")
        if "hardware_approval_gate" in deps:
            self._gate = deps.get("hardware_approval_gate")
        if "session_id" in deps:
            self._session_id = str(deps.get("session_id") or "")
        if "effect_scope" in deps:
            self._scope = deps.get("effect_scope")

        self._tools = None
        if self._registry is None:
            return

        self._register_teardown()

    def _register_teardown(self) -> None:
        """Close device connections when the owning scope unwinds.

        Registered through ``async_effect`` rather than ``effect``: ``close_all`` is
        a coroutine, and a coroutine handed to the synchronous variant is dropped
        without being awaited -- the connections would simply stay open.
        """
        if self._scope is None:
            return
        register = getattr(self._scope, "async_effect", None)
        if register is None:
            logger.warning(
                "Hardware plugin received an effect scope without async_effect; "
                "device connections will not be closed on teardown"
            )
            return
        try:
            register(self._registry.close_all)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Could not register hardware teardown effect: %s", exc, exc_info=True)

    @property
    def tools(self) -> list[ToolMetadata]:
        """Return the tool set, empty until a registry is bound."""
        if self._registry is None:
            return []
        if self._tools is None:
            from leapflow.hardware.tools import HardwareTools, build_hardware_tools

            self._tools = build_hardware_tools(
                HardwareTools(self._registry, gate=self._gate, session_id=self._session_id)
            )
        return list(self._tools)


plugin = HardwareContextPlugin()


__all__ = ["HardwareContextPlugin", "plugin"]
