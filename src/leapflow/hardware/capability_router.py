# Copyright (c) Alibaba, Inc. and its affiliates.
"""Capability router: maps declared affordances to callable tools.

When a device declares ``capabilities.affordances = ("grasp", "place", "push")``,
the router generates high-level semantic tools that the LLM agent can invoke
directly.  Each generated tool delegates to the appropriate device, channels,
and inference strategy through the existing HCP execution pipeline.

This is the bridge between *what a device can do* (declared in YAML) and
*what an agent can ask for* (tools in the LLM context).  Without it, the
agent must reason about raw channels and joint positions; with it, the
agent works at the task level.

The router is deliberately a thin routing layer.  It never implements
manipulation logic: an affordance command resolves to one of the existing
execution paths (policy inference or a batch actuate transaction), so every
physical write still flows through the same approval chain, trust ledger, and
audit record that a hand-written ``hw_*`` tool would.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from leapflow.hardware.context import HardwareEffect
from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)

# Metadata shared by every generated affordance tool.  An affordance always
# commands the physical world, so it is declared high-risk, state-mutating, and
# non-replayable: re-issuing "grasp" from an unknown pose is not a safe retry.
# ``effect_scope="external"`` is what makes the execution policy classify a
# failed affordance as an external side effect rather than an idempotent
# mutation, which is what stops the loop from replaying it blindly.
_AFFORDANCE_METADATA: dict[str, Any] = {
    "category": "hardware",
    "risk_level": "high",
    "requires_approval": True,
    "effect_scope": "external",
    "idempotency_scope": "session",
    "mutates_state": True,
    "execution_policy": "serial",
    "schema_cost": "low",
}


@dataclass(frozen=True)
class CapabilityEntry:
    """One device's ability to perform an affordance.

    Immutable snapshot built from a device's ``CapabilityDeclaration`` and its
    channel set.  ``actuate_channels`` are the writable ACTUATE endpoints a
    command can move; ``sensor_channels`` are the readable scalar/state
    endpoints used to observe current state and verify the result; and
    ``frame_channels`` are the camera endpoints an observation-driven policy
    consumes.
    """

    device_id: str
    affordance: str
    device_class: str  # "robot_arm", "mobile_robot", etc.
    kinematics: Any  # KinematicsDeclaration or None
    reach_m: float
    payload_kg: float
    actuate_channels: tuple[str, ...]  # writable ACTUATE channels
    sensor_channels: tuple[str, ...]  # readable channels for verification
    frame_channels: tuple[str, ...]  # camera channels

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "affordance": self.affordance,
            "device_class": self.device_class,
            "reach_m": self.reach_m,
            "payload_kg": self.payload_kg,
            "actuate_channels": list(self.actuate_channels),
            "sensor_channels": list(self.sensor_channels),
            "frame_channels": list(self.frame_channels),
        }


class CapabilityIndex:
    """Index of all affordances across all registered devices.

    Built from ``HardwareRegistry.contexts()`` -- scans every device's
    ``CapabilityDeclaration`` and builds a lookup from affordance name
    to the devices that can provide it.

    Thread-safe: rebuilt on demand when the registry changes.  Construction is
    lazy -- the first ``resolve``/``all_affordances`` call triggers a scan, and
    callers may force a re-scan with ``rebuild`` after the registry reloads.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._index: dict[str, list[CapabilityEntry]] = {}
        self._built = False
        self._lock = threading.RLock()

    def rebuild(self) -> None:
        """Scan all devices and rebuild the affordance index.

        Fail-soft: a registry that cannot enumerate its contexts yields an
        empty index rather than propagating, because a router that raises on
        access would take down tool assembly for every other plugin.
        """
        with self._lock:
            index: dict[str, list[CapabilityEntry]] = {}
            for context in self._iter_contexts():
                capabilities = getattr(context, "capabilities", None)
                if capabilities is None:
                    continue
                affordances = tuple(getattr(capabilities, "affordances", ()) or ())
                if not affordances:
                    continue
                actuate, sensors, frames = self._classify_channels(context)
                for affordance in affordances:
                    name = str(affordance).strip()
                    if not name:
                        continue
                    entry = CapabilityEntry(
                        device_id=context.device_id,
                        affordance=name,
                        device_class=str(getattr(context, "device_class", "") or ""),
                        kinematics=getattr(context, "kinematics", None),
                        reach_m=float(getattr(capabilities, "reach_m", 0.0) or 0.0),
                        payload_kg=float(getattr(capabilities, "payload_kg", 0.0) or 0.0),
                        actuate_channels=actuate,
                        sensor_channels=sensors,
                        frame_channels=frames,
                    )
                    index.setdefault(name, []).append(entry)
            self._index = index
            self._built = True

    def resolve(self, affordance: str) -> list[CapabilityEntry]:
        """Return all devices that declare the given affordance."""
        self._ensure_built()
        with self._lock:
            return list(self._index.get(str(affordance).strip(), ()))

    def all_affordances(self) -> frozenset[str]:
        """Return the union of all declared affordances."""
        self._ensure_built()
        with self._lock:
            return frozenset(self._index.keys())

    def _ensure_built(self) -> None:
        with self._lock:
            already = self._built
        if not already:
            self.rebuild()

    def _iter_contexts(self) -> tuple[Any, ...]:
        if self._registry is None:
            return ()
        try:
            return tuple(self._registry.contexts())
        except Exception as exc:  # noqa: BLE001 - a scan failure must not raise
            logger.warning("Capability index scan failed: %s", exc, exc_info=True)
            return ()

    @staticmethod
    def _classify_channels(context: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Partition a device's channels into actuate / sensor / frame sets.

        A channel is an actuator when it is writable and declares
        ``effect=actuate``; a sensor when it is readable and carries a value
        (not bytes); a frame when it carries media.  The partition is what lets
        the executor read state, drive motion, and feed a vision policy without
        re-deriving channel roles at every call.
        """
        actuate: list[str] = []
        sensors: list[str] = []
        frames: list[str] = []
        for channel in getattr(context, "channels", ()) or ():
            if getattr(channel, "is_media", False):
                frames.append(channel.channel_id)
                continue
            if channel.is_writable and channel.effect == HardwareEffect.ACTUATE.value:
                actuate.append(channel.channel_id)
            if channel.is_readable:
                sensors.append(channel.channel_id)
        return tuple(actuate), tuple(sensors), tuple(frames)


class CapabilityExecutor:
    """Executes a high-level affordance command on a device.

    This is the runtime component that receives a tool call like
    ``hw_grasp(device_id="robot.arm", target="red cup")`` and translates it
    into the HCP execution pipeline:

    1. Resolve device and capability entry from ``CapabilityIndex``.
    2. Read current state via ``registry.read_batch()``.
    3. Determine execution strategy:
       a. If ``PhysicalSkillPlugin`` has a loaded policy -> delegate to
          ``hw_policy_infer``.
       b. If the caller supplied an explicit joint-space target -> use
          ``hw_batch_actuate``.
       c. Otherwise -> return an error with a suggestion to load a policy.
    4. Execute through the approval chain (owned by the delegated tool).
    5. Optionally verify via the delegated tool's verification path.

    The executor does NOT implement manipulation logic itself -- it routes to
    the appropriate execution path.
    """

    def __init__(
        self,
        index: CapabilityIndex,
        registry: Any,
        skill_plugin: Any = None,  # PhysicalSkillPlugin reference
        gate: Any = None,  # approval gate
    ) -> None:
        self._index = index
        self._registry = registry
        self._skill_plugin = skill_plugin
        self._gate = gate
        self._hw_tools: Any = None

    async def execute(self, affordance: str, params: dict) -> dict:
        """Execute an affordance command.  Returns a structured result."""
        params = dict(params or {})
        entries = self._index.resolve(affordance)
        if not entries:
            return {
                "ok": False,
                "error": "unsupported_affordance",
                "affordance": affordance,
                "hint": (
                    "No admitted device declares this affordance. Declared "
                    f"affordances: {sorted(self._index.all_affordances()) or '(none)'}."
                ),
            }

        entry = self._select(entries, str(params.get("device_id") or ""))
        if entry is None:
            candidates = [e.device_id for e in entries]
            return {
                "ok": False,
                "error": "ambiguous_device",
                "affordance": affordance,
                "candidates": candidates,
                "hint": (
                    "Multiple devices provide this affordance; pass device_id to "
                    f"choose one of: {', '.join(candidates)}."
                ),
            }

        # Read current state (best effort): it is context for the operator and
        # the starting point a policy or verification compares against. A read
        # failure must not block the command -- it is diagnostic, not gating.
        state = await self._read_state(entry)
        verify = bool(params.get("verify"))

        # Strategy (a): a loaded policy owns the manipulation logic.
        policy = self._selected_policy()
        if policy is not None:
            result = await self._execute_via_policy(entry, affordance, params, policy)
            return self._decorate(result, entry, state, strategy="policy")

        # Strategy (b): an explicit joint-space target maps directly to a batch
        # actuate transaction; no inverse kinematics is performed here.
        commands = self._commands_from_position(entry, params.get("position"))
        if commands:
            result = await self._execute_via_batch(entry, commands, verify=verify)
            return self._decorate(result, entry, state, strategy="batch_actuate")

        # Strategy (c): nothing can turn the target into motion.
        return {
            "ok": False,
            "error": "no_execution_strategy",
            "affordance": affordance,
            "device_id": entry.device_id,
            "current_state": state,
            "suggestion": (
                "No policy is loaded and no explicit joint-space target was given. "
                "Load a manipulation policy (hw_policy_infer with a policy path), or "
                "issue joint positions directly via hw_batch_actuate. A natural-language "
                "target alone cannot be turned into motion without a policy."
            ),
        }

    # -- Strategy routing --------------------------------------------------

    async def _execute_via_policy(
        self, entry: CapabilityEntry, affordance: str, params: dict, policy: str
    ) -> dict:
        """Delegate to the physical skill plugin's policy inference tool."""
        task = str(params.get("target") or affordance)
        infer = getattr(self._skill_plugin, "policy_infer", None)
        if infer is None:
            return {"ok": False, "error": "skill_plugin_unavailable"}
        try:
            return await infer(
                {
                    "device_id": entry.device_id,
                    "policy": policy,
                    "task": task,
                    "execute": True,
                    "verify": bool(params.get("verify")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface as a structured failure
            logger.warning(
                "Affordance %s policy inference failed on %s: %s",
                affordance,
                entry.device_id,
                exc,
                exc_info=True,
            )
            return {"ok": False, "error": "policy_inference_failed", "detail": str(exc)}

    async def _execute_via_batch(
        self, entry: CapabilityEntry, commands: list[dict], *, verify: bool
    ) -> dict:
        """Route a joint-space target through the batch actuate tool.

        Builds a ``HardwareTools`` instance lazily so the full approval chain,
        envelope checks, and audit record apply exactly as they would for a
        hand-issued ``hw_batch_actuate`` call.
        """
        tools = self._hardware_tools()
        if tools is None:
            return {"ok": False, "error": "hardware_tools_unavailable"}
        try:
            return await tools.batch_actuate(
                {"device_id": entry.device_id, "commands": commands, "verify": verify}
            )
        except Exception as exc:  # noqa: BLE001 - surface as a structured failure
            logger.warning(
                "Affordance batch actuate failed on %s: %s",
                entry.device_id,
                exc,
                exc_info=True,
            )
            return {"ok": False, "error": "batch_actuate_failed", "detail": str(exc)}

    def _hardware_tools(self) -> Any:
        if self._hw_tools is None and self._registry is not None:
            from leapflow.hardware.tools import HardwareTools

            self._hw_tools = HardwareTools(self._registry, gate=self._gate)
        return self._hw_tools

    async def _read_state(self, entry: CapabilityEntry) -> dict[str, Any]:
        """Read the device's sensor and actuator channels, best effort."""
        channels = tuple(dict.fromkeys((*entry.sensor_channels, *entry.actuate_channels)))
        if not channels or self._registry is None:
            return {"available": False, "reason": "no readable channels"}
        try:
            batch = await self._registry.read_batch(entry.device_id, channels)
        except Exception as exc:  # noqa: BLE001 - state read is diagnostic, not gating
            logger.debug("State read failed for %s: %s", entry.device_id, exc, exc_info=True)
            return {"available": False, "reason": str(exc)}
        readings = getattr(batch, "readings", ()) or ()
        values = {
            getattr(r, "channel_id", ""): getattr(r, "value", None)
            for r in readings
        }
        return {"available": True, "values": values}

    # -- Resolution helpers ------------------------------------------------

    @staticmethod
    def _select(entries: list[CapabilityEntry], device_id: str) -> CapabilityEntry | None:
        """Pick the entry to act on: the named device, or the sole candidate."""
        if device_id:
            return next((e for e in entries if e.device_id == device_id), None)
        if len(entries) == 1:
            return entries[0]
        return None

    def _selected_policy(self) -> str | None:
        """Return a loaded policy key from the skill plugin, if any.

        Reads the plugin's policy cache directly: without a cached policy there
        is no path to hand ``hw_policy_infer``, so the executor falls through to
        the batch or error strategy instead of guessing one.
        """
        if self._skill_plugin is None:
            return None
        policies = getattr(self._skill_plugin, "_policies", None)
        if not policies:
            return None
        # The most recently loaded policy is the most likely intended one.
        try:
            return next(reversed(list(policies.keys())))
        except StopIteration:
            return None

    @staticmethod
    def _commands_from_position(entry: CapabilityEntry, position: Any) -> list[dict]:
        """Turn an explicit joint-space ``position`` mapping into batch commands.

        Only a mapping of ``channel_id -> value`` naming this device's actuate
        channels is accepted.  A cartesian ``{x, y, z}`` target is deliberately
        rejected here: converting it to joint angles is inverse kinematics, and
        the executor refuses to guess it.
        """
        if not isinstance(position, dict):
            return []
        actuate = set(entry.actuate_channels)
        commands: list[dict] = []
        for channel_id, value in position.items():
            if channel_id in actuate and isinstance(value, (int, float)):
                commands.append({"channel_id": channel_id, "value": float(value)})
        return commands

    @staticmethod
    def _decorate(
        result: dict, entry: CapabilityEntry, state: dict, *, strategy: str
    ) -> dict:
        """Attach routing provenance to a delegated tool result."""
        if not isinstance(result, dict):
            result = {"ok": False, "error": "invalid_result"}
        result.setdefault("device_id", entry.device_id)
        result["affordance"] = entry.affordance
        result["strategy"] = strategy
        result.setdefault("current_state", state)
        return result


class CapabilityToolGenerator:
    """Generates ``ToolMetadata`` for each declared affordance.

    For each affordance in the ``CapabilityIndex``, generates one tool:
    - name: ``hw_{affordance}`` (e.g. ``hw_grasp``, ``hw_place``)
    - parameters: device_id (auto-selected if only one), target description,
      optional position and verify flag
    - x_leapflow: category=hardware, risk_level=high, mutates_state=True
    - handler: delegates to ``CapabilityExecutor``

    Generated tools are registered through the plugin assembly pipeline, so
    they appear in the PCD tool catalog and go through the full approval chain.
    """

    def __init__(self, index: CapabilityIndex, executor: CapabilityExecutor) -> None:
        self._index = index
        self._executor = executor

    def generate(self) -> list[Any]:  # list[ToolMetadata]
        """Generate one ToolMetadata per unique affordance."""
        tools: list[ToolMetadata] = []
        for affordance in sorted(self._index.all_affordances()):
            entries = self._index.resolve(affordance)
            if not entries:
                continue
            tools.append(self._tool_for_affordance(affordance, entries))
        return tools

    def _tool_for_affordance(
        self, affordance: str, entries: list[CapabilityEntry]
    ) -> Any:
        """Build ToolMetadata for one affordance."""
        devices = ", ".join(sorted(e.device_id for e in entries))
        description = (
            f"Perform the '{affordance}' affordance on a robot device. This commands "
            "physical motion at the task level: describe the target in natural language "
            "and the router selects the device, reads current state, and routes to a "
            "loaded manipulation policy (or an explicit joint-space command). Physical "
            "and irreversible -- a repeat after a failure is not a safe retry. "
            f"Available on: {devices}."
        )
        schema = {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": (
                        "Target device from hw_list (optional if only one device "
                        f"provides '{affordance}')."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        f"What to {affordance} (natural language, e.g. 'the red cup')."
                    ),
                },
                "position": {
                    "type": "object",
                    "description": (
                        "Optional explicit joint-space target: a mapping of actuate "
                        "channel_id to value. Used only when no manipulation policy is "
                        "loaded; a cartesian {x, y, z} target is not accepted here."
                    ),
                },
                "verify": {
                    "type": "boolean",
                    "description": "Run post-operation verification (default false).",
                },
            },
            "required": [],
        }
        return ToolMetadata(
            name=f"hw_{affordance}",
            description=description,
            parameters_schema=schema,
            handler=self._make_handler(affordance),
            x_leapflow=dict(_AFFORDANCE_METADATA),
            mutates_state=True,
            execution_policy="serial",
            provides_capabilities=(f"hw.affordance.{affordance}",),
            requires_capabilities=("hardware_control",),
        )

    def _make_handler(self, affordance: str) -> Any:
        """Return an async handler bound to one affordance.

        Accepts keyword arguments so ``invoke_tool_handler`` dispatches the
        LLM-provided arguments straight through to the executor.
        """
        executor = self._executor

        async def handler(**kwargs: Any) -> dict:
            return await executor.execute(affordance, kwargs)

        return handler


class CapabilityRouterPlugin:
    """ToolPlugin that exposes auto-generated affordance tools.

    Dependencies:
    - hardware_registry: to scan device capabilities
    - hardware_approval_gate: for generated tools' approval chain
    - physical_skill_plugin: optional, for policy-based execution

    Tool generation is lazy: tools are generated on first access to
    ``self.tools`` and regenerated when the registry changes.
    """

    plugin_id = "capability_router"
    category = "hardware"
    dependencies = ["hardware_registry", "hardware_approval_gate"]

    def __init__(self) -> None:
        self._registry: Any = None
        self._gate: Any = None
        self._skill_plugin: Any = None
        self._tools: list[ToolMetadata] | None = None

    def bind_runtime(self, **deps: Any) -> None:
        """Receive the registry, the approval gate, and an optional skill plugin.

        The registry is optional on purpose: with hardware disabled nothing
        binds it, ``tools`` stays empty, and the tool index is byte-identical to
        a build without this plugin. Any dependency change invalidates the
        generated tool cache so a registry reload is reflected on next access.
        """
        changed = False
        if "hardware_registry" in deps:
            self._registry = deps.get("hardware_registry")
            changed = True
        if "hardware_approval_gate" in deps:
            self._gate = deps.get("hardware_approval_gate")
            changed = True
        # Optional: the physical skill plugin enables the policy execution path.
        # Not in ``dependencies`` because it is an enhancement, not a requirement;
        # accepted opportunistically when the assembler injects it.
        if "physical_skill_plugin" in deps:
            self._skill_plugin = deps.get("physical_skill_plugin")
            changed = True
        if changed:
            self._tools = None

    @property
    def tools(self) -> list[ToolMetadata]:
        """Return generated affordance tools, empty if no device has capabilities."""
        if self._registry is None:
            return []
        if self._tools is None:
            index = CapabilityIndex(self._registry)
            executor = CapabilityExecutor(
                index,
                self._registry,
                skill_plugin=self._skill_plugin,
                gate=self._gate,
            )
            generator = CapabilityToolGenerator(index, executor)
            self._tools = generator.generate()
        return list(self._tools)


plugin = CapabilityRouterPlugin()


__all__ = [
    "CapabilityEntry",
    "CapabilityExecutor",
    "CapabilityIndex",
    "CapabilityRouterPlugin",
    "CapabilityToolGenerator",
    "plugin",
]
