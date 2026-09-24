# Copyright (c) Alibaba, Inc. and its affiliates.
"""Physical skill plugin: VLA policy inference as LeapFlow tools.

This plugin bridges the gap between learned manipulation policies and
LeapFlow's tool execution pipeline.  A VLA model's ``select_action``
becomes a tool call that the agent can invoke, with the full governance
chain (PCD disclosure, risk classification, approval, trust, audit)
applied to the physical execution that follows.

Inference is delegated to the ``InferenceStrategyRegistry``: a policy
string is resolved to an :class:`InferenceStrategy` (``VLALocalStrategy``
for a local checkpoint, ``VLARemoteStrategy`` for a ``remote:host:port``
address) that is registered under a unique ``strategy_id`` and reused
across sessions.  Both paths produce an :class:`InferenceResult` whose
action vector is written to the robot through ``BatchTransport.write_batch()``
(or sequential writes as fallback).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.tool_plugins._physical_skill_helpers import (
    action_to_commands as _action_to_commands,
    build_physical_skill_tools as _build_physical_skill_tools,
    coerce_compute_budget as _coerce_compute_budget,
    discover_local_policies as _discover_local_policies,
)
from leapflow.robot.inference.registry import (
    InferenceStrategyRegistry,
    get_default_registry,
)
from leapflow.robot.inference.strategy import (
    ComputeBudget,
    InferenceResult,
    InferenceStrategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class PhysicalSkillPlugin:
    """ToolPlugin that exposes VLA policy inference as LeapFlow tools.

    Dependencies (injected via bind_runtime):
    - hardware_registry: HardwareRegistry instance for device access
    - hardware_approval_gate: approval orchestrator for physical operations
    - hardware_trust_gate: trust gate for progressive trust
    - effect_scope: EffectScope for cleanup registration
    - session_id: current session identifier
    """

    def __init__(
        self,
        *,
        strategy_registry: InferenceStrategyRegistry | None = None,
    ) -> None:
        self._registry: Any = None
        self._gate: Any = None
        self._trust_gate: Any = None
        self._scope: Any = None
        self._session_id: str = ""
        self._tools: list[ToolMetadata] | None = None
        # Strategy cache: policy_path -> InferenceStrategy instance.
        # The plugin keeps its own reference so scope teardown can close only
        # the strategies this plugin instance is aware of, without unregistering
        # them from the process-wide default registry (a shared local model
        # loaded once must survive one session tearing down).
        self._policies: dict[str, InferenceStrategy] = {}
        # Default strategy registry; injected for tests, otherwise the
        # module-level singleton.  A shared registry lets a heavy local model
        # loaded by one session serve every subsequent session in the process.
        self._strategy_registry: InferenceStrategyRegistry = (
            strategy_registry if strategy_registry is not None else get_default_registry()
        )
        # Episode state (per-plugin, session-level, not persisted).  When an
        # episode is active this holds the strategy it was started with, so
        # every ``policy_infer`` inside it re-uses the same instance and its
        # chunked-action queue rather than a fresh strategy per turn.
        self._active_episode: dict[str, Any] | None = None
        self._teardown_registered: bool = False

    @property
    def plugin_id(self) -> str:
        return "physical_skill"

    @property
    def category(self) -> str:
        return "hardware"

    @property
    def dependencies(self) -> list[str]:
        return [
            "hardware_registry",
            "hardware_approval_gate",
            "hardware_trust_gate",
            "effect_scope",
            "session_id",
        ]

    def bind_runtime(self, **deps: Any) -> None:
        """Receive dependencies.  Registry is optional -- no registry means empty tools."""
        registry_changed = False
        if "hardware_registry" in deps:
            self._registry = deps.get("hardware_registry")
            registry_changed = True
        if "hardware_approval_gate" in deps:
            self._gate = deps.get("hardware_approval_gate")
        if "hardware_trust_gate" in deps:
            self._trust_gate = deps.get("hardware_trust_gate")
        if "session_id" in deps:
            self._session_id = str(deps.get("session_id") or "")
        if "effect_scope" in deps:
            self._scope = deps.get("effect_scope")
            self._teardown_registered = False

        if registry_changed:
            self._tools = None
            self._policies.clear()
            self._active_episode = None
            self._teardown_registered = False

        if self._registry is None:
            return

        self._register_teardown()

    @property
    def tools(self) -> list[ToolMetadata]:
        """Return tool metadata, empty until registry is bound."""
        if self._registry is None:
            return []
        if self._tools is None:
            self._tools = _build_physical_skill_tools(self)
        return list(self._tools)

    # -- Tool handlers ------------------------------------------------

    async def policy_infer(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute one inference step and optionally write the action to hardware."""
        device_id: str = params.get("device_id", "")
        policy_path: str = params.get("policy", "")
        execute: bool = params.get("execute", True)
        chunk_size: int = max(1, int(params.get("chunk_size", 1) or 1))
        verify: bool = params.get("verify", False)
        budget = _coerce_compute_budget(params.get("compute_budget"), chunk_size)

        if not device_id:
            return {"ok": False, "error": "device_id is required"}
        if not policy_path:
            return {"ok": False, "error": "policy path is required"}

        # 1. Resolve the inference strategy for this policy path.
        try:
            strategy = await self._load_policy(policy_path)
        except Exception as exc:
            return {"ok": False, "error": f"failed to load policy: {exc}"}

        # 2. Read current observation from the robot.
        try:
            observation = await self._read_observation(device_id)
        except Exception as exc:
            return {"ok": False, "error": f"failed to read observation: {exc}"}

        # 3. Run policy inference through the strategy contract.
        try:
            result = await self._run_inference(
                strategy, observation, budget=budget,
            )
        except Exception as exc:
            return {"ok": False, "error": f"inference failed: {exc}"}
        action = result.action

        # 4. Optionally write the action to the robot.
        execution: dict[str, Any] | None = None
        if execute:
            try:
                execution = await self._execute_action(
                    device_id, action, verify=verify,
                )
            except Exception as exc:
                return self._format_result(
                    result, {"ok": False, "error": str(exc)}, None,
                )

        # 5. Verification result is embedded in execution if requested.
        verification: dict[str, Any] | None = None
        if execution and "verification" in execution:
            verification = execution.pop("verification", None)

        # 6. Update episode stats if an episode is active.
        if self._active_episode is not None:
            self._active_episode["steps"] += 1
            self._active_episode["last_action_time"] = time.time()

        return self._format_result(result, execution, verification)

    async def policy_episode(self, params: dict[str, Any]) -> dict[str, Any]:
        """Manage episode lifecycle: start / stop / status."""
        action: str = params.get("action", "")
        if action not in ("start", "stop", "status"):
            return {"ok": False, "error": "action must be 'start', 'stop', or 'status'"}

        if action == "status":
            if self._active_episode is None:
                return {"ok": True, "active": False, "episode": None}
            elapsed = time.time() - self._active_episode["start_time"]
            return {
                "ok": True,
                "active": True,
                "episode": {
                    **self._active_episode,
                    "elapsed_s": round(elapsed, 2),
                },
            }

        if action == "start":
            device_id: str = params.get("device_id", "")
            policy_path: str = params.get("policy", "")
            task_desc: str = params.get("task", "")
            if not device_id:
                return {"ok": False, "error": "device_id is required for episode start"}
            if not policy_path:
                return {"ok": False, "error": "policy path is required for episode start"}

            if self._active_episode is not None:
                return {
                    "ok": False,
                    "error": "an episode is already active; stop it before starting a new one",
                }

            # Load and reset the policy.
            try:
                strategy = await self._load_policy(policy_path)
                await strategy.reset()
            except Exception as exc:
                return {"ok": False, "error": f"failed to initialise episode: {exc}"}

            self._active_episode = {
                "device_id": device_id,
                "policy": policy_path,
                "task": task_desc,
                "start_time": time.time(),
                "steps": 0,
                "last_action_time": None,
            }
            return {"ok": True, "action": "started", "episode": dict(self._active_episode)}

        # action == "stop"
        if self._active_episode is None:
            return {"ok": False, "error": "no active episode to stop"}

        episode = dict(self._active_episode)
        elapsed = time.time() - episode["start_time"]
        episode["elapsed_s"] = round(elapsed, 2)
        episode["avg_step_time_s"] = (
            round(elapsed / episode["steps"], 3) if episode["steps"] > 0 else None
        )
        self._active_episode = None
        return {"ok": True, "action": "stopped", "summary": episode}

    async def policy_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """List available policies and their capabilities."""
        source: str = params.get("source", "all")
        policies: list[dict[str, Any]] = []

        # List cached / loaded strategies keyed by policy path.
        if source in ("local", "all"):
            for path, strategy in self._policies.items():
                if path.startswith("remote:"):
                    continue
                entry: dict[str, Any] = {
                    "name": path,
                    "type": "local",
                    "loaded": True,
                    "strategy_id": getattr(strategy, "strategy_id", ""),
                }
                profile = getattr(strategy, "compute_profile", None)
                if profile is not None:
                    entry["latency_range_ms"] = list(profile.latency_range_ms)
                    entry["supports_chunking"] = profile.supports_chunking
                policies.append(entry)

        if source in ("remote", "all"):
            for path, strategy in self._policies.items():
                if not path.startswith("remote:"):
                    continue
                entry = {
                    "name": path,
                    "type": "remote",
                    "loaded": True,
                    "address": path.removeprefix("remote:"),
                    "strategy_id": getattr(strategy, "strategy_id", ""),
                }
                policies.append(entry)

        # Discover local policies through LeapRobot if available.
        if source in ("local", "all"):
            discovered = _discover_local_policies()
            # Deduplicate against already-loaded.
            loaded_names = {p["name"] for p in policies}
            for d in discovered:
                if d["name"] not in loaded_names:
                    policies.append(d)

        return {"ok": True, "policies": policies, "count": len(policies)}

    # -- Policy loading ------------------------------------------------

    async def _load_policy(self, policy_path: str) -> InferenceStrategy:
        """Resolve a policy path to an :class:`InferenceStrategy` instance.

        The strategy is registered in the shared registry under a unique
        ``strategy_id`` (``vla_local:<path>`` or ``vla_remote:<address>``)
        so a heavy local model loaded once is reused by every subsequent
        session, while the plugin keeps a local reference in
        ``self._policies`` so scope teardown can close the strategies it
        touched without unregistering shared instances.
        """
        cached = self._policies.get(policy_path)
        if cached is not None:
            return cached

        if policy_path.startswith("remote:"):
            address = policy_path.removeprefix("remote:")
            strategy_id = f"vla_remote:{address}"
            strategy = self._strategy_registry.get(strategy_id)
            if strategy is None:
                from leapflow.robot.inference.vla_remote import VLARemoteStrategy

                strategy = VLARemoteStrategy(server_address=address)
                # Override the class-level id so every remote endpoint gets
                # its own registry slot -- otherwise the second address would
                # collide with the first under the shared ``vla_remote`` key.
                strategy.strategy_id = strategy_id  # type: ignore[misc]
                self._strategy_registry.register(strategy)
            self._policies[policy_path] = strategy
            logger.info("Remote VLA strategy registered for %s", address)
            return strategy

        # Local policy path: build a VLALocalStrategy.  The strategy loads the
        # underlying model lazily on first infer, so construction is cheap and
        # does not require torch to be importable here.
        strategy_id = f"vla_local:{policy_path}"
        strategy = self._strategy_registry.get(strategy_id)
        if strategy is None:
            from leapflow.robot.inference.vla_local import VLALocalStrategy

            strategy = VLALocalStrategy(policy_path=policy_path)
            strategy.strategy_id = strategy_id  # type: ignore[misc]
            self._strategy_registry.register(strategy)
        self._policies[policy_path] = strategy
        logger.info("Local VLA strategy registered for %s", policy_path)
        return strategy

    async def _run_inference(
        self,
        strategy: InferenceStrategy,
        observation: dict[str, Any],
        *,
        budget: ComputeBudget | None = None,
    ) -> InferenceResult:
        """Run inference through the :class:`InferenceStrategy` contract.

        The strategy owns model loading, transport lifecycle, and any
        budget interpretation (chunk size, ensemble, refinement).  The
        plugin passes the observation through unchanged and receives an
        :class:`InferenceResult` whose ``action`` is turned into hardware
        commands downstream.
        """
        return await strategy.infer(observation, budget=budget)

    # -- Observation reading -------------------------------------------

    async def _read_observation(self, device_id: str) -> dict[str, Any]:
        """Read full observation via registry.  Prefers read_batch for atomicity."""
        if self._registry is None:
            raise RuntimeError("hardware registry is not bound")

        # Attempt batch read first.
        device = await self._resolve_device(device_id)
        transport = getattr(device, "transport", None)
        if transport is None:
            transport = device

        # Collect readable channel ids.
        channels: list[str] = []
        context = getattr(device, "context", None)
        if context is not None:
            for ch in context.channels:
                # Skip frame channels -- they are not part of observation vectors.
                if getattr(ch, "representation", "") == "frame":
                    continue
                channels.append(ch.channel_id)

        from leapflow.hardware.transport import BatchTransport

        if isinstance(transport, BatchTransport) and channels:
            batch = await transport.read_batch(tuple(channels))
            return {r.channel_id: r.value for r in batch.readings}

        # Fallback: sequential reads.
        obs: dict[str, Any] = {}
        for cid in channels:
            try:
                reading = await self._registry.read(device_id, cid)
                obs[cid] = reading.value
            except Exception:
                logger.debug(
                    "Failed to read channel %s/%s for observation",
                    device_id, cid, exc_info=True,
                )
        return obs

    # -- Action execution ----------------------------------------------

    async def _execute_action(
        self,
        device_id: str,
        action: Any,
        *,
        verify: bool = False,
    ) -> dict[str, Any]:
        """Write action vector to robot and optionally verify."""
        if self._registry is None:
            raise RuntimeError("hardware registry is not bound")

        device = await self._resolve_device(device_id)
        transport = getattr(device, "transport", None)
        if transport is None:
            transport = device

        # Build command tuples from the action vector.
        commands = _action_to_commands(action, device)

        # Approval gate -- every physical write must be authorised before it
        # touches an actuator.  Fail closed: an absent or failing gate denies.
        if commands and not await self._approve_write(device_id, commands):
            return {
                "ok": False,
                "side_effect_state": "none",
                "channels_written": 0,
                "error": "Physical operation not approved by hardware approval gate",
            }

        from leapflow.hardware.transport import BatchTransport, BatchWriteOutcome

        result: dict[str, Any]
        if isinstance(transport, BatchTransport) and commands:
            outcome: BatchWriteOutcome = await transport.write_batch(tuple(commands))
            result = {
                "ok": outcome.ok,
                "side_effect_state": outcome.side_effect_state,
                "channels_written": len(commands),
            }
        else:
            # Fallback: sequential writes.
            written = 0
            last_error = ""
            for cid, val in commands:
                try:
                    w = await self._registry.write(device_id, cid, val)
                    if w.ok:
                        written += 1
                    else:
                        last_error = w.error
                except Exception as exc:
                    last_error = str(exc)
            result = {
                "ok": written == len(commands),
                "channels_written": written,
                "channels_total": len(commands),
            }
            if last_error:
                result["last_error"] = last_error

        # Optional verification.
        if verify and result.get("ok"):
            verification = await self._verify_execution(device_id, action, commands)
            if verification:
                result["verification"] = verification

        return result

    async def _approve_write(
        self,
        device_id: str,
        commands: list[tuple[str, Any]],
    ) -> bool:
        """Fail-closed approval check for a physical write.

        Mirrors ``HardwareTools._evaluate``: no gate installed, or a gate that
        raises, both deny.  For a physical device the cost of a wrong "allow"
        is not measured in data, so a broken gate must never become an open
        door.
        """
        if self._gate is None:
            logger.error(
                "No hardware approval gate is bound; refusing physical write "
                "to %s (fail-closed).",
                device_id,
            )
            return False

        descriptor = self._build_approval_descriptor(device_id, commands)
        try:
            result = await self._gate.evaluate(descriptor)
        except Exception as exc:  # noqa: BLE001 - a failing gate must deny, not propagate
            logger.error(
                "Hardware approval gate raised for %s: %s",
                device_id, exc, exc_info=True,
            )
            return False
        return bool(getattr(result, "approved", False))

    def _build_approval_descriptor(
        self,
        device_id: str,
        commands: list[tuple[str, Any]],
    ) -> Any:
        """Build an ActionDescriptor for a batched physical actuation.

        Mirrors the batch descriptor built by ``HardwareTools`` so the same
        approval orchestrator evaluates policy-driven writes and direct
        hardware writes through one contract.
        """
        from leapflow.security.actions import ActionDescriptor, ActionKind

        channel_summaries = [
            {"channel_id": cid, "value": val} for cid, val in commands
        ]
        return ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id=device_id,
            channel_id=commands[0][0] if commands else "",
            quantity="batch",
            value=f"{len(commands)} channels",
            unit="",
            envelope_band="batch",
            metadata={
                "batch": True,
                "commands": channel_summaries,
                "session_id": self._session_id,
                "source": "hw_policy_infer",
            },
        )

    async def _verify_execution(
        self,
        device_id: str,
        action: Any,
        commands: list[tuple[str, Any]],
    ) -> dict[str, Any] | None:
        """Run post-execution verification using OperationVerifier."""
        if not commands:
            return None

        try:
            from leapflow.hardware.verification import collect_evidence
        except ImportError:
            return None

        # Verify the first commanded channel as representative.
        channel_id, intended = commands[0]
        try:
            evidence = await collect_evidence(
                self._registry,
                device_id,
                channel_id,
                intended_value=intended,
                outcome=None,
                settle_delay_s=0.05,
            )
        except Exception as exc:
            logger.debug("Evidence collection failed: %s", exc, exc_info=True)
            return {"status": "inconclusive", "detail": f"evidence collection failed: {exc}"}

        # Find an applicable verifier.
        verifiers = getattr(self._registry, "verifiers", None)
        if not verifiers:
            return {"status": "skipped", "detail": "no verifiers registered"}

        for v in verifiers:
            try:
                verdict = await v.verify(evidence)
                verdict_dict = verdict.to_dict()
                self._update_trust_from_verdict(
                    device_id, channel_id, verdict_dict.get("status"),
                )
                return verdict_dict
            except Exception as exc:
                logger.debug("Verifier %s failed: %s", v, exc, exc_info=True)

        return {"status": "skipped", "detail": "no applicable verifier succeeded"}

    def _update_trust_from_verdict(
        self,
        device_id: str,
        channel_id: str,
        status: str | None,
    ) -> None:
        """Feed a verification verdict into the progressive-trust gate.

        Only definitive outcomes move trust: ``success`` promotes, ``failure``
        demotes.  ``inconclusive`` and ``skipped`` leave trust untouched --
        absence of evidence is not evidence of failure.
        """
        if self._trust_gate is None:
            return
        from leapflow.hardware.verification import VerdictStatus

        try:
            if status == VerdictStatus.SUCCESS.value:
                self._trust_gate.record_success(device_id, channel_id)
            elif status == VerdictStatus.FAILURE.value:
                self._trust_gate.record_failure(device_id, channel_id)
        except Exception as exc:  # noqa: BLE001 - trust bookkeeping must not fail the turn
            logger.debug("Trust gate update failed: %s", exc, exc_info=True)

    async def _resolve_device(self, device_id: str) -> Any:
        """Resolve a device from the registry by id."""
        get_device = getattr(self._registry, "get_device", None)
        if get_device is not None:
            device = get_device(device_id)
            if device is not None:
                return device

        # Fallback: return the registry itself as a proxy.
        return self._registry

    def _format_result(
        self,
        result: InferenceResult,
        execution: dict[str, Any] | None,
        verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the tool result dict from an :class:`InferenceResult`.

        The ``action`` payload is JSON-serialised (tensors flattened to a
        list, dicts kept intact) while the strategy's latency and confidence
        metadata surface alongside so callers can drive test-time scaling
        without inspecting private fields.
        """
        action = result.action
        out: dict[str, Any] = {"ok": True}

        # Serialise the action for JSON transport.
        if hasattr(action, "tolist"):
            out["action"] = action.tolist()
        elif hasattr(action, "cpu"):
            out["action"] = action.cpu().tolist()
        elif isinstance(action, dict):
            out["action"] = action
        else:
            out["action"] = str(action) if action is not None else None

        # Surface strategy telemetry so callers see the true latency and can
        # adapt budgets without reaching into the strategy instance.
        out["latency_ms"] = round(float(result.latency_ms), 3)
        out["confidence"] = float(result.confidence)
        out["chunk_size"] = int(result.chunk_size)
        if result.metadata:
            out["strategy_metadata"] = dict(result.metadata)

        if execution is not None:
            out["execution"] = execution
            if not execution.get("ok", True):
                out["ok"] = False
        if verification is not None:
            out["verification"] = verification

        return out

    def _register_teardown(self) -> None:
        """Register scope cleanup for policy cache and episode state."""
        if self._teardown_registered:
            return
        if self._scope is None:
            return
        register = getattr(self._scope, "async_effect", None) or getattr(
            self._scope, "effect", None
        )
        if register is None:
            logger.warning(
                "Physical skill plugin received an effect scope without "
                "async_effect; policy cache will not be cleaned on teardown"
            )
            return
        try:
            register(self._cleanup)
            self._teardown_registered = True
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not register physical skill teardown effect: %s",
                exc, exc_info=True,
            )

    async def _cleanup(self) -> None:
        """Release cached policies and reset episode state."""
        await self._close_remote_policies()
        self._policies.clear()
        self._active_episode = None
        logger.debug("Physical skill plugin: cleaned up policy cache and episode state")

    async def _close_remote_policies(self) -> None:
        """Close every remote strategy this plugin has referenced.

        Only strategies exposing an ``async close()`` (currently
        :class:`VLARemoteStrategy`) are touched; local strategies own no
        socket and stay resident in the shared registry so a subsequent
        session can reuse the loaded model.  Failures are swallowed --
        teardown must not propagate secondary errors.
        """
        for path, strategy in list(self._policies.items()):
            close = getattr(strategy, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:  # noqa: BLE001 - teardown must not propagate
                logger.debug(
                    "Failed to close inference strategy for %s", path, exc_info=True,
                )




plugin = PhysicalSkillPlugin()


__all__ = ["PhysicalSkillPlugin", "plugin"]
