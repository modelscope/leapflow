# Copyright (c) Alibaba, Inc. and its affiliates.
"""Control hierarchy: System-1/1.5/2 layered control orchestration.

Three control layers with distinct frequencies and responsibilities:

- System-2 (LLM, 0.2-1 Hz): High-level task planning and reasoning.
  Produces a sequence of subtask goals from natural language instructions.
  Runs in the AgentEngine's async loop.

- System-1.5 (VLA/Policy, 5-30 Hz): Learned or scripted policy execution.
  Maps observations + goals to coarse action sequences (action chunks).
  Runs via InferenceStrategy in async or dedicated thread.

- System-1 (Servo, 100-1000 Hz): Closed-loop PID/impedance control.
  Refines System-1.5 outputs into smooth actuator commands.
  Runs in HighFrequencyControlBus dedicated thread.

This module manages the interfaces between layers: goal passing,
state reporting, frequency budgeting, and escalation on failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class ControlLayer(str, Enum):
    """Identifies one of the three control layers."""

    SYSTEM_2 = "system_2"      # LLM planning
    SYSTEM_1_5 = "system_1_5"  # Policy inference
    SYSTEM_1 = "system_1"      # Servo control


@dataclass(frozen=True)
class LayerState:
    """Runtime state of one control layer."""

    layer: str  # ControlLayer value
    status: str  # "idle", "running", "error", "escalated"
    frequency_hz: float  # actual measured frequency
    target_frequency_hz: float
    cycle_count: int = 0
    last_error: str = ""
    compute_budget_used_ms: float = 0.0


@dataclass(frozen=True)
class SubtaskGoal:
    """A goal passed from System-2 to System-1.5.

    The LLM decomposes a natural language task into a sequence of these
    typed goals, each consumable by a specific InferenceStrategy.
    """

    goal_id: str
    description: str  # natural language for context
    goal_type: str  # "reach_position", "grasp", "place", "follow_trajectory", etc.
    target: Mapping[str, Any]  # goal-type-specific parameters
    strategy_hint: str = ""  # preferred InferenceStrategy id
    timeout_s: float = 30.0
    verify: bool = True


@dataclass(frozen=True)
class EscalationEvent:
    """Signals that a lower layer needs higher-layer intervention.

    System-1 detects servo error -> escalate to System-1.5 for replanning.
    System-1.5 detects policy failure -> escalate to System-2 for task replanning.
    """

    source_layer: str
    target_layer: str
    reason: str
    detail: str
    device_id: str = ""
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Goal execution verdicts
# ---------------------------------------------------------------------------


_VERDICT_OK = "ok"
_VERDICT_TIMEOUT = "timeout"
_VERDICT_ERROR = "error"
_VERDICT_ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# ControlHierarchy
# ---------------------------------------------------------------------------


class ControlHierarchy:
    """Orchestrates the three-layer control hierarchy for one device.

    Manages the lifecycle of all three layers, routes goals downward
    and escalations upward.

    Dependency injection keeps this module free of reverse imports into
    ``leapflow.engine``: all heavy objects arrive as constructor arguments.
    """

    def __init__(
        self,
        device_id: str,
        *,
        registry: Any,
        inference_registry: Any = None,  # InferenceStrategyRegistry
        control_bus: Any = None,  # HighFrequencyControlBus
        realtime_loop: Any = None,  # RealtimeControlLoop
        event_bus: Any = None,
    ) -> None:
        self._device_id = device_id
        self._registry = registry
        self._inference_registry = inference_registry
        self._control_bus = control_bus
        self._realtime_loop = realtime_loop
        self._event_bus = event_bus

        # Per-layer mutable state — only touched from the async orchestrator.
        self._layer_states: dict[str, LayerState] = {
            ControlLayer.SYSTEM_2.value: LayerState(
                layer=ControlLayer.SYSTEM_2.value,
                status="idle",
                frequency_hz=0.0,
                target_frequency_hz=1.0,
            ),
            ControlLayer.SYSTEM_1_5.value: LayerState(
                layer=ControlLayer.SYSTEM_1_5.value,
                status="idle",
                frequency_hz=0.0,
                target_frequency_hz=10.0,
            ),
            ControlLayer.SYSTEM_1.value: LayerState(
                layer=ControlLayer.SYSTEM_1.value,
                status="idle",
                frequency_hz=0.0,
                target_frequency_hz=500.0,
            ),
        }

        self._goal_queue: list[SubtaskGoal] = []
        self._escalation_handlers: dict[str, Callable[..., Any]] = {}
        self._active_goal: SubtaskGoal | None = None
        self._shutdown_requested = False

    # -- System-2 interface (called by PhysicalSkillPlugin / AgentEngine) --

    async def execute_task(
        self, goals: tuple[SubtaskGoal, ...]
    ) -> dict[str, Any]:
        """Execute a sequence of subtask goals through the hierarchy.

        For each goal:
        1. Select InferenceStrategy from inference_registry (System-1.5)
        2. Start RealtimeControlLoop with PolicyChain (System-1.5 -> 1)
        3. Monitor execution until goal achieved or timeout
        4. On failure: escalate or move to next goal

        Returns a summary dict with per-goal verdicts.
        """
        self._update_layer(
            ControlLayer.SYSTEM_2.value, status="running",
        )

        results: list[dict[str, Any]] = []
        completed = 0
        failed = 0

        for goal in goals:
            if self._shutdown_requested:
                results.append({
                    "goal_id": goal.goal_id,
                    "verdict": "aborted",
                    "reason": "hierarchy shutdown requested",
                })
                failed += 1
                continue

            result = await self.execute_single_goal(goal)
            results.append(result)
            if result.get("verdict") == _VERDICT_OK:
                completed += 1
            else:
                failed += 1
                # Escalate to System-2 on failure so the caller can replan.
                if result.get("verdict") in (_VERDICT_ERROR, _VERDICT_TIMEOUT):
                    self._emit_escalation(EscalationEvent(
                        source_layer=ControlLayer.SYSTEM_1_5.value,
                        target_layer=ControlLayer.SYSTEM_2.value,
                        reason=result.get("verdict", "unknown"),
                        detail=result.get("error", ""),
                        device_id=self._device_id,
                        timestamp=time.monotonic(),
                    ))

        self._update_layer(ControlLayer.SYSTEM_2.value, status="idle")

        return {
            "device_id": self._device_id,
            "total": len(goals),
            "completed": completed,
            "failed": failed,
            "results": results,
        }

    async def execute_single_goal(
        self, goal: SubtaskGoal
    ) -> dict[str, Any]:
        """Execute one subtask goal.  Returns result with verdict."""
        self._active_goal = goal
        self._goal_queue = [g for g in self._goal_queue if g.goal_id != goal.goal_id]

        start_t = time.monotonic()
        self._update_layer(
            ControlLayer.SYSTEM_1_5.value, status="running",
        )

        try:
            result = await self._execute_with_system_1_5(goal)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_t
            self._update_layer(
                ControlLayer.SYSTEM_1_5.value,
                status="error",
                last_error=f"timeout after {elapsed:.1f}s",
            )
            result = {
                "goal_id": goal.goal_id,
                "verdict": _VERDICT_TIMEOUT,
                "error": f"goal timed out after {elapsed:.1f}s",
                "elapsed_s": round(elapsed, 3),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start_t
            self._update_layer(
                ControlLayer.SYSTEM_1_5.value,
                status="error",
                last_error=str(exc),
            )
            result = {
                "goal_id": goal.goal_id,
                "verdict": _VERDICT_ERROR,
                "error": str(exc),
                "elapsed_s": round(elapsed, 3),
            }
        finally:
            self._active_goal = None
            self._update_layer(
                ControlLayer.SYSTEM_1_5.value, status="idle",
            )

        self._emit_telemetry("goal.completed", result)
        return result

    # -- Escalation handling --

    def on_escalation(self, event: EscalationEvent) -> None:
        """Handle an escalation from a lower layer.

        System-1 -> System-1.5: replan action chunk
        System-1.5 -> System-2: request LLM replanning (emit event for AgentEngine)
        """
        logger.warning(
            "Escalation %s -> %s: %s (%s)",
            event.source_layer,
            event.target_layer,
            event.reason,
            event.detail,
        )

        handler = self._escalation_handlers.get(event.target_layer)
        if handler is not None:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Escalation handler for %s failed: %s",
                    event.target_layer,
                    exc,
                    exc_info=True,
                )

        # Publish to EventBus for cross-module visibility.
        self._emit_telemetry("escalation", {
            "source_layer": event.source_layer,
            "target_layer": event.target_layer,
            "reason": event.reason,
            "device_id": event.device_id,
        })

    def register_escalation_handler(
        self, target_layer: str, handler: Callable[..., Any]
    ) -> None:
        """Register a handler for escalations targeting a specific layer."""
        self._escalation_handlers[target_layer] = handler

    # -- State and monitoring --

    def layer_state(self, layer: str) -> LayerState:
        """Return the current state of a single layer."""
        state = self._layer_states.get(layer)
        if state is None:
            raise KeyError(f"unknown control layer: {layer!r}")
        return state

    def all_layer_states(self) -> dict[str, LayerState]:
        """Return a snapshot of all layer states."""
        return dict(self._layer_states)

    async def status(self) -> dict[str, Any]:
        """Complete hierarchy status for tools/dashboard."""
        layers: dict[str, dict[str, Any]] = {}
        for key, state in self._layer_states.items():
            layers[key] = {
                "status": state.status,
                "frequency_hz": state.frequency_hz,
                "target_frequency_hz": state.target_frequency_hz,
                "cycle_count": state.cycle_count,
                "last_error": state.last_error,
                "compute_budget_used_ms": state.compute_budget_used_ms,
            }

        active_goal: dict[str, Any] | None = None
        if self._active_goal is not None:
            active_goal = {
                "goal_id": self._active_goal.goal_id,
                "goal_type": self._active_goal.goal_type,
                "description": self._active_goal.description,
            }

        bus_running = False
        bus_stats: dict[str, Any] | None = None
        if self._control_bus is not None:
            bus_running = self._control_bus.is_running
            bus_stats = self._control_bus.stats.to_dict()

        return {
            "device_id": self._device_id,
            "layers": layers,
            "active_goal": active_goal,
            "queued_goals": len(self._goal_queue),
            "control_bus_running": bus_running,
            "control_bus_stats": bus_stats,
        }

    # -- Lifecycle --

    async def shutdown(self) -> None:
        """Gracefully stop all layers (bottom-up: System-1 first)."""
        self._shutdown_requested = True

        # 1. System-1: stop servo control bus.
        if self._control_bus is not None and self._control_bus.is_running:
            try:
                await self._control_bus.async_stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "System-1 shutdown error on %s: %s",
                    self._device_id, exc, exc_info=True,
                )
        self._update_layer(ControlLayer.SYSTEM_1.value, status="idle")

        # 2. System-1.5: stop realtime loop (if running independently).
        if self._realtime_loop is not None:
            try:
                await self._realtime_loop.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "System-1.5 shutdown error on %s: %s",
                    self._device_id, exc, exc_info=True,
                )
        self._update_layer(ControlLayer.SYSTEM_1_5.value, status="idle")

        # 3. System-2: clear goal queue and mark idle.
        self._goal_queue.clear()
        self._active_goal = None
        self._update_layer(ControlLayer.SYSTEM_2.value, status="idle")

        logger.info(
            "Control hierarchy shutdown complete for device %s",
            self._device_id,
        )

    # -- Internal: goal execution --

    async def _execute_with_system_1_5(
        self, goal: SubtaskGoal
    ) -> dict[str, Any]:
        """Select strategy, run inference, execute through control stack."""
        start_t = time.monotonic()

        # 1. Select inference strategy.
        strategy = self._select_strategy(goal)
        if strategy is None:
            return {
                "goal_id": goal.goal_id,
                "verdict": _VERDICT_ERROR,
                "error": (
                    f"no inference strategy found for goal "
                    f"{goal.goal_type!r} (hint={goal.strategy_hint!r})"
                ),
                "elapsed_s": 0.0,
            }

        # 2. Read observation.
        observation = await self._read_observation()

        # Enrich observation with goal information so the policy has context.
        observation["_goal_type"] = goal.goal_type
        observation["_goal_target"] = dict(goal.target)
        observation["_goal_description"] = goal.description

        # 3. Run System-1.5 inference.
        self._update_layer(
            ControlLayer.SYSTEM_1_5.value, status="running",
        )
        inference_start = time.monotonic()

        from leapflow.robot.inference.strategy import ComputeBudget

        budget = ComputeBudget(
            max_latency_ms=goal.timeout_s * 1000.0 * 0.5,  # half of goal timeout
        )
        result = await strategy.infer(observation, budget=budget)

        inference_ms = (time.monotonic() - inference_start) * 1000.0
        self._update_layer(
            ControlLayer.SYSTEM_1_5.value,
            compute_budget_used_ms=inference_ms,
        )

        # 4. Execute action through System-1.
        action = result.action
        if action is None:
            return {
                "goal_id": goal.goal_id,
                "verdict": _VERDICT_ERROR,
                "error": "inference returned null action",
                "elapsed_s": round(time.monotonic() - start_t, 3),
            }

        execution_result = await self._execute_through_system_1(
            goal, action,
        )

        elapsed = time.monotonic() - start_t
        verdict = _VERDICT_OK if execution_result.get("ok") else _VERDICT_ERROR

        return {
            "goal_id": goal.goal_id,
            "goal_type": goal.goal_type,
            "verdict": verdict,
            "elapsed_s": round(elapsed, 3),
            "inference_ms": round(inference_ms, 3),
            "confidence": float(result.confidence),
            "strategy_id": getattr(strategy, "strategy_id", ""),
            "execution": execution_result,
        }

    async def _execute_through_system_1(
        self, goal: SubtaskGoal, action: Any
    ) -> dict[str, Any]:
        """Execute an action through the System-1 servo layer.

        If a RealtimeControlLoop and ControlBus are available, builds a
        PolicyChain (System-1.5 action -> System-1 PID) for smooth execution.
        Falls back to direct registry writes when the RT stack is unavailable.
        """
        self._update_layer(ControlLayer.SYSTEM_1.value, status="running")

        try:
            # Attempt RT execution path.
            if self._realtime_loop is not None and self._control_bus is not None:
                return await self._execute_via_rt_loop(goal, action)

            # Fallback: write action directly via registry.
            return await self._execute_direct_write(action)
        except Exception as exc:  # noqa: BLE001
            self._update_layer(
                ControlLayer.SYSTEM_1.value,
                status="error",
                last_error=str(exc),
            )
            # Escalate System-1 error to System-1.5.
            self._emit_escalation(EscalationEvent(
                source_layer=ControlLayer.SYSTEM_1.value,
                target_layer=ControlLayer.SYSTEM_1_5.value,
                reason="servo_error",
                detail=str(exc),
                device_id=self._device_id,
                timestamp=time.monotonic(),
            ))
            return {"ok": False, "error": str(exc)}
        finally:
            if self._layer_states[ControlLayer.SYSTEM_1.value].status == "running":
                self._update_layer(
                    ControlLayer.SYSTEM_1.value, status="idle",
                )

    async def _execute_via_rt_loop(
        self, goal: SubtaskGoal, action: Any
    ) -> dict[str, Any]:
        """Execute action through the RealtimeControlLoop stack."""
        assert self._realtime_loop is not None

        # Convert action to joint commands mapping.
        commands = self._action_to_joint_targets(action)
        if not commands:
            return {"ok": False, "error": "could not interpret action as joint targets"}

        joint_ids = tuple(commands.keys())

        # Choose inner controller based on goal type.
        inner_controller = "impedance" if goal.goal_type in (
            "grasp", "insert", "polish", "compliant_contact",
        ) else "pid"

        session_id = await self._realtime_loop.start_pid(
            self._device_id,
            joint_ids,
            commands,
        ) if inner_controller == "pid" else await self._realtime_loop.start_impedance(
            self._device_id,
            joint_ids,
            commands,
        )

        # Wait for convergence or timeout.
        deadline = time.monotonic() + min(goal.timeout_s, 30.0)
        while time.monotonic() < deadline:
            status = await self._realtime_loop.status()
            if status.get("status") != "running":
                break
            if status.get("is_complete"):
                break
            await asyncio.sleep(0.05)

        result = await self._realtime_loop.stop()
        self._update_layer(
            ControlLayer.SYSTEM_1.value,
            status="idle",
            cycle_count=result.get("stats", {}).get("cycles", 0),
        )
        return {
            "ok": True,
            "session_id": session_id,
            "execution_mode": f"rt_{inner_controller}",
            "stats": result.get("stats", {}),
        }

    async def _execute_direct_write(
        self, action: Any
    ) -> dict[str, Any]:
        """Fallback: write action directly to hardware registry."""
        commands = self._action_to_joint_targets(action)
        if not commands:
            return {"ok": False, "error": "could not interpret action as joint targets"}

        from leapflow.hardware.transport import BatchTransport

        transport = await self._registry.transport(self._device_id)
        if isinstance(transport, BatchTransport):
            write_tuples = tuple(commands.items())
            outcome = await transport.write_batch(write_tuples)
            return {
                "ok": outcome.ok,
                "channels_written": len(write_tuples),
                "execution_mode": "direct_batch",
            }

        # Sequential fallback.
        written = 0
        for ch_id, val in commands.items():
            try:
                await transport.write(ch_id, val)
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Direct write failed for %s.%s: %s",
                    self._device_id, ch_id, exc,
                )
        return {
            "ok": written == len(commands),
            "channels_written": written,
            "channels_total": len(commands),
            "execution_mode": "direct_sequential",
        }

    # -- Internal: strategy selection --

    def _select_strategy(self, goal: SubtaskGoal) -> Any:
        """Select an InferenceStrategy for a goal.

        Precedence:
        1. Explicit strategy_hint in the goal
        2. Auto-select from the inference registry based on budget
        """
        if self._inference_registry is None:
            return None

        # 1. Try explicit hint.
        if goal.strategy_hint:
            strategy = self._inference_registry.get(goal.strategy_hint)
            if strategy is not None:
                return strategy
            logger.warning(
                "Strategy hint %r not found in registry; falling back to auto-select",
                goal.strategy_hint,
            )

        # 2. Auto-select: let the registry pick the best fit.
        from leapflow.robot.inference.strategy import ComputeBudget

        budget = ComputeBudget(
            max_latency_ms=min(goal.timeout_s * 1000.0 * 0.5, 500.0),
        )
        return self._inference_registry.select(budget=budget)

    # -- Internal: observation reading --

    async def _read_observation(self) -> dict[str, Any]:
        """Read current device observation through the registry."""
        context = self._registry.context(self._device_id)
        if context is None:
            return {}

        channels = [
            ch.channel_id
            for ch in context.channels
            if ch.is_readable and getattr(ch, "representation", "") != "frame"
        ]

        from leapflow.hardware.transport import BatchTransport

        transport = await self._registry.transport(self._device_id)
        if isinstance(transport, BatchTransport) and channels:
            batch = await transport.read_batch(tuple(channels))
            return {r.channel_id: r.value for r in batch.readings}

        obs: dict[str, Any] = {}
        for cid in channels:
            try:
                reading = await self._registry.read(self._device_id, cid)
                obs[cid] = reading.value
            except Exception:  # noqa: BLE001
                pass
        return obs

    # -- Internal: helpers --

    @staticmethod
    def _action_to_joint_targets(action: Any) -> dict[str, float]:
        """Convert an action value to a {channel_id: float} mapping."""
        if isinstance(action, dict):
            return {str(k): float(v) for k, v in action.items()}
        if hasattr(action, "tolist"):
            # numpy/torch tensor: index-based channel ids.
            vals = action.tolist()
            if isinstance(vals, list):
                return {str(i): float(v) for i, v in enumerate(vals)}
        if isinstance(action, (list, tuple)):
            return {str(i): float(v) for i, v in enumerate(action)}
        return {}

    def _update_layer(
        self,
        layer: str,
        *,
        status: str | None = None,
        frequency_hz: float | None = None,
        cycle_count: int | None = None,
        last_error: str | None = None,
        compute_budget_used_ms: float | None = None,
    ) -> None:
        """Update one layer's state immutably."""
        current = self._layer_states.get(layer)
        if current is None:
            return
        self._layer_states[layer] = LayerState(
            layer=current.layer,
            status=status if status is not None else current.status,
            frequency_hz=(
                frequency_hz if frequency_hz is not None else current.frequency_hz
            ),
            target_frequency_hz=current.target_frequency_hz,
            cycle_count=(
                cycle_count if cycle_count is not None else current.cycle_count
            ),
            last_error=(
                last_error if last_error is not None else current.last_error
            ),
            compute_budget_used_ms=(
                compute_budget_used_ms
                if compute_budget_used_ms is not None
                else current.compute_budget_used_ms
            ),
        )

    def _emit_escalation(self, event: EscalationEvent) -> None:
        """Route an escalation through registered handlers and EventBus."""
        self.on_escalation(event)

    def _emit_telemetry(self, topic: str, payload: Any) -> None:
        """Best-effort telemetry publish to EventBus."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(f"control.hierarchy.{topic}", payload)
        except Exception:  # noqa: BLE001 — telemetry must not break control
            pass


# ---------------------------------------------------------------------------
# TaskDecomposer — System-2 -> System-1.5 bridge
# ---------------------------------------------------------------------------


class TaskDecomposer:
    """Decomposes high-level task descriptions into SubtaskGoal sequences.

    This is a lightweight bridge -- the actual decomposition happens in
    the LLM (System-2 via AgentEngine).  This class provides the
    structured representation that the hierarchy can consume.

    Not LLM-powered itself: it is a type converter that structures
    the LLM's already-decomposed output into typed SubtaskGoals.
    """

    def parse_plan(
        self, plan: Mapping[str, Any]
    ) -> tuple[SubtaskGoal, ...]:
        """Parse a structured plan (from LLM tool output) into goals.

        Expected plan format::

            {
                "steps": [
                    {
                        "type": "reach_position",
                        "target": {"joint.shoulder": 1.0, ...},
                        "description": "move to above the cup",
                        "strategy_hint": "vla_local:pick_place",
                        "timeout_s": 10.0,
                        "verify": true
                    },
                    {"type": "grasp", "target": {"gripper": "close"}},
                    {"type": "reach_position", "target": {...}},
                    {"type": "place", "target": {"gripper": "open"}},
                ]
            }

        Missing optional fields receive sensible defaults.
        """
        steps = plan.get("steps")
        if not steps or not isinstance(steps, (list, tuple)):
            return ()

        goals: list[SubtaskGoal] = []
        for idx, step in enumerate(steps):
            if not isinstance(step, Mapping):
                logger.warning(
                    "TaskDecomposer: skipping non-mapping step at index %d", idx,
                )
                continue

            goal_type = str(step.get("type", "unknown"))
            target = step.get("target", {})
            if not isinstance(target, Mapping):
                target = {}

            goals.append(SubtaskGoal(
                goal_id=step.get("goal_id", f"goal_{idx}_{uuid.uuid4().hex[:6]}"),
                description=str(step.get("description", goal_type)),
                goal_type=goal_type,
                target=target,
                strategy_hint=str(step.get("strategy_hint", "")),
                timeout_s=float(step.get("timeout_s", 30.0)),
                verify=bool(step.get("verify", True)),
            ))

        return tuple(goals)

    def from_action_chunks(
        self,
        chunks: list[Mapping[str, float]],
        *,
        chunk_duration_s: float = 0.5,
    ) -> tuple[SubtaskGoal, ...]:
        """Convert raw action chunks from a VLA policy into trajectory goals.

        Each chunk becomes a ``follow_trajectory`` goal whose target carries
        the joint-value mapping and the chunk's duration.  This is useful
        when a VLA policy returns a sequence of action chunks that should be
        executed sequentially through the control hierarchy.
        """
        if not chunks:
            return ()

        goals: list[SubtaskGoal] = []
        for idx, chunk in enumerate(chunks):
            goals.append(SubtaskGoal(
                goal_id=f"chunk_{idx}_{uuid.uuid4().hex[:6]}",
                description=f"action chunk {idx}/{len(chunks)}",
                goal_type="follow_trajectory",
                target={
                    "joint_targets": dict(chunk),
                    "duration_s": chunk_duration_s,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                },
                timeout_s=chunk_duration_s * 3.0,  # generous timeout
                verify=idx == len(chunks) - 1,  # verify only last chunk
            ))

        return tuple(goals)


__all__ = [
    "ControlHierarchy",
    "ControlLayer",
    "EscalationEvent",
    "LayerState",
    "SubtaskGoal",
    "TaskDecomposer",
]
