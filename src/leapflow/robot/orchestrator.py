# Copyright (c) Alibaba, Inc. and its affiliates.
"""Multi-device orchestration: coordinated control across multiple robots.

Sequences, parallelizes, and synchronizes physical operations across
multiple devices.  A single manipulation task may involve two arms
handing off an object, a leader-follower teleoperation pair, or a
pipeline where one device's completion triggers the next.

Safety is the orchestrator's first concern: any failure in a parallel
group triggers ``halt_all`` on every participating device, because a
half-completed multi-device operation can leave the workspace in a
physically unsafe state (one arm holding, the other released).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestration primitives
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    """How a group of device operations is coordinated."""

    SEQUENTIAL = "sequential"
    """One device completes before the next starts."""

    PARALLEL = "parallel"
    """All devices act simultaneously."""

    BARRIER = "barrier"
    """All devices reach a state, then proceed together."""


@dataclass(frozen=True)
class DeviceOperation:
    """One operation targeting one device in an orchestration.

    ``commands`` is a tuple of ``(channel_id, value)`` pairs that will be
    written to the device.  ``verify`` enables post-write verification
    through the existing ``OperationVerifier`` path.  ``settle_after_s``
    adds a settling delay after the write, matching the HCP Envelope
    convention.
    """

    device_id: str
    commands: tuple[tuple[str, Any], ...]
    verify: bool = False
    settle_after_s: float = 0.0


@dataclass(frozen=True)
class OrchestrationStep:
    """One step of a multi-device orchestration.

    ``mode`` is an :class:`ExecutionMode` value controlling how
    ``operations`` execute relative to one another.  ``barrier_timeout_s``
    applies only when ``mode`` is ``BARRIER``.
    """

    mode: str
    operations: tuple[DeviceOperation, ...]
    barrier_timeout_s: float = 10.0
    label: str = ""


@dataclass(frozen=True)
class OrchestrationResult:
    """Outcome of an orchestration step or plan.

    ``ok`` is True only when every operation succeeded.
    ``halted`` indicates that ``halt_all`` was triggered as a safety
    response to a failure.
    """

    ok: bool
    step_results: tuple[dict, ...]
    halted: bool = False
    halt_reason: str = ""
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class MultiDeviceOrchestrator:
    """Coordinates operations across multiple hardware devices.

    Dependencies:

    - ``registry``: :class:`~leapflow.hardware.registry.HardwareRegistry`
      for device access, I/O locks, and transport retrieval.
    - ``hardware_tools``: :class:`~leapflow.hardware.tools.HardwareTools`
      for approval-gated batch writes (optional).  When provided, every
      device write flows through the approval chain.
    - ``capability_index``: :class:`~leapflow.hardware.capability_router.CapabilityIndex`
      for affordance-based device routing (optional).
    """

    def __init__(
        self,
        registry: Any,
        *,
        hardware_tools: Any = None,
        capability_index: Any = None,
    ) -> None:
        self._registry = registry
        self._hardware_tools = hardware_tools
        self._capability_index = capability_index

    # ── Public API ────────────────────────────────────────────────

    async def execute_step(self, step: OrchestrationStep) -> OrchestrationResult:
        """Execute one orchestration step according to its mode.

        - ``SEQUENTIAL``: execute operations one at a time, stopping on first failure.
        - ``PARALLEL``: execute all operations concurrently via ``asyncio.gather``;
          on any failure trigger ``halt_all`` on every participating device.
        - ``BARRIER``: command all devices, then poll until all reach the target
          state (within ``barrier_timeout_s``), or halt on timeout.

        Returns an :class:`OrchestrationResult` summarising the outcome.
        """
        t0 = time.monotonic()
        mode = step.mode.value if isinstance(step.mode, ExecutionMode) else str(step.mode)

        try:
            if mode == ExecutionMode.SEQUENTIAL.value:
                results = await self._execute_sequential(step.operations)
            elif mode == ExecutionMode.PARALLEL.value:
                results = await self._execute_parallel(step.operations)
            elif mode == ExecutionMode.BARRIER.value:
                results = await self._execute_barrier(
                    step.operations, step.barrier_timeout_s
                )
            else:
                return OrchestrationResult(
                    ok=False,
                    step_results=({"error": f"unknown mode: {mode}"},),
                    elapsed_s=time.monotonic() - t0,
                )
        except Exception as exc:
            logger.error(
                "Orchestration step %r raised unexpectedly: %s",
                step.label or mode,
                exc,
                exc_info=True,
            )
            device_ids = tuple(op.device_id for op in step.operations)
            halt_map = await self.halt_all(device_ids)
            return OrchestrationResult(
                ok=False,
                step_results=({"error": str(exc), "halt_map": halt_map},),
                halted=True,
                halt_reason=str(exc),
                elapsed_s=time.monotonic() - t0,
            )

        ok = all(r.get("ok", False) for r in results)
        halted = any(r.get("halted", False) for r in results)
        halt_reason = next(
            (r.get("halt_reason", "") for r in results if r.get("halted")),
            "",
        )
        return OrchestrationResult(
            ok=ok,
            step_results=tuple(results),
            halted=halted,
            halt_reason=halt_reason,
            elapsed_s=time.monotonic() - t0,
        )

    async def execute_plan(
        self, steps: tuple[OrchestrationStep, ...]
    ) -> OrchestrationResult:
        """Execute a sequence of orchestration steps.

        Steps run sequentially (each step may itself be parallel internally).
        A failed step halts the plan and triggers ``halt_all`` on all devices
        that participated in the failed step.
        """
        t0 = time.monotonic()
        all_results: list[dict] = []

        for step in steps:
            result = await self.execute_step(step)
            all_results.extend(result.step_results)
            if not result.ok:
                return OrchestrationResult(
                    ok=False,
                    step_results=tuple(all_results),
                    halted=result.halted,
                    halt_reason=result.halt_reason or f"step '{step.label}' failed",
                    elapsed_s=time.monotonic() - t0,
                )

        return OrchestrationResult(
            ok=True,
            step_results=tuple(all_results),
            elapsed_s=time.monotonic() - t0,
        )

    async def halt_all(self, device_ids: tuple[str, ...]) -> dict:
        """Emergency stop all specified devices concurrently.

        Calls ``transport.halt()`` on every device in parallel.  ``halt()``
        is lock-free (HCP requirement) so this does not wait for in-flight
        operations.  Returns a per-device halt status map.

        This is the safety backstop: called automatically on any parallel
        group failure, and callable directly for emergency stop.
        """
        if not device_ids:
            return {}

        async def _halt_one(device_id: str) -> tuple[str, dict]:
            try:
                transport = await self._registry.transport(device_id)
                status = await transport.halt()
                return device_id, {
                    "halted": True,
                    "halt_supported": getattr(status, "halt_supported", True),
                }
            except Exception as exc:
                logger.warning(
                    "halt_all: device %r halt failed: %s",
                    device_id,
                    exc,
                    exc_info=True,
                )
                return device_id, {"halted": False, "error": str(exc)}

        unique_ids = tuple(dict.fromkeys(device_ids))
        results = await asyncio.gather(
            *(_halt_one(did) for did in unique_ids), return_exceptions=True
        )
        halt_map: dict[str, Any] = {}
        for item in results:
            if isinstance(item, Exception):
                logger.warning("halt_all: gather exception: %s", item, exc_info=True)
                continue
            did, status = item
            halt_map[did] = status
        return halt_map

    def resolve_by_affordance(self, affordance: str) -> tuple[str, ...]:
        """Return device_ids that can perform the given affordance.

        Uses ``capability_index`` to find candidate devices for multi-device
        task planning (e.g. "which arms can grasp?").
        """
        if self._capability_index is None:
            return ()
        entries = self._capability_index.resolve(affordance)
        return tuple(entry.device_id for entry in entries)

    # ── Internal execution modes ──────────────────────────────────

    async def _execute_sequential(
        self, operations: tuple[DeviceOperation, ...]
    ) -> list[dict]:
        """Execute operations one at a time, stopping on first failure."""
        results: list[dict] = []
        for op in operations:
            result = await self._execute_operation(op)
            results.append(result)
            if not result.get("ok", False):
                break
        return results

    async def _execute_parallel(
        self, operations: tuple[DeviceOperation, ...]
    ) -> list[dict]:
        """Execute operations concurrently via ``asyncio.gather``.

        Uses ``return_exceptions=True`` so one failure does not cancel the
        others mid-flight (which could leave devices in inconsistent
        states).  After gather, if any failed, calls ``halt_all``.
        """
        if not operations:
            return []

        raw_results = await asyncio.gather(
            *(self._execute_operation(op) for op in operations),
            return_exceptions=True,
        )

        results: list[dict] = []
        for i, raw in enumerate(raw_results):
            if isinstance(raw, Exception):
                results.append({
                    "ok": False,
                    "device_id": operations[i].device_id,
                    "error": str(raw),
                })
            else:
                results.append(raw)

        any_failed = any(not r.get("ok", False) for r in results)
        if any_failed:
            device_ids = tuple(op.device_id for op in operations)
            halt_map = await self.halt_all(device_ids)
            failure_reasons = [
                r.get("error", "unknown")
                for r in results
                if not r.get("ok", False)
            ]
            for r in results:
                r["halted"] = True
                r["halt_reason"] = "; ".join(failure_reasons)
                r["halt_map"] = halt_map

        return results

    async def _execute_barrier(
        self, operations: tuple[DeviceOperation, ...], timeout_s: float
    ) -> list[dict]:
        """Command all devices, then poll until all reach target or timeout.

        Phase 1: send commands to all devices concurrently.
        Phase 2: poll each device until the commanded value is reached or
        the barrier timeout expires.
        """
        if not operations:
            return []

        # Phase 1: command all devices concurrently.
        raw_results = await asyncio.gather(
            *(self._execute_operation(op) for op in operations),
            return_exceptions=True,
        )

        results: list[dict] = []
        for i, raw in enumerate(raw_results):
            if isinstance(raw, Exception):
                results.append({
                    "ok": False,
                    "device_id": operations[i].device_id,
                    "error": str(raw),
                })
            else:
                results.append(raw)

        # If any command failed, halt and return immediately.
        any_failed = any(not r.get("ok", False) for r in results)
        if any_failed:
            device_ids = tuple(op.device_id for op in operations)
            halt_map = await self.halt_all(device_ids)
            for r in results:
                r["halted"] = True
                r["halt_reason"] = "barrier command phase failed"
                r["halt_map"] = halt_map
            return results

        # Phase 2: wait for all devices to reach their target.
        deadline = time.monotonic() + timeout_s
        settled = set[int]()
        poll_interval = min(0.1, timeout_s / 10) if timeout_s > 0 else 0

        while len(settled) < len(operations):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            for idx, op in enumerate(operations):
                if idx in settled:
                    continue
                if await self._check_reached(op):
                    settled.add(idx)

            if len(settled) < len(operations) and poll_interval > 0:
                await asyncio.sleep(poll_interval)

        if len(settled) < len(operations):
            device_ids = tuple(op.device_id for op in operations)
            halt_map = await self.halt_all(device_ids)
            unsettled = [
                operations[i].device_id
                for i in range(len(operations))
                if i not in settled
            ]
            for r in results:
                r["halted"] = True
                r["halt_reason"] = (
                    f"barrier timeout ({timeout_s}s); "
                    f"unsettled devices: {unsettled}"
                )
                r["halt_map"] = halt_map
                r["ok"] = False
            return results

        for r in results:
            r["barrier_settled"] = True
        return results

    # ── Single operation execution ────────────────────────────────

    async def _execute_operation(self, op: DeviceOperation) -> dict:
        """Execute one device operation through the approval-gated path.

        Delegates to ``hardware_tools.batch_actuate`` if available (for the
        approval chain), otherwise uses ``transport.write`` directly.
        """
        if self._hardware_tools is not None:
            return await self._execute_via_tools(op)
        return await self._execute_direct(op)

    async def _execute_via_tools(self, op: DeviceOperation) -> dict:
        """Write through HardwareTools.batch_actuate for approval gating."""
        commands = [
            {"channel_id": ch_id, "value": value}
            for ch_id, value in op.commands
        ]
        try:
            result = await self._hardware_tools.batch_actuate({
                "device_id": op.device_id,
                "commands": commands,
                "verify": op.verify,
            })
            if not isinstance(result, dict):
                result = {"ok": False, "error": "unexpected result type"}
            result["device_id"] = op.device_id
            return result
        except Exception as exc:
            logger.warning(
                "Orchestrator: batch_actuate for %r failed: %s",
                op.device_id,
                exc,
                exc_info=True,
            )
            return {"ok": False, "device_id": op.device_id, "error": str(exc)}

    async def _execute_direct(self, op: DeviceOperation) -> dict:
        """Write directly through the transport (no approval chain)."""
        try:
            transport = await self._registry.transport(op.device_id)
        except Exception as exc:
            return {"ok": False, "device_id": op.device_id, "error": str(exc)}

        outcomes: list[dict] = []
        all_ok = True
        for channel_id, value in op.commands:
            try:
                outcome = await transport.write(channel_id, value)
                ok = getattr(outcome, "ok", False)
                outcomes.append({
                    "channel_id": channel_id,
                    "ok": ok,
                    "side_effect_state": getattr(
                        outcome, "side_effect_state", "unknown"
                    ),
                })
                if not ok:
                    all_ok = False
            except Exception as exc:
                outcomes.append({
                    "channel_id": channel_id,
                    "ok": False,
                    "error": str(exc),
                })
                all_ok = False

        if op.settle_after_s > 0:
            await asyncio.sleep(op.settle_after_s)

        return {
            "ok": all_ok,
            "device_id": op.device_id,
            "outcomes": outcomes,
        }

    async def _check_reached(self, op: DeviceOperation) -> bool:
        """Check whether all commanded channels reached their targets.

        Used by the barrier mode to poll for convergence.  Returns True
        when every commanded channel's current value is within a default
        tolerance of the target.  Returns True on read failure (fail-open)
        because a missing reading should not block the barrier indefinitely;
        the barrier's timeout is the safety backstop.
        """
        tolerance = 0.05
        for channel_id, target_value in op.commands:
            try:
                reading = await self._registry.read(op.device_id, channel_id)
                actual = getattr(reading, "value", None)
                if actual is None:
                    continue
                target_f = float(target_value)
                actual_f = float(actual)
                if abs(actual_f - target_f) > tolerance:
                    return False
            except Exception:
                # Read failure: do not block, let the timeout handle it.
                continue
        return True


__all__ = [
    "DeviceOperation",
    "ExecutionMode",
    "MultiDeviceOrchestrator",
    "OrchestrationResult",
    "OrchestrationStep",
]
