# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fail-closed execution of declared hardware degradation policies.

Streaming and health-monitoring paths detect different faults, but a device must
reach the same governed state regardless of which path found it.  This module is
the single execution point for that state transition.  An action is successful
only when its required physical transition has completed; every incomplete or
unknown action falls back to an immediate transport halt.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.transport import SIDE_EFFECT_COMMITTED, SIDE_EFFECT_NONE

logger = logging.getLogger(__name__)


class DegradationAction(str, Enum):
    """Actions admitted by a :class:`DegradationPolicy`."""

    HALT = "halt"
    HOLD_POSITION = "hold_position"
    SAFE_RETURN = "safe_return"
    SWITCH_SENSOR = "switch_sensor"
    CONTINUE_BLIND = "continue_blind"


@dataclass(frozen=True)
class DegradationOutcome:
    """Auditable result of a degradation transition."""

    device_id: str
    action: str
    reason: str
    completed: bool
    latched: bool
    fallback_action: str = ""
    active_channel_id: str = ""
    verification: str = ""
    side_effect_state: str = SIDE_EFFECT_NONE
    error: str = ""
    requires_manual_clear: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome for audit, diagnostics, and benchmarks."""
        return {
            "device_id": self.device_id,
            "action": self.action,
            "reason": self.reason,
            "completed": self.completed,
            "latched": self.latched,
            "fallback_action": self.fallback_action,
            "active_channel_id": self.active_channel_id,
            "verification": self.verification,
            "side_effect_state": self.side_effect_state,
            "error": self.error,
            "requires_manual_clear": self.requires_manual_clear,
        }


SensorSwitcher = Callable[[str], Awaitable[bool] | bool]


class DegradationCoordinator:
    """Execute degradation actions through one fail-closed transition point."""

    def __init__(self, registry: Any, context: HardwareContext) -> None:
        self._registry = registry
        self._context = context

    async def execute(
        self,
        action: str,
        *,
        reason: str,
        failed_channel_id: str = "",
        sensor_switcher: SensorSwitcher | None = None,
    ) -> DegradationOutcome:
        """Execute *action* or halt when its completion cannot be proven."""
        try:
            parsed = DegradationAction(action)
        except ValueError:
            return await self._halt_fallback(
                action,
                reason,
                f"unknown degradation action {action!r}",
            )

        if parsed is DegradationAction.HALT:
            return await self._halt(reason, action=parsed.value)
        if parsed is DegradationAction.HOLD_POSITION:
            outcome = self._outcome(parsed.value, reason, completed=True)
            self._registry.latch_degradation(outcome)
            return outcome
        if parsed is DegradationAction.CONTINUE_BLIND:
            outcome = self._outcome(
                parsed.value,
                reason,
                completed=True,
                verification="observation continues; writes remain blocked",
                requires_manual_clear=True,
            )
            self._registry.latch_degradation(outcome)
            return outcome
        if parsed is DegradationAction.SWITCH_SENSOR:
            return await self._switch_sensor(
                reason,
                failed_channel_id,
                sensor_switcher,
            )
        return await self._safe_return(reason)

    def _outcome(
        self,
        action: str,
        reason: str,
        *,
        completed: bool,
        fallback_action: str = "",
        active_channel_id: str = "",
        verification: str = "",
        side_effect_state: str = SIDE_EFFECT_NONE,
        error: str = "",
        requires_manual_clear: bool = False,
    ) -> DegradationOutcome:
        return DegradationOutcome(
            device_id=self._context.device_id,
            action=action,
            reason=reason,
            completed=completed,
            latched=True,
            fallback_action=fallback_action,
            active_channel_id=active_channel_id,
            verification=verification,
            side_effect_state=side_effect_state,
            error=error,
            requires_manual_clear=requires_manual_clear,
        )

    async def _halt(self, reason: str, *, action: str, error: str = "") -> DegradationOutcome:
        try:
            transport = await self._registry.transport(self._context.device_id)
            status = await transport.halt()
            completed = bool(getattr(status, "connected", False)) and bool(
                getattr(status, "halt_supported", True)
            )
            if not completed:
                error = error or "transport did not confirm emergency halt"
        except Exception as exc:  # noqa: BLE001 - failure must remain auditable
            completed = False
            error = error or f"halt failed: {exc}"

        outcome = self._outcome(
            action,
            reason,
            completed=completed,
            verification="transport halt confirmed" if completed else "transport halt unconfirmed",
            side_effect_state=SIDE_EFFECT_COMMITTED if completed else SIDE_EFFECT_NONE,
            error=error,
        )
        self._registry.latch_degradation(outcome)
        if completed:
            logger.warning("DegradationPolicy halted device %s (%s)", outcome.device_id, reason)
        else:
            logger.error("DegradationPolicy halt could not be confirmed for %s: %s", outcome.device_id, error)
        return outcome

    async def _halt_fallback(
        self, action: str, reason: str, error: str
    ) -> DegradationOutcome:
        halted = await self._halt(reason, action=action, error=error)
        outcome = DegradationOutcome(
            **{
                **halted.to_dict(),
                "fallback_action": DegradationAction.HALT.value,
            }
        )
        self._registry.latch_degradation(outcome)
        return outcome

    async def _switch_sensor(
        self,
        reason: str,
        failed_channel_id: str,
        sensor_switcher: SensorSwitcher | None,
    ) -> DegradationOutcome:
        policy = self._context.degradation
        target = ""
        if policy is not None:
            target = str(policy.sensor_fallbacks.get(failed_channel_id, ""))
        if not target or sensor_switcher is None:
            return await self._halt_fallback(
                DegradationAction.SWITCH_SENSOR.value,
                reason,
                "declared healthy sensor fallback is unavailable",
            )

        target_channel = self._context.channel(target)
        if target_channel is None or not target_channel.is_readable:
            return await self._halt_fallback(
                DegradationAction.SWITCH_SENSOR.value,
                reason,
                f"fallback channel {target!r} is not readable",
            )

        try:
            switched = sensor_switcher(target)
            if inspect.isawaitable(switched):
                switched = await switched
        except Exception as exc:  # noqa: BLE001 - transition must fail closed
            switched = False
            switch_error = f"sensor switch failed: {exc}"
        else:
            switch_error = ""

        if not switched:
            return await self._halt_fallback(
                DegradationAction.SWITCH_SENSOR.value,
                reason,
                switch_error or f"fallback channel {target!r} did not verify healthy",
            )

        outcome = self._outcome(
            DegradationAction.SWITCH_SENSOR.value,
            reason,
            completed=True,
            active_channel_id=target,
            verification="fallback channel readback is healthy",
        )
        self._registry.clear_degradation(self._context.device_id, recovery="verified_sensor_fallback")
        logger.warning(
            "DegradationPolicy switched device %s from %s to fallback sensor %s",
            self._context.device_id,
            failed_channel_id,
            target,
        )
        return outcome

    async def _safe_return(self, reason: str) -> DegradationOutcome:
        """Drive a declared home trajectory and retain the degradation latch.

        The motion is admitted only when transport health, declared writable
        joints, control-bus execution, and readback verification all succeed.
        Any missing precondition or failed verification immediately falls back
        to a lock-free halt.
        """
        kinematics = self._context.kinematics
        policy = self._context.degradation
        targets = dict(getattr(kinematics, "home_joint_targets", {}) or {})
        if not targets:
            return await self._halt_fallback(
                DegradationAction.SAFE_RETURN.value,
                reason,
                "safe_return requires declared home_joint_targets",
            )
        context_channels = {channel.channel_id: channel for channel in self._context.channels}
        if any(channel_id not in context_channels or not context_channels[channel_id].is_writable for channel_id in targets):
            return await self._halt_fallback(
                DegradationAction.SAFE_RETURN.value,
                reason,
                "safe_return targets must be declared writable channels",
            )
        try:
            transport = await self._registry.transport(self._context.device_id)
            status = await transport.probe()
            if not status.connected:
                raise RuntimeError("transport is not connected")

            from leapflow.hardware.realtime import PIDJointController, TrajectoryTracker

            controller = PIDJointController(
                tuple(targets), kp=1.0, ki=0.0, kd=0.0,
                max_output=100.0, output_mode="position",
            )
            tracker = TrajectoryTracker(
                controller,
                ((0.0, targets),),
                loop_mode="hold_final",
            )
            bus = self._registry.create_control_bus(self._context.device_id)
            await bus.async_start(self._context.device_id, tracker)
            await asyncio.sleep(min(max(policy.safe_return_timeout_s if policy else 5.0, 0.1), 5.0))
            await bus.async_stop()

            deviations: dict[str, float] = {}
            for channel_id, target in targets.items():
                reading = await transport.read(channel_id)
                deviations[channel_id] = abs(float(reading.value) - float(target))
            tolerance = 1e-3
            if any(deviation > tolerance for deviation in deviations.values()):
                raise RuntimeError(f"home position verification failed: {deviations}")
        except Exception as exc:  # noqa: BLE001 - any uncertainty must halt
            return await self._halt_fallback(
                DegradationAction.SAFE_RETURN.value,
                reason,
                f"safe_return failed: {exc}",
            )

        outcome = self._outcome(
            DegradationAction.SAFE_RETURN.value,
            reason,
            completed=True,
            verification="home trajectory reached and readback verified",
            side_effect_state=SIDE_EFFECT_COMMITTED,
        )
        self._registry.latch_degradation(outcome)
        logger.warning("DegradationPolicy returned device %s to declared home pose", outcome.device_id)
        return outcome


__all__ = ["DegradationAction", "DegradationCoordinator", "DegradationOutcome"]
