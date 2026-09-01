"""Declarative alert policies: event kind → automated response.

A policy is a YAML rule that maps an observed ``EventKind`` to an action. The
mapping is loaded from ``hardware.alert_policies`` at startup and evaluated on
every event the stream emits. The two actions are:

``hw_estop``
    Halt the device immediately via ``transport.halt()``. Exempt from approval
    because an emergency stop is a fail-safe reflex, not a deliberate operation
    -- and asking a human to click "approve" while a bench is breaching its
    envelope is exactly the delay the estop exists to eliminate.

Everything else
    Builds an ``ActionDescriptor`` and flows through ``ApprovalOrchestrator``
    like any other mutation, so no response path exists that bypasses consent.

``require_consecutive`` prevents a single transient reading from triggering an
irreversible response: the same *kind* on the same *channel* must fire at least
that many times in a row before the policy fires.  The default (3) is chosen so
a single spike is never actionable, matching the three-sample rule the quality
degradation detector already uses in ``HardwareEventDetector``.

Evaluation is synchronous: a policy check is a handful of dict lookups and
counter increments, never I/O. The asynchronous parts (transport.halt or
orchestrator.evaluate) are dispatched as fire-and-forget tasks so the sampling
loop is never blocked by an approval prompt or a slow transport.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_CONSECUTIVE = 3
"""Minimum consecutive hits before a policy fires.

Matches the three-sample rule used for quality degradation in the event
detector: a single transient reading must never trigger an irreversible
response.
"""


@dataclass(frozen=True)
class AlertRule:
    """One declarative rule: event_kind → action.

    ``channel_filter`` is optional: when absent the rule matches any channel on
    any device. When set it matches ``<device_id>.<channel_id>`` or a bare
    ``<channel_id>`` (which matches that channel on every device).
    """

    event_kind: str
    action: str
    channel_filter: str = ""
    require_consecutive: int = DEFAULT_CONSECUTIVE

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlertRule":
        raw_consecutive = data.get("require_consecutive")
        if raw_consecutive is not None:
            consecutive = int(raw_consecutive)
        else:
            consecutive = DEFAULT_CONSECUTIVE
        if consecutive < 1:
            consecutive = 1
        return cls(
            event_kind=str(data.get("event_kind", "")),
            action=str(data.get("action", "")),
            channel_filter=str(data.get("channel_filter", "") or ""),
            require_consecutive=consecutive,
        )

    def matches(self, event_kind: str, device_id: str, channel_id: str) -> bool:
        """Return whether this rule's kind and optional filter match the event."""
        if event_kind != self.event_kind:
            return False
        if not self.channel_filter:
            return True
        source = f"{device_id}.{channel_id}"
        return self.channel_filter == source or self.channel_filter == channel_id


@dataclass
class _ConsecutiveTracker:
    """Per-channel, per-kind consecutive hit counter."""

    counts: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _key(event_kind: str, device_id: str, channel_id: str) -> str:
        return f"{event_kind}:{device_id}.{channel_id}"

    def hit(self, event_kind: str, device_id: str, channel_id: str) -> int:
        """Increment and return the new consecutive count for this event."""
        key = self._key(event_kind, device_id, channel_id)
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        return count

    def reset(self, device_id: str, channel_id: str) -> None:
        """Clear all counters for a channel that has recovered."""
        prefix = f":{device_id}.{channel_id}"
        to_remove = [k for k in self.counts if k.endswith(prefix)]
        for k in to_remove:
            del self.counts[k]


class HardwareAlertPolicy:
    """Evaluate events against loaded rules, dispatching actions when matched.

    Thread-safety: ``evaluate`` is called from the sampling loop's
    ``_dispatch``, which is synchronous and single-threaded per source. The
    asynchronous action dispatch (halt / orchestrator) is delegated to
    ``asyncio.create_task`` so the sampling loop is never stalled.
    """

    def __init__(
        self,
        rules: Sequence[AlertRule] = (),
        *,
        orchestrator: Any = None,
        registry: Any = None,
    ) -> None:
        self._rules = tuple(rules)
        self._orchestrator = orchestrator
        self._registry = registry
        self._tracker = _ConsecutiveTracker()
        self._fired: int = 0

    @property
    def rules(self) -> tuple[AlertRule, ...]:
        return self._rules

    @property
    def fired_count(self) -> int:
        """Total number of policy actions that have been dispatched."""
        return self._fired

    def evaluate(self, event: Any) -> None:
        """Check *event* against all rules, dispatching on a match.

        Called synchronously from the sampling loop. Async work is spawned as
        tasks so the loop is never blocked.
        """
        kind = str(getattr(event, "kind", ""))
        device_id = str(getattr(event, "device_id", ""))
        channel_id = str(getattr(event, "channel_id", ""))

        for rule in self._rules:
            if not rule.matches(kind, device_id, channel_id):
                continue
            count = self._tracker.hit(kind, device_id, channel_id)
            if count < rule.require_consecutive:
                continue
            self._fire(rule, event, device_id)

    def reset_channel(self, device_id: str, channel_id: str) -> None:
        """Clear consecutive counters when a channel recovers.

        Called on ``settled`` events so a breach → recovery → breach cycle
        correctly restarts the consecutive-hit counter rather than carrying
        stale counts across a recovery.
        """
        self._tracker.reset(device_id, channel_id)

    def _fire(self, rule: AlertRule, event: Any, device_id: str) -> None:
        """Dispatch the rule's action, asynchronously when possible."""
        self._fired += 1
        detail = str(getattr(event, "detail", ""))
        channel_id = str(getattr(event, "channel_id", ""))

        if rule.action == "hw_estop":
            logger.warning(
                "Alert policy: estop %s (rule=%s, detail=%s)",
                device_id,
                rule.event_kind,
                detail,
            )
            self._dispatch_estop(device_id)
        else:
            logger.info(
                "Alert policy: %s on %s.%s (rule=%s, detail=%s)",
                rule.action,
                device_id,
                channel_id,
                rule.event_kind,
                detail,
            )
            self._dispatch_approval(rule, event, device_id, channel_id)

    def _dispatch_estop(self, device_id: str) -> None:
        """Halt the device. Exempt from approval -- this is a safety reflex."""
        registry = self._registry
        if registry is None:
            logger.error("Alert policy: cannot estop %s -- no registry bound", device_id)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("Alert policy: no running event loop for estop of %s", device_id)
            return
        loop.create_task(self._estop_async(device_id), name=f"hw_estop:{device_id}")

    async def _estop_async(self, device_id: str) -> None:
        """Execute the halt via the transport, logging the outcome."""
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.halt()
            if getattr(status, "halt_supported", False):
                logger.warning("Alert policy: estop of %s succeeded", device_id)
            else:
                logger.error(
                    "Alert policy: estop of %s -- transport reports halt not supported",
                    device_id,
                )
        except Exception as exc:  # noqa: BLE001 - estop failure must be logged, not propagated
            logger.error(
                "Alert policy: estop of %s failed: %s",
                device_id,
                exc,
                exc_info=True,
            )

    def _dispatch_approval(
        self,
        rule: AlertRule,
        event: Any,
        device_id: str,
        channel_id: str,
    ) -> None:
        """Build an ActionDescriptor and route through ApprovalOrchestrator."""
        orchestrator = self._orchestrator
        if orchestrator is None:
            logger.warning(
                "Alert policy: cannot dispatch %s -- no orchestrator bound",
                rule.action,
            )
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Alert policy: no running event loop for %s on %s",
                rule.action,
                device_id,
            )
            return
        loop.create_task(
            self._approve_async(rule, event, device_id, channel_id),
            name=f"hw_alert:{rule.action}:{device_id}",
        )

    async def _approve_async(
        self,
        rule: AlertRule,
        event: Any,
        device_id: str,
        channel_id: str,
    ) -> None:
        """Send the action through the approval chain."""
        try:
            from leapflow.security.actions import ActionDescriptor

            descriptor = ActionDescriptor.device(
                kind=f"device.alert.{rule.action}",
                device_id=device_id,
                channel_id=channel_id,
                quantity=str(getattr(event, "quantity", "")),
                value=getattr(event, "value", None),
                unit=str(getattr(event, "unit", "")),
                metadata={
                    "alert_action": rule.action,
                    "event_kind": rule.event_kind,
                    "detail": str(getattr(event, "detail", "")),
                },
            )
            result = await self._orchestrator.evaluate(descriptor)
            if not getattr(result, "approved", False):
                logger.info(
                    "Alert policy: %s on %s.%s denied by orchestrator",
                    rule.action,
                    device_id,
                    channel_id,
                )
        except Exception as exc:  # noqa: BLE001 - alert dispatch must not crash the sampling loop
            logger.error(
                "Alert policy: approval dispatch failed for %s on %s: %s",
                rule.action,
                device_id,
                exc,
                exc_info=True,
            )


def load_alert_policies(settings: Any) -> tuple[AlertRule, ...]:
    """Load alert rules from ``hardware.alert_policies`` in the settings.

    Returns an empty tuple when the key is absent, empty, or malformed, so a
    profile without policies keeps the hardware subsystem unchanged.
    """
    raw = getattr(settings, "hardware_alert_policies", None)
    if not raw:
        return ()
    if not isinstance(raw, (list, tuple)):
        logger.warning("hardware.alert_policies must be a list; ignoring")
        return ()
    rules: list[AlertRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            logger.warning("hardware.alert_policies[%d] is not a mapping; skipping", i)
            continue
        rule = AlertRule.from_dict(entry)
        if not rule.event_kind or not rule.action:
            logger.warning(
                "hardware.alert_policies[%d] missing event_kind or action; skipping",
                i,
            )
            continue
        rules.append(rule)
    return tuple(rules)


def build_alert_policy(
    settings: Any,
    *,
    orchestrator: Any = None,
    registry: Any = None,
) -> HardwareAlertPolicy | None:
    """Return a policy from settings, or None when no rules are configured."""
    rules = load_alert_policies(settings)
    if not rules:
        return None
    return HardwareAlertPolicy(
        rules,
        orchestrator=orchestrator,
        registry=registry,
    )


__all__ = [
    "AlertRule",
    "DEFAULT_CONSECUTIVE",
    "HardwareAlertPolicy",
    "build_alert_policy",
    "load_alert_policies",
]
