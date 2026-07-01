"""Trigger factory: create Trigger instances from type string and config dict."""

from __future__ import annotations

from typing import Dict, Type

from leapflow.scheduler.types import Trigger
from leapflow.scheduler.triggers.condition import ConditionTrigger
from leapflow.scheduler.triggers.cron import CronTrigger
from leapflow.scheduler.triggers.event import EventTrigger
from leapflow.scheduler.triggers.interval import IntervalTrigger

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TRIGGER_REGISTRY: Dict[str, Type] = {
    "interval": IntervalTrigger,
    "cron": CronTrigger,
    "event": EventTrigger,
    "condition": ConditionTrigger,
}


def register_trigger(trigger_type: str, cls: Type) -> None:
    """Register a custom trigger class for a given type string.

    Args:
        trigger_type: The type identifier (used in serialized configs).
        cls: The trigger class that implements the Trigger protocol.
    """
    _TRIGGER_REGISTRY[trigger_type] = cls


def create_trigger(trigger_type: str, config: dict) -> Trigger:
    """Factory: create a Trigger instance from type string and config dict.

    Args:
        trigger_type: One of "interval", "cron", "event", "condition",
                      or any registered custom type.
        config: Configuration dict to pass to the trigger's deserialize method.

    Returns:
        A Trigger instance ready for use.

    Raises:
        ValueError: If trigger_type is not registered.
    """
    cls = _TRIGGER_REGISTRY.get(trigger_type)
    if cls is None:
        available = ", ".join(sorted(_TRIGGER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown trigger type: {trigger_type!r}. "
            f"Available: {available}"
        )
    return cls.deserialize(config)  # type: ignore[return-value]


__all__ = [
    "create_trigger",
    "register_trigger",
    "IntervalTrigger",
    "CronTrigger",
    "EventTrigger",
    "ConditionTrigger",
]
