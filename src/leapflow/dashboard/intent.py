"""DashboardIntent: the single normalized request behind ``/board`` and the tool.

The **template** is the primary view dimension (a rendering lens). Most templates
analyze the current session and need nothing else; a per-device view needs to know
*which* device, so the intent also carries an optional target.

``device``/``channel`` are explicit fields rather than a generic params bag. The
board's request surface is small and worth keeping legible, and a typed field is what
lets ``select_template`` and the view builder decide what to do without inspecting a
dictionary of unknown shape. Control verbs
(``templates``/``refresh``/``pause``/``resume``/``stop``/``status``) are handled at
the command layer, so the intent that reaches the view builder is a lens plus, at
most, a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from leapflow.utils.shell_lex import split_args


@dataclass(frozen=True)
class DashboardIntent:
    """A normalized dashboard request: which lens, and optionally which target."""

    template: str = ""
    device: str = ""
    channel: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the wire form, omitting an absent target.

        Empty keys are dropped rather than sent as ``""`` so a caller cannot confuse
        "no device asked for" with "a device whose id is the empty string" -- the
        latter would resolve to a lookup failure and report an unknown device.
        """
        payload: dict[str, Any] = {"template": self.template}
        if self.device:
            payload["device"] = self.device
        if self.channel:
            payload["channel"] = self.channel
        return payload

    @classmethod
    def from_params(cls, data: Mapping[str, Any]) -> "DashboardIntent":
        """Build an intent from structured params (e.g. web ``?template=&device=``)."""
        data = data if isinstance(data, Mapping) else {}
        return cls(
            template=str(data.get("template", "") or "").strip(),
            device=str(data.get("device", "") or "").strip(),
            channel=str(data.get("channel", "") or "").strip(),
        )

    @classmethod
    def from_args(cls, args: str) -> "DashboardIntent":
        """Parse a slash argument string: ``<template> [device] [channel]``.

        Positional because that is how the surrounding slash commands read, and the
        order matches the drill-down a person is doing: a lens, then what to point it
        at.
        """
        try:
            tokens = split_args(args or "")
        except ValueError:
            tokens = tuple((args or "").split())
        parts = [token.strip() for token in tokens if token.strip()]
        return cls(
            template=parts[0] if parts else "",
            device=parts[1] if len(parts) > 1 else "",
            channel=parts[2] if len(parts) > 2 else "",
        )


__all__ = ["DashboardIntent"]
