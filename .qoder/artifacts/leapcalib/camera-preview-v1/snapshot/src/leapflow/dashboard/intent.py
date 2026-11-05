# Copyright (c) Alibaba, Inc. and its affiliates.
"""DashboardIntent: the single normalized request behind ``/board`` and the tool.

The **template** is the primary view dimension (a rendering lens). Most templates
analyze the current session and need nothing else; device and evolution views carry
an explicit target so they never infer another client's state.

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
    """A normalized dashboard request with explicit device or session scope."""

    template: str = ""
    device: str = ""
    channel: str = ""
    session_id: str = ""

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
        if self.session_id:
            payload["session_id"] = self.session_id
        return payload

    @classmethod
    def from_params(cls, data: Mapping[str, Any]) -> "DashboardIntent":
        """Build an intent from structured params (e.g. web ``?template=&device=``)."""
        data = data if isinstance(data, Mapping) else {}
        return cls(
            template=str(data.get("template", "") or "").strip(),
            device=str(data.get("device", "") or "").strip(),
            channel=str(data.get("channel", "") or "").strip(),
            session_id=str(data.get("session_id", data.get("session", "")) or "").strip(),
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
        template = parts[0] if parts else ""
        is_evolution = template in {"evolution", "evolution_live", "causal_trace"}
        return cls(
            template=template,
            device=parts[1] if len(parts) > 1 and not is_evolution else "",
            channel=parts[2] if len(parts) > 2 and not is_evolution else "",
            session_id=parts[1] if len(parts) > 1 and is_evolution else "",
        )


__all__ = ["DashboardIntent"]
