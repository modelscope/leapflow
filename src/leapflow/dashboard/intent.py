"""DashboardIntent: the single normalized request behind both entry doors.

Slash commands (``/dashboard ...``) and the engine ``dashboard`` tool both
produce a ``DashboardIntent``, so there is one implementation with two front
doors -- no duplicated routing and no keyword-matching taxonomy.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Mapping

KNOWN_ACTIONS = frozenset({
    "open", "home", "session", "watch", "new", "list", "refresh",
    "template", "close", "status", "pause", "resume", "stop", "mute", "findings",
})
_DEFAULT_ACTION = "open"
_TARGET_ACTIONS = frozenset({
    "watch", "refresh", "open", "pause", "resume", "stop", "mute", "findings",
})
_FLAG_KEYS = {
    "--template": "template",
    "--domain": "domain",
    "--name": "name",
    "--trigger": "trigger",
    "--sensitivity": "sensitivity",
}


@dataclass(frozen=True)
class DashboardIntent:
    """A normalized dashboard request from slash or natural-language entry."""

    action: str = _DEFAULT_ACTION
    domain: str = ""
    template: str = ""
    target: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "domain": self.domain,
            "template": self.template,
            "target": self.target,
            "params": dict(self.params),
        }

    @classmethod
    def from_params(cls, data: Mapping[str, Any]) -> "DashboardIntent":
        """Build an intent from structured tool/RPC params."""
        data = data if isinstance(data, Mapping) else {}
        action = str(data.get("action", "") or "").strip().lower()
        if action not in KNOWN_ACTIONS:
            action = _DEFAULT_ACTION
        return cls(
            action=action,
            domain=str(data.get("domain", "")).strip(),
            template=str(data.get("template", "")).strip(),
            target=str(data.get("target", "")).strip(),
            params=dict(data.get("params") or {}),
        )

    @classmethod
    def from_args(cls, args: str) -> "DashboardIntent":
        """Parse a slash argument string into an intent.

        The first token is the action when recognized; otherwise it is treated
        as a domain to create (``/dashboard finance`` -> new finance watch).
        """
        try:
            tokens = shlex.split(args or "")
        except ValueError:
            tokens = (args or "").split()
        if not tokens:
            return cls(action="open")

        first = tokens[0].lower()
        if first in KNOWN_ACTIONS:
            action, rest = first, tokens[1:]
        else:
            action, rest = "new", tokens

        domain = template = target = ""
        params: dict[str, Any] = {}
        positional: list[str] = []
        i = 0
        while i < len(rest):
            key = _FLAG_KEYS.get(rest[i])
            if key and i + 1 < len(rest):
                value = rest[i + 1]
                if key == "template":
                    template = value
                elif key == "domain":
                    domain = value
                else:
                    params[key] = value
                i += 2
            else:
                positional.append(rest[i])
                i += 1

        if action == "new" and positional and not domain:
            domain = positional[0]
        elif action in _TARGET_ACTIONS and positional:
            target = positional[0]
        elif action == "template" and positional:
            template = positional[0]
        elif action == "session":
            target = "session"
        return cls(action=action, domain=domain, template=template, target=target, params=params)


__all__ = ["DashboardIntent", "KNOWN_ACTIONS"]
