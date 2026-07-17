"""DashboardIntent: the single normalized request behind ``/board`` and the tool.

LeapBoard always analyzes the *current session*; the only view dimension is the
**template** (a rendering lens). Control verbs
(``templates``/``refresh``/``pause``/``resume``/``stop``/``status``) are handled
at the command layer, so the intent that reaches the view builder is simply a
template name — empty means the default (``generic``).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_TEMPLATE = "generic"


@dataclass(frozen=True)
class DashboardIntent:
    """A normalized dashboard request: which template to render the session with."""

    template: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"template": self.template}

    @classmethod
    def from_params(cls, data: Mapping[str, Any]) -> "DashboardIntent":
        """Build an intent from structured params (e.g. web ``?template=``)."""
        data = data if isinstance(data, Mapping) else {}
        return cls(template=str(data.get("template", "") or "").strip())

    @classmethod
    def from_args(cls, args: str) -> "DashboardIntent":
        """Parse a slash argument string; the first token is the template name."""
        try:
            tokens = shlex.split(args or "")
        except ValueError:
            tokens = (args or "").split()
        return cls(template=tokens[0].strip() if tokens else "")


__all__ = ["DashboardIntent", "DEFAULT_TEMPLATE"]
