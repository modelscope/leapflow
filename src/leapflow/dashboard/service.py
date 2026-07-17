"""DashboardViewBuilder: turn a DashboardIntent + live data into a ViewSpec.

The builder is transport-agnostic: it reads data through a small
``DashboardDataProvider`` protocol (satisfied by a DaemonClient adapter in the
server, or a fake in tests) and renders via the template library. Template
selection is convention-based (intent template, else a template named for the
domain), never a hardcoded domain->file map.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from leapflow.dashboard.intent import DashboardIntent
from leapflow.dashboard.templates import TemplateLibrary

logger = logging.getLogger(__name__)


@runtime_checkable
class DashboardDataProvider(Protocol):
    """Read-only data access the builder needs (watches and findings)."""

    async def watches(self) -> list[dict[str, Any]]:
        """Return all watch views."""
        ...

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """Return findings, optionally scoped to a watch."""
        ...


class DaemonDataProvider:
    """Adapt a DaemonClient's ``watch_*`` RPCs to the provider protocol."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def watches(self) -> list[dict[str, Any]]:
        return list(await self._client.watch_list())

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return list(await self._client.watch_findings(watch_id=watch_id, limit=limit))


def select_template(template: str, names: list[str]) -> str:
    """Return the requested template if available, else the generic fallback.

    The template is the single view dimension; an unknown name degrades to the
    built-in ``generic`` default rather than failing.
    """
    return template if template and template in names else "generic"


class DashboardViewBuilder:
    """Assemble ViewSpecs for dashboard intents."""

    def __init__(self, templates: TemplateLibrary | None = None) -> None:
        self._templates = templates or TemplateLibrary()

    async def build(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
        """Return a normalized ViewSpec: the current session rendered via a template.

        LeapBoard has one analysis target (the current session); the intent only
        carries which template lens to render it with.
        """
        return await self._build_session(intent.template, provider)

    async def _build_session(self, template: str, provider: DashboardDataProvider) -> dict[str, Any]:
        # The session watch emits an insight finding whose payload carries the
        # structured analysis plus observation transparency metadata.
        watches = await provider.watches()
        session_watch = next((w for w in watches if str(w.get("domain")) == "session"), {})
        findings = await provider.findings(watch_id="", limit=50)
        session_findings = [f for f in findings if str(f.get("domain")) == "session"]
        analysis = session_findings[0].get("payload") if session_findings else {}
        observation = dict((analysis or {}).get("observation_status") or {})
        if session_watch:
            observation.update({
                "watch_state": session_watch.get("state", ""),
                "watch_muted": session_watch.get("muted", False),
                "last_run_at": session_watch.get("last_run_at", 0),
                "next_due_at": session_watch.get("next_due_at", 0),
                "run_count": session_watch.get("run_count", 0),
            })
        data = {
            "title": "Session Analysis",
            "analysis": analysis or {},
            "observation": observation,
            "artifact_context": (analysis or {}).get("artifact_context") or [],
            "findings": session_findings,
            "watch": session_watch,
        }
        name = select_template(template, self._templates.names())
        return self._templates.render(name, data)


__all__ = [
    "DashboardDataProvider",
    "DaemonDataProvider",
    "DashboardViewBuilder",
    "select_template",
]
