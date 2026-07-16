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


def select_template(domain: str, intent_template: str, names: list[str]) -> str:
    """Pick a template by convention: explicit -> exact domain -> domain prefix -> generic."""
    if intent_template and intent_template in names:
        return intent_template
    if domain:
        if domain in names:
            return domain
        prefixed = sorted(n for n in names if n.split(".", 1)[0] == domain)
        if prefixed:
            return prefixed[0]
    return "generic"


class DashboardViewBuilder:
    """Assemble ViewSpecs for dashboard intents."""

    def __init__(self, templates: TemplateLibrary | None = None) -> None:
        self._templates = templates or TemplateLibrary()

    async def build(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
        """Return a normalized ViewSpec for the given intent."""
        if intent.action == "session" or intent.target == "session":
            return await self._build_session(intent, provider)
        if intent.target and intent.action in ("watch", "open", "refresh"):
            return await self._build_watch(intent, provider)
        return await self._build_overview(intent, provider)

    async def _build_overview(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
        watches = await provider.watches()
        findings = await provider.findings(watch_id="", limit=50)
        data = {"title": "LeapBoard", "watches": watches, "findings": findings}
        name = "overview" if "overview" in self._templates.names() else "generic"
        return self._templates.render(name, data)

    async def _build_watch(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
        watches = await provider.watches()
        watch = next(
            (w for w in watches if str(w.get("watch_id", "")).startswith(intent.target)),
            {},
        )
        domain = str(watch.get("domain", ""))
        findings = await provider.findings(watch_id=str(watch.get("watch_id") or intent.target), limit=100)
        data = {
            "title": watch.get("name") or "Watch",
            "watch": watch,
            "domain": domain,
            "findings": findings,
        }
        name = select_template(domain, intent.template, self._templates.names())
        return self._templates.render(name, data)

    async def _build_session(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
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
        name = select_template("session", intent.template, self._templates.names())
        return self._templates.render(name, data)


__all__ = [
    "DashboardDataProvider",
    "DaemonDataProvider",
    "DashboardViewBuilder",
    "select_template",
]
