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
    """Read-only data access the builder needs (watches, findings, signal metrics)."""

    async def watches(self) -> list[dict[str, Any]]:
        """Return all watch views."""
        ...

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """Return findings, optionally scoped to a watch."""
        ...

    async def signal_metrics(self) -> dict[str, Any]:
        """Return signal flow health metrics."""
        ...


class DaemonDataProvider:
    """Adapt a DaemonClient's ``watch_*`` RPCs to the provider protocol."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def watches(self) -> list[dict[str, Any]]:
        return list(await self._client.watch_list())

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return list(await self._client.watch_findings(watch_id=watch_id, limit=limit))

    async def signal_metrics(self) -> dict[str, Any]:
        """Return signal flow health metrics and live stream from the daemon."""
        result = await self._client.monitor_signal_metrics()
        if result.get("ok"):
            return {
                "metrics": result.get("metrics", {}),
                "signal_stream": result.get("signal_stream", []),
            }
        return {"metrics": {}, "signal_stream": []}


def select_template(template: str, names: list[str]) -> str:
    """Return the requested template if available, else the generic fallback.

    The template is the single view dimension; an unknown name degrades to the
    built-in ``generic`` default rather than failing.
    """
    return template if template and template in names else "generic"


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sum_int_values(value: Any) -> int:
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return 0
    return sum(_safe_int(item) for item in values)


def _distribution(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key) or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return [{"label": label, "value": count} for label, count in sorted(counts.items())]


def _event_family(event_type: str) -> str:
    normalized = str(event_type or "unknown").replace(":", ".")
    return normalized.split(".", 1)[0] or "unknown"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _signal_stream_item(evt: dict[str, Any]) -> dict[str, Any]:
    event_type = str(evt.get("event_type") or evt.get("type") or "")
    source = str(evt.get("source") or "")
    family = _event_family(event_type)
    timestamp = _safe_float(evt.get("ts") or evt.get("timestamp"))
    return {
        "title": event_type,
        "event_type": event_type,
        "summary": source,
        "source": source,
        "severity": "info",
        "family": family,
        "ts": timestamp,
    }


def _short_id(value: Any) -> str:
    text = str(value or "")
    return text[:8] if len(text) > 8 else text


_PAYLOAD_DOMAINS: dict[str, tuple[str, str]] = {
    # template -> (finding domain, data key the template binds to)
    "capability": ("capability_adaptation", "capability_plan"),
    "hardware": ("hardware", "hardware"),
}
"""Templates whose data is a producer's finding payload, not a session lens.

The distinction is real and was previously unrepresented. ``generic``/``finance``/
``sentiment``/``research`` are four renderings of *one* subject, the current
session, so they all read ``analysis``. These read the newest finding of their own
domain instead.

Held as a table because the alternative -- a name check per template in ``build``
-- had already gone wrong once: ``capability.yaml`` binds ``capability_plan.*``,
nothing supplied that key, and every value on the board rendered blank. The
producer ran, the template was valid, and nothing connected them.
"""


class DashboardViewBuilder:
    """Assemble ViewSpecs for dashboard intents."""

    def __init__(self, templates: TemplateLibrary | None = None) -> None:
        self._templates = templates or TemplateLibrary()

    async def build(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
        """Return a normalized ViewSpec for the requested template.

        Three data shapes, not one: the signal pipeline's own metrics, a producer's
        finding payload, or the session analysis every other lens renders.
        """
        template_name = intent.template
        if template_name == "signals":
            return await self._build_signals(template_name, provider)
        payload_domain = _PAYLOAD_DOMAINS.get(template_name)
        if payload_domain is not None:
            return await self._build_from_finding_payload(template_name, provider, *payload_domain)
        return await self._build_session(intent.template, provider)

    async def _build_from_finding_payload(
        self,
        template: str,
        provider: DashboardDataProvider,
        finding_domain: str,
        data_key: str,
    ) -> dict[str, Any]:
        """Render a template from the newest finding of one producer domain.

        Newest rather than merged: each of these payloads is a self-consistent
        snapshot of a subject at one instant, and stitching two together would show
        a state that never existed.
        """
        watches = await provider.watches()
        watch = next((w for w in watches if str(w.get("domain")) == finding_domain), {})
        findings = await provider.findings(watch_id="", limit=50)
        domain_findings = [f for f in findings if str(f.get("domain")) == finding_domain]
        payload = dict(domain_findings[0].get("payload") or {}) if domain_findings else {}
        data = {
            "title": template.replace("_", " ").title(),
            data_key: payload,
            "findings": domain_findings or None,
            "watch": watch,
            "observation": {
                "watch_state": watch.get("state", ""),
                "watch_muted": watch.get("muted", False),
                "last_run_at": watch.get("last_run_at", 0),
                "next_due_at": watch.get("next_due_at", 0),
                "run_count": watch.get("run_count", 0),
            },
        }
        return self._render(template, data)

    async def _build_session(self, template: str, provider: DashboardDataProvider) -> dict[str, Any]:
        # The session watch emits an insight finding whose payload carries the
        # structured analysis plus observation transparency metadata.
        watches = await provider.watches()
        session_watch = next((w for w in watches if str(w.get("domain")) == "session"), {})
        findings = await provider.findings(watch_id="", limit=50)
        session_findings = [f for f in findings if str(f.get("domain")) == "session"]
        analysis = dict(session_findings[0].get("payload") or {}) if session_findings else {}
        observation = dict(analysis.get("observation_status") or {})
        if session_watch:
            observation.update({
                "watch_state": session_watch.get("state", ""),
                "watch_muted": session_watch.get("muted", False),
                "last_run_at": session_watch.get("last_run_at", 0),
                "next_due_at": session_watch.get("next_due_at", 0),
                "run_count": session_watch.get("run_count", 0),
            })
        # De-weight process: fold any kind='process' insight into process_notes so
        # tool mechanics never render as prominent insight cards.
        insights = [i for i in (analysis.get("insights") or []) if isinstance(i, dict)]
        process_notes = [str(n) for n in (analysis.get("process_notes") or []) if str(n).strip()]
        kept_insights = []
        for item in insights:
            if str(item.get("kind", "")).lower() == "process":
                note = str(item.get("summary") or item.get("title") or "").strip()
                if note:
                    process_notes.append(note)
            else:
                kept_insights.append(item)
        analysis["insights"] = kept_insights
        analysis["process_notes"] = process_notes
        data = {
            "title": "Session Analysis",
            "analysis": analysis,
            "observation": observation,
            "artifact_context": analysis.get("artifact_context") or [],
            "findings": session_findings,
            "watch": session_watch,
        }
        return self._render(template, data)

    def _render(self, template: str, data: dict[str, Any]) -> dict[str, Any]:
        """Compile a template and attach the lens list the client switches on.

        Shared by every build path so a new one cannot forget the metadata and leave
        the web client with no way to offer the other lenses.
        """
        name = select_template(template, self._templates.names())
        spec = self._templates.render(name, data)
        if isinstance(spec, dict):
            meta = spec.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["templates"] = self._templates.visible_names()
                meta["hidden_templates"] = self._templates.hidden_names()
                meta["active_template"] = name
        return spec

    async def _build_signals(self, template: str, provider: DashboardDataProvider) -> dict[str, Any]:
        """Build signal flow observation view."""
        metrics_result = await provider.signal_metrics()
        watches = await provider.watches()
        findings = await provider.findings(limit=20)
        raw_metrics = metrics_result.get("metrics", {}) if isinstance(metrics_result, dict) else metrics_result
        metrics = dict(raw_metrics or {}) if isinstance(raw_metrics, dict) else {}

        # Use the raw signal event stream from the daemon ring buffer. Sort by
        # event timestamp descending so "recent" is true on first render; the
        # frontend still receives the full ring buffer and handles tab/limit UI.
        raw_stream = metrics_result.get("signal_stream", []) if isinstance(metrics_result, dict) else []
        stream_events = [dict(evt) for evt in raw_stream if isinstance(evt, dict)]
        stream_events.sort(key=lambda evt: _safe_float(evt.get("ts") or evt.get("timestamp")), reverse=True)
        signal_stream = [_signal_stream_item(evt) for evt in stream_events] or None

        debounce_total = _sum_int_values(metrics.get("debounce_stats"))
        metrics["total_debounced"] = debounce_total
        metrics["signal_stream_count"] = len(stream_events)
        metrics["total_drop_count"] = (
            _safe_int(metrics.get("signal_buffer_dropped"))
            + _safe_int(metrics.get("composite_source_dropped"))
        )

        trigger_rows = []
        for trigger in metrics.get("trigger_stats") or []:
            if not isinstance(trigger, dict):
                continue
            trigger_rows.append({
                "watch": _short_id(trigger.get("watch_id")),
                "pattern": str(trigger.get("pattern") or ""),
                "triggered": "yes" if trigger.get("triggered") else "no",
                "last_event": str(trigger.get("last_event") or ""),
            })

        event_family_rows = [{"family": str(evt.get("family") or "unknown")} for evt in (signal_stream or [])]
        data = {
            "signal_metrics": metrics,
            "signal_stream": signal_stream,
            "watches": watches if watches else None,
            "findings": findings if findings else None,
            "trigger_rows": trigger_rows,
            "event_family_distribution": _distribution(event_family_rows, "family") if event_family_rows else None,
            "watch_state_distribution": _distribution(watches, "state") if watches else None,
            "finding_severity_distribution": _distribution(findings, "severity") if findings else None,
        }
        return self._render(template, data)


__all__ = [
    "DashboardDataProvider",
    "DaemonDataProvider",
    "DashboardViewBuilder",
    "select_template",
]
