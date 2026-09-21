# Copyright (c) Alibaba, Inc. and its affiliates.
"""DashboardViewBuilder: turn a DashboardIntent + live data into a ViewSpec.

The builder is transport-agnostic: it reads data through a small
``DashboardDataProvider`` protocol (satisfied by a DaemonClient adapter in the
server, or a fake in tests) and renders via the template library. Template
selection is convention-based (intent template, else a template named for the
domain), never a hardcoded domain->file map.
"""

from __future__ import annotations

import logging
import time
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

    async def evolution_projection(self, *, session_id: str) -> dict[str, Any]:
        """Return one session-scoped evolution projection."""
        ...

    async def evolution_projection_aggregate(self) -> dict[str, Any]:
        """Return the explicitly cross-session evolution projection."""
        ...

    async def hardware_inventory(self) -> dict[str, Any]:
        """Return the admitted device fleet grouped by declared class."""
        ...

    async def hardware_device(self, device_id: str) -> dict[str, Any]:
        """Return one device's channels, sampled values, controls and previews."""
        ...

    async def subagent_state(self) -> dict[str, Any]:
        """Return subagent delegation state (active, recent, stats, config)."""
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

    async def evolution_projection(self, *, session_id: str) -> dict[str, Any]:
        return dict(await self._client.evolution_projection(session_id=session_id) or {})

    async def evolution_projection_aggregate(self) -> dict[str, Any]:
        return dict(await self._client.evolution_projection_aggregate() or {})

    async def hardware_inventory(self) -> dict[str, Any]:
        """Return the device fleet, tolerating a daemon without the RPC.

        A refusal is returned as data rather than raised: hardware is off by default,
        and a board whose whole page fails because a subsystem is disabled is worse
        than one that says so in a panel.
        """
        return await self._hardware_call(lambda: self._client.hardware_inventory())

    async def hardware_device(self, device_id: str) -> dict[str, Any]:
        """Return one device's live view, tolerating a daemon without the RPC."""
        return await self._hardware_call(lambda: self._client.hardware_device(device_id))

    async def subagent_state(self) -> dict[str, Any]:
        """Return subagent delegation state, tolerating a daemon without the RPC."""
        try:
            return dict(await self._client.subagent_state() or {})
        except AttributeError:
            return {}
        except Exception:  # noqa: BLE001 - one panel must not lose the board
            logger.debug("dashboard: subagent state read failed", exc_info=True)
            return {}

    @staticmethod
    async def _hardware_call(call: Any) -> dict[str, Any]:
        try:
            return dict(await call() or {})
        except AttributeError:
            # An older daemon than this view client. Named as its own case because the
            # generic handler below would report it as a runtime fault, and "your
            # daemon predates this panel" has a different fix.
            return {
                "ok": False,
                "code": "rpc_unavailable",
                "error": "This leapd build has no hardware inventory RPC; restart the daemon.",
            }
        except Exception as exc:  # noqa: BLE001 - one panel must not lose the board
            logger.debug("dashboard: hardware read failed", exc_info=True)
            return {"ok": False, "code": "unavailable", "error": str(exc)}


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


def _actionable_notes(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the admission decisions that took a capability away.

    The inventory carries every note, which is the truthful record and what the audit
    wants. The board wants the subset a person can act on: a demotion or a rejection
    changed what a device can do, while a ``warning`` is advisory and there is one per
    privacy-gated channel *by design* (rule V10).

    Filtered here rather than at the source because it is a presentation choice, and
    filtering the source would hide the notes from `/board devices` and the audit. Left
    unfiltered, a bench with three cameras showed "Admission decisions: 3" on every page
    under a heading promising to explain why a device is not what it declared -- when
    nothing had been demoted at all.
    """
    return [
        note
        for note in (inventory.get("notes") or [])
        if isinstance(note, dict) and str(note.get("outcome")) in ("demoted", "rejected")
    ]


def _empty_state(domain: str, watch: dict[str, Any]) -> dict[str, Any]:
    """Describe *why* a domain has no payload, so the page can say so.

    A board with no data used to render its metric row as em dashes and, once every
    other section learned to hide itself when empty, collapsed to a lone heading.
    That is indistinguishable from a broken page, and it is the first thing a new
    profile sees.

    Three states, three different next steps, and they are told apart by the watch
    rather than guessed:

    * **unscheduled** -- no watch armed for the domain, so the producer is never
      called. Nothing will ever appear; the scheduler is off or arming failed.
    * **waiting** -- a watch exists but has not completed a cycle yet. Data is
      coming, and the only useful thing to say is when.
    * **idle** -- the watch has run and produced nothing, which for this domain is a
      legitimate quiet answer rather than a fault.
    """
    if not watch:
        state = "unscheduled"
    elif int(watch.get("run_count") or 0) <= 0:
        state = "waiting"
    else:
        state = "idle"
    return {
        "state": state,
        "domain": domain,
        "watch_state": str(watch.get("state") or ""),
        "muted": bool(watch.get("muted")),
        "run_count": int(watch.get("run_count") or 0),
        "next_due_at": float(watch.get("next_due_at") or 0.0),
        "last_run_at": float(watch.get("last_run_at") or 0.0),
    }


def _provenance(payload: dict[str, Any], watch: dict[str, Any]) -> dict[str, Any]:
    """State when the rendered data was observed, when it was last confirmed, and
    whether the watch behind it is keeping its cadence.

    Two instants, deliberately not merged, because merging them is what made the
    board unreadable:

    * ``observed_at`` -- when the content on screen was produced. Findings dedup on
      a content fingerprint, so an unchanged subject keeps its previous finding and
      this can legitimately be hours old.
    * ``checked_at`` -- when a cycle last completed. It advances on every run,
      including one that concluded nothing had changed and therefore wrote nothing.

    Reporting only the first made a correct system look broken: pressing refresh ran
    a real cycle, the fingerprint matched, no finding was written, and every figure
    on the page -- including its own age -- stayed frozen. Reporting only the second
    would be worse: it would age a stale page from the clock and call it current.

    So ``stale`` is a verdict about the *watch*, not the content: a cycle has not
    completed in two cadences, which means the page cannot be trusted to reflect
    anything. Content that is old but freshly confirmed is not stale, it is stable,
    and ``unchanged_for_seconds`` says how long it has been so.

    Two cadences rather than a fixed number of seconds: one missed cycle is
    scheduling jitter, two is a cadence that is not being kept. With no declared
    cadence (event, cron, condition) the verdict is withheld rather than guessed --
    judging an event-driven watch against an invented interval would be the same
    class of error this block exists to remove.
    """
    now = time.time()
    observed_at = float(payload.get("observed_at") or 0.0)
    checked_at = float(watch.get("last_run_at") or 0.0)
    # A payload with no instant of its own is dated by the cycle that produced it.
    if observed_at <= 0.0:
        observed_at = checked_at
    cadence = float(watch.get("interval_seconds") or 0.0)
    next_due_at = float(watch.get("next_due_at") or 0.0)
    return {
        "observed_at": observed_at,
        "age_seconds": max(0.0, now - observed_at) if observed_at > 0.0 else 0.0,
        "checked_at": checked_at,
        "checked_age_seconds": max(0.0, now - checked_at) if checked_at > 0.0 else 0.0,
        # How long the subject has held still. Only meaningful once a check has
        # happened after the observation; otherwise there is nothing to compare.
        "unchanged_for_seconds": (
            max(0.0, checked_at - observed_at)
            if checked_at > 0.0 and observed_at > 0.0
            else 0.0
        ),
        "cadence_seconds": cadence,
        "next_due_at": next_due_at,
        "next_due_in_seconds": max(0.0, next_due_at - now) if next_due_at > 0.0 else 0.0,
        "run_count": int(watch.get("run_count") or 0),
        "watch_state": str(watch.get("state") or ""),
        "muted": bool(watch.get("muted")),
        # The identity of what produced the page, so the bar can offer a refresh
        # rather than only report that the page is behind. Being able to re-run the
        # cycle from here is what turns the freshness reading into a closed loop:
        # the reader changes something the board told them to change, refreshes, and
        # sees whether it landed.
        "watch_id": str(watch.get("watch_id") or ""),
        # Absent is not the same as fresh. A board whose watch never ran has no
        # instant to age, and saying "0s ago" there would invent one.
        "observed": observed_at > 0.0,
        "checked": checked_at > 0.0,
        "stale": bool(
            cadence > 0.0
            and checked_at > 0.0
            and (now - checked_at) > cadence * 2
        ),
    }


def _evidence_trend(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn the retained framework_evolution findings into one evolution axis.

    The board could say what the framework *is* and never what direction it is
    moving. Every panel was a cross-section of one instant; the only time-shaped
    element was the episode timeline, which is gated on episodes and therefore hidden
    in exactly the state where "has anything started yet" is the question being asked.

    So the axis is built from the findings already retained for the domain. They are
    persisted newest-first and deduped on content, which makes them a record of
    *changes* rather than of ticks -- one point per distinct state, which is the right
    granularity for this question and cheaper than sampling every cycle.

    A single point is emitted and drawn. It reads "one observation, here it is",
    which is a fact; suppressing it until a second arrives would hide the beginning
    of every evolution the board exists to show.
    """
    points: list[dict[str, Any]] = []
    for finding in reversed(findings or []):  # oldest first, so the axis reads left to right
        payload = finding.get("payload")
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        with_evidence = summary.get("segments_with_evidence")
        if with_evidence is None:
            continue
        observed_at = float(payload.get("observed_at") or finding.get("ts") or 0.0)
        points.append({
            "x": observed_at,
            "y": int(with_evidence),
            "at": observed_at,
        })
    total = 0
    for finding in findings or []:
        payload = finding.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
            total = int(payload["summary"].get("segments_total") or 0)
            break
    return {
        "series": [{"label": "Segments with runtime evidence", "points": points}] if points else [],
        "samples": len(points),
        "segments_total": total,
        "first_at": points[0]["at"] if points else 0.0,
        "last_at": points[-1]["at"] if points else 0.0,
        # Direction, stated rather than left to the reader's eye on a two-point line.
        "delta": (points[-1]["y"] - points[0]["y"]) if len(points) > 1 else 0,
    }


def _hardware_notice(
    inventory: dict[str, Any], digest: dict[str, Any]
) -> dict[str, str] | None:
    """Explain an empty bench, or return None when there is something to show.

    Every panel on this board is gated on the data it renders, which is what makes a new
    peripheral appear without a template edit -- and also what made the page render *bare*
    when there was no data at all. A blank tab under a title is not an answer to "what
    hardware do you see"; it is the same silently-gated-out failure the calibration panel
    had, one level up.

    Each branch names the next step, because the situations are genuinely different and
    only the message can tell them apart: hardware off is a config change, no devices is a
    scan or a declaration, and a daemon still starting will fix itself.
    """
    if inventory.get("ok") and (inventory.get("groups") or digest.get("devices")):
        return None

    code = str(inventory.get("code") or "")
    if code == "hardware_disabled":
        return {
            "title": "Hardware is disabled for this profile",
            "text": (
                "No peripherals are visible because the hardware subsystem is off. Enable "
                "it with `leap config set hardware.enabled true`, then restart the daemon "
                "(`leap daemon restart`). It is off by default: with it disabled LeapFlow "
                "exposes no device tools and behaves exactly as it did before the "
                "subsystem existed."
            ),
        }
    if code == "runtime_unavailable":
        return {
            "title": "leapd is still starting",
            "text": "This panel fills in on its own once the daemon has finished initializing.",
        }
    if code in ("rpc_unavailable", "unavailable"):
        return {
            "title": "This board cannot reach the hardware subsystem",
            "text": (
                f"{inventory.get('error') or 'The inventory could not be read.'} "
                "The board runs as its own long-lived process and does not pick up new "
                "code until it is restarted -- run `leap board` again if you have just "
                "upgraded."
            ),
        }
    return {
        "title": "No devices are attached",
        "text": (
            "Hardware is enabled but nothing was admitted. Run `/board rescan` after "
            "attaching a peripheral, or check `hardware.providers` and the profile's "
            "declaration directory. Admission decisions, including anything rejected, "
            "appear below when there are any."
        ),
    }


def _identity_rows(identity: dict[str, Any], device: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the declaration's provenance into ``{field, value}`` table rows.

    Flattened here rather than bound field-by-field in the template because the path
    resolver walks mapping keys and list indices only -- it cannot iterate a mapping's
    pairs, so a template asking for one renders nothing at all. Empty values are
    dropped so an unverified device shows a shorter table rather than a column of
    blanks.
    """
    candidates = (
        ("Device id", identity.get("device_id")),
        ("Vendor", device.get("vendor")),
        ("Model", device.get("model")),
        ("Location", device.get("location")),
        ("Declared by", identity.get("provenance_source")),
        ("Verified by", identity.get("verified_by") or "not verified"),
        ("Emergency stop", "supported" if device.get("halt_supported") else "not supported"),
        ("Notes", identity.get("notes")),
        ("Provenance notes", identity.get("provenance_notes")),
        ("Unmapped fields", ", ".join(identity.get("lossy_fields") or ())),
    )
    return [
        {"field": label, "value": str(value)}
        for label, value in candidates
        if value not in (None, "", [])
    ]


def _hardware_error(view: dict[str, Any], device_id: str) -> str:
    """Turn a structured hardware refusal into a sentence with a next step.

    The raw ``code`` is for the audit log. Surfacing it alone gives a person
    ``unknown_device`` as the entire content of a page, which names the problem
    without saying what to do about it.
    """
    code = str(view.get("code") or "")
    if code == "unknown_device":
        return (
            f"No device {device_id!r} is admitted. It may have been detached, or its "
            "declaration may have been rejected -- run `leap hw scan`, then check "
            "`leap hw list`."
        )
    if code == "hardware_disabled":
        return (
            "Hardware is disabled for this profile. Run "
            "`leap config set hardware.enabled true` and restart the daemon."
        )
    if code == "runtime_unavailable":
        return "leapd is still starting up. This panel will fill in once it is ready."
    return str(view.get("error") or "This device could not be read.")


def _short_id(value: Any) -> str:
    text = str(value or "")
    return text[:8] if len(text) > 8 else text


_PAYLOAD_DOMAINS: dict[str, tuple[str, str]] = {
    # template -> (finding domain, data key the template binds to)
    "capability": ("capability_adaptation", "capability_plan"),
    "evolution": ("framework_evolution", "evolution"),
    # The hidden demo/audit lens reads the same immutable evolution snapshot.
    # It adds no data source or mutation path.
    "causal_trace": ("framework_evolution", "evolution"),
    # The live lens consumes the same authoritative snapshot and only augments it
    # with presentation events delivered through the browser WebSocket.
    "evolution_live": ("framework_evolution", "evolution"),
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

HARDWARE_TEMPLATE = "hardware"
"""The one hardware lens. It renders the fleet, or one device when the intent names one.

They were two templates (``hardware`` and ``hardware_device``) and the split did not pay
for itself: the names read as near-synonyms in the lens list, and the second was not a
different *way of looking* at anything -- only a different subject. One lens with an
optional target is the honest shape, and it is also what makes a drill-down reversible
without the operator learning a second name.
"""

_DOMAIN_FINDINGS_LIMIT = 10
"""Rows fetched when a lens wants the newest finding of one domain.

Only the newest is rendered, so this is deliberately small: the fetch is scoped to a
single domain's watch, and a handful of rows is ample slack for that domain's own
dedup without pulling a batch of large payloads the page never reads.
"""


class DashboardViewBuilder:
    """Assemble ViewSpecs for dashboard intents."""

    def __init__(self, templates: TemplateLibrary | None = None) -> None:
        self._templates = templates or TemplateLibrary()

    async def build(self, intent: DashboardIntent, provider: DashboardDataProvider) -> dict[str, Any]:
        """Return a normalized ViewSpec for the requested template.

        Four data shapes, not one: the signal pipeline's own metrics, one device read
        live, a producer's finding payload, or the session analysis every other lens
        renders. The hardware lens picks between the second and third by whether the
        intent named a device.
        """
        template_name = intent.template
        if template_name == "signals":
            return await self._build_signals(template_name, provider)
        if template_name == "subagents":
            return await self._build_subagents(template_name, provider)
        if template_name == HARDWARE_TEMPLATE and intent.device:
            return await self._build_device(template_name, intent, provider)
        payload_domain = _PAYLOAD_DOMAINS.get(template_name)
        if payload_domain is not None:
            return await self._build_from_finding_payload(
                template_name,
                provider,
                *payload_domain,
                session_id=intent.session_id,
            )
        return await self._build_session(intent.template, provider)

    async def _build_device(
        self, template: str, intent: DashboardIntent, provider: DashboardDataProvider
    ) -> dict[str, Any]:
        """Render one device from a live read, with the fleet alongside for navigation.

        The inventory is fetched too so the page can offer the other devices without a
        round trip back to the fleet view -- a drill-down that dead-ends is a drill-down
        people stop using.

        Every section gates on the data it renders rather than a synthetic mode flag. That
        makes a device page contain its identity and channels, while the shared inventory
        remains usable to navigate to another device; an old builder can never blank a new
        template by omitting an invented key.
        """
        device_id = intent.device
        inventory = await provider.hardware_inventory()
        view = await provider.hardware_device(device_id)
        data = {
            "title": str(((view.get("device") or {}).get("label")) or device_id),
            "inventory": inventory,
            "admission_notes": _actionable_notes(inventory),
            "device_id": device_id,
            "channel_id": intent.channel,
        }
        if view.get("ok"):
            identity = view.get("identity") or {}
            previews = [dict(item) for item in (view.get("previews") or [])]
            # A Preview button named the target channel in the intent. Mark only that
            # panel so a fleet row has one unambiguous action: it opens the device *and*
            # starts that device's preview, instead of navigating somewhere that asks the
            # person to click another Start button.
            for preview in previews:
                preview["autostart"] = bool(
                    intent.channel and preview.get("channel_id") == intent.channel
                )
            data.update({
                "device": view.get("device") or {},
                "identity": identity,
                "identity_rows": _identity_rows(identity, view.get("device") or {}),
                "channels": view.get("channels") or None,
                "traces": view.get("traces") or None,
                "controls": view.get("controls") or None,
                "previews": previews or None,
                "events": view.get("events") or None,
            })
        else:
            data["device_error"] = _hardware_error(view, device_id)
        return self._render(template, data)

    async def _domain_findings(
        self, provider: DashboardDataProvider, domain: str, watch: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Return the newest findings of one *watch*, or of the domain when unarmed.

        Scoped by ``watch_id`` rather than fetched across all domains and filtered:
        findings return newest-first inside a byte-bounded batch, and the hardware
        payload is far larger than any other, so an unscoped read lets a burst of
        hardware findings fill the batch and push a still-current session or
        capability finding off the end -- blanking a board whose data plainly exists.
        A scoped read asks the store for only the domain the page renders. When no
        watch has been armed for the domain yet there is no id to scope by, so it
        falls back to an unscoped read and filters, which is correct because a
        domain with no watch also has no findings.

        Note the unit: this is per *watch*, not per domain. A domain armed with more
        than one watch must be resolved by :meth:`_domain_watch_payload`, which asks
        each of them; calling this once with an arbitrary watch was how a board went
        blank while its data existed under a sibling watch.
        """
        watch_id = str(watch.get("watch_id") or "")
        findings = await provider.findings(watch_id=watch_id, limit=_DOMAIN_FINDINGS_LIMIT)
        if watch_id:
            return findings
        return [f for f in findings if str(f.get("domain")) == domain]

    async def _domain_watch_payload(
        self, provider: DashboardDataProvider, finding_domain: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Resolve the newest finding of a domain across *every* watch armed for it.

        A domain can legitimately carry more than one watch: framework evolution is
        armed twice, once polled for state that has no event and once event-driven for
        change that polling would report minutes late. Picking the first match and
        scoping the finding read to it meant that whenever the arbitrary winner was the
        event watch -- which has produced nothing until something evolves -- the page
        received an empty payload and rendered a column of em dashes. The data existed
        the whole time, under the sibling watch.

        So each watch of the domain is asked, and the newest finding across all of them
        wins. The returned watch is the one that *produced* that finding, because the
        observation metadata (last run, next due, run count) describes how the rendered
        data was obtained; taking it from a different watch would report a cadence that
        had nothing to do with what is on screen.
        """
        watches = [
            w for w in await provider.watches() if str(w.get("domain")) == finding_domain
        ]
        if not watches:
            # No watch armed: one unscoped read, filtered by domain.
            findings = await self._domain_findings(provider, finding_domain, {})
            return ({}, findings)

        best_watch: dict[str, Any] = watches[0]
        best_findings: list[dict[str, Any]] = []
        best_ts = float("-inf")
        for candidate in watches:
            findings = await self._domain_findings(provider, finding_domain, candidate)
            if not findings:
                continue
            ts = float(findings[0].get("ts") or 0.0)
            if ts > best_ts:
                best_ts, best_findings, best_watch = ts, findings, candidate
        return (best_watch, best_findings)

    async def _build_from_finding_payload(
        self,
        template: str,
        provider: DashboardDataProvider,
        finding_domain: str,
        data_key: str,
        *,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Render a template from the newest finding of one producer domain.

        Newest rather than merged: each of these payloads is a self-consistent
        snapshot of a subject at one instant, and stitching two together would show
        a state that never existed.
        """
        watch, domain_findings = await self._domain_watch_payload(provider, finding_domain)
        payload = dict(domain_findings[0].get("payload") or {}) if domain_findings else {}
        event_projection: dict[str, Any] = {}
        if finding_domain == "framework_evolution":
            event_projection = await self._event_projection(provider, session_id=session_id)
            if event_projection:
                previous_summary = dict(payload.get("summary") or {})
                payload.update(event_projection)
                payload["summary"] = {
                    **previous_summary,
                    **dict(event_projection.get("summary") or {}),
                }
        data = {
            "title": template.replace("_", " ").title(),
            data_key: payload,
            "findings": domain_findings or None,
            "watch": watch,
            "provenance": _provenance(payload, watch),
            "observation": {
                "watch_state": watch.get("state", ""),
                "watch_muted": watch.get("muted", False),
                "last_run_at": watch.get("last_run_at", 0),
                "next_due_at": watch.get("next_due_at", 0),
                "run_count": watch.get("run_count", 0),
            },
        }
        if not payload:
            # An empty payload is a state the page must be able to explain, not a
            # reason to render a column of em dashes. Which state it is matters: a
            # domain whose watch has never run is waiting, one armed and running is
            # idle with nothing to report, and no watch at all means the producer is
            # not being scheduled -- three different next steps.
            data["empty"] = _empty_state(finding_domain, watch)
        if finding_domain == "framework_evolution":
            # A sibling top-level key, not folded into the bound payload. The producer
            # owns everything under ``evolution.*`` and a test asserts that contract;
            # this series is derived here because only the retained finding list
            # carries history, and hiding a service-derived figure inside the
            # producer's namespace would make that ownership unreadable.
            data["evidence_trend"] = _evidence_trend(domain_findings or [])
        if template == "hardware":
            # The fleet list comes from the live registry rather than the cycle payload.
            # The digest is capped at eight charted channels and is up to a monitor
            # interval old, so a device attached since the last cycle would be missing
            # from the one panel whose whole job is to say what is attached.
            inventory = await provider.hardware_inventory()
            data["inventory"] = inventory
            data["admission_notes"] = _actionable_notes(inventory)
            data["title"] = "Physical bench"
            notice = _hardware_notice(inventory, payload)
            if notice is not None:
                data["notice"] = notice
        rendered = self._render(template, data)
        if event_projection:
            rendered.setdefault("meta", {})["evolution_projection"] = {
                "scope": event_projection.get("scope", "aggregate"),
                "session_id": event_projection.get("session_id", ""),
                "last_sequence": event_projection.get("last_sequence", 0),
            }
        return rendered

    @staticmethod
    async def _event_projection(
        provider: DashboardDataProvider,
        *,
        session_id: str,
    ) -> dict[str, Any]:
        """Read the canonical event projection, degrading for older daemons."""
        try:
            if session_id:
                result = await provider.evolution_projection(session_id=session_id)
            else:
                result = await provider.evolution_projection_aggregate()
        except (AttributeError, RuntimeError, TypeError):
            logger.debug("dashboard: evolution projection unavailable", exc_info=True)
            return {}
        if not result.get("ok"):
            return {}
        projection = result.get("projection")
        return dict(projection) if isinstance(projection, dict) else {}

    async def _build_session(self, template: str, provider: DashboardDataProvider) -> dict[str, Any]:
        # The session watch emits an insight finding whose payload carries the
        # structured analysis plus observation transparency metadata.
        watches = await provider.watches()
        session_watch = next((w for w in watches if str(w.get("domain")) == "session"), {})
        session_findings = await self._domain_findings(provider, "session", session_watch)
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
            "provenance": _provenance(analysis, session_watch),
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
                # Freshness rides in ``meta`` rather than the component tree so it is
                # page chrome on every lens at once, and so a template author cannot
                # forget to bind it. Lifted here for the same reason the lens list is:
                # this is the one place every build path passes through. A build path
                # that computes no provenance says nothing rather than claiming the
                # page is current.
                provenance = data.get("provenance")
                if isinstance(provenance, dict):
                    meta["provenance"] = provenance
        return spec

    async def _build_subagents(self, template: str, provider: DashboardDataProvider) -> dict[str, Any]:
        """Build subagent monitor view from live SubagentManager state."""
        state = await provider.subagent_state()
        active = state.get("active") or []
        recent = state.get("recent") or []
        stats = state.get("stats") or {}
        config = state.get("config") or {}

        # Build timeline items from recent history
        recent_timeline = [
            {
                "title": item.get("goal", "")[:80],
                "summary": (
                    f"{item.get('status', '')} in {item.get('duration_s', 0)}s"
                    f" ({item.get('tool_calls', 0)} tools)"
                ),
                "severity": "success" if item.get("status") == "completed" else "error",
                "ts": item.get("timestamp", 0),
            }
            for item in recent
        ] or None

        # Build delegation tree (parent→child from recent + active)
        all_entries = list(active) + list(recent)
        delegation_tree = [
            {
                "parent_session_id": _short_id(entry.get("parent_session_id")),
                "subagent_id": _short_id(entry.get("subagent_id")),
                "goal": entry.get("goal", "")[:80],
                "depth": entry.get("depth", 0),
                "status": entry.get("status", "running"),
                "duration_s": entry.get("duration_s", entry.get("elapsed_s", "—")),
            }
            for entry in all_entries
        ] or None

        # Distributions for charts
        depth_dist = _distribution(all_entries, "depth") if all_entries else None
        outcome_dist = _distribution(
            [{"outcome": r.get("status", "unknown")} for r in recent],
            "outcome",
        ) if recent else None

        total_tool_calls = sum(r.get("tool_calls", 0) for r in recent)
        finished = stats.get("completed", 0) + stats.get("failed", 0)
        total_duration = round(stats.get("avg_duration", 0) * finished, 1)

        data: dict[str, Any] = {
            "title": "Sub-Agent Monitor",
            "subagent": {
                "active": active or None,
                "active_count": len(active),
                "recent": recent or None,
                "recent_timeline": recent_timeline,
                "stats": stats if stats.get("total_delegated", 0) > 0 else None,
                "config": config or None,
                "delegation_tree": delegation_tree,
                "depth_distribution": depth_dist,
                "outcome_distribution": outcome_dist,
                "total_tool_calls": total_tool_calls,
                "total_duration": total_duration,
            },
        }
        if not state or stats.get("total_delegated", 0) == 0:
            data["empty"] = {"state": "no_delegations", "config": config}
        return self._render(template, data)

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
    "HARDWARE_TEMPLATE",
    "DashboardDataProvider",
    "DaemonDataProvider",
    "DashboardViewBuilder",
    "select_template",
]
