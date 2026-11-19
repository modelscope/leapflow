# Copyright (c) Alibaba, Inc. and its affiliates.
"""The ``hardware`` monitor domain: episode notifications and current-state digests.

The only file here with a side effect, and the only one the daemon wires. It
implements ``MonitorProducer`` so device observation reaches the board through the
same watch schedule, finding store and push path as every other domain -- rather
than a second reporting mechanism that would need its own RPC, its own client and
its own failure modes.

Severity is derived from what was observed, not fixed, because it decides whether
the finding is merely persisted or actually pushed: an envelope breach must reach
someone, and a healthy bench must not.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Callable, Sequence

from leapflow.hardware.advisory import ADVISORY_KINDS
from leapflow.hardware.observability.digest import ALERT_KINDS, build_digest
from leapflow.hardware.observation import Episode
from leapflow.hardware.observability.series import _fit
from leapflow.monitor.types import Evidence, Finding, ProducerContext, Severity

logger = logging.getLogger(__name__)

DOMAIN = "hardware"

_ALERT_EVENTS = ALERT_KINDS
"""Event kinds that make a cycle worth pushing rather than only recording.

Shared with the digest's row severity rather than restated. Held as a second copy,
the two drifted the first time a kind was added: the board coloured the row as an
alert and the producer still declined to push it to anybody.
"""


class HardwareObservationProducer:
    """Observe every admitted condition, independently of chart and event-ring limits.

    Takes a provider rather than the registry itself, because the registry's
    persistence and experience store are bound during deferred initialization --
    later than this producer is constructed. Resolving per cycle means the first
    cycle after binding sees the store, instead of the producer holding a registry
    it captured before it was fully wired.
    """

    def __init__(
        self, registry_provider: Callable[[], Any], *, finding_store: Any = None, run_db: Any = None
    ) -> None:
        self._registry_provider = registry_provider
        self._finding_store = finding_store
        self._run_db = run_db
        self._checkpoints: dict[str, dict[str, Any]] = {}

    @property
    def domain(self) -> str:
        return DOMAIN

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        """Return uncheckpointed transitions or a current-state digest."""
        try:
            registry = self._registry_provider()
        except Exception as exc:  # noqa: BLE001 - a board panel is not worth a failed watch
            logger.debug("hardware watch: registry unavailable: %s", exc)
            return []
        if registry is None:
            return []

        digest = build_digest(registry, now=ctx.now)
        if digest.is_empty:
            # Hardware is off by default and a profile with no declarations is the
            # normal case. Emitting an empty finding every cycle would fill the
            # store with rows saying nothing happened.
            return []

        if self._finding_store is not None:

            def query() -> dict[str, dict[str, Any]]:
                return self._finding_store.latest_notification_checkpoints(DOMAIN)

            try:
                checkpoints = (
                    await self._run_db(query) if self._run_db else await asyncio.to_thread(query)
                )
            except Exception:  # noqa: BLE001 - failed checkpoint reads must not re-page incidents
                logger.warning("hardware watch: checkpoint read failed", exc_info=True)
                return []
        else:
            checkpoints = dict(self._checkpoints)
        acknowledge = getattr(registry, "acknowledge_observation", None)
        if callable(acknowledge):
            acknowledge(checkpoints)
        findings = []
        monitored = {row["key"] for row in digest.conditions}
        conditions = list(digest.conditions)
        # Command-path faults and older/external producers may not have a streaming
        # detector. Their individual evidence is still reported once, never as a set
        # of whatever happens to remain in the event ring.
        for event in reversed(digest.events):
            kind = str(event.get("kind", ""))
            condition = f"advisory:{event.get('quantity', '')}" if kind in ADVISORY_KINDS else kind
            key = f"{event.get('device_id', '')}/{event.get('channel_id', '')}/{condition}"
            if (kind not in _ALERT_EVENTS and kind not in ADVISORY_KINDS) or key in monitored:
                continue
            episode = hashlib.sha256(
                repr((key, event.get("x"), event.get("detail"))).encode()
            ).hexdigest()[:24]
            phase = "recovery" if kind == "advisory_cleared" else "entry"
            conditions.append(
                {
                    "key": key,
                    "pending": (
                        {
                            "key": key,
                            "condition": condition,
                            "episode": episode,
                            "runtime_id": episode,
                            "revision": 1,
                            "phase": phase,
                            "active": phase != "recovery",
                            "known": True,
                            "reason": event.get("detail", ""),
                            "device_id": event.get("device_id", ""),
                            "channel_id": event.get("channel_id", ""),
                            "observed_at": event.get("x", 0.0),
                            "event_only": True,
                            "identity": episode,
                        },
                    ),
                }
            )
        base = digest.to_payload()
        for row in conditions:
            pending = list(row.get("pending", ()))
            restored = checkpoints.get(row["key"], {})
            # Fresh healthy evidence can close a prior runtime's incident even when
            # debounce correctly prevented a new entry in this runtime.
            if (not pending and restored.get("active") and row.get("known")
                    and row.get("clear") and not row.get("active")
                    and row.get("identity") == restored.get("identity")
                    and row.get("runtime_id") != restored.get("runtime_id")):
                pending.append({**row, "phase": "recovery", "episode": row.get("episode", ""),
                                "revision": row.get("revision", 0)})
            for notice in pending:
                checkpoint = checkpoints.get(row["key"], {})
                notification = _notification(notice, checkpoint, ctx.now)
                if notification is None:
                    continue
                checkpoints[row["key"]] = notification
                phase, condition = notification["phase"], notice["condition"]
                hard = condition in _ALERT_EVENTS
                severity = Severity.NOTABLE if phase == "recovery" or not hard else Severity.ALERT
                disabled = condition == "threshold_exceeded" and _actuation_disabled(registry, notice, ctx.now)
                notification["actuation_disabled"] = disabled
                if disabled:
                    severity = Severity.NOTABLE
                where = f"{notice.get('device_id', '')}.{notice.get('channel_id', '')}"
                detail = (
                    f"{where}: {notice.get('reason', condition)}; raw={notice.get('value')}, "
                    f"soft={notice.get('soft_min')}..{notice.get('soft_max')}, "
                    f"hard={notice.get('hard_min')}..{notice.get('hard_max')}, "
                    f"margin={notice.get('margin', 0):g}, duration={notice.get('duration_s', 0):.2f}s"
                )
                if disabled and phase != "recovery":
                    detail += (
                        "; disabled resting pose is outside the declared command range; "
                        "motion from this pose may be refused. Check zero calibration or "
                        "return within range under supervision; do not move automatically."
                    )
                payload = _fit({**base, "notification": notification})
                findings.append(
                    Finding(
                        watch_id=ctx.spec.watch_id or DOMAIN,
                        domain=DOMAIN,
                        title=f"Hardware {condition} {phase}: {where}",
                        summary=detail,
                        severity=severity,
                        ts=ctx.now,
                        dedup_key=f"hardware:{row['key']}:{notification['episode']}:{notification['sequence']}",
                        tags=(DOMAIN, condition, phase),
                        evidence=(Evidence(kind="metric", label=where, value=detail),),
                        payload=payload,
                    )
                )
        if self._finding_store is None:
            self._checkpoints = checkpoints
        if findings:
            return findings
        _, headline = _assess(digest)
        # Current-state snapshots feed the existing board. Historical events are
        # evidence only; an unchanged condition cannot notify through chart churn.
        snapshot_severity = Severity.INFO
        if int(digest.storage.get("write_failures", 0) or 0) > 0:
            snapshot_severity, headline = Severity.NOTABLE, "Sampled history is not being persisted"
        elif any(_is_behind(row) for row in digest.sampling):
            snapshot_severity, headline = Severity.NOTABLE, "Sampling behind declared rate"
        return [
            Finding(
                watch_id=ctx.spec.watch_id or DOMAIN,
                domain=DOMAIN,
                title=headline,
                summary=_summary(digest),
                severity=snapshot_severity,
                ts=ctx.now,
                dedup_key=f"hardware:snapshot:{_state_fingerprint(digest)}",
                tags=(DOMAIN,),
                evidence=_evidence(digest),
                payload=base,
            )
        ]


def _assess(digest: Any) -> tuple[Severity, str]:
    """Return the severity and one-line headline for this cycle."""
    breaching = sorted(
        {
            str(row.get("condition", ""))
            for row in digest.conditions
            if row.get("active") and row.get("known") and row.get("condition") in _ALERT_EVENTS
        }
    )
    if breaching:
        return Severity.ALERT, f"Device envelope event: {', '.join(breaching)}"

    if int(digest.storage.get("write_failures", 0) or 0) > 0:
        # Not cosmetic: dropped windows are the one storage fault that leaves no
        # trace in the data, so the count is the only way it is ever noticed.
        return Severity.NOTABLE, "Sampled history is not being persisted"

    shortfall = [row for row in digest.sampling if _is_behind(row)]
    if shortfall:
        names = ", ".join(sorted(str(row.get("channel_id", "")) for row in shortfall)[:3])
        return Severity.NOTABLE, f"Sampling behind declared rate: {names}"

    warnings = [row for row in digest.conditions if row.get("active") and row.get("known")]
    if warnings:
        return Severity.NOTABLE, f"Operational hardware warning: {warnings[0].get('key', '')}"
    if not digest.conditions or any(not row.get("known") for row in digest.conditions):
        return Severity.INFO, "Hardware observation unknown or awaiting fresh evidence"
    return Severity.INFO, f"{len(digest.devices)} device(s) within declared envelopes"


def _is_behind(row: dict[str, Any]) -> bool:
    """Whether a channel is sampling materially slower than it declared.

    The 20% allowance is scheduling jitter, not drift; below that the loop is
    genuinely not keeping its cadence and every window under-counts.
    """
    declared = float(row.get("declared_hz") or 0.0)
    ratio = float(row.get("rate_ratio") or 0.0)
    return declared > 0 and 0.0 < ratio < 0.8


def _summary(digest: Any) -> str:
    parts = [
        f"{len(digest.devices)} device(s)",
        f"{len(digest.series)} charted channel(s)",
        f"{len(digest.events)} recent event(s)",
    ]
    failures = int(digest.storage.get("write_failures", 0) or 0)
    if failures:
        parts.append(f"{failures} unpersisted window(s)")
    return ", ".join(parts)


def _evidence(digest: Any) -> tuple[Evidence, ...]:
    """Cite the newest events. Evidence is the audit trail for a pushed finding."""
    rows: list[Evidence] = []
    for event in digest.events[:5]:
        where = f"{event.get('device_id', '')}.{event.get('channel_id', '')}"
        rows.append(
            Evidence(
                kind="metric",
                label=f"{event.get('kind', '')} · {where}",
                value=str(event.get("detail", "")),
            )
        )
    return tuple(rows)


def _state_fingerprint(digest: Any) -> str:
    """Identify the bench *state*, so an unchanged bench does not re-notify.

    Built from what a watcher would react to -- which channels are degraded, which
    event kinds are live, whether persistence is failing -- and deliberately not
    from the trace itself, which changes on every cycle and would defeat dedup
    entirely.
    """
    states = sorted(
        (row.get("key", ""), bool(row.get("active")), bool(row.get("known")))
        for row in digest.conditions
    )
    failing = int(digest.storage.get("write_failures", 0) or 0) > 0
    behind = sorted(str(row.get("channel_id", "")) for row in digest.sampling if _is_behind(row))
    return hashlib.sha256(repr((states, failing, behind)).encode()).hexdigest()[:24]


def _actuation_disabled(registry: Any, notice: dict[str, Any], now: float) -> bool:
    """Only fresh, explicit disabled evidence lowers reporting severity, never safety."""
    context_for = getattr(registry, "context", None)
    sample_for = getattr(registry, "latest_sample", None)
    if not callable(context_for) or not callable(sample_for):
        return False
    context = context_for(notice.get("device_id", ""))
    state_id = getattr(context, "actuation_state_channel", "")
    if not state_id or context.hc_version != "hc.v1":
        return False
    channel = context.channel(state_id)
    if channel is None or not channel.is_readable or channel.sample_rate_hz <= 0:
        return False
    reading = sample_for(context.device_id, state_id)
    if reading is None or not reading.is_trustworthy:
        return False
    age = now - reading.observed_at
    return 0 <= age <= 2.0 / channel.sample_rate_hz and (
        reading.value is False or type(reading.value) in (int, float) and reading.value == 0
    )


def _notification(
    notice: dict[str, Any], checkpoint: dict[str, Any], now: float
) -> dict[str, Any] | None:
    """Map source episodes to persisted notification progress, including restarts."""
    if (
        notice.get("event_only")
        and checkpoint
        and notice.get("observed_at", 0) <= checkpoint.get("observed_at", 0)
    ):
        return None
    same_runtime = checkpoint.get("runtime_id") == notice.get("runtime_id")
    if same_runtime and notice.get("revision", 0) <= checkpoint.get("revision", 0):
        return None
    phase = notice["phase"]
    continued = (
        checkpoint.get("active")
        and checkpoint.get("identity") == notice.get("identity")
        and (not same_runtime or checkpoint.get("source_episode") == notice.get("episode"))
    )
    if phase == "entry" and continued:
        phase = "reminder"  # restored progress is not restored healthy evidence
    confirm_s = float(notice.get("confirm_s", 0.5 if notice.get("condition") == "stale" else 0.0))
    if phase == "reminder" and continued and not same_runtime and notice.get("duration_s", 0.0) < confirm_s:
        return None
    if phase == "reminder" and continued and now < checkpoint.get("next_report_at", 0.0):
        return None
    episode = checkpoint["episode"] if continued else notice["episode"]
    sequence = (
        int(checkpoint.get("sequence", 0)) + 1 if continued else int(notice.get("sequence", 0))
    )
    after = float(notice.get("report_after_s", 60.0))
    maximum = float(notice.get("report_max_s", 900.0))
    clock = Episode(active=True, interval=float(checkpoint.get("interval", after)), next_report=0.0)
    if phase == "reminder":
        clock.observe(
            breached=True, clear=False, now=now, report_after_s=after, report_max_s=maximum
        )
    else:
        clock.interval, clock.next_report = after, now + after
    return {
        **notice,
        "phase": phase,
        "episode": episode,
        "sequence": sequence,
        "source_episode": notice["episode"],
        "active": phase != "recovery",
        "next_report_at": clock.next_report,
        "interval": clock.interval,
        "notified_at": now,
    }


__all__ = ["DOMAIN", "HardwareObservationProducer"]
