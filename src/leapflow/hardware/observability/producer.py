"""The ``hardware`` monitor domain: one finding per cycle, carrying the digest.

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

import logging
from typing import Any, Callable, Sequence

from leapflow.hardware.observability.digest import ALERT_KINDS, build_digest
from leapflow.monitor.types import Evidence, Finding, ProducerContext, Severity

logger = logging.getLogger(__name__)

DOMAIN = "hardware"

_ALERT_EVENTS = ALERT_KINDS
"""Event kinds that make a cycle worth pushing rather than only recording.

Shared with the digest's row severity rather than restated. Held as a second copy,
the two drifted the first time a kind was added: the board coloured the row as an
alert and the producer still declined to push it to anybody.
"""

_DEGRADED_QUALITIES = frozenset({"stale", "saturated"})


class HardwareObservationProducer:
    """Observe every admitted device and emit one finding describing the bench.

    Takes a provider rather than the registry itself, because the registry's
    persistence and experience store are bound during deferred initialization --
    later than this producer is constructed. Resolving per cycle means the first
    cycle after binding sees the store, instead of the producer holding a registry
    it captured before it was fully wired.
    """

    def __init__(self, registry_provider: Callable[[], Any]) -> None:
        self._registry_provider = registry_provider

    @property
    def domain(self) -> str:
        return DOMAIN

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        """Return at most one finding. Never raises."""
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

        severity, headline = _assess(digest)
        return [
            Finding(
                watch_id=ctx.spec.watch_id or DOMAIN,
                domain=DOMAIN,
                title=headline,
                summary=_summary(digest),
                severity=severity,
                ts=digest.generated_at,
                dedup_key=f"hardware:{_state_fingerprint(digest)}",
                tags=("hardware", *sorted({str(event["kind"]) for event in digest.events})),
                evidence=_evidence(digest),
                payload=digest.to_payload(),
            )
        ]


def _assess(digest: Any) -> tuple[Severity, str]:
    """Return the severity and one-line headline for this cycle."""
    kinds = {str(event["kind"]) for event in digest.events}
    breaching = sorted(kinds & _ALERT_EVENTS)
    if breaching:
        return Severity.ALERT, f"Device envelope event: {', '.join(breaching)}"

    degraded = [s for s in digest.series if s.quality_worst in _DEGRADED_QUALITIES]
    if degraded:
        names = ", ".join(sorted(s.id for s in degraded)[:3])
        return Severity.NOTABLE, f"Channel quality degraded: {names}"

    if int(digest.storage.get("write_failures", 0) or 0) > 0:
        # Not cosmetic: dropped windows are the one storage fault that leaves no
        # trace in the data, so the count is the only way it is ever noticed.
        return Severity.NOTABLE, "Sampled history is not being persisted"

    shortfall = [row for row in digest.sampling if _is_behind(row)]
    if shortfall:
        names = ", ".join(sorted(str(row.get("channel_id", "")) for row in shortfall)[:3])
        return Severity.NOTABLE, f"Sampling behind declared rate: {names}"

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
    degraded = sorted(s.id for s in digest.series if s.quality_worst != "ok")
    kinds = sorted({str(event["kind"]) for event in digest.events})
    failing = int(digest.storage.get("write_failures", 0) or 0) > 0
    return "|".join([",".join(degraded), ",".join(kinds), str(failing)])


__all__ = ["DOMAIN", "HardwareObservationProducer"]
