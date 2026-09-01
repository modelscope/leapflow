"""Prometheus-style metrics exporter for hardware observability.

Maps ``ReadingStore`` and ``HardwareStreamSource`` counters to named gauge and
counter values. Default **off** (``hardware.metrics_export_enabled=False``): when
disabled, the exporter is never instantiated and adds zero overhead to the
sampling path. When enabled, ``collect()`` returns the current snapshot as a list
of ``MetricSample`` tuples suitable for a ``/metrics`` endpoint to render in the
Prometheus exposition format.

No dependency on a Prometheus client library by design: this module owns the
metric names, labels, and values. A ``/metrics`` HTTP handler — wired elsewhere
when the flag is on — renders the output as text, keeping the exporter testable
and importable without any optional dependency installed.

Thread-safety: ``collect()`` reads counters that are only incremented from the
sampling loop (single-threaded per source), so it observes a consistent snapshot
without a lock. The snapshot is stale by up to one sampling interval, which is
the intended behaviour: metrics are polled, not pushed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSample:
    """One metric data point, shaped for text serialization.

    ``kind`` is ``gauge`` (current value, can go up and down) or ``counter``
    (monotonically increasing). ``labels`` are Prometheus-style key=value pairs
    used for filtering (device, channel, etc.).
    """

    name: str
    kind: str  # "gauge" | "counter"
    value: float
    help: str = ""
    labels: tuple[tuple[str, str], ...] = ()

    def prometheus_line(self) -> str:
        """Render one exposition-format line."""
        if self.labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in self.labels)
            return f"{self.name}{{{label_str}}} {self.value}"
        return f"{self.name} {self.value}"


# ── Metric name constants ──


WRITE_FAILURES_TOTAL = "leapflow_hw_write_failures_total"
RAW_WRITES_TOTAL = "leapflow_hw_raw_writes_total"
WINDOWS_WRITTEN_TOTAL = "leapflow_hw_windows_written_total"
ROWS_PRUNED_TOTAL = "leapflow_hw_rows_pruned_total"
SAMPLES_TOTAL = "leapflow_hw_samples_total"
DROPPED_TOTAL = "leapflow_hw_dropped_total"
EVENTS_PACED_OUT_TOTAL = "leapflow_hw_events_paced_out_total"
SKIPPED_SLOTS_TOTAL = "leapflow_hw_skipped_slots_total"
OBSERVED_HZ = "leapflow_hw_observed_hz"
RATE_RATIO = "leapflow_hw_rate_ratio"
STREAM_SOURCES_ACTIVE = "leapflow_hw_stream_sources_active"
DEVICES_ADMITTED = "leapflow_hw_devices_admitted"
ALERT_POLICY_FIRED_TOTAL = "leapflow_hw_alert_policy_fired_total"


class HardwareMetricsExporter:
    """Collect hardware counters into ``MetricSample`` snapshots.

    Constructed once at startup when the feature flag is on. ``collect()`` is
    called per scrape; it reads the current counters and returns a flat list of
    samples. No state is mutated during collection.
    """

    def __init__(
        self,
        *,
        registry: Any = None,
        reading_store: Any = None,
        alert_policy: Any = None,
    ) -> None:
        self._registry = registry
        self._reading_store = reading_store
        self._alert_policy = alert_policy

    def collect(self) -> list[MetricSample]:
        """Return a snapshot of all hardware metrics."""
        samples: list[MetricSample] = []
        self._collect_store(samples)
        self._collect_streams(samples)
        self._collect_registry(samples)
        self._collect_alert_policy(samples)
        return samples

    def render_prometheus(self) -> str:
        """Return the full exposition-format text."""
        lines: list[str] = []
        seen_help: set[str] = set()
        for sample in self.collect():
            if sample.name not in seen_help and sample.help:
                lines.append(f"# HELP {sample.name} {sample.help}")
                lines.append(f"# TYPE {sample.name} {sample.kind}")
                seen_help.add(sample.name)
            lines.append(sample.prometheus_line())
        return "\n".join(lines) + "\n"

    # ── Private collectors ──

    def _collect_store(self, out: list[MetricSample]) -> None:
        store = self._reading_store
        if store is None:
            return
        try:
            out.append(MetricSample(
                name=WRITE_FAILURES_TOTAL,
                kind="counter",
                value=float(getattr(store, "write_failures", 0) or 0),
                help="Total failed reading-store write batches.",
            ))
            out.append(MetricSample(
                name=RAW_WRITES_TOTAL,
                kind="counter",
                value=float(getattr(store, "raw_writes", 0) or 0),
                help="Total raw sample writes to the reading store.",
            ))
            out.append(MetricSample(
                name=WINDOWS_WRITTEN_TOTAL,
                kind="counter",
                value=float(getattr(store, "windows_written", 0) or 0),
                help="Total downsampled windows written to instrument.duckdb.",
            ))
            out.append(MetricSample(
                name=ROWS_PRUNED_TOTAL,
                kind="counter",
                value=float(getattr(store, "rows_pruned", 0) or 0),
                help="Total history rows pruned by retention.",
            ))
        except Exception as exc:  # noqa: BLE001 - metrics must not crash the caller
            logger.debug("hardware metrics: reading store collection failed: %s", exc)

    def _collect_streams(self, out: list[MetricSample]) -> None:
        registry = self._registry
        if registry is None:
            return
        sources: Sequence[Any] = ()
        try:
            sources = getattr(registry, "stream_sources", None) or ()
            if callable(sources):
                sources = sources()
        except Exception as exc:  # noqa: BLE001
            logger.debug("hardware metrics: stream sources unavailable: %s", exc)
            return

        active_count = 0
        for source in sources:
            active_count += 1
            health = getattr(source, "health", None)
            if not isinstance(health, dict):
                continue
            labels = (
                ("device_id", str(health.get("device_id", ""))),
                ("channel_id", str(health.get("channel_id", ""))),
            )
            out.append(MetricSample(
                name=SAMPLES_TOTAL,
                kind="counter",
                value=float(health.get("samples", 0)),
                help="Total raw samples collected per channel.",
                labels=labels,
            ))
            out.append(MetricSample(
                name=DROPPED_TOTAL,
                kind="counter",
                value=float(health.get("dropped", 0)),
                help="Total samples lost from the transport sequence per channel.",
                labels=labels,
            ))
            out.append(MetricSample(
                name=EVENTS_PACED_OUT_TOTAL,
                kind="counter",
                value=float(health.get("events_paced_out", 0)),
                help="Events suppressed by the rate floor per channel.",
                labels=labels,
            ))
            out.append(MetricSample(
                name=SKIPPED_SLOTS_TOTAL,
                kind="counter",
                value=float(health.get("skipped_slots", 0)),
                help="Sampling slots skipped because the loop fell behind per channel.",
                labels=labels,
            ))
            out.append(MetricSample(
                name=OBSERVED_HZ,
                kind="gauge",
                value=float(health.get("observed_hz", 0)),
                help="Observed sampling rate per channel.",
                labels=labels,
            ))
            out.append(MetricSample(
                name=RATE_RATIO,
                kind="gauge",
                value=float(health.get("rate_ratio", 0)),
                help="Observed-to-declared sampling rate ratio per channel.",
                labels=labels,
            ))

        out.append(MetricSample(
            name=STREAM_SOURCES_ACTIVE,
            kind="gauge",
            value=float(active_count),
            help="Number of active hardware stream sources.",
        ))

    def _collect_registry(self, out: list[MetricSample]) -> None:
        registry = self._registry
        if registry is None:
            return
        try:
            contexts = getattr(registry, "contexts", None)
            if callable(contexts):
                count = len(tuple(contexts()))
            else:
                count = 0
            out.append(MetricSample(
                name=DEVICES_ADMITTED,
                kind="gauge",
                value=float(count),
                help="Number of admitted hardware devices.",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("hardware metrics: device count unavailable: %s", exc)

    def _collect_alert_policy(self, out: list[MetricSample]) -> None:
        policy = self._alert_policy
        if policy is None:
            return
        try:
            out.append(MetricSample(
                name=ALERT_POLICY_FIRED_TOTAL,
                kind="counter",
                value=float(getattr(policy, "fired_count", 0) or 0),
                help="Total alert policy actions dispatched.",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("hardware metrics: alert policy collection failed: %s", exc)


def build_exporter(
    settings: Any,
    *,
    registry: Any = None,
    reading_store: Any = None,
    alert_policy: Any = None,
) -> HardwareMetricsExporter | None:
    """Return an exporter if the feature flag is on, else None (zero overhead)."""
    enabled = bool(getattr(settings, "hardware_metrics_export_enabled", False))
    if not enabled:
        return None
    return HardwareMetricsExporter(
        registry=registry,
        reading_store=reading_store,
        alert_policy=alert_policy,
    )


__all__ = [
    "ALERT_POLICY_FIRED_TOTAL",
    "DEVICES_ADMITTED",
    "DROPPED_TOTAL",
    "EVENTS_PACED_OUT_TOTAL",
    "HardwareMetricsExporter",
    "MetricSample",
    "OBSERVED_HZ",
    "RATE_RATIO",
    "RAW_WRITES_TOTAL",
    "ROWS_PRUNED_TOTAL",
    "SAMPLES_TOTAL",
    "SKIPPED_SLOTS_TOTAL",
    "STREAM_SOURCES_ACTIVE",
    "WINDOWS_WRITTEN_TOTAL",
    "WRITE_FAILURES_TOTAL",
    "build_exporter",
]
