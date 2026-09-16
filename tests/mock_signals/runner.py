# Copyright (c) Alibaba, Inc. and its affiliates.
"""Orchestrator: build pipeline, inject signals, collect metrics, report results.

The runner constructs a minimal in-memory LeapFlow pipeline (EventBus + MonitorManager)
and drives generated signals through it, measuring throughput, latency, debounce
behavior, and finding production.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.monitor.manager import MonitorManager
from leapflow.monitor.producers import ProducerRegistry
from leapflow.monitor.types import (
    Finding,
    ProducerContext,
    Severity,
    WatchSpec,
)
from leapflow.platform.event_bus import EventBus
from leapflow.storage.connection import LocalConnectionHolder

from tests.mock_signals.generators import GENERATOR_REGISTRY, BaseGenerator, SignalConfig
from tests.mock_signals.profiles import PROFILES


# ─── Passthrough producer ────────────────────────────────────────────────


class _PassthroughProducer:
    """Trivial producer that always emits a finding when triggered.

    Used to verify that the event-driven watch pipeline fires end-to-end.
    """

    def __init__(self, domain: str) -> None:
        self._domain = domain
        self.call_count = 0

    @property
    def domain(self) -> str:
        return self._domain

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        self.call_count += 1
        return [
            Finding(
                watch_id=ctx.spec.watch_id,
                domain=self._domain,
                title=f"Signal observed ({self._domain})",
                summary=f"Run #{self.call_count}",
                severity=Severity.INFO,
            )
        ]


# ─── Result dataclass ────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Aggregated results from a mock signal injection run."""

    profile: str
    daemon_mode: bool = False
    duration_s: float = 0.0
    events_injected: int = 0
    events_per_type: Dict[str, int] = field(default_factory=dict)
    triggers_fired: int = 0
    debounced_count: int = 0
    findings_produced: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted human-readable summary."""
        if self.daemon_mode:
            return self._daemon_summary()
        return self._local_summary()

    def _daemon_summary(self) -> str:
        """Summary format for daemon injection mode."""
        lines: List[str] = []
        sep = "\u2550" * 55
        lines.append(f"\n{sep}")
        lines.append(f"  LeapFlow Mock Signal Injection \u2014 {self.profile}")
        lines.append("  Mode: DAEMON (leapd RPC)")
        lines.append(sep)
        lines.append("")
        lines.append(f"  Duration:        {self.duration_s:.2f}s")
        lines.append(f"  Events injected: {self.events_injected}")
        lines.append("  Events by type:")
        total = max(self.events_injected, 1)
        for etype, count in sorted(self.events_per_type.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"    {etype:<20} {count:>5} ({pct:.1f}%)")
        lines.append("")
        if self.duration_s > 0:
            rate = self.events_injected / self.duration_s
            lines.append(f"  Injection rate:  {rate:.1f} events/s")
        lines.append(f"  Errors: {len(self.errors)}")
        if self.errors:
            for err in self.errors[:5]:
                lines.append(f"    - {err}")
        lines.append("")
        status = "\u2713 DONE (check TUI/LeapBoard for results)" if not self.errors else "\u2717 FAIL"
        lines.append(f"  Status: {status}")
        lines.append(sep)
        return "\n".join(lines)

    def _local_summary(self) -> str:
        """Summary format for local in-process mode."""
        lines: List[str] = []
        sep = "\u2550" * 55
        lines.append(f"\n{sep}")
        lines.append(f"  LeapFlow Mock Signal Injection \u2014 Profile: {self.profile}")
        lines.append(sep)
        lines.append("")
        lines.append(f"  Duration:          {self.duration_s:.2f}s")
        lines.append(f"  Events injected:   {self.events_injected}")
        lines.append("  Events by type:")
        total = max(self.events_injected, 1)
        for etype, count in sorted(self.events_per_type.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"    {etype:<20} {count:>5} ({pct:.1f}%)")
        lines.append("")
        lines.append("  Monitor results:")
        lines.append(f"    Triggers fired:  {self.triggers_fired:>5}")
        lines.append(f"    Debounced:       {self.debounced_count:>5}")
        lines.append(f"    Findings:        {self.findings_produced:>5}")
        lines.append("")
        lines.append("  Latency (event \u2192 trigger):")
        lines.append(f"    Avg:             {self.avg_latency_ms:.1f}ms")
        lines.append(f"    P99:             {self.p99_latency_ms:.1f}ms")
        lines.append(f"    Max:             {self.max_latency_ms:.1f}ms")
        lines.append("")
        lines.append(f"  Errors: {len(self.errors)}")
        if self.errors:
            for err in self.errors[:5]:
                lines.append(f"    - {err}")
        lines.append("")
        status = "\u2713 PASS" if not self.errors else "\u2717 FAIL"
        lines.append(f"  Status: {status}")
        lines.append(sep)
        return "\n".join(lines)


# ─── Runner ──────────────────────────────────────────────────────────────


class MockSignalRunner:
    """Orchestrates signal injection and validates end-to-end pipeline."""

    def __init__(
        self,
        profile_name: str = "normal",
        *,
        daemon_mode: bool = False,
        tmp_dir: Optional[Path] = None,
        duration_override: Optional[float] = None,
        freq_multiplier: Optional[float] = None,
    ) -> None:
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown profile: {profile_name!r}. "
                f"Available: {', '.join(sorted(PROFILES))}"
            )
        self._profile_name = profile_name
        self._profile = PROFILES[profile_name]
        self._daemon_mode = daemon_mode
        self._tmp_dir = tmp_dir
        self._duration_override = duration_override
        self._freq_multiplier = freq_multiplier

    async def run(self) -> RunResult:
        """Execute the full scenario and return results."""
        if self._daemon_mode:
            return await self._run_daemon_mode()
        return await self._run_local_mode()

    async def _run_daemon_mode(self) -> RunResult:
        """Inject signals into running leapd via RPC.

        Generators run concurrently (matching local mode) so that
        different signal types overlap realistically.  Inter-event
        waits use the profile-intended timing so the daemon's
        EventBridge debounce windows work as designed.
        """
        from leapflow.config import _bootstrap_profile_layout
        from leapflow.daemon.client import DaemonClient, DaemonUnavailableError

        result = RunResult(profile=self._profile_name, daemon_mode=True)

        # Resolve socket path from profile layout
        try:
            layout = _bootstrap_profile_layout()
            sock_path = layout.runtime_dir / "leapd.sock"
        except Exception as e:
            result.errors.append(f"Cannot resolve daemon socket path: {e}")
            return result

        client = DaemonClient(sock_path, timeout_s=10.0)

        # Verify daemon connectivity
        try:
            await client.status()
        except DaemonUnavailableError as e:
            result.errors.append(f"Cannot connect to leapd: {e}")
            return result
        except Exception as e:
            result.errors.append(f"Cannot connect to leapd: {e}")
            return result

        # Build generators (same as local mode)
        generators = self._build_generators()
        t0 = time.monotonic()
        _lock = asyncio.Lock()
        injected = 0
        errors: List[str] = []
        events_per_type: Dict[str, int] = {}
        _progress_count = 0

        async def _inject_from(gen: BaseGenerator) -> None:
            """Inject events from a single generator concurrently."""
            nonlocal injected, _progress_count
            for event_type, payload in gen.generate():
                if event_type == "__wait__":
                    wait_s = payload.get("seconds", 0.01)
                    await asyncio.sleep(wait_s)
                    continue
                try:
                    await client.signal_record(event_type, payload)
                except Exception as exc:
                    async with _lock:
                        errors.append(f"{event_type}: {exc}")
                    continue
                async with _lock:
                    injected += 1
                    _progress_count += 1
                    display_type = self._display_event_type(event_type)
                    events_per_type[display_type] = events_per_type.get(display_type, 0) + 1
                    if _progress_count % 20 == 0:
                        elapsed = time.monotonic() - t0
                        print(f"    \r  injected {_progress_count} events ({elapsed:.1f}s)", end="", flush=True)

        await asyncio.gather(*[_inject_from(gen) for gen in generators])

        # Clear progress line
        if _progress_count >= 20:
            print()

        result.duration_s = time.monotonic() - t0
        result.events_injected = injected
        result.events_per_type = events_per_type
        result.errors = errors
        return result

    async def _run_local_mode(self) -> RunResult:
        """Run in-process pipeline (original behavior)."""
        tmp = self._tmp_dir or Path(tempfile.mkdtemp(prefix="leapflow-mock-"))
        db_path = tmp / "mock.duckdb"

        # 1. Build infrastructure
        episodic = EpisodicMemoryProvider(ttl=300.0)
        working = WorkingMemoryProvider(max_tokens=2048)
        event_bus = EventBus(immediate=episodic, working=working)

        holder = LocalConnectionHolder(db_path)
        producers = ProducerRegistry()
        monitor = MonitorManager(holder=holder, producers=producers, tick_seconds=1)

        # 2. Wire EventBus -> EventBridge
        event_bus.subscribe(monitor.event_bridge.on_event)

        # 3. Register producers and arm watches for each signal domain
        domain_patterns = self._resolve_domain_patterns()
        producer_instances: Dict[str, _PassthroughProducer] = {}
        watch_ids: List[str] = []

        await monitor.start()

        for domain, pattern in domain_patterns.items():
            producer = _PassthroughProducer(domain)
            producers.register(producer)
            producer_instances[domain] = producer

            spec = WatchSpec(
                name=f"mock-{domain}",
                domain=domain,
                trigger_expr=f"event:{pattern}",
                sensitivity="info",
            )
            view = await monitor.arm_watch(spec)
            watch_ids.append(view.watch_id)

        # 4. Build generators
        generators = self._build_generators()

        # 5. Inject events concurrently from all generators
        result = RunResult(profile=self._profile_name)
        latencies: List[float] = []
        _lock = asyncio.Lock()
        t0 = time.monotonic()

        async def _inject_from(gen: BaseGenerator) -> None:
            """Inject events from a single generator."""
            for event_type, payload in gen.generate():
                if event_type == "__wait__":
                    wait_s = payload.get("seconds", 0.01)
                    await asyncio.sleep(min(wait_s, 0.05))
                    continue

                inject_ts = time.monotonic()
                try:
                    await event_bus.handle_event(event_type, payload)
                    latency_ms = (time.monotonic() - inject_ts) * 1000.0
                except Exception as exc:
                    async with _lock:
                        result.errors.append(f"{event_type}: {exc}")
                    continue

                async with _lock:
                    latencies.append(latency_ms)
                    result.events_injected += 1
                    display_type = self._display_event_type(event_type)
                    result.events_per_type[display_type] = (
                        result.events_per_type.get(display_type, 0) + 1
                    )

        await asyncio.gather(*[_inject_from(gen) for gen in generators])

        # Allow final triggers/ticks to process
        await asyncio.sleep(0.1)

        # Manually run each watch once to simulate scheduler executing
        # triggered event watches (the scheduler tick creates new trigger
        # instances from serialized config and can't see in-memory state).
        for wid in watch_ids:
            try:
                await monitor.run_watch_once(wid)
            except Exception:
                pass

        result.duration_s = time.monotonic() - t0

        # 6. Collect statistics
        # triggers_fired = non-debounced events that passed through EventBridge
        total_debounced = sum(
            monitor.event_bridge.debounced_count(wid) for wid in watch_ids
        )
        total_findings = sum(
            len(monitor.list_findings(watch_id=wid)) for wid in watch_ids
        )

        result.triggers_fired = sum(p.call_count for p in producer_instances.values())
        result.debounced_count = total_debounced
        result.findings_produced = total_findings

        if latencies:
            latencies.sort()
            result.avg_latency_ms = sum(latencies) / len(latencies)
            p99_idx = int(len(latencies) * 0.99)
            result.p99_latency_ms = latencies[min(p99_idx, len(latencies) - 1)]
            result.max_latency_ms = latencies[-1]

        # 7. Cleanup
        await monitor.stop()
        await event_bus.shutdown()
        holder.close()

        return result

    def _resolve_domain_patterns(self) -> Dict[str, str]:
        """Map generator event types to monitor domain + event pattern pairs."""
        mapping: Dict[str, str] = {}
        for gen_name, _ in self._profile.generators:
            cls = GENERATOR_REGISTRY.get(gen_name)
            if cls is None:
                continue
            etype = cls.event_type
            # Derive a domain key and an fnmatch pattern
            if etype.startswith("event."):
                # e.g. "event.fs_change" -> domain "fs", pattern "fs.*"
                suffix = etype.removeprefix("event.")
                domain = suffix.split("_")[0]
                # Normalized event type from EventBus fallback
                normalized = self._normalized_event_type(etype)
                pattern = f"{normalized}*" if not normalized.endswith("*") else normalized
                mapping[domain] = pattern
            elif etype.startswith("gateway."):
                domain = "gateway"
                mapping[domain] = "gateway.*"
            else:
                domain = etype.replace(".", "_")
                mapping[domain] = f"{etype}*"
        return mapping

    def _normalized_event_type(self, raw_type: str) -> str:
        """Predict the normalized event_type that EventBus will produce."""
        type_map = {
            "event.fs_change": "fs.change",
            "event.clipboard_change": "clipboard.change",
            "event.app_focus_change": "app.focus_change",
            "event.ui_action": "ui.action",
        }
        return type_map.get(raw_type, raw_type)

    def _display_event_type(self, raw_type: str) -> str:
        """Return a compact display name for output."""
        display_map = {
            "event.fs_change": "fs.change",
            "event.clipboard_change": "clipboard.change",
            "event.app_focus_change": "app.focus_change",
            "event.ui_action": "ui.action",
            "gateway.signal": "gateway.signal",
            "gateway.message.received": "gateway.message",
        }
        return display_map.get(raw_type, raw_type)

    def _build_generators(self) -> List[BaseGenerator]:
        """Instantiate generators from profile spec with optional overrides."""
        generators: List[BaseGenerator] = []
        for gen_name, kwargs in self._profile.generators:
            cls = GENERATOR_REGISTRY.get(gen_name)
            if cls is None:
                continue
            gen_kwargs = dict(kwargs)
            config = gen_kwargs.pop("config", SignalConfig())
            # Apply overrides
            if self._duration_override is not None:
                config = SignalConfig(
                    frequency_hz=config.frequency_hz,
                    burst_size=config.burst_size,
                    burst_interval_s=config.burst_interval_s,
                    duration_s=self._duration_override,
                    jitter_ms=config.jitter_ms,
                )
            if self._freq_multiplier is not None:
                config = SignalConfig(
                    frequency_hz=config.frequency_hz * self._freq_multiplier,
                    burst_size=config.burst_size,
                    burst_interval_s=config.burst_interval_s,
                    duration_s=config.duration_s,
                    jitter_ms=config.jitter_ms,
                )
            generators.append(cls(config, **gen_kwargs))
        return generators


__all__ = ["MockSignalRunner", "RunResult"]
