"""Per-plugin usage statistics accumulator.

Receives forwarded (tool_name, ok, duration_ms) from TurnUsageTracker
and maintains bounded, rolling statistics per tool. Cross-turn data survives
turn resets because PluginUsageTracker is session-scoped (or engine-scoped),
not turn-scoped.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import logging

from leapflow.learning.plugin_trust import PluginTrustLedger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PluginUsageSample:
    """Single recorded tool execution sample."""

    timestamp: float
    ok: bool
    duration_ms: float


@dataclass
class PluginStats:
    """Aggregated stats for a single plugin."""

    total_calls: int
    successes: int
    failures: int
    avg_duration_ms: float
    error_rate: float
    p95_duration_ms: float


class PluginUsageTracker:
    """Cross-turn accumulator. Bounded memory via deque(maxlen=N)."""

    def __init__(self, max_samples_per_tool: int = 500) -> None:
        self._max_samples = max(1, int(max_samples_per_tool))
        self._samples: Dict[str, deque[PluginUsageSample]] = defaultdict(
            self._make_deque
        )
        self._trust_ledger: Optional[PluginTrustLedger] = None
        # Lazy reverse index: tool_name → plugin_id
        self._tool_to_plugin: Optional[Dict[str, str]] = None
        self._registry_version: int = -1

    def _make_deque(self) -> deque[PluginUsageSample]:
        return deque(maxlen=self._max_samples)

    def set_trust_ledger(self, ledger: PluginTrustLedger) -> None:
        """Inject the trust ledger for automatic trust forwarding."""
        self._trust_ledger = ledger

    def set_quarantine_tracker(self, tracker: Any) -> None:
        """Inject the consecutive-failure tracker that feeds cold-path quarantine.

        Trust demotion has always been immediate here; quarantine had no feed at all,
        so a plugin could fail indefinitely without being disabled. The tracker keeps
        one integer per plugin and is drained by the cold-path co-evolution sweep --
        governance work itself must never run on this path.
        """
        self._quarantine_tracker = tracker

    def record(self, tool_name: str, ok: bool, duration_ms: float) -> None:
        """Called by TurnUsageTracker forward. Must be fast (<1μs hot path)."""
        sample = PluginUsageSample(time.time(), ok, duration_ms)
        self._samples[tool_name].append(sample)
        # Forward to trust ledger
        if self._trust_ledger is not None:
            plugin_id = self._resolve_plugin_id(tool_name)
            if plugin_id:
                if ok:
                    self._trust_ledger.record_success(plugin_id)
                else:
                    self._trust_ledger.record_failure(plugin_id)
                # Streak bookkeeping only: a dict lookup and an integer update. The
                # governance it may trigger runs later, on the sweep's cold path.
                #
                # Effect verification is deliberately *not* recorded here: this path
                # receives only ``ok``, and a tool's observed effect lives in its result
                # payload. The engine's result-observation path records outcomes, so
                # doing it here as well would double-count and would grade every
                # success as unverifiable.
                tracker = getattr(self, "_quarantine_tracker", None)
                if tracker is not None:
                    try:
                        tracker.record(plugin_id, tool_name, ok)
                    except Exception:  # noqa: BLE001 - never fail a tool call on it
                        logger.debug("quarantine streak not recorded", exc_info=True)

    def stats_for_plugin(self, plugin_id: str) -> Optional[PluginStats]:
        """Aggregate stats across all tools owned by a plugin."""
        tool_names = self._tools_for_plugin(plugin_id)
        if not tool_names:
            return None

        all_samples: list[PluginUsageSample] = []
        for tool_name in tool_names:
            if tool_name in self._samples:
                all_samples.extend(self._samples[tool_name])

        if not all_samples:
            return None

        total = len(all_samples)
        successes = sum(1 for s in all_samples if s.ok)
        failures = total - successes
        durations = [s.duration_ms for s in all_samples]
        avg_duration = sum(durations) / total if total else 0.0
        error_rate = failures / total if total else 0.0

        # p95 duration
        sorted_durations = sorted(durations)
        p95_idx = min(int(total * 0.95), total - 1)
        p95_duration = sorted_durations[p95_idx] if sorted_durations else 0.0

        return PluginStats(
            total_calls=total,
            successes=successes,
            failures=failures,
            avg_duration_ms=round(avg_duration, 2),
            error_rate=round(error_rate, 4),
            p95_duration_ms=round(p95_duration, 2),
        )

    def _resolve_plugin_id(self, tool_name: str) -> Optional[str]:
        """Map tool_name → plugin_id (lazy-built reverse index)."""
        index = self._get_reverse_index()
        return index.get(tool_name)

    def _tools_for_plugin(self, plugin_id: str) -> list[str]:
        """Return tool names belonging to the given plugin."""
        index = self._get_reverse_index()
        return [name for name, pid in index.items() if pid == plugin_id]

    def _get_reverse_index(self) -> Dict[str, str]:
        """Build/cache reverse index from tool_name → plugin_id."""
        try:
            from leapflow.plugins import get_registry
            reg = get_registry()
            version = getattr(reg, "_version", 0)
            if self._tool_to_plugin is not None and self._registry_version == version:
                return self._tool_to_plugin
            # Prefer the registry's live ownership map: it reflects first-wins
            # tool-name arbitration, so usage/trust accrues to the plugin whose
            # handler actually ran. Rebuilding from ``reg.plugins`` would let a
            # rejected duplicate tool claim steal the usage history.
            owners = getattr(reg, "tool_owners", None)
            if owners:
                mapping = {str(name): str(pid) for name, pid in dict(owners).items()}
            else:
                mapping = {}
                for pid, plugin in reg.plugins.items():
                    for tool_meta in plugin.tools:
                        mapping.setdefault(tool_meta.name, pid)
            self._tool_to_plugin = mapping
            self._registry_version = version
            return mapping
        except (ImportError, RuntimeError, AttributeError):
            return self._tool_to_plugin or {}

    # ── Persistence ──

    def to_state(self) -> Dict[str, Any]:
        """Serialize recent usage samples for persistence.

        Samples are keyed by tool name rather than plugin id: the tool → plugin
        mapping is derived from the live registry, so a plugin that is renamed or
        reinstalled still inherits the reliability history of the tools it owns.
        """
        return {
            "max_samples_per_tool": self._max_samples,
            "samples": {
                tool_name: [
                    [round(s.timestamp, 3), s.ok, round(s.duration_ms, 2)]
                    for s in list(samples)[-self._max_samples:]
                ]
                for tool_name, samples in self._samples.items()
                if samples
            },
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "PluginUsageTracker":
        """Restore a tracker from serialized state; malformed rows are skipped."""
        if not state:
            return cls()
        tracker = cls(
            max_samples_per_tool=int(state.get("max_samples_per_tool") or 500)
        )
        for tool_name, rows in (state.get("samples") or {}).items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                # A truncated or hand-edited blob must not break startup; a
                # dropped sample only dilutes signal, while a raise would cost
                # the whole session its history.
                try:
                    timestamp, ok, duration_ms = row
                    tracker._samples[str(tool_name)].append(
                        PluginUsageSample(float(timestamp), bool(ok), float(duration_ms))
                    )
                except (TypeError, ValueError):
                    continue
        return tracker
