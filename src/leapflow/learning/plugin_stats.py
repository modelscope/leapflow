# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-plugin usage statistics accumulator.

Receives forwarded (tool_name, ok, duration_ms) from TurnUsageTracker
and maintains bounded, rolling statistics per tool. Cross-turn data survives
turn resets because PluginUsageTracker is session-scoped (or engine-scoped),
not turn-scoped.

Each sample carries the result tier the engine graded
(``engine.tools.tool_execution.result_tier``) and its failure code, so the
statistics keep apart what ``ok`` alone conflates: a call the model or tool
reshaped (``repaired``), a request the tool declined without effect
(``refused``), and a real failure (``failed``). Only real outcomes move trust:
a correct refusal is the tool working, not evidence against it.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import logging

from leapflow.learning.plugin_trust import PluginTrustLedger

logger = logging.getLogger(__name__)

# Result tiers, as graded by ``engine.tools.tool_execution.result_tier``.
TIER_SUCCEEDED = "succeeded"
TIER_REPAIRED = "repaired"
TIER_REFUSED = "refused"
TIER_FAILED = "failed"
TIER_SKIPPED = "skipped"
_TIERS = frozenset({TIER_SUCCEEDED, TIER_REPAIRED, TIER_REFUSED, TIER_FAILED, TIER_SKIPPED})
# How many distinct failure codes a report names per tool, most frequent first.
_REPORTED_CODES = 5


def _tier(ok: bool, tier: str) -> str:
    """The sample's tier; an absent or unknown one falls back on ``ok``."""
    return tier if tier in _TIERS else (TIER_SUCCEEDED if ok else TIER_FAILED)


def trust_verdict(ok: bool, tier: str) -> Optional[bool]:
    """Whether one outcome is evidence for (True) or against (False) the tool, or neither.

    A refusal that changed nothing, a call rejected so the model could fix its
    arguments, and a suppressed duplicate say nothing about whether the tool
    works; charging them as failures demoted tools for declining bad requests.
    """
    graded = _tier(ok, tier)
    if graded == TIER_FAILED:
        return False
    if graded in {TIER_REFUSED, TIER_SKIPPED}:
        return None
    return True if ok else None


@dataclass(frozen=True, slots=True)
class PluginUsageSample:
    """Single recorded tool execution sample."""

    timestamp: float
    ok: bool
    duration_ms: float
    tier: str = ""
    failure_code: str = ""


@dataclass
class PluginStats:
    """Aggregated stats for a single plugin (or tool).

    ``failures`` and ``error_rate`` count real failures only; ``refusals`` and
    ``repairs`` are reported beside them so a tool that declines many requests
    does not read as broken, and a model that keeps misshaping calls is visible.
    """

    total_calls: int
    successes: int
    failures: int
    avg_duration_ms: float
    error_rate: float
    p95_duration_ms: float
    refusals: int = 0
    repairs: int = 0
    refusal_rate: float = 0.0
    repair_rate: float = 0.0
    failure_codes: Dict[str, int] = field(default_factory=dict)


def summarize_samples(samples: Iterable[PluginUsageSample]) -> Optional[PluginStats]:
    """Aggregate samples into ``PluginStats``; ``None`` when there are none."""
    rows = list(samples)
    if not rows:
        return None
    total = len(rows)
    tiers = Counter(_tier(s.ok, s.tier) for s in rows)
    failures = tiers[TIER_FAILED]
    durations = sorted(s.duration_ms for s in rows)
    p95_idx = min(int(total * 0.95), total - 1)
    codes = Counter(
        s.failure_code for s in rows if s.failure_code and _tier(s.ok, s.tier) != TIER_SUCCEEDED
    )
    return PluginStats(
        total_calls=total,
        successes=sum(1 for s in rows if s.ok),
        failures=failures,
        avg_duration_ms=round(sum(durations) / total, 2),
        error_rate=round(failures / total, 4),
        p95_duration_ms=round(durations[p95_idx], 2),
        refusals=tiers[TIER_REFUSED],
        repairs=tiers[TIER_REPAIRED],
        refusal_rate=round(tiers[TIER_REFUSED] / total, 4),
        repair_rate=round(tiers[TIER_REPAIRED] / total, 4),
        failure_codes=dict(codes.most_common(_REPORTED_CODES)),
    )


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

    def record(
        self,
        tool_name: str,
        ok: bool,
        duration_ms: float,
        *,
        tier: str = "",
        failure_code: str = "",
    ) -> None:
        """Called by TurnUsageTracker forward. Must be fast (<1μs hot path)."""
        graded = _tier(ok, tier)
        sample = PluginUsageSample(time.time(), ok, duration_ms, graded, str(failure_code or ""))
        self._samples[tool_name].append(sample)
        verdict = trust_verdict(ok, graded)
        # Forward to trust ledger: only real outcomes are evidence about the tool.
        if self._trust_ledger is not None and verdict is not None:
            plugin_id = self._resolve_plugin_id(tool_name)
            if plugin_id:
                if verdict:
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
                        tracker.record(plugin_id, tool_name, verdict)
                    except Exception:  # noqa: BLE001 - never fail a tool call on it
                        logger.debug("quarantine streak not recorded", exc_info=True)

    def stats_for_plugin(self, plugin_id: str) -> Optional[PluginStats]:
        """Aggregate stats across all tools owned by a plugin."""
        tool_names = self._tools_for_plugin(plugin_id)
        if not tool_names:
            return None
        return summarize_samples(
            sample
            for tool_name in tool_names
            if tool_name in self._samples
            for sample in self._samples[tool_name]
        )

    def stats_for_tool(self, tool_name: str) -> Optional[PluginStats]:
        """Aggregate stats for one tool."""
        samples = self._samples.get(tool_name)
        return summarize_samples(samples) if samples else None

    def tool_report(self, plugin_id: str) -> List[Dict[str, Any]]:
        """Per-tool outcome breakdown for a plugin's tools that have samples.

        Self-repair, refusal and failure rates are reported apart, with the
        most frequent failure codes, so whether a contract fix converged is read
        from data rather than from an impression that a tool "still errors".
        """
        report: List[Dict[str, Any]] = []
        for tool_name in sorted(self._tools_for_plugin(plugin_id)):
            stats = self.stats_for_tool(tool_name)
            if stats is None:
                continue
            report.append(
                {
                    "tool": tool_name,
                    "calls": stats.total_calls,
                    "failure_rate": stats.error_rate,
                    "refusal_rate": stats.refusal_rate,
                    "repair_rate": stats.repair_rate,
                    "failures": stats.failures,
                    "refusals": stats.refusals,
                    "repairs": stats.repairs,
                    "failure_codes": dict(stats.failure_codes),
                    "p95_duration_ms": stats.p95_duration_ms,
                }
            )
        return report

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
                    [round(s.timestamp, 3), s.ok, round(s.duration_ms, 2), s.tier, s.failure_code]
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
                    timestamp, ok, duration_ms, *graded = row
                    tier = str(graded[0]) if graded else ""
                    code = str(graded[1]) if len(graded) > 1 else ""
                    tracker._samples[str(tool_name)].append(
                        PluginUsageSample(
                            float(timestamp), bool(ok), float(duration_ms), _tier(bool(ok), tier), code
                        )
                    )
                except (TypeError, ValueError):
                    continue
        return tracker
