"""3-tier causal inference engine.

Tier 1: Rule-based (deterministic, zero cost, confidence ≥ 0.9)
Tier 2: Heuristic (probabilistic scoring, low cost, confidence 0.5-0.9)
Tier 3: VLM verification (high fidelity, async, for low-confidence edges)

Online learning via EMA adapts Tier 2 parameters over time.
Cold start protection: first 50 inferences use defaults only.
"""

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from leapflow.causal.channel import ChannelRegistry
from leapflow.causal.types import CausalEvent, CausalGraph, EventType
from leapflow.utils.diagnostics import PipelineTracer

logger = logging.getLogger(__name__)

# Default location for rule definitions co-located with this module.
_DEFAULT_RULES_PATH = Path(__file__).parent / "rules.yaml"

# Tolerance for "is value effectively zero / has it changed?" comparisons on
# floating-point reliability/confidence values. Threshold checks (``> 0.5``,
# ``< 3.0``) intentionally do NOT use this epsilon — they encode real
# semantic cutoffs rather than equality tests.
_FLOAT_EPSILON: float = 1e-9


# ── Tier 1: Rule-based inference ──


@dataclass(frozen=True)
class CausalRule:
    """A deterministic causal rule: parent pattern → child pattern → edge."""

    name: str
    parent_channel: str
    parent_type: Optional[EventType] = None
    parent_payload_match: Dict[str, Any] = field(default_factory=dict)
    child_channel: str = ""
    child_type: Optional[EventType] = None
    time_delta_max: float = 0.5
    spatial_distance_max: Optional[float] = None
    confidence: float = 0.95


def _coerce_event_type(value: Any) -> Optional[EventType]:
    """Coerce a YAML scalar to an EventType enum (case-insensitive)."""
    if value is None or value == "":
        return None
    if isinstance(value, EventType):
        return value
    try:
        return EventType[str(value).upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown EventType: {value!r}") from exc


def load_rules_from_yaml(path: Path) -> List[CausalRule]:
    """Load CausalRule definitions from a YAML file.

    The YAML schema mirrors the CausalRule dataclass fields. See
    ``src/leapflow/causal/rules.yaml`` for the canonical example and field
    documentation.

    Parameters
    ----------
    path:
        Filesystem path to a YAML document with a top-level ``rules`` list.

    Returns
    -------
    list[CausalRule]
        Parsed rules in declaration order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the document is malformed (missing required fields, unknown
        EventType, etc.).
    """
    with open(path, "r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    raw_rules = document.get("rules", []) or []
    if not isinstance(raw_rules, list):
        raise ValueError(f"{path}: top-level 'rules' must be a list")

    rules: List[CausalRule] = []
    for idx, entry in enumerate(raw_rules):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: rule #{idx} must be a mapping")
        try:
            rules.append(CausalRule(
                name=str(entry["name"]),
                parent_channel=str(entry["parent_channel"]),
                parent_type=_coerce_event_type(entry.get("parent_type")),
                parent_payload_match=dict(entry.get("parent_payload_match") or {}),
                child_channel=str(entry.get("child_channel", "") or ""),
                child_type=_coerce_event_type(entry.get("child_type")),
                time_delta_max=float(entry.get("time_delta_max", 0.5)),
                spatial_distance_max=(
                    float(entry["spatial_distance_max"])
                    if entry.get("spatial_distance_max") is not None
                    else None
                ),
                confidence=float(entry.get("confidence", 0.95)),
            ))
        except KeyError as exc:
            raise ValueError(
                f"{path}: rule #{idx} missing required field {exc!s}"
            ) from exc
    return rules


def _load_default_rules() -> List[CausalRule]:
    """Load rules from the bundled rules.yaml; fall back to legacy hardcoded list."""
    if _DEFAULT_RULES_PATH.exists():
        try:
            return load_rules_from_yaml(_DEFAULT_RULES_PATH)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to load default rules from %s: %s. Falling back to legacy DEFAULT_RULES.",
                _DEFAULT_RULES_PATH, exc,
            )
    return list(_LEGACY_DEFAULT_RULES)


# Legacy hardcoded rules — retained as a fallback for environments where the
# bundled rules.yaml is unavailable. Prefer editing rules.yaml; this list is
# **deprecated** and may be removed in a future release.
_LEGACY_DEFAULT_RULES: List[CausalRule] = [
    CausalRule(
        name="click_to_visual",
        parent_channel="click",
        parent_type=EventType.TRIGGER,
        child_channel="visual_change",
        child_type=EventType.RESPONSE,
        time_delta_max=0.5,
        spatial_distance_max=200.0,
        confidence=0.95,
    ),
    CausalRule(
        name="keyboard_to_visual",
        parent_channel="keyboard",
        parent_type=EventType.TRIGGER,
        child_channel="visual_change",
        child_type=EventType.RESPONSE,
        time_delta_max=0.5,
        confidence=0.90,
    ),
    CausalRule(
        name="app_switch_to_visual",
        parent_channel="app_switch",
        child_channel="visual_change",
        time_delta_max=0.3,
        confidence=0.90,
    ),
    CausalRule(
        name="cmd_c_to_clipboard",
        parent_channel="keyboard",
        parent_payload_match={"combo": "Cmd+C"},
        child_channel="clipboard",
        child_type=EventType.EFFECT,
        time_delta_max=0.2,
        confidence=0.99,
    ),
    CausalRule(
        name="cmd_v_to_visual",
        parent_channel="keyboard",
        parent_payload_match={"combo": "Cmd+V"},
        child_channel="visual_change",
        child_type=EventType.RESPONSE,
        time_delta_max=0.5,
        confidence=0.95,
    ),
    CausalRule(
        name="cmd_tab_to_app_switch",
        parent_channel="keyboard",
        parent_payload_match={"combo": "Cmd+Tab"},
        child_channel="app_switch",
        time_delta_max=0.3,
        confidence=0.95,
    ),
    CausalRule(
        name="drag_to_visual",
        parent_channel="drag",
        parent_type=EventType.TRIGGER,
        child_channel="visual_change",
        child_type=EventType.RESPONSE,
        time_delta_max=0.5,
        confidence=0.90,
    ),
]


def __getattr__(name: str) -> Any:
    """Lazy ``DEFAULT_RULES`` accessor that emits a DeprecationWarning.

    The canonical source of default rules is ``rules.yaml``; importers should
    rely on :class:`CausalInferenceEngine` (which loads the YAML automatically)
    or call :func:`load_rules_from_yaml` directly.
    """
    if name == "DEFAULT_RULES":
        warnings.warn(
            "DEFAULT_RULES is deprecated; load rules from rules.yaml via "
            "load_rules_from_yaml() or rely on CausalInferenceEngine defaults.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _load_default_rules()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class RuleEngine:
    """Tier 1: deterministic single-pass rule matching."""

    __slots__ = ("_rules",)

    def __init__(self, rules: Optional[List[CausalRule]] = None) -> None:
        self._rules = rules if rules is not None else _load_default_rules()

    def infer(self, events: List[CausalEvent], graph: CausalGraph) -> int:
        """Apply rules to establish edges. Returns number of edges added."""
        edges_added = 0
        for i, parent in enumerate(events):
            # Look forward within max window
            for j in range(i + 1, len(events)):
                child = events[j]
                dt = child.timestamp - parent.timestamp
                if dt > 3.0:
                    break
                rule = self._match_rule(parent, child, dt)
                if rule and child.caused_by is None:
                    graph.add_edge(parent.id, child.id)
                    child.confidence = max(child.confidence, min(1.0, rule.confidence * parent.confidence))
                    child.tags["inferred_by"] = "rule"
                    child.tags["rule_name"] = rule.name
                    edges_added += 1
        return edges_added

    def _match_rule(self, parent: CausalEvent, child: CausalEvent, dt: float) -> Optional[CausalRule]:
        for rule in self._rules:
            if not self._matches(parent, child, dt, rule):
                continue
            return rule
        return None

    def _matches(self, parent: CausalEvent, child: CausalEvent, dt: float, rule: CausalRule) -> bool:
        if parent.channel != rule.parent_channel:
            return False
        if rule.child_channel and child.channel != rule.child_channel:
            return False
        if rule.parent_type and parent.event_type != rule.parent_type:
            return False
        if rule.child_type and child.event_type != rule.child_type:
            return False
        if dt > rule.time_delta_max:
            return False
        # Payload match
        for key, val in rule.parent_payload_match.items():
            if parent.payload.get(key) != val:
                return False
        # Spatial distance check
        if rule.spatial_distance_max is not None:
            if "x" in parent.payload and "x" in child.payload:
                d = math.hypot(
                    child.payload["x"] - parent.payload["x"],
                    child.payload["y"] - parent.payload["y"],
                )
                if d > rule.spatial_distance_max:
                    return False
        return True


# ── Tier 2: Heuristic inference ──


def _build_semantic_prior_from_registry(registry: ChannelRegistry) -> Dict[Tuple[str, str], float]:
    """Extract semantic prior table from ChannelRegistry — single source of truth."""
    prior: Dict[Tuple[str, str], float] = {}
    for parent_channel in registry.channels:
        spec = registry.get_spec(parent_channel)
        if spec:
            for child_channel, weight in spec.semantic_prior.items():
                prior[(parent_channel, child_channel)] = weight
    return prior


class HeuristicEngine:
    """Tier 2: probabilistic causal scoring for uncovered pairs."""

    __slots__ = ("_registry", "_semantic_prior", "_confidence_threshold",
                 "_time_decay_s", "_space_decay_px")

    def __init__(
        self,
        registry: ChannelRegistry,
        semantic_prior: Optional[Dict[Tuple[str, str], float]] = None,
        confidence_threshold: float = 0.5,
        time_decay_s: Optional[float] = None,
        space_decay_px: Optional[float] = None,
    ) -> None:
        self._registry = registry
        self._semantic_prior = semantic_prior if semantic_prior is not None else _build_semantic_prior_from_registry(registry)
        self._confidence_threshold = confidence_threshold
        if time_decay_s is None or space_decay_px is None:
            from leapflow.config import get_settings
            settings = get_settings()
            if time_decay_s is None:
                time_decay_s = settings.heuristic_time_decay_s
            if space_decay_px is None:
                space_decay_px = settings.heuristic_space_decay_px
        self._time_decay_s = time_decay_s
        self._space_decay_px = space_decay_px

    def infer(self, events: List[CausalEvent], graph: CausalGraph) -> int:
        """Score uncovered event pairs and add edges above threshold."""
        edges_added = 0
        for i, parent in enumerate(events):
            if parent.event_type not in (EventType.TRIGGER, EventType.BOUNDARY):
                continue
            for j in range(i + 1, min(i + 15, len(events))):
                child = events[j]
                dt = child.timestamp - parent.timestamp
                if dt <= 0 or dt > 3.0:
                    continue
                if child.caused_by is not None:
                    continue

                score = self.causal_score(parent, child)
                if score >= self._confidence_threshold:
                    graph.add_edge(parent.id, child.id)
                    child.confidence = min(1.0, score * parent.confidence)
                    child.tags["inferred_by"] = "heuristic"
                    edges_added += 1
        return edges_added

    def causal_score(self, parent: CausalEvent, child: CausalEvent) -> float:
        dt = child.timestamp - parent.timestamp
        if dt <= 0 or dt > 3.0:
            return 0.0

        time_score = math.exp(-dt / self._time_decay_s)

        space_score = 1.0
        if "x" in parent.payload and "x" in child.payload:
            d = math.hypot(
                child.payload["x"] - parent.payload["x"],
                child.payload["y"] - parent.payload["y"],
            )
            space_score = math.exp(-d / self._space_decay_px)

        sem_key = (parent.channel, child.channel)
        sem_score = self._semantic_prior.get(sem_key, 0.3)

        return time_score * space_score * sem_score

    def update_prior(self, parent_channel: str, child_channel: str, observed: float, alpha: float = 0.1) -> None:
        """EMA update of semantic prior from VLM feedback."""
        key = (parent_channel, child_channel)
        current = self._semantic_prior.get(key, 0.3)
        updated = alpha * observed + (1 - alpha) * current
        self._semantic_prior[key] = OnlineLearningController.clamp(updated, "role_probability")


# ── Tier 3: VLM verification interface ──


@dataclass
class PendingVerification:
    """An edge awaiting VLM verification."""

    parent_id: str
    child_id: str
    confidence: float
    parent_channel: str
    child_channel: str


class VLMVerifier:
    """Tier 3: collect low-confidence edges for async VLM batch verification.

    This class collects pending edges; actual VLM calls are delegated to
    the external VLM provider to keep this module dependency-free.
    """

    __slots__ = ("_pending", "_confidence_threshold", "_batch_size")

    def __init__(self, confidence_threshold: float = 0.7, batch_size: int = 10) -> None:
        self._pending: List[PendingVerification] = []
        self._confidence_threshold = confidence_threshold
        self._batch_size = batch_size

    def collect_pending(self, graph: CausalGraph) -> List[PendingVerification]:
        """Scan graph for edges needing verification."""
        self._pending.clear()
        for parent_id, child_id, confidence in graph.iter_edges():
            if confidence < self._confidence_threshold:
                parent = graph.events.get(parent_id)
                child = graph.events.get(child_id)
                if parent and child:
                    self._pending.append(PendingVerification(
                        parent_id=parent_id,
                        child_id=child_id,
                        confidence=confidence,
                        parent_channel=parent.channel,
                        child_channel=child.channel,
                    ))
        return self._pending

    def get_batches(self) -> List[List[PendingVerification]]:
        """Split pending into batches for VLM calls."""
        batches: List[List[PendingVerification]] = []
        for i in range(0, len(self._pending), self._batch_size):
            batches.append(self._pending[i:i + self._batch_size])
        return batches

    def apply_results(
        self,
        graph: CausalGraph,
        results: List[Tuple[str, str, bool, float]],
        heuristic: Optional[HeuristicEngine] = None,
    ) -> None:
        """Apply VLM verification results to graph edges."""
        for parent_id, child_id, is_causal, confidence in results:
            if not is_causal:
                graph.remove_edge(parent_id, child_id)
            else:
                graph.update_edge_confidence(parent_id, child_id, confidence)
                child = graph.events.get(child_id)
                if child:
                    child.tags["inferred_by"] = "vlm"
                # Update heuristic prior from VLM feedback
                if heuristic:
                    parent = graph.events.get(parent_id)
                    if parent and child:
                        heuristic.update_prior(
                            parent.channel, child.channel,
                            confidence if is_causal else 0.0,
                        )


# ── Online Learning Controller ──


class OnlineLearningController:
    """Manages cold start protection and progressive parameter adaptation.

    Protocol:
        - First N inferences (cold start): use defaults, no learning
        - After N: enable EMA updates with gradually increasing alpha
        - Periodically snapshot parameters for rollback

    Hard bounds (§5.4.4):
        - causal_window: [0.1, 3.0] seconds
        - reliability: [0.1, 1.0]
        - role_probability: [0.05, 0.95]
    """

    BOUNDS = {
        "causal_window": (0.1, 3.0),
        "reliability": (0.1, 1.0),
        "role_probability": (0.05, 0.95),
    }

    __slots__ = (
        "_inference_count", "_cold_start_threshold", "_alpha_min",
        "_alpha_max", "_snapshots", "_max_snapshots",
    )

    def __init__(
        self,
        cold_start_threshold: int = 50,
        alpha_min: float = 0.01,
        alpha_max: float = 0.1,
        max_snapshots: int = 10,
    ) -> None:
        self._inference_count = 0
        self._cold_start_threshold = cold_start_threshold
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._snapshots: List[Dict[str, Any]] = []
        self._max_snapshots = max_snapshots

    @property
    def is_cold(self) -> bool:
        return self._inference_count < self._cold_start_threshold

    @property
    def current_alpha(self) -> float:
        if self.is_cold:
            return 0.0
        progress = min(1.0, (self._inference_count - self._cold_start_threshold) / 200.0)
        return self._alpha_min + progress * (self._alpha_max - self._alpha_min)

    def record_inference(self) -> None:
        self._inference_count += 1

    def snapshot(self, params: Dict[str, Any]) -> None:
        self._snapshots.append(dict(params))
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def rollback(self) -> Optional[Dict[str, Any]]:
        if self._snapshots:
            return self._snapshots.pop()
        return None

    @classmethod
    def clamp(cls, value: float, param_type: str) -> float:
        """Enforce hard bounds for a parameter type."""
        lo, hi = cls.BOUNDS.get(param_type, (0.0, 1.0))
        return max(lo, min(hi, value))

    @property
    def inference_count(self) -> int:
        return self._inference_count


# ── Unified Inference Engine ──


class CausalInferenceEngine:
    """Unified 3-tier causal inference engine.

    Orchestrates: Tier 1 (rules) → Tier 2 (heuristics) → Tier 3 (VLM queue).
    Synchronous path (Tier 1 + 2) stays under 10ms budget.
    """

    __slots__ = ("_rules", "_heuristic", "_verifier", "_learning")

    def __init__(
        self,
        registry: ChannelRegistry,
        rules: Optional[List[CausalRule]] = None,
        semantic_prior: Optional[Dict[Tuple[str, str], float]] = None,
        rules_path: Optional[Path] = None,
    ) -> None:
        """Construct the engine.

        Parameters
        ----------
        registry:
            Channel registry providing semantic priors for Tier 2 heuristics.
        rules:
            Pre-built CausalRule list. Takes precedence over ``rules_path``.
        semantic_prior:
            Optional override for the heuristic semantic prior table.
        rules_path:
            Optional path to a YAML rules file. When omitted (and ``rules``
            is also omitted), the bundled ``rules.yaml`` is used.
        """
        if rules is None and rules_path is not None:
            rules = self._load_rules(rules_path)
        self._rules = RuleEngine(rules)
        self._heuristic = HeuristicEngine(registry, semantic_prior)
        self._verifier = VLMVerifier()
        self._learning = OnlineLearningController()

    @staticmethod
    def _load_rules(path: Path) -> List[CausalRule]:
        """Load rules from a YAML file (delegates to :func:`load_rules_from_yaml`)."""
        return load_rules_from_yaml(path)

    def infer_sync(self, events: List[CausalEvent], graph: CausalGraph) -> Dict[str, int]:
        """Run synchronous inference (Tier 1 + 2). Must complete in <10ms."""
        tracer = PipelineTracer(
            "causal_inference",
            enabled=logger.isEnabledFor(logging.DEBUG),
        )
        stats: Dict[str, int] = {"rule_edges": 0, "heuristic_edges": 0}
        with tracer.stage("tier1_rules"):
            stats["rule_edges"] = self._rules.infer(events, graph)
            tracer.metric("edges", stats["rule_edges"])
        with tracer.stage("tier2_heuristic"):
            stats["heuristic_edges"] = self._heuristic.infer(events, graph)
            tracer.metric("edges", stats["heuristic_edges"])
        self._learning.record_inference()
        if tracer.enabled:
            tracer.metric("edges_added", stats["rule_edges"] + stats["heuristic_edges"])
            tracer.metric("cold_start", self._learning.is_cold)
            logger.debug(tracer.summary_line())
        return stats

    def collect_for_vlm(self, graph: CausalGraph) -> List[PendingVerification]:
        """Collect edges needing Tier 3 VLM verification (async path)."""
        return self._verifier.collect_pending(graph)

    def apply_vlm_results(
        self,
        graph: CausalGraph,
        results: List[Tuple[str, str, bool, float]],
    ) -> None:
        """Apply VLM verification results and update learning."""
        alpha = self._learning.current_alpha
        # ``current_alpha`` returns exactly 0.0 during cold start and
        # ``alpha_min`` (≥ 0.01) afterwards; using an explicit epsilon makes
        # the "learning enabled?" intent unambiguous and immune to any future
        # rounding in ``current_alpha`` arithmetic.
        if alpha > _FLOAT_EPSILON:
            self._verifier.apply_results(graph, results, self._heuristic)
        else:
            self._verifier.apply_results(graph, results, None)

    @property
    def learning(self) -> OnlineLearningController:
        return self._learning

    @property
    def heuristic(self) -> HeuristicEngine:
        return self._heuristic

    @property
    def verifier(self) -> VLMVerifier:
        return self._verifier
