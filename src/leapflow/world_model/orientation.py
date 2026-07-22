"""S4-D1: multi-layer orientation aggregation (observe-only).

A unified, read-only "orientation" query that merges the agent's orientation
across three layers — **immediate** (live signals / recent world-model change),
**working** (the current task's research ledger), and **long-term** (durable
cross-session findings / retrieved memory) — applying a per-layer base salience
and time decay (immediate fades fast, long-term slowly). This realizes the OODA
"Orient" as a first-class, layered view without changing any behavior; it is
pure and hermetic, and the aggregation is a read-only projection of existing
state (S1 ledger, world model, memory).

It is the foundation for later autonomy phases (D2–D4) but is useful on its own
as a single orientation query for dashboards / diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Per-layer base salience and half-life (seconds) for time decay.
_LAYER_BASE = {"immediate": 1.0, "working": 0.7, "long_term": 0.4}
_LAYER_HALF_LIFE = {"immediate": 300.0, "working": 3600.0, "long_term": 86400.0}
_LAYERS = ("immediate", "working", "long_term")
_MAX_ITEM_CHARS = 240

# A layer entry is either plain text (assumed fresh) or ``(text, timestamp)``.
LayerEntry = Union[str, Tuple[str, float]]


@dataclass(frozen=True)
class OrientationItem:
    """A single oriented fact with its layer and decayed salience weight."""

    text: str
    layer: str
    weight: float


@dataclass(frozen=True)
class Orientation:
    """A unified, weight-ranked orientation across the three layers."""

    items: Tuple[OrientationItem, ...] = ()

    def top(self, n: int = 10) -> List[OrientationItem]:
        return list(self.items[: max(0, n)])

    def by_layer(self, layer: str) -> List[OrientationItem]:
        return [it for it in self.items if it.layer == layer]

    def render(self, *, max_items: int = 12) -> str:
        """Compact, weight-ranked text block (for a prompt tail or a report)."""
        lines = [
            f"- ({it.layer}) {it.text}"
            for it in self.items[: max(0, max_items)]
        ]
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        by_layer: Dict[str, int] = {layer: 0 for layer in _LAYERS}
        for it in self.items:
            by_layer[it.layer] = by_layer.get(it.layer, 0) + 1
        return {
            "total": len(self.items),
            "by_layer": by_layer,
            "top": [
                {"layer": it.layer, "weight": it.weight, "text": it.text[:80]}
                for it in self.items[:5]
            ],
        }


def _decayed_weight(layer: str, age_seconds: float) -> float:
    base = _LAYER_BASE.get(layer, 0.5)
    half_life = _LAYER_HALF_LIFE.get(layer, 3600.0)
    if age_seconds <= 0 or half_life <= 0:
        return base
    return base * (0.5 ** (age_seconds / half_life))


def _normalize(entry: LayerEntry, now: float) -> Tuple[str, float]:
    if isinstance(entry, tuple):
        text, ts = entry
        return str(text), float(ts)
    return str(entry), now


def aggregate_orientation(
    *,
    immediate: Sequence[LayerEntry] = (),
    working: Sequence[LayerEntry] = (),
    long_term: Sequence[LayerEntry] = (),
    now: float,
    max_items: int = 24,
) -> Orientation:
    """Merge the three orientation layers into one weight-ranked view (pure).

    Each entry is plain text (assumed current) or ``(text, timestamp)``. The
    salience weight is the layer base times a half-life time decay, so recent
    immediate signals dominate while durable long-term facts persist quietly.
    """
    items: List[OrientationItem] = []
    for layer, entries in (("immediate", immediate), ("working", working), ("long_term", long_term)):
        for entry in entries:
            text, ts = _normalize(entry, now)
            text = text.strip()
            if not text:
                continue
            age = max(0.0, now - ts)
            items.append(OrientationItem(
                text=text[:_MAX_ITEM_CHARS],
                layer=layer,
                weight=round(_decayed_weight(layer, age), 4),
            ))
    items.sort(key=lambda it: it.weight, reverse=True)
    return Orientation(items=tuple(items[: max(0, max_items)]))


def build_orientation_from_ledger(
    ledger_state: Optional[Dict[str, Any]],
    *,
    now: float,
    immediate: Sequence[LayerEntry] = (),
    long_term: Sequence[LayerEntry] = (),
    max_items: int = 24,
) -> Orientation:
    """Aggregate an orientation from a research-ledger snapshot (working layer).

    Maps ``ResearchLedger.to_state()`` (findings / open_questions / next_step)
    into the working layer, optionally combined with caller-supplied immediate
    and long-term entries. Read-only; the ledger snapshot is not mutated.
    """
    state = ledger_state or {}
    working: List[LayerEntry] = []
    for finding in (state.get("findings") or []):
        working.append((str(finding), now))
    for question in (state.get("open_questions") or []):
        working.append((f"[open] {question}", now))
    next_step = str(state.get("next_step") or "").strip()
    if next_step:
        working.append((f"[next] {next_step}", now))
    return aggregate_orientation(
        immediate=immediate,
        working=working,
        long_term=long_term,
        now=now,
        max_items=max_items,
    )
