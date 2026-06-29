"""Adapters bridging CausalGraph to downstream pipeline interfaces.

Conversions:
    - graph_to_pair_context: CausalGraph → PairContext for VLM extraction
    - graph_to_semantic_actions: CausalGraph → SemanticAction list
    - chains_to_descriptions: CausalChain list → text descriptions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from leapflow.causal.channel import ChannelRegistry, build_default_registry
from leapflow.causal.types import CausalChain, CausalEvent, CausalGraph

if TYPE_CHECKING:
    from leapflow.perception.types import InteractionSignal, PairContext


_DEFAULT_REGISTRY: Optional[ChannelRegistry] = None


def _default_registry() -> ChannelRegistry:
    """Lazily build and cache the default :class:`ChannelRegistry`.

    Adapters accept an optional registry argument so callers can inject a
    custom one; when omitted we fall back to the bundled defaults so that
    short-template lookup remains consistent across the system.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY


def graph_to_pair_context(
    graph: CausalGraph,
    t0: float,
    t1: float,
    app: str = "",
) -> "PairContext":
    """Convert a CausalGraph time window into a PairContext for VLM prompts.

    This adapter enables the existing ContextEnrichedVLMExtractor to consume
    causal chain data without modification.
    """
    from leapflow.perception.types import InteractionSignal, PairContext

    chains = graph.chains_in_window(t0, t1)
    signals: List[InteractionSignal] = []

    for chain in chains:
        for ev in chain.all_events():
            if ev.timestamp < t0 or ev.timestamp > t1:
                continue
            sig = _event_to_signal(ev)
            if sig:
                signals.append(sig)

    signals.sort(key=lambda s: s.timestamp)
    # Cap at 8 signals per pair (existing pipeline limit)
    signals = signals[:8]

    return PairContext(
        time_delta=t1 - t0,
        app_b=app,
        signals=signals,
    )


def chains_to_descriptions(
    chains: List[CausalChain],
    registry: Optional[ChannelRegistry] = None,
) -> List[str]:
    """Convert chains to compact text descriptions for VLM prompts.

    Per-channel formatting is read from each :class:`~leapflow.causal.channel.ChannelSpec`'s
    ``short_template`` field via the registry, keeping channel additions purely
    declarative (no adapter changes needed for new channels).
    """
    reg = registry or _default_registry()
    descriptions: List[str] = []
    for chain in chains:
        parts: List[str] = []
        parts.append(_describe_event_short(chain.trigger, reg))
        for resp in chain.responses[:2]:
            parts.append(_describe_event_short(resp, reg))
        for eff in chain.effects[:1]:
            parts.append(_describe_event_short(eff, reg))
        descriptions.append(" → ".join(parts))
    return descriptions


def graph_to_semantic_actions(
    graph: CausalGraph,
    registry: Optional[ChannelRegistry] = None,
) -> List[Dict[str, Any]]:
    """Convert graph chains to a list of semantic action dicts.

    Compatible with the domain layer's SemanticAction format.
    """
    reg = registry or _default_registry()
    actions: List[Dict[str, Any]] = []
    for chain in graph.chains:
        trigger = chain.trigger
        action_name = _infer_action_name(trigger)
        params: Dict[str, Any] = {
            "_source": "causal_propagation",
            "_confidence": trigger.confidence,
            "_chain_id": chain.id,
            "_completeness": chain.completeness,
        }
        if "x" in trigger.payload:
            params["x"] = trigger.payload["x"]
            params["y"] = trigger.payload.get("y", 0)
        if "combo" in trigger.payload:
            params["shortcut"] = trigger.payload["combo"]

        actions.append({
            "action_name": action_name,
            "description": chain.semantic_label or _describe_event_short(trigger, reg),
            "parameters": params,
            "confidence": trigger.confidence * chain.completeness,
            "timestamp": trigger.timestamp,
        })
    return actions


# ── Internal helpers ──


def _event_to_signal(ev: CausalEvent) -> Optional["InteractionSignal"]:
    from leapflow.perception.types import InteractionSignal

    position = None
    if "x" in ev.payload and "y" in ev.payload:
        position = (ev.payload["x"], ev.payload["y"])

    detail = ev.payload.get("combo") or ev.payload.get("detail") or ""
    if not detail and "from" in ev.payload and "to" in ev.payload:
        detail = f"{ev.payload['from']}→{ev.payload['to']}"

    return InteractionSignal(
        timestamp=ev.timestamp,
        signal_type=ev.channel,
        position=position,
        detail=str(detail) if detail else None,
    )


_SHORT_TEMPLATES: Dict[str, str] = {
    # Deprecated: short-form templates now live on each ChannelSpec's
    # ``short_template`` field. Retained as a hard fallback for callers that
    # bypass the registry; new channels should populate ChannelSpec instead
    # of editing this dict.
    "click": "click({x},{y})",
    "keyboard": "{combo}",
    "drag": "drag",
    "app_switch": "switch→{to}",
    "scroll": "scroll",
    "clipboard": "clipboard",
    "clipboard_content": "clipboard",
    "visual_change": "visual_diff",
}


def _describe_event_short(
    ev: CausalEvent,
    registry: Optional[ChannelRegistry] = None,
) -> str:
    """Render a compact description for a CausalEvent.

    Looks up the per-channel ``short_template`` on the registry first; falls
    back to the legacy in-module table only when no registry/template is
    available.
    """
    reg = registry or _default_registry()
    spec = reg.get_spec(ev.channel)
    if spec and spec.short_template:
        return reg.describe_event_short(ev.channel, ev.payload)

    # Fallback for channels without a configured short_template.
    template = _SHORT_TEMPLATES.get(ev.channel, ev.channel)
    if "{" not in template:
        return template
    enriched = dict(ev.payload)
    enriched.setdefault("x", "?")
    enriched.setdefault("y", "?")
    enriched.setdefault("combo", enriched.get("key", ev.channel))
    enriched.setdefault("to", enriched.get("to_bundle", "?"))
    try:
        return template.format_map(enriched)
    except (KeyError, ValueError):
        return ev.channel


def _infer_action_name(trigger: CausalEvent) -> str:
    ch = trigger.channel
    if ch == "keyboard":
        combo = trigger.payload.get("combo", "")
        return f"shortcut.{combo.lower().replace('+', '_')}" if combo else "type"
    return ch
