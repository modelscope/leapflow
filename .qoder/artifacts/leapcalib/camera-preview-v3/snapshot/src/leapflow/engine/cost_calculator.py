# Copyright (c) Alibaba, Inc. and its affiliates.
"""Config-driven token cost calculator.

Computes dollar cost from token usage and a pricing overlay loaded through the
layered config system.  Pricing is keyed by exact model name or model family
prefix; resolution tries exact match first, then longest-prefix match.

Design:
- Frozen/stateless helper — no side effects, no network, no caching.
- Graceful degradation: missing pricing → cost unknown (None), never crash.
- Cold-path only: called once per turn summary, not per token.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """Resolved pricing for one model (per million tokens)."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_ratio: float = 0.1

    def validate(self) -> bool:
        """Return True if all values are non-negative."""
        return (
            self.input_per_mtok >= 0.0
            and self.output_per_mtok >= 0.0
            and 0.0 <= self.cached_input_ratio <= 1.0
        )


@dataclass(frozen=True)
class CostResult:
    """Immutable cost computation result."""

    dollar_cost: Optional[float] = None
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    cached_input_cost: Optional[float] = None
    model: str = ""
    pricing_source: str = ""  # "exact", "prefix", or "" if unknown

    @property
    def known(self) -> bool:
        """Whether pricing was resolved and cost is available."""
        return self.dollar_cost is not None


def resolve_pricing(
    model: str,
    pricing_config: Dict[str, Any],
) -> Optional[ModelPricing]:
    """Resolve pricing for a model from the config overlay.

    Resolution order:
    1. Exact model name match (case-insensitive).
    2. Longest prefix match among config keys (e.g. "deepseek" matches
       "deepseek-chat", "deepseek-reasoner").

    Returns None when no pricing entry matches (graceful degradation).
    """
    if not model or not pricing_config:
        return None

    model_lower = model.lower()

    # 1. Exact match
    for key, entry in pricing_config.items():
        if key.lower() == model_lower:
            return _parse_entry(entry)

    # 2. Longest prefix match
    best_key: Optional[str] = None
    best_len: int = 0
    for key in pricing_config:
        key_lower = key.lower()
        if model_lower.startswith(key_lower) and len(key_lower) > best_len:
            best_key = key
            best_len = len(key_lower)

    if best_key is not None:
        return _parse_entry(pricing_config[best_key])

    # 3. Regex match against config keys that look like patterns
    for key, entry in pricing_config.items():
        try:
            if re.search(key, model_lower):
                return _parse_entry(entry)
        except re.error:
            continue

    return None


def compute_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    model: str,
    pricing_config: Dict[str, Any],
) -> CostResult:
    """Compute dollar cost for a turn's token usage.

    Pricing handles cached tokens correctly:
    - ``miss_tokens = prompt_tokens - cached_tokens`` charged at full input rate.
    - ``cached_tokens`` charged at ``input_rate * cached_input_ratio``.
    - ``completion_tokens`` charged at the output rate.

    Returns a CostResult with ``dollar_cost=None`` when pricing is unavailable.
    """
    pricing = resolve_pricing(model, pricing_config)
    if pricing is None or not pricing.validate():
        return CostResult(model=model)

    miss_tokens = max(0, prompt_tokens - cached_tokens)
    cached = max(0, cached_tokens)

    input_cost = (miss_tokens / 1_000_000) * pricing.input_per_mtok
    cached_input_cost = (cached / 1_000_000) * pricing.input_per_mtok * pricing.cached_input_ratio
    output_cost = (completion_tokens / 1_000_000) * pricing.output_per_mtok

    total = round(input_cost + cached_input_cost + output_cost, 6)

    # Determine pricing source for diagnostics
    source = ""
    model_lower = model.lower()
    for key in pricing_config:
        if key.lower() == model_lower:
            source = "exact"
            break
    if not source:
        source = "prefix"

    return CostResult(
        dollar_cost=total,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        cached_input_cost=round(cached_input_cost, 6),
        model=model,
        pricing_source=source,
    )


def format_cost(cost: Optional[float]) -> str:
    """Human-readable cost string: '$0.0042' or 'unknown'."""
    if cost is None:
        return "unknown"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _parse_entry(entry: Any) -> Optional[ModelPricing]:
    """Parse a pricing config entry into a ModelPricing, or None on error."""
    if isinstance(entry, dict):
        try:
            return ModelPricing(
                input_per_mtok=float(entry.get("input_per_mtok", 0)),
                output_per_mtok=float(entry.get("output_per_mtok", 0)),
                cached_input_ratio=float(entry.get("cached_input_ratio", 0.1)),
            )
        except (TypeError, ValueError):
            logger.debug("Invalid pricing entry: %s", entry)
            return None
    return None


__all__ = [
    "CostResult",
    "ModelPricing",
    "compute_cost",
    "format_cost",
    "resolve_pricing",
]
