"""Copilot configuration model — all tuneable parameters in one place.

Every threshold, toggle, and budget is exposed here so that runtime behaviour
can be adjusted without code changes (e.g. via settings file or env vars).

SRP: Only declares configuration defaults — no logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CopilotConfig:
    """Workflow Copilot configuration — centralised tuning surface.

    Grouped by subsystem for readability:
      • Global toggles
      • Context encoder parameters
      • Prediction layer toggles & budgets
      • Display / idle gating
      • Feedback & evolution
      • Resource budgets / degradation
      • Speculative cache
    """

    # ── Global ────────────────────────────────────────────────────────────
    enabled: bool = True

    # ── Context Encoder ───────────────────────────────────────────────────
    action_ring_size: int = 10
    time_bucket_minutes: int = 60

    # ── Prediction Layers ─────────────────────────────────────────────────
    l0_enabled: bool = True
    l1_enabled: bool = True
    l2_enabled: bool = True
    l3_enabled: bool = True
    l3_complexity_threshold: float = 0.5

    # ── Display / Idle Gating ─────────────────────────────────────────────
    min_confidence_display: float = 0.3
    min_idle_ms: int = 300
    max_idle_ms: int = 3000

    # ── Feedback & Evolution ──────────────────────────────────────────────
    ema_alpha: float = 0.1
    ignore_decay: float = -0.1
    accept_boost: float = 1.0

    # ── Resource Budgets / Degradation ────────────────────────────────────
    memory_budget_mb: float = 50.0
    max_event_queue: int = 1000

    # ── Speculative Cache ─────────────────────────────────────────────────
    speculative_cache_size: int = 100
    cache_ttl_seconds: float = 30.0

    # ── Extensibility ─────────────────────────────────────────────────────
    extra_predictor_layers: List[str] = field(default_factory=list)
