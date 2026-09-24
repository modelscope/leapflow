# Copyright (c) Alibaba, Inc. and its affiliates.
"""Core contracts for the LeapRobot inference strategy framework.

An :class:`InferenceStrategy` maps an observation to an action through some
compute path -- a local VLA policy, a remote PolicyServer, a classical
controller, and so on.  The strategy declares a :class:`ComputeProfile`
describing what it can do (chunking, ensembling, refinement, GPU need) so the
system can match a :class:`ComputeBudget` to task complexity at inference time.

The Protocol is intentionally narrow: two async methods (``infer`` and
``reset``) plus two read-only properties (``strategy_id`` and
``compute_profile``).  Everything else -- policy loading, transport lifecycle,
budget interpretation -- is an implementation detail owned by the concrete
strategy, not the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = [
    "ComputeProfile",
    "ComputeBudget",
    "InferenceResult",
    "InferenceStrategy",
]


@dataclass(frozen=True)
class ComputeProfile:
    """Declared compute characteristics of an inference strategy.

    These are static declarations, not runtime measurements: the registry uses
    them to filter and rank candidate strategies against a
    :class:`ComputeBudget` before any inference call is made.  ``gpu_required``
    is about *this process* -- a remote strategy runs the model on the server,
    so it declares ``gpu_required=False`` even though a GPU is used elsewhere.
    """

    latency_range_ms: tuple[float, float]  # (typical_min, typical_max)
    supports_chunking: bool = False  # can predict N-step action chunks
    supports_ensemble: bool = False  # can run multiple samples + vote
    supports_refinement: bool = False  # can iteratively improve output
    gpu_required: bool = False
    max_chunk_size: int = 1


@dataclass(frozen=True)
class ComputeBudget:
    """Test-time compute budget for one inference call.

    Carries the knobs a caller uses to trade latency for quality: how long the
    call may take, how many refinement steps to run, how many ensemble samples
    to draw, the confidence below which more compute should be spent, and how
    many future action steps to predict in one chunk.

    A strategy that does not support a given knob (see :class:`ComputeProfile`)
    ignores it rather than failing -- the budget is advisory, and the profile is
    the authoritative statement of what a strategy can honour.
    """

    max_latency_ms: float = 1000.0
    max_refinement_steps: int = 1
    ensemble_size: int = 1
    confidence_threshold: float = 0.8
    chunk_size: int = 1


@dataclass(frozen=True)
class InferenceResult:
    """Output of one inference call.

    ``action`` is deliberately typed ``Any``: a strategy may return a dict of
    channel -> value, a torch tensor, a numpy array, or a plain list, and the
    caller adapts it to hardware commands downstream.  The remaining fields are
    telemetry the system uses to drive budget adaptation and trust: measured
    ``latency_ms``, a ``confidence`` score in ``[0, 1]``, the ``chunk_size`` the
    action represents, and strategy-specific ``metadata``.
    """

    action: Any  # typically a dict or tensor
    latency_ms: float = 0.0
    confidence: float = 1.0
    chunk_size: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceStrategy(Protocol):
    """Pluggable physical AI inference strategy.

    Any object satisfying this protocol can be registered with the
    :class:`~leapflow.robot.inference.registry.InferenceStrategyRegistry` and
    used by ``PhysicalSkillPlugin`` in place of the two hardcoded local/remote
    paths.  Implementations are free to lazily load heavy resources (models,
    transports) inside :meth:`infer` so that construction stays cheap and
    dependency-free.
    """

    @property
    def strategy_id(self) -> str:
        """Stable, unique identifier used as the registry key."""
        ...

    @property
    def compute_profile(self) -> ComputeProfile:
        """Declared compute characteristics used for budget-aware selection."""
        ...

    async def infer(
        self,
        observation: Mapping[str, Any],
        *,
        budget: ComputeBudget | None = None,
    ) -> InferenceResult:
        """Map an observation to an action under an optional compute budget.

        When *budget* is ``None`` the strategy uses its own defaults.  A
        strategy must honour the knobs its :class:`ComputeProfile` advertises
        and may ignore the rest.
        """
        ...

    async def reset(self) -> None:
        """Reset per-episode state (action queues, hidden states, caches)."""
        ...
