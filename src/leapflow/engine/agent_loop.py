"""Per-frame execution state for the agent OODA loop (W4-M1).

An ``AgentLoopFrame`` bundles everything that must be *fresh and isolated* for a
single loop frame. The top-level turn is depth 0; each delegated subagent will
run a deeper frame with its own budget, governance, ledger, commitment, usage
tracker, recovery coordinator, and compressor. Shared, cross-frame services
(LLM, settings, stores, working memory) deliberately do NOT live here -- they
belong in ``AgentLoopServices`` (introduced in M2 alongside the runner).

M1 introduces the value object and its pure depth-governance / tool-filter
helpers only. The runner that consumes a frame (M2/M3) and the child-frame
factory (M4) land in later, separately reviewed steps -- this keeps M1
zero-risk: no engine wiring, no runtime import of the bundled subsystems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional

if TYPE_CHECKING:  # imported lazily; never required at runtime for the value object
    from leapflow.engine.budget import IterationBudget
    from leapflow.engine.context_compressor import ContextCompressor
    from leapflow.engine.context_control import ContextGovernanceController
    from leapflow.engine.prefix_commitment import PrefixCommitmentController
    from leapflow.engine.recovery_coordinator import RecoveryCoordinator
    from leapflow.engine.research_ledger import ResearchLedger
    from leapflow.engine.turn_recovery import TurnRecoveryState
    from leapflow.engine.turn_usage import TurnUsageTracker


@dataclass
class AgentLoopFrame:
    """Isolated per-frame state for one OODA loop (top-level turn or subagent).

    Carries every mutable per-turn subsystem so the same loop can run the
    top-level turn (root frame) or a recursive subagent (a deeper frame with
    fresh subsystems) without cross-frame contamination. Mutable: per-turn
    fields such as ``last_context_snapshot`` are reassigned during the loop.
    """

    user_text: str
    depth: int = 0
    # Per-frame subsystems. Optional so the value object and its pure helpers are
    # testable in isolation; the engine constructs frames with all populated.
    budget: "Optional[IterationBudget]" = None
    governance: "Optional[ContextGovernanceController]" = None
    ledger: "Optional[ResearchLedger]" = None
    commitment: "Optional[PrefixCommitmentController]" = None
    usage_tracker: "Optional[TurnUsageTracker]" = None
    recovery: "Optional[TurnRecoveryState]" = None
    compressor: "Optional[ContextCompressor]" = None
    recovery_coordinator: "Optional[RecoveryCoordinator]" = None
    recovery_budget: Optional[Any] = None
    # Per-turn observability / continuity state, reassigned during the loop.
    last_context_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_turn_tool_categories: FrozenSet[str] = frozenset()
    # Progress-gated continuation state (P0): a fingerprint of task progress
    # (ledger findings/questions/decisions + governance evidence/sources) and a
    # count of consecutive rounds without progress. Drives budget extension vs
    # convergence so a productive long task continues while a stalled one stops.
    progress_marker: tuple = ()
    stalled_rounds: int = 0
    # ``None`` tool_filter means "all tools available"; a set restricts to it.
    tool_filter: Optional[FrozenSet[str]] = None
    enable_thinking: bool = False
    parent_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        """Whether this is the top-level (user-facing) frame."""
        return self.depth == 0

    @property
    def child_depth(self) -> int:
        """Depth a child frame spawned from this one would have."""
        return self.depth + 1

    def can_delegate(self, max_depth: int) -> bool:
        """Whether this frame may spawn a child within the depth budget.

        A child at ``child_depth`` is only permitted while it stays strictly
        below ``max_depth`` (depth 0 -> child 1 allowed when max_depth >= 2).
        """
        return self.child_depth < max_depth

    def allows_tool(self, name: str) -> bool:
        """Whether a tool name is permitted in this frame's filter."""
        return self.tool_filter is None or name in self.tool_filter
