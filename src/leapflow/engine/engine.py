# Copyright (c) Alibaba, Inc. and its affiliates.
"""Main ReAct-style engine with routing, skills, and audit logging."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import types
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from leapflow.platform.protocol import HostRpc
from leapflow.config import Settings
from leapflow.engine.budget import BudgetConfig, BudgetStatus, IterationBudget
from leapflow.engine.prefix_commitment import (
    PrefixCommitmentController,
)
from leapflow.engine.research_ledger import ResearchLedger
from leapflow.engine.agent_loop import AgentLoopFrame
from leapflow.engine.context.context_compressor import CompressorConfig, ContextCompressor
from leapflow.engine.context.context_control import (
    ContextBudgetEstimator,
    ContextGovernanceController,
    ContextPostureConfig,
    ContextWindowController,
    ToolEvidenceBuilder,
)
from leapflow.engine.context.context_disclosure import (
    CacheBoundary,
    DisclosurePlanner,
)
from leapflow.engine.context.context_focus import ReferenceResolution, SessionFocusState
from leapflow.engine.context.reference_resolver import ReferenceResolver
from leapflow.engine.recovery.error_classifier import (
    ErrorCategory,
    ErrorClassifier,
    build_recovery_map,
    jittered_backoff,
)
from leapflow.engine.tools.execution_trace import ExecutionMode, ExecutionTrace
from leapflow.engine.intent_classifier import IntentClassifier
from leapflow.engine.message_healer import MessageHealer
from leapflow.engine.message_sanitizer import MessageSanitizer
from leapflow.engine.prompt_cache import AnthropicCacheStrategy, CacheStrategy
from leapflow.engine.stale_stream import (
    StaleStreamError,
    stale_guarded_stream,
    build_continuation_prompt,
)
from leapflow.engine.recovery.turn_recovery import TurnRecoveryState
from leapflow.engine.turn_usage import (
    TurnUsageTracker,
)
from leapflow.engine.recovery.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.recovery.recovery_budget import RecoveryBudget
from leapflow.engine.recovery.unified_classifier import UnifiedErrorClassifier
from leapflow.engine.recovery.recovery_decision import RecoveryAction, RecoveryDecision
from leapflow.engine.recovery.strategies import default_strategies
from leapflow.engine.recovery.recovery_audit import JsonlAuditSink, create_audit_entry
from leapflow.engine.recovery.recovery_checkpoint import RecoveryCheckpoint, InMemoryCheckpointStore
from leapflow.engine.tools.tool_concurrency import (
    DefaultConcurrencyPolicy,
    ToolConcurrencyPolicy,
)
from leapflow.engine.tools.action_executor import ActionExecutor, RecordedActionExecutor
from leapflow.engine.tools.tool_execution import (
    ToolExecutionLedger,
)
from leapflow.engine.task_planning.graph_planner import GraphPlanner
from leapflow.engine.task_planning.scheduler import TaskScheduler
from leapflow.engine.session.session import SessionController
from leapflow.analysis.pipeline import ImitationPipeline
from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import (
    build_assistant_message,
    build_system_message,
    build_user_message_text,
)
from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.memory.providers.evolution import EvolutionMemoryProvider
from leapflow.memory.manager import MemoryManager
from leapflow.learning.active_learning import SkillMerger
from leapflow.storage.skill_library import SkillLibraryStore
from leapflow.storage.reentry_store import build_reentry_trigger
from leapflow.skills.registry import SkillRegistry
from leapflow.engine._stream_helpers import (
    BufferSink,
    OutputSink,
    StreamEvent,
    StreamSink,
    TaskContract,
)
from leapflow.engine._tool_helpers import (
    _normalize_tool_name,
    _concurrency_spec_lookup,
    _normalize_tool_call,
)
from leapflow.engine._message_helpers import (
    _EMPTY_RESPONSE_RETRY_PROMPT,
    _EMPTY_RESPONSE_DEGRADED_MESSAGE,
    _FORCED_FINALIZE_PROMPT,
    _truncate_result_for_budget,
    _tool_args_metadata,
    _tool_result_metadata,
    _is_retryable_unknown_tool_result,
    _has_completed_side_effect,
    _unknown_tool_retry_prompt,
    _is_permission_hard_stop_payload,
    _tool_result_counts_as_failure,
    _terminal_failure_text,
    _interaction_metadata,
    _permission_hard_stop_from_results,
    _build_native_tool_assistant_message,
    _permission_override_message,
    _last_tool_failures_recovery_message,
    _app_onboarding_recovery_message,
    _clear_indicator,
    _print_tool_result,
)
from leapflow.engine.session_persistence import SessionPersistence
from leapflow.engine.calibration import CalibrationManager
from leapflow.engine.learning_bridge import LearningBridge
from leapflow.engine.skill_dispatcher import SkillDispatcher
from leapflow.engine.prompt_assembler import PromptAssembler
from leapflow.engine.tool_dispatch_engine import ToolDispatchEngine

logger = logging.getLogger(__name__)



class AgentEngine:
    """Coordinates perception memory, LLM reasoning, RPC execution, and skills."""

    def __init__(
        self,
        settings: Settings,
        rpc: HostRpc,
        llm: LLMProvider,
        wm: WorkingMemoryProvider,
        lt: SemanticMemoryProvider,
        imm: EpisodicMemoryProvider,
        registry: SkillRegistry,
        classifier: IntentClassifier,
        imitation: Optional[ImitationPipeline] = None,
        skill_library: Optional[SkillLibraryStore] = None,
        graph_planner: Optional[GraphPlanner] = None,
        perception: Optional[Any] = None,
        execution: Optional[Any] = None,
        skill_activator: Optional[Any] = None,
        session: Optional[SessionController] = None,
        vlm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None,
        evolution: Optional[EvolutionMemoryProvider] = None,
        skill_injector: Optional[Any] = None,
        skill_index: Optional[Any] = None,
        concurrency_policy: Optional[ToolConcurrencyPolicy] = None,
        action_executor: Optional[ActionExecutor] = None,
    ) -> None:
        self._settings = settings
        self._rpc = rpc
        self._llm = llm
        self._vlm = vlm
        self._wm = wm
        self._lt = lt
        self._imm = imm
        self._registry = registry
        self._classifier = classifier
        self._imitation = imitation
        self._skill_library = skill_library
        self._skill_merger = SkillMerger(
            registry=registry,
            llm=llm,
            execution=execution,
        )
        self._graph_planner = graph_planner
        self._scheduler: Optional[TaskScheduler] = None
        self._perception = perception
        self._execution = execution
        self._activator = skill_activator
        self._session = session

        # Memory integration (MemoryManager + EvolutionProvider)
        self._memory_manager = memory_manager
        self._evolution = evolution

        # Skill index for compact prompt injection
        self._skill_index: Optional[Any] = skill_index

        # Skill discovery (SkillInjector for slash commands)
        self._skill_injector = skill_injector

        # Tool concurrency policy (None = sequential fallback)
        self._concurrency_policy: Optional[ToolConcurrencyPolicy] = (
            concurrency_policy
            if concurrency_policy is not None
            else DefaultConcurrencyPolicy(spec_lookup=_concurrency_spec_lookup)
        )

        # Session persistence (injected by CLI)
        self._conversation_store: Optional[Any] = None
        self._current_session_id: Optional[str] = None
        self._current_turn_id: str = ""
        self._current_command_id: str = ""
        self._tool_execution_ledger = ToolExecutionLedger()
        # The executor is profile-scoped while every invocation receives identity
        # from this session engine, preserving isolation across shallow copies.
        self._action_executor: ActionExecutor = action_executor or RecordedActionExecutor(None)
        if self._graph_planner is not None:
            self._scheduler = TaskScheduler(
                self._registry,
                graph_planner=self._graph_planner,
                action_dispatcher=self.execute_action,
            )

        self._current_request_id: str = ""

        # Memory context snapshot (frozen at session start for prefix cache stability)
        self._memory_context_snapshot: Optional[str] = None

        # Cancellation: tracks active task for interrupt support
        self._active_task: Optional[asyncio.Task] = None
        self._cancel_requested = False
        # Per-turn identity, set at turn start; initialized so reads and the
        # subagent frame save/restore never hit an unset attribute. NOTE:
        # _current_session_id is intentionally NOT re-initialized here — it is set
        # to None at the top of __init__, and the session-creation path relies on
        # that ``is None`` sentinel to mint a new session id, so overriding it to
        # "" would silently disable conversation persistence.
        self._current_turn_id: str = ""
        self._current_command_id: str = ""

        # Tool loop guardrails (injected by CLI)
        self._guardrail: Optional[Any] = None

        # Optional override for dynamic tool result budget (set by CLI wiring)
        self._tool_result_budget: Optional[int] = None

        # Per-turn usage tracking
        self._usage_tracker = TurnUsageTracker()
        # Wire plugin learning sink (process-global; graceful no-op if unavailable)
        try:
            from leapflow.engine.session.session_factory import _wire_plugin_stats_sink

            _wire_plugin_stats_sink(self._usage_tracker)
        except (ImportError, RuntimeError, AttributeError):
            pass

        # Per-tool timeout (seconds); can be overridden via set_tool_timeouts
        self._default_tool_timeout_s: float = 120.0
        self._tool_timeouts: Dict[str, float] = {}

        # Stale stream timeout
        self._stale_stream_timeout_s: float = 180.0

        # Evolution store for incremental persistence (injected by CLI)
        self._evolution_store: Optional[Any] = None

        # EventBus for learning signal emission (injected by CLI)
        self._event_bus: Optional[Any] = None

        # ExperienceStore bridge for world-model trajectory data (injected by CLI)
        self._experience_store: Optional[Any] = None

        # Model capability registry (injected by CLI)
        self._model_capabilities: Optional[Any] = None

        # Session-level counters (survive per-turn tracker resets)
        self._session_turn_count: int = 0
        self._last_context_tokens: int = 0

        # State-machine loop infrastructure (config-driven)
        self._budget_config = BudgetConfig(
            max_iterations=settings.agent_iter_floor,
            soft_limit=settings.react_soft_limit,
            warning_threshold=settings.react_warning_threshold,
            iter_ceiling=settings.agent_iter_ceiling,
            hard_cap=settings.agent_iter_hard_cap,
            scale_k=settings.agent_budget_scale_k,
        )
        # S3-L3: baseline difficulty weight, kept as the rollback/recompute anchor
        # so calibration never compounds and reset is exact.
        self._baseline_scale_k = settings.agent_budget_scale_k
        # S3-L4: calibrated finalize-posture threshold (None = use configured baseline).
        self._calibrated_finalizing_ratio: Optional[float] = None
        # S3 periodic re-calibration (opt-in): evolution store + root-turn counter.
        self._calibration_store: Optional[Any] = None
        self._calibration_event_store: Optional[Any] = None
        self._turns_since_calibration = 0
        self._error_classifier = ErrorClassifier(
            recovery_map=build_recovery_map(
                transient_max_retries=settings.error_transient_max_retries,
                rate_limit_base_delay=settings.error_rate_limit_base_delay,
            )
        )
        self._compressor = self._new_compressor()
        self._context_controller = ContextWindowController(
            estimator=ContextBudgetEstimator(),
            hard_limit_ratio=settings.context_hard_limit_ratio,
            warning_ratio=settings.context_warning_ratio,
        )
        self._context_governance_controller = self._new_governance()
        self._prefix_commitment = PrefixCommitmentController()
        self._research_ledger = ResearchLedger()
        self._research_ledger_store: Optional[Any] = None
        self._reentry_store: Optional[Any] = None
        self._active_frame: Optional[AgentLoopFrame] = None
        self._full_tools_tokens: int | None = None
        self._last_context_snapshot: dict[str, Any] = {}
        self._last_disclosure_metadata: dict[str, Any] = {}
        self._current_task_contract: TaskContract | None = None
        self._disclosure_planner = DisclosurePlanner()
        self._focus_state = SessionFocusState()
        self._reference_resolver = ReferenceResolver()
        self._last_reference_resolution: ReferenceResolution | None = None
        # Distilled knowledge is read on the hot path and written on the cold one, so
        # the engine holds the reader. Bound lazily rather than in the constructor
        # because the profile layout is not always present (tests, in-process CLI), and
        # a missing store must degrade context quality rather than fail construction.
        self._knowledge_store: Any = None
        # Set once a lookup has failed, so a persistent failure costs one attempt rather
        # than one per turn. The cold-path governor deliberately keeps retrying -- a
        # sweep runs once per session, so a transient error there should not disable
        # governance for the life of the process.
        self._knowledge_store_unavailable: bool = False
        # The environment the current session runs in. Compared against the environment
        # a fact was learned in, so a stale-looking fact can be disclosed *as* such
        # instead of being silently dropped or silently trusted.
        self._environment_fingerprint_id: str = ""
        # Tier 1 structural continuity gate: capability categories used by native
        # tool_calls in the most recently completed turn. Working memory only
        # stores a synthetic "[Called: ...]" summary (no structured tool_calls),
        # so this dedicated, reset-per-turn attribute is the actual source of
        # truth — never derived from re-parsing text.
        self._last_turn_tool_categories: frozenset[str] = frozenset()
        from leapflow.learning.capability_observation import CapabilityObservationBuffer

        self._capability_observation_buffer = CapabilityObservationBuffer()
        self._active_capability_plan: dict[str, Any] | None = None
        self._manifests_by_name: Dict[str, Any] | None = None
        # Semantic desktop schema cache. The plugin is re-resolved from the tool
        # registry on every read, and cache keys carry (plugin identity, version)
        # so an unregistered plugin yields zero schemas and a reloaded instance
        # (whose version counter restarts at 0) never collides with a cached
        # predecessor entry.
        self._semantic_plugin_key: Optional[tuple[int, int]] = None
        self._semantic_schemas: List[Dict[str, Any]] = []
        self._unified_catalog_key: Optional[tuple] = None
        self._unified_catalog: List[Dict[str, Any]] = []
        # Phase 5: tool execution/dispatch component (back-reference to engine).
        self._tool_dispatch = ToolDispatchEngine(self)
        # Capability discovery resolves the live catalog through this engine, so
        # runtime-injected categories (desktop) become expandable.
        from leapflow.plugins import get_registry

        _plugin_registry = get_registry()
        _plugin_registry.set_capability_catalog_provider(self._tool_dispatch._unified_tool_catalog)
        self._healer = MessageHealer()

        # B2: Prompt cache optimization (None = disabled)
        self._cache_strategy: CacheStrategy | None = None

        # B4: Output sanitization (None = disabled)
        self._sanitizer: MessageSanitizer | None = None

        # PCD cache-aware: frozen state for session restore (set by load_session
        # or session_factory when resuming a committed session; cleared on next
        # turn's _assemble_unified_prompt after being consumed).
        self._frozen_system_prompt: Optional[str] = None
        self._frozen_tool_schema: Optional[str] = None
        # PCD cache-aware: last-round tracking for snapshot persistence
        self._last_system_prompt: str = ""
        self._last_tool_definitions_json: str = ""
        self._last_disclosure_level: str = ""
        # PCD cache-aware: cache boundary from current assembly plan
        self._current_cache_boundary: CacheBoundary = CacheBoundary.NONE
        # PCD cache-aware: posture tracking for commitment breaking
        self._prev_context_posture: str = "baseline"
        # PCD cache-aware: dedicated compression provider (None = use primary)
        self._compression_provider: Optional[LLMProvider] = None
        try:
            self._compression_provider = self._build_compression_provider()
        except Exception:  # noqa: BLE001 - degrade to primary, never crash init
            logger.debug("compression provider build failed at init", exc_info=True)

        # Recovery coordinator infrastructure
        self._unified_classifier = UnifiedErrorClassifier(self._error_classifier)
        self._recovery_coordinator = RecoveryCoordinator()  # Re-created per turn
        self._checkpoint_store = InMemoryCheckpointStore()
        self._audit_sink = JsonlAuditSink(self._recovery_audit_path())

        # Extracted method-group components (Phase 3 refactor). Each holds a
        # back-reference to this engine so it reads live mutable state; place
        # after all engine attributes above are initialized.
        self._session_persistence = SessionPersistence(self)
        self._calibration_manager = CalibrationManager(self)
        self._learning_bridge = LearningBridge(self)
        self._skill_dispatcher = SkillDispatcher(self)
        self._prompt_assembler = PromptAssembler(self)

        # Apply startup-time tool configuration derived from settings.
        self._configure_tool_defaults()

    def _configure_tool_defaults(self) -> None:
        """Wire settings-derived values into module-level tool defaults at start-up.

        Kept in its own method so child frames and test fixtures can call it
        without re-running the full ``__init__`` body.
        """
        try:
            from leapflow.tools.shell_tools import set_max_shell_timeout

            set_max_shell_timeout(self._settings.max_shell_timeout_s)
        except Exception:  # noqa: BLE001 - optional; defaults remain if import fails
            logger.debug("_configure_tool_defaults: shell_tools not available")

    # ── Optional strategy setters (config-driven) ────────────────────────

    def set_cache_strategy(self, strategy: CacheStrategy | None) -> None:
        """Configure prompt cache optimization strategy."""
        self._cache_strategy = strategy

    def set_sanitizer(self, sanitizer: MessageSanitizer | None) -> None:
        """Configure output message sanitizer."""
        self._sanitizer = sanitizer

    def reconfigure_host_backend(
        self,
        *,
        rpc: HostRpc,
        perception: Optional[Any],
        execution: Optional[Any],
    ) -> None:
        """Refresh host RPC and adapters without resetting chat/session state."""
        self._rpc = rpc
        self._perception = perception
        self._execution = execution
        # Desktop semantic surfaces need no refresh here: the plugin is
        # re-resolved from the tool registry on every engine read, and the
        # context has already re-bound it via registry.bind_runtime.
        self._skill_merger = SkillMerger(
            registry=self._registry,
            llm=self._llm,
            execution=execution,
        )
        if self._settings.has_llm_credentials:
            self._scheduler = TaskScheduler(
                self._registry,
                graph_planner=self._graph_planner,
                action_dispatcher=self.execute_action,
            )
        else:
            self._scheduler = None

    def reconfigure_runtime(
        self,
        *,
        settings: Settings,
        llm: LLMProvider,
        vlm: Optional[Any],
        classifier: IntentClassifier,
    ) -> None:
        """Refresh runtime LLM configuration without resetting session state."""
        self._settings = settings
        self._llm = llm
        self._vlm = vlm
        self._classifier = classifier
        self._compressor = self._new_compressor()
        self._context_controller = ContextWindowController(
            estimator=ContextBudgetEstimator(),
            hard_limit_ratio=settings.context_hard_limit_ratio,
            warning_ratio=settings.context_warning_ratio,
        )
        self._context_governance_controller = self._new_governance()
        self._skill_merger = SkillMerger(
            registry=self._registry,
            llm=llm,
            execution=self._execution,
        )
        if settings.has_llm_credentials:
            self._graph_planner = GraphPlanner(self._llm, self._registry)
            self._scheduler = TaskScheduler(
                self._registry,
                graph_planner=self._graph_planner,
                action_dispatcher=self.execute_action,
            )
        else:
            self._graph_planner = None
            self._scheduler = None

    def set_tool_result_budget(self, budget: int) -> None:
        """Override per-tool result truncation budget (e.g. linked to model context)."""
        self._tool_result_budget = max(1, budget)

    def _effective_tool_result_budget(self) -> int:
        return self._tool_result_budget or self._settings.max_tool_result_chars

    async def _handle_api_error(
        self,
        classified: ErrorCategory,
        rec: Any,
        recovery: TurnRecoveryState,
        messages: list,
        budget: Any,
        *,
        use_native_tools: bool = False,
        tools_kwarg: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Legacy API error recovery dispatcher. Returns 'continue' to retry, else None.

        DEPRECATED: No longer called from main loops. Retained only for backward
        compat with any external subclass overrides. All recovery now flows through
        RecoveryCoordinator.evaluate().
        """
        if classified == ErrorCategory.CONTEXT_OVERFLOW and recovery.try_compress():
            messages[:] = self._compressor.force_compress(messages)
            logger.info("recovery: force_compress on context overflow")
            if budget.remaining > 0:
                return "continue"

        if classified == ErrorCategory.IMAGE_TOO_LARGE and recovery.try_multimodal_strip():
            self._strip_images_from_messages(messages)
            logger.info("recovery: stripped images from messages")
            if budget.remaining > 0:
                return "continue"

        if rec.should_fallback and recovery.try_provider_failover():
            llm = self._llm
            if hasattr(llm, "_failover"):
                llm._failover("recovery: provider failover")
            logger.info("recovery: provider failover triggered")
            if budget.remaining > 0:
                return "continue"

        if rec.should_rotate_credential and recovery.try_credential_rotate():
            logger.info("recovery: credential rotation requested")
            if budget.remaining > 0:
                return "continue"

        if classified == ErrorCategory.FORMAT_ERROR and recovery.try_disable_thinking():
            logger.info("recovery: disabled thinking mode")
            if budget.remaining > 0:
                return "continue"

        if rec.retry and budget.remaining > 0:
            if rec.backoff:
                await asyncio.sleep(jittered_backoff(budget.used, base=rec.base_delay))
            return "continue"

        return None

    _DEFAULT_LIVE_SIGNAL_KINDS = frozenset(
        {
            "app.focus_change",
            "fs.change",
            "context.change",
            "intent.signal",
        }
    )

    def _inject_live_signals(self, messages: list, watermark: list) -> None:
        """Inject high-priority WM events arrived since ``watermark[0]``.

        Uses a mutable watermark list (single-element) so the caller's
        timestamp advances after each injection, preventing duplicate
        signal messages across loop iterations.
        """
        since_ts = watermark[0]
        raw = getattr(self._settings, "live_signal_kinds", "")
        signal_kinds = (
            frozenset(k.strip() for k in raw.split(",") if k.strip())
            if raw
            else self._DEFAULT_LIVE_SIGNAL_KINDS
        )
        recent = self._wm.get_events_since(since_ts)
        relevant = [e for e in recent if e.get("_event_kind") in signal_kinds]
        if not relevant:
            return
        lines = []
        for ev in relevant[-5:]:
            text = ev.get("_event_text", "")
            if text:
                lines.append(str(text)[:120])
            else:
                lines.append(str(ev.get("content", ""))[:120])
        summary = "; ".join(lines)
        messages.append(build_system_message(f"[LIVE SIGNAL] {summary}"))
        watermark[0] = time.time()

    @staticmethod
    def _strip_images_from_messages(messages: list) -> None:
        """Remove image content parts from messages in-place (multimodal strip)."""
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [
                    p
                    for p in content
                    if isinstance(p, dict)
                    and p.get("type") != "image_url"
                    and p.get("type") != "input_image"
                    and p.get("type") != "image"
                ]
                if len(text_parts) < len(content):
                    if text_parts:
                        messages[i] = {**msg, "content": text_parts}
                    else:
                        messages[i] = {**msg, "content": "[images removed to reduce context]"}

    def _execute_transform_decision(
        self,
        decision: "RecoveryDecision",
        messages: list,
    ) -> bool:
        """Execute a TRANSFORM_AND_RETRY decision. Returns True if transform succeeded.

        Handles different transform strategies:
        - context_compress: force-compress conversation history
        - multimodal_strip: remove image content from messages
        - native_to_text: disable native tool calling
        - thinking_disable: disable thinking mode (handled externally)
        """
        strategy_key = decision.strategy_key
        # Determine specific phase from audit_metadata if available
        phase = dict(decision.audit_metadata).get("phase", "")

        if strategy_key == "context_compress":
            if phase == "multimodal_to_text":
                self._strip_images_from_messages(messages)
            else:
                # Default: history_summarize and disclosure_shrink both use force_compress
                messages[:] = self._compressor.force_compress(messages)
            return True

        if strategy_key == "multimodal_strip":
            self._strip_images_from_messages(messages)
            return True

        if strategy_key == "native_to_text":
            # Handled by caller via tools_kwarg mutation
            return True

        if strategy_key == "thinking_disable":
            # Handled by caller via enable_thinking flag
            return True

        logger.warning("Unknown transform strategy: %s", strategy_key)
        return True

    def _post_failover_recompress(
        self,
        messages: list,
        coordinator: "RecoveryCoordinator",
        failover_decision: "RecoveryDecision",
    ) -> bool:
        """Recompress messages when a failover landed on a smaller-window provider.

        Called immediately after ``RecoveryAction.FAILOVER`` is applied. If the
        new provider's context window is smaller than the estimated prompt
        payload, a force-compress pass is run on the message list so the
        retry does not waste a round trip or fail.

        Compression is non-side-effecting, so ``SideEffectState`` gating
        permits it unconditionally.

        Returns True if recompression was applied, False if it was not needed.
        """
        new_window = self._active_context_length()
        estimated = self._context_controller.estimator.estimate_messages(messages)
        if estimated <= new_window:
            return False

        logger.info(
            "provider_context_handoff: recompressing after failover "
            "(estimated=%d tokens > new_window=%d)",
            estimated, new_window,
        )
        messages[:] = self._compressor.force_compress(messages)
        self._usage_tracker.mark_compression()

        # Record the handoff recompression through the coordinator audit trail.
        coordinator.on_strategy_outcome(
            failover_decision.decision_id, True,
        )
        self._audit_sink.update_outcome(
            failover_decision.decision_id,
            "success",
            reason=(
                f"post-failover recompression applied: "
                f"{estimated} tokens compressed to fit {new_window} window"
            ),
        )
        return True

    def _save_halt_checkpoint(
        self,
        decision: Any,
        envelope: Any,
        messages: List[Dict[str, Any]],
        *,
        budget_used: int,
        tools_kwarg: Optional[Dict[str, Any]] = None,
        use_native_tools: bool = False,
    ) -> None:
        """Persist a resumable checkpoint for a ``HALT_WITH_CHECKPOINT`` decision.

        Shared by every terminal dispatch site (streaming included): the gate
        that withholds a replay after a side effect emits this action from any
        path, and a halt that skips the save leaves the decision's
        ``resumption_key`` pointing at nothing. The envelope id and the
        interaction's request id are recorded so the client can find this
        checkpoint again from the ``InteractionRequest`` it was shown
        (``resumption_key`` == envelope id; lookup via ``list_pending``).
        """
        interaction = getattr(decision, "interaction", None)
        try:
            checkpoint = RecoveryCheckpoint(
                session_id=getattr(self, "_current_session_id", "") or "",
                turn_id=budget_used,
                failure_envelope_data={
                    "envelope_id": envelope.envelope_id,
                    "message": envelope.message,
                    "category": envelope.category,
                    "failure_code": envelope.failure_code,
                    "source": envelope.source.value,
                    "side_effect_state": envelope.side_effect_state.value,
                },
                interaction_request_id=(interaction.request_id if interaction is not None else ""),
                messages_snapshot=list(messages),
                context_data={
                    "resumption_key": getattr(interaction, "resumption_key", "") or "",
                    "tools_kwarg_keys": list((tools_kwarg or {}).keys()),
                    "use_native_tools": use_native_tools,
                    "budget_used": budget_used,
                },
            )
            self._checkpoint_store.save(checkpoint)
        except Exception:  # noqa: BLE001 - a failed save must not mask the halt itself
            logger.warning("recovery: failed to persist halt checkpoint", exc_info=True)

    def set_tool_timeouts(self, timeouts: Dict[str, float]) -> None:
        """Set per-tool execution timeout overrides (seconds)."""
        self._tool_timeouts = dict(timeouts)

    def set_default_tool_timeout(self, timeout_s: float) -> None:
        self._default_tool_timeout_s = max(5.0, timeout_s)

    def set_stale_stream_timeout(self, timeout_s: float) -> None:
        self._stale_stream_timeout_s = max(30.0, timeout_s)

    def set_evolution_store(self, store: Any) -> None:
        """Inject evolution store for incremental episode persistence."""
        self._evolution_store = store

    def set_distilled_knowledge_store(self, store: Any) -> None:
        """Inject the event-derived knowledge read model shared by session engines."""
        self._knowledge_store = store
        self._knowledge_store_unavailable = store is None

    def set_model_capabilities(self, registry: Any) -> None:
        """Inject model capability registry."""
        self._model_capabilities = registry

    def set_doc_store(self, doc_store: Any) -> None:
        """Inject SkillDocStore so SkillMerger can sync SKILL.md on approve."""
        self._skill_merger.set_doc_store(doc_store)

    def set_event_bus(self, event_bus: Any) -> None:
        """Inject EventBus for emitting learning signals (episode events)."""
        self._event_bus = event_bus

    def set_experience_store(self, store: Any) -> None:
        """Inject ExperienceStore for world-model trajectory bridge."""
        self._experience_store = store

    def set_conversation_store(self, store: Any) -> None:
        """Inject conversation persistence store."""
        self._conversation_store = store
        self._tool_execution_ledger.reset(store=store)

    def set_research_ledger_store(self, store: Any) -> None:
        """Inject the research-ledger persistence store (S1, optional).

        Wires the ledger change-listener so each note is persisted per session
        (durable Orient). Without a store, the ledger degrades gracefully to
        per-turn in-memory state.
        """
        self._research_ledger_store = store
        self._research_ledger.set_change_listener(self._persist_research_ledger)

    def set_reentry_store(self, store: Any) -> None:
        """Inject the re-entry store (S2, optional).

        Absent => ``schedule_reentry`` reports "not configured". Registration is
        additionally gated by ``agent_reentry_enabled`` (default off).
        """
        self._reentry_store = store

    def load_session(self, session_id: str) -> bool:
        """Resume a previous session by loading messages from DuckDB.

        Returns True if the session was found and messages loaded.
        """
        return self._session_persistence.load_session(session_id)

    def freeze_prefix_for_resume(
        self,
        *,
        system_prompt: Optional[str],
        tool_schema: Optional[str],
        disclosure_level: Optional[str],
    ) -> None:
        """Freeze a persisted prefix so the next turn reproduces it verbatim (5c)."""
        self._session_persistence.freeze_prefix_for_resume(
            system_prompt=system_prompt,
            tool_schema=tool_schema,
            disclosure_level=disclosure_level,
        )

    def apply_resume_cache_snapshot(self, session_id: str) -> bool:
        """Load and apply a persisted prefix snapshot on resume (5c)."""
        return self._session_persistence.apply_resume_cache_snapshot(session_id)

    def cancel(self) -> None:
        """Request cancellation of the active run/run_stream call.

        Thread-safe: can be called from signal handlers or other threads.
        Cancels the active asyncio task if one is tracked.
        """
        self._cancel_requested = True
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested

    @property
    def model_capabilities(self) -> Optional[Any]:
        """Model capability registry (``ModelCapabilityRegistry``)."""
        return self._model_capabilities

    @property
    def usage_tracker(self) -> "TurnUsageTracker":
        """Current-turn usage accumulator."""
        return self._usage_tracker

    @property
    def turn_count(self) -> int:
        """Number of completed user turns in this session."""
        return self._session_turn_count

    @property
    def context_token_count(self) -> int:
        """Estimated provider-visible prompt tokens from the most recent API call."""
        return self._last_context_tokens

    @property
    def context_budget_snapshot(self) -> dict[str, Any]:
        """Last prompt-budget snapshot for status/daemon metadata."""
        return dict(self._last_context_snapshot)

    def _active_context_length(self) -> int:
        """Return the runtime context budget for the active model/provider.

        ``llm_context_length`` is the *configured budget* and the registry holds
        *model capability*, so the effective window is the smaller of the two —
        but only when the registry entry actually describes this model. A
        family-wide entry carries whatever the vendor's line supported when it
        was written, and clamping to it silently shrank the window for every
        later generation (a 1M-class model matching "qwen" ran on 131K, i.e. 13%
        of its window, with every compression ratio computed against that wrong
        denominator). Model names always outrun a static table, so a
        non-authoritative match defers to the configured budget instead.

        Overshooting a model's real limit is recoverable: the provider reports
        overflow and recovery routes it to context compression. Silently running
        at a fraction of the window is not — nothing surfaces it.

        When the LLM backend is a FailoverChain, ``context_length`` reflects
        the *active* provider's declared window — which may be smaller than
        the primary's after a failover.  The live chain value is folded into
        the budget so post-failover turns compress against the right limit.
        """
        budget = max(1, int(getattr(self._settings, "llm_context_length", 0) or 1))

        # Chain-aware: FailoverChain.context_length tracks the active provider.
        llm_backend = getattr(self, "_llm", None)
        chain_cl = getattr(llm_backend, "context_length", None) if llm_backend is not None else None
        if chain_cl is not None:
            budget = min(budget, max(1, int(chain_cl)))

        # Use the active model name for capability lookup when the chain
        # exposes it, so a failover to a different model resolves the right
        # registry entry instead of the primary's.
        active_model = (
            getattr(llm_backend, "model", None) if llm_backend is not None else None
        ) or self._settings.llm_model

        if self._model_capabilities is None:
            return budget
        try:
            caps = self._model_capabilities.resolve(active_model)
        except Exception:
            logger.debug("model capability lookup failed", exc_info=True)
            return budget
        if not getattr(caps, "authoritative", True):
            return budget
        known = max(1, int(getattr(caps, "context_length", 0) or 1))
        return min(budget, known)

    @property
    def active_context_length(self) -> int:
        """Effective context budget in use, for status reporting.

        Exposed so clients report the window compression actually runs against.
        Reading ``settings.llm_context_length`` instead shows the configured
        budget, which differs whenever an authoritative capability caps it — the
        status bar then claims a window the engine is not using.
        """
        return self._active_context_length()

    def _focus_turn_id(self) -> int:
        """Return a stable monotonic turn id for focus observations."""
        try:
            return int(self._session_turn_count)
        except (TypeError, ValueError):
            return 0

    def focus_view(self) -> dict[str, Any]:
        """Return read-only semantic focus diagnostics for /orient and tests."""
        data = self._focus_state.summary()
        data["last_reference_resolution"] = (
            self._last_reference_resolution.to_dict()
            if self._last_reference_resolution is not None
            else None
        )
        return data

    def recalibrate_difficulty(self, store: Any) -> Any:
        """S3-L3: apply offline calibration (S3-L2) to the difficulty weight."""
        return self._calibration_manager.recalibrate_difficulty(store)

    def reset_calibration(self) -> None:
        """Revert any applied difficulty calibration to the configured baseline."""
        self._calibration_manager.reset_calibration()

    def recalibrate_thresholds(self, store: Any) -> Any:
        """S3-L4: tune the finalize posture threshold from stored signals."""
        return self._calibration_manager.recalibrate_thresholds(store)

    def reset_threshold_calibration(self) -> None:
        """Revert any applied finalize-threshold calibration to the baseline."""
        self._calibration_manager.reset_threshold_calibration()

    def set_calibration_store(self, store: Any) -> None:
        """Install the skill episode store used for periodic calibration input."""
        self._calibration_store = store

    def set_calibration_event_store(self, store: Any) -> None:
        """Install the append-only audit sink for applied calibration decisions."""
        self._calibration_event_store = store

    def _update_progress_and_stall(self, frame: AgentLoopFrame) -> None:
        """Advance the frame's stall counter: reset on progress, else increment."""
        marker = self._calibration_manager._task_progress_marker()
        if marker == frame.progress_marker:
            frame.stalled_rounds += 1
        else:
            frame.stalled_rounds = 0
            frame.progress_marker = marker
            # Genuine progress: re-arm content-level recovery one-shots so a long
            # task can recover again later (e.g. multiple max_tokens continuations
            # or force-compressions across a long turn). Storm-prone infrastructure
            # one-shots stay strict (bounded by the RecoveryBudget instead).
            if frame.recovery is not None:
                frame.recovery.rearm_after_progress()

    def _within_resource_limits(self) -> bool:
        """Whether real resource budgets (cost) still allow continuation.

        The absolute iteration hard cap is enforced by the budget itself; this
        guards the *cost* ceiling when configured (0 disables). Context pressure
        is handled separately by the finalizing posture.
        """
        multiple = float(getattr(self._settings, "agent_cost_ceiling_context_multiple", 0.0) or 0.0)
        if multiple <= 0:
            return True
        effective = self._usage_tracker.summary().effective_prompt_tokens()
        ceiling = multiple * float(self._active_context_length() or 0)
        return ceiling <= 0 or effective < ceiling

    def _should_extend_budget(self, frame: AgentLoopFrame) -> bool:
        """Progress-gated continuation decision (P0).

        Extend the iteration budget past the elastic ceiling only when the task
        is *productively unfinished*: within resource limits, still making
        progress (not stalled), and not already signalled complete by the ledger
        (zero open questions). A stalled, complete, or resource-exhausted task is
        allowed to converge and stop — so a productive long task continues while a
        spinning one halts.
        """
        if not self._within_resource_limits():
            return False
        stall_rounds = int(getattr(self._settings, "agent_stall_rounds", 6) or 6)
        if frame.stalled_rounds >= stall_rounds:
            return False
        open_q = self._ledger_open_questions()
        if open_q is not None and open_q == 0:
            return False
        return True

    def _ledger_open_questions(self) -> int | None:
        """Ledger sufficiency signal for convergence: None when the ledger is
        inactive/empty (fall back to the marginal heuristic), else the current
        open-question count. A positive count suppresses early answer-ready
        convergence so a long task with tracked open work is never cut short.
        """
        ledger = self._research_ledger
        return None if ledger.is_empty else ledger.open_question_count

    def orientation_view(self, *, now: Optional[float] = None) -> Any:
        """Read-only unified orientation across immediate/working/long-term layers (S4-D1).

        Observe-only aggregation of existing state: the current research ledger
        forms the working layer (findings / open questions / next step). Changes
        no state; usable by dashboards, diagnostics, and future autonomy phases.
        """
        from leapflow.world_model.orientation import build_orientation_from_ledger

        return build_orientation_from_ledger(
            self._research_ledger.to_state(),
            now=now if now is not None else time.time(),
        )

    def _persist_research_ledger(self) -> None:
        """Persist the ledger for the active session (best-effort; S1 durable Orient).

        Fired as the ledger change-listener after each note. No-op when no store
        is wired or no session is established yet.
        """
        store = self._research_ledger_store
        session_id = self._current_session_id
        if store is None or not session_id:
            return
        store.save(session_id, self._research_ledger.to_state())

    def _schedule_reentry(
        self,
        *,
        kind: str = "time",
        reason: str = "",
        delay_seconds: Any = 0.0,
        event_match: Any = None,
        max_reentries: Any = 1,
        deadline_seconds: Any = 0.0,
    ) -> Dict[str, Any]:
        """Register a re-entry trigger seeded with the current orientation (S2 N2).

        Gated by ``agent_reentry_enabled`` (default off). Only persists a trigger
        (Orient snapshot = research-ledger state + task contract + reason); the
        actual wake-up dispatch is a later phase (N3+).
        """
        if not getattr(self._settings, "agent_reentry_enabled", False):
            return {"ok": False, "error": "re-entry is disabled (set agent.reentry_enabled=true)"}
        if self._reentry_store is None:
            return {"ok": False, "error": "re-entry store not configured"}
        contract = self._current_task_contract
        task_id = contract.task_id if contract else (self._current_session_id or "task")
        try:
            trigger = build_reentry_trigger(
                task_id=task_id,
                session_id=self._current_session_id or "",
                ledger_state=self._research_ledger.to_state(),
                task_contract=asdict(contract) if contract else {},
                continuation_summary=reason,
                kind=kind,
                delay_seconds=float(delay_seconds or 0.0),
                event_match=dict(event_match or {}),
                max_reentries=int(max_reentries or 1),
                deadline_seconds=float(deadline_seconds or 0.0),
            )
            self._reentry_store.save(trigger)
        except Exception as exc:
            return {"ok": False, "error": f"failed to schedule re-entry: {exc}"}
        return {
            "ok": True,
            "trigger_id": trigger.trigger_id,
            "kind": trigger.kind,
            "due_at": trigger.due_at,
            "note": "registered; wake-up dispatch activates in a later phase",
        }

    @staticmethod
    def _tool_def_name(tool_def: Any) -> str:
        """Extract the tool name from an OpenAI-style tool definition, else ''."""
        if not isinstance(tool_def, dict):
            return ""
        fn = tool_def.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name", "") or "")
        return str(tool_def.get("name", "") or "")

    @staticmethod
    def _safe_tools_json(tool_definitions: Any) -> str:
        """Serialize tool definitions to a JSON string, degrading to '' on error."""
        try:
            return json.dumps(list(tool_definitions or ()), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _parse_tool_schema(schema_json: Optional[str]) -> List[Dict[str, Any]]:
        """Parse a persisted tool-schema JSON string into a list, else empty."""
        if not schema_json:
            return []
        try:
            parsed = json.loads(schema_json)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, list):
            return [td for td in parsed if isinstance(td, dict)]
        return []

    def _tools_kwarg_with_cache_marker(self, tools_kwarg: Dict[str, Any]) -> Dict[str, Any]:
        """Return a tools kwarg with the committed tool-cache marker applied.

        Cold-path helper invoked once per round right before ``achat``. Only the
        Anthropic strategy supports a frozen tool-array cache breakpoint and only
        when the boundary is ``COMMITTED``; every other case returns the kwarg
        unchanged (byte-identical to before). The marker is applied to a copy, so
        the caller's ``tools_kwarg`` (which may still be mutated by mid-turn tool
        expansion) is never touched.
        """
        tools = tools_kwarg.get("tools")
        if not tools or self._current_cache_boundary is not CacheBoundary.COMMITTED:
            return tools_kwarg
        if not isinstance(self._cache_strategy, AnthropicCacheStrategy):
            return tools_kwarg
        marked = AnthropicCacheStrategy._apply_tool_cache_marker(
            tools, self._current_cache_boundary
        )
        return {**tools_kwarg, "tools": marked}

    def _recovery_audit_path(self) -> Any:
        """Return the profile-owned path for the recovery audit trail, if declared.

        Recovery decisions are the only record of why a turn stopped, and an
        in-memory sink loses them with the turn — which is how an incident became
        undiagnosable after the fact. Falls back to ``None`` (memory only) when no
        profile layout is available, so an embedded engine still works.
        """
        layout = getattr(self._settings, "profile_layout", None)
        path = getattr(layout, "audit_log_path", None)
        return path

    def _record_llm_call_telemetry(self, resp: Any, *, recovery: Any = None) -> None:
        """Record usage and budget calibration for a provider call that succeeded.

        Deliberately self-contained: this is bookkeeping, and a defect in it must
        never be mistaken for a provider failure. It used to run inside the
        provider call's ``try`` block, where one mistyped attribute name became an
        ``AttributeError`` that the LLM classifier read as a context overflow (the
        name contained "context"), sending every round through compression,
        failover, and credential rotation before halting the turn with an
        unrelated message. Telemetry now fails loudly in the log and silently to
        the turn.

        The success signal is recorded first so a bookkeeping defect cannot also
        cost the loop its error-counter reset.
        """
        try:
            if recovery is not None:
                recovery.record_api_success()
            usage = getattr(resp, "usage", None) or {}
            self._usage_tracker.record_api_call(
                usage,
                provider=getattr(self._llm, "active_provider_name", ""),
                model=getattr(resp, "model", "") or "",
            )
            if int(usage.get("prompt_tokens", 0) or 0) > 0:
                self._record_provider_usage(getattr(resp, "model", "") or "", usage)
        except Exception:  # noqa: BLE001 - telemetry must never fail a turn
            logger.warning("llm call telemetry recording failed", exc_info=True)

    def _record_provider_usage(self, model: str, usage: Dict[str, Any]) -> None:
        """Prefer provider prompt usage when available and learn observed limits."""
        provider_prompt = int(usage.get("prompt_tokens", 0) or 0)
        if provider_prompt > 0:
            # Calibrate before overwriting: the snapshot still holds this turn's
            # estimate, so the pair (estimate, actual) is the only signal that can
            # correct the character heuristic for this model and language mix.
            self._calibrate_budget_estimator(provider_prompt)
            self._last_context_tokens = provider_prompt
            self._last_context_snapshot = {
                **self._last_context_snapshot,
                "provider_prompt_tokens": provider_prompt,
                "total_tokens": provider_prompt,
                "ratio": provider_prompt
                / max(1, int(self._last_context_snapshot.get("context_length") or 1)),
            }
        if self._model_capabilities and model and usage:
            self._model_capabilities.update_from_usage(model, usage)

    def _calibrate_budget_estimator(self, provider_prompt: int) -> None:
        """Feed this turn's (estimate, actual) pair to the budget estimator."""
        snapshot = self._last_context_snapshot or {}
        # Skip a snapshot already replaced by a provider count, otherwise the
        # estimator would calibrate against its own previous observation.
        if snapshot.get("provider_prompt_tokens"):
            return
        estimated = int(snapshot.get("total_tokens", 0) or 0)
        if estimated <= 0:
            return
        estimator = getattr(self._context_controller, "estimator", None)
        observe = getattr(estimator, "observe_actual", None)
        if observe is None:
            return
        try:
            observe(estimated=estimated, actual=provider_prompt)
        except Exception:  # noqa: BLE001 - calibration must never break a turn
            logger.debug("budget estimator calibration failed", exc_info=True)

    async def run(self, user_text: str, *, enable_thinking: bool = False) -> str:
        """Entrypoint: simplified routing with unified tool loop as default path."""
        self._session_turn_count += 1
        logger.info("audit.user_input chars=%s", len(user_text))
        self._prompt_assembler._begin_turn_context(user_text)
        self._learning_bridge._emit_chat_event("user_message", {"content": user_text[:500]})

        # 1. Slash command (skill injection — zero-ambiguity activation)
        if user_text.startswith("/") and self._skill_injector:
            self._skill_dispatcher._inject_pending_skill_reminder()
            self._wm.remember_chat(build_user_message_text(user_text))
            logger.debug("route.slash command=%s", user_text.split()[0])
            return await self._unified_tool_loop(user_text, enable_thinking=enable_thinking)

        self._skill_dispatcher._inject_pending_skill_reminder()
        self._wm.remember_chat(build_user_message_text(user_text))

        # 2. Teach command (special session mode switch)
        if self._skill_dispatcher._is_teach_command(user_text):
            return await self._skill_dispatcher._handle_learn_command(user_text)

        # 3. Everything else → unified tool loop (LLM decides tools vs direct response)
        logger.debug("route.unified user_text_len=%d", len(user_text))
        if not self._settings.has_llm_credentials:
            msg = self._error_classifier.friendly_message(ErrorCategory.AUTH_PERMANENT)
            self._wm.remember_chat(build_assistant_message(msg))
            return msg
        return await self._unified_tool_loop(user_text, enable_thinking=enable_thinking)

    async def run_stream(
        self, user_text: str, *, enable_thinking: bool = False, request_id: str = ""
    ) -> AsyncIterator[Union[str, StreamEvent]]:
        """Like run(), but yields text chunks for streamable responses.

        Yields:
            str: legacy plain-text chunks (teach commands).
            StreamEvent(type="chunk"): real-time token fragments.
            StreamEvent(type="final"): complete assembled response.
            StreamEvent(type="tool_call"): internal tool invocation (suppress display).
        """
        self._session_turn_count += 1
        self._current_request_id = request_id
        logger.info("audit.user_input chars=%s", len(user_text))
        self._prompt_assembler._begin_turn_context(user_text)
        self._learning_bridge._emit_chat_event("user_message", {"content": user_text[:500]})

        # 1. Slash command (skill injection)
        if user_text.startswith("/") and self._skill_injector:
            self._skill_dispatcher._inject_pending_skill_reminder()
            self._wm.remember_chat(build_user_message_text(user_text))
            logger.debug("route.slash command=%s", user_text.split()[0])
            async for event in self._stream_via_sink(
                user_text, enable_thinking=enable_thinking
            ):
                yield event
            return

        self._skill_dispatcher._inject_pending_skill_reminder()
        self._wm.remember_chat(build_user_message_text(user_text))

        # 2. Teach command (special session mode switch)
        if self._skill_dispatcher._is_teach_command(user_text):
            result = await self._skill_dispatcher._handle_learn_command(user_text)
            yield result
            return

        # 3. Everything else → unified tool loop (streaming)
        logger.debug("route.unified user_text_len=%d", len(user_text))
        if not self._settings.has_llm_credentials:
            msg = self._error_classifier.friendly_message(ErrorCategory.AUTH_PERMANENT)
            self._wm.remember_chat(build_assistant_message(msg))
            yield StreamEvent(type="final", content=msg)
            return
        async for event in self._stream_via_sink(
            user_text, enable_thinking=enable_thinking
        ):
            yield event

    async def _stream_via_sink(
        self, user_text: str, *, enable_thinking: bool = False
    ) -> AsyncIterator[StreamEvent]:
        """Bridge: run the unified loop with a StreamSink and yield events.

        Creates a ``StreamSink`` backed by an ``asyncio.Queue``, kicks the
        unified ``_run_agent_loop`` off as a background task (push side),
        and yields ``StreamEvent`` objects from the queue (pull side).
        """
        sink = StreamSink()
        frame = self._build_root_frame(user_text, enable_thinking=enable_thinking)

        loop_error: Optional[BaseException] = None

        async def _run_loop() -> None:
            nonlocal loop_error
            try:
                await self._run_agent_loop(frame, sink=sink)
            except BaseException as exc:
                loop_error = exc
                try:
                    await sink.emit_error(str(exc))
                except Exception:
                    pass  # Sink might already be closed
            finally:
                await sink.close()

        task = asyncio.create_task(_run_loop())
        try:
            async for event in sink:
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            else:
                # Retrieve the task result to surface unexpected exceptions
                try:
                    task.result()
                except (asyncio.CancelledError, Exception):
                    pass

    # ── Unified Tool Loop (chat scenarios) ───────────────────────────────

    def _new_compressor(self) -> ContextCompressor:
        """Fresh context compressor (per engine, or per isolated child frame).

        The summarization callback is routed through the dedicated compression
        provider when one is configured (``self._compression_provider``),
        falling back to the primary LLM otherwise. Routing compression to a
        separate provider keeps the main conversation's cache prefix intact:
        an interleaved compression call on the primary provider would otherwise
        break the byte-stable prefix the prefix cache depends on.
        """
        ctx_len = self._settings.llm_context_length
        return ContextCompressor(
            CompressorConfig(
                token_budget=max(1, int(ctx_len * self._settings.context_hard_limit_ratio)),
                context_length=ctx_len,
                threshold=self._settings.compress_threshold,
                keep_tail=self._settings.compress_keep_tail,
                max_output_chars=self._settings.max_tool_output_chars,
                summarize_fn=self._make_compression_summarize_fn(),
                protect_first_n=self._settings.compression_protect_first_n,
                summarize_keep_recent=self._settings.compression_keep_recent_n,
            )
        )

    def _make_compression_summarize_fn(self) -> Any:
        """Build a summarize callback that prefers the dedicated compression provider.

        Returns ``None`` when no provider is available (neither a dedicated
        compression provider nor a primary LLM), so the compressor degrades to
        its deterministic non-LLM fallback rather than crashing. The provider is
        resolved lazily at call time so a compression provider built after the
        compressor still takes effect.
        """
        async def _summarize(prompt: str) -> str:
            provider = self._compression_provider or self._llm
            if provider is None:
                return ""
            resp = await provider.achat(
                [build_user_message_text(prompt)],
                stream=False,
                enable_thinking=False,
            )
            return (getattr(resp, "content", "") or "").strip()

        return _summarize

    def _build_compression_provider(self) -> Optional[LLMProvider]:
        """Build a dedicated LLM provider for context compression, or ``None``.

        Activates only when ``compression_provider`` or ``compression_model`` is
        configured. Empty ``compression_*`` fields fall back to the primary
        LLM's corresponding ``llm_*`` configuration, so a partial configuration
        (e.g. only a cheaper model on the same endpoint) is valid. Constructs an
        ``OpenAIChat`` provider directly, mirroring how the primary LLM is built
        (the primary is an ``OpenAIChat`` wired in the CLI context), so the
        compression endpoint speaks the same OpenAI-compatible protocol.

        Returns ``None`` when unconfigured (the common path) so behaviour is
        byte-identical to before. A construction failure degrades to ``None``
        (compression then uses the primary provider) rather than crashing engine
        construction — an auxiliary provider must never fail a turn.
        """
        settings = self._settings
        provider_name = str(getattr(settings, "compression_provider", "") or "").strip()
        model = str(getattr(settings, "compression_model", "") or "").strip()
        if not provider_name and not model:
            return None
        api_key = str(getattr(settings, "compression_api_key", "") or "").strip()
        base_url = str(getattr(settings, "compression_base_url", "") or "").strip()
        # Empty compression_* fields fall back to the primary LLM configuration.
        # Credentials are already resolved from ``secret://`` refs at config-load
        # time (config_loader), so they are used verbatim here just like the
        # primary provider does with ``settings.llm_api_key``. Primary provider
        # behavior is inferred from its OpenAI-compatible base URL; there is no
        # separate ``llm.provider`` setting.
        effective_provider = provider_name
        effective_model = model or str(getattr(settings, "llm_model", "") or "")
        effective_api_key = api_key or str(getattr(settings, "llm_api_key", "") or "")
        effective_base_url = base_url or str(getattr(settings, "llm_base_url", "") or "")
        if not effective_api_key or not effective_base_url or not effective_model:
            logger.debug(
                "compression provider not built: incomplete config "
                "(model=%s base_url set=%s api_key set=%s)",
                effective_model, bool(effective_base_url), bool(effective_api_key),
            )
            return None
        try:
            from leapflow.llm.openai_provider import OpenAIChat

            provider = OpenAIChat(
                api_key=effective_api_key,
                base_url=effective_base_url,
                model=effective_model,
                max_retries=int(getattr(settings, "llm_max_retries", 3) or 3),
                provider=effective_provider or None,
            )
            logger.info(
                "compression provider built: model=%s (independent of primary)",
                effective_model,
            )
            return provider
        except (ImportError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "compression provider construction failed; using primary provider: %s", exc
            )
            return None

    def _new_governance(self) -> ContextGovernanceController:
        """Fresh context-governance controller (per engine, or per child frame)."""
        ctx_len = self._settings.llm_context_length
        return ContextGovernanceController(
            evidence_builder=ToolEvidenceBuilder(
                max_content_chars=self._settings.tool_evidence_max_chars,
                context_length=ctx_len,
            ),
            repeated_read_limit=self._settings.repeated_read_limit,
            convergence_round=self._settings.long_task_convergence_round,
            convergence_round_ceiling=self._settings.convergence_round_ceiling,
            convergence_scale=self._settings.convergence_scale,
            checkpoint_interval=self._settings.agent_checkpoint_interval,
            posture_config=ContextPostureConfig(
                expanded_ratio=self._settings.context_expanded_ratio,
                finalizing_ratio=(
                    getattr(self, "_calibrated_finalizing_ratio", None)
                    or self._settings.context_finalizing_ratio
                ),
                expanded_evidence_threshold=self._settings.context_expanded_evidence_threshold,
                expanded_tool_call_threshold=self._settings.context_expanded_tool_call_threshold,
                research_source_threshold=self._settings.context_research_source_threshold,
                research_evidence_threshold=self._settings.context_research_evidence_threshold,
            ),
        )

    def _new_usage_tracker(self) -> TurnUsageTracker:
        """Fresh TurnUsageTracker with plugin learning sink wired."""
        tracker = TurnUsageTracker()
        try:
            from leapflow.engine.session.session_factory import _wire_plugin_stats_sink

            _wire_plugin_stats_sink(tracker)
        except (ImportError, RuntimeError, AttributeError):
            pass
        return tracker

    def _build_child_frame(
        self,
        user_text: str,
        *,
        depth: int,
        tool_filter: "frozenset[str] | None" = None,
        enable_thinking: bool = False,
        parent_session_id: Optional[str] = None,
    ) -> AgentLoopFrame:
        """Build an isolated child frame with fresh per-turn subsystems.

        A recursive subagent runs the same ``_run_agent_loop`` on this frame; the
        fresh budget/recovery/governance/ledger/commitment/usage/compressor keep
        its OODA loop from contaminating the parent frame's state.
        """
        return AgentLoopFrame(
            user_text=user_text,
            depth=depth,
            budget=IterationBudget.for_react(self._budget_config),
            recovery=TurnRecoveryState(),
            governance=self._new_governance(),
            ledger=ResearchLedger(),
            commitment=PrefixCommitmentController(),
            usage_tracker=self._new_usage_tracker(),
            compressor=self._new_compressor(),
            tool_filter=tool_filter,
            enable_thinking=enable_thinking,
            parent_session_id=parent_session_id,
        )

    def _install_frame(self, frame: AgentLoopFrame) -> Dict[str, Any]:
        """Install a frame's per-turn subsystems as the engine's active state.

        Returns the previous per-turn state for restoration. This lets a child
        frame run the full loop on the shared engine while the parent frame's
        subsystems stay untouched (see ``_run_child_frame``).
        """
        saved: Dict[str, Any] = {
            "governance": self._context_governance_controller,
            "ledger": self._research_ledger,
            "commitment": self._prefix_commitment,
            "usage": self._usage_tracker,
            "compressor": self._compressor,
            "coordinator": self._recovery_coordinator,
            "snapshot": self._last_context_snapshot,
            "categories": self._last_turn_tool_categories,
            # Per-turn identity is turn-start-set and non-propagating, so it is
            # scoped to the frame (a child that reassigns it must not leak to the
            # parent). NOTE: _cancel_requested is deliberately NOT saved here — it
            # is a cross-frame signal that must propagate into a running child.
            "session_id": self._current_session_id,
            "turn_id": self._current_turn_id,
            "command_id": self._current_command_id,
            "frame": self._active_frame,
        }
        self._context_governance_controller = frame.governance
        self._research_ledger = frame.ledger
        self._prefix_commitment = frame.commitment
        self._usage_tracker = frame.usage_tracker
        self._compressor = frame.compressor
        if frame.recovery_coordinator is not None:
            self._recovery_coordinator = frame.recovery_coordinator
        self._last_context_snapshot = frame.last_context_snapshot
        self._last_turn_tool_categories = frame.last_turn_tool_categories
        self._active_frame = frame
        return saved

    def _restore_per_turn_state(self, saved: Dict[str, Any]) -> None:
        """Restore per-turn state previously saved by ``_install_frame``."""
        self._context_governance_controller = saved["governance"]
        self._research_ledger = saved["ledger"]
        self._prefix_commitment = saved["commitment"]
        self._usage_tracker = saved["usage"]
        self._compressor = saved["compressor"]
        self._recovery_coordinator = saved["coordinator"]
        self._last_context_snapshot = saved["snapshot"]
        self._last_turn_tool_categories = saved["categories"]
        self._current_session_id = saved["session_id"]
        self._current_turn_id = saved["turn_id"]
        self._current_command_id = saved["command_id"]
        self._active_frame = saved["frame"]

    async def _run_child_frame(self, frame: AgentLoopFrame) -> str:
        """Run a recursive subagent's isolated frame through the full loop.

        Swaps the engine's per-turn state to the child frame for the duration of
        the child loop, then restores the parent's state — so recursion is fully
        state-isolated without duplicating the loop body.
        """
        saved = self._install_frame(frame)
        try:
            return await self._run_agent_loop(frame)
        finally:
            self._restore_per_turn_state(saved)

    async def _run_subagent_goal(
        self,
        goal: str,
        *,
        depth: int,
        tool_filter: "frozenset[str] | None" = None,
        enable_thinking: bool = False,
    ) -> str:
        """Run a subagent goal as an isolated child frame through the full loop.

        Bridge for ``EngineFrameSubagentExecutor`` (opt-in full-loop subagents):
        the child frame's fresh subsystems + per-frame swap keep the subagent
        from contaminating the parent turn's state.
        """
        frame = self._build_child_frame(
            goal,
            depth=depth,
            tool_filter=tool_filter,
            enable_thinking=enable_thinking,
        )
        return await self._run_child_frame(frame)

    def _build_frame(
        self,
        user_text: str,
        enable_thinking: bool,
        budget: Any,
        recovery: Any,
    ) -> AgentLoopFrame:
        """Bundle per-frame state around the given budget/recovery.

        The root frame wraps the engine's (freshly reset) per-turn subsystems so
        loop-path reads through ``self._active_frame`` are byte-equivalent to the
        singletons; recursive subagents (later) build frames with fresh subsystems.
        """
        return AgentLoopFrame(
            user_text=user_text,
            enable_thinking=enable_thinking,
            budget=budget,
            recovery=recovery,
            governance=self._context_governance_controller,
            ledger=self._research_ledger,
            commitment=self._prefix_commitment,
            usage_tracker=self._usage_tracker,
            compressor=self._compressor,
            session_id=self._current_session_id,
            turn_id=self._current_turn_id,
            command_id=self._current_command_id,
        )

    def _build_root_frame(self, user_text: str, *, enable_thinking: bool = False) -> AgentLoopFrame:
        """Build the top-level (depth-0) agent-loop frame for a turn."""
        return self._build_frame(
            user_text,
            enable_thinking,
            IterationBudget.for_react(self._budget_config),
            TurnRecoveryState(),
        )

    async def _unified_tool_loop(self, user_text: str, *, enable_thinking: bool = False) -> str:
        """Entry adapter: build the root frame and run the unified agent loop."""
        return await self._run_agent_loop(
            self._build_root_frame(user_text, enable_thinking=enable_thinking)
        )

    async def _run_agent_loop(
        self, frame: AgentLoopFrame, *, sink: Optional[OutputSink] = None
    ) -> str:
        """Unified adaptive OODA loop over an isolated per-frame state.

        Per-frame execution state (budget, recovery) lives on ``frame`` so the
        same loop serves the top-level turn (root frame) and recursive
        subagents (deeper frames with their own budget). Output delivery is
        abstracted behind ``sink``: a ``BufferSink`` for ``run()`` (returns
        text), a ``StreamSink`` for ``run_stream()`` (pushes ``StreamEvent``
        objects via an asyncio queue).

        Capabilities remain engine methods; the LLM dynamically decides
        tools vs direct reply.
        """
        if sink is None:
            sink = BufferSink()
        user_text = frame.user_text
        enable_thinking = frame.enable_thinking
        budget = frame.budget
        recovery = frame.recovery
        self._active_frame = frame

        # Detect slash command → inject skill context
        if user_text.startswith("/"):
            slash_name = user_text.split()[0][1:]  # Remove leading /
            remaining = user_text[len(slash_name) + 1 :].strip()
            if self._skill_injector:
                injection = self._skill_injector.build_injection_message(slash_name, remaining)
                if injection:
                    user_text = injection  # Replace user_text with skill injection

        # A restricted frame (e.g. a subagent) is offered only its permitted
        # tools; the root frame (tool_filter=None) sees the full registry,
        # including semantic desktop tools while perception is online.
        tool_defs = self._tool_dispatch._unified_tool_catalog()
        tool_handlers = self._tool_dispatch._unified_tool_handlers()
        if frame.tool_filter is not None:
            tool_defs = [
                td
                for td in tool_defs
                if td.get("function", {}).get("name", "") in frame.tool_filter
            ]
            tool_handlers = {
                name: fn for name, fn in tool_handlers.items() if name in frame.tool_filter
            }

        trace = ExecutionTrace()
        assembly = await self._prompt_assembler._assemble_unified_prompt(
            user_text,
            tool_definitions=tool_defs,
            enable_thinking=enable_thinking,
            slash_command=user_text.startswith("/"),
        )
        planned_enable_thinking = self._prompt_assembler._planned_enable_thinking(assembly.plan, enable_thinking)
        # Reset the Tier 1 continuity state now that this turn's plan has been
        # assembled from the *previous* turn's value; it accumulates fresh from
        # this turn's own tool_calls for the *next* turn's plan.
        self._last_turn_tool_categories = frozenset()

        messages: List[Dict[str, Any]] = [build_system_message(assembly.system)]
        if assembly.volatile_context:
            messages.append({
                "role": "system",
                "content": assembly.volatile_context,
                "_volatile_context": True,
            })
        messages.extend(assembly.prior_turns)
        messages.append(build_user_message_text(user_text))

        content = ""
        fatal_error: Optional[str] = None
        # P3: Initialize recovery coordinator for this turn
        recovery_budget = RecoveryBudget(
            turn_deadline_s=self._settings.recovery_turn_deadline_s,
            total_recovery_actions=self._settings.recovery_total_actions,
            max_retry_per_category=self._settings.recovery_max_retry_per_category,
        )
        recovery_budget.start_deadline()
        self._recovery_coordinator = RecoveryCoordinator(
            strategies=default_strategies(
                credential_availability=self._llm
                if hasattr(self._llm, "has_rotatable_credentials") else None,
            ),
            budget=recovery_budget,
        )
        self._recovery_coordinator.new_turn(turn_id=budget.used)
        use_native_tools = assembly.plan.native_tools
        result_budget = self._effective_tool_result_budget()
        unknown_tool_retry_used = False
        empty_response_retry_used = False
        self._usage_tracker.reset()

        tools_kwarg: Dict[str, Any] = self._prompt_assembler._planned_tools_kwarg(assembly.plan)

        self._cancel_requested = False
        _signal_watermark = [time.time()]

        session_id = self._session_persistence._ensure_session_for_frame(frame, user_text)

        # Prime per-turn guardrail baselines with the initial message state
        # (prior turns only) so that TurnCapGuard counts only calls added
        # during THIS turn, not the pre-existing prior-turn calls.
        if self._guardrail is not None:
            self._guardrail.check(messages)

        while not budget.exhausted:
            if self._cancel_requested:
                logger.info("unified_loop: cancelled by user")
                break

            status = budget.consume()
            if status == BudgetStatus.EXHAUSTED:
                # Progress-gated continuation: a productively-unfinished task
                # (open ledger work, still progressing, within resource limits)
                # extends past the elastic ceiling toward the hard cap instead of
                # terminating; a stalled/complete/over-budget task stops here.
                if budget.can_extend and self._should_extend_budget(frame):
                    budget.grant_extension(self._settings.agent_iter_extension_step)
                    if budget.status() == BudgetStatus.EXHAUSTED:
                        break  # absolute hard cap reached
                    logger.info(
                        "unified_loop: budget extended (progress-gated) to %d (stalled=%d)",
                        budget.effective_max,
                        frame.stalled_rounds,
                    )
                    status = budget.status()
                else:
                    break

            self._inject_live_signals(messages, _signal_watermark)

            healed = self._healer.heal(messages)
            compressed = self._prompt_assembler._prepare_llm_messages(
                healed,
                tools=tools_kwarg.get("tools") if use_native_tools else None,
                round_number=budget.used,
                defer_cache_optimization=True,
            )
            self._calibration_manager._widen_budget_for_difficulty(budget)
            self._update_progress_and_stall(frame)
            self._calibration_manager._evaluate_prefix_commitment(budget)
            # PCD 2d: a posture upgrade or slash injection disrupts the frozen
            # prefix, so break enforcement and resume normal PCD next round.
            _posture_now = str(self._last_context_snapshot.get("context_posture") or "baseline")
            self._calibration_manager._maybe_break_commitment(
                posture_changed=_posture_now != self._prev_context_posture,
                slash_command=user_text.startswith("/"),
            )
            self._prev_context_posture = _posture_now
            # Apply markers only after this round's commitment evaluation (and
            # any same-round break), eliminating the first-commit boundary skew.
            compressed = self._prompt_assembler._apply_message_cache_strategy(compressed)

            # ── LLM call: native-tools path ─────────────────────────────
            if use_native_tools and tools_kwarg:
                try:
                    resp = await self._llm.achat(
                        compressed,
                        stream=False,
                        enable_thinking=planned_enable_thinking,
                        **self._tools_kwarg_with_cache_marker(tools_kwarg),
                    )
                except Exception as exc:
                    _clear_indicator()
                    classified = self._error_classifier.classify(exc)
                    category_str = classified.value if hasattr(classified, "value") else str(classified)
                    recovery.record_api_error(category_str)

                    # Classify through unified coordinator and execute recovery
                    envelope = self._unified_classifier.classify_llm_error(
                        exc,
                        provider=getattr(self._llm, "provider", ""),
                        model=getattr(self._llm, "model", ""),
                    )
                    logger.error(
                        "unified_loop: llm call failed (%s/%s)",
                        envelope.category,
                        envelope.failure_code,
                        exc_info=True,
                    )
                    _recovery_break, _recovery_updates = await self._handle_llm_recovery(
                        envelope, recovery, budget, messages, tools_kwarg,
                        use_native_tools, planned_enable_thinking, sink,
                    )
                    if _recovery_break == "continue":
                        use_native_tools = _recovery_updates.get("use_native_tools", use_native_tools)
                        planned_enable_thinking = _recovery_updates.get("planned_enable_thinking", planned_enable_thinking)
                        tools_kwarg = _recovery_updates.get("tools_kwarg", tools_kwarg)
                        continue
                    elif _recovery_break == "fatal":
                        fatal_error = _recovery_updates.get("fatal_error", "")
                        break
                    break  # "break" sentinel
                _clear_indicator()
                self._record_llm_call_telemetry(resp, recovery=recovery)

                content = (resp.content or "").strip()
                if self._sanitizer:
                    content = self._sanitizer.sanitize(content)

                # Surface provider reasoning/thinking to sink
                thinking = getattr(resp, "thinking_content", None)
                if thinking and thinking.strip():
                    await sink.emit_thinking(thinking.strip())

                # Length continuation
                finish = getattr(resp, "finish_reason", None)
                if finish in ("length", "max_tokens") and recovery.try_length_continuation():
                    logger.info("unified_loop: length continuation (finish_reason=%s)", finish)
                    messages.append(build_assistant_message(content))
                    messages.append(build_user_message_text(build_continuation_prompt(content)))
                    continue

                native_calls = getattr(resp, "tool_calls", None) or []
                if native_calls:
                    # Surface pre-tool-call reasoning as thinking
                    if content:
                        await sink.emit_thinking(content)
                    content = ""
                    assistant_msg = _build_native_tool_assistant_message(
                        native_calls,
                        thinking_content=thinking,
                    )
                    messages.append(assistant_msg)
                    self._session_persistence._persist_message(
                        session_id, "assistant", "", tool_calls=assistant_msg.get("tool_calls")
                    )

                    # Emit tool_start events
                    for tc in native_calls:
                        resolved_call = _normalize_tool_call(
                            {"name": tc.name, "arguments": tc.arguments}
                        )
                        normalized_name = str(resolved_call["name"])
                        original_name = str(resolved_call.get("original_tool_name") or tc.name)
                        await sink.emit_tool_start(
                            normalized_name,
                            _tool_args_metadata(
                                normalized_name,
                                tc.arguments,
                                original_tool_name=original_name,
                                tool_call_id=str(tc.id),
                            ),
                        )

                    results = await self._tool_dispatch._execute_tools_concurrent(
                        native_calls,
                        tool_handlers,
                        trace=trace,
                        messages=messages,
                    )
                    self._prompt_assembler._record_tool_call_categories(native_calls)
                    self._learning_bridge._observe_capability_results(results)
                    tools_kwarg = self._tool_dispatch._merge_expanded_tool_schemas(tools_kwarg, results)

                    # Emit tool_complete events
                    result_by_id = {str(item.get("id")): item for item in results}
                    for tc in native_calls:
                        item = result_by_id.get(str(tc.id), {})
                        normalized_name = str(item.get("name") or _normalize_tool_name(tc.name))
                        original_name = str(item.get("original_tool_name") or tc.name)
                        await sink.emit_tool_complete(
                            normalized_name,
                            {
                                **_tool_result_metadata(
                                    normalized_name,
                                    tc.arguments,
                                    item.get("result"),
                                    original_tool_name=original_name,
                                    tool_call_id=str(tc.id),
                                ),
                                **self._tool_dispatch._tool_context_metadata(
                                    normalized_name, tc.arguments, item.get("result")
                                ),
                            },
                        )

                    permission_hard_stop = _permission_hard_stop_from_results(results)
                    if permission_hard_stop:
                        logger.info(
                            "unified_loop: permission hard-stop after %s/%s",
                            permission_hard_stop.get("platform", "platform"),
                            permission_hard_stop.get("capability")
                            or permission_hard_stop.get("action")
                            or "action",
                        )
                        break

                    retryable_unknown = next(
                        (
                            item.get("result")
                            for item in results
                            if _is_retryable_unknown_tool_result(item.get("result"))
                        ),
                        None,
                    )
                    if retryable_unknown and not unknown_tool_retry_used:
                        unknown_tool_retry_used = True
                        # PCD 2d: the frozen tool subset proved insufficient; break
                        # enforcement before escalating to the full catalog.
                        self._calibration_manager._maybe_break_commitment(tool_error=True)
                        tools_kwarg = self._tool_dispatch._expand_tools_kwarg_full(tools_kwarg, tool_defs)
                        use_native_tools = bool(tools_kwarg)
                        messages.append(
                            build_user_message_text(_unknown_tool_retry_prompt(retryable_unknown))
                        )
                        continue

                    halt_reason = self._tool_dispatch._evaluate_tool_failures(
                        [
                            (item.get("name") or "", item["result"])
                            for item in results
                            if isinstance(item.get("result"), dict)
                            and _tool_result_counts_as_failure(item["result"])
                        ],
                        turn_id=budget.used,
                    )
                    if halt_reason:
                        fatal_error = halt_reason
                        break

                    # Guardrail check after tool execution
                    if self._tool_dispatch._check_guardrail(messages) == "halt":
                        break

                    self._wm.remember_chat(
                        build_assistant_message(
                            f"[Called: {', '.join(tc.name for tc in native_calls)}]"
                        )
                    )

                    if status == BudgetStatus.SOFT_LIMIT and not self._should_extend_budget(frame):
                        messages.append(
                            build_user_message_text(
                                "SYSTEM: Approaching limit. Provide final answer now."
                            )
                        )
                    elif _has_completed_side_effect(results):
                        messages.append(
                            build_user_message_text(
                                "SYSTEM: Side-effect action completed (result has completed:true). "
                                "Do not re-invoke it with the same parameters. "
                                "If all user-requested actions are done, provide the final answer."
                            )
                        )
                    continue
                # native_tools path but LLM returned text — fall through to text handling

            # ── LLM call: text path (streaming or non-streaming) ────────
            else:
                if sink.supports_streaming and self._settings.stream_output:
                    # Real-time streaming
                    content_parts: list[str] = []
                    try:
                        _clear_indicator()
                        raw_stream = self._llm.achat_stream(
                            compressed,
                            enable_thinking=planned_enable_thinking,
                        )
                        guarded = stale_guarded_stream(
                            raw_stream,
                            timeout_s=self._stale_stream_timeout_s,
                        )
                        async for chunk in guarded:
                            content_parts.append(chunk)
                            await sink.emit_chunk(chunk)
                        recovery.record_api_success()
                    except StaleStreamError as stale_exc:
                        _clear_indicator()
                        partial = stale_exc.partial_text or "".join(content_parts)
                        if partial.strip() and recovery.try_length_continuation():
                            logger.warning(
                                "stale_stream: recovering with %d chars partial", len(partial)
                            )
                            content = partial.strip()
                            messages.append(build_assistant_message(content))
                            messages.append(
                                build_user_message_text(build_continuation_prompt(content))
                            )
                            continue
                        await sink.emit_error(str(stale_exc))
                        break
                    except Exception as exc:
                        _clear_indicator()
                        classified = self._error_classifier.classify(exc)
                        category_str = classified.value if hasattr(classified, "value") else str(classified)
                        recovery.record_api_error(category_str)
                        envelope = self._unified_classifier.classify_llm_error(
                            exc,
                            provider=getattr(self._llm, "provider", ""),
                            model=getattr(self._llm, "model", ""),
                        )
                        logger.error(
                            "unified_loop: stream llm call failed (%s/%s)",
                            envelope.category,
                            envelope.failure_code,
                            exc_info=True,
                        )
                        _recovery_break, _recovery_updates = await self._handle_llm_recovery(
                            envelope, recovery, budget, messages, tools_kwarg,
                            use_native_tools, planned_enable_thinking, sink,
                        )
                        if _recovery_break == "continue":
                            use_native_tools = _recovery_updates.get("use_native_tools", use_native_tools)
                            planned_enable_thinking = _recovery_updates.get("planned_enable_thinking", planned_enable_thinking)
                            tools_kwarg = _recovery_updates.get("tools_kwarg", tools_kwarg)
                            continue
                        elif _recovery_break == "fatal":
                            fatal_error = _recovery_updates.get("fatal_error", "")
                            break
                        break

                    content = "".join(content_parts).strip()
                    if self._sanitizer:
                        content = self._sanitizer.sanitize(content)
                    # Streaming text path: achat_stream() yields only text
                    # chunks — no response object carries usage.  Record the
                    # API call so the tracker counts it; token counters stay
                    # at zero when the provider's stream omits usage data.
                    _stream_resp = types.SimpleNamespace(
                        usage=None,
                        model=getattr(self._llm, "model", ""),
                    )
                    self._record_llm_call_telemetry(
                        _stream_resp, recovery=recovery,
                    )
                else:
                    # Non-streaming text path
                    try:
                        resp = await self._llm.achat(
                            compressed,
                            stream=False,
                            enable_thinking=planned_enable_thinking,
                        )
                    except Exception as exc:
                        _clear_indicator()
                        classified = self._error_classifier.classify(exc)
                        category_str = classified.value if hasattr(classified, "value") else str(classified)
                        recovery.record_api_error(category_str)
                        envelope = self._unified_classifier.classify_llm_error(
                            exc,
                            provider=getattr(self._llm, "provider", ""),
                            model=getattr(self._llm, "model", ""),
                        )
                        logger.error(
                            "unified_loop: llm call failed (%s/%s)",
                            envelope.category,
                            envelope.failure_code,
                            exc_info=True,
                        )
                        _recovery_break, _recovery_updates = await self._handle_llm_recovery(
                            envelope, recovery, budget, messages, tools_kwarg,
                            use_native_tools, planned_enable_thinking, sink,
                        )
                        if _recovery_break == "continue":
                            use_native_tools = _recovery_updates.get("use_native_tools", use_native_tools)
                            planned_enable_thinking = _recovery_updates.get("planned_enable_thinking", planned_enable_thinking)
                            tools_kwarg = _recovery_updates.get("tools_kwarg", tools_kwarg)
                            continue
                        elif _recovery_break == "fatal":
                            fatal_error = _recovery_updates.get("fatal_error", "")
                            break
                        break
                    _clear_indicator()
                    self._record_llm_call_telemetry(resp, recovery=recovery)
                    content = (resp.content or "").strip()
                    if self._sanitizer:
                        content = self._sanitizer.sanitize(content)

                    # Surface provider reasoning/thinking to sink
                    thinking = getattr(resp, "thinking_content", None)
                    if thinking and thinking.strip():
                        await sink.emit_thinking(thinking.strip())

                    # Length continuation for non-stream path
                    finish = getattr(resp, "finish_reason", None)
                    if finish in ("length", "max_tokens") and recovery.try_length_continuation():
                        logger.info("unified_loop: length continuation (finish_reason=%s)", finish)
                        messages.append(build_assistant_message(content))
                        messages.append(build_user_message_text(build_continuation_prompt(content)))
                        continue

            # ── Text-mode tool handling (shared by all paths) ───────────
            self._session_persistence._persist_message(session_id, "assistant", content)
            # PCD 5b: snapshot the assembled prefix so a cache-priority resume
            # can reproduce it verbatim and hit the provider cache immediately.
            self._session_persistence._persist_session_snapshot(session_id)
            tool_call = self._tool_dispatch._parse_tool_call_from_content(content)

            if tool_call is None:
                if not content and not empty_response_retry_used:
                    # Empty successful response: treat as a transient failure and
                    # retry once with an explicit nudge.
                    empty_response_retry_used = True
                    logger.warning(
                        "unified_loop: empty LLM response "
                        "(model=%s provider=%s stream=%s); retrying once",
                        getattr(self._llm, "model", ""),
                        getattr(self._llm, "active_provider_name", "")
                        or getattr(self._llm, "provider", ""),
                        self._settings.stream_output,
                    )
                    messages.append(build_user_message_text(_EMPTY_RESPONSE_RETRY_PROMPT))
                    continue
                self._wm.remember_chat(build_assistant_message(content))
                trace.record(ExecutionMode.COMPLETE)
                break

            # Text-mode preamble exclusion: only store call summary in WM,
            # not the natural language preamble that surrounds the tool_call tag.
            normalized_tool_call = _normalize_tool_call(tool_call)
            tool_name = str(normalized_tool_call["name"])
            original_tool_name = str(normalized_tool_call.get("original_tool_name", tool_name))
            self._wm.remember_chat(build_assistant_message(f"[Called: {tool_name}]"))

            messages.append(build_assistant_message(content))
            tool_arguments = normalized_tool_call.get("arguments")
            self._learning_bridge._emit_chat_event(
                "tool_call",
                {
                    "tool_name": tool_name,
                    "arguments_summary": json.dumps(
                        tool_arguments, default=str, ensure_ascii=False
                    )[:300]
                    if tool_arguments
                    else "",
                },
            )
            await sink.emit_tool_start(
                tool_name,
                _tool_args_metadata(
                    tool_name,
                    tool_arguments,
                    original_tool_name=original_tool_name,
                ),
            )
            result = await self._tool_dispatch._execute_tool_with_ledger(
                normalized_tool_call,
                tool_handlers,
                tool_call_id=f"text-{budget.used}",
            )
            _clear_indicator()
            self._learning_bridge._emit_chat_event(
                "tool_result",
                {
                    "tool_name": tool_name,
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
                    "summary": json.dumps(result, default=str, ensure_ascii=False)[:300]
                    if isinstance(result, dict)
                    else str(result)[:300],
                },
            )
            await sink.emit_tool_complete(
                tool_name,
                {
                    **_tool_result_metadata(
                        tool_name,
                        tool_arguments,
                        result,
                        original_tool_name=original_tool_name,
                    ),
                    **self._tool_dispatch._tool_context_metadata(
                        tool_name,
                        tool_arguments,
                        result,
                    ),
                },
            )
            _print_tool_result(tool_name, result, enabled=self._settings.verbose_progress)
            trace.record(
                ExecutionMode.ACTING,
                action=normalized_tool_call,
                observation=result if isinstance(result, dict) else {"result": str(result)},
            )

            is_error = isinstance(result, dict) and _tool_result_counts_as_failure(result)
            if is_error:
                recovery.record_tool_failure()
            else:
                recovery.record_tool_success()
            self._learning_bridge._record_tool_focus(tool_name, tool_arguments, result)
            self._learning_bridge._observe_capability_result(result)
            result_payload = self._tool_dispatch._compact_tool_result(tool_name, tool_arguments, result)
            result_text = _truncate_result_for_budget(result_payload, result_budget)
            messages.append(build_user_message_text(f"Tool result ({tool_name}):\n{result_text}"))
            self._session_persistence._persist_message(
                session_id,
                "tool",
                result_text,
                tool_name=tool_name,
                tool_call_id=f"text-{budget.used}",
                metadata=self._tool_dispatch._tool_execution_metadata_with_focus(
                    tool_name, tool_arguments, result
                ),
            )

            if _is_permission_hard_stop_payload(result):
                logger.info(
                    "unified_loop: permission hard-stop after %s/%s",
                    result.get("platform", "platform"),
                    result.get("capability") or result.get("action") or tool_name,
                )
                break

            if _is_retryable_unknown_tool_result(result) and not unknown_tool_retry_used:
                unknown_tool_retry_used = True
                # PCD 2d: frozen tool subset insufficient; break enforcement.
                self._calibration_manager._maybe_break_commitment(tool_error=True)
                messages.append(build_user_message_text(_unknown_tool_retry_prompt(result)))
                continue

            if is_error:
                halt_reason = self._tool_dispatch._evaluate_tool_failures(
                    [(tool_name, result)], turn_id=budget.used
                )
                if halt_reason:
                    fatal_error = halt_reason
                    break

            if self._tool_dispatch._check_guardrail(messages) == "halt":
                break

            if status == BudgetStatus.SOFT_LIMIT and not self._should_extend_budget(
                self._active_frame
            ):
                messages.append(
                    build_user_message_text("SYSTEM: Approaching limit. Provide final answer now.")
                )

        # ── Post-loop finalization ──────────────────────────────────────
        # Turn-end learning/memory-sync are top-level-turn concerns; a recursive
        # child frame (subagent) must not pollute the parent's evolution/memory
        # (its result flows back via SubagentResult) nor leak background tasks.
        if (
            getattr(self._active_frame, "is_root", True)
            and self._memory_manager
            and self._settings.memory_integration_enabled
        ):
            asyncio.create_task(self._sync_turn_safe(messages))

        if getattr(self._active_frame, "is_root", True) and self._evolution is not None and content:
            asyncio.create_task(self._learning_bridge._post_turn_review(messages, content))

        llm = self._llm
        if hasattr(llm, "try_restore_primary"):
            llm.try_restore_primary()

        logger.info("turn_usage: %s", self._usage_tracker.format_log_line())

        if content:
            permission_override = _permission_override_message(messages)
            final = permission_override or content
            self._learning_bridge._emit_chat_event("response", {"content": final[:500]})
            await sink.emit_final(final)
            return final

        # The loop stopped without a written answer (repetition halt or
        # exhausted budget). Give the model one tool-free round to answer
        # from what it gathered before falling back to the canned notice;
        # a genuine terminal failure (fatal_error) still surfaces.
        if empty_response_retry_used:
            # Empty response persisted after retry — use degraded message
            logger.warning(
                "unified_loop: empty LLM response persisted after retry "
                "(model=%s); emitting transparent degraded message",
                getattr(self._llm, "model", ""),
            )
            fallback = (
                _app_onboarding_recovery_message(messages)
                or _EMPTY_RESPONSE_DEGRADED_MESSAGE
            )
        else:
            fallback = (
                _app_onboarding_recovery_message(messages)
                or _last_tool_failures_recovery_message(messages)
                or fatal_error
                or await self._synthesize_forced_answer(messages)
                or self._budget_exhausted_response(messages)
            )
        self._learning_bridge._emit_chat_event("response", {"content": fallback[:500]})
        await sink.emit_final(fallback)
        return fallback

    # ── LLM Recovery Helper ────────────────────────────────────────────

    async def _handle_llm_recovery(
        self,
        envelope: Any,
        recovery: TurnRecoveryState,
        budget: IterationBudget,
        messages: List[Dict[str, Any]],
        tools_kwarg: Dict[str, Any],
        use_native_tools: bool,
        planned_enable_thinking: bool,
        sink: OutputSink,
    ) -> tuple[str, Dict[str, Any]]:
        """Shared LLM-error recovery logic for the unified loop.

        Returns ``(action, updates)`` where *action* is one of
        ``"continue"`` (retry/transform succeeded — caller should ``continue``
        the while-loop), ``"fatal"`` (unrecoverable — caller should ``break``
        and use ``updates["fatal_error"]``), or ``"break"`` (terminal halt,
        error already emitted to *sink*).

        *updates* may contain ``tools_kwarg``, ``use_native_tools``, and
        ``planned_enable_thinking`` when a transform mutated them.
        """
        coordinator = self._recovery_coordinator
        try:
            decision = coordinator.evaluate(envelope)
        except Exception as coord_exc:
            logger.error("recovery_coordinator.evaluate() failed: %s", coord_exc)
            err_msg = f"Internal recovery error: {coord_exc}"
            await sink.emit_error(err_msg)
            return "fatal", {"fatal_error": err_msg}
        self._audit_sink.record(
            create_audit_entry(
                envelope,
                decision,
                coordinator.budget,
                session_id=getattr(self, "_current_session_id", "") or "",
                turn_id=budget.used,
            )
        )

        if decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
            if decision.retry_semantics.backoff_config:
                await asyncio.sleep(
                    jittered_backoff(
                        budget.used,
                        base=decision.retry_semantics.backoff_config.base_delay,
                    )
                )
            return "continue", {}

        elif decision.action == RecoveryAction.TRANSFORM_AND_RETRY:
            # PCD 2d: recovery transform breaks the frozen prefix.
            self._calibration_manager._maybe_break_commitment(transform_retry=True)
            updates: Dict[str, Any] = {}
            if decision.strategy_key == "native_to_text":
                updates["tools_kwarg"] = {}
                updates["use_native_tools"] = False
                transform_ok = True
            elif decision.strategy_key == "thinking_disable":
                updates["planned_enable_thinking"] = False
                transform_ok = True
            else:
                transform_ok = self._execute_transform_decision(decision, messages)
            if transform_ok:
                self._usage_tracker.mark_compression()
            coordinator.on_strategy_outcome(decision.decision_id, transform_ok)
            if not transform_ok:
                return "fatal", {"fatal_error": f"Transform failed: {decision.reason}"}
            return "continue", updates

        elif decision.action == RecoveryAction.FAILOVER:
            if hasattr(self._llm, "_failover"):
                self._llm._failover(f"recovery: {decision.reason}")
            self._post_failover_recompress(messages, coordinator, decision)
            coordinator.on_strategy_outcome(decision.decision_id, True)
            return "continue", {}

        else:
            # Terminal: HALT_CLEAN, HALT_WITH_CHECKPOINT, ASK_USER, etc.
            if decision.action == RecoveryAction.HALT_WITH_CHECKPOINT:
                self._save_halt_checkpoint(
                    decision,
                    envelope,
                    messages,
                    budget_used=budget.used,
                    tools_kwarg=tools_kwarg,
                    use_native_tools=use_native_tools,
                )
            elif decision.action in (
                RecoveryAction.HALT_CLEAN,
            ):
                self._audit_sink.update_outcome(
                    decision.decision_id,
                    "failure",
                    reason="Terminal halt",
                )
            fatal_error = _terminal_failure_text(decision)
            await sink.emit_error(fatal_error, _interaction_metadata(decision))
            return "break", {"fatal_error": fatal_error}

    # ── Unified Loop Helpers ───────────────────────────────────────────────


    # ── Helpers ──────────────────────────────────────────────────────────

    async def _synthesize_forced_answer(self, messages: List[Dict[str, Any]]) -> str:
        """One tool-free LLM round to answer with what the turn already gathered.

        A turn can stop before the model has written a final answer: a detected
        repetition loop is halted, or the iteration budget runs out. Breaking
        cold then hands the user a canned "reasoning step limit" notice and none
        of the information the tools already returned. This gives the model
        exactly one chance to answer from the accumulated context, with tools
        withheld so it cannot resume the loop.

        Returns the answer text, or "" on any failure so the caller falls back to
        the canned notice \u2014 a best-effort finalize must never turn a clean
        stop into a crash.
        """
        try:
            prompt = list(messages)
            prompt.append(build_user_message_text(_FORCED_FINALIZE_PROMPT))
            compressed = self._prompt_assembler._prepare_llm_messages(
                self._healer.heal(prompt), tools=None, round_number=0
            )
            resp = await self._llm.achat(
                compressed, stream=False, enable_thinking=False
            )
            text = (resp.content or "").strip()
            if self._sanitizer:
                text = self._sanitizer.sanitize(text)
            return text
        except Exception:
            logger.warning("forced final-answer synthesis failed", exc_info=True)
            return ""

    def _budget_exhausted_response(self, messages: List[Dict[str, Any]]) -> str:
        """Response when the iteration hard cap is reached.

        When the research ledger shows unfinished work, surface the remaining
        open-question count and next step so the stop is informative and
        continuable (not a bare dead-stop); otherwise the plain notice.
        """
        base = (
            "I've reached my reasoning step limit. Here's my best answer based on progress so far."
        )
        led = self._research_ledger
        if led.is_empty or led.open_question_count == 0:
            return base
        d = led.as_dict()
        parts = [
            base,
            "",
            f"Note: {led.open_question_count} open question(s) remain — the task is not fully complete.",
        ]
        next_step = d.get("next_step", "")
        if next_step:
            parts.append(f"Suggested next step: {next_step}")
        return "\n".join(parts)

    @staticmethod
    def _error_response(observation: Any) -> str:
        """Format error observation as user-facing response."""
        if isinstance(observation, dict):
            return f"Action failed: {observation.get('error', 'unknown error')}"
        return f"Action failed: {observation}"

    async def _sync_turn_safe(self, messages: List[Dict[str, Any]]) -> None:
        """Non-blocking wrapper for MemoryManager.sync_turn."""
        try:
            assert self._memory_manager is not None
            workspace_root = (
                self._current_task_contract.workspace_root if self._current_task_contract else ""
            )
            await asyncio.wait_for(
                self._memory_manager.sync_turn(
                    messages,
                    workspace_root=workspace_root,
                    session_id=self._current_session_id or "",
                ),
                timeout=self._settings.memory_prefetch_timeout_s,
            )
            logger.debug("memory.sync_turn completed")
        except asyncio.TimeoutError:
            logger.debug("memory.sync_turn timed out")
        except Exception:
            logger.debug("memory.sync_turn failed", exc_info=True)

    async def execute_action(self, action: Dict[str, Any], user_goal: str) -> Any:
        """Execute a no-LLM action (memory/skill/bridge/tool) via the dispatcher.

        Public entry point retained on the engine (referenced by the task
        scheduler's ``action_dispatcher`` wiring); delegates to the extracted
        :class:`SkillDispatcher`.
        """
        return await self._skill_dispatcher.execute_action(action, user_goal)

