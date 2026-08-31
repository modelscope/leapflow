"""CLI runtime context — assembles and manages the LeapFlow component graph."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional


from leapflow.platform.cua_client import CuaDriverClient
from leapflow.platform.event_bus import EventBus
from leapflow.platform.mock import MockBridge
from leapflow.config import Settings, _build_settings_from_env
from leapflow.config_loader import config_signature, load_config_bundle
from leapflow.engine.context_compressor import adaptive_tool_result_chars
from leapflow.engine.engine import AgentEngine, build_default_registry
from leapflow.engine.graph_planner import GraphPlanner
from leapflow.engine.intent_classifier import (
    FallbackClassifier,
    IntentClassifier,
    LLMIntentClassifier,
)
from leapflow.engine.scheduler import TaskScheduler
from leapflow.engine.session import SessionController
from leapflow.recording.attention import build_attention_filters
from leapflow.analysis.pipeline import ImitationPipeline
from leapflow.storage.session_store import LearningSessionStore
from leapflow.storage.trajectory_store import TrajectoryStore
from leapflow.llm.openai_provider import OpenAIChat
from leapflow.llm.provider_chain import (
    AuxiliaryClient,
    FailoverChain,
    parse_credential_pools,
    parse_provider_configs,
)
from leapflow.memory import (
    MemoryManager, WorkingMemoryProvider, EpisodicMemoryProvider,
    SemanticMemoryProvider, EvolutionMemoryProvider, NarrativeProvider,
    MemoryFragment,
)
from leapflow.skills.activator import SkillActivator
from leapflow.skills.evolution import EMAConfidencePolicy
from leapflow.skills.index import SkillIndex
from leapflow.skills.injector import SkillInjector
from leapflow.skills.discovery import configure as configure_skill_discovery
from leapflow.learning.active_learning import ActiveLearningObserver
from leapflow.learning.codegen import CompositeSkillCodeGenerator, LLMSkillCodeGenerator
from leapflow.learning.distiller import LLMSkillDistiller, SkillDistiller
from leapflow.learning.doc_generator import CompositeSkillDocGenerator, LLMSkillDocGenerator
from leapflow.storage.skill_docs import SkillDocStore
from leapflow.storage.connection import LocalConnectionHolder
from leapflow.learning.feedback import FeedbackEvaluator
from leapflow.storage.skill_library import SkillLibraryStore
from leapflow.skills.registry import SkillRegistry
from leapflow.engine.audit import AuditLogger
from leapflow.learning.similarity import HeuristicSimilarityScorer, LLMSimilarityScorer
from leapflow.platform.adapters.darwin import DarwinExecutionAdapter, DarwinPerceptionAdapter
from leapflow.domain.platform import PlatformManifest
from leapflow.engine.situational_assessor import LLMSituationalAssessor
from leapflow.platform.facade import VirtualSystemInterface
from leapflow.platform.normalizer import EventNormalizer


logger = logging.getLogger(__name__)

_MCP_THREAT_BLOCK_SEVERITY = 0.8
"""Severity at or above which an MCP tool description is refused registration.

Set where the classic injection patterns sit ("ignore all previous instructions", "your
new instructions are") rather than lower, because a tool description is *supposed* to
contain imperative language about what the tool does. Blocking on weak signals would
reject legitimate tools; blocking on nothing leaves an injection payload sitting in the
model's tool index for every subsequent turn.
"""


def _active_tool_workspace_root(fallback_workspace: str) -> str:
    """Return the current tool context workspace, or a stable fallback."""
    try:
        from leapflow.tools.execution_context import current_tool_context

        tool_ctx = current_tool_context()
        root = getattr(tool_ctx, "workspace_root", None) if tool_ctx is not None else None
    except Exception:
        root = None
    if root:
        return str(Path(str(root)).expanduser().resolve())
    return fallback_workspace


if TYPE_CHECKING:
    from leapflow.platform.observers import RecordingProfile
    from leapflow.security.approval import ApprovalDecision, ApprovalRequest
    from leapflow.storage.skill_library import StoredSkill


class _TUIApprovalGate:
    """Approval gate that delegates to the active TUI surface when available."""

    def __init__(self) -> None:
        self._handler: Optional[Callable[["ApprovalRequest"], Any]] = None

    def set_handler(self, handler: Optional[Callable[["ApprovalRequest"], Any]]) -> None:
        """Set the active TUI approval handler."""
        self._handler = handler

    _CATEGORY_LABELS = {
        "shell_dangerous": ("Shell Command", "yellow"),
        "shell.command": ("Shell Command", "yellow"),
        "file.read": ("Sensitive File Read", "yellow"),
        "file.write": ("File Write", "yellow"),
        "file_write": ("File Write", "yellow"),
        "gateway_send": ("External Message", "cyan"),
        "gateway.send": ("External Message", "cyan"),
    }

    async def request_approval(
        self, request: "ApprovalRequest",
    ) -> "ApprovalDecision":
        handler = self._handler
        if handler is not None:
            result = handler(request)
            if hasattr(result, "__await__"):
                return await result
            return result
        from leapflow.cli.approval_view import prompt_approval

        return await prompt_approval(request)

    async def check(self, command: str) -> bool:
        """``CommandApprovalGate`` compatibility — shell_tools calls this."""
        from leapflow.security.approval import ApprovalDecision, ApprovalRequest
        from leapflow.security.actions import ActionDescriptor

        action = ActionDescriptor.shell(command)
        decision = await self.request_approval(ApprovalRequest(
            category=action.kind,
            detail=command,
            risk_hint=0.7,
            action=action,
        ))
        return decision in {
            ApprovalDecision.ALLOW,
            ApprovalDecision.ALLOW_ONCE,
            ApprovalDecision.ALLOW_SESSION,
            ApprovalDecision.ALLOW_ALWAYS,
        }


def _default_recording_profile(settings: Settings) -> Optional["RecordingProfile"]:
    """Build a RecordingProfile for high-fidelity recording during teach.

    Always returns a profile so that teach sessions activate InputTapObserver
    and tighten FS debounce regardless of recording mode.
    """
    from leapflow.platform.observers import RecordingProfile
    return RecordingProfile()


def _emit_status(msg: str) -> None:
    """Write a dim status line to stderr (non-blocking, safe pre-logging)."""
    if sys.stderr.isatty():
        sys.stderr.write(f"\033[2m\u2192 {msg}\033[0m\n")
    else:
        sys.stderr.write(f"→ {msg}\n")
    sys.stderr.flush()


def _build_visual_components(
    settings: Settings, rpc: Any,
) -> Optional[Any]:
    """Build perception session if visual track is enabled.

    Returns perception_session, or *None* when visual track is disabled.
    """
    if not settings.visual_track_enabled:
        return None

    from leapflow.perception.config import PerceptionConfig
    from leapflow.perception.session import PerceptionSession

    vlm_api_key = settings.vlm_api_key or settings.llm_api_key
    if not vlm_api_key.strip():
        message = (
            "Visual perception disabled: LEAPFLOW_VLM_API_KEY or "
            "LEAPFLOW_LLM_API_KEY is required when visual track is enabled."
        )
        logger.warning(message)
        _emit_status(message)
        return None

    vlm_base_url = settings.vlm_base_url or settings.llm_base_url
    vlm_model = settings.vlm_model or settings.llm_model
    vlm_provider = OpenAIChat(
        api_key=vlm_api_key,
        base_url=vlm_base_url,
        model=vlm_model,
    )

    perception_config = PerceptionConfig.from_settings(settings)
    perception_session = PerceptionSession(
        config=perception_config,
        rpc=rpc,
        vlm=vlm_provider,
    )

    return perception_session


def _build_video_components(settings: Settings, rpc: Any, vlm: Any):
    """Build video-mode components when RecordingMode.VIDEO is active.

    Returns (TrajectoryRecorder, VideoAnalyzer, VideoSegmenter, SignalTimeline).
    """
    from leapflow.cache.manager import CacheManager
    from leapflow.layout import workspace_id_for_path
    from leapflow.perception.video.analyzer import VideoAnalyzer
    from leapflow.perception.video.cache_manager import VideoCacheManager
    from leapflow.perception.video.recorder import TrajectoryRecorder
    from leapflow.perception.video.segmenter import VideoSegmenter
    from leapflow.perception.video.timeline import SignalTimeline

    workspace_id = workspace_id_for_path(settings.workspace_root)
    cache_index = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    video_cache_manager = VideoCacheManager(
        settings.video_cache_dir,
        max_age_days=settings.video_cache_max_age_days,
        max_size_gb=settings.video_cache_max_size_gb,
        cache_manager=cache_index,
        workspace_id=workspace_id,
    )
    video_cache_manager.cleanup()

    recorder = TrajectoryRecorder(
        rpc,
        settings.video_cache_dir,
        fps=settings.video_fps,
        resolution_scale=settings.video_resolution_scale,
        codec=settings.video_codec,
        max_segment_s=settings.video_max_segment_s,
        cache_manager=cache_index,
        workspace_id=workspace_id,
    )
    analyzer = VideoAnalyzer(
        vlm,
        l2_enabled=settings.video_l2_enabled,
        l3_enabled=settings.video_l3_enabled,
        max_l2_requests=settings.video_max_l2_requests,
        max_l3_requests=settings.video_max_l3_requests,
        l2_time_window_s=settings.video_l2_time_window_s,
        frame_extractor=recorder,
        url_scheme=settings.video_vlm_url_scheme,
        vlm_max_retries=settings.video_vlm_max_retries,
        vlm_retry_backoff_s=settings.video_vlm_retry_backoff_s,
    )
    segmenter = VideoSegmenter(
        min_segment_s=settings.video_segmenter_min_s,
        max_segment_s=settings.video_segmenter_max_s,
        idle_gap_s=settings.video_segmenter_idle_gap_s,
        app_switch_gap_s=settings.video_segmenter_app_gap_s,
        min_split_s=settings.video_segmenter_min_split_s,
    )
    timeline = SignalTimeline(
        max_markers=settings.video_timeline_max_markers,
        compress_max=settings.video_timeline_compress_max,
        merge_channels=frozenset(
            s.strip() for s in settings.video_timeline_merge_channels.split(",") if s.strip()
        ),
    )
    return recorder, analyzer, segmenter, timeline


def _build_promotion_callback(lt: SemanticMemoryProvider):
    def _promote(frag: MemoryFragment) -> None:
        lt.insert_raw(
            frag.event_type,
            frag.content,
            path=frag.path,
            metadata=frag.metadata,
        )

    return _promote


def _make_stored_skill_fn(stored: "StoredSkill", llm: Any):
    """Create an LLM-backed execution function from a StoredSkill."""
    steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(stored.steps))

    async def _run(*, user_goal: str = "", **kwargs: Any) -> str:
        from leapflow.llm.message_builder import build_system_message, build_user_message_text

        params_desc = ""
        if kwargs:
            params_desc = "\n".join(f"- {k}: {v}" for k, v in kwargs.items())
            params_desc = f"\nProvided parameters:\n{params_desc}"

        goal = user_goal or stored.title
        system = (
            f"You are executing a learned skill: {stored.title}\n"
            f"Steps:\n{steps_text}\n"
            f"Pre-conditions: {', '.join(stored.pre_conditions) or 'none'}\n"
            f"Apps involved: {', '.join(stored.app_sequence) or 'any'}"
        )
        user_msg = f"Execute this skill. Goal: {goal}{params_desc}"

        resp = await llm.achat(
            [
                build_system_message(system),
                build_user_message_text(user_msg),
            ],
            stream=False,
            enable_thinking=False,
        )
        return resp.content or ""

    return _run


def _register_stored_skill_fallbacks(
    skill_lib: SkillLibraryStore,
    registry: SkillRegistry,
    llm: Any,
) -> int:
    """Register StoredSkills that lack a parameterized or doc counterpart."""
    from leapflow.learning.document import title_to_kebab
    from leapflow.skills.registry import Skill, SkillMetadata

    registered_names = set(registry.names()) if hasattr(registry, 'names') else {s.name for s in registry.list_all()}
    stored = skill_lib.load_all_active()
    count = 0

    for s in stored:
        # Same naming function as the SKILL.md write paths, otherwise the
        # dedup below misses doc-backed skills and registers a duplicate.
        name = title_to_kebab(s.title)
        if name in registered_names:
            continue
        if not s.trigger_phrases:
            continue

        skill_fn = _make_stored_skill_fn(s, llm)
        skill = Skill(
            name=name,
            description=s.title,
            run=skill_fn,
            triggers=list(s.trigger_phrases),
            metadata=SkillMetadata(
                source="distilled",
                source_trajectory_id=s.source_trajectory_id,
                source_episode_id=s.source_episode_id,
                confidence=s.confidence,
                version=s.version,
            ),
        )
        registry.register(skill)
        registered_names.add(name)
        count += 1

    return count


class Context:
    """Shared runtime context assembled once, used by all subcommands."""

    def __init__(self, settings: Settings, mock_host: bool) -> None:
        self.settings = settings
        try:
            from leapflow.security.path_sensitivity import configure_path_sensitivity_roots
            configure_path_sensitivity_roots((settings.layout.root,))
        except Exception:
            logger.debug("Path sensitivity root configuration skipped", exc_info=True)
        self.effective_mock = bool(mock_host or settings.mock_host)

        # Shared DuckDB connection holder — single leap.duckdb for all stores (P1).
        # Created here (lazy, not yet opened) so __init__-time providers such as
        # SemanticMemoryProvider can bind to it. Eager open + lock detection
        # happens later in initialize().
        self._db_holder = LocalConnectionHolder(
            settings.duckdb_path,
            volatile_on_lock=True,
        )

        # Memory subsystem — provider-based architecture (dual-layer)
        working = WorkingMemoryProvider(max_tokens=settings.memory_working_max_tokens)
        semantic = SemanticMemoryProvider(source=self._db_holder)
        episodic = EpisodicMemoryProvider(
            ttl=settings.memory_episodic_ttl_s,
            max_entries=settings.memory_episodic_max_entries,
            on_promote=_build_promotion_callback(semantic),
        )
        evolution = EvolutionMemoryProvider(max_episodes=settings.memory_evolution_max_episodes)
        narrative = NarrativeProvider(
            memory_dir=settings.profile_layout.memory_dir,
            workspace_path=str(Path.cwd()),
        )

        self.memory = MemoryManager()
        self.memory.add_provider(working)
        self.memory.add_provider(episodic)
        self.memory.add_provider(narrative)
        self.memory.add_provider(semantic)
        self.memory.add_provider(evolution)

        # Shorthands used by engine/event_bus
        self.wm = working
        self.lt = semantic
        self.imm = episodic
        self._evolution = evolution

        from leapflow.privacy.policy import PrivacyManager, PrivacyPolicy, DataRetentionConfig

        _exclude_paths_raw = os.environ.get("LEAPFLOW_PRIVACY_EXCLUDE_PATHS", "")
        _exclude_paths = frozenset(
            p.strip() for p in _exclude_paths_raw.split(",") if p.strip()
        ) if _exclude_paths_raw else frozenset()

        privacy_policy = PrivacyPolicy(
            exclude_apps=frozenset(settings.privacy_sensitive_apps),
            exclude_paths=_exclude_paths,
            retention=DataRetentionConfig(
                episodic_ttl_s=settings.memory_episodic_ttl_s,
            ),
        )
        self.privacy_manager = PrivacyManager(privacy_policy)

        self.event_bus = EventBus(
            immediate=self.imm,
            working=self.wm,
            privacy_filter=self.privacy_manager,
        )
        self.rpc: CuaDriverClient | MockBridge
        if self.effective_mock or not settings.use_cua_driver:
            if not self.effective_mock:
                _emit_status("cua-driver disabled by LEAPFLOW_USE_CUA_DRIVER=false")
                _emit_status("Running in degraded mode (no OS execution)")
            self.rpc = MockBridge()
            self.rpc.on_event(self.event_bus.handle_event)
        else:
            from leapflow.platform.cua_client import cua_driver_available
            if not cua_driver_available():
                _emit_status(
                    "WARNING: cua-driver not found on PATH"
                )
                _emit_status(
                    "  Install with: leap host install  |  Diagnose with: leap host doctor"
                )
            self.rpc = CuaDriverClient()

        self._config_signature = self._runtime_config_signature(settings)
        self._configure_llm_clients(settings)

        self.audit = AuditLogger(settings.audit_log_path)

        self.assessor: Optional[LLMSituationalAssessor] = None

        self.perception_session: Optional[Any] = None
        self.registry: Optional[SkillRegistry] = None
        self.imitation: Optional[ImitationPipeline] = None
        self.skill_lib: Optional[SkillLibraryStore] = None
        self.doc_store: Optional[SkillDocStore] = None
        self.session: Optional[SessionController] = None
        self.session_store: Optional[LearningSessionStore] = None
        self.engine: Optional[AgentEngine] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self._platform_manifest: Optional[PlatformManifest] = None
        self._platform_perception: Optional[Any] = None
        self._platform_execution: Optional[Any] = None
        self._platform_event_callback: Optional[Callable[[Any], None]] = None

        # World Model components (wired during initialize)
        self.learning_budget: Optional[Any] = None
        self.experience_store: Optional[Any] = None
        self.prediction_loop: Optional[Any] = None
        self.curiosity: Optional[Any] = None
        self.replay_engine: Optional[Any] = None
        self.snapshot_service: Optional[Any] = None
        self.trajectory_grader: Optional[Any] = None
        self.active_observer: Optional[Any] = None

        # Daemon / Observer
        self._observation_daemon: Optional[Any] = None
        self._pipeline_observer: Optional[Any] = None

        # Skill evolution & PatternMiner
        self._evolution_policy: Optional[EMAConfidencePolicy] = None
        self._pattern_miner: Optional[Any] = None

        # Deferred initialization tracking
        self._deferred_initialized: bool = False
        self._deferred_lock: asyncio.Lock = asyncio.Lock()
        self._deferred_attempts: int = 0
        # Single runner task executing initialize_deferred(); callers only
        # wait on it (shielded), so a caller timeout never cancels the init.
        self._deferred_task: Optional["asyncio.Task[None]"] = None
        # Dedicated single-thread executor for the heavy synchronous DuckDB
        # work inside initialize_deferred(): keeps the event loop responsive
        # and serializes deferred DB operations among themselves.
        self._deferred_db_executor: Optional[ThreadPoolExecutor] = None

        # Intermediate attributes shared between critical and deferred init
        self._critical_codegen: Optional[Any] = None
        self._critical_traj_store: Optional[Any] = None
        self._critical_distiller: Optional[Any] = None
        self._critical_intent_inferrer: Optional[Any] = None
        self._critical_attention_filters: Optional[List[Any]] = None
        self._critical_surprise_annotator: Optional[Any] = None
        self._critical_scorer: Optional[Any] = None
        self._critical_llm_scorer: Optional[Any] = None
        self._critical_feedback_evaluator: Optional[Any] = None
        self._critical_activator: Optional[Any] = None

        # Unified approval gate is resource-free; create it in __init__ so all
        # initialize() wiring paths can safely reference the same session gate.
        from leapflow.security.approval import SessionAwareGate
        from leapflow.security.grants import ApprovalAuditLog, JsonApprovalGrantStore
        from leapflow.security.orchestrator import ApprovalOrchestrator
        from leapflow.security.policy import ApprovalPolicyEngine

        # Hardware is resolved here, before the orchestrator, because the risk
        # classifier is composed at construction: a device action must be assessed by
        # the hardware classifier from the very first turn. With hardware disabled the
        # registry is None and build_risk_classifier returns the unmodified default,
        # so this costs nothing and changes nothing.
        from leapflow.hardware.registry import build_registry as _build_hardware_registry
        from leapflow.hardware.risk import build_risk_classifier as _build_risk_classifier

        self._hardware_registry = _build_hardware_registry(settings)

        approval_layout = settings.profile_layout.approval
        self._tui_approval = _TUIApprovalGate()
        self._approval_gate = SessionAwareGate(self._tui_approval)
        self._approval_orchestrator = ApprovalOrchestrator(
            self._approval_gate,
            risk_classifier=_build_risk_classifier(self._hardware_registry),
            policy=ApprovalPolicyEngine(bypass=settings.approval_bypass),
            grants=JsonApprovalGrantStore(approval_layout.grants_path),
            audit=ApprovalAuditLog(approval_layout.audit_path),
        )

    def set_approval_handler(self, handler: Optional[Callable[["ApprovalRequest"], Any]]) -> None:
        """Bind the current interactive surface as the approval renderer."""
        self._tui_approval.set_handler(handler)

    _DEFERRED_MAX_ATTEMPTS: int = 2

    async def _ensure_deferred(self) -> None:
        """Wait for deferred init, starting the runner task if none is active.

        The initialization itself always runs inside a dedicated runner task
        (``_run_deferred_once``); callers only *wait* on it through
        ``asyncio.shield``. This makes the wait cancellation-safe: when a
        caller wraps this in ``asyncio.wait_for`` and times out, only the
        wait is cancelled — the background initialization keeps running and
        later callers pick up the completed state.

        Gives up after _DEFERRED_MAX_ATTEMPTS failed attempts: components stay
        uninitialized and the engine keeps running in critical-only mode.
        """
        if self._deferred_initialized:
            return
        task = getattr(self, "_deferred_task", None)
        if task is None or task.done():
            task = asyncio.create_task(self._run_deferred_once())
            self._deferred_task = task
        await asyncio.shield(task)

    async def _run_deferred_once(self) -> None:
        """Single deferred-init attempt; only ever runs in the runner task."""
        async with self._deferred_lock:
            if self._deferred_initialized:
                return
            if self._deferred_attempts >= self._DEFERRED_MAX_ATTEMPTS:
                logger.error(
                    "Deferred initialization abandoned after %d failed attempts; "
                    "engine continues in critical-only degraded mode",
                    self._deferred_attempts,
                )
                return
            self._deferred_attempts += 1
            await self.initialize_deferred()
            self._deferred_initialized = True

    async def _run_deferred_db(self, fn: Callable[[], Any]) -> Any:
        """Run a blocking (DuckDB/file) operation off the event loop.

        Uses a dedicated single-thread executor so that:
        - the event loop stays responsive during heavy deferred-init work
          (RPCs such as daemon.status keep answering), and
        - deferred DB operations are serialized among themselves and never
          hit the shared DuckDB connection concurrently.
        """
        if self._deferred_db_executor is None:
            self._deferred_db_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="leap-deferred-db",
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._deferred_db_executor, fn)

    async def _ensure_skill_system(self) -> None:
        """Ensure skill registry is fully populated (deferred skills loaded)."""
        if self._deferred_initialized:
            return
        await self._ensure_deferred()

    async def _ensure_world_model(self) -> None:
        """Ensure world model components are ready."""
        if self.prediction_loop is not None:
            return
        await self._ensure_deferred()

    def _configure_llm_clients(self, settings: Settings) -> None:
        """Build LLM/VLM clients from a settings snapshot."""
        provider_configs = parse_provider_configs(
            settings.llm_api_key or "missing",
            settings.llm_base_url,
            settings.llm_model,
            fallback_json=settings.llm_fallback_providers,
            primary_context_length=settings.llm_context_length,
        )
        credential_pools = parse_credential_pools(
            provider_configs,
            cooldown_s=settings.llm_credential_cooldown_s,
        )
        if len(provider_configs) > 1 or credential_pools:
            self.llm_chain = FailoverChain(
                provider_configs,
                credential_pools=credential_pools,
                circuit_failure_threshold=settings.circuit_breaker_threshold,
                circuit_cooldown_s=settings.circuit_breaker_cooldown_s,
            )
            self.llm = self.llm_chain
            logger.info(
                "LLM chain: %d providers, %d credential pools",
                len(provider_configs), len(credential_pools),
            )
        else:
            self.llm_chain = None
            self.llm = OpenAIChat(
                api_key=settings.llm_api_key or "missing",
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                max_retries=settings.llm_max_retries,
            )

        self.auxiliary: Optional[AuxiliaryClient] = None
        if settings.llm_aux_model:
            aux_llm = OpenAIChat(
                api_key=settings.llm_aux_api_key or settings.llm_api_key or "missing",
                base_url=settings.llm_aux_base_url or settings.llm_base_url,
                model=settings.llm_aux_model,
                max_retries=2,
            )
            self.auxiliary = AuxiliaryClient(aux_llm)
            logger.info("Auxiliary LLM configured: %s", settings.llm_aux_model)
        elif settings.has_llm_credentials:
            self.auxiliary = AuxiliaryClient(self.llm)

        self.vlm: Optional[OpenAIChat] = None
        if (
            settings.vlm_model
            and settings.vlm_model != settings.llm_model
            and settings.has_vlm_credentials
        ):
            self.vlm = OpenAIChat(
                api_key=settings.vlm_api_key or settings.llm_api_key,
                base_url=settings.vlm_base_url or settings.llm_base_url,
                model=settings.vlm_model,
            )

    def _effective_llm_context_length(self, settings: Settings) -> int:
        """Return the configured runtime context budget for the active provider."""
        if self.llm_chain is not None:
            return max(1, int(self.llm_chain.context_length))
        return max(1, int(settings.llm_context_length))

    def _build_model_capability_registry(self, settings: Settings) -> Any:
        """Build model capabilities where explicit runtime config wins over static hints."""
        from leapflow.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry

        cap_registry = ModelCapabilityRegistry()
        base_caps = cap_registry.resolve(settings.llm_model)
        cap_registry.register(
            settings.llm_model,
            ModelCapabilities(
                context_length=self._effective_llm_context_length(settings),
                max_output_tokens=base_caps.max_output_tokens,
                supports_tools=settings.native_tool_calling_enabled,
                supports_vision=base_caps.supports_vision,
                supports_thinking=base_caps.supports_thinking,
                supports_streaming_tools=base_caps.supports_streaming_tools,
                tokens_per_image=base_caps.tokens_per_image,
            ),
        )
        return cap_registry

    def _sync_engine_runtime_budget(self, settings: Settings) -> None:
        """Sync engine-visible budgets and model capability metadata from settings."""
        if self.engine is None:
            return

        context_length = self._effective_llm_context_length(settings)
        dynamic_result_budget = adaptive_tool_result_chars(
            settings.max_tool_result_chars, context_length,
        )
        if dynamic_result_budget != settings.max_tool_result_chars:
            logger.info(
                "Dynamic tool result budget: %d (context=%d)",
                dynamic_result_budget,
                context_length,
            )
        self.engine.set_tool_result_budget(dynamic_result_budget)
        self.engine.set_model_capabilities(self._build_model_capability_registry(settings))

        compressor = getattr(self.engine, "_compressor", None)
        if compressor is not None and hasattr(compressor, "reconfigure"):
            token_budget = max(1, int(context_length * settings.context_hard_limit_ratio))
            compressor.reconfigure(
                token_budget=token_budget,
                context_length=context_length,
            )

        logger.debug("Model capability registry wired")

    @staticmethod
    def _runtime_config_signature(settings: Settings) -> tuple:
        """Return a stable signature for user-editable runtime config files."""
        paths = tuple(settings.watched_config_paths or settings.layout.watched_config_paths(
            settings.profile,
            settings.workspace_root,
        ))
        return config_signature(paths)

    @staticmethod
    def _bool_env(value: str, default: bool) -> bool:
        text = value.strip().lower()
        if not text:
            return default
        return text in {"1", "true", "yes", "on"}

    def _load_runtime_settings_from_files(self) -> Settings:
        """Reload hot-swappable settings from structured config sources."""
        bundle = load_config_bundle(
            self.settings.layout,
            self.settings.profile_layout,
            self.settings.workspace_root,
        )
        original_env = dict(os.environ)
        for key, value in bundle.env.items():
            os.environ[key] = value
        try:
            return _build_settings_from_env(
                layout=self.settings.layout,
                profile_layout=self.settings.profile_layout,
                profile_manifest=self.settings.profile_manifest,
                config_sources=tuple(str(source.path) for source in bundle.sources),
                watched_config_paths=bundle.watched_paths,
                config_warnings=bundle.warnings,
            )
        finally:
            # Restore original env to avoid polluting daemon global state.
            for key in bundle.env:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)

    async def _authorize_mcp_call(self, schema: Any, params: dict) -> tuple[bool, str]:
        """Run one MCP tool call through the approval orchestrator.

        Fails closed on absence and on exception. No orchestrator, or one that raises,
        both mean deny with a message the model can act on: a broken gate must never
        become an open door, and this gate stands in front of arbitrary third-party code.

        ``mcp.approval_mode`` selects the policy. The default, ``mutating_only``, gates
        every tool that does not declare itself read-only -- absence of a declaration is
        not a claim of safety, so an old server that carries no annotations is gated in
        full. ``off`` exists because a bench of trusted local servers is a real setup, but
        it is logged once per process so the choice is discoverable in a diagnosis.
        """
        mode = str(getattr(self.settings, "mcp_approval_mode", "mutating_only") or "mutating_only")
        read_only = bool(getattr(schema, "read_only", False))
        if mode == "off":
            if not self._mcp_approval_off_logged:
                self._mcp_approval_off_logged = True
                logger.warning(
                    "mcp.approval_mode=off: MCP tool calls run without approval. "
                    "Third-party server code executes with this agent's privileges."
                )
            return True, ""
        if mode == "mutating_only" and read_only:
            return True, ""

        orchestrator = getattr(self, "_approval_orchestrator", None)
        if orchestrator is None:
            return False, (
                "No approval gate is installed for MCP tools, so the call was refused. "
                "This is a configuration fault, not a user decision."
            )

        from leapflow.security.actions import ActionDescriptor

        descriptor = ActionDescriptor.mcp_tool(
            server=str(getattr(schema, "server_name", "") or ""),
            tool=str(getattr(schema, "original_name", "") or getattr(schema, "name", "")),
            arguments=params,
            description=str(getattr(schema, "description", "") or ""),
            read_only=read_only,
        )
        try:
            result = await orchestrator.evaluate(descriptor)
        except Exception as exc:
            logger.error(
                "MCP approval gate raised for %r: %s", descriptor.resource, exc, exc_info=True
            )
            return False, "The approval gate failed while assessing this call, so it was refused."
        if getattr(result, "approved", False):
            return True, ""
        # The orchestrator's own wording states that the user withheld consent and that
        # the outcome must not be pursued another way. Substituting a generic tool error
        # would let the agent reroute around a refusal.
        message = getattr(result, "denial_message", "") or getattr(result, "reason", "")
        return False, str(message or "The MCP tool call was not approved.")

    def _configure_mcp_manager(self, settings: Settings) -> None:
        """Rebuild MCP manager and global MCP tool registrations from layout config."""
        from leapflow.plugins import get_registry
        _tool_registry = get_registry()

        previous_names = set(getattr(self, "_mcp_tool_names", ()))
        previous_manager = getattr(self, "_mcp_manager", None)
        if previous_manager is not None:
            try:
                previous_manager.close()
            except Exception:
                logger.debug("MCP manager close failed during rebuild", exc_info=True)
        if previous_names:
            _tool_registry.tool_definitions[:] = [
                definition for definition in _tool_registry.tool_definitions
                if str((definition.get("function") or {}).get("name") or "") not in previous_names
            ]
            for name in previous_names:
                _tool_registry.tool_handlers.pop(name, None)
        self._mcp_manager = None
        self._mcp_tool_names = ()
        self._mcp_approval_off_logged = False

        try:
            mcp_config_path = settings.layout.mcp_servers_path
            if not mcp_config_path.exists():
                return
            import json as _json_mcp
            from leapflow.platform.mcp_manager import McpManager, McpServerConfig
            from leapflow.security.threat_patterns import scan_mcp_description

            raw_configs = _json_mcp.loads(mcp_config_path.read_text(encoding="utf-8"))
            if not isinstance(raw_configs, dict):
                logger.warning("MCP config must be a JSON object: %s", mcp_config_path)
                return
            server_configs = [
                McpServerConfig(
                    name=name,
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    parallel_safe=cfg.get("parallel_safe", False),
                )
                for name, cfg in raw_configs.items()
                if isinstance(cfg, dict) and cfg.get("enabled", True) and cfg.get("command")
            ]
            if not server_configs:
                return

            mgr = McpManager()
            total_tools = 0
            for sc in server_configs:
                try:
                    schemas = mgr.add_server(sc)
                    total_tools += len(schemas)
                except Exception:
                    logger.warning("MCP server '%s' failed to connect", sc.name)

            tool_names: list[str] = []

            def _build_mcp_handler(manager, schema):
                """Wrap one MCP tool call in the single approval entry point.

                An MCP tool is third-party code reached over a local transport, running
                with this agent's privileges, and the protocol tells us nothing about what
                it does. Before this gate existed, every such call executed with no risk
                classification, no consent, and no audit record -- the only sensitive
                capability in the process reachable without passing through the
                orchestrator.

                The orchestrator is resolved per call rather than captured here on
                purpose: ``ApprovalCoordinator.install_gate`` *replaces*
                ``ctx._approval_orchestrator`` when leapd starts, so a captured reference
                would keep routing prompts to the in-process gate and a daemon session
                would never see them.
                """

                async def _handler(params: dict) -> dict:
                    allowed, denial = await self._authorize_mcp_call(schema, params)
                    if not allowed:
                        return {"ok": False, "error": denial, "failure_code": "approval_denied"}
                    return await manager.call_tool(schema.name, params)

                return _handler

            for schema in mgr.get_tool_schemas():
                threats = scan_mcp_description(schema.description)
                blocking = [t for t in threats if t.severity >= _MCP_THREAT_BLOCK_SEVERITY]
                if blocking:
                    # Refused, not merely logged. A tool description is injected verbatim
                    # into the model's tool index, so a description carrying "ignore all
                    # previous instructions" is an attack delivered through the capability
                    # catalogue itself. Registering it and warning leaves the payload in
                    # place for every subsequent turn.
                    logger.error(
                        "Refusing MCP tool %r from server %r: description matches "
                        "prompt-injection patterns %s",
                        schema.name,
                        schema.server_name,
                        [t.pattern_name for t in blocking],
                    )
                    continue
                if threats:
                    logger.warning(
                        "MCP tool '%s' description has threats: %s",
                        schema.name,
                        [t.pattern_name for t in threats],
                    )
                _tool_registry.tool_definitions.append(schema.to_openai_function())
                _tool_registry.tool_handlers[schema.name] = _build_mcp_handler(mgr, schema)
                tool_names.append(schema.name)

            if tool_names:
                self._mcp_manager = mgr
                self._mcp_tool_names = tuple(tool_names)
                self._install_mcp_transport_client()
                logger.info("MCP Manager: %d servers, %d tools registered to agent", len(server_configs), total_tools)
            else:
                mgr.close()
        except Exception:
            logger.debug("MCP Manager initialization skipped", exc_info=True)

    def _install_mcp_transport_client(self) -> None:
        """Let a device declared with ``kind: mcp`` reach the configured servers.

        A resolver rather than the manager itself, because ``_configure_mcp_manager``
        runs again on every runtime config reload: a captured manager would outlive
        the servers it was built for, and the device would keep calling a closed
        session. Installed here because this is the one place the manager exists.
        """
        try:
            from leapflow.hardware.transports.mcp import set_mcp_client_provider

            set_mcp_client_provider(lambda: getattr(self, "_mcp_manager", None))
        except Exception:
            logger.debug("MCP hardware transport client not installed", exc_info=True)

    def reload_runtime_config_if_changed(self, *, force: bool = False) -> bool:
        """Hot-reload LLM/VLM config when user-editable config files changed."""
        signature = self._runtime_config_signature(self.settings)
        if not force and signature == self._config_signature:
            return False

        previous = self.settings
        updated = self._load_runtime_settings_from_files()
        self._config_signature = self._runtime_config_signature(updated)
        llm_changed = (
            previous.llm_api_key != updated.llm_api_key
            or previous.llm_base_url != updated.llm_base_url
            or previous.llm_model != updated.llm_model
            or previous.llm_max_retries != updated.llm_max_retries
            or previous.llm_context_length != updated.llm_context_length
            or previous.vlm_api_key != updated.vlm_api_key
            or previous.vlm_base_url != updated.vlm_base_url
            or previous.vlm_model != updated.vlm_model
            or previous.visual_track_enabled != updated.visual_track_enabled
        )
        self.settings = updated
        self._configure_mcp_manager(updated)
        if not llm_changed:
            if self.engine is not None:
                self.engine.reconfigure_runtime(
                    settings=updated,
                    llm=self.llm,
                    vlm=self.vlm,
                    classifier=self.intent_classifier,
                )
            logger.info("Runtime configuration reloaded")
            return True

        self._configure_llm_clients(updated)
        classifier: IntentClassifier = (
            LLMIntentClassifier(self.llm)
            if updated.has_llm_credentials
            else FallbackClassifier()
        )
        self.intent_classifier = classifier
        self.assessor = (
            LLMSituationalAssessor(self.llm)
            if updated.has_llm_credentials
            else None
        )
        if self.engine is not None:
            self.engine.reconfigure_runtime(
                settings=updated,
                llm=self.llm,
                vlm=self.vlm,
                classifier=classifier,
            )
            self._sync_engine_runtime_budget(updated)
        logger.info("Runtime configuration reloaded")
        return True

    def _host_backend_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.rpc, "status_snapshot", None)
        if callable(snapshot):
            try:
                return dict(snapshot())
            except Exception as exc:
                return {
                    "backend": type(self.rpc).__name__,
                    "started": False,
                    "pid": None,
                    "pid_source": "unavailable",
                    "last_error": str(exc),
                }
        return {
            "backend": type(self.rpc).__name__,
            "started": bool(getattr(self.rpc, "connected", False)),
            "pid": None,
            "pid_source": "unavailable",
        }

    async def host_backend_status(self) -> dict[str, Any]:
        """Return the current daemon-owned host backend status."""
        return self._host_backend_snapshot()

    async def host_backend_start(self) -> dict[str, Any]:
        """Start CuaDriver and rewire runtime host adapters in-place."""
        if self.effective_mock:
            return {
                "ok": False,
                "backend": "mock",
                "started": False,
                "pid": None,
                "pid_source": "unavailable",
                "last_error": "host backend is disabled by mock mode",
            }
        if isinstance(self.rpc, CuaDriverClient) and self.rpc.connected:
            status = self._host_backend_snapshot()
            status.update({"ok": True, "changed": False})
            return status

        previous_rpc = self.rpc
        next_rpc = CuaDriverClient()
        try:
            next_rpc.start()
            manifest = await VirtualSystemInterface(next_rpc).handshake()
            await self._rewire_host_backend(next_rpc, manifest, bridge_online=True)
        except Exception as exc:
            try:
                next_rpc.stop()
            except Exception:
                logger.debug("CuaDriverClient cleanup failed after start error", exc_info=True)
            return {
                "ok": False,
                "backend": "cua-driver",
                "started": False,
                "pid": None,
                "pid_source": "unavailable",
                "last_error": str(exc),
            }

        if isinstance(previous_rpc, CuaDriverClient) and previous_rpc is not next_rpc:
            try:
                previous_rpc.stop()
            except Exception:
                logger.debug("previous CuaDriverClient stop failed after host start", exc_info=True)
        status = self._host_backend_snapshot()
        status.update({"ok": True, "changed": True})
        return status

    async def host_backend_stop(self) -> dict[str, Any]:
        """Stop CuaDriver while preserving chat, memory, and daemon runtime state."""
        previous_rpc = self.rpc
        stop_error = ""
        if isinstance(previous_rpc, CuaDriverClient):
            try:
                previous_rpc.stop()
            except Exception as exc:
                stop_error = str(exc)
                logger.debug("CuaDriverClient stop failed during host_backend_stop", exc_info=True)

        mock_rpc = MockBridge()
        mock_rpc.on_event(self.event_bus.handle_event)
        manifest = PlatformManifest.default_darwin()
        await self._rewire_host_backend(mock_rpc, manifest, bridge_online=False)
        status = self._host_backend_snapshot()
        status.update({"ok": not stop_error, "changed": True})
        if stop_error:
            status["last_error"] = stop_error
        return status

    async def host_backend_restart(self) -> dict[str, Any]:
        """Restart CuaDriver and keep daemon/session state intact."""
        await self.host_backend_stop()
        return await self.host_backend_start()

    async def _rewire_host_backend(
        self,
        rpc: CuaDriverClient | MockBridge,
        manifest: PlatformManifest,
        *,
        bridge_online: bool,
    ) -> None:
        previous_callback = self._platform_event_callback
        if previous_callback is not None:
            self.event_bus.unsubscribe(previous_callback)
            self._platform_event_callback = None

        self.rpc = rpc
        self.event_bus.set_normalizer(EventNormalizer(manifest))

        perception: Any
        execution_adapter: Any
        if not self.effective_mock and bridge_online:
            perception = DarwinPerceptionAdapter(rpc, manifest)
            execution_adapter = DarwinExecutionAdapter(rpc, manifest)
            self.event_bus.subscribe(perception.enqueue_event)
            self._platform_event_callback = perception.enqueue_event
            home = str(Path.home())
            cwd = str(Path.cwd())
            for watch_path in dict.fromkeys([home, cwd]):
                try:
                    result = await rpc.call("fs.subscribe", {"path": watch_path})
                    logger.info("Subscribed to FS events after host switch: %s", watch_path)
                    for evt in (result.get("recent") or []):
                        await self.event_bus.handle_event("event.fs_change", dict(evt))
                except Exception as exc:
                    logger.warning("Failed to subscribe FS events for %s: %s", watch_path, exc)
        else:
            from leapflow.platform.adapters.mock import MockExecutionAdapter, MockPerceptionAdapter
            perception = MockPerceptionAdapter()
            execution_adapter = MockExecutionAdapter()

        self._platform_manifest = manifest
        self._platform_perception = perception
        self._platform_execution = execution_adapter

        if self.engine is not None:
            from leapflow.plugins import get_registry

            # Bind perception/execution to the desktop semantic plugin
            get_registry().bind_runtime(perception=perception, execution=execution_adapter)
            self.engine.reconfigure_host_backend(
                rpc=rpc,
                perception=perception,
                execution=execution_adapter,
            )

    def _bind_hardware_experience(self) -> None:
        """Give the hardware registry the experience store once it exists.

        Called from deferred initialization because that is where the store is built,
        while hardware persistence is bound during critical initialization -- reading
        ``experience_store`` there would only ever find None. Without this second pass the
        outcome recorder stays disabled and physical commands are never learned from, which
        is exactly the kind of "wired but never reached" gap that is invisible in review.
        """
        registry = getattr(self, "_hardware_registry", None)
        store = getattr(self, "experience_store", None)
        if registry is None or store is None:
            return
        try:
            registry.bind_persistence(experience_store=store)
        except Exception:
            logger.warning(
                "Could not bind the experience store to hardware outcomes", exc_info=True
            )

    async def _start_hardware_streams(self) -> None:
        """Begin sampling channels that declare a sample rate.

        Started here rather than handed to ``ActiveSourceManager`` because that manager
        has no production caller today; delegating to it would ship a sampling loop that
        never runs. Handing it over later also needs a ``HardwareEvent`` ->
        ``InteractionSignal`` adapter, since its queue is typed for the latter.

        Failures are contained: a bench that cannot be sampled must not prevent the
        process from finishing initialization.
        """
        registry = getattr(self, "_hardware_registry", None)
        if registry is None:
            return
        self._bind_hardware_persistence(registry)
        try:
            # Installed before starting, and used by the command path too: a refusal
            # to command an unreachable device must reach the same signal path as a
            # threshold breach, or a stalled bench stays invisible.
            registry.set_event_emitter(self._hardware_event_emitter())
            await registry.start_streams()
        except Exception:
            logger.warning("Hardware streaming failed to start", exc_info=True)

    def _hardware_event_emitter(self) -> Any:
        """Return the sink that puts derived device events on the shared signal path.

        This is the step that makes physical observation actionable. Without it the
        detector still runs and still records events for ``hw_status``, but nothing
        reacts to them: an overnight run could leave its declared envelope and no
        watch, board or turn would ever hear about it. Devices go onto ``EventBus``
        rather than a private channel so they reach the same noise gate, watch
        activation and board stream as every other environment signal -- ``hw`` is a
        family there by virtue of the event type, with nothing enumerated anywhere.

        Returns ``None`` when there is no bus, so the registry keeps recording events
        for status instead of failing to sample.
        """
        event_bus = getattr(self, "event_bus", None)
        if event_bus is None or not hasattr(event_bus, "handle_event"):
            logger.debug("No event bus for hardware events; sampling records for status only")
            return None

        def _emit(event: Any) -> None:
            # Sampling runs on this loop, and ingestion is async, so the handoff is a
            # task. Safe to spawn per event only because the source paces each kind;
            # an unpaced 10 Hz channel would otherwise queue tasks at sampling rate.
            try:
                asyncio.create_task(
                    event_bus.handle_event(event.event_type, event.to_payload()),
                    name=f"hw-event:{event.kind}",
                )
            except RuntimeError:
                # No running loop (teardown). Dropping one event is correct here;
                # raising would surface inside the sampling loop's dispatch.
                logger.debug("Dropped hardware event %s: no running loop", event.event_type)

        return _emit

    def _bind_hardware_persistence(self, registry: Any) -> None:
        """Point the reading store at session-scoped, layout-owned paths.

        Raw physical samples are treated like the session's visual and VLM artifacts:
        session cache scope, marked sensitive and non-syncable, TTL bounded. A qPCR curve
        can carry patient sample information and a production temperature trace can be a
        trade secret, so this data must never leave the machine and must expire.

        Bound here rather than at registry construction because the path is session
        scoped, and no session exists when the registry is built.
        """
        try:
            from leapflow.cache.manager import CacheManager, CacheScope
            from leapflow.hardware.reading_store import READINGS_CATEGORY

            settings = self.settings
            cache_layout = settings.profile_layout.cache
            session_id = str(getattr(self, "session_id", "") or "default")
            workspace_id = str(getattr(settings, "workspace_id", "") or "default")
            readings_dir = cache_layout.category_dir(
                scope=CacheScope.SESSION.value,
                category=READINGS_CATEGORY,
                workspace_id=workspace_id,
                session_id=session_id,
            )
            registry.bind_persistence(
                cache_manager=CacheManager(
                    cache_layout, profile_id=settings.profile_manifest.profile_id
                ),
                readings_dir=readings_dir,
                session_id=session_id,
                # Physical outcomes are the first clean ground truth the world model can
                # get: the command was 37.0, the device settled at 36.8, the error is 0.2
                # and needs no model call to judge. Absent a store the recorder stays off
                # rather than accumulating comparisons nothing will resolve.
                experience_store=getattr(self, "experience_store", None),
            )
        except Exception:
            # Reduced to in-memory sampling rather than no sampling: observing the device
            # is still worth more than nothing, and the failure is visible here.
            logger.warning(
                "Hardware reading persistence unavailable; samples will not be stored",
                exc_info=True,
            )

    def _bind_hardware_plugin(self) -> None:
        """Bind the hardware registry and approval gate into the hardware plugin.

        Skipped entirely when hardware is disabled: the plugin then keeps an empty tool
        list, so the LLM tool index is byte-identical to a build without the subsystem.
        That equivalence is what makes the feature default-off and reversible, and it is
        also what keeps the journey cassette fingerprints valid.

        The gate passed here is the orchestrator, not a bare gate: hardware commands go
        through the same single entry point as every other sensitive capability.
        """
        registry = getattr(self, "_hardware_registry", None)
        if registry is None:
            return
        from leapflow.plugins import get_registry as _get_tool_registry

        _get_tool_registry().bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=self._approval_orchestrator,
        )
        report = registry.report
        logger.info(
            "Hardware plugin bound: %d device(s) admitted, %d rejected",
            len(report.admitted),
            len(report.rejected),
        )

    @property
    def storage_volatile(self) -> bool:
        """Return True when this process uses non-persistent fallback storage."""
        return bool(getattr(self._db_holder, "is_volatile", False))

    async def initialize(self) -> None:
        """Full initialization - used by CLI direct mode."""
        await self.initialize_critical()
        await self.initialize_deferred()
        self._deferred_initialized = True

    async def initialize_critical(self) -> None:
        """Critical-path initialization: platform, memory, engine core.

        Must complete before service.start() returns. Provides enough state
        for the engine to handle basic chat requests.
        """
        settings = self.settings

        await self.memory.initialize_all()
        if self.storage_volatile:
            _emit_status(
                "Primary database is locked; running with volatile session storage."
            )
            _emit_status(
                "This window can chat, but new memory/session data will not persist."
            )

        vsi = VirtualSystemInterface(self.rpc)
        bridge_online = False

        if isinstance(self.rpc, CuaDriverClient):
            try:
                self.rpc.start()
                manifest = await vsi.handshake()
                bridge_online = True
            except RuntimeError as exc:
                _emit_status(f"cua-driver connection failed: {exc}")
                if "not found" in str(exc).lower():
                    _emit_status(
                        "  Install with: leap host install"
                    )
                else:
                    _emit_status(
                        "  Check permissions (macOS TCC) or run: leap host doctor"
                    )
                _emit_status("Running in degraded mode (no OS execution)")
                self.rpc = MockBridge()
                self.rpc.on_event(self.event_bus.handle_event)
                vsi = VirtualSystemInterface(self.rpc)
                manifest = PlatformManifest.default_darwin()
                vsi._manifest = manifest
        else:
            manifest = await vsi.handshake()
            bridge_online = True

        normalizer = EventNormalizer(manifest)
        self.event_bus.set_normalizer(normalizer)

        perception: Any = None
        execution_adapter: Any = None

        if not self.effective_mock and bridge_online:
            perception = DarwinPerceptionAdapter(self.rpc, manifest)
            execution_adapter = DarwinExecutionAdapter(self.rpc, manifest)
            self.event_bus.subscribe(perception.enqueue_event)
            self._platform_event_callback = perception.enqueue_event
            home = str(Path.home())
            cwd = str(Path.cwd())
            for watch_path in dict.fromkeys([home, cwd]):
                try:
                    result = await self.rpc.call("fs.subscribe", {"path": watch_path})
                    logger.info("Subscribed to FS events: %s", watch_path)
                    for evt in (result.get("recent") or []):
                        await self.event_bus.handle_event("event.fs_change", dict(evt))
                except Exception as exc:
                    logger.warning("Failed to subscribe FS events for %s: %s", watch_path, exc)
        else:
            from leapflow.platform.adapters.mock import MockExecutionAdapter, MockPerceptionAdapter
            perception = MockPerceptionAdapter()
            execution_adapter = MockExecutionAdapter()
            self._platform_event_callback = None

        self._platform_manifest = manifest
        self._platform_perception = perception
        self._platform_execution = execution_adapter

        logger.info(
            "Platform: %s (v%s) | Capabilities: %d",
            manifest.platform_id.value,
            manifest.os_version,
            len(manifest.capabilities),
        )

        codegen = None
        if settings.has_llm_credentials:
            codegen = CompositeSkillCodeGenerator(LLMSkillCodeGenerator(self.llm))

        # Holder was created in __init__ and may already have opened during
        # memory initialization. Access once here to preserve early lock/fallback
        # detection before persistent stores are assembled.
        _ = self._db_holder.connection

        try:
            traj_store = TrajectoryStore(self._db_holder)
        except Exception as exc:
            logger.error("TrajectoryStore init failed: %s", exc)
            self._db_holder.close()
            raise SystemExit(f"\nFailed to initialize trajectory store: {exc}") from exc
        distiller: SkillDistiller
        if settings.has_llm_credentials:
            distiller = LLMSkillDistiller(self.llm)
        else:
            distiller = SkillDistiller()
        intent_inferrer = None
        if settings.has_llm_credentials and settings.intent_inference_enabled:
            from leapflow.analysis.intent_inferrer import LLMIntentInferrer
            intent_inferrer = LLMIntentInferrer(
                self.llm, language=settings.intent_inference_language
            )
        else:
            from leapflow.analysis.intent_inferrer import RuleBasedIntentInferrer
            intent_inferrer = RuleBasedIntentInferrer()

        perception_session = _build_visual_components(settings, self.rpc)
        self.perception_session = perception_session

        platform_hint = manifest.platform_id.value

        attention_filters = build_attention_filters(
            foreground_gate=settings.attention_foreground_gate,
            noise_patterns=settings.attention_noise_patterns,
            working_dir_inference=settings.attention_working_dir_inference,
            domain_whitelist=settings.attention_domain_whitelist,
            platform_hint=platform_hint,
            perceptual_field_enabled=settings.perceptual_field_enabled,
            perceptual_field_config=settings.perceptual_field_config,
        )

        # SurpriseAnnotator — event-level surprise detection (post-filter)
        surprise_annotator = None
        if settings.surprise_enabled:
            from leapflow.recording.attention import SurpriseAnnotator, SurpriseConfig
            surprise_annotator = SurpriseAnnotator(SurpriseConfig(
                stat_weight=settings.surprise_stat_weight,
                temporal_weight=settings.surprise_temporal_weight,
                pattern_weight=settings.surprise_pattern_weight,
                annotation_threshold=settings.surprise_annotation_threshold,
                warmup_events=settings.surprise_warmup_events,
            ))

        # Store intermediates for deferred phase
        self._critical_codegen = codegen
        self._critical_traj_store = traj_store
        self._critical_distiller = distiller
        self._critical_intent_inferrer = intent_inferrer
        self._critical_attention_filters = attention_filters
        self._critical_surprise_annotator = surprise_annotator

        # NOTE: ImitationPipeline, Video components are assembled in initialize_deferred()

        self.skill_lib = SkillLibraryStore(self._db_holder, audit_logger=self.audit)
        scorer = HeuristicSimilarityScorer()
        llm_scorer = LLMSimilarityScorer(self.llm) if settings.has_llm_credentials else None
        feedback_evaluator = FeedbackEvaluator(
            traj_store, llm=self.llm if settings.has_llm_credentials else None,
        )

        self.registry = build_default_registry(self.rpc, self.llm, self.wm, self.lt)
        
        # Store scorers for deferred phase
        self._critical_scorer = scorer
        self._critical_llm_scorer = llm_scorer
        self._critical_feedback_evaluator = feedback_evaluator
        
        # NOTE: World Model, SkillActivator, Learning Pipeline, Doc/Stored skills
        # are assembled in initialize_deferred()
        
        graph_planner = GraphPlanner(self.llm, self.registry) if settings.has_llm_credentials else None
        scheduler = TaskScheduler(
            self.registry, self.rpc, graph_planner=graph_planner,
        ) if graph_planner else None

        # Bind perception/execution to the desktop semantic plugin
        from leapflow.plugins import get_registry as _get_tool_registry
        _get_tool_registry().bind_runtime(perception=perception, execution=execution_adapter)
        logger.info("Desktop semantic plugin bound (perception=%s)", perception is not None)

        self._bind_hardware_plugin()
        await self._start_hardware_streams()

        # Initialize skill discovery (SkillIndex + SkillInjector)
        skills_dir = Path(settings.skills_dir).expanduser()
        skill_index = SkillIndex(skills_dir, min_quality=settings.skill_min_quality)
        self.skill_index = skill_index
        skill_injector = SkillInjector(skills_dir)
        configure_skill_discovery(
            skill_index, skill_injector,
            registry=self.registry,
            skill_view_max_chars=settings.skill_view_max_chars,
        )
        logger.info("Skill discovery initialized: %s", skills_dir)

        self.session_store = LearningSessionStore(self._db_holder)

        # NOTE: Learnability, SessionController are assembled in initialize_deferred()

        classifier: IntentClassifier = (
            LLMIntentClassifier(self.llm) if settings.has_llm_credentials else FallbackClassifier()
        )
        self.intent_classifier = classifier
        if settings.has_llm_credentials:
            self.assessor = LLMSituationalAssessor(self.llm)

        # NOTE: Copilot pipeline is assembled in initialize_deferred()
        self.copilot_pipeline = None
        self.copilot_idle = None
        self.copilot_encoder = None
        self.copilot_feedback = None
        self.copilot_evolution = None
        self.copilot_config = None

        # ── Wire memory tools into TOOL_HANDLERS (late binding) ──
        from leapflow.plugins import get_registry
        _tool_registry = get_registry()
        _tool_registry.set_memory_manager(self.memory)

        # ── Config tools: bind this Context so a config write reloads the live
        # session, the same way `/config set` does. Without it the write lands on
        # disk while the in-process settings keep the old value, and an immediate
        # read-back looks like the write failed.
        from leapflow.tools.config_tools import set_config_context
        set_config_context(self)

        # ── Gateway server (late-bound tool wiring) ──
        from leapflow.gateway.server import GatewayServer
        from leapflow.gateway.router import GatewayRouter
        from leapflow.gateway.events import (
            GatewayMessageReceived,
            GatewaySessionCreated,
            GatewaySessionEnded,
        )
        from leapflow.gateway.connectors.protocol import BackendEvent
        from leapflow.tools.gateway_tool import set_gateway_approval_gate

        async def _on_gateway_event(event: object) -> None:
            """Bridge gateway events to episodic memory and logging."""
            if isinstance(event, GatewayMessageReceived):
                logger.info(
                    "gateway.inbound platform=%s session=%s len=%d",
                    event.source.platform,
                    event.session_key or "(filtered)",
                    len(event.text),
                )
                episodic = self.memory.get_provider("episodic")
                if episodic is not None and hasattr(episodic, "ingest"):
                    episodic.ingest(
                        "gateway.message",
                        f"[{event.source.platform}:{event.source.user_name or event.source.user_id}] "
                        f"{event.text[:500]}",
                        metadata={
                            "platform": event.source.platform,
                            "session": event.session_key,
                        },
                    )
            elif isinstance(event, GatewaySessionCreated):
                logger.info("gateway.session_created key=%s", event.session_key)
            elif isinstance(event, GatewaySessionEnded):
                logger.info(
                    "gateway.session_ended key=%s reason=%s",
                    event.session_key,
                    event.reason,
                )
                router = getattr(self, "_gateway_router", None)
                if router is not None:
                    router.clear_session(event.session_key)
            elif isinstance(event, BackendEvent):
                logger.debug(
                    "gateway.signal platform=%s type=%s",
                    event.platform_id, event.event_type,
                )
                episodic = self.memory.get_provider("episodic")
                if episodic is not None and hasattr(episodic, "ingest"):
                    episodic.ingest(
                        "gateway.signal",
                        f"[{event.platform_id}] {event.event_type}",
                        metadata={
                            "platform": event.platform_id,
                            "event_type": event.event_type,
                            "event_id": event.event_id,
                        },
                    )

        from leapflow.gateway.checkpoint_store import DuckDBCheckpointStore, DuckDBDeduplicationStore
        from leapflow.gateway.event_bridge import GatewayEventBridge

        _checkpoint_store = DuckDBCheckpointStore(self._db_holder)
        _dedup_store = DuckDBDeduplicationStore(self._db_holder)
        self._gateway_event_bridge = GatewayEventBridge(self.event_bus)

        async def _on_gateway_event_with_bridge(event: object) -> None:
            await _on_gateway_event(event)
            await self._gateway_event_bridge.on_gateway_event(event)
            # N4: observe inbound messages for re-entry EVENT-trigger matches
            # (independent of agent activation; matcher is set by the daemon).
            observer = getattr(self, "_reentry_event_observer", None)
            if observer is not None and isinstance(event, GatewayMessageReceived):
                try:
                    await observer(
                        platform=event.source.platform,
                        chat=event.source.chat_id,
                        text=event.text,
                    )
                except Exception:
                    logger.debug("reentry event observer failed", exc_info=True)

        self.gateway_server = GatewayServer(
            settings.profile_layout,
            extra_manifest_dirs=[settings.profile_layout.gateway.manifests_dir],
            on_event=_on_gateway_event_with_bridge,
            checkpoint_store=_checkpoint_store,
            dedup_store=_dedup_store,
        )
        self.gateway_server.discover_manifests()
        _tool_registry.set_gateway_server(self.gateway_server)
        set_gateway_approval_gate(self._approval_orchestrator)
        # Config writes are gated too: several writable keys weaken safety
        # machinery (guardrail.enabled, confirm.default_level, codegen.sandbox),
        # so config_set must not be able to disable its own supervision.
        from leapflow.tools.config_tools import set_config_approval_gate
        set_config_approval_gate(self._approval_orchestrator)
        # Outbound fetches to internal targets (loopback, private ranges, cloud
        # instance metadata) need the same review; public reads stay unprompted.
        from leapflow.tools.web_fetch import set_web_approval_gate
        set_web_approval_gate(self._approval_orchestrator)

        self._register_gateway_normalizers(settings)

        async def _gateway_send(source: Any, text: str) -> None:
            await self.gateway_server.send_reply(source, text)

        async def _gateway_indicator(
            source: Any, message_id: str, phase: str,
        ) -> None:
            """Processing indicator via platform reactions (fire-and-forget)."""
            platform = getattr(source, "platform", "")
            if not platform or not message_id:
                return
            gw = self.gateway_server
            if phase == "start":
                await gw.execute_platform_action(
                    platform, "im.add_reaction",
                    {"message_id": message_id, "emoji_type": "OnIt"},
                )
            elif phase in ("done", "error"):
                await gw.execute_platform_action(
                    platform, "im.remove_reaction",
                    {"message_id": message_id, "emoji_type": "OnIt"},
                )

        async def _gateway_stream_send(
            source: Any, text: str, message_id: str,
        ) -> str:
            """Send or update a message for streaming replies.

            When *message_id* is empty, sends a new message and returns its ID.
            When *message_id* is provided, updates the existing message in place.
            """
            platform = getattr(source, "platform", "")
            chat_id = getattr(source, "chat_id", "")
            gw = self.gateway_server
            if not message_id:
                result = await gw.send_message(platform, chat_id, text)
                return str(result.get("message_id", ""))
            else:
                await gw.execute_platform_action(
                    platform, "im.update_message",
                    {"message_id": message_id, "text": text},
                )
                return message_id

        async def _gateway_context_fetch(
            platform: str, message_id: str, field: str,
        ) -> str:
            """Fetch parent message text via platform action for thread context."""
            if not platform or not message_id:
                return ""
            gw = self.gateway_server
            result = await gw.execute_platform_action(
                platform, "im.get_messages",
                {"message_ids": message_id},
            )
            if not result.get("ok"):
                return ""
            data = result.get("data", {})
            items = data.get("items") or data.get("messages") or []
            if isinstance(items, list) and items:
                msg = items[0] if isinstance(items[0], dict) else {}
                return str(msg.get("content") or msg.get("text") or "")
            return ""

        from leapflow.plugins import get_registry
        _tool_registry_gw = get_registry()

        self._gateway_router = GatewayRouter(
            llm=self.llm,
            system_prompt=(
                "You are LeapFlow, a helpful AI assistant responding "
                "through an external messaging platform.  Be concise "
                "and conversational."
            ),
            send_fn=_gateway_send,
            tool_definitions=_tool_registry_gw.tool_definitions,
            tool_handlers=_tool_registry_gw.tool_handlers,
            persistence=getattr(self, "_conversation_store", None),
            indicator_fn=_gateway_indicator,
            stream_send_fn=_gateway_stream_send,
            context_fetch_fn=_gateway_context_fetch,
        )
        self.gateway_server.set_message_handler(
            self._gateway_router.handle_message,
        )
        self.gateway_server.set_callback_handler(
            self._gateway_router.handle_callback,
        )

        # ── Build CompressorConfig with LLM callbacks ──

        from leapflow.engine.context_compressor import CompressorConfig

        async def _summarize_via_llm(prompt: str) -> str:
            from leapflow.llm.message_builder import build_user_message_text
            resp = await self.llm.achat(
                [build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return (resp.content or "").strip()

        ctx_len = self._effective_llm_context_length(settings)
        compressor_config = CompressorConfig(
            token_budget=max(1, int(ctx_len * settings.context_hard_limit_ratio)),
            context_length=ctx_len,
            threshold=settings.compress_threshold,
            keep_tail=settings.compress_keep_tail,
            max_output_chars=settings.max_tool_output_chars,
            summarize_fn=_summarize_via_llm if settings.has_llm_credentials else None,
        )

        # ── Initialize DuckDBConversationStore ──
        self._conversation_store = None
        try:
            from leapflow.storage.conversation_store import DuckDBConversationStore
            self._conversation_store = DuckDBConversationStore(self._db_holder)
            logger.info("ConversationStore initialized")
        except Exception:
            logger.warning("ConversationStore initialization failed", exc_info=True)

        # ── Initialize ResearchLedgerStore (S1 durable Orient) ──
        self._research_ledger_store = None
        try:
            from leapflow.storage.research_ledger_store import ResearchLedgerStore
            self._research_ledger_store = ResearchLedgerStore(self._db_holder)
            logger.info("ResearchLedgerStore initialized")
        except Exception:
            logger.warning("ResearchLedgerStore initialization failed", exc_info=True)

        # ── Initialize ReentryStore (S2 event-driven re-entry) ──
        self._reentry_store = None
        try:
            from leapflow.storage.reentry_store import ReentryStore
            self._reentry_store = ReentryStore(self._db_holder)
            logger.info("ReentryStore initialized")
        except Exception:
            logger.warning("ReentryStore initialization failed", exc_info=True)

        if self._conversation_store and hasattr(self, "_gateway_router"):
            self._gateway_router._persistence = self._conversation_store

        # ── Initialize MCP Manager + register tools into agent surface ──
        self._configure_mcp_manager(settings)

        # ── Unified approval gate wiring (shell, file, gateway) ──
        try:
            from leapflow.tools.shell_tools import set_approval_gate
            set_approval_gate(self._approval_orchestrator)
            logger.debug("Shell approval gate: action orchestrator mode")
        except Exception:
            logger.debug("Shell approval gate setup skipped", exc_info=True)
        try:
            from leapflow.plugins import get_registry
            _tool_reg_desktop = get_registry()
            _tool_reg_desktop.set_desktop_gate(self._approval_orchestrator)
            logger.debug("Desktop approval gate: action orchestrator mode")
        except Exception:
            logger.debug("Desktop approval gate setup skipped", exc_info=True)

        self.engine = AgentEngine(
            settings, self.rpc, self.llm, self.wm, self.lt, self.imm,
            self.registry, classifier,
            imitation=None,  # wired in initialize_deferred()
            skill_library=self.skill_lib,
            graph_planner=graph_planner,
            scheduler=scheduler,
            perception=perception,
            execution=execution_adapter,
            skill_activator=None,  # wired in initialize_deferred()
            session=None,  # wired in initialize_deferred()
            vlm=self.vlm,
            memory_manager=self.memory,
            evolution=self._evolution,
            skill_injector=skill_injector,
            skill_index=skill_index,
        )

        # ── Wire CompressorConfig with archive_fn into engine ──
        from leapflow.engine.context_compressor import ContextCompressor

        async def _archive_to_semantic(messages: List[Dict[str, Any]]) -> None:
            """Archive evicted messages to SemanticMemoryProvider."""
            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "")
                if content and isinstance(content, str) and len(content) > 20:
                    self.lt.insert_raw(
                        f"archived_{role}",
                        content[:2000],
                        metadata={"source": "compression_archive", "role": role},
                    )

        compressor_config.archive_fn = _archive_to_semantic
        self.engine._compressor = ContextCompressor(compressor_config)

        # ── Enable PrefixCacheOptimizer ──
        from leapflow.engine.prompt_cache import PrefixCacheOptimizer
        self.engine.set_cache_strategy(PrefixCacheOptimizer())

        # ── Wire ConversationStore into engine for session persistence ──
        if self._conversation_store:
            self.engine.set_conversation_store(self._conversation_store)

        # ── Wire ResearchLedgerStore into engine (S1 durable Orient) ──
        if self._research_ledger_store:
            self.engine.set_research_ledger_store(self._research_ledger_store)

        # ── Wire ReentryStore into engine (S2 event-driven re-entry) ──
        if self._reentry_store:
            self.engine.set_reentry_store(self._reentry_store)

        # ── Wire SubagentManager + delegate_task tool ──
        try:
            from leapflow.engine.subagent import DefaultSubagentExecutor, SubagentManager
            from leapflow.plugins import get_registry
            _tool_reg_sub = get_registry()
            _tool_reg_sub.assemble()  # idempotent: no-op once assembled
            _TD = _tool_reg_sub.tool_definitions
            _TH = _tool_reg_sub.tool_handlers
            if getattr(settings, "agent_subagent_full_loop", False):
                # Opt-in: subagents run the engine's full adaptive loop on an
                # isolated child frame (state-isolated via per-frame swap).
                from leapflow.engine.subagent import EngineFrameSubagentExecutor
                sub_executor = EngineFrameSubagentExecutor(
                    run_child=self.engine._run_subagent_goal,
                    tool_names=list(_TH.keys()),
                    settings=settings,
                )
            else:
                sub_executor = DefaultSubagentExecutor(
                    llm=self.llm,
                    tool_handlers=_TH,
                    tool_definitions=_TD,
                    settings=settings,
                )
            self._subagent_manager = SubagentManager(
                executor=sub_executor,
                max_depth=settings.agent_subagent_max_depth,
                max_concurrent=settings.agent_subagent_max_concurrent,
            )
            _tool_reg_sub.set_subagent_manager(self._subagent_manager)
            logger.info("SubagentManager wired with delegate_task tool")
        except Exception:
            self._subagent_manager = None
            logger.debug("SubagentManager setup skipped", exc_info=True)

        # NOTE: EvolutionStore + calibration are in initialize_deferred()
        self._evolution_store = None

        # ── Wire tool loop guardrails (progress-aware; thresholds from config) ──
        try:
            if getattr(settings, "guardrail_enabled", True):
                from leapflow.engine.tool_guardrails import CompositeGuardrail
                self.engine._guardrail = CompositeGuardrail(
                    max_repeats=settings.guardrail_max_repeats,
                    stagnation_window=settings.guardrail_stagnation_window,
                    min_success_rate=settings.guardrail_min_success_rate,
                    max_consecutive_same=settings.guardrail_max_consecutive_same,
                )
                logger.debug("Tool loop guardrails enabled")
            else:
                self.engine._guardrail = None
                logger.debug("Tool loop guardrails disabled by config")
        except Exception:
            logger.debug("Tool guardrails setup skipped", exc_info=True)

        # ── Seamless ripgrep provisioning for code_search (best-effort, background) ──
        # code_search always works via the pure-Python fallback; this just tries to
        # provision the faster ripgrep backend without blocking startup or searches.
        # The attempt is persisted in the profile cache so daemon restarts never
        # re-trigger the installer's process storm after a failed install.
        try:
            if getattr(settings, "tools_ripgrep_autoinstall", True):
                import threading
                from leapflow.tools.file_operations import ensure_ripgrep_available
                provision_marker = settings.profile_layout.cache.profile_dir / "ripgrep_provision.json"
                threading.Thread(
                    target=ensure_ripgrep_available,
                    kwargs={"autoinstall": True, "marker_path": provision_marker},
                    daemon=True,
                ).start()
        except Exception:
            logger.debug("ripgrep background provisioning skipped", exc_info=True)

        # ── Wire developer verification + terminal-session tool config ──
        try:
            from leapflow.tools.dev_tools import set_dev_commands
            set_dev_commands(
                test_command=getattr(settings, "tools_test_command", "") or "",
                lint_command=getattr(settings, "tools_lint_command", "") or "",
            )
            from leapflow.tools.terminal_session import set_terminal_sessions_enabled
            set_terminal_sessions_enabled(bool(getattr(settings, "tools_terminal_session_enabled", False)))
            from leapflow.tools.file_operations import set_edit_verification
            set_edit_verification(bool(getattr(settings, "tools_verify_edits", True)))
        except Exception:
            logger.debug("dev/terminal tool config wiring skipped", exc_info=True)

        # ── Wire Smart Approval (auxiliary LLM for command risk) ──
        if self.auxiliary is not None:
            try:
                aux = self.auxiliary

                class _SmartApprovalGate:
                    """LLM-assisted shell approval adapter that preserves policy authority."""

                    def __init__(self, delegate: Any) -> None:
                        self._delegate = delegate

                    async def evaluate(self, action: Any) -> Any:
                        return await self._delegate.evaluate(action)

                    async def check(self, command: str) -> bool:
                        try:
                            risk = await aux.classify_risk(command)
                        except Exception:
                            risk = 0.5
                        if risk < 0.3:
                            logger.debug("smart_approval: low auxiliary risk hint (risk=%.2f)", risk)
                        return await self._delegate.check(command)

                from leapflow.tools.shell_tools import set_approval_gate
                set_approval_gate(_SmartApprovalGate(self._approval_orchestrator))
                logger.debug("Smart approval gate enabled (auxiliary LLM)")
            except Exception:
                logger.debug("Smart approval setup skipped", exc_info=True)

        # ── Wire File Read Approval Gate ──
        try:
            from leapflow.security.actions import ActionDescriptor
            from leapflow.plugins import get_registry
            _tool_reg_fread = get_registry()

            approval_orchestrator = self._approval_orchestrator

            class _FileReadGate:
                """File read approval via the action approval orchestrator."""

                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(
                    self,
                    path: str,
                    mode: str = "raw",
                    sensitivity_meta: dict | None = None,
                ) -> bool:
                    meta = dict(sensitivity_meta or {})
                    action = ActionDescriptor.file_read(path, mode=mode, metadata=meta)
                    result = await approval_orchestrator.evaluate(action)
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            _tool_reg_fread.set_file_read_gate(_FileReadGate())
            logger.debug("File read approval gate: action orchestrator")
        except Exception:
            logger.debug("File read gate setup skipped", exc_info=True)

        # ── Wire File Write Approval Gate ──
        try:
            from leapflow.security.actions import ActionDescriptor
            from leapflow.plugins import get_registry
            _tool_reg_fwrite = get_registry()

            approval_orchestrator = self._approval_orchestrator

            class _FileWriteGate:
                """File write approval via the action approval orchestrator."""

                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(
                    self,
                    path: str,
                    content: str,
                    mode: str = "overwrite",
                    sensitivity_meta: dict | None = None,
                ) -> bool:
                    meta = dict(sensitivity_meta or {})
                    action = ActionDescriptor.file_write(path, content, mode=mode, metadata=meta)
                    result = await approval_orchestrator.evaluate(action)
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            _tool_reg_fwrite.set_file_write_gate(_FileWriteGate())
            logger.debug("File write approval gate: action orchestrator")
        except Exception:
            logger.debug("File write gate setup skipped", exc_info=True)

        # ── Token budget and model capability metadata ─────────────────────
        if self.engine is not None:
            self._sync_engine_runtime_budget(settings)
            self.engine.set_stale_stream_timeout(settings.stale_stream_timeout_s)
            self.engine.set_default_tool_timeout(settings.default_tool_timeout_s)

            # NOTE: evolution_store and experience_store wired in initialize_deferred()

            if hasattr(self, "doc_store") and self.doc_store is not None:
                self.engine.set_doc_store(self.doc_store)

            self.engine.set_event_bus(self.event_bus)

        # ── Register persisted session tools ──
        if self._conversation_store:
            try:
                from leapflow.daemon.session_coordinator import SessionCoordinator
                from leapflow.plugins import get_registry
                _tool_reg_session = get_registry()
                _tool_reg_session.assemble()  # idempotent: no-op once assembled
                conv_store = self._conversation_store
                session_reader = SessionCoordinator()
                fallback_workspace_cwd = str(Path(str(getattr(self.settings, "workspace_root", "") or os.getcwd())).expanduser().resolve())

                def _json_result(payload: Any) -> str:
                    import json as _json_sessions
                    return _json_sessions.dumps(payload, ensure_ascii=False)

                def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
                    try:
                        parsed = int(value)
                    except (TypeError, ValueError):
                        parsed = default
                    return min(max(parsed, minimum), maximum)

                # ── Register session_search tool ──
                async def _session_search_handler(params: dict) -> dict:
                    query = str(params.get("query", "") or "")
                    limit = _bounded_int(params.get("limit", 5), 5, minimum=1, maximum=50)
                    if not query:
                        return {"ok": False, "error": "Missing query parameter"}
                    workspace_cwd = _active_tool_workspace_root(fallback_workspace_cwd)
                    results = conv_store.search_messages(query, limit=limit, cwd=workspace_cwd)
                    if not results:
                        return {"ok": True, "result": "No matching sessions found."}
                    items = [
                        {
                            "session_id": r.session_id,
                            "session": r.session_title or r.session_id[:8],
                            "session_title": r.session_title,
                            "message_id": r.message_id,
                            "role": r.role,
                            "content": r.content[:300],
                            "score": round(r.score, 3),
                            "created_at": r.created_at,
                            "date": _format_ts(r.created_at),
                        }
                        for r in results
                    ]
                    return {"ok": True, "result": _json_result(items)}

                _tool_reg_session.register_late_tool(
                    {
                        "type": "function",
                        "function": {
                            "name": "session_search",
                            "description": "Search past conversation sessions in the current workspace for relevant context.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search keywords"},
                                    "limit": {"type": "integer", "description": "Max results (default: 5)"},
                                },
                                "required": ["query"],
                            },
                        },
                    },
                    _session_search_handler,
                    "session_search",
                )
                logger.debug("session_search tool registered")

                # ── Register session_list tool ──
                def _format_ts(ts: float) -> str:
                    if not ts:
                        return ""
                    import datetime
                    try:
                        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    except (TypeError, ValueError, OSError):
                        return str(ts)[:16]

                async def _session_list_handler(params: dict) -> dict:
                    limit = _bounded_int(params.get("limit", 10), 10, minimum=1, maximum=30)
                    workspace_cwd = _active_tool_workspace_root(fallback_workspace_cwd)
                    sessions = conv_store.list_sessions(limit=limit, active_only=False, cwd=workspace_cwd)
                    items = []
                    for s in sessions:
                        session_id = str(getattr(s, "session_id", "") or "")
                        item = {
                            "session_id": session_id,
                            "title": getattr(s, "title", "") or session_id[:8],
                            "date": _format_ts(getattr(s, "updated_at", 0) or getattr(s, "created_at", 0)),
                            "created_at": getattr(s, "created_at", 0),
                            "updated_at": getattr(s, "updated_at", 0),
                            "messages": getattr(s, "message_count", 0),
                            "cwd": getattr(s, "cwd", "") or "",
                        }
                        summary = getattr(s, "summary", "")
                        if summary:
                            item["summary"] = summary[:200]
                        items.append(item)
                    return {"ok": True, "result": _json_result(items)}

                _tool_reg_session.register_late_tool(
                    {
                        "type": "function",
                        "function": {
                            "name": "session_list",
                            "description": (
                                "List recent conversation sessions in the current workspace with stable ids, titles, dates, and summaries. "
                                "Use for browsing past tasks or when user asks to see history without specific search terms. "
                                "No keywords needed \u2014 returns chronological list."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "limit": {"type": "integer", "description": "Max sessions to return (default: 10, max: 30)"},
                                },
                                "required": [],
                            },
                        },
                    },
                    _session_list_handler,
                    "session_list",
                )
                logger.debug("session_list tool registered")

                # ── Register session_detail tool ──
                async def _session_detail_handler(params: dict) -> dict:
                    session_id = str(params.get("session_id", "") or "").strip()
                    limit = _bounded_int(params.get("limit", 200), 200, minimum=1, maximum=1000)
                    offset = _bounded_int(params.get("offset", 0), 0, minimum=0, maximum=1_000_000)
                    include_inactive = bool(params.get("include_inactive", True))
                    workspace_cwd = _active_tool_workspace_root(fallback_workspace_cwd)
                    detail = await session_reader.get_detail(
                        self,
                        self.settings,
                        session_id,
                        limit=limit,
                        offset=offset,
                        include_inactive=include_inactive,
                        workspace_root=workspace_cwd,
                    )
                    ok = bool(detail.get("ok", False))
                    response = {"ok": ok, "result": _json_result(detail)}
                    if not ok:
                        response["error"] = str(detail.get("error", "session detail unavailable"))
                    return response

                _tool_reg_session.register_late_tool(
                    {
                        "type": "function",
                        "function": {
                            "name": "session_detail",
                            "description": (
                                "Read a paginated persisted transcript for one past conversation session in the current workspace. "
                                "Use after session_list or session_search returns a session_id."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "session_id": {"type": "string", "description": "Exact session_id from session_list or session_search"},
                                    "limit": {"type": "integer", "description": "Max messages to return (default: 200, max: 1000)"},
                                    "offset": {"type": "integer", "description": "Message offset for pagination (default: 0)"},
                                    "include_inactive": {"type": "boolean", "description": "Include inactive or compacted messages (default: true)"},
                                },
                                "required": ["session_id"],
                            },
                        },
                    },
                    _session_detail_handler,
                    "session_detail",
                )
                logger.debug("session_detail tool registered")
            except Exception:
                logger.debug("persisted session tool registration failed", exc_info=True)

        # NOTE: PipelineObserver, ObservationDaemon, ColdStart, PatternMiner,
        # ImplicitFeedback are assembled in initialize_deferred()

    async def initialize_deferred(self) -> None:
        """Deferred initialization: rich features that can run in background.

        Assembles: ImitationPipeline, World Model, Skill System full load,
        Learning Pipeline, Copilot, Observers. Components auto-initialize
        on first use via _ensure_deferred() if this hasn't completed.
        """
        settings = self.settings
        perception = self._platform_perception
        execution_adapter = self._platform_execution
        perception_session = self.perception_session
        codegen = self._critical_codegen
        traj_store = self._critical_traj_store
        distiller = self._critical_distiller
        intent_inferrer = self._critical_intent_inferrer
        attention_filters = self._critical_attention_filters
        surprise_annotator = self._critical_surprise_annotator
        scorer = self._critical_scorer
        llm_scorer = self._critical_llm_scorer
        feedback_evaluator = self._critical_feedback_evaluator

        # Give leapd's control plane a scheduling point before deferred init
        # begins constructing optional subsystems; this task is background work.
        await asyncio.sleep(0)

        # ── Video-mode components ──
        video_recorder = None
        video_analyzer = None
        video_segmenter = None
        signal_timeline = None
        if settings.recording_mode.uses_video and settings.visual_track_enabled:
            if settings.has_vlm_credentials:
                video_recorder, video_analyzer, video_segmenter, signal_timeline = (
                    _build_video_components(settings, self.rpc, self.vlm or self.llm)
                )
            else:
                message = (
                    "Video analysis disabled: LEAPFLOW_VLM_API_KEY or "
                    "LEAPFLOW_LLM_API_KEY is required for visual recording mode."
                )
                logger.warning(message)
                _emit_status(message)

        # ── ImitationPipeline full assembly ──
        from leapflow.analysis.abstractor import ActionAbstractor
        platform_hint = getattr(self._platform_manifest, 'platform_id', None)
        platform_hint = platform_hint.value if platform_hint else "darwin"
        abstractor = ActionAbstractor(platform_hint=platform_hint)

        self.imitation = ImitationPipeline(
            store=traj_store, distiller=distiller, codegen=codegen,
            intent_inferrer=intent_inferrer,
            abstractor=abstractor,
            perception_session=perception_session,
            goal_relevance_threshold=settings.attention_goal_relevance_threshold,
            attention_filters=attention_filters,
            surprise_annotator=surprise_annotator,
            rpc=self.rpc,
            event_bus=self.event_bus,
            text_capture_enabled=settings.text_capture_enabled,
            text_capture_exclude_apps=settings.text_capture_exclude_apps,
            text_capture_secure_roles=settings.text_capture_secure_roles,
            text_capture_max_length=settings.text_capture_max_length,
            clipboard_max_length=settings.clipboard_max_length,
            recording_mode=settings.recording_mode,
            mhms_fusion_enabled=settings.mhms_fusion_enabled,
            video_recorder=video_recorder,
            video_analyzer=video_analyzer,
            video_segmenter=video_segmenter,
            signal_timeline=signal_timeline,
            observation_daemon=self._observation_daemon,
            recording_profile=_default_recording_profile(settings),
        )
        self.event_bus.subscribe(self.imitation.recorder.on_event)

        if perception_session:
            perception_session._recording_context = self.imitation.recorder.attention_context
            perception_session.set_recording_mode(settings.recording_mode)
            self.event_bus.subscribe(perception_session.on_system_event)

        if settings.recording_mode.uses_video and signal_timeline is not None:
            if self.event_bus is not None:
                self.event_bus.subscribe(signal_timeline.record_event)
                logger.info("EventBus -> SignalTimeline subscription established")
        elif settings.recording_mode.uses_video and signal_timeline is None:
            logger.warning(
                "SignalTimeline is None — skipping EventBus subscription "
                "(video mode active but timeline unavailable)"
            )

        # Yield to the event loop so heartbeats/RPC callbacks stay responsive
        # during this long synchronous initialization (see daemon keepalive).
        await asyncio.sleep(0)

        # ── World Model assembly ──
        if settings.prediction_enabled:
            from leapflow.world_model import (
                LearningBudgetController,
                ExperienceStore,
                CuriosityConfig,
                CuriositySignal,
                PredictionLoop,
                ExperienceReplayEngine,
                TrajectoryGrader,
            )
            from leapflow.perception.state_snapshot import StateSnapshotService

            self.learning_budget = LearningBudgetController(
                prediction_budget=settings.prediction_budget,
                comparison_budget=settings.comparison_budget,
                replay_budget=settings.replay_budget,
                grading_budget=settings.grading_budget,
                distillation_budget=settings.distillation_budget,
                discovery_baseline=settings.budget_discovery_baseline,
                regression_baseline=settings.budget_regression_baseline,
            )

            embedding_provider = None
            if settings.semantic_embedding_provider != "none":
                from leapflow.world_model.embedding import (
                    TFIDFEmbeddingProvider,
                    LLMEmbeddingProvider,
                )
                if settings.semantic_embedding_provider == "llm":
                    embedding_provider = LLMEmbeddingProvider(self.llm)
                else:
                    embedding_provider = TFIDFEmbeddingProvider()

            self.experience_store = ExperienceStore(
                self.lt,
                embedding_provider=embedding_provider,
                semantic_weight=settings.semantic_rerank_weight,
            )
            self._bind_hardware_experience()
            self.snapshot_service = StateSnapshotService(self.rpc, self.imm)
            self.curiosity = CuriositySignal(
                CuriosityConfig(
                    alpha=settings.curiosity_alpha,
                    beta=settings.curiosity_beta,
                    gamma=settings.curiosity_gamma,
                    auto_balance=settings.curiosity_auto_balance,
                ),
                experience_store=self.experience_store,
            )
            self.prediction_loop = PredictionLoop(
                llm=self.llm,
                snapshot_service=self.snapshot_service,
                experience_store=self.experience_store,
                budget=self.learning_budget,
                enabled=settings.prediction_enabled,
                delta_threshold=settings.prediction_delta_threshold,
                structural_blend_weight=settings.prediction_structural_blend,
                semantic_blend_weight=settings.prediction_semantic_blend,
                semantic_compare_threshold=settings.prediction_semantic_threshold,
                rag_advantage_floor=settings.prediction_rag_advantage_floor,
                failure_advantage=settings.prediction_failure_advantage,
            )
            insight_callback = self._build_insight_callback()
            self.replay_engine = ExperienceReplayEngine(
                llm=self.llm,
                experience_store=self.experience_store,
                budget=self.learning_budget,
                on_insight=insight_callback,
                regression_sample_size=settings.replay_regression_sample_size,
            )
            self.trajectory_grader = TrajectoryGrader(
                llm=self.llm,
                experience_store=self.experience_store,
                budget=self.learning_budget,
            )
            self.registry.set_prediction_loop(self.prediction_loop)

            if perception_session is not None:
                self.curiosity.set_causal_graph(perception_session.causal_graph)
                freq = perception_session.causal_graph.metadata.get("frequency_counter")
                if freq:
                    self.curiosity.load_frequency_counter(freq)

            _ss = self.snapshot_service
            def _on_focus_for_snapshot(event: Any) -> None:
                if getattr(event, "event_type", "") == "app.focus_change":
                    bid = event.payload.get("bundle_id", "")
                    title = event.payload.get("window_title", "")
                    if bid:
                        _ss.update_focus(bid, title)
            self.event_bus.subscribe(_on_focus_for_snapshot)

            logger.info("World model initialized (prediction + curiosity + replay + OPD grading)")

        # Event-loop yield point after world model assembly
        await asyncio.sleep(0)

        # ── SkillActivator ──
        activator = None
        if perception and execution_adapter:
            activator = SkillActivator(
                self.registry, self.skill_lib, execution_adapter, perception,
                codegen=codegen,
            )
            # Heavy DuckDB skill-library load: run off the event loop
            n_activated = await self._run_deferred_db(activator.load_and_activate_all)
            if n_activated:
                logger.info("Activated %d learned skills from library", n_activated)
        self._critical_activator = activator

        # Event-loop yield point after heavy skill-library DuckDB loading
        await asyncio.sleep(0)

        # ── Learning Pipeline ──
        from leapflow.analysis.consensus import MultiTrajectoryDistiller
        consensus_distiller = MultiTrajectoryDistiller(self.imitation)

        self.doc_store = SkillDocStore(settings.skills_dir)
        doc_generator: Optional[CompositeSkillDocGenerator] = None
        if settings.has_llm_credentials:
            doc_generator = CompositeSkillDocGenerator(
                llm_generator=LLMSkillDocGenerator(self.llm),
            )
        else:
            doc_generator = CompositeSkillDocGenerator()

        self.active_observer = ActiveLearningObserver(
            self.skill_lib, scorer, self.wm,
            llm_scorer=llm_scorer,
            feedback_evaluator=feedback_evaluator,
            skill_activator=activator,
            consensus_distiller=consensus_distiller,
            doc_generator=doc_generator,
            doc_store=self.doc_store,
            skill_registry=self.registry,
            llm=self.llm,
            execution=execution_adapter,
        )
        observer = self.active_observer
        self.imitation.set_on_candidates_ready(observer.on_candidates_ready)

        # Wire curiosity signal from world model → active learning + attention tuner
        if self.prediction_loop is not None and self.curiosity is not None:
            _es = self.experience_store

            from leapflow.recording.attention_tuner import AttentionTuner
            pf_filter = None
            for f in attention_filters:
                if type(f).__name__ == "PerceptualFieldFilter":
                    pf_filter = f
                    break
            self.attention_tuner = AttentionTuner(
                self.imitation.recorder.attention_context,
                perceptual_filter=pf_filter,
                curiosity_expand_threshold=settings.attention_curiosity_expand_threshold,
                accuracy_contract_threshold=settings.attention_accuracy_contract_threshold,
            )
            _tuner = self.attention_tuner

            _ctx = self

            def _on_prediction_outcome(outcome: Any) -> None:
                score = _ctx.curiosity.compute(outcome)
                exp_id = getattr(outcome, "experience_id", "")
                if exp_id and _es is not None:
                    _es.update_curiosity_score(exp_id, score.total)
                _tuner.on_curiosity_signal(score, outcome)
                observer.on_curiosity_signal(score, outcome)

                delta = getattr(outcome, "delta", 0.0)
                evo_policy = _ctx._evolution_policy
                skill_lib = _ctx.skill_lib
                if evo_policy is not None and skill_lib is not None:
                    action_desc = ""
                    pred = getattr(outcome, "prediction", None)
                    if pred is not None:
                        action_desc = getattr(pred, "action_description", "")
                    skill_title = action_desc.split(":", 1)[-1] if ":" in action_desc else action_desc
                    if skill_title and (delta > 0.4 or delta < 0.15):
                        try:
                            stored = skill_lib.load_skill_by_title(skill_title)
                            if stored:
                                evo_outcome = evo_policy.on_execution_result(
                                    stored.title,
                                    success=(delta < 0.2),
                                    duration_s=0.0,
                                    current_confidence=stored.confidence,
                                    current_version=stored.version,
                                )
                                skill_lib.update_skill_confidence(
                                    stored.title, evo_outcome.new_confidence
                                )
                                if evo_outcome.tier_changed:
                                    logger.info(
                                        "Delta-driven evolution: '%s' confidence → %.3f",
                                        stored.title, evo_outcome.new_confidence,
                                    )
                        except Exception:
                            logger.debug("delta-driven evolution update failed", exc_info=True)

            self.prediction_loop._on_outcome = _on_prediction_outcome

        # ── Doc skills + stored fallbacks ──
        n_doc_skills = 0
        # SKILL.md loading is blocking file/DB IO: run off the event loop
        doc_skills = await self._run_deferred_db(
            lambda: list(self.doc_store.load_all_as_skills(
                self.llm, execution=execution_adapter, perception=perception,
            ))
        )
        for skill in doc_skills:
            self.registry.register(skill)
            n_doc_skills += 1
        if n_doc_skills:
            logger.info("Registered %d SKILL.md skills", n_doc_skills)

        n_fallback = await self._run_deferred_db(
            lambda: _register_stored_skill_fallbacks(
                self.skill_lib, self.registry, self.llm,
            )
        )
        if n_fallback:
            logger.info("Registered %d stored skills as fallback", n_fallback)

        # Event-loop yield point after doc-skill/fallback registration
        await asyncio.sleep(0)

        # ── Learnability + SessionController ──
        from leapflow.engine.confirmation import ConfirmationHandler
        confirmation = ConfirmationHandler(skill_store=self.skill_lib)

        learnability_assessor = None
        if settings.learnability_enabled:
            from leapflow.learning.learnability import DefaultLearnabilityAssessor, LearnabilityConfig
            learnability_config = LearnabilityConfig(
                min_steps=settings.learnability_min_steps,
                min_duration_s=settings.learnability_min_duration_s,
                max_idle_ratio=settings.learnability_max_idle_ratio,
                min_action_diversity=settings.learnability_min_action_diversity,
                learn_threshold=settings.learnability_learn_threshold,
                ask_threshold=settings.learnability_ask_threshold,
                vlm_enabled=settings.learnability_vlm_enabled,
                llm_enabled=settings.learnability_llm_enabled,
                rule_weight=settings.learnability_rule_weight,
                vlm_weight=settings.learnability_vlm_weight,
                llm_weight=settings.learnability_llm_weight,
            )
            learnability_assessor = DefaultLearnabilityAssessor(
                llm=self.llm if settings.has_llm_credentials else None,
                vlm=self.vlm,
                config=learnability_config,
            )

        self._evolution_policy = EMAConfidencePolicy()
        self.session = SessionController(
            self.imitation,
            self.registry,
            idle_timeout=settings.learn_idle_timeout,
            auto_learn=settings.learn_auto_distill,
            confirmation=confirmation,
            audit=self.audit,
            storage_path=str(settings.duckdb_path),
            audit_log_path=str(settings.audit_log_path),
            active_learning_observer=observer,
            session_store=self.session_store,
            learnability_assessor=learnability_assessor,
            evolution_policy=self._evolution_policy,
            skill_store=self.skill_lib,
        )

        # ── Wire deferred components to engine ──
        if self.engine is not None:
            self.engine._imitation = self.imitation
            self.engine._session = self.session
            if activator:
                self.engine._skill_activator = activator
            if self.experience_store is not None:
                self.engine.set_experience_store(self.experience_store)

        # Event-loop yield point after SessionController/engine wiring
        await asyncio.sleep(0)

        # ── EvolutionStore (DuckDB persistence for skill episodes) ──
        try:
            from leapflow.storage.evolution_store import DuckDBEvolutionStore

            def _load_evolution_store() -> "tuple[Any, list[dict[str, Any]]]":
                # Construction runs schema init; both are blocking DuckDB work
                store = DuckDBEvolutionStore(self._db_holder)
                episodes = store.load_recent_episodes(
                    limit=settings.memory_evolution_max_episodes,
                )
                return store, episodes

            self._evolution_store, persisted = await self._run_deferred_db(
                _load_evolution_store
            )
            for ep in persisted:
                self._evolution.record_episode(
                    skill_name=ep["skill_name"],
                    actions=ep["actions"],
                    outcome=ep["outcome"],
                    reward=ep["reward"],
                    context=ep.get("context"),
                    episode_id=ep["episode_id"],
                    timestamp=ep.get("timestamp"),
                )
            if persisted:
                logger.info("Evolution: hydrated %d episodes from DuckDB", len(persisted))
            self._evolution._persistent_store = self._evolution_store
        except Exception:
            logger.debug("EvolutionStore initialization skipped", exc_info=True)

        # Calibration
        try:
            if getattr(settings, "agent_calibration_enabled", False) and self._evolution_store is not None:
                self.engine.set_calibration_store(self._evolution_store)
                diff_result = await self._run_deferred_db(
                    lambda: self.engine.recalibrate_difficulty(self._evolution_store)
                )
                if getattr(diff_result, "applied", False):
                    logger.info("Difficulty calibration applied: %s", getattr(diff_result, "reason", ""))
                thr_result = await self._run_deferred_db(
                    lambda: self.engine.recalibrate_thresholds(self._evolution_store)
                )
                if getattr(thr_result, "applied", False):
                    logger.info("Threshold calibration applied: %s", getattr(thr_result, "reason", ""))
        except Exception:
            logger.debug("Difficulty/threshold calibration skipped", exc_info=True)

        if self._evolution_store and self.engine is not None:
            self.engine.set_evolution_store(self._evolution_store)

        # Event-loop yield point after EvolutionStore hydration/calibration
        await asyncio.sleep(0)

        # ── Copilot pipeline ──
        if settings.copilot_enabled:
            from leapflow.copilot import (
                CopilotConfig,
                ContextEncoder,
                CopilotEventSubscriber,
                PredictionEngine,
                SpeculativePipeline,
                IdleDetector,
                FeedbackCollector,
                EvolutionLoop,
            )
            from leapflow.copilot.predictors import (
                L0HashPredictor,
                L1MarkovPredictor,
            )

            copilot_config = CopilotConfig(
                enabled=True,
                action_ring_size=settings.copilot_action_ring_size,
                min_idle_ms=settings.copilot_min_idle_ms,
                max_idle_ms=settings.copilot_max_idle_ms,
                cache_ttl_seconds=settings.copilot_cache_ttl_s,
                speculative_cache_size=settings.copilot_speculative_cache_size,
            )

            copilot_encoder = ContextEncoder(copilot_config)
            from leapflow.copilot.predictors.l0_hash import InMemoryContextHashStore

            l0_store = InMemoryContextHashStore()
            if hasattr(self, 'lt') and self.lt is not None:
                from leapflow.copilot.adapters import SemanticHashAdapter
                l0_store = SemanticHashAdapter(self.lt)

            l1_markov = L1MarkovPredictor()
            self._l1_markov = l1_markov
            # Semantic-memory DuckDB read: run off the event loop
            await self._run_deferred_db(lambda: self._hydrate_l1_markov(l1_markov))

            predictors = [
                L0HashPredictor(l0_store),
                l1_markov,
            ]
            if hasattr(self, 'experience_store') and self.experience_store is not None:
                from leapflow.copilot.adapters import ExperienceEmbedAdapter
                from leapflow.copilot.predictors.l2_embed import L2EmbeddingPredictor
                from leapflow.copilot.predictors.l3_llm import L3LLMPredictor

                l2_provider = ExperienceEmbedAdapter(self.experience_store)
                predictors.append(L2EmbeddingPredictor(l2_provider))

                if settings.has_llm_credentials:
                    from leapflow.copilot.adapters import MemoryRAGAdapter as _RAG
                    rag_provider = _RAG(self.wm, self.experience_store)

                    class _CopilotLLMClient:
                        def __init__(self, llm):
                            self._llm = llm
                        async def complete(self, prompt: str) -> str:
                            from leapflow.llm.message_builder import build_user_message_text
                            resp = await self._llm.achat(
                                [build_user_message_text(prompt)], stream=False,
                            )
                            return resp.content or ""

                    predictors.append(L3LLMPredictor(
                        _CopilotLLMClient(self.llm), rag_provider=rag_provider,
                    ))

            copilot_engine = PredictionEngine(predictors, copilot_config)
            copilot_pipeline = SpeculativePipeline(copilot_engine, copilot_config)
            copilot_feedback = FeedbackCollector()
            copilot_evolution = EvolutionLoop(copilot_config, predictors)

            async def _copilot_on_idle(duration_ms: int) -> None:
                pass

            copilot_idle = IdleDetector(copilot_config, on_idle=_copilot_on_idle)

            warmup_raw = getattr(settings, "copilot_warmup_event_types", "")
            warmup_types = frozenset(k.strip() for k in warmup_raw.split(",") if k.strip()) if warmup_raw else None
            copilot_subscriber = CopilotEventSubscriber(
                copilot_encoder,
                tracker=None,
                working_memory=self.wm if hasattr(self, 'wm') else None,
                pipeline=copilot_pipeline,
                warmup_event_types=warmup_types,
            )
            self.event_bus.subscribe(copilot_subscriber.on_system_event)

            self.copilot_pipeline = copilot_pipeline
            self.copilot_idle = copilot_idle
            self.copilot_encoder = copilot_encoder
            self.copilot_feedback = copilot_feedback
            self.copilot_evolution = copilot_evolution
            self.copilot_config = copilot_config
            logger.info("Copilot initialized: L0+L1 predictors active")

        # Event-loop yield point after copilot assembly (L1 Markov hydration)
        await asyncio.sleep(0)

        # Pipeline Observer (A6: learning pipeline observability)
        from leapflow.engine.pipeline_observer import StructuredPipelineLogger
        self._pipeline_observer = StructuredPipelineLogger()

        # ObservationDaemon conditional auto-start (A4)
        if settings.observer_auto_start and not self.effective_mock:
            from leapflow.platform.observers import ObserverConfig
            from leapflow.platform.observers.daemon import ObservationDaemon
            observer_config = ObserverConfig(enabled=settings.observer_enabled_set)
            self._observation_daemon = ObservationDaemon(
                bus=self.event_bus, config=observer_config,
            )
            try:
                await self._observation_daemon.start()
                logger.info("ObservationDaemon auto-started: %s", self._observation_daemon.status)
            except Exception:
                logger.warning("ObservationDaemon auto-start failed", exc_info=True)
                self._observation_daemon = None

        # ColdStartManager: adaptive threshold management
        from leapflow.learning.cold_start import ColdStartManager, ColdStartConfig
        self._cold_start = ColdStartManager(ColdStartConfig(mode="prompt"))
        initial_skills = (
            await self._run_deferred_db(lambda: len(self.skill_lib.load_all_active()))
            if self.skill_lib else 0
        )
        self._cold_start.update_stats(skills_count=initial_skills)

        # LearningEffectivenessTracker: metrics observability
        from leapflow.learning.effectiveness import LearningEffectivenessTracker
        self._effectiveness_tracker = LearningEffectivenessTracker()

        # Event-loop yield point before the observer/miner tail phase
        await asyncio.sleep(0)

        # PatternMiner → ActiveLearningObserver bridge (closed loop)
        if settings.observer_auto_start and settings.has_llm_credentials:
            try:
                from leapflow.learning.pattern_miner import PatternMiner
                active_obs = self.active_observer

                def _on_miner_candidates(candidates: list) -> None:
                    if active_obs is not None:
                        active_obs.on_pattern_candidate(candidates)
                    self._effectiveness_tracker.record_pattern_discovered()

                base_freq = 5
                adjusted_freq = self._cold_start.get_adjusted_min_frequency(base_freq)

                self._pattern_miner = PatternMiner(
                    memory=self.imm,
                    llm=self.llm,
                    min_frequency=adjusted_freq,
                    on_candidates=_on_miner_candidates,
                )
                self.event_bus.register_consumer(self._pattern_miner)
                logger.info("PatternMiner registered (min_freq=%d, cold=%s)",
                            adjusted_freq, self._cold_start.phase.value)
            except Exception:
                logger.warning("PatternMiner initialization failed", exc_info=True)

        # ImplicitFeedbackObserver: detect user struggle signals
        if settings.observer_auto_start:
            try:
                from leapflow.perception.implicit_feedback import ImplicitFeedbackObserver
                self._implicit_feedback = ImplicitFeedbackObserver(self.event_bus)
                await self._implicit_feedback.start()
                logger.info("ImplicitFeedbackObserver started")
            except Exception:
                logger.warning("ImplicitFeedbackObserver start failed", exc_info=True)

    _L1_MARKOV_MEMORY_ID = "copilot_l1_markov_state"
    _L1_MARKOV_KIND = "copilot_state"

    def _hydrate_l1_markov(self, l1: Any) -> None:
        """Restore L1 Markov transition matrix from semantic memory."""
        try:
            hits = self.lt.search_keywords(
                [self._L1_MARKOV_MEMORY_ID], kinds=[self._L1_MARKOV_KIND], limit=1,
            )
            if hits:
                import json
                state = json.loads(hits[0].content)
                l1.import_state(state)
        except Exception:
            logger.debug("L1 Markov hydration skipped", exc_info=True)

    def _persist_l1_markov(self) -> None:
        """Save L1 Markov transition matrix to semantic memory for next session."""
        l1 = getattr(self, "_l1_markov", None)
        if l1 is None:
            return
        try:
            import json
            state = l1.export_state()
            if not state.get("transitions"):
                return
            content = json.dumps(state, ensure_ascii=False)
            self.lt.upsert_raw(
                self._L1_MARKOV_KIND, content,
                memory_id=self._L1_MARKOV_MEMORY_ID,
            )
            logger.info("L1 Markov: persisted %d transition keys", len(state.get("transitions", {})))
        except Exception:
            logger.debug("L1 Markov persistence failed", exc_info=True)

    def _build_insight_callback(self) -> Callable:
        """Build callback for replay insights — routes ALL insight types."""
        ps = self.perception_session

        def _on_insight(insight: Any) -> None:
            metadata = getattr(insight, "metadata", None) or {}
            if isinstance(metadata, str):
                return
            insight_type = getattr(insight, "insight_type", None) or metadata.get("type", "unknown")

            # Route 1: Causal rules → causal heuristic (existing)
            causal_rule = metadata.get("causal_rule")
            if causal_rule and isinstance(causal_rule, dict) and ps is not None:
                parent = causal_rule.get("parent_channel", "")
                child = causal_rule.get("child_channel", "")
                confidence = float(causal_rule.get("confidence", 0.5))
                if parent and child:
                    try:
                        heuristic = ps.causal_pipeline.inference.heuristic
                        heuristic.update_prior(parent, child, confidence)
                        logger.debug(
                            "insight applied: %s\u2192%s confidence=%.2f",
                            parent, child, confidence,
                        )
                    except Exception:
                        logger.debug("insight causal application failed", exc_info=True)

            # Route 2: Skill performance and corrective insights → evolution policy
            if insight_type in ("edge_correction", "correction", "heuristic"):
                skill_name = metadata.get("skill_name")
                if skill_name and self._evolution_policy is not None:
                    delta = float(metadata.get("delta", 0.0))
                    if delta < -0.3:  # Significant regression
                        try:
                            outcome = self._evolution_policy.on_regression_detected(
                                skill_name,
                                current_confidence=float(metadata.get("confidence", 0.5)),
                                current_version=int(metadata.get("version", 1)),
                            )
                            if self.skill_lib and outcome.tier_changed:
                                self.skill_lib.update_skill_confidence(
                                    skill_name, outcome.new_confidence
                                )
                                logger.info(
                                    "Insight-driven regression: %s confidence \u2192 %.3f",
                                    skill_name, outcome.new_confidence,
                                )
                        except Exception:
                            logger.debug("insight evolution update failed", exc_info=True)

            if insight_type == "pattern_discovered":
                pattern_desc = metadata.get("pattern", "")
                logger.info("Insight: new pattern discovered — %s", pattern_desc)

        return _on_insight

    async def _generate_session_summary(self) -> str | None:
        """Generate a structured task summary from the session's conversation."""
        store = getattr(self, '_conversation_store', None)
        if not store:
            return None
        session_id = getattr(self.engine, '_current_session_id', None) or ""
        if not session_id:
            return None
        try:
            # Fetch all messages (up to 200) to find both first user goal and final outcome.
            # get_messages returns ASC order; we use a larger limit to capture session endpoints.
            messages = store.get_messages(session_id, limit=200)
        except Exception:
            return None
        if not messages or len(messages) < 2:
            return None

        first_user = ""
        for m in messages:
            if getattr(m, 'role', '') == "user" and getattr(m, 'content', ''):
                first_user = m.content[:150]
                break

        tool_names = sorted(set(
            getattr(m, 'tool_name', '') or ''
            for m in messages
            if getattr(m, 'tool_name', '')
        ))[:8]

        last_assistant = ""
        for m in reversed(messages):
            content = getattr(m, 'content', '') or ''
            if getattr(m, 'role', '') == "assistant" and len(content.strip()) > 20:
                last_assistant = content[:200]
                break

        if not first_user:
            return None

        parts = [f"Goal: {first_user}"]
        if tool_names:
            parts.append(f"Tools: {', '.join(tool_names)}")
        if last_assistant:
            parts.append(f"Outcome: {last_assistant}")
        return "\n".join(parts)

    async def _persist_session_summary(self) -> None:
        """Generate and persist session summary at end of session."""
        try:
            summary = await self._generate_session_summary()
            if not summary:
                return

            # Persist to memory via SemanticMemoryProvider
            from leapflow.memory.protocol import MemoryEntry, MemoryKind, SignalDomain
            session_id = getattr(self.engine, '_current_session_id', None) or ""
            entry = MemoryEntry(
                kind=MemoryKind.SESSION_SUMMARY,
                domain=SignalDomain.SYSTEM,
                content=summary,
                metadata={
                    "_session_id": session_id,
                    "workspace": str(self.settings.workspace_root),
                },
            )
            if hasattr(self, 'memory') and self.memory:
                await self.memory.insert(entry, session_id=session_id)
                # MemoryManager routes SESSION_SUMMARY to narrative (MEMORY.md) first,
                # but query_recent_summaries() reads from DuckDB (semantic provider).
                # Explicitly persist to semantic to enable cross-session querying.
                semantic = self.memory.get_provider("semantic")
                if semantic is not None:
                    try:
                        await semantic.insert(entry, session_id=session_id)
                    except Exception:
                        logger.debug("semantic insert for session summary failed", exc_info=True)
                logger.debug("Session summary persisted for session=%s", session_id[:8])

            # Update session title with the goal line
            goal_line = summary.split("\n")[0].removeprefix("Goal: ").strip()
            conv_store = getattr(self, '_conversation_store', None)
            if conv_store and session_id and goal_line:
                try:
                    conv_store.end_session(session_id, title=goal_line, summary=summary[:500])
                except Exception:
                    logger.debug("session title update failed", exc_info=True)
        except Exception:
            logger.debug("session summary persistence failed", exc_info=True)

    async def _on_session_end_learning(self) -> None:
        """End-of-session OPD learning pipeline (8 phases) with full observability.

        Executes in order:
        1. Trajectory grading (teacher with full hindsight) — grades consumed by replay_engine
        2. Off-policy experience replay (high-delta)
        3. Curiosity-targeted replay (high-curiosity apps) — curiosity fed to attention_tuner
        4. Regression-gated self-distillation (causal rules)
        5. Attention statistics feedback (AttentionTuner)
        6. Long-term memory maintenance (prune old rows)
        7. Budget rebalancing from session outcomes
        8. VLM Tier 3 verification (if enabled)
        """
        observer = self._pipeline_observer
        if observer is None:
            # Deferred init never completed (degraded/critical-only mode):
            # no learning components were assembled, nothing to flush.
            logger.debug("Session-end learning skipped: pipeline observer not initialized")
            return
        pipeline_start = time.perf_counter()
        phases_ok = 0
        phases_failed = 0
        trajectory: list = []

        # Phase 1: Trajectory grading — FIX A5-1: grades now consumed by replay_engine
        if self.trajectory_grader is not None and self.prediction_loop is not None:
            observer.on_phase_start("trajectory_grading")
            t0 = time.perf_counter()
            try:
                trajectory, goal = self.prediction_loop.flush_trajectory()
                if trajectory:
                    grades = await self.trajectory_grader.grade_trajectory(
                        trajectory, goal=goal,
                    )
                    if grades and self.replay_engine is not None:
                        self.replay_engine.set_replay_priorities(grades)
                    observer.on_phase_success(
                        "trajectory_grading", time.perf_counter() - t0,
                        {"actions_graded": len(grades) if grades else 0},
                    )
                    phases_ok += 1
                else:
                    observer.on_phase_success(
                        "trajectory_grading", time.perf_counter() - t0,
                        {"actions_graded": 0, "note": "empty_trajectory"},
                    )
                    phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("trajectory_grading", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 2: Off-policy replay
        if self.replay_engine is not None:
            observer.on_phase_start("off_policy_replay")
            t0 = time.perf_counter()
            try:
                insights = await self.replay_engine.replay_session()
                observer.on_phase_success(
                    "off_policy_replay", time.perf_counter() - t0,
                    {"insights_discovered": len(insights) if insights else 0},
                )
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("off_policy_replay", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 3: Curiosity-targeted replay — FIX A5-2: feed curiosity to attention_tuner
        if self.replay_engine is not None and self.active_observer is not None:
            observer.on_phase_start("curiosity_replay")
            t0 = time.perf_counter()
            try:
                curious_apps = self.active_observer.drain_high_curiosity_apps()
                for app_ctx in curious_apps:
                    await self.replay_engine.replay_targeted(app_ctx)
                tuner = getattr(self, "attention_tuner", None)
                if tuner is not None and curious_apps:
                    tuner.boost_curiosity_domains(curious_apps)
                observer.on_phase_success(
                    "curiosity_replay", time.perf_counter() - t0,
                    {"curious_apps": len(curious_apps)},
                )
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("curiosity_replay", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 4: Regression-gated self-distillation
        if self.replay_engine is not None and trajectory:
            observer.on_phase_start("regression_distillation")
            t0 = time.perf_counter()
            try:
                if self.replay_engine.detect_regression(trajectory):
                    distilled = await self.replay_engine.self_distill()
                    observer.on_phase_success(
                        "regression_distillation", time.perf_counter() - t0,
                        {"regression_detected": True, "rules_distilled": len(distilled)},
                    )
                else:
                    observer.on_phase_success(
                        "regression_distillation", time.perf_counter() - t0,
                        {"regression_detected": False},
                    )
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("regression_distillation", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 5: Attention statistics feedback
        tuner = getattr(self, "attention_tuner", None)
        if tuner is not None and trajectory:
            observer.on_phase_start("attention_feedback")
            t0 = time.perf_counter()
            try:
                from collections import defaultdict
                app_sums: dict = defaultdict(lambda: [0.0, 0])
                for step in trajectory:
                    app = step.get("app_context", "")
                    delta = step.get("delta", 0.0)
                    if app:
                        app_sums[app][0] += delta
                        app_sums[app][1] += 1
                app_deltas = {a: s[0] / s[1] for a, s in app_sums.items() if s[1] > 0}
                if app_deltas:
                    tuner.on_session_stats(app_deltas)
                observer.on_phase_success(
                    "attention_feedback", time.perf_counter() - t0,
                    {"apps_tracked": len(app_deltas)},
                )
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("attention_feedback", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 6: Long-term memory maintenance
        observer.on_phase_start("memory_prune")
        t0 = time.perf_counter()
        try:
            pruned = self.lt.prune(max_age_days=self.settings.memory_prune_age_days)
            observer.on_phase_success(
                "memory_prune", time.perf_counter() - t0,
                {"rows_pruned": pruned or 0},
            )
            phases_ok += 1
        except Exception as exc:
            observer.on_phase_failure("memory_prune", exc, time.perf_counter() - t0)
            phases_failed += 1

        # Phase 6.5: Skill inactivity decay (C4 — after memory prune, before budget rebalance)
        if self._evolution_policy is not None and self.skill_lib is not None:
            observer.on_phase_start("skill_decay")
            t0 = time.perf_counter()
            try:
                all_skills = self.skill_lib.load_all_active_parameterized()
                decayed_count = 0
                for skill in all_skills:
                    last_used = skill.get("updated_at", 0.0)
                    days_inactive = (time.time() - last_used) / 86400.0
                    if days_inactive > 30:  # Only decay after 30 days of inactivity
                        outcome = self._evolution_policy.decay_inactive(
                            skill.get("name", ""),
                            current_confidence=skill.get("confidence", 0.5),
                            current_version=skill.get("version", 1),
                            last_used_ts=last_used,
                        )
                        if outcome.tier_changed:
                            self.skill_lib.update_skill_confidence(
                                skill.get("name", ""), outcome.new_confidence
                            )
                            decayed_count += 1
                observer.on_phase_success(
                    "skill_decay", time.perf_counter() - t0,
                    {"skills_decayed": decayed_count, "skills_checked": len(all_skills)},
                )
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("skill_decay", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 7: Budget rebalancing
        budget = getattr(self, "learning_budget", None)
        if budget is not None:
            observer.on_phase_start("budget_rebalance")
            t0 = time.perf_counter()
            try:
                skills_discovered = 0
                regressions_detected = 0
                avg_delta = 0.0
                if trajectory:
                    deltas = [s.get("delta", 0.0) for s in trajectory if isinstance(s, dict)]
                    avg_delta = sum(deltas) / max(len(deltas), 1)
                    regressions_detected = sum(
                        1 for s in trajectory
                        if isinstance(s, dict) and s.get("verdict") == "regressed"
                    )
                replay_engine = getattr(self, "replay_engine", None)
                if replay_engine is not None:
                    skills_discovered = getattr(replay_engine, "session_discoveries", 0)
                budget.rebalance_from_session_outcome(
                    skills_discovered=skills_discovered,
                    regressions_detected=regressions_detected,
                    avg_prediction_delta=avg_delta,
                )
                observer.on_phase_success(
                    "budget_rebalance", time.perf_counter() - t0,
                    {"avg_delta": round(avg_delta, 4), "regressions": regressions_detected},
                )
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("budget_rebalance", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Phase 8: VLM Tier 3 verification
        if self.settings.causal_tier3_enabled:
            observer.on_phase_start("vlm_tier3")
            t0 = time.perf_counter()
            try:
                ps = self.perception_session
                if ps is not None:
                    pipeline = ps.causal_pipeline
                    graph = ps.causal_graph

                    async def _vlm_call(prompt: str) -> str:
                        vlm = self.vlm or self.llm
                        resp = await vlm.achat(
                            [{"role": "user", "content": prompt}],
                            stream=False,
                            enable_thinking=False,
                        )
                        return (resp.content or "").strip()

                    await pipeline.run_vlm_verification(graph, vlm_call=_vlm_call)
                observer.on_phase_success("vlm_tier3", time.perf_counter() - t0, {})
                phases_ok += 1
            except Exception as exc:
                observer.on_phase_failure("vlm_tier3", exc, time.perf_counter() - t0)
                phases_failed += 1

        # Pipeline complete
        observer.on_pipeline_complete(
            time.perf_counter() - pipeline_start, phases_ok, phases_failed,
        )

    _NORMALIZER_FACTORIES: dict[str, type] = {}

    @staticmethod
    def _get_normalizer_factories() -> dict[str, type]:
        """Lazily load normalizer classes for known platforms."""
        if not Context._NORMALIZER_FACTORIES:
            from leapflow.gateway.normalizers.dingtalk import DingTalkEventNormalizer
            from leapflow.gateway.normalizers.feishu import FeishuEventNormalizer
            from leapflow.gateway.normalizers.telegram import TelegramEventNormalizer

            Context._NORMALIZER_FACTORIES = {
                "feishu": FeishuEventNormalizer,
                "telegram": TelegramEventNormalizer,
                "dingtalk": DingTalkEventNormalizer,
            }
        return Context._NORMALIZER_FACTORIES

    def _register_gateway_normalizers(self, settings: Any) -> None:
        """Register event normalizers and trigger policies for all known platforms."""
        from leapflow.gateway.trigger_policy import TriggerMode, TriggerPolicy

        gw = self.gateway_server
        profile = getattr(settings, "active_profile", "default")
        factories = self._get_normalizer_factories()

        for platform_id, factory in factories.items():
            normalizer = factory(profile=profile)
            gw.register_normalizer(platform_id, normalizer)

            opts = gw.platform_options(platform_id)
            raw_mode = str(opts.get("trigger_mode", "mention_only"))
            try:
                trigger_mode = TriggerMode(raw_mode)
            except ValueError:
                logger.warning(
                    "Unknown trigger_mode '%s' for %s, defaulting to mention_only",
                    raw_mode, platform_id,
                )
                trigger_mode = TriggerMode.MENTION_ONLY

            def _to_frozenset(val: Any) -> frozenset[str]:
                if isinstance(val, str):
                    return frozenset(v.strip() for v in val.split(",") if v.strip())
                if isinstance(val, (list, tuple, set, frozenset)):
                    return frozenset(str(v) for v in val)
                return frozenset()

            policy = TriggerPolicy(
                mode=trigger_mode,
                allowed_chats=_to_frozenset(opts.get("allowed_chats")),
                blocked_chats=_to_frozenset(opts.get("blocked_chats")),
                allowed_users=_to_frozenset(opts.get("allowed_users")),
                blocked_users=_to_frozenset(opts.get("blocked_users")),
                keywords=tuple(
                    str(k) for k in (opts.get("keywords") or [])
                ) if opts.get("keywords") else (),
                max_events_per_minute=int(opts.get("max_events_per_minute", 30)),
                cooldown_per_chat_s=float(opts.get("cooldown_per_chat_s", 1.0)),
            )
            gw.register_trigger_policy(platform_id, policy)

    async def cleanup(self) -> None:
        # Physical devices come first. A sampling loop still reading from a transport
        # that is being torn down logs a failure per channel on the way out, burying
        # whatever actually caused the shutdown; and a device left commanded -- a fan
        # still spinning, a serial port still held -- outlives the process that opened
        # it. Unlike every store below, this one has consequences outside the machine.
        registry = getattr(self, "_hardware_registry", None)
        if registry is not None:
            try:
                await registry.close_all()
            except Exception:
                logger.warning("Hardware teardown failed", exc_info=True)

        # Drain the deferred-DB executor first so no worker thread touches the
        # shared DuckDB connection while stores below persist/close it.
        db_executor = getattr(self, "_deferred_db_executor", None)
        if db_executor is not None:
            db_executor.shutdown(wait=True, cancel_futures=True)
            self._deferred_db_executor = None

        # Persist evolution episodes to DuckDB before shutdown
        evo_store = getattr(self, "_evolution_store", None)
        if evo_store is not None and self._evolution is not None:
            try:
                for eid, episode in self._evolution._episodes.items():
                    evo_store.save_episode(
                        episode_id=eid,
                        skill_name=episode.skill_name,
                        actions=episode.actions,
                        outcome=episode.outcome,
                        reward=episode.reward,
                        context=episode.context,
                        timestamp=episode.timestamp,
                    )
                logger.info("Evolution: persisted %d episodes to DuckDB", len(self._evolution._episodes))
            except Exception:
                logger.debug("Evolution persistence failed", exc_info=True)
            finally:
                try:
                    evo_store.close()
                except Exception:
                    pass

        # Stop gateway server
        gw = getattr(self, "gateway_server", None)
        if gw is not None:
            try:
                await gw.stop()
            except Exception:
                logger.debug("GatewayServer stop failed", exc_info=True)

        # Cancel engine if running
        if self.engine is not None:
            self.engine.cancel()

        # Close MCP manager
        mcp = getattr(self, "_mcp_manager", None)
        if mcp is not None:
            try:
                mcp.close()
            except Exception:
                logger.debug("MCP manager close failed", exc_info=True)

        # Close conversation store
        conv_store = getattr(self, "_conversation_store", None)
        if conv_store is not None:
            try:
                conv_store.close()
            except Exception:
                logger.debug("ConversationStore close failed", exc_info=True)
        # Stop ImplicitFeedbackObserver
        implicit = getattr(self, "_implicit_feedback", None)
        if implicit is not None:
            try:
                await implicit.stop()
            except Exception:
                logger.debug("ImplicitFeedbackObserver stop failed", exc_info=True)
        # Stop ObservationDaemon
        if self._observation_daemon is not None:
            try:
                await self._observation_daemon.stop()
            except Exception:
                logger.warning("ObservationDaemon stop failed", exc_info=True)
        # Emit final effectiveness metrics
        tracker = getattr(self, "_effectiveness_tracker", None)
        if tracker is not None:
            tracker.maybe_emit()
        # Persist L1 Markov state before shutdown
        self._persist_l1_markov()
        # Flush EventBus tail events before learning pipeline
        try:
            await self.event_bus.shutdown()
        except Exception:
            logger.debug("EventBus shutdown failed", exc_info=True)
        # OPD end-of-session learning pipeline
        if self.settings.replay_on_session_end:
            await self._on_session_end_learning()
        # Persist session summary before memory shutdown
        await self._persist_session_summary()
        # Shutdown all memory providers (stops GC, closes DB)
        await self.memory.shutdown_all()
        if isinstance(self.rpc, CuaDriverClient):
            try:
                self.rpc.stop()
            except Exception:
                logger.debug("CuaDriverClient stop failed during cleanup", exc_info=True)
        if self.skill_lib:
            self.skill_lib.close()
        if self.session_store:
            self.session_store.close()
        if self.imitation:
            self.imitation.store.close()
        # Close the shared DuckDB connection last (after all stores are done)
        db_holder = getattr(self, "_db_holder", None)
        if db_holder is not None:
            db_holder.close()
        self.audit.close()
