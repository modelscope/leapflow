"""CLI runtime context — assembles and manages the LeapFlow component graph."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from leapflow.platform.client import BridgeClient
from leapflow.platform.cua_client import CuaDriverClient
from leapflow.platform.event_bus import EventBus
from leapflow.platform.mock import MockBridge
from leapflow.config import Settings, load_config
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
from leapflow.memory import (
    MemoryManager, WorkingMemoryProvider, EpisodicMemoryProvider,
    SemanticMemoryProvider, EvolutionMemoryProvider, MemoryFragment,
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
from leapflow.learning.feedback import FeedbackEvaluator
from leapflow.storage.skill_library import SkillLibraryStore
from leapflow.skills.registry import SkillRegistry
from leapflow.engine.audit import AuditLogger
from leapflow.learning.similarity import HeuristicSimilarityScorer, LLMSimilarityScorer
from leapflow.platform.adapters.darwin import DarwinExecutionAdapter, DarwinPerceptionAdapter
from leapflow.domain.platform import PlatformManifest
from leapflow.engine.shortcuts import ShortcutStore
from leapflow.engine.situational_assessor import LLMSituationalAssessor
from leapflow.platform.facade import VirtualSystemInterface
from leapflow.platform.normalizer import EventNormalizer


logger = logging.getLogger(__name__)


# ─── Host Readiness ────────────────────────────────────────────────────────


class HostReadiness(str, Enum):
    """Result of OS Host pre-flight check during initialization."""

    RUNNING = "running"       # Host confirmed running, proceed normally
    STARTED = "started"       # Host just started by this process
    DEGRADED = "degraded"     # Host unavailable, running in offline mode


def _emit_status(msg: str) -> None:
    """Write a dim status line to stderr (non-blocking, safe pre-logging)."""
    sys.stderr.write(f"\033[2m\u2192 {msg}\033[0m\n")
    sys.stderr.flush()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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

    Returns (VideoRecorder, VideoAnalyzer, VideoSegmenter, SignalTimeline).
    """
    from leapflow.perception.video.analyzer import VideoAnalyzer
    from leapflow.perception.video.cache_manager import VideoCacheManager
    from leapflow.perception.video.recorder import VideoRecorder
    from leapflow.perception.video.segmenter import VideoSegmenter
    from leapflow.perception.video.timeline import SignalTimeline

    # Clean up stale video cache before allocating new recording resources
    cache_manager = VideoCacheManager(
        settings.video_cache_dir,
        max_age_days=settings.video_cache_max_age_days,
        max_size_gb=settings.video_cache_max_size_gb,
    )
    cache_manager.cleanup()

    recorder = VideoRecorder(
        rpc,
        settings.video_cache_dir,
        fps=settings.video_fps,
        resolution_scale=settings.video_resolution_scale,
        codec=settings.video_codec,
        max_segment_s=settings.video_max_segment_s,
    )
    analyzer = VideoAnalyzer(
        vlm,
        l2_enabled=settings.video_l2_enabled,
        l3_enabled=settings.video_l3_enabled,
        max_l2_requests=settings.video_max_l2_requests,
        max_l3_requests=settings.video_max_l3_requests,
        l2_time_window_s=settings.video_l2_time_window_s,
        frame_extractor=recorder,  # VideoRecorder implements FrameExtractor Protocol
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


def sanitize_skill_name(title: str) -> str:
    """Convert a skill title to a registry-safe name."""
    name = re.sub(r"[^\w\s-]", "", title.lower())
    name = re.sub(r"[\s]+", "-", name.strip())
    return name or "unnamed-skill"


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
    from leapflow.storage.skill_library import StoredSkill
    from leapflow.skills.registry import Skill, SkillMetadata

    registered_names = set(registry.names()) if hasattr(registry, 'names') else {s.name for s in registry.list_all()}
    stored = skill_lib.load_all_active()
    count = 0

    for s in stored:
        name = sanitize_skill_name(s.title)
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
        self.effective_mock = bool(mock_host or settings.mock_host)

        # Memory subsystem — provider-based architecture
        working = WorkingMemoryProvider(max_tokens=settings.memory_working_max_tokens)
        semantic = SemanticMemoryProvider(db_path=settings.duckdb_path)
        episodic = EpisodicMemoryProvider(
            ttl=settings.memory_episodic_ttl_s,
            max_entries=settings.memory_episodic_max_entries,
            on_promote=_build_promotion_callback(semantic),
        )
        evolution = EvolutionMemoryProvider(max_episodes=settings.memory_evolution_max_episodes)

        self.memory = MemoryManager()
        self.memory.add_provider(working)
        self.memory.add_provider(episodic)
        self.memory.add_provider(semantic)
        self.memory.add_provider(evolution)

        # Shorthands used by engine/event_bus
        self.wm = working
        self.lt = semantic
        self.imm = episodic
        self._evolution = evolution

        self.event_bus = EventBus(immediate=self.imm, working=self.wm)
        self.rpc: BridgeClient | CuaDriverClient | MockBridge
        if self.effective_mock:
            self.rpc = MockBridge()
            self.rpc.on_event(self.event_bus.handle_event)
        elif settings.use_cua_driver:
            from leapflow.platform.cua_client import cua_driver_available
            if not cua_driver_available():
                _emit_status(
                    "WARNING: use_cua_driver=True but cua-driver not found on PATH"
                )
                _emit_status(
                    "  Install with: leap host install  |  Diagnose with: leap host doctor"
                )
            self.rpc = CuaDriverClient()
        else:
            self.rpc = BridgeClient(
                settings.bridge_socket,
                default_timeout=settings.rpc_timeout_default,
            )
            self.rpc.on_event(self.event_bus.handle_event)

        self.llm = OpenAIChat(
            api_key=settings.llm_api_key or "missing",
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            max_retries=settings.llm_max_retries,
        )

        self.vlm: Optional[OpenAIChat] = None
        if settings.vlm_model and settings.vlm_model != settings.llm_model:
            self.vlm = OpenAIChat(
                api_key=settings.vlm_api_key or settings.llm_api_key or "missing",
                base_url=settings.vlm_base_url or settings.llm_base_url,
                model=settings.vlm_model,
            )

        self.audit = AuditLogger(settings.audit_log_path)

        self.shortcuts = ShortcutStore(Path.cwd() / ".leapflow" / "shortcuts.yaml")
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
        self._bg_connect_task: Optional[asyncio.Task] = None

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

    async def _bg_connect(self) -> None:
        """Background bridge reconnection with post-connect initialization.
    
        After connecting, sends fs.subscribe and fires reconnect callbacks
        so the system transitions from offline/mock state to live.
    
        Uses longer retry window (up to ~90s) with increasing delays to
        handle slow host startup and transient unavailability.
        """
        _BG_MAX_ATTEMPTS = 12
        _BG_DELAYS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 8.0, 10.0, 10.0, 15.0, 15.0]
        for attempt in range(_BG_MAX_ATTEMPTS):
            delay = _BG_DELAYS[attempt] if attempt < len(_BG_DELAYS) else 15.0
            await asyncio.sleep(delay)
            ok = await self.rpc.try_connect()
            if ok:
                _emit_status("Bridge connected")
                await self._post_connect_setup()
                return
        _emit_status("Bridge unavailable, running offline")

    async def _resubscribe_fs(self) -> None:
        """Re-subscribe to FS events after reconnect."""
        home = str(Path.home())
        cwd = str(Path.cwd())
        for watch_path in dict.fromkeys([home, cwd]):
            try:
                await self.rpc.call("fs.subscribe", {"path": watch_path})
                logger.info("Re-subscribed to FS events: %s", watch_path)
            except Exception as exc:
                logger.warning("fs.subscribe failed for %s: %s", watch_path, exc)

    async def _post_connect_setup(self) -> None:
        """Post-connect initialization: register fs callback + fire all reconnect callbacks.

        Registering _resubscribe_fs before firing ensures it runs as part of
        fire_reconnect_callbacks, avoiding a duplicate call.
        """
        if isinstance(self.rpc, BridgeClient):
            self.rpc.on_reconnect(self._resubscribe_fs)
            await self.rpc.fire_reconnect_callbacks()

    # ── Host Readiness (legacy — OS Host removed, cua-driver pending) ────────

    async def _ensure_host_ready(self) -> HostReadiness:
        """OS Host has been removed; always returns DEGRADED (offline mode).

        The legacy OS Host module is deprecated in favor of cua-driver.
        Until cua-driver integration is complete, the system runs in offline mode.
        """
        _emit_status("OS Host: removed (cua-driver migration pending)")
        return HostReadiness.DEGRADED

    async def initialize(self) -> None:
        """Async initialization: VSI handshake, pipeline assembly.

        Phases:
            1. Memory providers initialization
            2. OS Host readiness check (diagnose → auto-start → feedback)
            3. Bridge connection (non-blocking, background fallback)
            4. Platform adapter registration
        """
        settings = self.settings

        # Initialize all memory providers (opens DB, starts GC, etc.)
        await self.memory.initialize_all()

        # Phase 1: Host readiness (OS Host removed; mock or offline)
        host_readiness: HostReadiness
        if self.effective_mock:
            # Mock mode: no host needed, bridge is in-process.
            host_readiness = HostReadiness.RUNNING
        else:
            # OS Host removed — run in offline/degraded mode until cua-driver ready
            host_readiness = await self._ensure_host_ready()

        # Phase 2: Bridge connection
        vsi = VirtualSystemInterface(self.rpc)
        bridge_online = False

        if isinstance(self.rpc, BridgeClient):
            # Skip connection attempt when host is known-degraded
            if host_readiness == HostReadiness.DEGRADED:
                _emit_status("Running in offline mode")
                manifest = PlatformManifest.default_darwin()
                vsi._manifest = manifest
                self._bg_connect_task = asyncio.create_task(self._bg_connect())
            else:
                bridge_online = await self.rpc.try_connect()
                if bridge_online:
                    manifest = await vsi.handshake()
                else:
                    _emit_status("Bridge not available, connecting in background...")
                    manifest = PlatformManifest.default_darwin()
                    vsi._manifest = manifest
                    self._bg_connect_task = asyncio.create_task(self._bg_connect())
        elif isinstance(self.rpc, CuaDriverClient):
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
            if isinstance(self.rpc, BridgeClient):
                self.rpc.on_reconnect(self._resubscribe_fs)
        else:
            from leapflow.platform.adapters.mock import MockExecutionAdapter, MockPerceptionAdapter
            perception = MockPerceptionAdapter()
            execution_adapter = MockExecutionAdapter()

        logger.info(
            "Platform: %s (v%s) | Capabilities: %d",
            manifest.platform_id.value,
            manifest.os_version,
            len(manifest.capabilities),
        )

        codegen = None
        if settings.has_llm_credentials:
            codegen = CompositeSkillCodeGenerator(LLMSkillCodeGenerator(self.llm))

        traj_db_path = settings.duckdb_path.parent / "trajectories.duckdb"
        traj_store = TrajectoryStore(traj_db_path)
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

        # Video-mode components
        video_recorder = None
        video_analyzer = None
        video_segmenter = None
        signal_timeline = None
        if settings.recording_mode.uses_video and settings.visual_track_enabled:
            video_recorder, video_analyzer, video_segmenter, signal_timeline = (
                _build_video_components(settings, self.rpc, self.vlm or self.llm)
            )

        platform_hint = manifest.platform_id.value

        from leapflow.analysis.abstractor import ActionAbstractor
        abstractor = ActionAbstractor(platform_hint=platform_hint)

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
        )
        self.event_bus.subscribe(self.imitation.recorder.on_event)

        # Wire perception session into EventBus with shared attention context
        if perception_session:
            perception_session._recording_context = self.imitation.recorder.attention_context
            perception_session.set_recording_mode(settings.recording_mode)
            self.event_bus.subscribe(perception_session.on_system_event)

        # Wire signal timeline into EventBus for video mode
        if settings.recording_mode.uses_video and signal_timeline is not None:
            if self.event_bus is not None:
                self.event_bus.subscribe(signal_timeline.record_event)
                logger.info("EventBus -> SignalTimeline subscription established")
            else:
                logger.warning(
                    "EventBus is None — skipping SignalTimeline subscription"
                )
        elif settings.recording_mode.uses_video and signal_timeline is None:
            logger.warning(
                "SignalTimeline is None — skipping EventBus subscription "
                "(video mode active but timeline unavailable)"
            )

        skill_lib_path = settings.duckdb_path.parent / "skill_library.duckdb"
        self.skill_lib = SkillLibraryStore(skill_lib_path, audit_logger=self.audit)
        scorer = HeuristicSimilarityScorer()
        llm_scorer = LLMSimilarityScorer(self.llm) if settings.has_llm_credentials else None
        feedback_evaluator = FeedbackEvaluator(
            traj_store, llm=self.llm if settings.has_llm_credentials else None,
        )

        self.registry = build_default_registry(self.rpc, self.llm, self.wm, self.lt)

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

            # Bridge CausalGraph → CuriositySignal (frequency data + graph reference)
            if perception_session is not None:
                self.curiosity.set_causal_graph(perception_session.causal_graph)
                freq = perception_session.causal_graph.metadata.get("frequency_counter")
                if freq:
                    self.curiosity.load_frequency_counter(freq)

            # Wire StateSnapshotService.update_focus() from EventBus
            _ss = self.snapshot_service
            def _on_focus_for_snapshot(event: Any) -> None:
                if getattr(event, "event_type", "") == "app.focus_change":
                    bid = event.payload.get("bundle_id", "")
                    title = event.payload.get("window_title", "")
                    if bid:
                        _ss.update_focus(bid, title)
            self.event_bus.subscribe(_on_focus_for_snapshot)

            logger.info("World model initialized (prediction + curiosity + replay + OPD grading)")

        activator = None
        if perception and execution_adapter:
            activator = SkillActivator(
                self.registry, self.skill_lib, execution_adapter, perception,
                codegen=codegen,
            )
            n_activated = activator.load_and_activate_all()
            if n_activated:
                logger.info("Activated %d learned skills from library", n_activated)

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

            _evo_policy = self._evolution_policy
            _skill_lib = self.skill_lib

            def _on_prediction_outcome(outcome: Any) -> None:
                score = self.curiosity.compute(outcome)
                exp_id = getattr(outcome, "experience_id", "")
                if exp_id and _es is not None:
                    _es.update_curiosity_score(exp_id, score.total)
                _tuner.on_curiosity_signal(score, outcome)
                observer.on_curiosity_signal(score, outcome)

                # Delta-driven skill evolution (断裂点4)
                delta = getattr(outcome, "delta", 0.0)
                if delta > 0.4 and _evo_policy is not None and _skill_lib is not None:
                    action_desc = ""
                    pred = getattr(outcome, "prediction", None)
                    if pred is not None:
                        action_desc = getattr(pred, "action_description", "")
                    if action_desc:
                        try:
                            stored = _skill_lib.load_skill_by_title(action_desc)
                            if stored:
                                evo_outcome = _evo_policy.on_execution_result(
                                    stored.title,
                                    success=(delta < 0.2),
                                    duration_s=0.0,
                                    current_confidence=stored.confidence,
                                    current_version=stored.version,
                                )
                                _skill_lib.update_skill_confidence(
                                    stored.title, evo_outcome.new_confidence
                                )
                                if evo_outcome.tier_changed:
                                    logger.info(
                                        "Delta-driven evolution: '%s' confidence \u2192 %.3f",
                                        stored.title, evo_outcome.new_confidence,
                                    )
                        except Exception:
                            logger.debug("delta-driven evolution update failed", exc_info=True)

            self.prediction_loop._on_outcome = _on_prediction_outcome

        n_doc_skills = 0
        for skill in self.doc_store.load_all_as_skills(
            self.llm, execution=execution_adapter, perception=perception
        ):
            self.registry.register(skill)
            n_doc_skills += 1
        if n_doc_skills:
            logger.info("Registered %d SKILL.md skills", n_doc_skills)

        n_fallback = _register_stored_skill_fallbacks(
            self.skill_lib, self.registry, self.llm,
        )
        if n_fallback:
            logger.info("Registered %d stored skills as fallback", n_fallback)

        graph_planner = GraphPlanner(self.llm, self.registry) if settings.has_llm_credentials else None
        scheduler = TaskScheduler(
            self.registry, self.rpc, graph_planner=graph_planner,
        ) if graph_planner else None

        # Build ToolBridge with general-purpose tools for unified execution
        from leapflow.skills.bridge_factory import build_tool_bridge
        from leapflow.tools import bootstrap_tools

        tool_bridge = build_tool_bridge(execution_adapter, perception)
        tool_count = bootstrap_tools(tool_bridge)
        logger.info("Registered %d general-purpose tools", tool_count)

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

        from leapflow.engine.confirmation import ConfirmationHandler
        confirmation = ConfirmationHandler(skill_store=self.skill_lib)

        self.session_store = LearningSessionStore(traj_db_path)

        # Learnability assessor
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
            storage_path=str(skill_lib_path),
            audit_log_path=str(settings.audit_log_path),
            active_learning_observer=observer,
            session_store=self.session_store,
            learnability_assessor=learnability_assessor,
            evolution_policy=self._evolution_policy,
            skill_store=self.skill_lib,
        )

        classifier: IntentClassifier = (
            LLMIntentClassifier(self.llm) if settings.has_llm_credentials else FallbackClassifier()
        )
        self.intent_classifier = classifier
        if settings.has_llm_credentials:
            self.assessor = LLMSituationalAssessor(self.llm)

        # ── Workflow Copilot (proactive prediction pipeline) ──
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

            # Context encoder
            copilot_encoder = ContextEncoder(copilot_config)

            # Predictors (L0 + L1 always; L2/L3 only if LLM available)
            from leapflow.copilot.predictors.l0_hash import InMemoryContextHashStore

            l0_store = InMemoryContextHashStore()
            # Use SemanticHashAdapter for persistent storage if semantic provider available
            if hasattr(self, 'lt') and self.lt is not None:
                from leapflow.copilot.adapters import SemanticHashAdapter
                l0_store = SemanticHashAdapter(self.lt)

            predictors = [
                L0HashPredictor(l0_store),
                L1MarkovPredictor(),
            ]
            # L2/L3: wire Memory adapters when ExperienceStore is available
            if hasattr(self, 'experience_store') and self.experience_store is not None:
                from leapflow.copilot.adapters import ExperienceEmbedAdapter, MemoryRAGAdapter
                from leapflow.copilot.predictors.l2_embed import L2EmbeddingPredictor
                from leapflow.copilot.predictors.l3_llm import L3LLMPredictor

                l2_provider = ExperienceEmbedAdapter(self.experience_store)
                predictors.append(L2EmbeddingPredictor(l2_provider))

                if settings.has_llm_credentials:
                    from leapflow.copilot.adapters import MemoryRAGAdapter as _RAG
                    rag_provider = _RAG(self.wm, self.experience_store)

                    class _CopilotLLMClient:
                        """Adapt OpenAIChat to L3's LLMClient protocol."""
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

            # Prediction engine
            copilot_engine = PredictionEngine(predictors, copilot_config)

            # Speculative pipeline
            copilot_pipeline = SpeculativePipeline(copilot_engine, copilot_config)

            # Feedback
            copilot_feedback = FeedbackCollector()
            copilot_evolution = EvolutionLoop(copilot_config, predictors)

            # Idle detector (callback will be wired in Phase 2)
            async def _copilot_on_idle(duration_ms: int) -> None:
                pass  # Phase 2 will implement ghost hint rendering here

            copilot_idle = IdleDetector(copilot_config, on_idle=_copilot_on_idle)

            # Event subscriber — register to EventBus
            copilot_subscriber = CopilotEventSubscriber(
                copilot_encoder,
                tracker=None,  # CrossAppContextTracker if available
                working_memory=self.wm if hasattr(self, 'wm') else None,
            )
            self.event_bus.subscribe(copilot_subscriber.on_system_event)

            # Store references on Context for Phase 2 access
            self.copilot_pipeline = copilot_pipeline
            self.copilot_idle = copilot_idle
            self.copilot_encoder = copilot_encoder
            self.copilot_feedback = copilot_feedback
            self.copilot_evolution = copilot_evolution
            self.copilot_config = copilot_config

            logger.info("Copilot initialized: L0+L1 predictors active, idle detection armed")
        else:
            self.copilot_pipeline = None
            self.copilot_idle = None
            self.copilot_encoder = None
            self.copilot_feedback = None
            self.copilot_evolution = None
            self.copilot_config = None

        self.engine = AgentEngine(
            settings, self.rpc, self.llm, self.wm, self.lt, self.imm,
            self.registry, classifier,
            imitation=self.imitation,
            skill_library=self.skill_lib,
            graph_planner=graph_planner,
            scheduler=scheduler,
            perception=perception,
            execution=execution_adapter,
            skill_activator=activator,
            session=self.session,
            shortcuts=self.shortcuts,
            vlm=self.vlm,
            memory_manager=self.memory,
            evolution=self._evolution,
            tool_bridge=tool_bridge,
            skill_injector=skill_injector,
            skill_index=skill_index,
        )

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

        # PatternMiner integration — 断裂点5: bridge EventBus → PatternMiner → ActiveLearning
        if settings.observer_auto_start and settings.has_llm_credentials:
            try:
                from leapflow.learning.pattern_miner import PatternMiner
                self._pattern_miner = PatternMiner(
                    memory=self.imm,
                    llm=self.llm,
                    min_frequency=5,
                )
                self.event_bus.register_consumer(self._pattern_miner)
                logger.info("PatternMiner registered as EventBus consumer")
            except ImportError:
                logger.debug("PatternMiner not available (module not yet created)")
            except Exception:
                logger.warning("PatternMiner initialization failed", exc_info=True)

    def _build_insight_callback(self) -> Callable:
        """Build callback for replay insights — routes ALL insight types."""
        ps = self.perception_session

        def _on_insight(insight: Any) -> None:
            metadata = getattr(insight, "metadata", None) or {}
            if isinstance(metadata, str):
                return
            insight_type = metadata.get("type", "unknown")

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

            # Route 2: Skill performance insights → evolution policy
            if insight_type in ("skill_regression", "skill_improvement", "execution_delta"):
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

            # Route 3: Pattern discovery insights → log for observability
            if insight_type == "pattern_discovered":
                pattern_desc = metadata.get("pattern", "")
                logger.info("Insight: new pattern discovered \u2014 %s", pattern_desc)
                # Route to ActiveLearningObserver if available
                if self.active_observer is not None and hasattr(self.active_observer, "on_pattern_candidate"):
                    try:
                        self.active_observer.on_pattern_candidate(metadata)
                    except Exception:
                        logger.debug("pattern candidate routing failed", exc_info=True)

        return _on_insight

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
                    # FIX A5-1: Feed grades to replay engine as priority weights
                    if hasattr(self.replay_engine, 'set_replay_priorities') and grades:
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
                # FIX A5-2: Feed curiosity signal to attention_tuner for next session
                tuner = getattr(self, "attention_tuner", None)
                if tuner is not None and curious_apps and hasattr(tuner, 'boost_curiosity_domains'):
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

    async def cleanup(self) -> None:
        if self._bg_connect_task and not self._bg_connect_task.done():
            self._bg_connect_task.cancel()
            try:
                await self._bg_connect_task
            except asyncio.CancelledError:
                pass
        # Stop ObservationDaemon if running
        if self._observation_daemon is not None:
            try:
                await self._observation_daemon.stop()
            except Exception:
                logger.warning("ObservationDaemon stop failed", exc_info=True)
        # OPD end-of-session learning pipeline
        if self.settings.replay_on_session_end:
            await self._on_session_end_learning()
        # Shutdown all memory providers (stops GC, closes DB)
        await self.memory.shutdown_all()
        if isinstance(self.rpc, BridgeClient):
            await self.rpc.close()
        elif isinstance(self.rpc, CuaDriverClient):
            self.rpc.stop()
        if self.skill_lib:
            self.skill_lib.close()
        if self.imitation:
            self.imitation.store.close()
        self.audit.close()
        if self.skill_lib:
            self.skill_lib.close()
        if self.imitation:
            self.imitation.store.close()
        self.audit.close()
