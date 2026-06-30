"""CLI runtime context — assembles and manages the LeapFlow component graph."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from leapflow.platform.client import BridgeClient
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
        self.rpc: BridgeClient | MockBridge
        if self.effective_mock:
            self.rpc = MockBridge()
            self.rpc.on_event(self.event_bus.handle_event)
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

    # ── Host Readiness ──────────────────────────────────────────────────────

    def _build_host_manager(self) -> "HostManager":  # noqa: F821
        """Construct a HostManager from current settings."""
        from leapflow.host import HostManager
        return HostManager(
            host_root=self.settings.host_root,
            host_socket=self.settings.host_socket,
            pid_file=self.settings.host_pid_file,
            log_file=self.settings.host_log_file,
            bundle_id=self.settings.host_bundle_id,
        )

    async def _ensure_host_ready(self) -> HostReadiness:
        """Structured OS Host health check with transparent status feedback.

        Performs fast-path diagnosis, attempts auto-start when needed, and
        reports state to the user in real-time. Never blocks longer than the
        configured start timeout (default 5s).

        Returns:
            HostReadiness indicating whether the host is usable or degraded.
        """
        from leapflow.host import HostState

        mgr = self._build_host_manager()
        diag = mgr.diagnose()

        # Fast path: already running
        if diag.state == HostState.RUNNING:
            pid_info = f" (PID {diag.pid})" if diag.pid else ""
            _emit_status(f"OS Host: running{pid_info}")
            return HostReadiness.RUNNING

        # Binary not installed — guide user, degrade gracefully
        if not diag.executable_found:
            _emit_status("OS Host: not installed")
            _emit_status("  Run 'leap host setup' to install and configure.")
            return HostReadiness.DEGRADED

        # Stale state detected — announce cleanup
        if diag.state == HostState.STALE:
            _emit_status("OS Host: cleaning stale state...")

        # Attempt start
        _emit_status("OS Host: starting...")
        try:
            status = await mgr.start(timeout=5.0)
            pid_info = f" (PID {status.pid})" if status.pid else ""
            _emit_status(f"OS Host: started{pid_info}")
            return HostReadiness.STARTED
        except FileNotFoundError:
            _emit_status("OS Host: binary not found")
            _emit_status("  Run 'leap host setup' to install and configure.")
            return HostReadiness.DEGRADED
        except TimeoutError:
            _emit_status("OS Host: start timed out")
            _emit_status("  Check 'leap host logs' for details.")
            return HostReadiness.DEGRADED
        except RuntimeError as exc:
            _emit_status(f"OS Host: crashed on startup")
            _emit_status(f"  {exc}")
            return HostReadiness.DEGRADED
        except OSError as exc:
            _emit_status(f"OS Host: start failed ({exc})")
            return HostReadiness.DEGRADED
        except Exception as exc:
            logger.debug("OS Host start unexpected error: %s", exc, exc_info=True)
            _emit_status(f"OS Host: error ({type(exc).__name__})")
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

        # Phase 1: OS Host readiness — diagnose, auto-start, report status
        host_readiness: HostReadiness
        if self.effective_mock:
            # Mock mode: no host needed, bridge is in-process.
            host_readiness = HostReadiness.RUNNING
        elif settings.host_auto_start:
            host_readiness = await self._ensure_host_ready()
        else:
            # Auto-start disabled: user manages host externally (e.g. launchd).
            # Assume available; bridge connection will validate.
            host_readiness = HostReadiness.RUNNING

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

            def _on_prediction_outcome(outcome: Any) -> None:
                score = self.curiosity.compute(outcome)
                exp_id = getattr(outcome, "experience_id", "")
                if exp_id and _es is not None:
                    _es.update_curiosity_score(exp_id, score.total)
                _tuner.on_curiosity_signal(score, outcome)
                observer.on_curiosity_signal(score, outcome)
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
        )

        classifier: IntentClassifier = (
            LLMIntentClassifier(self.llm) if settings.has_llm_credentials else FallbackClassifier()
        )
        self.intent_classifier = classifier
        if settings.has_llm_credentials:
            self.assessor = LLMSituationalAssessor(self.llm)
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

    def _build_insight_callback(self) -> Any:
        """Build a callback that routes replay insights to the causal heuristic engine."""
        ps = self.perception_session
        if ps is None:
            return None

        def _on_insight(insight: Any) -> None:
            causal_rule = getattr(insight, "metadata", {}).get("causal_rule")
            if not causal_rule or not isinstance(causal_rule, dict):
                return
            parent = causal_rule.get("parent_channel", "")
            child = causal_rule.get("child_channel", "")
            confidence = float(causal_rule.get("confidence", 0.5))
            if parent and child:
                try:
                    heuristic = ps.causal_pipeline.inference.heuristic
                    heuristic.update_prior(parent, child, confidence)
                    logger.debug(
                        "insight applied: %s→%s confidence=%.2f",
                        parent, child, confidence,
                    )
                except Exception:
                    logger.debug("insight application failed", exc_info=True)

        return _on_insight

    async def _on_session_end_learning(self) -> None:
        """End-of-session OPD learning pipeline (8 phases).

        Executes in order:
        1. Trajectory grading (teacher with full hindsight)
        2. Off-policy experience replay (high-delta)
        3. Curiosity-targeted replay (high-curiosity apps)
        4. Regression-gated self-distillation (causal rules)
        5. Attention statistics feedback (AttentionTuner)
        6. Long-term memory maintenance (prune old rows)
        7. Budget rebalancing from session outcomes
        8. VLM Tier 3 verification (if enabled)
        """
        trajectory: list = []

        # Phase 1: Trajectory grading
        if self.trajectory_grader is not None and self.prediction_loop is not None:
            try:
                trajectory, goal = self.prediction_loop.flush_trajectory()
                if trajectory:
                    grades = await self.trajectory_grader.grade_trajectory(
                        trajectory, goal=goal,
                    )
                    logger.info("session_end: graded %d actions", len(grades))
            except Exception:
                logger.debug("session_end.trajectory_grading failed", exc_info=True)

        # Phase 2: Off-policy replay
        if self.replay_engine is not None:
            try:
                insights = await self.replay_engine.replay_session()
                if insights:
                    logger.info("session_end: discovered %d insights", len(insights))
            except Exception:
                logger.debug("session_end.replay failed", exc_info=True)

        # Phase 3: Targeted replay for high-curiosity domains
        if self.replay_engine is not None and self.active_observer is not None:
            try:
                curious_apps = self.active_observer.drain_high_curiosity_apps()
                for app_ctx in curious_apps:
                    await self.replay_engine.replay_targeted(app_ctx)
            except Exception:
                logger.debug("session_end.curiosity_replay failed", exc_info=True)

        # Phase 4: Regression-gated self-distillation
        if self.replay_engine is not None and trajectory:
            try:
                if self.replay_engine.detect_regression(trajectory):
                    logger.warning("session_end: regression detected, triggering self-distillation")
                    distilled = await self.replay_engine.self_distill()
                    logger.info("session_end: self-distilled %d rules", len(distilled))
            except Exception:
                logger.debug("session_end.distillation failed", exc_info=True)

        # Phase 5: Feed per-app accuracy stats to AttentionTuner (contract mastered domains)
        tuner = getattr(self, "attention_tuner", None)
        if tuner is not None and trajectory:
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
                    logger.debug("session_end: attention_tuner stats for %d apps", len(app_deltas))
            except Exception:
                logger.debug("session_end.attention_stats failed", exc_info=True)

        # Phase 6: Long-term memory maintenance (prune old, low-value rows)
        try:
            pruned = self.lt.prune(
                max_age_days=self.settings.memory_prune_age_days,
            )
            if pruned:
                logger.info("session_end: pruned %d old memory rows", pruned)
        except Exception:
            logger.debug("session_end.memory_prune failed", exc_info=True)

        # Phase 7: Budget rebalancing from session outcomes
        budget = getattr(self, "learning_budget", None)
        if budget is not None:
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
            except Exception:
                logger.debug("session_end.budget_rebalance failed", exc_info=True)

        # Phase 8: VLM Tier 3 verification (if enabled)
        if self.settings.causal_tier3_enabled:
            try:
                ps = getattr(self, "_perception_session", None)
                if ps is not None:
                    pipeline = ps.causal_pipeline
                    graph = ps.causal_graph

                    async def _vlm_call(prompt: str) -> str:
                        vlm = getattr(self, "vlm", None) or self.llm
                        return await vlm.chat([{"role": "user", "content": prompt}])

                    await pipeline.run_vlm_verification(
                        graph, vlm_call=_vlm_call,
                    )
            except Exception:
                logger.debug("session_end.vlm_tier3 failed", exc_info=True)

    async def cleanup(self) -> None:
        if self._bg_connect_task and not self._bg_connect_task.done():
            self._bg_connect_task.cancel()
            try:
                await self._bg_connect_task
            except asyncio.CancelledError:
                pass
        # OPD end-of-session learning pipeline
        if self.settings.replay_on_session_end:
            await self._on_session_end_learning()
        # Shutdown all memory providers (stops GC, closes DB)
        await self.memory.shutdown_all()
        if isinstance(self.rpc, BridgeClient):
            await self.rpc.close()
        if self.skill_lib:
            self.skill_lib.close()
        if self.imitation:
            self.imitation.store.close()
        self.audit.close()
