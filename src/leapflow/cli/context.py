"""CLI runtime context — assembles and manages the LeapFlow component graph."""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from dotenv import dotenv_values

from leapflow.platform.cua_client import CuaDriverClient
from leapflow.platform.event_bus import EventBus
from leapflow.platform.mock import MockBridge
from leapflow.config import Settings, _load_yaml_overlay
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
from leapflow.engine.shortcuts import ShortcutStore
from leapflow.engine.situational_assessor import LLMSituationalAssessor
from leapflow.platform.facade import VirtualSystemInterface
from leapflow.platform.normalizer import EventNormalizer


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from leapflow.platform.observers import RecordingProfile
    from leapflow.security.approval import ApprovalDecision, ApprovalRequest
    from leapflow.storage.skill_library import StoredSkill


class _TUIApprovalGate:
    """Rich-styled approval gate for the interactive TUI.

    Displays a styled panel with the action details and accepts:
    - ``y``/``yes`` → allow this one time
    - ``a``/``always`` → allow and skip future prompts for this category
    - ``n``/``no``/Enter → deny

    Implements both ``ApprovalGate`` (unified) and ``CommandApprovalGate``
    (backward-compatible with ``shell_tools.py``).
    """

    _CATEGORY_LABELS = {
        "shell_dangerous": ("Shell Command", "yellow"),
        "file_write": ("File Write", "yellow"),
        "gateway_send": ("External Message", "cyan"),
    }

    async def request_approval(
        self, request: "ApprovalRequest",
    ) -> "ApprovalDecision":
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
    """Build a RecordingProfile if the recording mode supports video."""
    if not settings.recording_mode.uses_video:
        return None
    from leapflow.platform.observers import RecordingProfile
    return RecordingProfile()


def _emit_status(msg: str) -> None:
    """Write a dim status line to stderr (non-blocking, safe pre-logging)."""
    if sys.stderr.isatty():
        sys.stderr.write(f"\033[2m\u2192 {msg}\033[0m\n")
    else:
        sys.stderr.write(f"→ {msg}\n")
    sys.stderr.flush()


def configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Install RedactingFormatter to prevent secret leakage in logs
    try:
        from leapflow.security.redact import RedactingFormatter
        formatter = RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    except ImportError:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=log_level, handlers=[handler])


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
    from leapflow.perception.video.analyzer import VideoAnalyzer
    from leapflow.perception.video.cache_manager import VideoCacheManager
    from leapflow.perception.video.recorder import TrajectoryRecorder
    from leapflow.perception.video.segmenter import VideoSegmenter
    from leapflow.perception.video.timeline import SignalTimeline

    cache_manager = VideoCacheManager(
        settings.video_cache_dir,
        max_age_days=settings.video_cache_max_age_days,
        max_size_gb=settings.video_cache_max_size_gb,
    )
    cache_manager.cleanup()

    recorder = TrajectoryRecorder(
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
            memory_dir=settings.profile_dir / "memory",
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
        if self.effective_mock:
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

        # Unified approval gate is resource-free; create it in __init__ so all
        # initialize() wiring paths can safely reference the same session gate.
        from leapflow.security.approval import SessionAwareGate
        from leapflow.security.grants import ApprovalAuditLog, JsonApprovalGrantStore
        from leapflow.security.orchestrator import ApprovalOrchestrator

        approval_dir = settings.profile_dir / "approval"
        self._tui_approval = _TUIApprovalGate()
        self._approval_gate = SessionAwareGate(self._tui_approval)
        self._approval_orchestrator = ApprovalOrchestrator(
            self._approval_gate,
            grants=JsonApprovalGrantStore(approval_dir / "grants.json"),
            audit=ApprovalAuditLog(approval_dir / "audit.jsonl"),
        )

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
        dynamic_result_budget = min(settings.max_tool_result_chars, context_length // 20)
        if dynamic_result_budget != settings.max_tool_result_chars:
            logger.info(
                "Dynamic tool result budget: %d (context=%d)",
                dynamic_result_budget,
                context_length,
            )
        self.engine.set_tool_result_budget(dynamic_result_budget)
        self.engine.set_model_capabilities(self._build_model_capability_registry(settings))
        logger.debug("Model capability registry wired")

    @staticmethod
    def _runtime_config_signature(settings: Settings) -> tuple:
        """Return a stable signature for user-editable runtime config files."""
        paths = (
            settings.data_dir / ".env",
            settings.data_dir / "config.yaml",
            Path.cwd() / ".env",
        )
        signature = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                signature.append((str(path), 0, 0))
            else:
                signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    @staticmethod
    def _bool_env(value: str, default: bool) -> bool:
        text = value.strip().lower()
        if not text:
            return default
        return text in {"1", "true", "yes", "on"}

    def _load_runtime_settings_from_files(self) -> Settings:
        """Reload hot-swappable LLM/VLM settings directly from config files."""
        values: dict[str, str] = {}
        data_env = self.settings.data_dir / ".env"
        cwd_env = Path.cwd() / ".env"
        if data_env.exists():
            values.update({
                key: value
                for key, value in dotenv_values(data_env).items()
                if value is not None
            })
        values.update(_load_yaml_overlay(self.settings.data_dir))
        if cwd_env.exists():
            values.update({
                key: value
                for key, value in dotenv_values(cwd_env).items()
                if value is not None
            })

        def _value(key: str, current: str) -> str:
            env_value = os.environ.get(key, "").strip()
            if env_value:
                return env_value
            return str(values.get(key, current)).strip()

        max_retries_raw = _value(
            "LEAPFLOW_LLM_MAX_RETRIES",
            str(self.settings.llm_max_retries),
        )
        try:
            max_retries = max(1, int(max_retries_raw))
        except ValueError:
            max_retries = self.settings.llm_max_retries

        context_length_raw = _value(
            "LEAPFLOW_LLM_CONTEXT_LENGTH",
            str(self.settings.llm_context_length),
        )
        try:
            llm_context_length = max(1, int(context_length_raw))
        except ValueError:
            llm_context_length = self.settings.llm_context_length

        return replace(
            self.settings,
            llm_api_key=_value("LEAPFLOW_LLM_API_KEY", self.settings.llm_api_key),
            llm_base_url=_value("LEAPFLOW_LLM_BASE_URL", self.settings.llm_base_url).rstrip("/"),
            llm_model=_value("LEAPFLOW_LLM_MODEL", self.settings.llm_model),
            llm_max_retries=max_retries,
            llm_context_length=llm_context_length,
            vlm_api_key=_value("LEAPFLOW_VLM_API_KEY", self.settings.vlm_api_key),
            vlm_base_url=_value("LEAPFLOW_VLM_BASE_URL", self.settings.vlm_base_url).rstrip("/"),
            vlm_model=_value("LEAPFLOW_VLM_MODEL", self.settings.vlm_model),
            visual_track_enabled=self._bool_env(
                _value(
                    "LEAPFLOW_VISUAL_TRACK_ENABLED",
                    "1" if self.settings.visual_track_enabled else "0",
                ),
                self.settings.visual_track_enabled,
            ),
        )

    def reload_runtime_config_if_changed(self) -> bool:
        """Hot-reload LLM/VLM config when user-editable config files changed."""
        signature = self._runtime_config_signature(self.settings)
        if signature == self._config_signature:
            return False

        previous = self.settings
        updated = self._load_runtime_settings_from_files()
        self._config_signature = signature
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
        if not llm_changed:
            return False

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
        logger.info("Runtime LLM configuration reloaded")
        return True

    @property
    def storage_volatile(self) -> bool:
        """Return True when this process uses non-persistent fallback storage."""
        return bool(getattr(self._db_holder, "is_volatile", False))

    async def initialize(self) -> None:
        """Async initialization: VSI handshake, pipeline assembly.

        Phases:
            1. Memory providers initialization
            2. CuaDriver connection / mock setup
            3. Platform adapter registration
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

        # Video-mode components
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
            observation_daemon=self._observation_daemon,
            recording_profile=_default_recording_profile(settings),
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

        self.skill_lib = SkillLibraryStore(self._db_holder, audit_logger=self.audit)
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

            _ctx = self

            def _on_prediction_outcome(outcome: Any) -> None:
                score = _ctx.curiosity.compute(outcome)
                exp_id = getattr(outcome, "experience_id", "")
                if exp_id and _es is not None:
                    _es.update_curiosity_score(exp_id, score.total)
                _tuner.on_curiosity_signal(score, outcome)
                observer.on_curiosity_signal(score, outcome)

                # Delta-driven skill evolution: high delta means prediction
                # was inaccurate (failure); low delta means accurate (success)
                delta = getattr(outcome, "delta", 0.0)
                evo_policy = _ctx._evolution_policy
                skill_lib = _ctx.skill_lib
                if evo_policy is not None and skill_lib is not None:
                    action_desc = ""
                    pred = getattr(outcome, "prediction", None)
                    if pred is not None:
                        action_desc = getattr(pred, "action_description", "")
                    # Strip "skill:" / "bridge:" prefix for title lookup
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

        self.session_store = LearningSessionStore(self._db_holder)

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
            storage_path=str(settings.duckdb_path),
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

            l1_markov = L1MarkovPredictor()
            self._l1_markov = l1_markov
            self._hydrate_l1_markov(l1_markov)

            predictors = [
                L0HashPredictor(l0_store),
                l1_markov,
            ]
            # L2/L3: wire Memory adapters when ExperienceStore is available
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

        # ── Wire memory tools into TOOL_HANDLERS (late binding) ──
        from leapflow.tools.registry_bootstrap import set_memory_manager
        set_memory_manager(self.memory)

        # ── Gateway server (late-bound tool wiring) ──
        from leapflow.gateway.server import GatewayServer
        from leapflow.gateway.router import GatewayRouter
        from leapflow.gateway.events import (
            GatewayMessageReceived,
            GatewaySessionCreated,
            GatewaySessionEnded,
        )
        from leapflow.tools.registry_bootstrap import set_gateway_server
        from leapflow.tools.gateway_tool import set_gateway_approval_gate

        async def _on_gateway_event(event: object) -> None:
            """Bridge gateway events to episodic memory and logging."""
            if isinstance(event, GatewayMessageReceived):
                logger.info(
                    "gateway.inbound platform=%s session=%s len=%d",
                    event.source.platform,
                    event.session_key,
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

        self.gateway_server = GatewayServer(
            settings.profile_dir,
            extra_manifest_dirs=[settings.profile_dir / "gateway" / "manifests"],
            on_event=_on_gateway_event,
        )
        self.gateway_server.discover_manifests()
        set_gateway_server(self.gateway_server)
        set_gateway_approval_gate(self._approval_orchestrator)

        async def _gateway_send(source: Any, text: str) -> None:
            await self.gateway_server.send_reply(source, text)

        from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS

        self._gateway_router = GatewayRouter(
            llm=self.llm,
            system_prompt=(
                "You are LeapFlow, a helpful AI assistant responding "
                "through an external messaging platform.  Be concise "
                "and conversational."
            ),
            send_fn=_gateway_send,
            tool_definitions=TOOL_DEFINITIONS,
            tool_handlers=TOOL_HANDLERS,
        )
        self.gateway_server.set_message_handler(
            self._gateway_router.handle_message,
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

        compressor_config = CompressorConfig(
            threshold=settings.compress_threshold,
            keep_tail=settings.compress_keep_tail,
            max_output_chars=settings.max_tool_output_chars,
            summarize_fn=_summarize_via_llm if settings.has_llm_credentials else None,
            token_count_fn=lambda text: len(text) // 4,
        )

        # ── Initialize DuckDBConversationStore ──
        self._conversation_store = None
        try:
            from leapflow.storage.conversation_store import DuckDBConversationStore
            self._conversation_store = DuckDBConversationStore(self._db_holder)
            logger.info("ConversationStore initialized")
        except Exception:
            logger.warning("ConversationStore initialization failed", exc_info=True)

        # ── Initialize MCP Manager + register tools into agent surface ──
        self._mcp_manager = None
        try:
            mcp_config_path = settings.data_dir / "mcp_servers.json"
            if mcp_config_path.exists():
                import json as _json_mcp
                from leapflow.platform.mcp_manager import McpManager, McpServerConfig
                from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS

                raw_configs = _json_mcp.loads(mcp_config_path.read_text())
                server_configs = [
                    McpServerConfig(
                        name=name,
                        command=cfg.get("command", ""),
                        args=cfg.get("args", []),
                        env=cfg.get("env", {}),
                        parallel_safe=cfg.get("parallel_safe", False),
                    )
                    for name, cfg in raw_configs.items()
                    if cfg.get("enabled", True) and cfg.get("command")
                ]
                if server_configs:
                    mgr = McpManager()
                    total_tools = 0
                    for sc in server_configs:
                        try:
                            schemas = mgr.add_server(sc)
                            total_tools += len(schemas)
                        except Exception:
                            logger.warning("MCP server '%s' failed to connect", sc.name)

                    if total_tools > 0:
                        self._mcp_manager = mgr

                        from leapflow.security.threat_patterns import scan_mcp_description

                        def _build_mcp_handler(manager, tool_name: str):
                            async def _handler(params: dict) -> dict:
                                return await manager.call_tool(tool_name, params)
                            return _handler

                        for schema in mgr.get_tool_schemas():
                            threats = scan_mcp_description(schema.description)
                            if threats:
                                logger.warning("MCP tool '%s' description has threats: %s",
                                               schema.name, [t.pattern_name for t in threats])
                            TOOL_DEFINITIONS.append(schema.to_openai_function())
                            TOOL_HANDLERS[schema.name] = _build_mcp_handler(mgr, schema.name)

                        logger.info("MCP Manager: %d servers, %d tools registered to agent",
                                    len(server_configs), total_tools)
        except Exception:
            logger.debug("MCP Manager initialization skipped", exc_info=True)

        # ── Unified approval gate wiring (shell, file, gateway) ──
        try:
            from leapflow.tools.shell_tools import set_approval_gate
            set_approval_gate(self._approval_orchestrator)
            logger.debug("Shell approval gate: action orchestrator mode")
        except Exception:
            logger.debug("Shell approval gate setup skipped", exc_info=True)

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

        # ── Wire SubagentManager + delegate_task tool ──
        try:
            from leapflow.engine.subagent import DefaultSubagentExecutor, SubagentManager
            from leapflow.tools.registry_bootstrap import (
                TOOL_DEFINITIONS as _TD, TOOL_HANDLERS as _TH,
                set_subagent_manager,
            )
            sub_executor = DefaultSubagentExecutor(
                llm=self.llm,
                tool_handlers=_TH,
                tool_definitions=_TD,
                settings=settings,
            )
            self._subagent_manager = SubagentManager(executor=sub_executor)
            set_subagent_manager(self._subagent_manager)
            logger.info("SubagentManager wired with delegate_task tool")
        except Exception:
            self._subagent_manager = None
            logger.debug("SubagentManager setup skipped", exc_info=True)

        # ── Wire EvolutionStore (DuckDB persistence for skill episodes) ──
        self._evolution_store = None
        try:
            from leapflow.storage.evolution_store import DuckDBEvolutionStore
            self._evolution_store = DuckDBEvolutionStore(self._db_holder)
            # Hydrate in-memory provider from persisted episodes
            persisted = self._evolution_store.load_recent_episodes(
                limit=settings.memory_evolution_max_episodes,
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

        # ── Wire tool loop guardrails ──
        try:
            from leapflow.engine.tool_guardrails import CompositeGuardrail
            self.engine._guardrail = CompositeGuardrail()
            logger.debug("Tool loop guardrails enabled")
        except Exception:
            logger.debug("Tool guardrails setup skipped", exc_info=True)

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

        # ── Wire File Write Approval Gate ──
        try:
            from leapflow.security.actions import ActionDescriptor
            from leapflow.tools.registry_bootstrap import set_file_write_gate

            approval_orchestrator = self._approval_orchestrator

            class _FileWriteGate:
                """File write approval via the action approval orchestrator."""

                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(self, path: str, content: str, mode: str = "overwrite") -> bool:
                    action = ActionDescriptor.file_write(path, content, mode=mode)
                    result = await approval_orchestrator.evaluate(action)
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            set_file_write_gate(_FileWriteGate())
            logger.debug("File write approval gate: action orchestrator")
        except Exception:
            logger.debug("File write gate setup skipped", exc_info=True)

        # ── Token budget and model capability metadata ─────────────────────
        if self.engine is not None:
            self._sync_engine_runtime_budget(settings)
            self.engine.set_stale_stream_timeout(settings.stale_stream_timeout_s)
            self.engine.set_default_tool_timeout(settings.default_tool_timeout_s)

            if self._evolution_store is not None:
                self.engine.set_evolution_store(self._evolution_store)

            if hasattr(self, "doc_store") and self.doc_store is not None:
                self.engine.set_doc_store(self.doc_store)

            self.engine.set_event_bus(self.event_bus)

            if hasattr(self, "experience_store") and self.experience_store is not None:
                self.engine.set_experience_store(self.experience_store)

        # ── Register session_search tool ──
        if self._conversation_store:
            try:
                from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS
                conv_store = self._conversation_store

                TOOL_DEFINITIONS.append({
                    "type": "function",
                    "function": {
                        "name": "session_search",
                        "description": "Search past conversation sessions for relevant context.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search keywords"},
                                "limit": {"type": "integer", "description": "Max results (default: 5)"},
                            },
                            "required": ["query"],
                        },
                    },
                })

                async def _session_search_handler(params: dict) -> dict:
                    query = params.get("query", "")
                    limit = int(params.get("limit", 5))
                    if not query:
                        return {"ok": False, "error": "Missing query parameter"}
                    results = conv_store.search_messages(query, limit=limit)
                    if not results:
                        return {"ok": True, "result": "No matching sessions found."}
                    items = [
                        {
                            "session": r.session_title or r.session_id[:8],
                            "role": r.role,
                            "content": r.content[:300],
                            "score": round(r.score, 3),
                        }
                        for r in results
                    ]
                    import json as _json_ss
                    return {"ok": True, "result": _json_ss.dumps(items, ensure_ascii=False)}

                TOOL_HANDLERS["session_search"] = _session_search_handler
                TOOL_HANDLERS["gp_session_search"] = _session_search_handler
                logger.debug("session_search tool registered")
            except Exception:
                logger.debug("session_search tool registration failed", exc_info=True)

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
        initial_skills = len(self.skill_lib.load_all_active()) if self.skill_lib else 0
        self._cold_start.update_stats(skills_count=initial_skills)

        # LearningEffectivenessTracker: metrics observability
        from leapflow.learning.effectiveness import LearningEffectivenessTracker
        self._effectiveness_tracker = LearningEffectivenessTracker()

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

    async def cleanup(self) -> None:
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
        # Shutdown all memory providers (stops GC, closes DB)
        await self.memory.shutdown_all()
        if isinstance(self.rpc, CuaDriverClient):
            self.rpc.stop()
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
