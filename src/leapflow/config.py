"""Configuration loading from structured YAML with environment overrides.

Loading priority (highest wins):
    1. Process environment variables and explicit override files
    2. Workspace ``.leapflow/config.yaml`` overrides
    3. ``profiles/<profile>/config/*.yaml`` profile config
    4. ``config/user.yaml`` user defaults
    5. Hard-coded fallbacks
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from leapflow.config_loader import load_config_bundle
from leapflow.domain.trajectory import RecordingMode
from leapflow.layout import (
    PathLayout,
    ProfileLayout,
    ProfileManifest,
    build_layout,
    validate_profile_name,
    workspace_id_for_path,
)

logger = logging.getLogger(__name__)

DEFAULT_LLM_CONTEXT_LENGTH = 256_000
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_profile_name(profile: str) -> str:
    """Return a safe profile name or raise before any path construction."""
    return validate_profile_name(profile)

ALL_SIGNAL_CHANNELS = frozenset({
    "click", "app_switch", "clipboard", "clipboard_content",
    "keyboard", "scroll", "drag",
})


def _expand_path(value: str) -> Path:
    return Path(value.replace("~", str(Path.home()))).expanduser()


# Legacy constants are kept only for older import sites. Directory creation is
# owned by PathLayout/ProfileLayout.
_GLOBAL_SUBDIRS = ["logs"]
_PROFILE_SUBDIRS = []


def ensure_data_dir(data_dir: Path, *, profile: str = "default") -> None:
    """Create the LeapFlow data directory using the canonical layout."""
    build_layout(data_dir).ensure(profile_id=_validate_profile_name(profile))


def ensure_default_env(data_dir: Path) -> bool:
    """Deprecated: global .env is no longer a persistent configuration source."""
    ensure_data_dir(data_dir)
    return False


@dataclass(frozen=True)
class Settings:
    """Runtime settings for LeapFlow."""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_max_retries: int
    mock_host: bool
    duckdb_path: Path
    log_level: str

    # ── Memory Providers ──
    memory_working_max_tokens: int = 8192
    memory_episodic_ttl_s: float = 300.0        # 5 minutes
    memory_episodic_max_entries: int = 200
    memory_evolution_max_episodes: int = 1000
    memory_integration_enabled: bool = True      # wire MemoryManager into main loop
    memory_prefetch_timeout_s: float = 2.0       # max wait for prefetch in PREPARING
    memory_prefetch_limit: int = 5               # max entries injected per turn

    # ── Data Root & Profile ──
    data_dir: Path = Path("~/.leapflow")
    profile: str = "default"
    workspace_root: Path = Path(".")
    layout: PathLayout = field(default_factory=lambda: build_layout("~/.leapflow"))
    profile_layout: ProfileLayout = field(
        default_factory=lambda: build_layout("~/.leapflow").profile("default")
    )
    profile_manifest: ProfileManifest = field(
        default_factory=lambda: ProfileManifest.default("default")
    )
    config_sources: tuple[str, ...] = ()
    watched_config_paths: tuple[Path, ...] = ()
    runtime_dir: Path = field(
        default_factory=lambda: build_layout("~/.leapflow").profile("default").runtime_dir
    )

    # Audit
    audit_log_path: Path = field(
        default_factory=lambda: build_layout("~/.leapflow").profile("default").audit_log_path
    )

    # Imitation Learning
    pattern_library_path: str = ""        # Custom patterns.yaml path (empty = use default)
    snapshot_level_default: str = "light"  # Default snapshot level

    # Skill Code Generation
    codegen_sandbox: bool = True           # Enable AST safety validation
    codegen_max_retries: int = 2           # LLM code generation retries

    # Task DAG
    max_dag_concurrency: int = 3           # DAG max parallelism
    dag_node_timeout: float = 300.0        # Per-node timeout (seconds)

    # Intent Inference
    intent_inference_enabled: bool = True   # Enable LLM intent inference
    intent_inference_language: str = "zh"   # Intent inference language

    # Learning / Execution
    learn_idle_timeout: float = 300.0       # Auto-stop learning after idle (seconds)
    learn_auto_distill: bool = True         # Auto-trigger distillation on learn stop
    confirm_default_level: str = "confirm"  # Default confirmation level

    # Skill Documents (SKILL.md)
    skills_dir: Path = field(
        default_factory=lambda: build_layout("~/.leapflow").profile("default").skills_dir
    )
    skill_view_max_chars: int = 5000  # Max chars returned by skill_view tool
    skill_min_quality: float = 0.5    # SkillIndex quality threshold

    # Visual Track
    visual_track_enabled: bool = False
    visual_frame_cache_dir: Path = field(
        default_factory=lambda: build_layout("~/.leapflow").profile("default").cache.category_dir(
            scope="workspace",
            category="perception/keyframes",
            workspace_id=workspace_id_for_path(Path.cwd()),
        )
    )
    visual_sample_strategy: str = "keyframe"  # keyframe | periodic | all
    vlm_model: str = ""  # 空则复用 llm_model
    vlm_api_key: str = ""  # 空则复用 llm_api_key
    vlm_base_url: str = ""  # 空则复用 llm_base_url
    privacy_sensitive_apps: tuple = ()  # 隐私应用 bundle_id 黑名单

    # ── VLM Optimization ──

    # P1: Prefiltering
    vlm_prefilter_enabled: bool = True
    vlm_prefilter_skip_actions: tuple = (
        "file.create", "file.delete", "file.rename",
        "file.modify", "clipboard.copy", "app.switch", "ui.type",
    )
    vlm_prefilter_confidence_threshold: float = 0.85

    # P2: Frame Result Cache
    vlm_cache_enabled: bool = True
    vlm_cache_ttl: float = 300.0          # 缓存过期时间（秒）
    vlm_cache_max_size: int = 1000        # 最大缓存条目数

    # P3: Image Compression
    vlm_compression_enabled: bool = True
    vlm_compression_max_resolution: int = 1024  # 最大长边像素
    vlm_compression_quality: int = 75           # JPEG 质量 (1-100)
    vlm_compression_adaptive: bool = True       # 根据动作类型自适应

    # P4: Frame Tiling
    vlm_tiling_enabled: bool = True
    vlm_tiling_max_frames: int = 4              # 单次拼接最大帧数
    vlm_tiling_tile_size: int = 384             # 单帧缩放后最大长边
    vlm_tiling_gap: int = 4                     # 网格间距像素

    # ── Perception Depth ──
    text_capture_enabled: bool = True
    text_capture_exclude_apps: tuple = ()
    text_capture_secure_roles: tuple = ("AXSecureTextField",)
    text_capture_max_length: int = 500
    clipboard_max_length: int = 1024

    # ── Perceptual Field ──
    perceptual_field_enabled: bool = False
    perceptual_field_config: str = ""

    # ── Context Learning Attention ──
    attention_foreground_gate: bool = True
    attention_noise_patterns: tuple = ()
    attention_goal_relevance_threshold: float = 0.1
    attention_working_dir_inference: bool = True
    attention_domain_whitelist: bool = True

    # ── Recording Mode ──
    recording_mode: RecordingMode = RecordingMode.DEFAULT

    # ── Causal Inference ──
    causal_reorder_window_ms: float = 300.0
    causal_burst_limit: int = 50
    causal_max_chain_events: int = 20
    heuristic_time_decay_s: float = 0.5
    heuristic_space_decay_px: float = 200.0

    # ── World Model (Prediction Loop & Curiosity-Driven Learning) ──
    prediction_enabled: bool = True
    prediction_delta_threshold: float = 0.3
    curiosity_alpha: float = 0.4
    curiosity_beta: float = 0.3
    curiosity_gamma: float = 0.3
    curiosity_auto_balance: bool = True
    prediction_budget: int = 50
    comparison_budget: int = 20
    replay_budget: int = 3
    grading_budget: int = 5
    distillation_budget: int = 2
    replay_on_session_end: bool = True
    prediction_structural_blend: float = 0.4
    prediction_semantic_blend: float = 0.6
    prediction_semantic_threshold: float = 0.1
    prediction_rag_advantage_floor: float = -0.3
    prediction_failure_advantage: float = -0.5
    replay_regression_sample_size: int = 200
    memory_prune_age_days: float = 90.0

    # ── Semantic RAG ──
    semantic_embedding_provider: str = "tfidf"
    semantic_rerank_weight: float = 0.4

    # ── Budget Rebalancing ──
    budget_discovery_baseline: int = 2
    budget_regression_baseline: int = 1

    # ── AST Pre-Check ──
    ast_precheck_enabled: bool = True

    # ── Causal Chain Builder ──
    causal_window_s: float = 3.0
    causal_app_window_overrides: Dict[str, float] = field(default_factory=dict)
    causal_tier3_enabled: bool = False
    causal_tier3_confidence_threshold: float = 0.7
    mhms_fusion_enabled: bool = False

    # ── Attention Learning Feedback ──
    attention_curiosity_expand_threshold: float = 0.7
    attention_accuracy_contract_threshold: float = 0.1

    # ── Surprise Detection ──
    surprise_enabled: bool = True
    surprise_stat_weight: float = 0.4
    surprise_temporal_weight: float = 0.3
    surprise_pattern_weight: float = 0.3
    surprise_annotation_threshold: float = 0.5
    surprise_warmup_events: int = 50

    # ── Video Recording ──
    video_fps: int = 5
    video_resolution_scale: float = 0.75
    video_codec: str = "h264"
    video_max_segment_s: int = 600
    video_cache_dir: Path = field(
        default_factory=lambda: build_layout("~/.leapflow").profile("default").cache.category_dir(
            scope="workspace",
            category="video",
            workspace_id=workspace_id_for_path(Path.cwd()),
        )
    )
    video_cache_max_age_days: int = 7           # 视频缓存最大保留天数
    video_cache_max_size_gb: float = 5.0        # 视频缓存最大占用空间(GB)
    video_l2_enabled: bool = True
    video_l3_enabled: bool = True
    video_segmenter_min_s: float = 30.0
    video_segmenter_max_s: float = 600.0
    video_segmenter_idle_gap_s: float = 15.0
    video_segmenter_app_gap_s: float = 5.0
    video_segmenter_min_split_s: float = 1.0   # 分割最小片段长度（秒）
    video_max_l2_requests: int = 10
    video_max_l3_requests: int = 5
    video_start_timeout_s: float = 10.0         # 录制启动超时（秒）
    video_vlm_max_retries: int = 2               # VLM 调用最大重试次数
    video_vlm_retry_backoff_s: float = 1.0       # VLM 重试退避基数（秒）
    video_vlm_url_scheme: str = "base64"            # "base64" for local dev, HTTPS URL prefix for production
    video_l2_time_window_s: float = 2.0              # L2分析时间窗口（前后各N秒）

    # ── Video Timeline ──
    video_timeline_max_markers: int = 5000    # 在线 timeline 最大标记数
    video_timeline_compress_max: int = 200    # 压缩后保留的最大标记数
    video_timeline_merge_channels: str = "keyboard,scroll"  # 合并策略通道（逗号分隔）

    # ── Learnability Assessment ──
    learnability_enabled: bool = True
    learnability_min_steps: int = 3
    learnability_min_duration_s: float = 5.0
    learnability_max_idle_ratio: float = 0.80
    learnability_min_action_diversity: int = 2
    learnability_learn_threshold: float = 0.65
    learnability_ask_threshold: float = 0.40
    learnability_vlm_enabled: bool = True
    learnability_llm_enabled: bool = True
    learnability_rule_weight: float = 0.4
    learnability_vlm_weight: float = 0.3
    learnability_llm_weight: float = 0.3

    # ── Execution Loop Budget ──
    react_max_iterations: int = 20
    react_soft_limit: int = 14
    react_warning_threshold: int = 10
    tool_max_iterations: int = 30
    native_tool_calling_enabled: bool = True   # Use native OpenAI tool_calls when available
    stream_output: bool = True                   # Enable LLM streaming in interactive mode
    verbose_progress: bool = True                # Show detailed tool execution progress

    # Context Compression
    compress_threshold: int = 16
    compress_keep_tail: int = 4
    max_tool_output_chars: int = 2000
    max_tool_result_chars: int = 3000  # Per-tool result truncation for LLM context
    context_hard_limit_ratio: float = 0.92
    context_warning_ratio: float = 0.75
    tool_evidence_max_chars: int = 1200
    repeated_read_limit: int = 2
    long_task_convergence_round: int = 12
    context_expanded_ratio: float = 0.60
    context_finalizing_ratio: float = 0.90
    context_expanded_evidence_threshold: int = 2
    context_expanded_tool_call_threshold: int = 3
    context_research_source_threshold: int = 3
    context_research_evidence_threshold: int = 5

    # ── Error Recovery ──
    error_transient_max_retries: int = 3
    error_rate_limit_base_delay: float = 5.0
    max_consecutive_tool_failures: int = 3

    # ── Session Persistence ──
    session_persistence_enabled: bool = True

    # ── Multi-Provider LLM ──
    llm_fallback_providers: str = ""  # JSON array of fallback provider configs
    llm_aux_model: str = ""  # Auxiliary model for cheap operations (empty = reuse primary)
    llm_aux_api_key: str = ""  # Aux API key (empty = reuse primary)
    llm_aux_base_url: str = ""  # Aux base URL (empty = reuse primary)
    llm_context_length: int = DEFAULT_LLM_CONTEXT_LENGTH  # Primary provider's runtime context budget
    llm_credential_cooldown_s: float = 60.0  # Per-key rate-limit cooldown

    # ── Stream & Tool Robustness ──
    stale_stream_timeout_s: float = 180.0  # Idle timeout for streaming responses
    default_tool_timeout_s: float = 120.0  # Default per-tool execution timeout
    circuit_breaker_threshold: int = 5  # Consecutive failures before circuit opens
    circuit_breaker_cooldown_s: float = 60.0  # Circuit breaker cooldown period

    # ── Signal Fusion (vision_only mode) ──
    # Default is the full 7-channel set; ``LEAPFLOW_SIGNAL_CHANNELS=none`` disables
    # signal collection entirely (V0 baseline).
    signal_channels: frozenset = frozenset()
    signal_reactive_capture: bool = False

    # ── RPC Transport ──
    # Default fallback timeout (seconds) used by CuaDriverClient when no
    # method-specific timeout applies.
    rpc_timeout_default: float = 30.0

    # ── Cua Driver ──
    use_cua_driver: bool = True
    cua_driver_cmd: str = "cua-driver"

    # ── Workflow Copilot ──
    copilot_enabled: bool = True
    copilot_min_idle_ms: int = 500
    copilot_max_idle_ms: int = 5000
    copilot_cache_ttl_s: float = 30.0
    copilot_speculative_cache_size: int = 100
    copilot_action_ring_size: int = 10
    copilot_warmup_event_types: str = "app.focus_change,context.change,ui.action"
    notification_renderer: str = "auto"  # "os" | "stderr" | "log" | "auto" | "none"

    # ── Engine Live Signals ──
    live_signal_kinds: str = "app.focus_change,fs.change,context.change,intent.signal"

    # ── Hub (cloud collaboration) ──
    hub_type: str = "modelscope"
    hub_default_owner: str = ""
    hub_default_visibility: str = "private"
    hub_sync_strategy: str = "remote-wins"
    hub_sync_copilot: bool = True
    hub_repo_prefix: str = "leapflow-"
    hub_search_sources: str = "modelscope"  # comma-separated backend names for multi-source search

    # ── Observer / Daemon ──
    observer_auto_start: bool = False   # Auto-start ObservationDaemon on initialize
    observer_enabled_set: Dict[str, bool] = field(default_factory=lambda: {
        "fs_watcher": True,
        "app_focus": True,
        "clipboard": True,
        "input_tap": False,
    })

    # ── Scheduler ──
    scheduler_enabled: bool = True
    scheduler_tick_seconds: int = 60
    scheduler_grace_seconds: float = 120.0
    scheduler_default_tier: str = "auto"  # auto | local | cloud

    @property
    def profile_dir(self) -> Path:
        """Root directory for the active profile."""
        return self.profile_layout.root

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def has_vlm_credentials(self) -> bool:
        """Return True when visual perception can call a VLM provider."""
        return bool((self.vlm_api_key or self.llm_api_key).strip())


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into base. Overlay values take precedence."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_overlay(data_dir: Path) -> Dict[str, str]:
    """Compatibility wrapper returning flattened structured config values."""
    layout = build_layout(data_dir)
    profile = _validate_profile_name(os.getenv("LEAPFLOW_PROFILE", "default"))
    workspace_root = _expand_path(
        os.getenv("LEAPFLOW_WORKSPACE_ROOT", str(Path.cwd())).strip() or str(Path.cwd())
    ).resolve()
    profile_layout = layout.profile(profile)
    bundle = load_config_bundle(layout, profile_layout, workspace_root)
    return bundle.env


def _flatten_yaml(
    node: Any, *, prefix: str, env_vars: Dict[str, str]
) -> None:
    """Flatten nested YAML dict to LEAPFLOW_X_Y_Z = value strings."""
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}_{key}".upper()
            _flatten_yaml(value, prefix=child_prefix, env_vars=env_vars)
    else:
        env_vars[prefix] = str(node)


def load_config(*, env_file: str | Path | None = None) -> Settings:
    """Load settings from structured YAML and process environment overrides."""
    if env_file is not None:
        load_dotenv(env_file, override=False)

    data_dir = _expand_path(os.getenv("LEAPFLOW_DATA_DIR", "~/.leapflow").strip())
    profile = _validate_profile_name(os.getenv("LEAPFLOW_PROFILE", "default"))
    workspace_root = _expand_path(
        os.getenv("LEAPFLOW_WORKSPACE_ROOT", str(Path.cwd())).strip() or str(Path.cwd())
    ).resolve()
    layout = build_layout(data_dir)
    profile_layout = layout.ensure(profile_id=profile)
    bundle = load_config_bundle(layout, profile_layout, workspace_root)

    original_env = dict(os.environ)
    injected_keys: list[str] = []
    for key, value in bundle.env.items():
        if key not in original_env:
            os.environ[key] = value
            injected_keys.append(key)
    try:
        settings = _build_settings_from_env(
            layout=layout,
            profile_layout=profile_layout,
            profile_manifest=profile_layout.load_manifest(),
            config_sources=tuple(str(source.path) for source in bundle.sources),
            watched_config_paths=bundle.watched_paths,
        )
    finally:
        for key in injected_keys:
            os.environ.pop(key, None)
    return settings


def _build_settings_from_env(
    *,
    layout: PathLayout | None = None,
    profile_layout: ProfileLayout | None = None,
    profile_manifest: ProfileManifest | None = None,
    config_sources: tuple[str, ...] = (),
    watched_config_paths: tuple[Path, ...] = (),
) -> Settings:
    """Build Settings from current os.environ."""
    api_key = os.getenv("LEAPFLOW_LLM_API_KEY", "").strip()
    base_url = os.getenv(
        "LEAPFLOW_LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    model = os.getenv("LEAPFLOW_LLM_MODEL", "qwen-plus").strip()
    max_retries = int(os.getenv("LEAPFLOW_LLM_MAX_RETRIES", "3"))

    data_dir = _expand_path(os.getenv("LEAPFLOW_DATA_DIR", "~/.leapflow").strip())
    profile = _validate_profile_name(os.getenv("LEAPFLOW_PROFILE", "default"))
    workspace_root = _expand_path(
        os.getenv("LEAPFLOW_WORKSPACE_ROOT", str(Path.cwd())).strip() or str(Path.cwd())
    ).resolve()
    layout = layout or build_layout(data_dir)
    profile_layout = profile_layout or layout.profile(profile)
    profile_manifest = profile_manifest or profile_layout.load_manifest()
    workspace_id = workspace_id_for_path(workspace_root)
    _profile_dir = profile_layout.root
    runtime_dir = os.getenv("LEAPFLOW_RUNTIME_DIR", str(profile_layout.runtime_dir)).strip()

    mock_host = os.getenv("LEAPFLOW_MOCK_HOST", "0").strip() in ("1", "true", "True", "yes")
    duckdb = os.getenv("LEAPFLOW_DUCKDB_PATH", str(profile_layout.duckdb_path)).strip()
    log_level = os.getenv("LEAPFLOW_LOG_LEVEL", "INFO").strip()

    # Memory Providers
    memory_working_max_tokens = int(os.getenv("LEAPFLOW_MEMORY_WORKING_MAX_TOKENS", "8192"))
    memory_episodic_ttl_s = float(os.getenv("LEAPFLOW_MEMORY_EPISODIC_TTL_S", "300.0"))
    memory_episodic_max_entries = int(os.getenv("LEAPFLOW_MEMORY_EPISODIC_MAX_ENTRIES", "200"))
    memory_evolution_max_episodes = int(os.getenv("LEAPFLOW_MEMORY_EVOLUTION_MAX_EPISODES", "1000"))
    memory_integration_enabled = os.getenv("LEAPFLOW_MEMORY_INTEGRATION_ENABLED", "1").strip() in ("1", "true", "True", "yes")
    memory_prefetch_timeout_s = float(os.getenv("LEAPFLOW_MEMORY_PREFETCH_TIMEOUT_S", "2.0"))
    memory_prefetch_limit = int(os.getenv("LEAPFLOW_MEMORY_PREFETCH_LIMIT", "5"))

    # Audit
    audit_log_path = os.getenv("LEAPFLOW_AUDIT_LOG_PATH", str(profile_layout.audit_log_path)).strip()

    # Visual Track
    visual_track_enabled = os.getenv("LEAPFLOW_VISUAL_TRACK_ENABLED", "0").strip() in ("1", "true", "True", "yes")
    visual_frame_cache_dir = os.getenv(
        "LEAPFLOW_VISUAL_FRAME_CACHE_DIR",
        str(profile_layout.cache.category_dir(
            scope="workspace",
            category="perception/keyframes",
            workspace_id=workspace_id,
        )),
    ).strip()
    visual_sample_strategy = os.getenv("LEAPFLOW_VISUAL_SAMPLE_STRATEGY", "keyframe").strip()
    vlm_model = os.getenv("LEAPFLOW_VLM_MODEL", "").strip() or model
    vlm_api_key = os.getenv("LEAPFLOW_VLM_API_KEY", "").strip()
    vlm_base_url = os.getenv("LEAPFLOW_VLM_BASE_URL", "").strip()
    privacy_sensitive_apps_raw = os.getenv("LEAPFLOW_PRIVACY_SENSITIVE_APPS", "").strip()
    privacy_sensitive_apps = tuple(b.strip() for b in privacy_sensitive_apps_raw.split(",") if b.strip()) if privacy_sensitive_apps_raw else ()

    # Skill Documents
    skills_dir = os.getenv("LEAPFLOW_SKILLS_DIR", str(profile_layout.skills_dir)).strip()
    skill_view_max_chars = int(os.getenv("LEAPFLOW_SKILL_VIEW_MAX_CHARS", "5000"))
    skill_min_quality = float(os.getenv("LEAPFLOW_SKILL_MIN_QUALITY", "0.5"))

    _bool = lambda key, default: os.getenv(key, default).strip().lower() in ("1", "true", "yes")  # noqa: E731

    # VLM Optimization – P1: Prefiltering
    vlm_prefilter_enabled = _bool("LEAPFLOW_VLM_PREFILTER_ENABLED", "true")
    vlm_prefilter_skip_raw = os.getenv(
        "LEAPFLOW_VLM_PREFILTER_SKIP_ACTIONS",
        "file.create,file.delete,file.rename,file.modify,clipboard.copy,app.switch,ui.type",
    )
    vlm_prefilter_skip_actions = tuple(s.strip() for s in vlm_prefilter_skip_raw.split(",") if s.strip())
    vlm_prefilter_confidence_threshold = float(os.getenv("LEAPFLOW_VLM_PREFILTER_CONFIDENCE_THRESHOLD", "0.85"))

    # VLM Optimization – P2: Frame Result Cache
    vlm_cache_enabled = _bool("LEAPFLOW_VLM_CACHE_ENABLED", "true")
    vlm_cache_ttl = float(os.getenv("LEAPFLOW_VLM_CACHE_TTL", "300"))
    vlm_cache_max_size = int(os.getenv("LEAPFLOW_VLM_CACHE_MAX_SIZE", "1000"))

    # VLM Optimization – P3: Image Compression
    vlm_compression_enabled = _bool("LEAPFLOW_VLM_COMPRESSION_ENABLED", "true")
    vlm_compression_max_resolution = int(os.getenv("LEAPFLOW_VLM_COMPRESSION_MAX_RESOLUTION", "1024"))
    vlm_compression_quality = int(os.getenv("LEAPFLOW_VLM_COMPRESSION_QUALITY", "75"))
    vlm_compression_adaptive = _bool("LEAPFLOW_VLM_COMPRESSION_ADAPTIVE", "true")

    # VLM Optimization – P4: Frame Tiling
    vlm_tiling_enabled = _bool("LEAPFLOW_VLM_TILING_ENABLED", "true")
    vlm_tiling_max_frames = int(os.getenv("LEAPFLOW_VLM_TILING_MAX_FRAMES", "4"))
    vlm_tiling_tile_size = int(os.getenv("LEAPFLOW_VLM_TILING_TILE_SIZE", "384"))
    vlm_tiling_gap = int(os.getenv("LEAPFLOW_VLM_TILING_GAP", "4"))

    # Perception Depth
    text_capture_enabled = _bool("LEAPFLOW_TEXT_CAPTURE_ENABLED", "true")
    text_capture_exclude_apps_raw = os.getenv("LEAPFLOW_TEXT_CAPTURE_EXCLUDE_APPS", "").strip()
    text_capture_exclude_apps = tuple(
        b.strip() for b in text_capture_exclude_apps_raw.split(",") if b.strip()
    ) if text_capture_exclude_apps_raw else ()
    text_capture_secure_roles_raw = os.getenv("LEAPFLOW_TEXT_CAPTURE_SECURE_ROLES", "AXSecureTextField").strip()
    text_capture_secure_roles = tuple(
        r.strip() for r in text_capture_secure_roles_raw.split(",") if r.strip()
    )
    text_capture_max_length = int(os.getenv("LEAPFLOW_TEXT_CAPTURE_MAX_LENGTH", "500"))
    clipboard_max_length = int(os.getenv("LEAPFLOW_CLIPBOARD_MAX_LENGTH", "1024"))

    # Perceptual Field
    perceptual_field_enabled = _bool("LEAPFLOW_PERCEPTUAL_FIELD_ENABLED", "false")
    perceptual_field_config = os.getenv(
        "LEAPFLOW_PERCEPTUAL_FIELD_CONFIG", str(profile_layout.config_dir / "perceptual_fields.yaml")
    ).strip()

    # Context Learning Attention
    attention_foreground_gate = _bool("LEAPFLOW_ATTENTION_FOREGROUND_GATE", "true")
    attention_noise_patterns_raw = os.getenv("LEAPFLOW_ATTENTION_NOISE_PATTERNS", "").strip()
    attention_noise_patterns = tuple(
        p.strip() for p in attention_noise_patterns_raw.split(",") if p.strip()
    ) if attention_noise_patterns_raw else ()
    attention_goal_relevance_threshold = float(
        os.getenv("LEAPFLOW_ATTENTION_GOAL_RELEVANCE_THRESHOLD", "0.1")
    )
    attention_working_dir_inference = _bool("LEAPFLOW_ATTENTION_WORKING_DIR_INFERENCE", "true")
    attention_domain_whitelist = _bool("LEAPFLOW_ATTENTION_DOMAIN_WHITELIST", "true")

    # Recording Mode
    recording_mode = RecordingMode.from_str(os.getenv("LEAPFLOW_RECORDING_MODE", "default"))

    # Causal Inference
    causal_reorder_window_ms = float(os.getenv("LEAPFLOW_CAUSAL_REORDER_WINDOW_MS", "300"))
    causal_burst_limit = int(os.getenv("LEAPFLOW_CAUSAL_BURST_LIMIT", "50"))
    causal_max_chain_events = int(os.getenv("LEAPFLOW_CAUSAL_MAX_CHAIN_EVENTS", "20"))
    heuristic_time_decay_s = float(os.getenv("LEAPFLOW_HEURISTIC_TIME_DECAY_S", "0.5"))
    heuristic_space_decay_px = float(os.getenv("LEAPFLOW_HEURISTIC_SPACE_DECAY_PX", "200.0"))

    # World Model
    prediction_enabled = _bool("LEAPFLOW_PREDICTION_ENABLED", "true")
    prediction_delta_threshold = float(os.getenv("LEAPFLOW_PREDICTION_DELTA_THRESHOLD", "0.3"))
    curiosity_alpha = float(os.getenv("LEAPFLOW_CURIOSITY_ALPHA", "0.4"))
    curiosity_beta = float(os.getenv("LEAPFLOW_CURIOSITY_BETA", "0.3"))
    curiosity_gamma = float(os.getenv("LEAPFLOW_CURIOSITY_GAMMA", "0.3"))
    curiosity_auto_balance = _bool("LEAPFLOW_CURIOSITY_AUTO_BALANCE", "true")
    prediction_budget = int(os.getenv("LEAPFLOW_PREDICTION_BUDGET", "50"))
    comparison_budget = int(os.getenv("LEAPFLOW_COMPARISON_BUDGET", "20"))
    replay_budget = int(os.getenv("LEAPFLOW_REPLAY_BUDGET", "3"))
    grading_budget = int(os.getenv("LEAPFLOW_GRADING_BUDGET", "5"))
    distillation_budget = int(os.getenv("LEAPFLOW_DISTILLATION_BUDGET", "2"))
    replay_on_session_end = _bool("LEAPFLOW_REPLAY_ON_SESSION_END", "true")
    prediction_structural_blend = float(os.getenv("LEAPFLOW_PREDICTION_STRUCTURAL_BLEND", "0.4"))
    prediction_semantic_blend = float(os.getenv("LEAPFLOW_PREDICTION_SEMANTIC_BLEND", "0.6"))
    prediction_semantic_threshold = float(os.getenv("LEAPFLOW_PREDICTION_SEMANTIC_THRESHOLD", "0.1"))
    prediction_rag_advantage_floor = float(os.getenv("LEAPFLOW_PREDICTION_RAG_ADVANTAGE_FLOOR", "-0.3"))
    prediction_failure_advantage = float(os.getenv("LEAPFLOW_PREDICTION_FAILURE_ADVANTAGE", "-0.5"))
    replay_regression_sample_size = int(os.getenv("LEAPFLOW_REPLAY_REGRESSION_SAMPLE_SIZE", "200"))
    memory_prune_age_days = float(os.getenv("LEAPFLOW_MEMORY_PRUNE_AGE_DAYS", "90.0"))
    semantic_embedding_provider = os.getenv("LEAPFLOW_SEMANTIC_EMBEDDING_PROVIDER", "tfidf")
    semantic_rerank_weight = float(os.getenv("LEAPFLOW_SEMANTIC_RERANK_WEIGHT", "0.4"))
    budget_discovery_baseline = int(os.getenv("LEAPFLOW_BUDGET_DISCOVERY_BASELINE", "2"))
    budget_regression_baseline = int(os.getenv("LEAPFLOW_BUDGET_REGRESSION_BASELINE", "1"))
    ast_precheck_enabled = _bool("LEAPFLOW_AST_PRECHECK_ENABLED", "true")
    causal_window_s = float(os.getenv("LEAPFLOW_CAUSAL_WINDOW_S", "3.0"))
    causal_app_window_overrides: dict = {}
    _raw_overrides = os.getenv("LEAPFLOW_CAUSAL_APP_WINDOW_OVERRIDES", "")
    if _raw_overrides:
        import json as _json
        try:
            causal_app_window_overrides = {str(k): float(v) for k, v in _json.loads(_raw_overrides).items()}
        except Exception:
            logger.warning("Invalid LEAPFLOW_CAUSAL_APP_WINDOW_OVERRIDES: %s", _raw_overrides)
    causal_tier3_enabled = os.getenv("LEAPFLOW_CAUSAL_TIER3_ENABLED", "").lower() in ("1", "true", "yes")
    causal_tier3_confidence_threshold = float(os.getenv("LEAPFLOW_CAUSAL_TIER3_CONFIDENCE_THRESHOLD", "0.7"))
    mhms_fusion_enabled = os.getenv("LEAPFLOW_MHMS_FUSION_ENABLED", "").lower() in ("1", "true", "yes")
    # Attention learning feedback
    attention_curiosity_expand_threshold = float(os.getenv("LEAPFLOW_ATTENTION_CURIOSITY_EXPAND_THRESHOLD", "0.7"))
    attention_accuracy_contract_threshold = float(os.getenv("LEAPFLOW_ATTENTION_ACCURACY_CONTRACT_THRESHOLD", "0.1"))
    # Surprise detection
    surprise_enabled = _bool("LEAPFLOW_SURPRISE_ENABLED", "true")
    surprise_stat_weight = float(os.getenv("LEAPFLOW_SURPRISE_STAT_WEIGHT", "0.4"))
    surprise_temporal_weight = float(os.getenv("LEAPFLOW_SURPRISE_TEMPORAL_WEIGHT", "0.3"))
    surprise_pattern_weight = float(os.getenv("LEAPFLOW_SURPRISE_PATTERN_WEIGHT", "0.3"))
    surprise_annotation_threshold = float(os.getenv("LEAPFLOW_SURPRISE_ANNOTATION_THRESHOLD", "0.5"))
    surprise_warmup_events = int(os.getenv("LEAPFLOW_SURPRISE_WARMUP_EVENTS", "50"))

    # Video Recording
    video_fps = int(os.getenv("LEAPFLOW_VIDEO_FPS", "5"))
    video_resolution_scale = float(os.getenv("LEAPFLOW_VIDEO_RESOLUTION_SCALE", "0.75"))
    video_codec = os.getenv("LEAPFLOW_VIDEO_CODEC", "h264").strip()
    video_max_segment_s = int(os.getenv("LEAPFLOW_VIDEO_MAX_SEGMENT_S", "600"))
    video_cache_dir = os.getenv(
        "LEAPFLOW_VIDEO_CACHE_DIR",
        str(profile_layout.cache.category_dir(scope="workspace", category="video", workspace_id=workspace_id)),
    ).strip()
    video_cache_max_age_days = int(os.getenv("LEAPFLOW_VIDEO_CACHE_MAX_AGE_DAYS", "7"))
    video_cache_max_size_gb = float(os.getenv("LEAPFLOW_VIDEO_CACHE_MAX_SIZE_GB", "5.0"))
    video_l2_enabled = _bool("LEAPFLOW_VIDEO_L2_ENABLED", "true")
    video_l3_enabled = _bool("LEAPFLOW_VIDEO_L3_ENABLED", "true")
    video_segmenter_min_s = float(os.getenv("LEAPFLOW_VIDEO_SEGMENTER_MIN_S", "30"))
    video_segmenter_max_s = float(os.getenv("LEAPFLOW_VIDEO_SEGMENTER_MAX_S", "600"))
    video_segmenter_idle_gap_s = float(os.getenv("LEAPFLOW_VIDEO_SEGMENTER_IDLE_GAP_S", "15"))
    video_segmenter_app_gap_s = float(os.getenv("LEAPFLOW_VIDEO_SEGMENTER_APP_GAP_S", "5"))
    video_segmenter_min_split_s = float(os.getenv("LEAPFLOW_VIDEO_SEGMENTER_MIN_SPLIT_S", "1.0"))
    video_max_l2_requests = int(os.getenv("LEAPFLOW_VIDEO_MAX_L2_REQUESTS", "10"))
    video_max_l3_requests = int(os.getenv("LEAPFLOW_VIDEO_MAX_L3_REQUESTS", "5"))
    video_start_timeout_s = float(os.getenv("LEAPFLOW_VIDEO_START_TIMEOUT_S", "10.0"))
    video_vlm_max_retries = int(os.getenv("LEAPFLOW_VIDEO_VLM_MAX_RETRIES", "2"))
    video_vlm_retry_backoff_s = float(os.getenv("LEAPFLOW_VIDEO_VLM_RETRY_BACKOFF_S", "1.0"))
    video_vlm_url_scheme = os.getenv("LEAPFLOW_VIDEO_VLM_URL_SCHEME", "base64").strip()
    video_l2_time_window_s = float(os.environ.get("LEAPFLOW_VIDEO_L2_TIME_WINDOW_S", "2.0"))
    video_timeline_max_markers = int(os.getenv("LEAPFLOW_VIDEO_TIMELINE_MAX_MARKERS", "5000"))
    video_timeline_compress_max = int(os.getenv("LEAPFLOW_VIDEO_TIMELINE_COMPRESS_MAX", "200"))
    video_timeline_merge_channels = os.getenv("LEAPFLOW_VIDEO_TIMELINE_MERGE_CHANNELS", "keyboard,scroll").strip()

    # Learnability Assessment
    learnability_enabled = _bool("LEAPFLOW_LEARNABILITY_ENABLED", "true")
    learnability_min_steps = int(os.getenv("LEAPFLOW_LEARNABILITY_MIN_STEPS", "3"))
    learnability_min_duration_s = float(os.getenv("LEAPFLOW_LEARNABILITY_MIN_DURATION_S", "5.0"))
    learnability_max_idle_ratio = float(os.getenv("LEAPFLOW_LEARNABILITY_MAX_IDLE_RATIO", "0.80"))
    learnability_min_action_diversity = int(os.getenv("LEAPFLOW_LEARNABILITY_MIN_ACTION_DIVERSITY", "2"))
    learnability_learn_threshold = float(os.getenv("LEAPFLOW_LEARNABILITY_LEARN_THRESHOLD", "0.65"))
    learnability_ask_threshold = float(os.getenv("LEAPFLOW_LEARNABILITY_ASK_THRESHOLD", "0.40"))
    learnability_vlm_enabled = _bool("LEAPFLOW_LEARNABILITY_VLM_ENABLED", "true")
    learnability_llm_enabled = _bool("LEAPFLOW_LEARNABILITY_LLM_ENABLED", "true")
    learnability_rule_weight = float(os.getenv("LEAPFLOW_LEARNABILITY_RULE_WEIGHT", "0.4"))
    learnability_vlm_weight = float(os.getenv("LEAPFLOW_LEARNABILITY_VLM_WEIGHT", "0.3"))
    learnability_llm_weight = float(os.getenv("LEAPFLOW_LEARNABILITY_LLM_WEIGHT", "0.3"))

    # Execution Loop Budget
    react_max_iterations = int(os.getenv("LEAPFLOW_REACT_MAX_ITERATIONS", "20"))
    react_soft_limit = int(os.getenv("LEAPFLOW_REACT_SOFT_LIMIT", "14"))
    react_warning_threshold = int(os.getenv("LEAPFLOW_REACT_WARNING_THRESHOLD", "10"))
    tool_max_iterations = int(os.getenv("LEAPFLOW_TOOL_MAX_ITERATIONS", "30"))
    native_tool_calling_enabled = os.getenv("LEAPFLOW_NATIVE_TOOL_CALLING_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    stream_output = os.getenv("LEAPFLOW_STREAM_OUTPUT", "1").strip().lower() in ("1", "true", "yes")
    verbose_progress = os.getenv("LEAPFLOW_VERBOSE_PROGRESS", "1").strip().lower() in ("1", "true", "yes")

    # Context Compression
    compress_threshold = int(os.getenv("LEAPFLOW_COMPRESS_THRESHOLD", "16"))
    compress_keep_tail = int(os.getenv("LEAPFLOW_COMPRESS_KEEP_TAIL", "4"))
    max_tool_output_chars = int(os.getenv("LEAPFLOW_MAX_TOOL_OUTPUT_CHARS", "2000"))
    max_tool_result_chars = int(os.getenv("LEAPFLOW_MAX_TOOL_RESULT_CHARS", "3000"))
    context_hard_limit_ratio = float(os.getenv("LEAPFLOW_CONTEXT_HARD_LIMIT_RATIO", "0.92"))
    context_warning_ratio = float(os.getenv("LEAPFLOW_CONTEXT_WARNING_RATIO", "0.75"))
    tool_evidence_max_chars = int(os.getenv("LEAPFLOW_TOOL_EVIDENCE_MAX_CHARS", "1200"))
    repeated_read_limit = int(os.getenv("LEAPFLOW_REPEATED_READ_LIMIT", "2"))
    long_task_convergence_round = int(os.getenv("LEAPFLOW_LONG_TASK_CONVERGENCE_ROUND", "12"))
    context_expanded_ratio = float(os.getenv("LEAPFLOW_CONTEXT_EXPANDED_RATIO", "0.60"))
    context_finalizing_ratio = float(os.getenv("LEAPFLOW_CONTEXT_FINALIZING_RATIO", "0.90"))
    context_expanded_evidence_threshold = int(os.getenv("LEAPFLOW_CONTEXT_EXPANDED_EVIDENCE_THRESHOLD", "2"))
    context_expanded_tool_call_threshold = int(os.getenv("LEAPFLOW_CONTEXT_EXPANDED_TOOL_CALL_THRESHOLD", "3"))
    context_research_source_threshold = int(os.getenv("LEAPFLOW_CONTEXT_RESEARCH_SOURCE_THRESHOLD", "3"))
    context_research_evidence_threshold = int(os.getenv("LEAPFLOW_CONTEXT_RESEARCH_EVIDENCE_THRESHOLD", "5"))

    # Error Recovery
    error_transient_max_retries = int(os.getenv("LEAPFLOW_ERROR_TRANSIENT_MAX_RETRIES", "3"))
    error_rate_limit_base_delay = float(os.getenv("LEAPFLOW_ERROR_RATE_LIMIT_BASE_DELAY", "5.0"))
    max_consecutive_tool_failures = int(os.getenv("LEAPFLOW_MAX_CONSECUTIVE_TOOL_FAILURES", "3"))

    # Session Persistence
    session_persistence_enabled = _bool("LEAPFLOW_SESSION_PERSISTENCE_ENABLED", "true")

    # Multi-Provider LLM
    llm_fallback_providers = os.getenv("LEAPFLOW_LLM_FALLBACK_PROVIDERS", "").strip()
    llm_aux_model = os.getenv("LEAPFLOW_LLM_AUX_MODEL", "").strip()
    llm_aux_api_key = os.getenv("LEAPFLOW_LLM_AUX_API_KEY", "").strip()
    llm_aux_base_url = os.getenv("LEAPFLOW_LLM_AUX_BASE_URL", "").strip()
    llm_context_length = int(os.getenv("LEAPFLOW_LLM_CONTEXT_LENGTH", str(DEFAULT_LLM_CONTEXT_LENGTH)))
    llm_credential_cooldown_s = float(os.getenv("LEAPFLOW_LLM_CREDENTIAL_COOLDOWN_S", "60.0"))

    # Stream & Tool Robustness
    stale_stream_timeout_s = float(os.getenv("LEAPFLOW_STALE_STREAM_TIMEOUT_S", "180.0"))
    default_tool_timeout_s = float(os.getenv("LEAPFLOW_DEFAULT_TOOL_TIMEOUT_S", "120.0"))
    circuit_breaker_threshold = int(os.getenv("LEAPFLOW_CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_breaker_cooldown_s = float(os.getenv("LEAPFLOW_CIRCUIT_BREAKER_COOLDOWN_S", "60.0"))

    # Signal Fusion
    # Default = "all": collect every supported channel (V7 full fusion). Set
    # to "none" or empty list to disable; comma-separated list selects a
    # specific subset for ablation experiments. See .env.example for the
    # complete ablation matrix.
    signal_channels_raw = os.getenv("LEAPFLOW_SIGNAL_CHANNELS", "all").strip().lower()
    if signal_channels_raw == "none":
        signal_channels: frozenset = frozenset()
    elif signal_channels_raw in ("", "all"):
        signal_channels = ALL_SIGNAL_CHANNELS
    else:
        signal_channels = frozenset(
            ch.strip() for ch in signal_channels_raw.split(",") if ch.strip()
        ) & ALL_SIGNAL_CHANNELS
    signal_reactive_capture = _bool("LEAPFLOW_SIGNAL_REACTIVE_CAPTURE", "false")

    # RPC Transport
    rpc_timeout_default = float(os.getenv("LEAPFLOW_RPC_TIMEOUT_DEFAULT", "30.0"))

    # Cua Driver
    use_cua_driver = _bool("LEAPFLOW_USE_CUA_DRIVER", "true")
    cua_driver_cmd = os.getenv("LEAPFLOW_CUA_DRIVER_CMD", "cua-driver").strip()

    # Workflow Copilot
    copilot_enabled = _bool("LEAPFLOW_COPILOT_ENABLED", "true")
    copilot_min_idle_ms = int(os.getenv("LEAPFLOW_COPILOT_MIN_IDLE_MS", "500"))
    copilot_max_idle_ms = int(os.getenv("LEAPFLOW_COPILOT_MAX_IDLE_MS", "5000"))
    copilot_cache_ttl_s = float(os.getenv("LEAPFLOW_COPILOT_CACHE_TTL_S", "30.0"))
    copilot_speculative_cache_size = int(os.getenv("LEAPFLOW_COPILOT_SPECULATIVE_CACHE_SIZE", "100"))
    copilot_action_ring_size = int(os.getenv("LEAPFLOW_COPILOT_ACTION_RING_SIZE", "10"))
    copilot_warmup_event_types = os.getenv(
        "LEAPFLOW_COPILOT_WARMUP_EVENT_TYPES", "app.focus_change,context.change,ui.action",
    )
    notification_renderer = os.getenv("LEAPFLOW_NOTIFICATION_RENDERER", "auto")

    live_signal_kinds = os.getenv(
        "LEAPFLOW_LIVE_SIGNAL_KINDS", "app.focus_change,fs.change,context.change,intent.signal",
    )

    # Hub (cloud collaboration)
    hub_type = os.getenv("LEAPFLOW_HUB_TYPE", "modelscope")
    hub_default_owner = os.getenv("LEAPFLOW_HUB_DEFAULT_OWNER", "")
    hub_default_visibility = os.getenv("LEAPFLOW_HUB_DEFAULT_VISIBILITY", "private")
    hub_sync_strategy = os.getenv("LEAPFLOW_HUB_SYNC_STRATEGY", "remote-wins")
    hub_sync_copilot = _bool("LEAPFLOW_HUB_SYNC_COPILOT", "true")
    hub_repo_prefix = os.getenv("LEAPFLOW_HUB_REPO_PREFIX", "leapflow-")
    hub_search_sources = os.getenv("LEAPFLOW_HUB_SEARCH_SOURCES", "modelscope")

    # Observer / Daemon
    observer_auto_start = _bool("LEAPFLOW_OBSERVER_AUTO_START", "false")
    _observer_enabled_raw = os.getenv("LEAPFLOW_OBSERVER_ENABLED_SET", "")
    observer_enabled_set: Dict[str, bool] = {
        "fs_watcher": True, "app_focus": True, "clipboard": True, "input_tap": False,
    }
    if _observer_enabled_raw:
        import json as _json_obs
        try:
            observer_enabled_set = {str(k): bool(v) for k, v in _json_obs.loads(_observer_enabled_raw).items()}
        except Exception:
            logger.warning("Invalid LEAPFLOW_OBSERVER_ENABLED_SET: %s", _observer_enabled_raw)

    # Scheduler
    scheduler_enabled = _bool("LEAPFLOW_SCHEDULER_ENABLED", "true")
    scheduler_tick_seconds = int(os.getenv("LEAPFLOW_SCHEDULER_TICK_SECONDS", "60"))
    scheduler_grace_seconds = float(os.getenv("LEAPFLOW_SCHEDULER_GRACE_SECONDS", "120.0"))
    scheduler_default_tier = os.getenv("LEAPFLOW_SCHEDULER_DEFAULT_TIER", "auto")

    settings = Settings(
        llm_api_key=api_key,
        llm_base_url=base_url.rstrip("/"),
        llm_model=model,
        llm_max_retries=max(1, max_retries),
        mock_host=mock_host,
        duckdb_path=_expand_path(duckdb),
        log_level=log_level,
        # Memory Providers
        memory_working_max_tokens=memory_working_max_tokens,
        memory_episodic_ttl_s=memory_episodic_ttl_s,
        memory_episodic_max_entries=memory_episodic_max_entries,
        memory_evolution_max_episodes=memory_evolution_max_episodes,
        memory_integration_enabled=memory_integration_enabled,
        memory_prefetch_timeout_s=memory_prefetch_timeout_s,
        memory_prefetch_limit=memory_prefetch_limit,
        data_dir=data_dir,
        profile=profile,
        workspace_root=workspace_root,
        layout=layout,
        profile_layout=profile_layout,
        profile_manifest=profile_manifest,
        config_sources=config_sources,
        watched_config_paths=watched_config_paths,
        runtime_dir=_expand_path(runtime_dir),
        audit_log_path=_expand_path(audit_log_path),
        skills_dir=_expand_path(skills_dir),
        skill_view_max_chars=skill_view_max_chars,
        skill_min_quality=skill_min_quality,
        visual_track_enabled=visual_track_enabled,
        visual_frame_cache_dir=_expand_path(visual_frame_cache_dir),
        visual_sample_strategy=visual_sample_strategy,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
        vlm_base_url=vlm_base_url,
        privacy_sensitive_apps=privacy_sensitive_apps,
        # VLM Optimization
        vlm_prefilter_enabled=vlm_prefilter_enabled,
        vlm_prefilter_skip_actions=vlm_prefilter_skip_actions,
        vlm_prefilter_confidence_threshold=vlm_prefilter_confidence_threshold,
        vlm_cache_enabled=vlm_cache_enabled,
        vlm_cache_ttl=vlm_cache_ttl,
        vlm_cache_max_size=vlm_cache_max_size,
        vlm_compression_enabled=vlm_compression_enabled,
        vlm_compression_max_resolution=vlm_compression_max_resolution,
        vlm_compression_quality=vlm_compression_quality,
        vlm_compression_adaptive=vlm_compression_adaptive,
        vlm_tiling_enabled=vlm_tiling_enabled,
        vlm_tiling_max_frames=vlm_tiling_max_frames,
        vlm_tiling_tile_size=vlm_tiling_tile_size,
        vlm_tiling_gap=vlm_tiling_gap,
        # Perceptual Field
        perceptual_field_enabled=perceptual_field_enabled,
        perceptual_field_config=perceptual_field_config,
        # Perception Depth
        text_capture_enabled=text_capture_enabled,
        text_capture_exclude_apps=text_capture_exclude_apps,
        text_capture_secure_roles=text_capture_secure_roles,
        text_capture_max_length=text_capture_max_length,
        clipboard_max_length=clipboard_max_length,
        # Context Learning Attention
        attention_foreground_gate=attention_foreground_gate,
        attention_noise_patterns=attention_noise_patterns,
        attention_goal_relevance_threshold=attention_goal_relevance_threshold,
        attention_working_dir_inference=attention_working_dir_inference,
        attention_domain_whitelist=attention_domain_whitelist,
        # Recording Mode
        recording_mode=recording_mode,
        # Causal Inference
        causal_reorder_window_ms=causal_reorder_window_ms,
        causal_burst_limit=causal_burst_limit,
        causal_max_chain_events=causal_max_chain_events,
        heuristic_time_decay_s=heuristic_time_decay_s,
        heuristic_space_decay_px=heuristic_space_decay_px,
        # World Model
        prediction_enabled=prediction_enabled,
        prediction_delta_threshold=prediction_delta_threshold,
        curiosity_alpha=curiosity_alpha,
        curiosity_beta=curiosity_beta,
        curiosity_gamma=curiosity_gamma,
        curiosity_auto_balance=curiosity_auto_balance,
        prediction_budget=prediction_budget,
        comparison_budget=comparison_budget,
        replay_budget=replay_budget,
        grading_budget=grading_budget,
        distillation_budget=distillation_budget,
        replay_on_session_end=replay_on_session_end,
        prediction_structural_blend=prediction_structural_blend,
        prediction_semantic_blend=prediction_semantic_blend,
        prediction_semantic_threshold=prediction_semantic_threshold,
        prediction_rag_advantage_floor=prediction_rag_advantage_floor,
        prediction_failure_advantage=prediction_failure_advantage,
        replay_regression_sample_size=replay_regression_sample_size,
        memory_prune_age_days=memory_prune_age_days,
        semantic_embedding_provider=semantic_embedding_provider,
        semantic_rerank_weight=semantic_rerank_weight,
        budget_discovery_baseline=budget_discovery_baseline,
        budget_regression_baseline=budget_regression_baseline,
        ast_precheck_enabled=ast_precheck_enabled,
        causal_window_s=causal_window_s,
        causal_app_window_overrides=causal_app_window_overrides,
        causal_tier3_enabled=causal_tier3_enabled,
        causal_tier3_confidence_threshold=causal_tier3_confidence_threshold,
        mhms_fusion_enabled=mhms_fusion_enabled,
        # Attention learning feedback
        attention_curiosity_expand_threshold=attention_curiosity_expand_threshold,
        attention_accuracy_contract_threshold=attention_accuracy_contract_threshold,
        # Surprise detection
        surprise_enabled=surprise_enabled,
        surprise_stat_weight=surprise_stat_weight,
        surprise_temporal_weight=surprise_temporal_weight,
        surprise_pattern_weight=surprise_pattern_weight,
        surprise_annotation_threshold=surprise_annotation_threshold,
        surprise_warmup_events=surprise_warmup_events,
        # Video Recording
        video_fps=video_fps,
        video_resolution_scale=video_resolution_scale,
        video_codec=video_codec,
        video_max_segment_s=video_max_segment_s,
        video_cache_dir=_expand_path(video_cache_dir),
        video_cache_max_age_days=video_cache_max_age_days,
        video_cache_max_size_gb=video_cache_max_size_gb,
        video_l2_enabled=video_l2_enabled,
        video_l3_enabled=video_l3_enabled,
        video_segmenter_min_s=video_segmenter_min_s,
        video_segmenter_max_s=video_segmenter_max_s,
        video_segmenter_idle_gap_s=video_segmenter_idle_gap_s,
        video_segmenter_app_gap_s=video_segmenter_app_gap_s,
        video_segmenter_min_split_s=video_segmenter_min_split_s,
        video_max_l2_requests=video_max_l2_requests,
        video_max_l3_requests=video_max_l3_requests,
        video_start_timeout_s=video_start_timeout_s,
        video_vlm_max_retries=video_vlm_max_retries,
        video_vlm_retry_backoff_s=video_vlm_retry_backoff_s,
        video_vlm_url_scheme=video_vlm_url_scheme,
        video_l2_time_window_s=video_l2_time_window_s,
        video_timeline_max_markers=video_timeline_max_markers,
        video_timeline_compress_max=video_timeline_compress_max,
        video_timeline_merge_channels=video_timeline_merge_channels,
        # Learnability Assessment
        learnability_enabled=learnability_enabled,
        learnability_min_steps=learnability_min_steps,
        learnability_min_duration_s=learnability_min_duration_s,
        learnability_max_idle_ratio=learnability_max_idle_ratio,
        learnability_min_action_diversity=learnability_min_action_diversity,
        learnability_learn_threshold=learnability_learn_threshold,
        learnability_ask_threshold=learnability_ask_threshold,
        learnability_vlm_enabled=learnability_vlm_enabled,
        learnability_llm_enabled=learnability_llm_enabled,
        learnability_rule_weight=learnability_rule_weight,
        learnability_vlm_weight=learnability_vlm_weight,
        learnability_llm_weight=learnability_llm_weight,
        # Execution Loop Budget
        react_max_iterations=react_max_iterations,
        react_soft_limit=react_soft_limit,
        react_warning_threshold=react_warning_threshold,
        tool_max_iterations=tool_max_iterations,
        native_tool_calling_enabled=native_tool_calling_enabled,
        stream_output=stream_output,
        verbose_progress=verbose_progress,
        # Context Compression
        compress_threshold=compress_threshold,
        compress_keep_tail=compress_keep_tail,
        max_tool_output_chars=max_tool_output_chars,
        max_tool_result_chars=max_tool_result_chars,
        context_hard_limit_ratio=context_hard_limit_ratio,
        context_warning_ratio=context_warning_ratio,
        tool_evidence_max_chars=tool_evidence_max_chars,
        repeated_read_limit=repeated_read_limit,
        long_task_convergence_round=long_task_convergence_round,
        context_expanded_ratio=context_expanded_ratio,
        context_finalizing_ratio=context_finalizing_ratio,
        context_expanded_evidence_threshold=context_expanded_evidence_threshold,
        context_expanded_tool_call_threshold=context_expanded_tool_call_threshold,
        context_research_source_threshold=context_research_source_threshold,
        context_research_evidence_threshold=context_research_evidence_threshold,
        # Error Recovery
        error_transient_max_retries=error_transient_max_retries,
        error_rate_limit_base_delay=error_rate_limit_base_delay,
        max_consecutive_tool_failures=max_consecutive_tool_failures,
        # Session Persistence
        session_persistence_enabled=session_persistence_enabled,
        # Multi-Provider LLM
        llm_fallback_providers=llm_fallback_providers,
        llm_aux_model=llm_aux_model,
        llm_aux_api_key=llm_aux_api_key,
        llm_aux_base_url=llm_aux_base_url,
        llm_context_length=llm_context_length,
        llm_credential_cooldown_s=llm_credential_cooldown_s,
        # Stream & Tool Robustness
        stale_stream_timeout_s=stale_stream_timeout_s,
        default_tool_timeout_s=default_tool_timeout_s,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_s=circuit_breaker_cooldown_s,
        # Signal Fusion
        signal_channels=signal_channels,
        signal_reactive_capture=signal_reactive_capture,
        # RPC Transport
        rpc_timeout_default=rpc_timeout_default,
        # Cua Driver
        use_cua_driver=use_cua_driver,
        cua_driver_cmd=cua_driver_cmd,
        # Workflow Copilot
        copilot_enabled=copilot_enabled,
        copilot_min_idle_ms=copilot_min_idle_ms,
        copilot_max_idle_ms=copilot_max_idle_ms,
        copilot_cache_ttl_s=copilot_cache_ttl_s,
        copilot_speculative_cache_size=copilot_speculative_cache_size,
        copilot_action_ring_size=copilot_action_ring_size,
        copilot_warmup_event_types=copilot_warmup_event_types,
        notification_renderer=notification_renderer,
        live_signal_kinds=live_signal_kinds,
        # Hub
        hub_type=hub_type,
        hub_default_owner=hub_default_owner,
        hub_default_visibility=hub_default_visibility,
        hub_sync_strategy=hub_sync_strategy,
        hub_sync_copilot=hub_sync_copilot,
        hub_repo_prefix=hub_repo_prefix,
        hub_search_sources=hub_search_sources,
        # Observer / Daemon
        observer_auto_start=observer_auto_start,
        observer_enabled_set=observer_enabled_set,
        # Scheduler
        scheduler_enabled=scheduler_enabled,
        scheduler_tick_seconds=scheduler_tick_seconds,
        scheduler_grace_seconds=scheduler_grace_seconds,
        scheduler_default_tier=scheduler_default_tier,
    )

    if not settings.llm_api_key:
        logger.warning("LEAPFLOW_LLM_API_KEY is empty; LLM calls will fail until configured.")

    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    for warning in validate_settings(settings):
        logger.warning("config: %s", warning)

    return settings


def validate_settings(settings: Settings) -> list[str]:
    """Post-load validation with actionable error messages.

    Returns a list of warning strings (empty = no issues).
    Does not raise — all issues are advisory.
    """
    warnings: list[str] = []

    if settings.llm_context_length < 1024:
        warnings.append(
            f"llm_context_length={settings.llm_context_length} is suspiciously low; "
            "expected at least 1024. Check LEAPFLOW_LLM_CONTEXT_LENGTH."
        )

    if settings.stale_stream_timeout_s < 10.0:
        warnings.append(
            f"stale_stream_timeout_s={settings.stale_stream_timeout_s} is very short; "
            "may cause premature stream cancellations."
        )

    if settings.default_tool_timeout_s < 5.0:
        warnings.append(
            f"default_tool_timeout_s={settings.default_tool_timeout_s} is very short; "
            "tools may timeout before completing."
        )

    if settings.react_soft_limit >= settings.react_max_iterations:
        warnings.append(
            f"react_soft_limit ({settings.react_soft_limit}) >= react_max_iterations "
            f"({settings.react_max_iterations}); soft limit will never trigger."
        )

    if settings.llm_aux_model and not settings.llm_aux_api_key and not settings.llm_api_key:
        warnings.append(
            "llm_aux_model is set but no API key available (neither aux nor primary). "
            "Auxiliary LLM calls will fail."
        )

    if settings.llm_fallback_providers:
        import json as _json
        try:
            _json.loads(settings.llm_fallback_providers)
        except _json.JSONDecodeError as e:
            warnings.append(
                f"LEAPFLOW_LLM_FALLBACK_PROVIDERS is not valid JSON: {e}. "
                "Fallback providers will be ignored."
            )

    return warnings


# ── Global settings accessor (lazy singleton) ──

_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings singleton, creating it on first access.

    Safe to call from any module; avoids circular-import issues when used
    inside constructors (deferred import pattern).
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = load_config()
    return _settings_instance
