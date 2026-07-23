"""Configuration loading from structured YAML with environment overrides.

Loading priority (highest wins):
    1. Process environment variables
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

DEFAULT_LLM_MODEL = "qwen3.7-plus"
DEFAULT_LLM_CONTEXT_LENGTH = 1_000_000
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


# Directory creation is owned by PathLayout/ProfileLayout.
def ensure_data_dir(data_dir: Path, *, profile: str = "default") -> None:
    """Create the LeapFlow data directory using the canonical layout."""
    build_layout(data_dir).ensure(profile_id=_validate_profile_name(profile))


def _bootstrap_data_dir() -> Path:
    return _expand_path(os.getenv("LEAPFLOW_DATA_DIR", "~/.leapflow").strip())


def _bootstrap_profile() -> str:
    return _validate_profile_name(os.getenv("LEAPFLOW_PROFILE", "default"))


def _bootstrap_layout() -> PathLayout:
    return build_layout(_bootstrap_data_dir())


def _bootstrap_profile_layout() -> ProfileLayout:
    return _bootstrap_layout().profile(_bootstrap_profile())


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
    data_dir: Path = field(default_factory=_bootstrap_data_dir)
    profile: str = field(default_factory=_bootstrap_profile)
    workspace_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    layout: PathLayout = field(default_factory=_bootstrap_layout)
    profile_layout: ProfileLayout = field(default_factory=_bootstrap_profile_layout)
    profile_manifest: ProfileManifest = field(
        default_factory=lambda: ProfileManifest.default(_bootstrap_profile())
    )
    config_sources: tuple[str, ...] = ()
    watched_config_paths: tuple[Path, ...] = ()
    config_warnings: tuple[str, ...] = ()
    runtime_dir: Path = field(default_factory=lambda: _bootstrap_profile_layout().runtime_dir)

    # Audit
    audit_log_path: Path = field(default_factory=lambda: _bootstrap_profile_layout().audit_log_path)

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
    skills_dir: Path = field(default_factory=lambda: _bootstrap_profile_layout().skills_dir)
    skill_view_max_chars: int = 5000  # Max chars returned by skill_view tool
    skill_min_quality: float = 0.5    # SkillIndex quality threshold

    # Visual Track
    visual_track_enabled: bool = False
    visual_frame_cache_dir: Path = field(
        default_factory=lambda: _bootstrap_profile_layout().cache.category_dir(
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
        default_factory=lambda: _bootstrap_profile_layout().cache.category_dir(
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
    # Adaptive-depth elastic budget: baseline floor -> difficulty-scaled ceiling.
    agent_iter_floor: int = 12
    agent_iter_ceiling: int = 200
    agent_budget_scale_k: float = 1.0
    # Progress-gated continuation: when a task is productively unfinished (open
    # ledger questions + still surfacing progress) and within resource limits,
    # the effective cap extends past the elastic ceiling toward this hard cap in
    # steps of ``agent_iter_extension_step``. Stall (no progress for
    # ``agent_stall_rounds`` rounds) stops extension. The hard cap is the true
    # backstop so a long task is bounded by progress/resources, not a fixed count.
    agent_iter_hard_cap: int = 500
    agent_iter_extension_step: int = 25
    agent_stall_rounds: int = 6
    agent_cost_ceiling_context_multiple: float = 0.0
    agent_subagent_max_depth: int = 2
    agent_subagent_max_concurrent: int = 3
    agent_subagent_max_iterations: int = 15
    agent_max_parallel_tools: int = 8  # Max tool calls run in parallel within one response's batch
    agent_subagent_full_loop: bool = False
    agent_calibration_enabled: bool = False
    agent_calibration_min_confidence: float = 0.3
    agent_calibration_interval_turns: int = 0
    agent_compression_writeback: bool = False
    agent_reentry_enabled: bool = False
    agent_reentry_tick_seconds: float = 30.0
    agent_reentry_global_budget: int = 100
    agent_reentry_send_enabled: bool = False
    agent_reentry_send_rate_per_hour: int = 4
    agent_reentry_send_global_budget: int = 50
    agent_reentry_send_verified_at: int = 3
    tool_max_iterations: int = 30
    native_tool_calling_enabled: bool = True   # Use native OpenAI tool_calls when available
    stream_output: bool = True                   # Enable LLM streaming in interactive mode
    verbose_progress: bool = True                # Show detailed tool execution progress

    # Context Compression
    compress_threshold: int = 16
    compress_keep_tail: int = 4
    max_tool_output_chars: int = 3000
    max_tool_result_chars: int = 3000  # Per-tool result truncation for LLM context
    # code_search: best-effort seamless ripgrep auto-install (macOS/Homebrew, no
    # sudo) when missing; always falls back to the pure-Python search + a manual
    # install hint, so search works with zero install regardless.
    tools_ripgrep_autoinstall: bool = True
    tools_test_command: str = ""  # empty => auto-detect (pytest/npm/go/cargo)
    tools_lint_command: str = ""  # empty => auto-detect (ruff/eslint/go vet/clippy)
    tools_terminal_session_enabled: bool = False  # persistent shell sessions (opt-in, high risk)
    tools_verify_edits: bool = True  # post-edit syntax check (advisory) for edit_file/file_write
    agent_validate_tool_args: bool = True  # pre-execution required-argument validation + self-repair
    context_hard_limit_ratio: float = 0.92
    context_warning_ratio: float = 0.75
    tool_evidence_max_chars: int = 1200
    repeated_read_limit: int = 2
    long_task_convergence_round: int = 12
    # Adaptive convergence ceiling: on high-difficulty tasks the effective
    # convergence round scales up (convergence_round * (1 + difficulty *
    # convergence_scale)), bounded by this ceiling so genuinely stuck tasks
    # still eventually converge.
    convergence_round_ceiling: int = 40
    convergence_scale: float = 2.0
    # Shell safety ceiling: the internal shell-run process timeout is clamped
    # here so individual shell calls can safely run beyond 2 minutes.
    max_shell_timeout_s: float = 300.0
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
    # Per-turn recovery budget (bounds recovery attempts within one agent turn).
    # A non-positive deadline means unlimited wall-clock time so a long-running
    # task is never denied recovery for a late transient error; the action-count
    # budget remains the real bound and scales for long tasks.
    recovery_turn_deadline_s: float = 0.0
    recovery_total_actions: int = 24
    recovery_max_retry_per_category: int = 4

    # ── Tool-loop Guardrails ──
    # Progress-aware loop guards (repetition / stagnation / single-tool
    # domination). Halts and finalize nudges are suppressed while the task is
    # still making progress, so legitimate batch/sequential work on a long task
    # is not cut short. Thresholds are configurable; the guard can be disabled.
    guardrail_enabled: bool = True
    guardrail_max_repeats: int = 3
    guardrail_max_consecutive_same: int = 8
    guardrail_stagnation_window: int = 10
    guardrail_min_success_rate: float = 0.2

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
    daemon_request_ledger_ttl_s: float = 600.0  # Replay cache retention for completed engine requests
    daemon_request_ledger_max_entries: int = 128  # Maximum completed engine requests kept for replay
    # Concurrent turn execution (Stage 3). N=3 lets several fresh TUI sessions
    # run concurrently by default on isolated per-session engines (turns within
    # one session stay serialized). Set to 1 for strict serialized fallback.
    daemon_max_concurrent_turns: int = 3
    daemon_max_live_sessions: int = 16
    daemon_session_idle_ttl_s: float = 1800.0
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

    # ── Dashboard (monitoring web view) ──
    dashboard_enabled: bool = True
    dashboard_bind: str = "127.0.0.1"
    dashboard_port: int = 8765
    dashboard_auto_open: bool = True
    dashboard_token_ref: str = ""  # secret ref for the local dashboard access token

    # ── Session analysis dashboard (domain=session watch) ──
    monitor_session_batch_turns: int = 6
    monitor_session_batch_tokens: int = 4000
    monitor_session_use_model_salience: bool = False
    monitor_session_debounce_s: float = 15.0
    monitor_session_max_refresh_per_min: int = 4

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


def load_config() -> Settings:
    """Load settings from structured YAML and process environment overrides."""
    data_dir = _expand_path(os.getenv("LEAPFLOW_DATA_DIR", "~/.leapflow").strip())
    profile = _validate_profile_name(os.getenv("LEAPFLOW_PROFILE", "default"))
    workspace_root = _expand_path(
        os.getenv("LEAPFLOW_WORKSPACE_ROOT", str(Path.cwd())).strip() or str(Path.cwd())
    ).resolve()
    layout = build_layout(data_dir)
    try:
        from leapflow.security.path_sensitivity import configure_path_sensitivity_roots
        configure_path_sensitivity_roots((layout.root,))
    except Exception:
        logger.debug("Path sensitivity root configuration skipped", exc_info=True)
    profile_layout = layout.ensure(profile_id=profile)
    workspace_id = workspace_id_for_path(workspace_root)
    try:
        layout.write_workspace_manifest(workspace_root)
        profile_layout.cache.write_workspace_manifest(workspace_id, workspace_root)
    except OSError:
        logger.debug("Workspace manifest write skipped", exc_info=True)
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
            config_warnings=bundle.warnings,
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
    config_warnings: tuple[str, ...] = (),
) -> Settings:
    """Build Settings from current os.environ."""
    api_key = os.getenv("LEAPFLOW_LLM_API_KEY", "").strip()
    base_url = os.getenv(
        "LEAPFLOW_LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    model = os.getenv("LEAPFLOW_LLM_MODEL", DEFAULT_LLM_MODEL).strip()
    max_retries = int(os.getenv("LEAPFLOW_LLM_MAX_RETRIES", "3"))

    data_dir = _expand_path(os.getenv("LEAPFLOW_DATA_DIR", "~/.leapflow").strip())
    profile = _validate_profile_name(os.getenv("LEAPFLOW_PROFILE", "default"))
    workspace_root = _expand_path(
        os.getenv("LEAPFLOW_WORKSPACE_ROOT", str(Path.cwd())).strip() or str(Path.cwd())
    ).resolve()
    layout = layout or build_layout(data_dir)
    try:
        from leapflow.security.path_sensitivity import configure_path_sensitivity_roots
        configure_path_sensitivity_roots((layout.root,))
    except Exception:
        logger.debug("Path sensitivity root configuration skipped", exc_info=True)
    profile_layout = profile_layout or layout.profile(profile)
    profile_manifest = profile_manifest or profile_layout.load_manifest()
    workspace_id = workspace_id_for_path(workspace_root)
    try:
        layout.write_workspace_manifest(workspace_root)
        profile_layout.cache.write_workspace_manifest(workspace_id, workspace_root)
    except OSError:
        logger.debug("Workspace manifest write skipped", exc_info=True)
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
    agent_iter_floor = int(os.getenv("LEAPFLOW_AGENT_ITER_FLOOR", "12"))
    agent_iter_ceiling = int(os.getenv("LEAPFLOW_AGENT_ITER_CEILING", "200"))
    agent_budget_scale_k = float(os.getenv("LEAPFLOW_AGENT_BUDGET_SCALE_K", "1.0"))
    agent_iter_hard_cap = int(os.getenv("LEAPFLOW_AGENT_ITER_HARD_CAP", "500"))
    agent_iter_extension_step = int(os.getenv("LEAPFLOW_AGENT_ITER_EXTENSION_STEP", "25"))
    agent_stall_rounds = int(os.getenv("LEAPFLOW_AGENT_STALL_ROUNDS", "6"))
    agent_cost_ceiling_context_multiple = float(os.getenv("LEAPFLOW_AGENT_COST_CEILING_CONTEXT_MULTIPLE", "0.0"))
    agent_subagent_max_depth = int(os.getenv("LEAPFLOW_AGENT_SUBAGENT_MAX_DEPTH", "2"))
    agent_subagent_max_concurrent = int(os.getenv("LEAPFLOW_AGENT_SUBAGENT_MAX_CONCURRENT", "3"))
    agent_subagent_max_iterations = int(os.getenv("LEAPFLOW_AGENT_SUBAGENT_MAX_ITERATIONS", "15"))
    agent_max_parallel_tools = int(os.getenv("LEAPFLOW_AGENT_MAX_PARALLEL_TOOLS", "8"))
    agent_subagent_full_loop = os.getenv("LEAPFLOW_AGENT_SUBAGENT_FULL_LOOP", "0").strip().lower() in ("1", "true", "yes")
    agent_calibration_enabled = os.getenv("LEAPFLOW_AGENT_CALIBRATION_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    agent_calibration_min_confidence = float(os.getenv("LEAPFLOW_AGENT_CALIBRATION_MIN_CONFIDENCE", "0.3"))
    agent_calibration_interval_turns = int(os.getenv("LEAPFLOW_AGENT_CALIBRATION_INTERVAL_TURNS", "0"))
    agent_compression_writeback = os.getenv("LEAPFLOW_AGENT_COMPRESSION_WRITEBACK", "0").strip().lower() in ("1", "true", "yes")
    agent_reentry_enabled = os.getenv("LEAPFLOW_AGENT_REENTRY_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    agent_reentry_tick_seconds = float(os.getenv("LEAPFLOW_AGENT_REENTRY_TICK_SECONDS", "30"))
    agent_reentry_global_budget = int(os.getenv("LEAPFLOW_AGENT_REENTRY_GLOBAL_BUDGET", "100"))
    agent_reentry_send_enabled = os.getenv("LEAPFLOW_AGENT_REENTRY_SEND_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    agent_reentry_send_rate_per_hour = int(os.getenv("LEAPFLOW_AGENT_REENTRY_SEND_RATE_PER_HOUR", "4"))
    agent_reentry_send_global_budget = int(os.getenv("LEAPFLOW_AGENT_REENTRY_SEND_GLOBAL_BUDGET", "50"))
    agent_reentry_send_verified_at = int(os.getenv("LEAPFLOW_AGENT_REENTRY_SEND_VERIFIED_AT", "3"))
    tool_max_iterations = int(os.getenv("LEAPFLOW_TOOL_MAX_ITERATIONS", "30"))
    native_tool_calling_enabled = os.getenv("LEAPFLOW_NATIVE_TOOL_CALLING_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    stream_output = os.getenv("LEAPFLOW_STREAM_OUTPUT", "1").strip().lower() in ("1", "true", "yes")
    verbose_progress = os.getenv("LEAPFLOW_VERBOSE_PROGRESS", "1").strip().lower() in ("1", "true", "yes")

    # Context Compression
    compress_threshold = int(os.getenv("LEAPFLOW_COMPRESS_THRESHOLD", "16"))
    compress_keep_tail = int(os.getenv("LEAPFLOW_COMPRESS_KEEP_TAIL", "4"))
    max_tool_output_chars = int(os.getenv("LEAPFLOW_MAX_TOOL_OUTPUT_CHARS", "3000"))
    max_tool_result_chars = int(os.getenv("LEAPFLOW_MAX_TOOL_RESULT_CHARS", "3000"))
    tools_ripgrep_autoinstall = os.getenv("LEAPFLOW_TOOLS_RIPGREP_AUTOINSTALL", "1").strip().lower() in ("1", "true", "yes")
    tools_test_command = os.getenv("LEAPFLOW_TOOLS_TEST_COMMAND", "").strip()
    tools_lint_command = os.getenv("LEAPFLOW_TOOLS_LINT_COMMAND", "").strip()
    tools_terminal_session_enabled = os.getenv("LEAPFLOW_TOOLS_TERMINAL_SESSION_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    tools_verify_edits = os.getenv("LEAPFLOW_TOOLS_VERIFY_EDITS", "1").strip().lower() in ("1", "true", "yes")
    agent_validate_tool_args = os.getenv("LEAPFLOW_AGENT_VALIDATE_TOOL_ARGS", "1").strip().lower() in ("1", "true", "yes")
    context_hard_limit_ratio = float(os.getenv("LEAPFLOW_CONTEXT_HARD_LIMIT_RATIO", "0.92"))
    context_warning_ratio = float(os.getenv("LEAPFLOW_CONTEXT_WARNING_RATIO", "0.75"))
    tool_evidence_max_chars = int(os.getenv("LEAPFLOW_TOOL_EVIDENCE_MAX_CHARS", "1200"))
    repeated_read_limit = int(os.getenv("LEAPFLOW_REPEATED_READ_LIMIT", "2"))
    long_task_convergence_round = int(os.getenv("LEAPFLOW_LONG_TASK_CONVERGENCE_ROUND", "12"))
    convergence_round_ceiling = int(os.getenv("LEAPFLOW_CONVERGENCE_ROUND_CEILING", "40"))
    convergence_scale = float(os.getenv("LEAPFLOW_CONVERGENCE_SCALE", "2.0"))
    max_shell_timeout_s = float(os.getenv("LEAPFLOW_MAX_SHELL_TIMEOUT_S", "300.0"))
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
    recovery_turn_deadline_s = float(os.getenv("LEAPFLOW_RECOVERY_TURN_DEADLINE_S", "0"))
    recovery_total_actions = int(os.getenv("LEAPFLOW_RECOVERY_TOTAL_ACTIONS", "24"))
    recovery_max_retry_per_category = int(os.getenv("LEAPFLOW_RECOVERY_MAX_RETRY_PER_CATEGORY", "4"))
    guardrail_enabled = os.getenv("LEAPFLOW_GUARDRAIL_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    guardrail_max_repeats = int(os.getenv("LEAPFLOW_GUARDRAIL_MAX_REPEATS", "3"))
    guardrail_max_consecutive_same = int(os.getenv("LEAPFLOW_GUARDRAIL_MAX_CONSECUTIVE_SAME", "8"))
    guardrail_stagnation_window = int(os.getenv("LEAPFLOW_GUARDRAIL_STAGNATION_WINDOW", "10"))
    guardrail_min_success_rate = float(os.getenv("LEAPFLOW_GUARDRAIL_MIN_SUCCESS_RATE", "0.2"))

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
    daemon_request_ledger_ttl_s = float(os.getenv("LEAPFLOW_DAEMON_REQUEST_LEDGER_TTL_S", "600.0"))
    daemon_request_ledger_max_entries = int(os.getenv("LEAPFLOW_DAEMON_REQUEST_LEDGER_MAX_ENTRIES", "128"))
    daemon_max_concurrent_turns = int(os.getenv("LEAPFLOW_DAEMON_MAX_CONCURRENT_TURNS", "3"))
    daemon_max_live_sessions = int(os.getenv("LEAPFLOW_DAEMON_MAX_LIVE_SESSIONS", "16"))
    daemon_session_idle_ttl_s = float(os.getenv("LEAPFLOW_DAEMON_SESSION_IDLE_TTL_S", "1800.0"))
    circuit_breaker_threshold = int(os.getenv("LEAPFLOW_CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_breaker_cooldown_s = float(os.getenv("LEAPFLOW_CIRCUIT_BREAKER_COOLDOWN_S", "60.0"))

    # Signal Fusion
    # Default = "all": collect every supported channel (V7 full fusion). Set
    # to "none" or empty list to disable; comma-separated list selects a
    # specific subset for ablation experiments.
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

    # Dashboard
    dashboard_enabled = _bool("LEAPFLOW_DASHBOARD_ENABLED", "true")
    dashboard_bind = os.getenv("LEAPFLOW_DASHBOARD_BIND", "127.0.0.1").strip() or "127.0.0.1"
    dashboard_port = int(os.getenv("LEAPFLOW_DASHBOARD_PORT", "8765"))
    dashboard_auto_open = _bool("LEAPFLOW_DASHBOARD_AUTO_OPEN", "true")
    dashboard_token_ref = os.getenv("LEAPFLOW_DASHBOARD_TOKEN_REF", "").strip()

    # Session analysis dashboard
    monitor_session_batch_turns = int(os.getenv("LEAPFLOW_MONITOR_SESSION_BATCH_TURNS", "6"))
    monitor_session_batch_tokens = int(os.getenv("LEAPFLOW_MONITOR_SESSION_BATCH_TOKENS", "4000"))
    monitor_session_use_model_salience = _bool("LEAPFLOW_MONITOR_SESSION_USE_MODEL_SALIENCE", "false")
    monitor_session_debounce_s = float(os.getenv("LEAPFLOW_MONITOR_SESSION_DEBOUNCE_S", "15.0"))
    monitor_session_max_refresh_per_min = int(os.getenv("LEAPFLOW_MONITOR_SESSION_MAX_REFRESH_PER_MIN", "4"))

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
        config_warnings=config_warnings,
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
        agent_iter_floor=agent_iter_floor,
        agent_iter_ceiling=agent_iter_ceiling,
        agent_budget_scale_k=agent_budget_scale_k,
        agent_iter_hard_cap=agent_iter_hard_cap,
        agent_iter_extension_step=agent_iter_extension_step,
        agent_stall_rounds=agent_stall_rounds,
        agent_cost_ceiling_context_multiple=agent_cost_ceiling_context_multiple,
        agent_subagent_max_depth=agent_subagent_max_depth,
        agent_subagent_max_concurrent=agent_subagent_max_concurrent,
        agent_subagent_max_iterations=agent_subagent_max_iterations,
        agent_max_parallel_tools=agent_max_parallel_tools,
        agent_subagent_full_loop=agent_subagent_full_loop,
        agent_calibration_enabled=agent_calibration_enabled,
        agent_calibration_min_confidence=agent_calibration_min_confidence,
        agent_calibration_interval_turns=agent_calibration_interval_turns,
        agent_compression_writeback=agent_compression_writeback,
        agent_reentry_enabled=agent_reentry_enabled,
        agent_reentry_tick_seconds=agent_reentry_tick_seconds,
        agent_reentry_global_budget=agent_reentry_global_budget,
        agent_reentry_send_enabled=agent_reentry_send_enabled,
        agent_reentry_send_rate_per_hour=agent_reentry_send_rate_per_hour,
        agent_reentry_send_global_budget=agent_reentry_send_global_budget,
        agent_reentry_send_verified_at=agent_reentry_send_verified_at,
        tool_max_iterations=tool_max_iterations,
        native_tool_calling_enabled=native_tool_calling_enabled,
        stream_output=stream_output,
        verbose_progress=verbose_progress,
        # Context Compression
        compress_threshold=compress_threshold,
        compress_keep_tail=compress_keep_tail,
        max_tool_output_chars=max_tool_output_chars,
        max_tool_result_chars=max_tool_result_chars,
        tools_ripgrep_autoinstall=tools_ripgrep_autoinstall,
        tools_test_command=tools_test_command,
        tools_lint_command=tools_lint_command,
        tools_terminal_session_enabled=tools_terminal_session_enabled,
        tools_verify_edits=tools_verify_edits,
        agent_validate_tool_args=agent_validate_tool_args,
        context_hard_limit_ratio=context_hard_limit_ratio,
        context_warning_ratio=context_warning_ratio,
        tool_evidence_max_chars=tool_evidence_max_chars,
        repeated_read_limit=repeated_read_limit,
        long_task_convergence_round=long_task_convergence_round,
        convergence_round_ceiling=convergence_round_ceiling,
        convergence_scale=convergence_scale,
        max_shell_timeout_s=max_shell_timeout_s,
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
        recovery_turn_deadline_s=recovery_turn_deadline_s,
        recovery_total_actions=recovery_total_actions,
        recovery_max_retry_per_category=recovery_max_retry_per_category,
        guardrail_enabled=guardrail_enabled,
        guardrail_max_repeats=guardrail_max_repeats,
        guardrail_max_consecutive_same=guardrail_max_consecutive_same,
        guardrail_stagnation_window=guardrail_stagnation_window,
        guardrail_min_success_rate=guardrail_min_success_rate,
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
        daemon_request_ledger_ttl_s=daemon_request_ledger_ttl_s,
        daemon_request_ledger_max_entries=daemon_request_ledger_max_entries,
        daemon_max_concurrent_turns=daemon_max_concurrent_turns,
        daemon_max_live_sessions=daemon_max_live_sessions,
        daemon_session_idle_ttl_s=daemon_session_idle_ttl_s,
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
        # Dashboard
        dashboard_enabled=dashboard_enabled,
        dashboard_bind=dashboard_bind,
        dashboard_port=dashboard_port,
        dashboard_auto_open=dashboard_auto_open,
        dashboard_token_ref=dashboard_token_ref,
        monitor_session_batch_turns=monitor_session_batch_turns,
        monitor_session_batch_tokens=monitor_session_batch_tokens,
        monitor_session_use_model_salience=monitor_session_use_model_salience,
        monitor_session_debounce_s=monitor_session_debounce_s,
        monitor_session_max_refresh_per_min=monitor_session_max_refresh_per_min,
    )

    if not settings.llm_api_key:
        logger.warning("LLM API key is empty; run `leap config llm key` before making LLM calls.")

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
