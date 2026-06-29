"""Context Learning Attention Mechanism — signal/noise filtering for demonstration recording.

Implements layered attention filters that improve the signal-to-noise ratio
of recorded trajectories. Filters are composable and configurable.

Priority levels:
    P0:   ForegroundGateFilter — only record events from user's active apps
    P0.5: DomainWhitelistFilter — goal-driven domain-scoped whitelist
    P1:   GoalRelevanceFilter — post-hoc relevance scoring against user's goal
    P2:   NoiseRuleFilter — configurable source/path patterns for known noise
    P3:   WorkingDirFilter — infer working directory, filter unrelated FS events
"""

from __future__ import annotations

import os
import os.path
import re
import time as _time_mod
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from leapflow.domain.events import SystemEvent

_DEFAULT_NOISE_PATTERNS: Tuple[str, ...] = (
    r"/\.sogouinput/",
    r"/\.openclaw-bundle/",
    r"/\.Trash/",
    r"/\.DS_Store$",
    r"/\.localized$",
    r"/Library/Caches/",
    r"/Library/Saved Application State/",
    r"/\.CFUserTextEncoding$",
    r"/\.git/objects/",
)

_HARD_REJECT_NOISE_PATTERNS: Tuple[str, ...] = (
    r"/\.openclaw-bundle/auto_update\.lock$",
    r"/\.sogouinput/",
    r"\.swp$",
    r"~$",
    r"/\.git/objects/",
    r"/\.git/refs/",
    r"/\.git/logs/",
    r"/\.Trash/",
    r"/__pycache__/",
    r"/\.DS_Store$",
    r"/\.localized$",
    r"/\.CFUserTextEncoding$",
    r"/node_modules/",
    r"/Library/Caches/",
    r"/Library/Saved Application State/",
    r"/Library/Preferences/ByHost/",
    r"/Library/Cookies/",
    r"/\.cache/",
)

# ── Goal seeding helpers ──

_GOAL_APP_HINTS: Dict[str, str] = {
    "finder": "com.apple.finder",
    "preview": "com.apple.Preview",
    "terminal": "com.apple.Terminal",
    "safari": "com.apple.Safari",
    "chrome": "com.google.Chrome",
    "firefox": "org.mozilla.firefox",
    "vscode": "com.microsoft.VSCode",
    "code": "com.microsoft.VSCode",
    "notes": "com.apple.Notes",
    "pages": "com.apple.iWork.Pages",
    "numbers": "com.apple.iWork.Numbers",
    "keynote": "com.apple.iWork.Keynote",
    "textedit": "com.apple.TextEdit",
}

_KNOWN_EXTENSIONS: FrozenSet[str] = frozenset({
    ".pdf", ".txt", ".md", ".doc", ".docx", ".xlsx", ".csv",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml",
    ".zip", ".tar", ".gz", ".dmg", ".pkg",
    ".mp3", ".mp4", ".mov", ".avi",
})

_EXTENSION_PATTERN = re.compile(r"\.([a-zA-Z]{2,5})\b")
_BARE_EXTENSION_PATTERN = re.compile(r"\b([a-zA-Z]{2,5})\b")
_PATH_PATTERN = re.compile(r"[~/][^\s,;:\"']+")

# Platform-specific system bundles (never user-relevant)
_PLATFORM_SYSTEM_APPS: Dict[str, FrozenSet[str]] = {
    "darwin": frozenset({
        "com.apple.SecurityAgent",
        "com.apple.universalaccessd",
        "com.apple.UserNotificationCenter",
        "com.apple.inputmethod.SCIM",
        "com.apple.inputmethod.TCIM",
        "com.apple.TextInputMenuAgent",
        "com.apple.TextInputSwitcher",
        "com.apple.Spotlight",
        "com.apple.notificationcenterui",
        "com.sogou.inputmethod.sogou",
        "com.apple.loginwindow",
    }),
    "linux": frozenset({
        "org.gnome.Shell",
        "org.kde.plasmashell",
        "org.freedesktop.IBus",
        "org.fcitx.Fcitx5",
    }),
}

if TYPE_CHECKING:
    from leapflow.domain.trajectory import Trajectory, TrajectoryStep


class FilterVerdict(Enum):
    """Outcome of an attention filter evaluation."""

    ACCEPT = "accept"
    ANNOTATE_NOISE = "annotate_noise"
    REJECT = "reject"


@dataclass(frozen=True)
class FilterResult:
    """Result from a single attention filter evaluation."""

    verdict: FilterVerdict
    reason: str = ""
    confidence: float = 1.0


_USER_COMMON_DIRS: Tuple[str, ...] = (
    "~/Documents", "~/Desktop", "~/Downloads",
    "~/Pictures", "~/Music", "~/Movies",
)

_FINDER_BUNDLES = frozenset({
    "com.apple.finder",
    "org.gnome.Nautilus",
    "org.kde.dolphin",
})

_WINDOW_MGMT_UI_ACTIONS = frozenset({"move", "resize", "minimize", "maximize", "zoom"})


@dataclass
class RecordingContext:
    """Shared mutable context maintained across the recording session."""

    focused_apps: Set[str] = field(default_factory=set)
    current_focused_app: str = ""
    goal: str = ""
    working_dirs: Set[str] = field(default_factory=set)
    self_host_app: str = ""
    _fs_paths_seen: List[str] = field(default_factory=list)
    _working_dir_inferred: bool = False

    # Domain-scoped whitelist state
    fs_scope: Set[str] = field(default_factory=set)
    app_scope: Set[str] = field(default_factory=set)
    goal_keywords: Set[str] = field(default_factory=set)
    goal_extensions: Set[str] = field(default_factory=set)
    _bootstrap_count: int = 0
    _bootstrap_limit: int = 5

    # App context
    last_window_title: str = ""

    # Transient visit tracking: app_bundle_id → interaction count in current visit
    _app_visit_counts: Dict[str, int] = field(default_factory=dict)
    _app_visit_threshold: int = 2

    # Perceptual field context tracking (behavioral inference signals)
    current_context: "Optional[Any]" = None
    context_history: List["Any"] = field(default_factory=list)
    _context_interaction_counts: Dict[str, int] = field(default_factory=dict)
    _context_dwell_start: Dict[str, float] = field(default_factory=dict)

    @property
    def bootstrap_complete(self) -> bool:
        return self._bootstrap_count >= self._bootstrap_limit

    def update_focus(self, bundle_id: str) -> None:
        """Track an app focus event."""
        self.current_focused_app = bundle_id
        if bundle_id and bundle_id != self.self_host_app:
            self.focused_apps.add(bundle_id)

    def record_app_interaction(self, bundle_id: str) -> None:
        """Record a UI interaction in the given app (for transient visit detection)."""
        if bundle_id:
            self._app_visit_counts[bundle_id] = self._app_visit_counts.get(bundle_id, 0) + 1

    def is_transient_visit(self, bundle_id: str) -> bool:
        """Check if app has had fewer than threshold interactions (likely a glance)."""
        return self._app_visit_counts.get(bundle_id, 0) < self._app_visit_threshold

    _MAX_CONTEXT_HISTORY = 500

    def update_context(self, ctx: "Any", timestamp: float) -> None:
        """Update the current perceptual field context (driven by ContextExtractor)."""
        self.current_context = ctx
        key = f"{ctx.app_bundle_id}::{ctx.context_value}"
        if key not in self._context_dwell_start:
            self._context_dwell_start[key] = timestamp
        if len(self.context_history) < self._MAX_CONTEXT_HISTORY:
            self.context_history.append((timestamp, ctx))

    def record_context_interaction(self, ctx: "Any") -> None:
        """Record a substantive interaction within a context (learning signal)."""
        key = f"{ctx.app_bundle_id}::{ctx.context_value}"
        self._context_interaction_counts[key] = self._context_interaction_counts.get(key, 0) + 1

    def is_transient_context(self, ctx: "Any", threshold: int = 2) -> bool:
        """Check if a context has had fewer than threshold interactions."""
        key = f"{ctx.app_bundle_id}::{ctx.context_value}"
        return self._context_interaction_counts.get(key, 0) < threshold

    def observe_fs_path(self, path: str) -> None:
        """Feed a filesystem path for working directory inference."""
        if self._working_dir_inferred or not path:
            return
        self._fs_paths_seen.append(path)
        if len(self._fs_paths_seen) >= 3:
            self._infer_working_dir()

    def _infer_working_dir(self) -> None:
        """Infer working directory from the common prefix of observed paths."""
        dirs = [os.path.dirname(p) for p in self._fs_paths_seen if p.startswith("/")]
        if not dirs:
            return
        prefix = os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0]
        if prefix and prefix != "/":
            self.working_dirs.add(prefix)
            self._working_dir_inferred = True

    def seed_from_goal(self, goal: str) -> None:
        """Extract scope hints from the user's stated goal (lightweight, no LLM)."""
        if not goal:
            return
        self.goal = goal
        lower = goal.lower()

        for match in _EXTENSION_PATTERN.finditer(lower):
            ext = "." + match.group(1)
            if ext in _KNOWN_EXTENSIONS:
                self.goal_extensions.add(ext)

        for match in _BARE_EXTENSION_PATTERN.finditer(lower):
            candidate = "." + match.group(1)
            if candidate in _KNOWN_EXTENSIONS:
                self.goal_extensions.add(candidate)

        for match in _PATH_PATTERN.finditer(goal):
            path = os.path.expanduser(match.group().rstrip(".,;:)]}"))
            if path and len(path) > 2:
                self.fs_scope.add(path)

        for keyword, bundle in _GOAL_APP_HINTS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
                self.app_scope.add(bundle)

        self.goal_keywords = {w for w in re.split(r"[^a-z0-9]+", lower) if len(w) > 2}

        # Auto-add common user directories when goal mentions file operations
        file_action_keywords = {"organize", "move", "copy", "sort", "clean", "file", "folder", "dir"}
        if file_action_keywords & self.goal_keywords:
            home = os.path.expanduser("~")
            for d in _USER_COMMON_DIRS:
                expanded = os.path.expanduser(d)
                if os.path.isdir(expanded):
                    self.fs_scope.add(expanded)
            # Also add home directory itself as scope
            self.fs_scope.add(home)

    def observe_for_bootstrap(self, event: "SystemEvent") -> None:
        """Feed events during bootstrap phase to build initial scope."""
        if self.bootstrap_complete:
            return
        self._bootstrap_count += 1

        if event.event_type == "fs.change":
            path = str(event.payload.get("path", event.source))
            if path and path.startswith("/"):
                directory = os.path.dirname(path)
                if directory and directory != "/":
                    self.fs_scope.add(directory)
        elif event.event_type == "app.focus_change":
            bundle_id = str(event.payload.get("bundle_id", event.source))
            if bundle_id:
                self.app_scope.add(bundle_id)

    def observe_ui_for_fs_scope(self, event: "SystemEvent") -> None:
        """Domain inter-cooperation: UI events inform FS scope.

        When a user clicks or navigates in a file manager (Finder, Nautilus),
        the target may reveal which directory they're working in. This expands
        the FS whitelist based on cross-domain inference.
        """
        app = str(event.payload.get("app_bundle_id", ""))
        if app not in _FINDER_BUNDLES:
            return

        label = str(event.payload.get("label", ""))
        target = str(event.payload.get("target", ""))
        candidate = label or target

        if candidate and ("/" in candidate or candidate.startswith("~")):
            path = os.path.expanduser(candidate)
            if path.startswith("/"):
                self.expand_fs_scope(path)

    def expand_fs_scope(self, path: str) -> None:
        """Add directory to FS whitelist (from user's explicit action)."""
        if not path:
            return
        directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
        if directory and directory != "/":
            self.fs_scope.add(directory)

    def expand_app_scope(self, bundle_id: str) -> None:
        """Add app to whitelist (user explicitly switched to it)."""
        if bundle_id:
            self.app_scope.add(bundle_id)

    def reset(self) -> None:
        """Reset context for a new recording session."""
        self.focused_apps.clear()
        self.current_focused_app = ""
        self.working_dirs.clear()
        self._fs_paths_seen.clear()
        self._working_dir_inferred = False
        self.fs_scope.clear()
        self.app_scope.clear()
        self.goal_keywords.clear()
        self.goal_extensions.clear()
        self._bootstrap_count = 0
        self._app_visit_counts.clear()
        self.current_context = None
        self.context_history.clear()
        self._context_interaction_counts.clear()
        self._context_dwell_start.clear()


# ── Filter Protocol ──


class AttentionFilter(ABC):
    """Base class for recording-time attention filters."""

    @abstractmethod
    def evaluate(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        """Evaluate whether an event should be recorded."""


# ── P0: Foreground App Gating ──


class ForegroundGateFilter(AttentionFilter):
    """Only accept events from apps the user has actively focused.

    Events from apps the user has never switched to during this session
    are annotated as noise — they likely come from background processes.
    app.focus_change events are always accepted (they build the focused set).
    """

    def evaluate(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        if event.event_type == "app.focus_change":
            return FilterResult(FilterVerdict.ACCEPT)

        event_app = self._extract_app(event, context)
        if not event_app:
            return FilterResult(FilterVerdict.ACCEPT)

        if event_app in context.focused_apps:
            return FilterResult(FilterVerdict.ACCEPT)

        return FilterResult(
            FilterVerdict.ANNOTATE_NOISE,
            reason=f"app_not_focused:{event_app}",
            confidence=0.85,
        )

    @staticmethod
    def _extract_app(event: SystemEvent, context: RecordingContext) -> str:
        """Extract the app bundle_id associated with this event."""
        if event.event_type == "ui.action":
            return str(event.payload.get("app_bundle_id", "")) or context.current_focused_app
        if event.event_type == "app.focus_change":
            return str(event.payload.get("bundle_id", event.source))
        return context.current_focused_app


# ── P0.5: Domain Whitelist Filter ──


_FS_HARD_REJECT_DARWIN: Tuple[re.Pattern[str], ...] = (
    re.compile(r"/Library/"),
    re.compile(r"\.app/Contents/"),
)

_FS_HARD_REJECT_LINUX: Tuple[re.Pattern[str], ...] = (
    re.compile(r"/proc/"),
    re.compile(r"/sys/"),
    re.compile(r"/run/"),
    re.compile(r"/snap/[^/]+/"),
)

_FS_HARD_REJECT_BY_PLATFORM: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "darwin": _FS_HARD_REJECT_DARWIN,
    "linux": _FS_HARD_REJECT_LINUX,
}


class DomainWhitelistFilter(AttentionFilter):
    """Goal-driven whitelist filter with per-domain scope inference.

    Three-layer decision per event:
      1. Whitelist check → ACCEPT if event matches domain scope
      2. Hard blacklist → REJECT if matches known system noise
      3. Unknown zone → ANNOTATE_NOISE (preserve but downweight)

    Position: P0.5 (after ForegroundGateFilter, before NoiseRuleFilter).
    Cross-platform: system bundle set and FS hard reject patterns are
    selected by platform_hint.
    Domain inter-cooperation: UI events in file managers expand FS scope.
    """

    def __init__(self, *, goal: str = "", platform_hint: str = "darwin") -> None:
        self._goal = goal
        platform_key = platform_hint.split("_")[0] if platform_hint else "darwin"
        self._system_bundles = _PLATFORM_SYSTEM_APPS.get(platform_key, frozenset())
        self._fs_hard_reject = _FS_HARD_REJECT_BY_PLATFORM.get(platform_key, ())

    def evaluate(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        if not context.bootstrap_complete:
            context.observe_for_bootstrap(event)
            return FilterResult(FilterVerdict.ACCEPT)

        if event.event_type == "fs.change":
            return self._evaluate_fs(event, context)
        elif event.event_type == "app.focus_change":
            return self._evaluate_app(event, context)
        elif event.event_type == "ui.action":
            return self._evaluate_ui(event, context)
        return FilterResult(FilterVerdict.ACCEPT)

    def _evaluate_fs(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        path = str(event.payload.get("path", event.source))
        if not path:
            return FilterResult(FilterVerdict.ACCEPT)

        # 1. Explicit scope takes highest priority
        for scope_dir in context.fs_scope:
            if path.startswith(scope_dir):
                return FilterResult(FilterVerdict.ACCEPT)

        # 2. Hard reject platform-specific system paths (before extension match —
        #    a .pdf in /Library/Caches/ is system noise, not user intent)
        for pattern in self._fs_hard_reject:
            if pattern.search(path):
                return FilterResult(
                    FilterVerdict.REJECT,
                    reason=f"fs_hard_reject:{path}",
                )

        # 3. Extension match in non-system paths
        if context.goal_extensions:
            ext = os.path.splitext(path)[1].lower()
            if ext and ext in context.goal_extensions:
                context.expand_fs_scope(path)
                return FilterResult(FilterVerdict.ACCEPT)

        # 4. Unknown zone — preserve but downweight
        return FilterResult(
            FilterVerdict.ANNOTATE_NOISE,
            reason=f"outside_fs_scope:{path}",
            confidence=0.6,
        )

    def _evaluate_app(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        bundle_id = str(event.payload.get("bundle_id", event.source))
        if not bundle_id:
            return FilterResult(FilterVerdict.ACCEPT)
        if bundle_id in self._system_bundles or self._is_system_app(bundle_id):
            return FilterResult(
                FilterVerdict.REJECT,
                reason=f"system_bundle:{bundle_id}",
            )
        if bundle_id in context.app_scope:
            return FilterResult(FilterVerdict.ACCEPT)
        # App mentioned in goal → accept immediately
        if context.goal_keywords:
            app_short = bundle_id.split(".")[-1].lower()
            if app_short in context.goal_keywords:
                context.expand_app_scope(bundle_id)
                return FilterResult(FilterVerdict.ACCEPT)
        # First focus: accept the event (switch is always informative) but don't
        # expand scope until we confirm it's not a transient visit.
        if context.is_transient_visit(bundle_id):
            return FilterResult(FilterVerdict.ACCEPT)
        # App has accumulated enough interactions → promote to scope
        context.expand_app_scope(bundle_id)
        return FilterResult(FilterVerdict.ACCEPT)

    @staticmethod
    def _is_system_app(bundle_id: str) -> bool:
        """Heuristic check for system/input method apps not in the static list."""
        return ".inputmethod." in bundle_id

    def _evaluate_ui(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        app = str(event.payload.get("app_bundle_id", ""))
        sub_type = str(event.payload.get("sub_type", ""))

        # Window management actions are low-signal noise (don't count as interaction)
        if sub_type in _WINDOW_MGMT_UI_ACTIONS:
            return FilterResult(
                FilterVerdict.ANNOTATE_NOISE,
                reason=f"window_mgmt:{sub_type}",
                confidence=0.8,
            )

        # Record meaningful interaction for transient visit tracking
        if app and sub_type not in _WINDOW_MGMT_UI_ACTIONS:
            context.record_app_interaction(app)
            # Promote app if it now meets the interaction threshold
            if app not in context.app_scope and not context.is_transient_visit(app):
                context.expand_app_scope(app)

        # Domain inter-cooperation: UI in file manager → expand FS scope
        if app in _FINDER_BUNDLES:
            context.observe_ui_for_fs_scope(event)

        if not app or app in context.app_scope:
            return FilterResult(FilterVerdict.ACCEPT)

        # Functional UI in unknown apps (has identifiable target) gets lower noise confidence
        has_label = bool(event.payload.get("label", ""))
        has_role = bool(event.payload.get("role", ""))
        if has_label or has_role:
            return FilterResult(
                FilterVerdict.ANNOTATE_NOISE,
                reason=f"ui_outside_app_scope:{app}",
                confidence=0.4,
            )

        return FilterResult(
            FilterVerdict.ANNOTATE_NOISE,
            reason=f"ui_outside_app_scope:{app}",
            confidence=0.6,
        )


# ── P2: Noise Rule Filter ──


class NoiseRuleFilter(AttentionFilter):
    """Pattern-based noise filtering using configurable regex rules.

    Two tiers:
      - Hard-reject patterns: fs.change events matching confirmed garbage paths
        are REJECT-ed outright (never enter the trajectory).
      - Soft patterns: matching events are ANNOTATE_NOISE (preserved but downweighted).
    """

    def __init__(
        self,
        patterns: Sequence[str] = (),
        reject_patterns: Sequence[str] = (),
    ) -> None:
        self._compiled = [re.compile(p) for p in patterns if p]
        self._reject_compiled = [re.compile(p) for p in reject_patterns if p]

    def evaluate(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        candidates = self._extract_matchable_strings(event)

        if self._reject_compiled and event.event_type == "fs.change":
            for text in candidates:
                for pattern in self._reject_compiled:
                    if pattern.search(text):
                        return FilterResult(
                            FilterVerdict.REJECT,
                            reason=f"noise_reject:{pattern.pattern}",
                            confidence=0.95,
                        )

        if not self._compiled:
            return FilterResult(FilterVerdict.ACCEPT)

        for text in candidates:
            for pattern in self._compiled:
                if pattern.search(text):
                    return FilterResult(
                        FilterVerdict.ANNOTATE_NOISE,
                        reason=f"noise_rule:{pattern.pattern}",
                        confidence=0.9,
                    )

        return FilterResult(FilterVerdict.ACCEPT)

    @staticmethod
    def _extract_matchable_strings(event: SystemEvent) -> List[str]:
        """Extract strings from the event to match against noise patterns."""
        strings = [event.source]
        payload = event.payload
        if "path" in payload:
            strings.append(str(payload["path"]))
        if "bundle_id" in payload:
            strings.append(str(payload["bundle_id"]))
        if "app_bundle_id" in payload:
            strings.append(str(payload["app_bundle_id"]))
        return [s for s in strings if s]


# ── P3: Working Directory Filter ──


class WorkingDirFilter(AttentionFilter):
    """Filter filesystem events outside the inferred working directory.

    Only applies to fs.change events. If a working directory has been inferred,
    file events outside that scope are annotated as noise.
    """

    def evaluate(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        if event.event_type != "fs.change":
            return FilterResult(FilterVerdict.ACCEPT)

        if not context.working_dirs:
            return FilterResult(FilterVerdict.ACCEPT)

        path = str(event.payload.get("path", event.source))
        if not path:
            return FilterResult(FilterVerdict.ACCEPT)

        for wd in context.working_dirs:
            if path.startswith(wd):
                return FilterResult(FilterVerdict.ACCEPT)

        return FilterResult(
            FilterVerdict.ANNOTATE_NOISE,
            reason=f"outside_working_dir:{path}",
            confidence=0.7,
        )


# ── P1: Goal Relevance Filter (post-hoc) ──


class GoalRelevanceFilter:
    """Post-hoc filter: scores trajectory steps against the user's stated goal.

    Uses lightweight keyword overlap to identify steps whose target/app
    are semantically unrelated to the goal. Steps below threshold are
    annotated as noise but preserved in the trajectory.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self._threshold = threshold

    def filter_trajectory(self, trajectory: "Trajectory", goal: str) -> "Trajectory":
        """Annotate irrelevant steps as noise. Returns the same trajectory (mutated)."""
        if not goal or not trajectory.steps:
            return trajectory

        goal_tokens = self._tokenize(goal)
        if not goal_tokens:
            return trajectory

        for step in trajectory.steps:
            if self._has_noise_annotation(step, "goal_irrelevant"):
                continue
            relevance = self._compute_relevance(step, goal_tokens)
            if relevance < self._threshold:
                noise = step.action.params.setdefault("_noise", [])
                noise.append({
                    "signal_type": "goal_irrelevant",
                    "confidence": 1.0 - relevance,
                    "related_step": -1,
                })

        return trajectory

    def _compute_relevance(self, step: "TrajectoryStep", goal_tokens: Set[str]) -> float:
        """Compute relevance score between a step and the goal tokens."""
        step_tokens = self._step_to_tokens(step)
        if not step_tokens:
            return 0.5  # neutral — don't penalize steps with no textual signal

        intersection = goal_tokens & step_tokens
        if not intersection:
            return 0.0
        return len(intersection) / min(len(goal_tokens), len(step_tokens))

    @staticmethod
    def _step_to_tokens(step: "TrajectoryStep") -> Set[str]:
        """Extract meaningful tokens from a trajectory step."""
        parts: List[str] = []
        action = step.action
        if action.target:
            parts.append(action.target)
        if action.target_label:
            parts.append(action.target_label)
        if action.app_bundle_id:
            parts.append(action.app_bundle_id.split(".")[-1])
        if action.app_name:
            parts.append(action.app_name)

        text = " ".join(parts).lower()
        return {w for w in re.split(r"[^a-z0-9一-鿿]+", text) if len(w) > 1}

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Tokenize text into lowercase keyword set."""
        return {w for w in re.split(r"[^a-z0-9一-鿿]+", text.lower()) if len(w) > 1}

    @staticmethod
    def _has_noise_annotation(step: "TrajectoryStep", signal_type: str) -> bool:
        noise = step.action.params.get("_noise", [])
        return any(n.get("signal_type") == signal_type for n in noise)


# ═══════════════════════════════════════════════════════════════════════
# Surprise Annotator — event-level surprise detection
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SurpriseConfig:
    """Configuration for event-level surprise detection."""

    stat_weight: float = 0.4
    temporal_weight: float = 0.3
    pattern_weight: float = 0.3
    z_score_threshold: float = 2.0
    annotation_threshold: float = 0.5
    warmup_events: int = 50


class SurpriseAnnotator:
    """Event-level surprise detection for the attention system.

    Does NOT filter events — instead annotates them with a surprise score
    when they deviate from the observed distribution. Positioned after all
    filters in the chain; only sees events that passed all filters.

    Three detection layers:
      - Statistical: frequency anomaly via Z-score of (channel, sub_type, app)
      - Temporal: inter-event interval anomaly (sudden bursts or pauses)
      - Pattern: bigram sequence novelty (unexpected event transitions)
    """

    def __init__(self, config: SurpriseConfig = SurpriseConfig()) -> None:
        from collections import Counter, deque as _deque

        self._config = config
        self._event_counter: Counter = Counter()
        self._interval_window: _deque = _deque(maxlen=100)
        self._bigram_counter: Counter = Counter()
        self._last_event_key: str = ""
        self._last_event_time: float = 0.0
        self._total_events: int = 0

    def annotate(
        self, event: SystemEvent, context: RecordingContext,
    ) -> Optional[Dict[str, Any]]:
        """Compute surprise score; return annotation dict if above threshold."""
        self._total_events += 1
        if self._total_events < self._config.warmup_events:
            self._update_stats(event, context)
            return None

        stat = self._stat_surprise(event, context)
        temporal = self._temporal_surprise(event)
        pattern = self._pattern_surprise(event, context)

        self._update_stats(event, context)

        cfg = self._config
        total = cfg.stat_weight * stat + cfg.temporal_weight * temporal + cfg.pattern_weight * pattern

        if total < cfg.annotation_threshold:
            return None

        return {
            "signal_type": "surprise",
            "total": round(total, 3),
            "stat": round(stat, 3),
            "temporal": round(temporal, 3),
            "pattern": round(pattern, 3),
        }

    def _event_key(self, event: SystemEvent, context: RecordingContext) -> str:
        et = event.event_type if isinstance(event.event_type, str) else event.event_type.value
        channel = et.split(".")[0] if "." in et else et
        sub = str(event.payload.get("sub_type", event.payload.get("action", ""))) or et
        app = context.current_focused_app or ""
        return f"{channel}:{sub}:{app}"

    def _stat_surprise(self, event: SystemEvent, context: RecordingContext) -> float:
        """Z-score based frequency anomaly detection."""
        key = self._event_key(event, context)
        count = self._event_counter.get(key, 0)

        if not self._event_counter:
            return 0.0
        if count == 0:
            return 1.0

        counts = list(self._event_counter.values())
        if len(counts) < 2:
            return 0.0

        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std = variance ** 0.5
        if std < 1e-6:
            return 0.0

        z = abs(count - mean) / std
        threshold = self._config.z_score_threshold
        if z > threshold:
            return min(1.0, z / (threshold * 2))
        return 0.0

    def _temporal_surprise(self, event: SystemEvent) -> float:
        """Detect abnormal inter-event intervals."""
        now = getattr(event, "timestamp", 0.0) or _time_mod.time()
        if self._last_event_time <= 0:
            return 0.0

        interval = now - self._last_event_time
        if len(self._interval_window) < 5:
            return 0.0

        intervals = list(self._interval_window)
        mean = sum(intervals) / len(intervals)
        if mean < 1e-6:
            return 0.0

        variance = sum((i - mean) ** 2 for i in intervals) / len(intervals)
        std = variance ** 0.5
        if std < 1e-6:
            return 0.0

        z = abs(interval - mean) / std
        threshold = self._config.z_score_threshold
        if z > threshold:
            return min(1.0, z / (threshold * 2))
        return 0.0

    def _pattern_surprise(self, event: SystemEvent, context: RecordingContext) -> float:
        """Detect novel event bigram transitions."""
        cur_key = self._event_key(event, context)
        if not self._last_event_key:
            return 0.0

        bigram = (self._last_event_key, cur_key)
        count = self._bigram_counter.get(bigram, 0)
        if count == 0:
            return 1.0

        total_bigrams = sum(self._bigram_counter.values())
        if total_bigrams < 10:
            return 0.0

        expected = total_bigrams / max(1, len(self._bigram_counter))
        ratio = count / max(1e-6, expected)
        if ratio < 0.2:
            return min(1.0, 1.0 - ratio)
        return 0.0

    def _update_stats(self, event: SystemEvent, context: RecordingContext) -> None:
        """Update internal frequency and timing statistics."""
        key = self._event_key(event, context)
        self._event_counter[key] += 1

        now = getattr(event, "timestamp", 0.0) or _time_mod.time()
        if self._last_event_time > 0:
            self._interval_window.append(now - self._last_event_time)
        self._last_event_time = now

        if self._last_event_key:
            self._bigram_counter[(self._last_event_key, key)] += 1
        self._last_event_key = key


# ── Factory ──


def build_attention_filters(
    *,
    foreground_gate: bool = True,
    noise_patterns: Sequence[str] = (),
    working_dir_inference: bool = True,
    domain_whitelist: bool = True,
    goal: str = "",
    platform_hint: str = "darwin",
    perceptual_field_enabled: bool = False,
    perceptual_field_config: "Optional[str]" = None,
    perceptual_field_rules: "Sequence[Any] | None" = None,
) -> List[AttentionFilter]:
    """Build the default attention filter chain from configuration."""
    from pathlib import Path

    filters: List[AttentionFilter] = []
    if foreground_gate:
        filters.append(ForegroundGateFilter())
    if domain_whitelist:
        filters.append(DomainWhitelistFilter(goal=goal, platform_hint=platform_hint))

    if perceptual_field_enabled:
        from leapflow.recording.field_policy_loader import FieldPolicyLoader
        from leapflow.recording.perceptual_field import (
            ContextExtractor,
            PerceptualFieldFilter,
        )
        loader = FieldPolicyLoader()
        config_path = Path(perceptual_field_config) if perceptual_field_config else None
        policy = loader.load(
            config_path=config_path,
            goal=goal,
            extra_rules=perceptual_field_rules or (),
        )
        extractor = ContextExtractor()
        filters.append(PerceptualFieldFilter(extractor, policy))

    all_patterns = list(_DEFAULT_NOISE_PATTERNS) + list(noise_patterns)
    filters.append(NoiseRuleFilter(
        patterns=all_patterns,
        reject_patterns=_HARD_REJECT_NOISE_PATTERNS,
    ))
    if working_dir_inference:
        filters.append(WorkingDirFilter())
    return filters
