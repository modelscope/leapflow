# Copyright (c) Alibaba, Inc. and its affiliates.
"""Signal noise gate for monitor and LeapBoard live-stream ingestion.

This gate is intentionally *signal-attribute based* rather than natural-language
routing: it classifies stable, observable noise properties (cache dirs, transient
suffixes, same-source bursts) before events wake monitor watches or enter the
LeapBoard live stream. The raw EventBus pipeline can still ingest/remember the
normalized event; this gate controls the monitor/display boundary where high
frequency OS/tool churn has the highest user-facing cost.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from leapflow.domain.events import SystemEvent

_DEFAULT_NOISY_PATH_FRAGMENTS = (
    "/Library/Caches/",
    "/Library/Logs/",
    "/Library/Preferences/",
    "/Application Support/Qoder/User/globalStorage/",
    "/Application Support/Qoder/SharedClientCache/",
    "/Application Support/Qoder/SharedCredentialCache/",
    "/Application Support/Qoder/Partitions/native-browser/Cache/",
    "/Application Support/Cursor/User/globalStorage/",
    "/Application Support/DuetExpertCenter/",
    "/.cache/",
    "/.hermes/",
    "/.r2c/logs/",
)

_DEFAULT_NOISY_DIR_NAMES = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "Cache",
    "Caches",
    "SharedCredentialCache",
    "globalStorage",
    "Code Cache",
    "GPUCache",
)

_DEFAULT_NOISY_SUFFIXES = (
    ".tmp",
    ".temp",
    ".swp",
    ".lock",
    ".log",
    ".pyc",
    ".pyo",
    "-journal",
    "-shm",
    "-wal",
    ".db-shm",
    ".db-wal",
    ".sqlite-journal",
)


@dataclass(frozen=True)
class SignalNoiseConfig:
    """Runtime policy for monitor/display signal noise suppression."""

    enabled: bool = True
    workspace_root: str = ""
    same_source_cooldown_s: float = 2.0
    allow_fs_outside_workspace: bool = False
    path_fragments: tuple[str, ...] = _DEFAULT_NOISY_PATH_FRAGMENTS
    dir_names: tuple[str, ...] = _DEFAULT_NOISY_DIR_NAMES
    suffixes: tuple[str, ...] = _DEFAULT_NOISY_SUFFIXES

    @classmethod
    def from_settings(cls, settings: Any) -> "SignalNoiseConfig":
        """Build policy from Settings-like objects while keeping safe defaults."""
        return cls(
            enabled=bool(getattr(settings, "signal_noise_gate_enabled", True)),
            workspace_root=str(getattr(settings, "workspace_root", "") or ""),
            same_source_cooldown_s=max(
                0.0, float(getattr(settings, "signal_noise_same_source_cooldown_s", 2.0) or 0.0)
            ),
            allow_fs_outside_workspace=bool(
                getattr(settings, "signal_noise_allow_fs_outside_workspace", False)
            ),
            path_fragments=tuple(
                getattr(settings, "signal_noise_path_fragments", _DEFAULT_NOISY_PATH_FRAGMENTS)
                or ()
            ),
            dir_names=tuple(
                getattr(settings, "signal_noise_dir_names", _DEFAULT_NOISY_DIR_NAMES)
                or ()
            ),
            suffixes=tuple(
                getattr(settings, "signal_noise_suffixes", _DEFAULT_NOISY_SUFFIXES)
                or ()
            ),
        )


@dataclass
class SignalNoiseStats:
    """Counters exposed to dashboard metrics for transparency."""

    seen: int = 0
    passed: int = 0
    suppressed: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)
    by_family: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        """Serialize counters without exposing sampled file paths."""
        return {
            "seen": self.seen,
            "passed": self.passed,
            "suppressed": self.suppressed,
            "by_reason": dict(self.by_reason),
            "by_family": dict(self.by_family),
        }


@dataclass(frozen=True)
class SignalNoiseDecision:
    """One classification verdict."""

    pass_event: bool
    reason: str = ""


class SignalNoiseGate:
    """Suppress low-value monitor/display events before they trigger UI churn."""

    def __init__(self, config: SignalNoiseConfig | None = None) -> None:
        self.config = config or SignalNoiseConfig()
        self._stats = SignalNoiseStats()
        self._last_seen_by_source: dict[str, float] = {}

    @property
    def stats(self) -> dict[str, Any]:
        """Return a stable snapshot for metrics collectors."""
        return self._stats.to_dict()

    def update_config(self, config: SignalNoiseConfig) -> None:
        """Apply new policy without resetting suppression counters."""
        self.config = config

    def should_pass(self, event: SystemEvent) -> bool:
        """Classify and update counters. True means monitor/display may see it."""
        self._stats.seen += 1
        decision = self._decide(event)
        if decision.pass_event:
            self._stats.passed += 1
            self._remember_source(event)
            return True
        self._stats.suppressed += 1
        reason = decision.reason or "noise"
        self._stats.by_reason[reason] += 1
        self._stats.by_family[_event_family(event.event_type)] += 1
        return False

    def _decide(self, event: SystemEvent) -> SignalNoiseDecision:
        if not self.config.enabled:
            return SignalNoiseDecision(True)
        event_type = str(event.event_type or "")
        if event_type != "fs.change":
            return SignalNoiseDecision(True)
        source = str(event.source or event.payload.get("path") or "")
        if not source:
            return SignalNoiseDecision(True)
        normalized = source.replace("\\", "/")
        workspace_root = self.config.workspace_root.replace("\\", "/").rstrip("/")
        in_workspace = bool(workspace_root and _is_relative_to(normalized, workspace_root))

        if self._is_same_source_burst(event):
            return SignalNoiseDecision(False, "same_source_burst")
        if self._matches_suffix(normalized):
            return SignalNoiseDecision(False, "transient_suffix")
        if self._matches_path_fragment(normalized):
            return SignalNoiseDecision(False, "path_fragment")
        if self._matches_noisy_dir(normalized):
            return SignalNoiseDecision(False, "noisy_dir")
        if workspace_root and not in_workspace and not self.config.allow_fs_outside_workspace:
            return SignalNoiseDecision(False, "outside_workspace")
        # Outside-workspace hidden app-state roots (e.g. ~/.hermes, ~/.cache)
        # are almost always runtime churn; keep workspace hidden dirs governed by
        # the explicit dir/suffix checks above so project dotfiles remain visible.
        if not in_workspace and _has_hidden_state_segment(normalized):
            return SignalNoiseDecision(False, "hidden_state")
        return SignalNoiseDecision(True)

    def _remember_source(self, event: SystemEvent) -> None:
        key = _event_key(event)
        if key:
            self._last_seen_by_source[key] = time.monotonic()

    def _is_same_source_burst(self, event: SystemEvent) -> bool:
        cooldown = float(self.config.same_source_cooldown_s or 0.0)
        if cooldown <= 0:
            return False
        key = _event_key(event)
        if not key:
            return False
        previous = self._last_seen_by_source.get(key, 0.0)
        return (time.monotonic() - previous) < cooldown

    def _matches_path_fragment(self, path: str) -> bool:
        return any(fragment and fragment in path for fragment in self.config.path_fragments)

    def _matches_suffix(self, path: str) -> bool:
        lowered = path.lower()
        return any(suffix and lowered.endswith(str(suffix).lower()) for suffix in self.config.suffixes)

    def _matches_noisy_dir(self, path: str) -> bool:
        parts = _path_parts(path)
        return any(name in parts for name in self.config.dir_names)


def _event_family(event_type: str) -> str:
    normalized = str(event_type or "unknown").replace(":", ".")
    return normalized.split(".", 1)[0] or "unknown"


def _event_key(event: SystemEvent) -> str:
    source = str(event.source or event.payload.get("path") or "")
    return f"{event.event_type}:{source}" if source else ""


def _is_relative_to(path: str, root: str) -> bool:
    path = path.rstrip("/")
    return path == root or path.startswith(root + "/")


def _path_parts(path: str) -> set[str]:
    """Split POSIX, Windows, and mixed-separator paths without OS-dependent Path.parts."""
    return {part for part in str(path).replace("\\", "/").split("/") if part}


def _has_hidden_state_segment(path: str) -> bool:
    return bool(_path_parts(path) & {".cache", ".hermes", ".qoder", ".r2c"})


__all__ = [
    "SignalNoiseConfig",
    "SignalNoiseDecision",
    "SignalNoiseGate",
    "SignalNoiseStats",
]
