# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic tests for monitor/display signal noise suppression."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

from leapflow.domain.events import SystemEvent
from leapflow.monitor.signal_metrics import SignalMetricsCollector
from leapflow.monitor.signal_noise import SignalNoiseConfig, SignalNoiseGate


def _event(event_type: str, source: str) -> SystemEvent:
    return SystemEvent(event_type=event_type, source=source, payload={"path": source}, timestamp=time.time())


def _gate() -> SignalNoiseGate:
    return SignalNoiseGate(SignalNoiseConfig(workspace_root="/repo", same_source_cooldown_s=2.0))


def test_signal_noise_gate_passes_workspace_source_file() -> None:
    gate = _gate()

    assert gate.should_pass(_event("fs.change", "/repo/src/leapflow/dashboard/service.py")) is True
    assert gate.stats["passed"] == 1
    assert gate.stats["suppressed"] == 0


def test_signal_noise_gate_suppresses_cache_paths_and_tracks_reason() -> None:
    gate = _gate()

    assert gate.should_pass(_event(
        "fs.change",
        "/Users/jason/Library/Application Support/Qoder/User/globalStorage/state.vscdb-shm",
    )) is False

    stats = gate.stats
    assert stats["suppressed"] == 1
    assert stats["by_reason"] == {"transient_suffix": 1}
    assert stats["by_family"] == {"fs": 1}


def test_signal_noise_gate_suppresses_noisy_dirs_inside_workspace() -> None:
    gate = _gate()

    assert gate.should_pass(_event("fs.change", "/repo/.git/index.lock")) is False

    assert gate.stats["suppressed"] == 1
    assert gate.stats["by_reason"] == {"transient_suffix": 1}


def test_signal_noise_gate_suppresses_windows_style_noisy_dirs() -> None:
    gate = SignalNoiseGate(SignalNoiseConfig(workspace_root="C:/repo", same_source_cooldown_s=0.0))

    assert gate.should_pass(_event("fs.change", r"C:\repo\.cache\state.db")) is False
    assert gate.should_pass(_event("fs.change", "C:/repo/node_modules/pkg/index.js")) is False

    assert gate.stats["suppressed"] == 2
    assert gate.stats["by_reason"] == {"path_fragment": 1, "noisy_dir": 1}


def test_signal_noise_gate_handles_mixed_windows_separators_for_workspace() -> None:
    gate = SignalNoiseGate(SignalNoiseConfig(workspace_root=r"C:\repo", same_source_cooldown_s=0.0))

    assert gate.should_pass(_event("fs.change", "C:/repo/src/main.py")) is True

    assert gate.stats["passed"] == 1
    assert gate.stats["suppressed"] == 0


def test_signal_noise_gate_suppresses_fs_changes_outside_workspace_by_default() -> None:
    gate = _gate()

    assert gate.should_pass(_event("fs.change", "/Users/jason/other-project/src/main.py")) is False

    assert gate.stats["suppressed"] == 1
    assert gate.stats["by_reason"] == {"outside_workspace": 1}


def test_signal_noise_gate_can_allow_fs_changes_outside_workspace() -> None:
    gate = SignalNoiseGate(SignalNoiseConfig(
        workspace_root="/repo",
        allow_fs_outside_workspace=True,
        same_source_cooldown_s=0.0,
    ))

    assert gate.should_pass(_event("fs.change", "/Users/jason/other-project/src/main.py")) is True

    assert gate.stats["passed"] == 1
    assert gate.stats["suppressed"] == 0


def test_signal_noise_gate_suppresses_same_source_burst_but_allows_later() -> None:
    gate = _gate()
    clock = [100.0]

    def _monotonic() -> float:
        return clock[0]

    with patch("leapflow.monitor.signal_noise.time.monotonic", side_effect=_monotonic):
        assert gate.should_pass(_event("fs.change", "/repo/src/main.py")) is True
        clock[0] = 101.0
        assert gate.should_pass(_event("fs.change", "/repo/src/main.py")) is False
        clock[0] = 103.1
        assert gate.should_pass(_event("fs.change", "/repo/src/main.py")) is True

    assert gate.stats["passed"] == 2
    assert gate.stats["suppressed"] == 1
    assert gate.stats["by_reason"] == {"same_source_burst": 1}


def test_signal_noise_gate_keeps_non_fs_signals() -> None:
    gate = _gate()

    assert gate.should_pass(_event("gateway.signal", "lark")) is True
    assert gate.should_pass(_event("ui.action", "keyboard")) is True
    assert gate.should_pass(_event("clipboard.change", "system.clipboard")) is True

    assert gate.stats["passed"] == 3
    assert gate.stats["suppressed"] == 0


def test_signal_noise_gate_can_be_disabled() -> None:
    gate = SignalNoiseGate(SignalNoiseConfig(enabled=False, workspace_root="/repo"))

    assert gate.should_pass(_event("fs.change", "/Users/jason/Library/Caches/noise.log")) is True

    assert gate.stats["passed"] == 1
    assert gate.stats["suppressed"] == 0


def test_signal_noise_config_reads_settings_overrides() -> None:
    settings = SimpleNamespace(
        workspace_root="/work",
        signal_noise_gate_enabled=False,
        signal_noise_same_source_cooldown_s=0.5,
        signal_noise_allow_fs_outside_workspace=True,
        signal_noise_path_fragments=("/custom/noise/",),
        signal_noise_dir_names=("Generated",),
        signal_noise_suffixes=(".cachefile",),
    )

    config = SignalNoiseConfig.from_settings(settings)

    assert config.enabled is False
    assert config.workspace_root == "/work"
    assert config.same_source_cooldown_s == 0.5
    assert config.allow_fs_outside_workspace is True
    assert config.path_fragments == ("/custom/noise/",)
    assert config.dir_names == ("Generated",)
    assert config.suffixes == (".cachefile",)


def test_signal_metrics_collector_includes_noise_gate_stats() -> None:
    gate = _gate()
    gate.should_pass(_event("fs.change", "/repo/src/main.py"))
    gate.should_pass(_event("fs.change", "/Users/jason/.hermes/kanban.db-shm"))

    snapshot = SignalMetricsCollector().collect(signal_noise_gate=gate)

    assert snapshot.signal_noise_seen == 2
    assert snapshot.signal_noise_passed == 1
    assert snapshot.signal_noise_suppressed == 1
    assert snapshot.signal_noise_by_reason == {"transient_suffix": 1}
    assert snapshot.signal_noise_by_family == {"fs": 1}
