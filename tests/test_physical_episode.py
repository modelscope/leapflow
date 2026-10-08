# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for PhysicalEpisodeRecorder and EpisodeExporter."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from leapflow.learning.physical_episode import (
    EPISODE_SCHEMA_VERSION,
    PhysicalEpisode,
    PhysicalEpisodeRecorder,
    make_episode_id,
)
from leapflow.robot.episode_export import EpisodeExporter
from leapflow.robot.trajectory import (
    PhysicalTrajectory,
    PhysicalTrajectoryStep,
    make_trajectory_id,
)


# ---------------------------------------------------------------------------
# Fixtures: mock registry
# ---------------------------------------------------------------------------


@dataclass
class _MockReading:
    channel_id: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"channel_id": self.channel_id, "value": self.value}


@dataclass
class _MockBatch:
    readings: tuple[_MockReading, ...]


@dataclass
class _MockChannel:
    channel_id: str
    is_readable: bool = True


@dataclass
class _MockContext:
    channels: tuple[_MockChannel, ...]


class MockRegistry:
    """Minimal mock HardwareRegistry for episode tests."""

    def __init__(self, state: dict[str, float] | None = None) -> None:
        self._state = state or {"joint_0": 0.5, "joint_1": 1.0}
        self._call_count = 0

    def context(self, device_id: str) -> _MockContext:
        channels = tuple(
            _MockChannel(channel_id=ch) for ch in self._state
        )
        return _MockContext(channels=channels)

    async def read_batch(self, device_id: str, channels: tuple[str, ...]) -> _MockBatch:
        self._call_count += 1
        readings = tuple(
            _MockReading(channel_id=ch, value=self._state.get(ch, 0.0))
            for ch in channels
        )
        return _MockBatch(readings=readings)

    def set_state(self, new_state: dict[str, float]) -> None:
        self._state = dict(new_state)


def _make_step(seq: int = 0, source: str = "teleop") -> PhysicalTrajectoryStep:
    """Create a test trajectory step."""
    return PhysicalTrajectoryStep(
        timestamp=time.time(),
        monotonic_at=time.monotonic(),
        joint_positions={"joint_0": 0.1 * seq, "joint_1": 0.2 * seq},
        joint_velocities={"joint_0": 0.01, "joint_1": 0.02},
        gripper_state=0.5,
        action_source=source,
        sequence=seq,
    )


def _make_step_with_camera(seq: int = 0, camera_ref: str = "/tmp/f.png") -> PhysicalTrajectoryStep:
    """Create a step with a camera frame reference."""
    return PhysicalTrajectoryStep(
        timestamp=time.time(),
        monotonic_at=time.monotonic(),
        joint_positions={"joint_0": 0.1 * seq},
        camera_frames={"cam0": camera_ref},
        action_source="teleop",
        sequence=seq,
    )


# ---------------------------------------------------------------------------
# PhysicalEpisode dataclass tests
# ---------------------------------------------------------------------------


class TestPhysicalEpisode:
    """Test the PhysicalEpisode frozen dataclass."""

    def test_basic_construction(self) -> None:
        trajectory = PhysicalTrajectory(
            trajectory_id="traj-1",
            device_id="robot.arm0",
            goal="pick",
            steps=(_make_step(0), _make_step(1)),
            started_at=100.0,
            ended_at=110.0,
        )
        ep = PhysicalEpisode(
            episode_id="ep-1",
            device_id="robot.arm0",
            task="pick",
            trajectory=trajectory,
            initial_state={"joint_0": 0.0},
            final_state={"joint_0": 1.0},
            success=True,
            started_at=100.0,
            ended_at=110.0,
        )
        assert ep.episode_id == "ep-1"
        assert ep.success is True
        assert ep.duration_s == pytest.approx(10.0)
        assert ep.step_count == 2

    def test_to_dict_roundtrip(self) -> None:
        ep = PhysicalEpisode(
            episode_id="ep-2",
            device_id="robot.arm0",
            task="place",
            trajectory=None,
            initial_state={"a": 1},
            final_state={"a": 2},
            evidence_ids=("ev-1", "ev-2"),
            success=False,
        )
        d = ep.to_dict()
        assert d["episode_id"] == "ep-2"
        assert d["evidence_ids"] == ["ev-1", "ev-2"]
        assert d["success"] is False
        assert d["trajectory"] is None

    def test_zero_duration_when_no_timestamps(self) -> None:
        ep = PhysicalEpisode(
            episode_id="ep-3",
            device_id="d",
            task="t",
            trajectory=None,
            initial_state={},
            final_state={},
        )
        assert ep.duration_s == 0.0
        assert ep.step_count == 0


class TestMakeEpisodeId:
    def test_unique(self) -> None:
        ids = {make_episode_id() for _ in range(100)}
        assert len(ids) == 100

    def test_length(self) -> None:
        assert len(make_episode_id()) == 16


# ---------------------------------------------------------------------------
# PhysicalEpisodeRecorder lifecycle tests
# ---------------------------------------------------------------------------


class TestPhysicalEpisodeRecorder:
    """Test the full episode lifecycle: begin → record → end."""

    @pytest.fixture()
    def registry(self) -> MockRegistry:
        return MockRegistry()

    @pytest.mark.asyncio
    async def test_basic_lifecycle(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)

        eid = await recorder.begin_episode("robot.arm0", "pick_cube")
        assert recorder.is_recording
        assert isinstance(eid, str) and len(eid) == 16

        recorder.record_step(_make_step(0))
        recorder.record_step(_make_step(1))
        recorder.record_evidence("ev-abc")

        episode = await recorder.end_episode()
        assert not recorder.is_recording
        assert episode.episode_id == eid
        assert episode.device_id == "robot.arm0"
        assert episode.task == "pick_cube"
        assert episode.step_count == 2
        assert "ev-abc" in episode.evidence_ids
        assert episode.success is False  # default when no verdict

    @pytest.mark.asyncio
    async def test_success_from_verdict(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        await recorder.begin_episode("d", "t")
        recorder.record_step(_make_step(0))

        verdict = MagicMock()
        verdict.is_success = True
        verdict.status = "success"
        verdict.confidence = 0.95

        episode = await recorder.end_episode(verdict)
        assert episode.success is True
        assert episode.verdict is verdict

    @pytest.mark.asyncio
    async def test_failure_verdict(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        await recorder.begin_episode("d", "t")

        verdict = MagicMock()
        verdict.is_success = False
        verdict.status = "failure"
        verdict.confidence = 0.9

        episode = await recorder.end_episode(verdict)
        assert episode.success is False

    @pytest.mark.asyncio
    async def test_explicit_success_flag(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        await recorder.begin_episode("d", "t")
        episode = await recorder.end_episode(success=True)
        assert episode.success is True

    @pytest.mark.asyncio
    async def test_double_begin_raises(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        await recorder.begin_episode("d", "t")
        with pytest.raises(RuntimeError, match="already in progress"):
            await recorder.begin_episode("d2", "t2")

    @pytest.mark.asyncio
    async def test_end_without_begin_raises(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            await recorder.end_episode()

    @pytest.mark.asyncio
    async def test_record_step_without_begin_raises(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            recorder.record_step(_make_step(0))

    @pytest.mark.asyncio
    async def test_record_evidence_without_begin_raises(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            recorder.record_evidence("ev-1")

    @pytest.mark.asyncio
    async def test_initial_and_final_state_captured(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        await recorder.begin_episode("d", "t")
        # Change state between begin and end.
        registry.set_state({"joint_0": 9.9, "joint_1": 8.8})
        episode = await recorder.end_episode()

        assert episode.initial_state == {"joint_0": 0.5, "joint_1": 1.0}
        assert episode.final_state == {"joint_0": 9.9, "joint_1": 8.8}

    @pytest.mark.asyncio
    async def test_episodes_recorded_counter(self, registry: MockRegistry) -> None:
        recorder = PhysicalEpisodeRecorder(registry)
        assert recorder.episodes_recorded == 0

        await recorder.begin_episode("d", "t")
        await recorder.end_episode()
        assert recorder.episodes_recorded == 1

        await recorder.begin_episode("d", "t2")
        await recorder.end_episode()
        assert recorder.episodes_recorded == 2


# ---------------------------------------------------------------------------
# DuckDB persistence tests
# ---------------------------------------------------------------------------


class TestEpisodePersistence:
    """Test DuckDB write/query round-trip."""

    @pytest.mark.asyncio
    async def test_persist_and_query(self, tmp_path: Path) -> None:
        registry = MockRegistry()
        db_path = tmp_path / "episodes.duckdb"
        recorder = PhysicalEpisodeRecorder(registry, db_path=db_path)

        await recorder.begin_episode("robot.arm0", "pick")
        recorder.record_step(_make_step(0))
        await recorder.end_episode(success=True)

        await recorder.begin_episode("robot.arm0", "place")
        recorder.record_step(_make_step(0))
        await recorder.end_episode(success=False)

        # Query all.
        all_eps = await recorder.query_episodes(device_id="robot.arm0")
        assert len(all_eps) == 2

        # Query success only.
        success_eps = await recorder.query_episodes(
            device_id="robot.arm0", success_only=True,
        )
        assert len(success_eps) == 1
        assert success_eps[0]["task"] == "pick"

        # Query by task.
        task_eps = await recorder.query_episodes(task="place")
        assert len(task_eps) == 1

        recorder.close()

    @pytest.mark.asyncio
    async def test_query_without_db(self) -> None:
        registry = MockRegistry()
        recorder = PhysicalEpisodeRecorder(registry)  # no db_path
        result = await recorder.query_episodes()
        assert result == []


# ---------------------------------------------------------------------------
# Evolution integration tests
# ---------------------------------------------------------------------------


class TestEvolutionFeed:
    """Test best-effort evolution feed."""

    @pytest.mark.asyncio
    async def test_evolution_feed_success(self) -> None:
        registry = MockRegistry()
        mock_outbox = AsyncMock()
        mock_recorder = MagicMock()
        mock_recorder._outbox = mock_outbox

        recorder = PhysicalEpisodeRecorder(
            registry, action_recorder=mock_recorder,
        )
        await recorder.begin_episode("d", "t")
        await recorder.end_episode(success=True)

        mock_outbox.publish.assert_called_once()
        event = mock_outbox.publish.call_args[0][0]
        assert event.payload["ok"] is True
        assert event.payload["action_type"] == "physical_episode"

    @pytest.mark.asyncio
    async def test_evolution_feed_failure_episode(self) -> None:
        registry = MockRegistry()
        mock_outbox = AsyncMock()
        mock_recorder = MagicMock()
        mock_recorder._outbox = mock_outbox

        recorder = PhysicalEpisodeRecorder(
            registry, action_recorder=mock_recorder,
        )
        await recorder.begin_episode("d", "t")
        await recorder.end_episode(success=False)

        event = mock_outbox.publish.call_args[0][0]
        assert event.payload["ok"] is False

    @pytest.mark.asyncio
    async def test_evolution_feed_skipped_without_recorder(self) -> None:
        registry = MockRegistry()
        recorder = PhysicalEpisodeRecorder(registry)
        await recorder.begin_episode("d", "t")
        # Should not raise.
        episode = await recorder.end_episode()
        assert episode is not None

    @pytest.mark.asyncio
    async def test_evolution_feed_tolerates_broken_outbox(self) -> None:
        registry = MockRegistry()
        mock_outbox = AsyncMock()
        mock_outbox.publish.side_effect = RuntimeError("outbox broken")
        mock_recorder = MagicMock()
        mock_recorder._outbox = mock_outbox

        recorder = PhysicalEpisodeRecorder(
            registry, action_recorder=mock_recorder,
        )
        await recorder.begin_episode("d", "t")
        # Should not raise despite broken outbox.
        episode = await recorder.end_episode()
        assert episode is not None


# ---------------------------------------------------------------------------
# EpisodeExporter tests
# ---------------------------------------------------------------------------


class TestEpisodeExporter:
    """Test export to temporary directory."""

    def _make_episode(self, steps: int = 3) -> PhysicalEpisode:
        """Create a minimal episode for export tests."""
        step_list = tuple(_make_step(i) for i in range(steps))
        trajectory = PhysicalTrajectory(
            trajectory_id="traj-x",
            device_id="robot.arm0",
            goal="export_test",
            steps=step_list,
            started_at=100.0,
            ended_at=110.0,
        )
        return PhysicalEpisode(
            episode_id="ep-export",
            device_id="robot.arm0",
            task="export_test",
            trajectory=trajectory,
            initial_state={"joint_0": 0.0},
            final_state={"joint_0": 1.0},
            success=True,
            started_at=100.0,
            ended_at=110.0,
        )

    @pytest.mark.asyncio
    async def test_export_episode_json_fallback(self, tmp_path: Path) -> None:
        """Test export when pyarrow is NOT available (JSON fallback)."""
        exporter = EpisodeExporter(tmp_path, fps=30.0)
        episode = self._make_episode()

        with patch("leapflow.robot.episode_export._PYARROW_AVAILABLE", False):
            result = await exporter.export_episode(episode)

        assert result["episode_id"] == "ep-export"
        assert result["step_count"] == 3
        assert result["format"] == "json"

        # Check JSON data file exists.
        data_file = tmp_path / "data" / "episode_ep-export.json"
        assert data_file.exists()
        data = json.loads(data_file.read_text())
        assert len(data) == 3

        # Check info.json.
        info_file = tmp_path / "meta" / "info.json"
        assert info_file.exists()
        info = json.loads(info_file.read_text())
        assert info["fps"] == 30.0

        # Check episode manifest.
        manifest = tmp_path / "meta" / "episodes.jsonl"
        assert manifest.exists()

    @pytest.mark.asyncio
    async def test_export_episode_parquet_when_available(self, tmp_path: Path) -> None:
        """Test export with pyarrow available."""
        pytest.importorskip("pyarrow")

        exporter = EpisodeExporter(tmp_path, fps=15.0)
        episode = self._make_episode(steps=5)
        result = await exporter.export_episode(episode)

        assert result["format"] == "parquet"
        parquet_file = tmp_path / "data" / "episode_ep-export.parquet"
        assert parquet_file.exists()

    @pytest.mark.asyncio
    async def test_export_trajectory_raw(self, tmp_path: Path) -> None:
        """Test exporting a raw PhysicalTrajectory."""
        exporter = EpisodeExporter(tmp_path)
        trajectory = PhysicalTrajectory(
            trajectory_id="traj-raw",
            device_id="robot.arm0",
            goal="demo",
            steps=(_make_step(0), _make_step(1)),
            started_at=50.0,
            ended_at=55.0,
        )

        with patch("leapflow.robot.episode_export._PYARROW_AVAILABLE", False):
            result = await exporter.export_trajectory(trajectory, task="demo_task")

        assert result["episode_id"] == "traj-raw"
        assert result["step_count"] == 2

    @pytest.mark.asyncio
    async def test_export_with_camera_frames_fallback(self, tmp_path: Path) -> None:
        """Test camera frame export when opencv is unavailable."""
        steps = (
            _make_step_with_camera(0, "/tmp/frame0.png"),
            _make_step_with_camera(1, "/tmp/frame1.png"),
        )
        trajectory = PhysicalTrajectory(
            trajectory_id="traj-cam",
            device_id="d",
            goal="cam_test",
            steps=steps,
            started_at=0.0,
            ended_at=1.0,
        )
        episode = PhysicalEpisode(
            episode_id="ep-cam",
            device_id="d",
            task="cam_test",
            trajectory=trajectory,
            initial_state={},
            final_state={},
        )
        exporter = EpisodeExporter(tmp_path)

        with patch("leapflow.robot.episode_export._PYARROW_AVAILABLE", False), \
             patch("leapflow.robot.episode_export._CV2_AVAILABLE", False):
            result = await exporter.export_episode(episode)

        # Check frame refs file was written.
        refs_file = tmp_path / "videos" / "episode_ep-cam" / "cam0_refs.json"
        assert refs_file.exists()
        refs = json.loads(refs_file.read_text())
        assert len(refs) == 2

    @pytest.mark.asyncio
    async def test_episodes_exported_counter(self, tmp_path: Path) -> None:
        exporter = EpisodeExporter(tmp_path)
        assert exporter.episodes_exported == 0

        with patch("leapflow.robot.episode_export._PYARROW_AVAILABLE", False):
            await exporter.export_episode(self._make_episode(1))
        assert exporter.episodes_exported == 1

    @pytest.mark.asyncio
    async def test_empty_episode_export(self, tmp_path: Path) -> None:
        """Export an episode with zero steps."""
        trajectory = PhysicalTrajectory(
            trajectory_id="traj-empty",
            device_id="d",
            goal="nothing",
            steps=(),
        )
        episode = PhysicalEpisode(
            episode_id="ep-empty",
            device_id="d",
            task="empty",
            trajectory=trajectory,
            initial_state={},
            final_state={},
        )
        exporter = EpisodeExporter(tmp_path)

        with patch("leapflow.robot.episode_export._PYARROW_AVAILABLE", False):
            result = await exporter.export_episode(episode)

        assert result["step_count"] == 0
        data_file = tmp_path / "data" / "episode_ep-empty.json"
        assert data_file.exists()
        assert json.loads(data_file.read_text()) == []
