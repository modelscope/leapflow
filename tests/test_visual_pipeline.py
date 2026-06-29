"""Scenario-based tests for the video-first perception pipeline."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from leapflow.domain.events import SystemEvent
from leapflow.domain.trajectory import RecordingMode
from leapflow.llm.base import LLMChatResponse
from leapflow.perception.types import (
    MacroAnalysisResult,
    TimelineMarker,
    VideoAction,
    VideoSegment,
)
from leapflow.perception.video.analyzer import VideoAnalyzer
from leapflow.perception.video.prompts import (
    AnalysisPromptStrategy,
    DefaultAnalysisPrompts,
    VLMMessageBuilder,
)
from leapflow.perception.video.segmenter import AnalysisSegment, VideoSegmenter
from leapflow.perception.video.timeline import (
    SignalTimeline,
    TimelineReader,
    TimelineWriter,
)


# ── Fixtures ──


def _make_segment(
    sid: str = "seg_000",
    session: str = "sess_001",
    start: float = 0.0,
    duration: float = 120.0,
    fps: float = 5.0,
) -> VideoSegment:
    return VideoSegment(
        segment_id=sid,
        session_id=session,
        file_path=Path(f"/tmp/test/{session}/{sid}.mp4"),
        start_time=start,
        end_time=start + duration,
        duration=duration,
        fps=fps,
        resolution=(960, 540),
        codec="h264",
        file_size_bytes=1024 * 1024,
    )


def _make_markers(count: int = 10, start: float = 0.0, gap: float = 5.0) -> List[TimelineMarker]:
    markers = []
    for i in range(count):
        markers.append(TimelineMarker(
            timestamp=start + i * gap,
            channel="click" if i % 3 else "app_switch",
            action=f"action_{i}",
            app=f"app_{i % 3}",
            coordinates=(100 + i * 10, 200),
        ))
    return markers


# ── RecordingMode tests ──


def test_recording_mode_video() -> None:
    """RecordingMode.VIDEO has expected properties."""
    mode = RecordingMode.VIDEO
    assert mode.uses_video is True
    assert mode.skip_structural_events is False
    assert mode.needs_visual_polling is False


def test_recording_mode_from_str() -> None:
    """RecordingMode.from_str parses 'video' correctly."""
    assert RecordingMode.from_str("video") == RecordingMode.VIDEO
    assert RecordingMode.from_str("VIDEO") == RecordingMode.VIDEO
    assert RecordingMode.from_str("unknown") == RecordingMode.DEFAULT


# ── SignalTimeline tests ──


def test_timeline_record_and_compress() -> None:
    """Timeline records markers and compresses them."""
    timeline = SignalTimeline()
    timeline.set_start_time(100.0)

    for m in _make_markers(5, start=100.0, gap=2.0):
        with timeline._lock:
            timeline._markers.append(m)

    compressed = timeline.compress(max_markers=100)
    assert len(compressed) > 0
    assert all(isinstance(m, TimelineMarker) for m in compressed)


def test_timeline_format_for_prompt() -> None:
    """Timeline formats markers as readable text."""
    timeline = SignalTimeline()
    timeline.set_start_time(0.0)

    with timeline._lock:
        timeline._markers.extend(_make_markers(3, start=0.0, gap=5.0))

    text = timeline.format_for_prompt()
    assert "[00:" in text
    assert "app_switch" in text or "click" in text


def test_timeline_empty_format() -> None:
    """Empty timeline returns sentinel text."""
    tl = SignalTimeline()
    assert tl.format_for_prompt() == "(no events recorded)"


def test_timeline_keyboard_merge() -> None:
    """Consecutive keyboard markers are merged into a single typed marker."""
    tl = SignalTimeline()
    for i in range(5):
        with tl._lock:
            tl._markers.append(TimelineMarker(
                timestamp=float(i),
                channel="keyboard",
                action="type",
                app="TextEdit",
                payload_digest=chr(65 + i),
            ))
    compressed = tl.compress()
    typed = [m for m in compressed if "typed" in m.action]
    assert len(typed) == 1
    assert "ABCDE" in typed[0].action


# ── VideoSegmenter tests ──


def test_segmenter_single_segment() -> None:
    """Single short segment is returned as-is."""
    seg = _make_segment(duration=60.0)
    markers = _make_markers(5, start=0.0, gap=5.0)

    segmenter = VideoSegmenter(min_segment_s=10.0)
    result = segmenter.segment([seg], markers)

    assert len(result) >= 1
    assert all(isinstance(r, AnalysisSegment) for r in result)
    total = sum(r.duration for r in result)
    assert abs(total - 60.0) < 1.0


def test_segmenter_idle_gap_split() -> None:
    """A long idle gap triggers segment splitting."""
    seg = _make_segment(duration=120.0)
    markers = [
        TimelineMarker(timestamp=10.0, channel="click", action="click", app="App1"),
        TimelineMarker(timestamp=15.0, channel="click", action="click", app="App1"),
        TimelineMarker(timestamp=80.0, channel="click", action="click", app="App1"),
        TimelineMarker(timestamp=85.0, channel="click", action="click", app="App1"),
    ]

    segmenter = VideoSegmenter(idle_gap_s=15.0, min_segment_s=5.0)
    result = segmenter.segment([seg], markers)

    assert len(result) >= 2


def test_segmenter_merge_short() -> None:
    """Very short segments are merged with neighbours."""
    seg = _make_segment(duration=20.0)
    markers = _make_markers(2, start=0.0, gap=5.0)

    segmenter = VideoSegmenter(min_segment_s=30.0)
    result = segmenter.segment([seg], markers)

    assert len(result) == 1


# ── VideoSegment data type tests ──


def test_video_segment_frozen() -> None:
    """VideoSegment is frozen (immutable)."""
    seg = _make_segment()
    with pytest.raises(AttributeError):
        seg.segment_id = "changed"


def test_video_action_frozen() -> None:
    """VideoAction is frozen (immutable)."""
    action = VideoAction(
        action_name="click_button",
        description="Clicked save",
        start_time=10.0,
        end_time=12.0,
        app="TextEdit",
        confidence=0.9,
        analysis_level=1,
    )
    with pytest.raises(AttributeError):
        action.action_name = "changed"


def test_macro_analysis_result_defaults() -> None:
    """MacroAnalysisResult has sane defaults."""
    r = MacroAnalysisResult()
    assert r.actions == []
    assert r.overall_goal == ""
    assert r.detail_requests == []


# ── Integration tests for Tasks #60-#66 ──


def _make_analysis_segment(
    seg: Optional[VideoSegment] = None,
    start_offset: float = 0.0,
    end_offset: float = 60.0,
    app_summary: Optional[Dict[str, float]] = None,
) -> AnalysisSegment:
    """Helper to build an AnalysisSegment for analyzer tests."""
    return AnalysisSegment(
        segment=seg or _make_segment(duration=end_offset),
        start_offset=start_offset,
        end_offset=end_offset,
        app_summary=app_summary or {"TextEdit": 30.0, "Finder": 20.0},
    )


def _make_system_event(
    event_type: str = "ui.action",
    *,
    sub_action: str = "click",
    app_name: str = "TextEdit",
    ts: float = 1.0,
    x: int = 100,
    y: int = 200,
) -> SystemEvent:
    """Helper to create a SystemEvent suitable for timeline recording."""
    payload: Dict[str, Any] = {"sub_action": sub_action, "app_name": app_name}
    if x is not None:
        payload["x"] = x
        payload["y"] = y
    return SystemEvent(
        event_type=event_type,
        source=app_name,
        payload=payload,
        timestamp=ts,
    )


# ── 1. test_video_mode_analyze_routing ──


async def test_video_mode_analyze_routing(trajectory_store) -> None:
    """ImitationPipeline.analyze() in VIDEO mode routes to _build_episodes_from_video."""
    from leapflow.analysis.pipeline import ImitationPipeline

    # Create a minimal trajectory so analyze() has something to load
    from leapflow.domain.trajectory import (
        ActionType, Episode, RawAction, StateSnapshot, Trajectory, TrajectoryStep,
    )
    import time as _time
    now = _time.time()
    traj = Trajectory(
        trajectory_id="traj_video_test",
        steps=[
            TrajectoryStep(
                action=RawAction(
                    action_type=ActionType.UI_CLICK,
                    timestamp=now + i,
                    target=f"btn_{i}",
                ),
                state=StateSnapshot(timestamp=now + i),
            )
            for i in range(3)
        ],
    )
    trajectory_store.save_trajectory(traj)

    # Inject video actions
    video_actions = [
        VideoAction(
            action_name="open_file",
            description="Opened document",
            start_time=now,
            end_time=now + 5,
            app="TextEdit",
            confidence=0.9,
            analysis_level=1,
        ),
        VideoAction(
            action_name="edit_text",
            description="Typed paragraph",
            start_time=now + 6,
            end_time=now + 15,
            app="TextEdit",
            confidence=0.85,
            analysis_level=1,
        ),
    ]

    pipeline = ImitationPipeline(
        store=trajectory_store,
        recording_mode=RecordingMode.VIDEO,
    )
    pipeline._extracted_video_actions = video_actions

    episodes = await pipeline.analyze("traj_video_test", goal="edit document")

    assert len(episodes) >= 1, "VIDEO mode should produce at least one episode"
    for ep in episodes:
        for sa in ep.semantic_actions:
            assert sa.parameters.get("_source") == "video", (
                f"Expected _source='video' in parameters, got {sa.parameters}"
            )
            # Verify all required metadata fields are present
            assert "_analysis_level" in sa.parameters, (
                f"Missing _analysis_level in parameters: {sa.parameters}"
            )
            assert "_start_time" in sa.parameters, (
                f"Missing _start_time in parameters: {sa.parameters}"
            )
            assert "_end_time" in sa.parameters, (
                f"Missing _end_time in parameters: {sa.parameters}"
            )
            assert "_corroborating_events" in sa.parameters, (
                f"Missing _corroborating_events in parameters: {sa.parameters}"
            )

    # Verify DEFAULT mode does NOT use video path when no video actions
    pipeline_default = ImitationPipeline(
        store=trajectory_store,
        recording_mode=RecordingMode.DEFAULT,
    )
    pipeline_default._extracted_video_actions = []
    episodes_default = await pipeline_default.analyze("traj_video_test")
    for ep in episodes_default:
        for sa in ep.semantic_actions:
            assert sa.parameters.get("_source") != "video", (
                "DEFAULT mode should not produce video-sourced actions"
            )


# ── 2. test_eventbus_timeline_subscription ──


def test_eventbus_timeline_subscription() -> None:
    """EventBus events flow into SignalTimeline via subscribe."""
    timeline = SignalTimeline()
    timeline.set_start_time(0.0)

    events = [
        _make_system_event(sub_action="click", app_name="Finder", ts=1.0),
        _make_system_event(
            event_type="app.focus_change",
            sub_action="", app_name="Safari", ts=2.0,
        ),
        _make_system_event(sub_action="click", app_name="Safari", ts=3.0),
    ]

    # Simulate EventBus.subscribe + dispatch
    for evt in events:
        timeline.record_event(evt)

    with timeline._lock:
        count = len(timeline._markers)

    assert count == 3, f"Expected 3 markers, got {count}"

    compressed = timeline.compress(max_markers=100)
    channels = {m.channel for m in compressed}
    assert "click" in channels or "app_switch" in channels, (
        f"Expected click or app_switch channels, got {channels}"
    )


# ── 3. test_l3_analysis_with_mock_extractor ──


async def test_l3_analysis_with_mock_extractor() -> None:
    """L3 analysis triggers when frame_extractor is present and l3_enabled=True."""
    mock_vlm = AsyncMock()
    # L1 returns JSON with frame_requests
    l1_response = LLMChatResponse(content='{\n'
        '  "steps": [{"action": "click", "description": "Clicked button", '
        '"start_time": 5.0, "end_time": 10.0, "app": "TextEdit", "confidence": 0.8}],\n'
        '  "overall_goal": "test",\n'
        '  "detail_requests": [],\n'
        '  "frame_requests": [{"timestamp": 7.0, "reason": "check OCR"}]\n'
        '}')
    # L3 returns JSON with analysis
    l3_response = LLMChatResponse(content='{"content_type": "dialog", '
        '"text": "Save As", "ui_elements": ["Save", "Cancel"], "confidence": 0.9}')
    mock_vlm.achat = AsyncMock(side_effect=[l1_response, l3_response])

    mock_extractor = AsyncMock()
    mock_extractor.extract = AsyncMock(return_value=b"fake_frame_data")

    # Create a stub prompt strategy to avoid get_settings dependency
    class StubPrompts:
        def build_l1_messages(self, **kw): return [{"role": "user", "content": "l1"}]
        def build_l2_messages(self, **kw): return [{"role": "user", "content": "l2"}]
        def build_l3_messages(self, **kw): return [{"role": "user", "content": "l3"}]

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._vlm = mock_vlm
    analyzer._l2_enabled = False
    analyzer._l3_enabled = True
    analyzer._max_l2_requests = 10
    analyzer._max_l3_requests = 5
    analyzer._frame_extractor = mock_extractor
    analyzer._prompts = StubPrompts()
    analyzer._vlm_max_retries = 0
    analyzer._vlm_retry_backoff_s = 0.0

    timeline = SignalTimeline()
    seg = _make_analysis_segment(end_offset=60.0)

    actions = await analyzer.analyze([seg], timeline)

    l3_actions = [a for a in actions if a.analysis_level == 3]
    assert len(l3_actions) >= 1, "Expected at least one L3 action"
    assert mock_extractor.extract.call_count <= 5, (
        f"FrameExtractor.extract called {mock_extractor.extract.call_count} times, "
        f"should be <= max_l3_requests=5"
    )


# ── 4. test_l3_disabled_without_extractor ──


async def test_l3_disabled_without_extractor() -> None:
    """L3 gracefully skips when frame_extractor=None — no exceptions."""
    mock_vlm = AsyncMock()
    l1_response = LLMChatResponse(content='{\n'
        '  "steps": [{"action": "click", "description": "Clicked", '
        '"start_time": 0.0, "end_time": 5.0, "confidence": 0.7}],\n'
        '  "overall_goal": "test",\n'
        '  "detail_requests": [],\n'
        '  "frame_requests": [{"timestamp": 3.0, "reason": "need OCR"}]\n'
        '}')
    mock_vlm.achat = AsyncMock(return_value=l1_response)

    class StubPrompts:
        def build_l1_messages(self, **kw): return [{"role": "user", "content": "l1"}]
        def build_l2_messages(self, **kw): return [{"role": "user", "content": "l2"}]
        def build_l3_messages(self, **kw): return [{"role": "user", "content": "l3"}]

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._vlm = mock_vlm
    analyzer._l2_enabled = False
    analyzer._l3_enabled = True
    analyzer._max_l2_requests = 10
    analyzer._max_l3_requests = 5
    analyzer._frame_extractor = None  # No extractor!
    analyzer._prompts = StubPrompts()
    analyzer._vlm_max_retries = 0
    analyzer._vlm_retry_backoff_s = 0.0

    timeline = SignalTimeline()
    seg = _make_analysis_segment(end_offset=30.0)

    actions = await analyzer.analyze([seg], timeline)

    # Should return actions but none at L3
    for a in actions:
        assert a.analysis_level in (1, 2), (
            f"Expected level 1 or 2 without extractor, got {a.analysis_level}"
        )


# ── 5. test_vlm_failure_graceful_degradation ──


async def test_vlm_failure_graceful_degradation(caplog) -> None:
    """VLM failure returns empty list (no crash) and logs warning."""
    mock_vlm = AsyncMock()
    mock_vlm.achat = AsyncMock(side_effect=RuntimeError("VLM connection refused"))

    class StubPrompts:
        def build_l1_messages(self, **kw): return [{"role": "user", "content": "l1"}]
        def build_l2_messages(self, **kw): return [{"role": "user", "content": "l2"}]
        def build_l3_messages(self, **kw): return [{"role": "user", "content": "l3"}]

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._vlm = mock_vlm
    analyzer._l2_enabled = False
    analyzer._l3_enabled = False
    analyzer._max_l2_requests = 0
    analyzer._max_l3_requests = 0
    analyzer._frame_extractor = None
    analyzer._prompts = StubPrompts()
    analyzer._vlm_max_retries = 1
    analyzer._vlm_retry_backoff_s = 0.0

    timeline = SignalTimeline()
    seg = _make_analysis_segment(end_offset=30.0)

    with caplog.at_level(logging.WARNING, logger="leapflow.perception.video.analyzer"):
        actions = await analyzer.analyze([seg], timeline)

    assert actions == [], f"Expected empty list on VLM failure, got {actions}"
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("failed" in m.lower() or "l1_failed" in m.lower() for m in warning_msgs), (
        f"Expected warning about VLM failure, got: {warning_msgs}"
    )


# ── 6. test_vlm_retry_with_backoff ──


async def test_vlm_retry_with_backoff() -> None:
    """VLM retries N-1 failures then succeeds on Nth attempt."""
    success_response = LLMChatResponse(content='{\n'
        '  "steps": [{"action": "save", "description": "Saved file", '
        '"start_time": 0.0, "end_time": 2.0, "confidence": 0.9}],\n'
        '  "overall_goal": "save work",\n'
        '  "detail_requests": [],\n'
        '  "frame_requests": []\n'
        '}')

    call_count = 0

    async def flaky_achat(messages, *, stream=True, **kw):
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # Fail first 2 attempts
            raise RuntimeError("transient error")
        return success_response

    mock_vlm = AsyncMock()
    mock_vlm.achat = flaky_achat

    class StubPrompts:
        def build_l1_messages(self, **kw): return [{"role": "user", "content": "l1"}]
        def build_l2_messages(self, **kw): return [{"role": "user", "content": "l2"}]
        def build_l3_messages(self, **kw): return [{"role": "user", "content": "l3"}]

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._vlm = mock_vlm
    analyzer._l2_enabled = False
    analyzer._l3_enabled = False
    analyzer._max_l2_requests = 0
    analyzer._max_l3_requests = 0
    analyzer._frame_extractor = None
    analyzer._prompts = StubPrompts()
    analyzer._vlm_max_retries = 2  # up to 3 total attempts
    analyzer._vlm_retry_backoff_s = 0.0  # no wait in tests

    timeline = SignalTimeline()
    seg = _make_analysis_segment(end_offset=10.0)

    actions = await analyzer.analyze([seg], timeline)

    assert len(actions) >= 1, "Expected successful action after retry"
    assert call_count == 3, (
        f"Expected 3 VLM calls (2 failures + 1 success), got {call_count}"
    )


# ── 7. test_recorder_start_await_no_race ──


async def test_recorder_start_await_no_race(trajectory_store) -> None:
    """start_recording awaits VideoRecorder.start (no race condition)."""
    from leapflow.analysis.pipeline import ImitationPipeline

    start_completed = False

    async def slow_start(session_id: str) -> None:
        nonlocal start_completed
        await asyncio.sleep(0.05)
        start_completed = True

    mock_recorder = MagicMock()
    mock_recorder.start = slow_start
    mock_recorder.active = True

    pipeline = ImitationPipeline(
        store=trajectory_store,
        recording_mode=RecordingMode.VIDEO,
        video_recorder=mock_recorder,
    )

    await pipeline.start_recording(goal="test")

    assert start_completed, (
        "start_recording should await VideoRecorder.start before returning"
    )


# ── 8. test_recorder_start_timeout_fallback ──


async def test_recorder_start_timeout_fallback(trajectory_store, caplog) -> None:
    """start_recording handles timeout gracefully (no crash, logs warning)."""
    from leapflow.analysis.pipeline import ImitationPipeline

    async def hanging_start(session_id: str) -> None:
        await asyncio.sleep(10.0)  # Will be cancelled by timeout

    mock_recorder = MagicMock()
    mock_recorder.start = hanging_start
    mock_recorder.active = False

    pipeline = ImitationPipeline(
        store=trajectory_store,
        recording_mode=RecordingMode.VIDEO,
        video_recorder=mock_recorder,
    )

    with patch("leapflow.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(video_start_timeout_s=0.1)
        with caplog.at_level(logging.WARNING):
            tid = await pipeline.start_recording(goal="test")

    assert tid, "start_recording should return a trajectory ID even on timeout"
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("timed out" in m.lower() or "timeout" in m.lower() for m in warning_msgs), (
        f"Expected timeout warning, got: {warning_msgs}"
    )


# ── 9. test_timeline_overflow_warning ──


def test_timeline_overflow_warning(caplog) -> None:
    """Marker overflow triggers log warnings (first + every 100th)."""
    timeline = SignalTimeline(max_markers=10)

    with caplog.at_level(logging.WARNING, logger="leapflow.perception.video.timeline"):
        for i in range(110):
            evt = _make_system_event(ts=float(i), app_name=f"App_{i}")
            timeline.record_event(evt)

    with timeline._lock:
        stored = len(timeline._markers)

    assert stored <= 10, f"Expected at most 10 markers, got {stored}"

    overflow_warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "overflow" in r.message.lower()
    ]
    assert len(overflow_warnings) >= 2, (
        f"Expected at least 2 overflow warnings (1st + 100th), got {len(overflow_warnings)}"
    )


# ── 10. test_timeline_thread_safety ──


def test_timeline_thread_safety() -> None:
    """Concurrent record_event and set_start_time calls produce no errors."""
    timeline = SignalTimeline(max_markers=5000)
    errors: List[Exception] = []
    n_events_per_thread = 200
    n_threads = 5

    def writer(thread_id: int) -> None:
        try:
            for i in range(n_events_per_thread):
                evt = _make_system_event(
                    ts=float(thread_id * 1000 + i),
                    app_name=f"App_{thread_id}",
                )
                timeline.record_event(evt)
        except Exception as exc:
            errors.append(exc)

    def time_setter() -> None:
        try:
            for i in range(100):
                timeline.set_start_time(float(i))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    threads.append(threading.Thread(target=time_setter))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Thread safety violation: {errors}"

    with timeline._lock:
        actual = len(timeline._markers)

    assert actual <= n_events_per_thread * n_threads, (
        f"Marker count {actual} exceeds maximum possible {n_events_per_thread * n_threads}"
    )
    assert actual > 0, "Expected at least some markers to be recorded"


# ── 11. test_prompt_strategy_extensibility ──


async def test_prompt_strategy_extensibility() -> None:
    """Custom AnalysisPromptStrategy can be injected and used by VideoAnalyzer."""

    class CustomPrompts:
        def __init__(self):
            self.l1_called = False
            self.l2_called = False
            self.l3_called = False

        def build_l1_messages(self, **kw):
            self.l1_called = True
            return [{"role": "user", "content": "custom_l1"}]

        def build_l2_messages(self, **kw):
            self.l2_called = True
            return [{"role": "user", "content": "custom_l2"}]

        def build_l3_messages(self, **kw):
            self.l3_called = True
            return [{"role": "user", "content": "custom_l3"}]

    custom = CustomPrompts()
    mock_vlm = AsyncMock()
    mock_vlm.achat = AsyncMock(return_value=LLMChatResponse(
        content='{"steps": [{"action": "test", "description": "t", '
                '"start_time": 0, "end_time": 1, "confidence": 0.5}], '
                '"overall_goal": "g", "detail_requests": [], "frame_requests": []}',
    ))

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._vlm = mock_vlm
    analyzer._l2_enabled = False
    analyzer._l3_enabled = False
    analyzer._max_l2_requests = 0
    analyzer._max_l3_requests = 0
    analyzer._frame_extractor = None
    analyzer._prompts = custom
    analyzer._vlm_max_retries = 0
    analyzer._vlm_retry_backoff_s = 0.0

    timeline = SignalTimeline()
    seg = _make_analysis_segment(end_offset=10.0)

    await analyzer.analyze([seg], timeline)

    assert custom.l1_called, "Custom prompt strategy build_l1_messages was not called"


# ── 12. test_vlm_message_builder_replacement ──


def test_vlm_message_builder_replacement() -> None:
    """Custom VLMMessageBuilder is used by DefaultAnalysisPrompts."""

    class CustomBuilder:
        def __init__(self):
            self.video_calls = 0
            self.image_calls = 0

        def build_video_message(self, video_path, prompt, *, system=""):
            self.video_calls += 1
            return [{"role": "user", "content": f"custom_video: {prompt[:20]}"}]

        def build_image_message(self, image_data, prompt, *, system=""):
            self.image_calls += 1
            return [{"role": "user", "content": f"custom_image: {prompt[:20]}"}]

    builder = CustomBuilder()
    prompts = DefaultAnalysisPrompts(message_builder=builder)

    l1_msgs = prompts.build_l1_messages(
        video_path="/tmp/test.mp4",
        duration=60.0,
        top_apps="TextEdit",
        timeline_text="(none)",
        goal="test",
    )
    assert builder.video_calls == 1, "Custom builder.build_video_message not called for L1"
    assert "custom_video" in l1_msgs[0]["content"]

    l3_msgs = prompts.build_l3_messages(
        image_data=b"fake_png",
        timestamp=5.0,
        reason="OCR check",
        nearby_events="none",
    )
    assert builder.image_calls == 1, "Custom builder.build_image_message not called for L3"
    assert "custom_image" in l3_msgs[0]["content"]


# ── 13. test_timeline_writer_reader_isolation ──


def test_timeline_writer_reader_isolation() -> None:
    """SignalTimeline satisfies both TimelineWriter and TimelineReader protocols."""
    tl = SignalTimeline()

    # Verify writer methods exist (TimelineWriter protocol)
    writer_methods = ["set_start_time", "record_event", "clear"]
    for method in writer_methods:
        assert callable(getattr(tl, method, None)), (
            f"SignalTimeline missing TimelineWriter method: {method}"
        )

    # Verify reader methods exist (TimelineReader protocol)
    reader_methods = ["markers_in_range", "compress", "format_for_prompt"]
    for method in reader_methods:
        assert callable(getattr(tl, method, None)), (
            f"SignalTimeline missing TimelineReader method: {method}"
        )

    # Verify writer and reader work independently:
    # write via writer interface, read via reader interface
    tl.set_start_time(0.0)
    evt = _make_system_event(ts=1.0, app_name="TestApp")
    tl.record_event(evt)

    # Reader interface should see the written data
    markers = tl.markers_in_range(0.0, 2.0)
    assert len(markers) == 1, "Reader should see marker written via writer interface"
    text = tl.format_for_prompt()
    assert text != "(no events recorded)", "Reader format should reflect written data"


# ── 14. test_recording_mode_video_property ──


def test_recording_mode_video_property() -> None:
    """RecordingMode.VIDEO.uses_video is True; other modes are False."""
    assert RecordingMode.VIDEO.uses_video is True
    assert RecordingMode.DEFAULT.uses_video is False
    assert RecordingMode.VISION_ONLY.uses_video is False
