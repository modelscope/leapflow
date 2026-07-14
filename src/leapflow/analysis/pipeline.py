"""Imitation learning pipeline — orchestrates the full observe → distill workflow.

Coordinates:
    DemonstrationRecorder → SegmentDetector → ActionAbstractor → SkillDistiller

This is the single entry point for the engine/skills layer to interact with
the imitation learning subsystem.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from leapflow.analysis.abstractor import ActionAbstractor
from leapflow.recording.recorder import DemonstrationRecorder
from leapflow.analysis.segmenter import SegmentDetector
from leapflow.storage.trajectory_store import TrajectoryStore
from leapflow.domain.trajectory import Episode, RecordingMode, Trajectory
from leapflow.domain.skill_types import DistillationCandidate
from leapflow.learning.distiller import LLMSkillDistiller, SkillDistiller
from leapflow.platform.client import fire_and_forget
from leapflow.platform.protocol import HostRpc

if TYPE_CHECKING:
    from leapflow.analysis.intent_inferrer import IntentInferrer
    from leapflow.perception.session import PerceptionSession
    from leapflow.perception.types import VideoAction
    from leapflow.perception.video.analyzer import VideoAnalyzer
    from leapflow.perception.video.recorder import TrajectoryRecorder
    from leapflow.perception.video.segmenter import VideoSegmenter
    from leapflow.perception.video.timeline import SignalTimeline
    from leapflow.platform.event_bus import EventBus
    from leapflow.platform.observers import RecordingProfile
    from leapflow.platform.observers.daemon import ObservationDaemon
    from leapflow.learning.codegen import CodeGenContext, GeneratedSkill, SkillCodeGenerator

CandidatesCallback = Callable[[List[DistillationCandidate], List[Episode]], None]
ProgressCallback = Optional[Callable[[str, int, int], None]]
StopProgressCallback = Optional[Callable[[str, int, int], None]]

logger = logging.getLogger(__name__)


def _workflow_graph_to_mermaid(graph: Any) -> str:
    """Convert a WorkflowGraph to compact Mermaid string for Episode persistence."""
    lines = ["graph LR"]
    for node in getattr(graph, "nodes", []):
        label = getattr(node, "app_bundle", "") or getattr(node, "node_id", "?")
        role = getattr(node, "role", None)
        role_suffix = f" ({role.value})" if role and hasattr(role, "value") else ""
        nid = getattr(node, "node_id", "?")
        lines.append(f"    {nid}[{label}{role_suffix}]")
    for edge in getattr(graph, "edges", []):
        lines.append(f"    {edge.from_node_id} --> {edge.to_node_id}")
    return "\n".join(lines)


class ImitationPipeline:
    """End-to-end imitation learning pipeline.

    Manages the lifecycle of recording, analysis, and skill distillation.
    Designed to be instantiated once and shared across the application.
    """

    def __init__(
        self,
        store: TrajectoryStore,
        *,
        distiller: SkillDistiller | None = None,
        segmenter: SegmentDetector | None = None,
        abstractor: ActionAbstractor | None = None,
        intent_inferrer: Optional["IntentInferrer"] = None,
        user_id: str = "default",
        codegen: Optional["SkillCodeGenerator"] = None,
        perception_session: Optional["PerceptionSession"] = None,
        goal_relevance_threshold: float = 0.1,
        attention_filters: "Optional[list]" = None,
        surprise_annotator: "Optional[Any]" = None,
        rpc: Optional[HostRpc] = None,
        event_bus: "Optional[EventBus]" = None,
        text_capture_enabled: bool = True,
        text_capture_exclude_apps: "tuple[str, ...] | list[str]" = (),
        text_capture_secure_roles: "tuple[str, ...] | list[str]" = ("AXSecureTextField",),
        text_capture_max_length: int = 500,
        clipboard_max_length: int = 1024,
        recording_mode: RecordingMode = RecordingMode.DEFAULT,
        mhms_fusion_enabled: bool = False,
        video_recorder: "Optional[TrajectoryRecorder]" = None,
        video_analyzer: "Optional[VideoAnalyzer]" = None,
        video_segmenter: "Optional[VideoSegmenter]" = None,
        signal_timeline: "Optional[SignalTimeline]" = None,
        observation_daemon: "Optional[ObservationDaemon]" = None,
        recording_profile: "Optional[RecordingProfile]" = None,
    ) -> None:
        self._store = store
        self._segmenter = segmenter or SegmentDetector()
        self._abstractor = abstractor or ActionAbstractor()
        self._distiller = distiller or SkillDistiller()
        self._intent_inferrer = intent_inferrer
        self._perception_session = perception_session
        self._recording_mode = recording_mode
        self._recorder = DemonstrationRecorder(
            store, user_id=user_id,
            perception_session=perception_session,
            attention_filters=attention_filters,
            surprise_annotator=surprise_annotator,
            text_capture_enabled=text_capture_enabled,
            text_capture_exclude_apps=text_capture_exclude_apps,
            text_capture_secure_roles=text_capture_secure_roles,
            text_capture_max_length=text_capture_max_length,
            clipboard_max_length=clipboard_max_length,
            recording_mode=recording_mode,
        )
        self._codegen = codegen
        self._rpc = rpc
        self._event_bus = event_bus
        self._goal_relevance_threshold = goal_relevance_threshold
        self._on_candidates_ready: Optional[CandidatesCallback] = None
        self._progress_callback: ProgressCallback = None
        self._stop_progress_callback: StopProgressCallback = None
        self._mhms_fusion_enabled = mhms_fusion_enabled

        self._video_recorder = video_recorder
        self._video_analyzer = video_analyzer
        self._video_segmenter = video_segmenter
        self._signal_timeline = signal_timeline
        self._observation_daemon = observation_daemon
        self._recording_profile = recording_profile
        self._daemon_started_for_recording: bool = False
        self._extracted_video_actions: List[Any] = []
        self._video_available: bool = False
        self._current_goal = ""

    @property
    def recorder(self) -> DemonstrationRecorder:
        return self._recorder

    @property
    def store(self) -> TrajectoryStore:
        return self._store

    @property
    def progress_callback(self) -> ProgressCallback:
        return self._progress_callback

    @progress_callback.setter
    def progress_callback(self, cb: ProgressCallback) -> None:
        self._progress_callback = cb

    @property
    def stop_progress_callback(self) -> StopProgressCallback:
        return self._stop_progress_callback

    @stop_progress_callback.setter
    def stop_progress_callback(self, cb: StopProgressCallback) -> None:
        self._stop_progress_callback = cb

    def _report_stop(self, stage: str, current: int = 0, total: int = 0) -> None:
        if self._stop_progress_callback:
            self._stop_progress_callback(stage, current, total)

    def set_on_candidates_ready(self, callback: CandidatesCallback) -> None:
        """Register a callback invoked after distillation produces candidates."""
        self._on_candidates_ready = callback

    # ── Recording control ──

    def mark_control_input(self) -> None:
        """Signal that subsequent events are control input, not demonstration."""
        self._recorder.mark_control_input_start()

    def end_control_input(self) -> None:
        """Signal that control input has ended; resume normal recording."""
        self._recorder.end_control_input()

    async def start_recording(self, *, user_id: Optional[str] = None, goal: str = "") -> str:
        """Begin recording a demonstration. Returns trajectory ID."""
        self._current_goal = goal
        self._extracted_video_actions = []
        if self._event_bus is not None:
            self._event_bus.enable_reorder()
        tid = self._recorder.start(user_id=user_id)
        if goal:
            self._recorder.attention_context.seed_from_goal(goal)
        await self._ensure_observation_daemon()
        await self._apply_recording_profile()

        if self._recording_mode.uses_video and self._video_recorder:
            from leapflow.config import get_settings
            settings = get_settings()
            try:
                await asyncio.wait_for(
                    self._video_recorder.start(tid),
                    timeout=settings.video_start_timeout_s,
                )
                # VideoRecorder.start() catches internal RPC errors without
                # re-raising — check the actual active state to confirm success.
                self._video_available = self._video_recorder.active
                if not self._video_available:
                    logger.warning(
                        "Video recording start returned but recorder is not active, "
                        "continuing without video",
                    )
            except asyncio.TimeoutError:
                self._video_available = False
                logger.warning(
                    "Video recording start timed out after %.1fs, continuing without video",
                    settings.video_start_timeout_s,
                )
            except Exception as exc:
                self._video_available = False
                logger.warning(
                    "Video recording start failed: %s, continuing without video", exc,
                )
            if self._signal_timeline:
                import time as _time
                self._signal_timeline.set_start_time(_time.time())
        return tid

    def resume_recording(self, trajectory_id: str) -> str:
        """Resume recording on an existing trajectory. Returns trajectory ID."""
        traj = self._store.load_trajectory(trajectory_id)
        if traj is None:
            raise ValueError(f"Trajectory not found: {trajectory_id}")
        return self._recorder.resume_from(traj)

    async def stop_recording(self, *, discard: bool = False) -> Optional[Trajectory]:
        """Stop recording and return the completed trajectory.

        Args:
            discard: If True, skip VLM extraction / video analysis.
                     Used when the user abandons a learning session.
        """
        if self._event_bus is not None:
            self._report_stop("drain")
            await self._event_bus.disable_reorder()
        # Allow in-flight events to arrive after reorder buffer flush
        await asyncio.sleep(0.2)
        self._report_stop("save")
        traj = self._recorder.stop()
        await self._reset_recording_profile()

        if self._recording_mode.uses_video:
            await self._stop_video_recording(discard=discard)
        elif self._perception_session:
            await self._stop_screenshot_recording(discard=discard)

        return traj

    async def _stop_video_recording(self, *, discard: bool) -> None:
        """Finalize trajectory/video recording and run multi-scale analysis."""
        if not self._video_available:
            return
        if not self._video_recorder or not self._video_recorder.active:
            return

        trajectory_actions = self._video_recorder.load_trajectory_actions()
        if trajectory_actions:
            logger.info(
                "video_analysis: loaded %d trajectory actions", len(trajectory_actions),
            )

        self._report_stop("video_stop")
        logger.info("video_analysis: stopping trajectory recorder")
        segments = await self._video_recorder.stop()
        if discard or not segments:
            return

        logger.info("video_analysis: segmenting %d video segments", len(segments))
        if self._video_analyzer and self._video_segmenter and self._signal_timeline:
            self._report_stop("video_segment")
            await asyncio.sleep(0)
            markers = self._signal_timeline.compress()
            analysis_segs = self._video_segmenter.segment(segments, markers)
            if analysis_segs:
                logger.info("video_analysis: analyzing %d segments via VLM", len(analysis_segs))
                self._report_stop("video_analyze", 0, len(analysis_segs))
                try:
                    actions = await self._video_analyzer.analyze(
                        analysis_segs, self._signal_timeline,
                        goal=self._current_goal,
                        progress=lambda msg: self._report_stop(msg),
                        trajectory_actions=trajectory_actions,
                    )
                    self._extracted_video_actions = actions
                    logger.info("video_analysis: extracted %d video actions", len(actions))
                except Exception:
                    logger.warning("Video analysis failed", exc_info=True)

    async def _stop_screenshot_recording(self, *, discard: bool) -> None:
        """Legacy screenshot-mode finalization."""
        ps = self._perception_session
        if not ps:
            return
        self._report_stop("visual_stop")
        await ps.stop()
        if discard:
            return
        _MIN_VISUAL_FRAMES = 3
        fc = ps.frame_count
        if 0 < fc < _MIN_VISUAL_FRAMES:
            logger.warning(
                "Insufficient visual data (%d/%d frames) — skipping VLM extraction",
                fc, _MIN_VISUAL_FRAMES,
            )
        elif fc >= _MIN_VISUAL_FRAMES:
            self._report_stop("extract", 0, fc)
            try:
                await ps.extract(progress=self._stop_progress_callback)
                logger.info(
                    "perception.extraction complete: %d actions",
                    len(ps.extracted_actions),
                )
            except Exception as e:
                logger.warning("Perception extraction failed: %s", e)

    def pause_recording(self) -> None:
        self._recorder.pause()
        # Synchronously pause video recording (fire-and-forget async call)
        if self._video_recorder and self._video_recorder.active:
            fire_and_forget(self._video_recorder.pause())

    def unpause_recording(self) -> None:
        self._recorder.resume()
        # Synchronously resume video recording (fire-and-forget async call)
        if self._video_recorder and self._video_recorder.paused:
            fire_and_forget(self._video_recorder.resume())

    async def _ensure_observation_daemon(self) -> None:
        """Start ObservationDaemon on-demand if not already running.

        Enables OS-level signal capture (FS, focus, clipboard, input tap)
        even when observer_auto_start was disabled at init time.
        """
        if self._observation_daemon is not None:
            return
        if self._event_bus is None:
            return
        try:
            from leapflow.platform.observers import ObserverConfig
            from leapflow.platform.observers.daemon import ObservationDaemon
            daemon = ObservationDaemon(bus=self._event_bus, config=ObserverConfig())
            await daemon.start()
            self._observation_daemon = daemon
            self._daemon_started_for_recording = True
            logger.info("ObservationDaemon started on-demand for teach recording")
        except Exception:
            logger.debug("on-demand ObservationDaemon start failed", exc_info=True)

    async def _apply_recording_profile(self) -> None:
        """Switch observation daemon to high-fidelity recording mode."""
        if self._observation_daemon is None or self._recording_profile is None:
            return
        try:
            await self._observation_daemon.apply_profile(self._recording_profile)
        except Exception as e:
            logger.warning("apply_recording_profile failed: %s", e)

    async def _reset_recording_profile(self) -> None:
        """Restore observation daemon to idle-mode parameters."""
        if self._observation_daemon is None:
            return
        try:
            await self._observation_daemon.reset_profile()
        except Exception as e:
            logger.warning("reset_recording_profile failed: %s", e)
        if self._daemon_started_for_recording:
            try:
                await self._observation_daemon.stop()
            except Exception:
                logger.debug("on-demand ObservationDaemon stop failed", exc_info=True)
            self._observation_daemon = None
            self._daemon_started_for_recording = False

    # ── Video enrichment ──

    def _enrich_episodes_with_video(self, episodes: List[Episode], traj: Trajectory) -> None:
        """Merge video-derived actions into episode semantic_actions."""
        from leapflow.domain.trajectory import SemanticAction

        for ep in episodes:
            if not traj.steps:
                continue
            ep_start = traj.steps[ep.start_idx].action.timestamp if ep.start_idx < len(traj.steps) else 0
            ep_end = traj.steps[min(ep.end_idx, len(traj.steps)) - 1].action.timestamp if ep.end_idx > 0 else 0
            overlapping = [
                va for va in self._extracted_video_actions
                if va.confidence > 0 and va.start_time < ep_end and va.end_time > ep_start
            ]
            if overlapping:
                video_sas = [
                    SemanticAction(
                        action_name=va.action_name,
                        description=va.description,
                        parameters={
                            "app": va.app,
                            "goal": va.goal,
                            "_source": "video",
                            "_analysis_level": va.analysis_level,
                            "_start_time": va.start_time,
                            "_end_time": va.end_time,
                            "_corroborating_events": va.corroborating_events,
                        },
                        confidence=va.confidence,
                    )
                    for va in overlapping
                ]
                ep.semantic_actions = _merge_semantic_actions(ep.semantic_actions, video_sas)
                self._store.save_episode(ep)

    # ── Video Episode Construction ──

    async def _build_episodes_from_video(
        self,
        traj: Trajectory,
        video_actions: "List[VideoAction]",
        *,
        goal: str = "",
    ) -> List[Episode]:
        """Build Episode list from video-derived actions.

        Strategy: group consecutive VideoActions by application context and
        temporal proximity into coherent Episodes. Falls back to the
        traditional segmenter path when *video_actions* is empty.
        """
        if not video_actions:
            # Graceful fallback — delegate to trajectory-based segmentation
            episodes = self._segmenter.segment(traj)
            for ep in episodes:
                steps = ep.steps_from(traj)
                ep.semantic_actions = self._abstractor.abstract(steps)
                self._store.save_episode(ep)
            return episodes

        from leapflow.config import get_settings
        from leapflow.domain.trajectory import SemanticAction

        settings = get_settings()
        # Time gap threshold for splitting episodes (config-driven, no hardcoding)
        app_gap_s: float = settings.video_segmenter_app_gap_s

        # Filter out zero-confidence actions (uncertain detections)
        valid_actions = [va for va in video_actions if va.confidence > 0]
        if not valid_actions:
            # All actions had zero confidence — fallback to segmenter
            logger.info("video_episode_build: all %d actions filtered (confidence=0), fallback", len(video_actions))
            episodes = self._segmenter.segment(traj)
            for ep in episodes:
                steps = ep.steps_from(traj)
                ep.semantic_actions = self._abstractor.abstract(steps)
                self._store.save_episode(ep)
            return episodes

        # Group video actions into episode buckets
        groups: List[List["VideoAction"]] = []
        current_group: List["VideoAction"] = [valid_actions[0]]

        for prev_va, va in zip(valid_actions[:-1], valid_actions[1:]):
            app_changed = va.app != prev_va.app and va.app and prev_va.app
            time_gap = va.start_time - prev_va.end_time
            if app_changed or time_gap > app_gap_s:
                groups.append(current_group)
                current_group = [va]
            else:
                current_group.append(va)
        groups.append(current_group)

        # Convert each group into an Episode
        episodes: List[Episode] = []
        for group in groups:
            semantic_actions = [
                SemanticAction(
                    action_name=va.action_name,
                    description=va.description,
                    parameters={
                        "app": va.app,
                        "goal": va.goal,
                        "_source": "video",
                        "_analysis_level": va.analysis_level,
                        "_start_time": va.start_time,
                        "_end_time": va.end_time,
                        "_corroborating_events": va.corroborating_events,
                    },
                    confidence=va.confidence,
                )
                for va in group
            ]

            # Derive app_sequence (ordered unique apps in this group)
            seen_apps: set[str] = set()
            app_sequence: List[str] = []
            for va in group:
                if va.app and va.app not in seen_apps:
                    seen_apps.add(va.app)
                    app_sequence.append(va.app)

            # Episode goal: prefer group-level goal, then overall goal
            ep_goal = group[0].goal or goal

            # Confidence: average of action confidences
            avg_confidence = (
                sum(va.confidence for va in group) / len(group)
                if group
                else 0.0
            )

            episode = Episode(
                episode_id=uuid.uuid4().hex[:16],
                trajectory_id=traj.trajectory_id,
                start_idx=0,
                end_idx=len(traj.steps),
                inferred_goal=ep_goal,
                app_sequence=app_sequence,
                semantic_actions=semantic_actions,
                confidence=avg_confidence,
            )
            self._store.save_episode(episode)
            episodes.append(episode)

        logger.info(
            "video_episode_build: %d actions (%d after confidence filter) -> %d episodes",
            len(video_actions),
            len(valid_actions),
            len(episodes),
        )
        return episodes

    # ── Analysis ──

    async def analyze(
        self, trajectory_id: str, *, goal: str = "", progress: ProgressCallback = None,
    ) -> List[Episode]:
        """Segment and abstract a trajectory into episodes.

        Loads the trajectory from the store, segments it into episodes,
        runs action abstraction on each episode, and persists the results.
        If ``goal`` is provided, it is used as a fallback inferred_goal for
        episodes that lack one after intent inference.
        """
        traj = self._store.load_trajectory(trajectory_id)
        if traj is None:
            logger.warning("trajectory not found: %s", trajectory_id)
            return []

        self._store.delete_episodes(trajectory_id)
        logger.info("analyze: building episodes from %d steps", traj.step_count)

        # P1: Goal-directed attention filter (before segmentation)
        if goal and traj.steps:
            from leapflow.recording.attention import GoalRelevanceFilter
            goal_filter = GoalRelevanceFilter(
                threshold=self._goal_relevance_threshold,
            )
            traj = goal_filter.filter_trajectory(traj, goal)

        # Route episode construction based on recording mode
        if self._recording_mode.uses_video and self._extracted_video_actions:
            if progress:
                progress("segment", 0, 1)
            episodes = await self._build_episodes_from_video(
                traj, self._extracted_video_actions, goal=goal,
            )
            if progress:
                progress("segment", 1, 1)
        else:
            if progress:
                progress("segment", 0, 1)
            episodes = self._segmenter.segment(traj)
            if progress:
                progress("segment", 1, 1)

            for i, ep in enumerate(episodes):
                if progress:
                    progress("abstract", i + 1, len(episodes))
                steps = ep.steps_from(traj)
                ep.semantic_actions = self._abstractor.abstract(steps)
                self._store.save_episode(ep)

            if self._extracted_video_actions:
                self._enrich_episodes_with_video(episodes, traj)

        # Optional MHMS-SF multi-scale fusion enrichment
        if self._mhms_fusion_enabled and self._perception_session:
            try:
                await self._run_mhms_fusion(episodes, traj, goal)
            except Exception:
                logger.debug("mhms_fusion failed, continuing with basic analysis", exc_info=True)

        # User-supplied goal takes precedence over LLM inference
        if goal:
            for ep in episodes:
                ep.inferred_goal = goal
                self._store.save_episode(ep)

        # Intent inference for episodes still lacking a goal
        needs_inference = [ep for ep in episodes if not ep.inferred_goal]
        if self._intent_inferrer and needs_inference:
            logger.info("analyze: %d episodes built, starting intent inference", len(episodes))
            if progress:
                progress("intent", 0, len(needs_inference))
            from leapflow.utils.stream_progress import StreamProgressWriter
            intent_writer = StreamProgressWriter(prefix="  │    ")
            try:
                results = await self._intent_inferrer.infer_batch(
                    needs_inference, on_chunk=intent_writer,
                )
                for ep, result in zip(needs_inference, results):
                    if result.goal:
                        ep.inferred_goal = result.goal
                        ep.confidence = max(ep.confidence, result.confidence)
                        self._store.save_episode(ep)
            except Exception as e:
                logger.warning("Intent inference failed for batch: %s", e)
            finally:
                intent_writer.finish()
            if progress:
                progress("intent", len(needs_inference), len(needs_inference))

        logger.info(
            "analyzed trajectory=%s episodes=%d",
            trajectory_id,
            len(episodes),
        )
        return episodes

    async def _run_mhms_fusion(
        self,
        episodes: List[Episode],
        traj: Trajectory,
        goal: str,
    ) -> None:
        """Run MHMS-SF multi-scale fusion to enrich episodes with cross-scale insights."""
        ps = self._perception_session
        if ps is None:
            return

        visual_actions = getattr(ps, "extracted_actions", None) or []
        if not visual_actions and not traj.steps:
            return

        from leapflow.signal_fusion.pipeline import MHMSFusionPipeline
        from leapflow.signal_fusion.protocol import FusionContext

        pipeline = MHMSFusionPipeline.default(intent_inferrer=self._intent_inferrer)

        system_events: list = []
        if traj.steps:
            from leapflow.domain.events import SystemEvent
            for step in traj.steps:
                system_events.append(SystemEvent(
                    event_type=step.action.action_type.value,
                    source=step.action.app_bundle_id or "unknown",
                    payload=step.action.params,
                    timestamp=step.action.timestamp,
                ))

        context = FusionContext(
            visual_actions=list(visual_actions),
            system_events=system_events,
            keyframes=list(getattr(ps, "keyframes", None) or []),
            goal=goal,
        )
        result = await pipeline.fuse(context)

        if result.episodes:
            for ep in episodes:
                for enriched in result.episodes:
                    if enriched.workflow_graph:
                        ep.procedure_graph = _workflow_graph_to_mermaid(
                            enriched.workflow_graph,
                        )
                        self._store.save_episode(ep)
                        break

        logger.info(
            "mhms_fusion: %d actions, %d segments, %d episodes enriched",
            len(result.atomic_actions), len(result.segments), len(result.episodes),
        )

    # ── Distillation ──

    async def distill(
        self, trajectory_id: str, *, goal: str = "", progress: ProgressCallback = None,
    ) -> List[DistillationCandidate]:
        """Run the full pipeline: analyze + distill skills from a trajectory.

        Automatically uses LLM-enhanced distillation when the distiller supports it,
        falling back to heuristic-only extraction otherwise.
        """
        cb = progress or self._progress_callback
        episodes = await self.analyze(trajectory_id, goal=goal, progress=cb)

        if isinstance(self._distiller, LLMSkillDistiller):
            return await self._distill_episodes_async(episodes, progress=cb)
        return self._distill_episodes_sync(episodes, progress=cb)

    async def _distill_episodes_async(
        self, episodes: List[Episode], *, progress: ProgressCallback = None,
    ) -> List[DistillationCandidate]:
        """LLM-enhanced distillation path with pre-deduplication."""
        from leapflow.analysis.episode_dedup import deduplicate_episodes
        from leapflow.utils.stream_progress import StreamProgressWriter

        assert isinstance(self._distiller, LLMSkillDistiller)
        unique_episodes = deduplicate_episodes(episodes)
        logger.info("distill: processing %d episodes (%d unique) via LLM", len(episodes), len(unique_episodes))
        candidates: List[DistillationCandidate] = []
        matched_episodes: List[Episode] = []
        for i, ep in enumerate(unique_episodes):
            if progress:
                progress("distill", i + 1, len(unique_episodes))
            writer = StreamProgressWriter(prefix="  │    ")
            try:
                candidate = await self._distiller.propose_from_episode_async(
                    ep, on_chunk=writer,
                )
            finally:
                writer.finish()
            if candidate:
                candidates.append(candidate)
                matched_episodes.append(ep)
        if self._on_candidates_ready and candidates:
            self._on_candidates_ready(candidates, matched_episodes)
        logger.info("distill: %d candidates generated from %d episodes (%d unique, llm)", len(candidates), len(episodes), len(unique_episodes))
        return candidates

    def distill_episodes(
        self, episodes: List[Episode], *, progress: ProgressCallback = None,
    ) -> List[DistillationCandidate]:
        """Heuristic-only distillation (backward-compat entry point)."""
        return self._distill_episodes_sync(episodes, progress=progress)

    def _distill_episodes_sync(
        self, episodes: List[Episode], *, progress: ProgressCallback = None,
    ) -> List[DistillationCandidate]:
        """Distill skill candidates using heuristic-only extraction."""
        from leapflow.analysis.episode_dedup import deduplicate_episodes

        unique_episodes = deduplicate_episodes(episodes)
        logger.info("distill: processing %d episodes (%d unique) via heuristic", len(episodes), len(unique_episodes))
        candidates: List[DistillationCandidate] = []
        matched_episodes: List[Episode] = []
        for i, ep in enumerate(unique_episodes):
            if progress:
                progress("distill", i + 1, len(unique_episodes))
            candidate = self._distiller.propose_from_episode(ep)
            if candidate:
                candidates.append(candidate)
                matched_episodes.append(ep)
        if self._on_candidates_ready and candidates:
            self._on_candidates_ready(candidates, matched_episodes)
        logger.info("distill: %d candidates generated from %d episodes (%d unique, heuristic)", len(candidates), len(episodes), len(unique_episodes))
        return candidates

    # ── Query ──

    def list_trajectories(self, *, limit: int = 20) -> list:
        """List recent trajectories."""
        return self._store.list_trajectories(limit=limit)

    def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Load a specific trajectory."""
        return self._store.load_trajectory(trajectory_id)

    def get_episodes(self, trajectory_id: str) -> List[Episode]:
        """Load episodes for a trajectory (from store, not re-analyzing)."""
        return self._store.load_episodes(trajectory_id)

    # ── Code Generation ──

    async def distill_to_code(
        self,
        trajectory_id: str,
        context: Optional["CodeGenContext"] = None,
    ) -> List["GeneratedSkill"]:
        """Full pipeline: trajectory → episodes → candidates → generated skills.

        Steps:
            1. analyze (segment + abstract)
            2. distill (candidate extraction)
            3. code generation for each candidate
            4. validation (via codegen's internal validator)
            5. return valid GeneratedSkills
        """
        from leapflow.learning.codegen import build_default_context

        if self._codegen is None:
            logger.warning("distill_to_code called but no codegen configured")
            return []

        ctx = context or build_default_context()

        # 1. Analyze trajectory into episodes
        episodes = await self.analyze(trajectory_id)
        if not episodes:
            return []

        # 2+3. Distill each episode to executable code
        generated: List["GeneratedSkill"] = []
        for ep in episodes:
            skill = await self._distiller.distill_to_executable(ep, self._codegen, ctx)
            if skill and skill.is_valid:
                generated.append(skill)

        logger.info(
            "distill_to_code trajectory=%s episodes=%d generated=%d",
            trajectory_id, len(episodes), len(generated),
        )
        return generated

    def format_trajectory(self, trajectory_id: str) -> str:
        """Human-readable text representation of a trajectory."""
        traj = self._store.load_trajectory(trajectory_id)
        if traj is None:
            return f"Trajectory {trajectory_id} not found."

        lines = [
            f"Trajectory: {traj.trajectory_id}",
            f"Duration: {traj.duration:.1f}s | Steps: {traj.step_count}",
            f"Apps: {', '.join(traj.app_sequence) or 'none'}",
            "",
        ]
        for i, step in enumerate(traj.steps):
            a = step.action
            ts_offset = a.timestamp - traj.start_time
            lines.append(
                f"  [{ts_offset:6.1f}s] {a.action_type.value:20s} "
                f"{a.target or a.app_bundle_id or ''}"
            )
        return "\n".join(lines)


def _merge_semantic_actions(
    existing: "List[Any]", video_actions: "List[Any]",
) -> "List[Any]":
    """Merge video-derived semantic actions into existing ones.

    Video actions with higher confidence replace existing actions that
    share the same action_name; otherwise they are appended.
    """
    by_name = {a.action_name: a for a in existing}
    for va in video_actions:
        prev = by_name.get(va.action_name)
        if prev and va.confidence > prev.confidence:
            by_name[va.action_name] = va
        elif not prev:
            by_name[va.action_name] = va
    return list(by_name.values())
