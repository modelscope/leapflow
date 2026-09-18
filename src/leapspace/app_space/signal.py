# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapSignal — the in-sandbox leapflow entry for signal-mode runs.

One process owns the whole observation stack: EventBus (real normalizer
and privacy gate), ObservationDaemon, and ImitationPipeline. Control
flows through record_* sentinel files — the host polls
record_start.json, creates record_stop.json, and waits on record_done.json —
so no port, process signal, or RPC is needed. Host-side code never
imports this; it runs inside the sandbox via the image venv interpreter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leapspace.app_space.state import write_atomic

if TYPE_CHECKING:
    from leapflow.analysis.intent_inferrer import InferenceResult
    from leapflow.analysis.pipeline import ImitationPipeline
    from leapflow.domain.trajectory import Episode
    from leapflow.platform.event_bus import EventBus
    from leapflow.platform.observers.daemon import ObservationDaemon

logger = logging.getLogger(__name__)

# Control protocol between the in-sandbox runner and the host harness.
# record_start.json and record_done.json carry JSON payloads written by
# this process; record_stop.json is the host's JSON payload to end the
# stimulus window — this process polls its existence only.
RECORD_START_FILE = "record_start.json"
RECORD_STOP_FILE = "record_stop.json"
RECORD_DONE_FILE = "record_done.json"

SENTINEL_POLL_S = 0.2

# The eval store sits beside the control files, outside the fs watch surface.
EVAL_STORE_FILE = "eval.duckdb"


class LeapSignal:
    """Own the in-box observation stack for one signal run.

    Launch order encodes the recording window: the harness starts this
    process after the apps are ready and right before the stimulus, so
    recording starts at process start — no start round-trip needed.
    """

    def __init__(
        self,
        signal_dir: Path,
        *,
        goal: str = "",
        watch_paths: tuple[str, ...] = (),
    ) -> None:
        """Bind the run's control directory and fs watch surface."""
        self._dir = signal_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._goal = goal
        self._watch_paths = watch_paths
        self._bus: EventBus | None = None
        self._pipeline: ImitationPipeline | None = None
        self._daemon: ObservationDaemon | None = None
        self._trajectory_id = ""

    async def run(self) -> int:
        """Drive the full lifecycle; the exit code is the process verdict.

        Any death still writes record_done.json — the host's deadline poll
        then has something to diagnose instead of a bare timeout.
        """
        try:
            await self._start()
            await self._await_stop()
            summary = await self._stop()
            self._write(RECORD_DONE_FILE, {**summary, "ok": True})
            return 0
        except BaseException as exc:
            self._write(RECORD_DONE_FILE, {"ok": False, "error": repr(exc)})
            logger.exception("LeapSignal failed")
            return 1

    async def _start(self) -> None:
        """Assemble the production stack and begin recording."""
        import platform as _platform

        from leapflow.analysis.pipeline import ImitationPipeline
        from leapflow.domain.platform import PlatformID, PlatformManifest
        from leapflow.memory import EpisodicMemoryProvider, WorkingMemoryProvider
        from leapflow.platform.event_bus import EventBus
        from leapflow.platform.normalizer import EventNormalizer
        from leapflow.platform.observers import ObserverConfig, RecordingProfile
        from leapflow.platform.observers.daemon import ObservationDaemon
        from leapflow.privacy.policy import PrivacyManager, PrivacyPolicy
        from leapflow.storage.trajectory_store import TrajectoryStore

        # Minimal assembly mirroring cli/context.py: the real normalizer and
        # privacy gate; in-process memory providers — a single run's
        # assertions depend on trajectory/episode persistence only.
        self._bus = EventBus(
            immediate=EpisodicMemoryProvider(),
            working=WorkingMemoryProvider(),
            normalizer=EventNormalizer(
                PlatformManifest(
                    platform_id=PlatformID.resolve(),
                    os_version=_platform.version(),
                    capabilities=frozenset(),
                )
            ),
            privacy_filter=PrivacyManager(PrivacyPolicy()),
        )
        # The fs watch surface is scenario-declared; an empty list falls back
        # to Path.home() (fs_watcher.py:41).
        self._daemon = ObservationDaemon(
            bus=self._bus,
            config=ObserverConfig(fs_watch_paths=list(self._watch_paths)),
        )
        await self._daemon.start()

        self._pipeline = ImitationPipeline(
            TrajectoryStore(self._dir / EVAL_STORE_FILE),
            event_bus=self._bus,
            # Owned explicitly here: start/stop belong to this class, not
            # the pipeline's on-demand startup.
            observation_daemon=self._daemon,
            recording_profile=RecordingProfile(),
            intent_inferrer=FakeIntentInferrer(),
        )
        # Production wires the recorder subscription in cli/context.py:2027;
        # a bare assembly must do it itself.
        self._bus.subscribe(self._pipeline.recorder.on_event)
        # causal feed — online per-event CausalFusionPipeline.fuse, the
        # perception/session.py:292 shape — is the remaining wiring point.

        self._trajectory_id = await self._pipeline.start_recording(goal=self._goal)
        self._write(
            RECORD_START_FILE,
            {"trajectory_id": self._trajectory_id, "observers": self._daemon.status},
        )

    async def _await_stop(self) -> None:
        """Poll until the host creates record_stop.json."""
        while not (self._dir / RECORD_STOP_FILE).exists():
            await asyncio.sleep(SENTINEL_POLL_S)

    async def _stop(self) -> dict[str, Any]:
        """Stop recording (drain inside), analyze, and summarize the run."""
        # stop_recording drains internally: disable_reorder -> flush ->
        # 0.2s settle -> recorder.stop persists the trajectory.
        traj = await self._pipeline.stop_recording()
        episodes = await self._pipeline.analyze(self._trajectory_id, goal=self._goal)
        await self._daemon.stop()
        return {
            "trajectory_id": self._trajectory_id,
            "steps": traj.step_count if traj else 0,
            "episodes": len(episodes),
            # causal node/edge counts land here once the feed is wired.
        }

    def _write(self, name: str, payload: dict[str, Any]) -> None:
        """Write one control file atomically — the host reads it back."""
        write_atomic(self._dir / name, json.dumps(payload, ensure_ascii=False))


class FakeIntentInferrer:
    """Deterministic stand-in for the LLM boundary of intent inference.

    Echoes the episode's action names back as the goal, so the verdict
    program can assert on exactly what the pipeline showed the model.
    """

    async def infer(
        self, episode: Episode, context: dict[str, Any] | None = None
    ) -> InferenceResult:
        return (await self.infer_batch([episode], context))[0]

    async def infer_batch(
        self,
        episodes: list[Episode],
        context: dict[str, Any] | None = None,
        *,
        on_chunk: Any = None,
    ) -> list[InferenceResult]:
        from leapflow.analysis.intent_inferrer import InferenceResult

        return [
            InferenceResult(
                goal=("fake:" + " | ".join(
                    sa.action_name for sa in ep.semantic_actions
                ))[:80],
                confidence=1.0,
            )
            for ep in episodes
        ]


async def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LeapSpace in-box signal runner")
    parser.add_argument("signal_dir", type=Path)
    parser.add_argument("--goal", default="")
    parser.add_argument("--watch", action="append", default=[],
                        help="fs watch path inside the sandbox (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return await LeapSignal(
        args.signal_dir, goal=args.goal, watch_paths=tuple(args.watch)
    ).run()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
