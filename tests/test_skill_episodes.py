# Copyright (c) Alibaba, Inc. and its affiliates.
"""Episode recording: every executed actuating run leaves a trainable record.

The arm, camera and policy server are simulated; the runtime, gate, recorder
and builtin skills are the production ones. Pins the on-disk layout (manifest,
rows, commands, frames, refs), that a policy run carries its model identity
and chunk events, that failures and stops are recorded too, and that
recording never changes a verdict: a full disk or a dead camera only leaves
less data, and recording is off unless a directory is configured.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.skills.builtin.episode import skills as episode_skills
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.policy import skills as policy_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.episodes import (
    EPISODE_FORMAT,
    NULL_TAP,
    EpisodeSettings,
    EpisodeTap,
    latest_labels,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    STATUS_HALTED,
    SkillError,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    ARM_ID,
    CAMERA_ID,
    JOINTS,
    POLICY_ID,
    SEQUENCE,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimArmTransport,
    SimPolicy,
    SimRgbdCamera,
    SkillNodeRegistry,
    action_space,
    embodiment,
    move_by,
    write_workcell,
)

EE = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")
GRIPPER = "gripper.position"
_TINY_CAMERA = CameraIntrinsics(width=16, height=12, fx=20.0, fy=20.0, ppx=8.0, ppy=6.0)


def _episodes(tmp_path: Path, **overrides: Any) -> EpisodeSettings:
    values: dict[str, Any] = {"directory": tmp_path / "episodes", "fps": 50.0, "min_free_mb": 0}
    return EpisodeSettings(**{**values, **overrides})


def _cell(
    tmp_path: Path,
    *,
    episodes: EpisodeSettings | None = None,
    arm: SimArmTransport | None = None,
    camera: SimRgbdCamera | None = None,
    policy: SimPolicy | None = None,
) -> SkillRuntime:
    devices: dict[str, Any] = {CAMERA_ID: camera or SimRgbdCamera(intrinsics=_TINY_CAMERA)}
    workcell: dict[str, Any] = {
        **WORKCELL_YAML,
        "roles": {**WORKCELL_YAML["roles"], "camera": CAMERA_ID},
    }
    if policy is not None:
        devices[POLICY_ID] = policy
        workcell["models"] = {"cube": POLICY_ID}
        workcell["policies"] = {"cube": embodiment(policy.space, cameras={"top": "camera"})}
    registry = SkillNodeRegistry(
        arm or SimArmTransport(travel_reads=3),
        store_path=tmp_path / "instrument.duckdb",
        devices=devices,
    )
    registry.calibrate_cell()
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell)
    settings = SkillRuntimeSettings(workcells_dir=directory)
    if episodes is not None:
        settings = SkillRuntimeSettings(workcells_dir=directory, episodes=episodes)
    return SkillRuntime(
        registry,
        settings=settings,
        skills=[*motion_skills(), *sequence_skills(), *policy_skills(), *episode_skills()],
        poll_s=0.02,
    )


def _sim(runtime: SkillRuntime) -> SimArmTransport:
    return runtime._registry.sim  # type: ignore[attr-defined] - the test owns the sim


async def _start(
    runtime: SkillRuntime, skill: str, args: Mapping[str, Any], *, intent: str = ""
) -> SkillRun:
    plan = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(plan, SkillPlan), plan
    run = await runtime.execute(plan.digest, intent=intent)
    assert isinstance(run, SkillRun), run
    return run


async def _run(
    runtime: SkillRuntime, skill: str, args: Mapping[str, Any], *, intent: str = ""
) -> SkillOutcome:
    run = await _start(runtime, skill, args, intent=intent)
    finished = await runtime.wait(run.run_id, timeout_s=10.0)
    assert finished is not None and finished.outcome is not None
    return finished.outcome


def _manifest(root: Path) -> list[dict[str, Any]]:
    lines = (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Layout ───────────────────────────────────────────────────────────────────


async def test_a_motion_run_writes_rows_commands_and_frames(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    outcome = await _run(runtime, SEQUENCE, move_by(20))
    assert outcome.ok, outcome
    episode = outcome.evidence["episode"]
    root = tmp_path / "episodes"
    assert episode["path"] == str(root) and episode["steps"] >= 2
    assert episode["cameras"] == [CAMERA_ID] and episode["commands"] >= 1

    (entry,) = _manifest(root)
    assert entry["episode_id"] == episode["episode_id"]
    assert entry["success"] is True and entry["skill_id"] == SEQUENCE
    assert entry["workcell"] == WORKCELL_ID and entry["plan_digest"]
    assert entry["step_count"] == episode["steps"] and entry["dropped"] == {}
    assert _load(root / "meta" / "info.json")["format"] == EPISODE_FORMAT

    rows = _load(root / entry["data"])
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert all(set(row["ee_pose"]) == set(EE) for row in rows)
    assert all(set(row["joint_positions"]) == set(JOINTS) for row in rows)
    assert rows[0]["gripper_state"] is not None
    assert rows[-1]["ee_pose"]["ee.z"] > rows[0]["ee_pose"]["ee.z"]  # the arm went up
    assert rows[-1]["action"]["ee.z"] == rows[-1]["ee_pose"]["ee.z"]
    assert all(row["camera_skew_ms"] == 0.0 for row in rows)  # one camera: no skew

    recorded = _load(root / entry["commands"])["commands"]
    sent = [dict(batch) for batch in _sim(runtime).batches]
    assert [command["commands"] for command in recorded] == sent

    refs = _load(root / "videos" / f"episode_{entry['episode_id']}" / f"{CAMERA_ID}_refs.json")
    assert len(refs) == len(rows)
    first = root / refs[0]["frame_ref"]
    assert first.read_bytes()[:2] == b"\xff\xd8"  # a JPEG


async def test_runs_append_to_one_dataset(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    first = await _run(runtime, SEQUENCE, move_by(10))
    runtime._registry.forget_rate_baselines()  # type: ignore[attr-defined]
    second = await _run(runtime, SEQUENCE, move_by(-10))
    assert first.ok and second.ok
    ids = [entry["episode_id"] for entry in _manifest(tmp_path / "episodes")]
    assert ids == [
        first.evidence["episode"]["episode_id"],
        second.evidence["episode"]["episode_id"],
    ]


async def test_a_policy_episode_carries_its_model_and_chunks(tmp_path: Path) -> None:
    policy = SimPolicy(action_space(gripper={}), (0, 0, 1.0, 0, 0, 0, 0))
    runtime = _cell(
        tmp_path, episodes=_episodes(tmp_path), arm=SimArmTransport(travel_reads=1), policy=policy
    )
    args = {"policy": "cube", "task": "pick up the cube", "duration_s": 0.6}
    outcome = await _run(runtime, "policy.run", args)
    assert outcome.ok, outcome
    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["task"] == "pick up the cube" and entry["skill_id"] == "policy.run"
    binding = entry["annotations"]["policy"]
    assert binding["name"] == "cube" and binding["device_id"] == POLICY_ID
    assert binding["model_info"]["name"] == "sim-policy"
    assert binding["action_channels"] == [*EE, GRIPPER]
    assert binding["state_channels"] == [*EE, GRIPPER] and binding["cameras"] == {"top": CAMERA_ID}
    adapter = binding["adapter"]
    assert adapter["digest"] == outcome.result["policy"]["policy"]["adapter_digest"]
    assert adapter["steps"] == ["map", "validate"]
    assert adapter["spaces"]["action_space"]["fps"] == 20.0

    events = _load(tmp_path / "episodes" / entry["commands"])["events"]
    chunks = [event for event in events if event["kind"] == "chunk"]
    assert len(chunks) == outcome.result["policy"]["chunks"]
    # Both sides of the adapter: the model's native rows and the channel targets.
    assert all(len(chunk["native"][0]) == 7 for chunk in chunks)
    assert all(len(chunk["actions"][0]) == len(EE) + 1 for chunk in chunks)


# ── S2's annotations: intent and verified success ────────────────────────────


async def test_a_run_intent_is_the_episode_language_label(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    outcome = await _run(runtime, SEQUENCE, move_by(10), intent="lift the gripper clear")
    assert outcome.ok, outcome
    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["intent"] == entry["task"] == "lift the gripper clear"
    assert entry["commanded"] is True
    (run,) = runtime.status()
    assert run.to_dict()["intent"] == "lift the gripper clear"


async def test_a_label_records_s2_verification_and_the_latest_wins(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    outcome = await _run(runtime, SEQUENCE, move_by(10))
    episode_id = outcome.evidence["episode"]["episode_id"]
    first = runtime.label(episode_id, success=False, note="cup tipped over")
    assert first["success"] is False and first["source"] == "host"
    runtime.label(episode_id, success=True, note="re-checked: upright")
    label = latest_labels(tmp_path / "episodes")[episode_id]
    assert label["success"] is True and label["note"] == "re-checked: upright"
    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["success"] is True  # the run's own verdict is never rewritten


async def test_labels_are_refused_for_running_unknown_or_unrecorded_runs(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path), arm=SimArmTransport(stall=True))
    run = await _start(runtime, SEQUENCE, move_by(50))
    with pytest.raises(SkillError) as running:
        runtime.label(run.run_id, success=True)
    assert running.value.failure_code == FAILURE_INVALID_ARGS
    await runtime.stop(run.run_id, "test over")
    await runtime.wait(run.run_id, timeout_s=5.0)
    for bad in ("f" * 32, "../../etc"):
        with pytest.raises(SkillError) as unknown:
            runtime.label(bad, success=True)
        assert unknown.value.failure_code == FAILURE_SKILL_UNAVAILABLE

    (tmp_path / "off").mkdir()
    unrecorded = _cell(tmp_path / "off")
    done = await _run(unrecorded, SEQUENCE, move_by(10))
    (finished,) = unrecorded.status()
    assert done.ok
    with pytest.raises(SkillError) as off:
        unrecorded.label(finished.run_id, success=True)
    assert off.value.failure_code == FAILURE_SKILL_UNAVAILABLE


# ── Teleoperated demonstrations ──────────────────────────────────────────────


async def test_a_demonstration_holds_the_cell_and_commands_nothing(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    task = "stack the blocks"
    run = await _start(runtime, "episode.record", {"task": task, "duration_s": 30})
    await asyncio.sleep(0.2)  # the operator demonstrates; the recorder samples at 50 Hz
    assert runtime.holds_device(ARM_ID)
    other = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(other, SkillPlan)
    busy = await runtime.execute(other.digest)
    assert isinstance(busy, SkillOutcome) and busy.failure_code == "device_busy"

    await runtime.stop(run.run_id, "demonstration over")
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and finished.outcome is not None
    assert finished.outcome.failure_code == FAILURE_STOPPED
    assert finished.outcome.side_effect_state == "none"
    assert _sim(runtime).batches == [] and _sim(runtime).halts == 0
    assert not runtime.holds_device(ARM_ID)

    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["skill_id"] == "episode.record" and entry["task"] == task
    assert entry["commanded"] is False and entry["step_count"] >= 1
    runtime.label(run.run_id, success=True, note="clean demonstration")
    assert latest_labels(tmp_path / "episodes")[run.run_id]["success"] is True


async def test_a_demonstration_needs_recording_and_room_in_the_run_cap(tmp_path: Path) -> None:
    off = await _cell(tmp_path).plan(WORKCELL_ID, "episode.record", {"task": "wave"})
    assert isinstance(off, SkillOutcome) and off.failure_code == FAILURE_SKILL_UNAVAILABLE
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    long = await runtime.plan(WORKCELL_ID, "episode.record", {"task": "wave", "duration_s": 1e6})
    assert isinstance(long, SkillOutcome) and long.failure_code == FAILURE_INVALID_ARGS


async def test_a_full_demonstration_ends_by_itself(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path))
    outcome = await _run(runtime, "episode.record", {"task": "wave", "duration_s": 1})
    assert outcome.ok and outcome.side_effect_state == "none", outcome
    assert outcome.result["recorded_s"] == 1.0
    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["success"] is True and entry["commanded"] is False
    assert entry["step_count"] >= 10  # sampled throughout, at 50 Hz


# ── Verdicts are recorded, never changed ─────────────────────────────────────


async def test_a_stopped_run_is_still_recorded(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path), arm=SimArmTransport(stall=True))
    run = await _start(runtime, SEQUENCE, move_by(50))
    for _ in range(200):
        if _sim(runtime).batches:
            break
        await asyncio.sleep(0.01)
    await runtime.stop(run.run_id, "operator pressed stop")
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and finished.outcome is not None
    outcome = finished.outcome
    assert (outcome.status, outcome.failure_code) == (STATUS_HALTED, FAILURE_STOPPED)
    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["success"] is False and entry["status"] == STATUS_HALTED
    assert entry["failure_code"] == FAILURE_STOPPED
    assert entry["message"] == "operator pressed stop"


async def test_a_full_disk_skips_recording_without_failing_the_run(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path, min_free_mb=1e12))
    outcome = await _run(runtime, SEQUENCE, move_by(10))
    assert outcome.ok, outcome
    assert "MB free" in outcome.evidence["episode"]["skipped"]
    assert not (tmp_path / "episodes" / "meta" / "episodes.jsonl").exists()


async def test_a_dead_camera_drops_frames_without_failing_the_run(tmp_path: Path) -> None:
    camera = SimRgbdCamera(intrinsics=_TINY_CAMERA, fail=True)
    runtime = _cell(tmp_path, episodes=_episodes(tmp_path), camera=camera)
    outcome = await _run(runtime, SEQUENCE, move_by(10))
    assert outcome.ok, outcome
    (entry,) = _manifest(tmp_path / "episodes")
    assert entry["dropped"][CAMERA_ID] == entry["step_count"] >= 1
    rows = _load(tmp_path / "episodes" / entry["data"])
    assert all(row["camera_frames"] == {} for row in rows)
    assert all(row["camera_skew_ms"] is None for row in rows)


async def test_recording_is_off_without_a_directory(tmp_path: Path) -> None:
    runtime = _cell(tmp_path)
    outcome = await _run(runtime, SEQUENCE, move_by(10))
    assert outcome.ok, outcome
    assert "episode" not in outcome.evidence
    assert not (tmp_path / "episodes").exists()


def test_settings_and_null_tap() -> None:
    assert not EpisodeSettings().enabled
    assert not EpisodeSettings(directory=Path("/tmp/x"), fps=0).enabled
    assert EpisodeSettings(directory=Path("/tmp/x")).enabled
    assert isinstance(NULL_TAP, EpisodeTap)
    NULL_TAP.command("arm", (("ee.z", 1.0),))
    NULL_TAP.annotate("k", 1)
    NULL_TAP.event("chunk", index=0)
