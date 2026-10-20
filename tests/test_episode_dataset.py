# Copyright (c) Alibaba, Inc. and its affiliates.
"""Recorded episodes read back as training data (``episode_dataset``, ``adapt.encode``).

Pins which episodes a dataset takes (the newest S2 label over the run's own
verdict, verified successes only on request), how an episode is put on a
fixed-rate grid and what each frame's action target is (commanded, or the next
observed state of a demonstration), and that the ``model`` space is the very
adapter a policy ran under, inverted: decoding an encoded action gives back the
recorded target, and the encoded state is what the model was sent.
"""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from leapflow.hardware.skills.adapt import EpisodeEncoder
from leapflow.hardware.skills.adapt.observe import fit_image
from leapflow.hardware.skills.adapt.spec import (
    FIT_CROP,
    FIT_PAD,
    FIT_STRETCH,
    ImageSpec,
    stats_checksum,
)
from leapflow.hardware.skills.episode_dataset import (
    SPACE_EE,
    SPACE_JOINT,
    SPACE_MODEL,
    EpisodeFilter,
    RecordedEpisode,
    camera_keys,
    read_episodes,
    resample,
    training_space,
)
from leapflow.hardware.skills.episodes import (
    EPISODE_FORMAT,
    EPISODE_FORMAT_VERSION,
    EpisodeSettings,
)
from leapflow.hardware.skills.geometry import wrap_deg
from leapflow.hardware.skills.kinematics import DhChain
from tests._harness.policy_cell import (
    EE,
    GRIPPER,
    JOINT_START,
    execute_run,
    piper_chain,
    plan_run,
    policy_cell,
)
from tests._harness.skill_stubs import (
    JOINTS,
    TOP_IMAGE,
    SimArmTransport,
    SimPolicy,
    action_space,
    embodiment,
)

TOL = 1e-6
_ANGLES = {"ee.rx", "ee.ry", "ee.rz"}
_CONVERTER = Path(__file__).parents[1] / "temp" / "devices" / "deployment" / "convert_to_lerobot.py"


# ── A synthetic dataset ──────────────────────────────────────────────────────


def _row(t: float, z: float, *, grip: float | None = 40.0, **action: float) -> dict[str, Any]:
    return {
        "t": t,
        "joint_positions": {joint: 1.0 for joint in JOINTS},
        "ee_pose": {
            "ee.x": 300.0,
            "ee.y": 0.0,
            "ee.z": z,
            "ee.rx": 180.0,
            "ee.ry": 0.0,
            "ee.rz": 0.0,
        },
        "gripper_state": grip,
        "action": dict(action),
        "camera_frames": {},
    }


def _dataset(root: Path, episodes: Sequence[Mapping[str, Any]], labels: Sequence[Mapping]) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    info = {"format": EPISODE_FORMAT, "format_version": EPISODE_FORMAT_VERSION}
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    manifest = []
    for episode in episodes:
        rows = episode.get("rows") or [_row(0.0, 200.0), _row(0.1, 201.0)]
        entry = {k: v for k, v in episode.items() if k != "rows"}
        entry["data"] = f"data/episode_{entry['episode_id']}.json"
        (root / entry["data"]).write_text(json.dumps(rows), encoding="utf-8")
        manifest.append(json.dumps(entry))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    lines = [json.dumps(label) for label in labels]
    (root / "meta" / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _entry(episode_id: str, success: bool, **extra: Any) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "success": success,
        "skill_id": "motion.sequence",
        "workcell": "cell",
        **extra,
    }


@pytest.fixture
def labelled(tmp_path: Path) -> Path:
    return _dataset(
        tmp_path / "episodes",
        [
            _entry("ran_ok", True),
            _entry("ran_ok_s2_no", True),
            _entry("ran_failed_s2_yes", False),
            _entry("ran_ok_s2_yes", True, skill_id="policy.run", workcell="other"),
        ],
        [
            {"episode_id": "ran_ok_s2_no", "success": True},
            {"episode_id": "ran_ok_s2_no", "success": False},  # the newest wins
            {"episode_id": "ran_failed_s2_yes", "success": True},
            {"episode_id": "ran_ok_s2_yes", "success": True},
        ],
    )


def _ids(episodes: Sequence[RecordedEpisode]) -> list[str]:
    return [episode.episode_id for episode in episodes]


def test_s2_labels_decide_which_episodes_count_as_successes(labelled: Path) -> None:
    taken = read_episodes(labelled, EpisodeFilter())
    assert _ids(taken) == ["ran_ok", "ran_failed_s2_yes", "ran_ok_s2_yes"]
    assert [e.verified for e in taken] == [None, True, True]

    verified = read_episodes(labelled, EpisodeFilter(verified_only=True))
    assert _ids(verified) == ["ran_failed_s2_yes", "ran_ok_s2_yes"]

    every = read_episodes(labelled, EpisodeFilter(include_failures=True))
    assert len(every) == 4
    assert {e.episode_id: e.success for e in every}["ran_ok_s2_no"] is False


def test_episodes_are_selected_by_skill_workcell_and_length(labelled: Path) -> None:
    assert _ids(read_episodes(labelled, EpisodeFilter(skills=("policy.*",)))) == ["ran_ok_s2_yes"]
    assert _ids(read_episodes(labelled, EpisodeFilter(workcell="other"))) == ["ran_ok_s2_yes"]
    assert read_episodes(labelled, EpisodeFilter(min_rows=3)) == []


def test_another_format_is_refused(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "episodes", [], [])
    (root / "meta" / "info.json").write_text('{"format": "lerobot"}', encoding="utf-8")
    with pytest.raises(ValueError, match="is not a"):
        read_episodes(root, EpisodeFilter())


def _one(tmp_path: Path, rows: Sequence[Mapping[str, Any]], **extra: Any) -> RecordedEpisode:
    root = _dataset(tmp_path / "episodes", [_entry("e1", True, rows=rows, **extra)], [])
    (episode,) = read_episodes(root, EpisodeFilter())
    return episode


def test_a_commanded_target_holds_what_nothing_commanded(tmp_path: Path) -> None:
    rows = [_row(0.0, 200.0, grip=None), _row(0.05, 200.0), _row(0.1, 201.0, **{"ee.z": 210.0})]
    episode = _one(tmp_path, rows)
    space = training_space(SPACE_EE, [episode])
    assert space.state_names == ("x", "y", "z", "rx", "ry", "rz", GRIPPER)
    frames = resample(episode, space.channels, (), fps=20.0)
    assert len(frames) == 2  # the first tick has no gripper reading yet
    assert space.action(frames[0]) == space.state(frames[0])  # nothing commanded yet
    assert frames[1].target["ee.z"] == 210.0 and frames[1].sample["ee.z"] == 201.0
    assert space.action(frames[1])[0] == 300.0  # x was never commanded: hold the observation


def test_a_demonstration_targets_the_next_observed_state(tmp_path: Path) -> None:
    rows = [_row(0.1 * i, 200.0 + i) for i in range(4)]
    episode = _one(tmp_path, rows, commanded=False, skill_id="episode.record")
    space = training_space(SPACE_JOINT, [episode])
    assert space.channels == (*sorted(JOINTS), GRIPPER)
    frames = resample(episode, EE, (), fps=10.0)
    assert [f.target["ee.z"] for f in frames] == [201.0, 202.0, 203.0, 203.0]


def test_the_model_space_needs_one_pinned_binding(tmp_path: Path) -> None:
    episode = _one(tmp_path, [_row(0.0, 200.0)])
    with pytest.raises(ValueError, match="pin no encoding"):
        training_space(SPACE_MODEL, [episode])
    with pytest.raises(ValueError, match="must be one of"):
        training_space("lerobot", [episode])
    a = RecordedEpisode(
        {"episode_id": "a", "annotations": {"policy": {"encoding": {"x": 1}}}}, (), True
    )
    b = RecordedEpisode(
        {"episode_id": "b", "annotations": {"policy": {"encoding": {"x": 2}}}}, (), True
    )
    with pytest.raises(ValueError, match="different policy bindings"):
        training_space(SPACE_MODEL, [a, b])


def test_a_hand_written_encoding_encodes_a_demonstration(tmp_path: Path) -> None:
    rows = [_row(0.1 * i, 200.0 + i, grip=10.0 * i) for i in range(3)]
    episode = _one(tmp_path, rows, commanded=False)
    space = {
        "fps": 10.0,
        "width": 7,
        "groups": [
            {"name": "arm", "kind": "ee_pose", "dims": list(range(6)), "unit": "m"},
            {"name": "gripper", "kind": "gripper", "dims": [6], "unit": "normalized"},
        ],
    }
    record = {
        "spaces": {"action_space": space},
        "actions": [
            {"group": "arm", "channels": list(EE)},
            {"group": "gripper", "channels": [GRIPPER]},
        ],
        "gripper_mm": {"gripper": [0, 40]},
    }
    model = training_space(SPACE_MODEL, [episode], encoding=record)
    assert model.state_names == ()  # the model observes no state
    frames = resample(episode, model.channels, (), fps=10.0)
    first = model.action(frames[0])
    assert first[2] == pytest.approx(0.201) and first[6] == pytest.approx(0.25)


def test_a_model_dataset_keys_images_by_the_policys_camera_names(tmp_path: Path) -> None:
    policy = {"cameras": {"top": "realsense"}}
    ran = RecordedEpisode(
        {"episode_id": "a", "cameras": ["realsense", "wrist"], "annotations": {"policy": policy}},
        (),
        True,
    )
    assert camera_keys([ran], model=True) == {"realsense": "top"}  # what the policy observed
    assert camera_keys([ran], model=False) == {"realsense": "realsense", "wrist": "wrist"}
    shown = RecordedEpisode({"episode_id": "b", "cameras": ["realsense"]}, (), True)
    assert camera_keys([shown], model=True) == {"realsense": "realsense"}  # no policy ran
    assert training_space(SPACE_EE, [shown]).fps is None


def test_converter_keeps_model_fitting_when_a_camera_key_is_renamed(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    (tmp_path / "frame.png").parent.mkdir(exist_ok=True)
    image_module.new("RGB", (8, 4), "red").save(tmp_path / "frame.png")
    policy = {"cameras": {"top": "realsense"}}
    episode = RecordedEpisode(
        {"episode_id": "a", "cameras": ["realsense"], "annotations": {"policy": policy}},
        ({"camera_frames": {"realsense": "frame.png"}},),
        True,
    )
    declared = ImageSpec(width=6, height=6, fit=FIT_CROP)
    space = SimpleNamespace(action_names=(), state_names=(), image_specs={"top": declared})
    converter = runpy.run_path(str(_CONVERTER))
    streams = converter["_camera_streams"](
        [episode], space, model=True, renames={"realsense": "front"}
    )
    assert [(stream.device, stream.key, stream.spec) for stream in streams] == [
        ("realsense", "front", declared)
    ]
    spec = converter["features"](tmp_path, [episode], space, streams)
    assert spec["observation.images.front"]["shape"] == (6, 6, 3)


@pytest.mark.parametrize("fit", [FIT_STRETCH, FIT_CROP, FIT_PAD])
def test_converter_reuses_inference_image_fitting(tmp_path: Path, fit: str) -> None:
    image_module = pytest.importorskip("PIL.Image")
    source = image_module.new("RGB", (4, 2))
    source.putdata(
        [(255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)] * source.height
    )
    path = tmp_path / "frame.png"
    source.save(path)
    image_spec = ImageSpec(width=4, height=4, fit=fit)
    converter = runpy.run_path(str(_CONVERTER))
    converted = converter["_training_image"](path, image_spec)
    expected = fit_image(source, image_spec)
    assert converted.size == (4, 4)
    assert converted.tobytes() == expected.tobytes()
    if fit == FIT_CROP:
        pixels = {converted.getpixel((x, y)) for y in range(4) for x in range(4)}
        assert pixels == {(0, 255, 0)}
    elif fit == FIT_PAD:
        assert converted.getpixel((0, 0)) == (0, 0, 0)
        assert converted.getpixel((1, 1)) != (0, 0, 0)


def test_a_dh_table_survives_its_record() -> None:
    chain = piper_chain()
    again = DhChain.from_mapping(chain.to_dict())
    assert again == chain
    assert again.forward(JOINT_START).as_tuple() == chain.forward(JOINT_START).as_tuple()


# ── A recorded policy run, re-encoded by its own adapter ─────────────────────


def _smolvla() -> SimPolicy:
    mean = [math.radians(v) for v in JOINT_START] + [0.5]
    std = [0.5] * 7
    space = action_space("joints", fps=30.0, unit="rad", gripper={"unit": "normalized"})
    space["normalization"] = {
        "mode": "mean_std",
        "mean": mean,
        "std": std,
        "checksum": stats_checksum("mean_std", mean, std),
    }
    return SimPolicy(space, (0.02, 0, 0, 0, 0, 0, 0))


def _edgevla() -> SimPolicy:
    space = action_space(
        fps=10.0,
        unit="m",
        rotation="axis_angle",
        angle_unit="rad",
        reference="delta",
        frame="tool",
        gripper={"unit": "normalized", "binary": True},
    )
    rows = [[0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 1.0]] * 5
    return SimPolicy(space, (0.0,) * 7, observation={"images": TOP_IMAGE}, rows=rows)


def _pose_on_joints() -> SimPolicy:
    return SimPolicy(action_space(gripper={}), (0, 0, -1.0, 0, 0, 0, 0))


_IK_DECL = {"actions": {"arm": list(JOINTS), "gripper": [GRIPPER]}, "cameras": {"top": "camera"}}


def _joint_arm() -> SimArmTransport:
    return SimArmTransport(travel_reads=1, joints=JOINT_START, forward=piper_chain().forward)


def _close(actual: Mapping[str, float], expected: Mapping[str, float], skip: set[str]) -> None:
    for channel, value in actual.items():
        if channel in skip:
            continue
        delta = value - expected[channel]
        if channel in _ANGLES:
            delta = wrap_deg(delta)
        assert abs(delta) < TOL, (channel, value, expected[channel])


@pytest.mark.parametrize(
    ("make", "decl", "joint_arm", "lossy", "tol"),
    [
        (_smolvla, None, True, set(), TOL),
        # A binary gripper only says open or closed: 60 mm is "open".
        (_edgevla, None, False, {GRIPPER}, TOL),
        # IK solves the pose to its declared tolerance (0.1 mm / 0.1 deg).
        (_pose_on_joints, _IK_DECL, True, set(JOINTS), 0.1),
    ],
    ids=["smolvla", "edgevla", "ik"],
)
async def test_a_policy_episode_round_trips_through_its_adapter(
    tmp_path: Path,
    make: Any,
    decl: Mapping[str, Any] | None,
    joint_arm: bool,
    lossy: set[str],
    tol: float,
) -> None:
    policy = make()
    episodes = EpisodeSettings(directory=tmp_path / "episodes", fps=50.0, min_free_mb=0)
    arm = _joint_arm() if joint_arm else SimArmTransport(travel_reads=1)
    binding = decl or embodiment(policy.space, cameras={"top": "camera"})
    runtime = policy_cell(tmp_path, policy, arm=arm, decl=binding, episodes=episodes)
    outcome = await execute_run(runtime, await plan_run(runtime))
    assert outcome.ok, outcome

    (episode,) = read_episodes(tmp_path / "episodes", EpisodeFilter(skills=("policy.run",)))
    model = training_space(SPACE_MODEL, [episode])
    assert model.fps == policy.space["fps"]
    assert set(camera_keys([episode], model=True).values()) == {"top"}
    encoder: EpisodeEncoder = model.encoder  # type: ignore[attr-defined]
    frames = resample(episode, model.channels, (), fps=policy.space["fps"])
    assert len(frames) > 3
    for frame in frames:
        native = model.action(frame)
        assert len(native) == len(model.action_names) == policy.space["width"]
        decoded = encoder.decode_action(native, frame.sample)
        _close(decoded, frame.target, lossy)
        # What the decoded targets command encodes back to the same native action.
        again = encoder.encode_action({**frame.target, **decoded}, frame.sample)
        # Euler degrees may come back on the other side of +-180.
        gaps = [min(abs(a - b), abs(wrap_deg(a - b))) for a, b in zip(again, native)]
        assert max(gaps) < tol, (again, native)

    sent = policy.requests[0]["state"]  # the preview's observation, before anything moved
    assert len(model.state_names) == len(sent)
    if sent:
        assert model.state(frames[0]) == pytest.approx(sent, abs=1e-6)
