# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real registry/cache/vision wiring with an external camera boundary; no device writes."""

from __future__ import annotations

import asyncio
import json
import stat
import sys
import time
from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapcalib.contracts.identity import ModelIdentity, digest_of
from leapcalib.contracts.motion import MODE_FIXED, MODE_WRIST
from leapcalib.geometry.se3 import Transform
from leapflow.cache.manager import CacheManager
from leapflow.hardware.calibration.measurements import tracked_motion, si_transform
from leapflow.hardware.calibration.readiness import CATEGORY
from leapflow.hardware.context import HardwareContext
from leapflow.hardware.lhp.server import LhpNodeServer, WRITE_MODE_OBSERVE_ONLY
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills.args import validate_args
from leapflow.hardware.skills.builtin.metric_readiness import MetricReadiness
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.geometry import Transform as HardwareTransform
from leapflow.hardware.skills.protocol import SkillError
from leapflow.hardware.skills.vision import WorkcellVision
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.transport import FrameReading
from leapflow.layout import build_layout
from tests._harness.skill_stubs import ARM_ID, piper_context, write_workcell
from tools.calib_regression import camera_regression


def frame(sequence: int = 1, seconds: float = 1) -> RgbdFrame:
    return RgbdFrame(
        "camera",
        16,
        12,
        b"\0" * (16 * 12 * 3),
        array("H", [5000] * (16 * 12)).tobytes(),
        0.0001,
        CameraIntrinsics(16, 12, 10, 10, 8, 6),
        1700000000,
        seconds,
        sequence=sequence,
        metadata={"aligned_to_color": True, "profile_digest": "live"},
    )


class Camera:
    """External capture boundary only: production registry locks and cache are real."""

    def __init__(self) -> None:
        self.sequence = 0
        self.active = 0
        self.maximum_active = 0
        self.stale = False
        self.profile_change = False
        self.future = False
        self.preview_sequence = 0
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1

    async def read_frame(
        self,
        channel_id: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0,
        min_sequence: int = 0,
    ) -> FrameReading:
        await asyncio.sleep(0.001)
        self.preview_sequence += 1
        return FrameReading(
            "camera",
            channel_id,
            b"bounded-preview",
            width=16,
            height=12,
            sequence=self.preview_sequence,
        )

    async def capture_rgbd(self, *, after_monotonic: float | None = None) -> RgbdFrame:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0.002)
            self.sequence += 1
            captured = frame(self.sequence, time.monotonic() if not self.stale else 1)
            if self.future:
                captured = replace(captured, monotonic_at=time.monotonic() + 1)
            if self.profile_change and self.sequence > 1:
                captured = replace(captured, depth_scale_m=0.001)
            return captured
        finally:
            self.active -= 1


@pytest.fixture
def production_port(tmp_path: Path, monkeypatch):
    layout = build_layout(tmp_path / "home").profile("default")
    cache = CacheManager(layout.cache, profile_id="test_profile")
    registry = HardwareRegistry(settings=HardwareSettings(enabled=True))
    registry.bind_persistence(cache_manager=cache)
    declaration = HardwareContext.from_mapping(
        {
            "hc_version": "hc.v1",
            "device_id": "camera",
            "transport": {"kind": "media"},
            "channels": [{"channel_id": "frame", "direction": "read", "representation": "frame"}],
        }
    )
    arm = piper_context()
    arm = replace(arm, provenance=replace(arm.provenance, verified_by="offline-test-fixture"))
    registry.register_external_contexts("test_declarations", (arm, declaration))
    report = registry.load()
    assert set(report.admitted) == {ARM_ID, "camera"}, report.to_dict()
    camera = Camera()

    async def transport(device_id: str):
        assert device_id == "camera"
        return camera

    monkeypatch.setattr(registry, "transport", transport)
    return WorkcellVision(registry=registry, camera="camera"), cache, camera


@pytest.mark.asyncio
async def test_metric_readiness_keeps_raw_samples_local_private_and_indexed(
    production_port,
) -> None:
    port, cache, camera = production_port
    summary, _ = await asyncio.gather(
        port.metric_readiness(
            frames=4,
            points=32,
            role="fixed",
            stamp_uncertainty_s=0.05,
            ttl_s=60,
            quota_bytes=1048576,
        ),
        port._capture(),
    )
    assert camera.maximum_active == 1
    assert summary["frame_count"] == 4 and not summary["raw_exported"]
    assert not summary["motion_support"] and not summary["hardware_accuracy_certified"]
    assert "observations" not in summary
    entries = cache.list_entries(category=CATEGORY)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.sensitive and not entry.syncable and entry.expires_at
    assert stat.S_IMODE(entry.path.stat().st_mode) == 0o600
    payload = json.loads(entry.path.read_text())
    assert payload["digest"] == digest_of(payload["content"]) == summary["evidence_digest"]
    assert len(payload["content"]["observations"]) == 4
    assert payload["content"]["observations"][0]["samples"][0]["depth_m"] == pytest.approx(0.5)
    assert all(s["depth_scale_m"] == 0.0001 for s in summary["frames"])
    assert all(s["valid_ratio"] == 1 for s in summary["frames"])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["stale", "future", "profile_change"])
async def test_readiness_refuses_bad_capture_without_persisting(
    production_port, change: str
) -> None:
    port, cache, camera = production_port
    setattr(camera, change, True)
    with pytest.raises(SkillError):
        await port.metric_readiness(
            frames=2,
            points=32,
            role="fixed",
            stamp_uncertainty_s=0.05,
            ttl_s=60,
            quota_bytes=1048576,
        )
    assert not cache.list_entries(category=CATEGORY)


@pytest.mark.asyncio
async def test_discovered_skill_uses_the_optional_port_without_actuation(production_port) -> None:
    port, _, _ = production_port
    skill = MetricReadiness()
    args = validate_args(skill.spec.args_schema, {"frames": 2, "points": 32})
    ctx = SimpleNamespace(camera_vision=lambda role: port)
    plan = await skill.plan(ctx, args)
    outcome = await skill.execute(ctx, plan)
    assert outcome.side_effect_state == "none"
    assert skill.spec.effect == "read"


@pytest.mark.parametrize("mode", [MODE_FIXED, MODE_WRIST])
def test_metric_motion_adapter_preserves_si_and_frame_direction(mode: str) -> None:
    first, second = frame(), frame(2, 2)
    before = HardwareTransform.from_pose(Pose(100, 0, 200, 0, 0, 0))
    after = HardwareTransform.from_pose(Pose(110, 20, 200, 0, 0, 15))
    sources = tracked_motion(
        first,
        second,
        before,
        after,
        observation_id="pair",
        model=ModelIdentity("raw", 6, "chain"),
        tracks={"identified": ((3, 3), (4, 3))},
        mode=mode,
        scene_id="scene",
        settled=True,
        max_speed_rad_s=0,
        stamp_uncertainty_s=0.05,
    )
    a, b, motion = sources
    assert motion.points[0].first_m[2] == pytest.approx(0.5)
    nominal_a, nominal_b = si_transform(before), si_transform(after)
    expected = (
        nominal_b.compose(nominal_a.inverse())
        if mode == MODE_FIXED
        else nominal_a.inverse().compose(nominal_b)
    )
    assert motion.robot_motion.distance_m(expected) < 1e-12
    assert motion.robot_motion.angle_to_rad(expected) < 1e-7
    assert a.stamp.semantics == b.stamp.semantics == "process_receipt"
    assert motion.source_ids == (a.observation_id, b.observation_id)


def test_adapter_refuses_stream_change_and_sequence_relabeling() -> None:
    identity = HardwareTransform()
    options = dict(
        observation_id="pair",
        model=ModelIdentity("raw", 6, "chain"),
        tracks={"track": ((3, 3), (3, 3))},
        mode=MODE_FIXED,
        scene_id="scene",
        settled=True,
        max_speed_rad_s=0,
        stamp_uncertainty_s=0.05,
    )
    with pytest.raises(ValueError, match="increasing"):
        tracked_motion(frame(), frame(), identity, identity, **options)
    with pytest.raises(ValueError, match="profile changed"):
        tracked_motion(
            frame(), replace(frame(2, 2), depth_scale_m=0.001), identity, identity, **options
        )
    assert si_transform(identity).translation_m == Transform().translation_m


@pytest.mark.asyncio
@pytest.mark.parametrize("role,preview", [("camera", False), ("camera.wrist", True)])
async def test_regression_runner_real_readonly_rpc_auth_and_private_persistence(
    production_port,
    tmp_path: Path,
    role: str,
    preview: bool,
) -> None:
    port, cache, camera = production_port
    directory = tmp_path / "workcells"
    write_workcell(
        directory,
        {
            "workcell": "readiness_cell",
            "roles": {"arm": ARM_ID, role: "camera"},
            "skills": ["calib.metric_readiness"],
        },
    )
    runtime = SkillRuntime(port._registry, settings=SkillRuntimeSettings(workcells_dir=directory))
    assert runtime.list()["workcells"][0]["workcell"] == "readiness_cell"
    server = LhpNodeServer(
        port._registry,
        node_id="test_node",
        token="test-only-token",
        write_mode=WRITE_MODE_OBSERVE_ONLY,
        skill_runtime=runtime,
    )
    host, number = await server.start(host="127.0.0.1", port=0)
    token_file = tmp_path / "test-token"
    token_file.write_text("test-only-token")
    try:
        result = await camera_regression(
            f"tcp://{host}:{number}",
            "readiness_cell",
            role,
            8,
            32,
            0.05,
            str(token_file),
            preview_device="camera" if preview else None,
            preview_channel="frame" if preview else None,
        )
        assert result["node_id"] == "test_node"
        outcome = result["result"]["outcome"]
        assert outcome["status"] == "succeeded", outcome
        assert outcome["side_effect_state"] == "none"
        assert outcome["result"]["frame_count"] == 8
        assert not outcome["result"]["raw_exported"]
        assert "data_b64" not in json.dumps(result) and "test-only-token" not in json.dumps(result)
        assert len(cache.list_entries(category=CATEGORY)) == 1
        assert runtime.status() == ()
        assert result["preview_coexistence"]["verified"] is preview
        if preview:
            assert camera.preview_sequence >= 3
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).parents[1] / "tools" / "calib_regression.py"),
            "camera",
            "--endpoint",
            f"tcp://{host}:{number}",
            "--workcell",
            "unknown_cell",
            "--token-file",
            str(token_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        assert process.returncode == 1, stderr
        assert (
            json.loads(stdout)["content"]["result"]["outcome"]["failure_code"] == "unknown_workcell"
        )
    finally:
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_unavailable", [False, True])
async def test_actual_node_startup_binds_evidence_cache_or_refuses_diagnostics(
    production_port, tmp_path: Path, monkeypatch, cache_unavailable: bool
) -> None:
    from leapflow import config as config_module
    from leapflow.cache import manager as cache_module
    from leapflow.hardware import registry as registry_module
    from leapflow.hardware.lhp.runtime import serve_node

    port, cache, camera = production_port
    registry = port._registry
    registry._cache_manager = None
    layout = build_layout(tmp_path / "home").profile("default")
    write_workcell(
        layout.hardware.workcells_dir,
        {
            "workcell": "readiness_cell",
            "roles": {"arm": ARM_ID, "camera": "camera"},
            "skills": ["calib.metric_readiness"],
        },
    )
    config = SimpleNamespace(
        profile_layout=layout,
        profile_manifest=SimpleNamespace(profile_id="test_profile"),
        hardware_providers="yaml",
        hardware_enabled=True,
    )
    monkeypatch.setattr(config_module, "load_config", lambda: config)

    def use_admitted_registry(settings):
        registry._settings = settings
        return registry

    monkeypatch.setattr(registry_module, "HardwareRegistry", use_admitted_registry)
    if cache_unavailable:

        def unavailable_cache(*args, **kwargs):
            raise OSError("cache index unavailable")

        monkeypatch.setattr(cache_module, "CacheManager", unavailable_cache)
    callbacks = []
    monkeypatch.setattr(
        asyncio.get_running_loop(),
        "add_signal_handler",
        lambda sig, callback: callbacks.append(callback),
    )
    start = LhpNodeServer.start

    async def start_and_exercise(server, **kwargs):
        host, number = await start(server, **kwargs)
        result = await camera_regression(
            f"tcp://{host}:{number}",
            "readiness_cell",
            "camera",
            2,
            32,
            0.05,
        )
        outcome = result["result"]["outcome"]
        if cache_unavailable:
            assert outcome["status"] == "failed"
            assert outcome["failure_code"] == "skill_unavailable"
            assert registry.evidence_cache is None
            assert camera.sequence == 0
        else:
            assert outcome["status"] == "succeeded", outcome
            assert registry.evidence_cache.index_path == cache.index_path
            assert len(cache.list_entries(category=CATEGORY)) == 1
        assert not registry.settings.persist_readings and not registry.settings.stream_enabled
        callbacks[0]()
        return host, number

    monkeypatch.setattr(LhpNodeServer, "start", start_and_exercise)
    assert await serve_node(node_id="test_node", port=0, write_mode=WRITE_MODE_OBSERVE_ONLY) == 0


@pytest.mark.asyncio
async def test_regression_runner_rejects_preview_of_another_device(
    production_port, tmp_path: Path
) -> None:
    port, cache, camera = production_port
    directory = tmp_path / "workcells"
    write_workcell(
        directory,
        {
            "workcell": "readiness_cell",
            "roles": {"arm": ARM_ID, "camera": "camera"},
            "skills": ["calib.metric_readiness"],
        },
    )
    runtime = SkillRuntime(port._registry, settings=SkillRuntimeSettings(workcells_dir=directory))
    server = LhpNodeServer(port._registry, node_id="test_node", skill_runtime=runtime)
    host, number = await server.start(host="127.0.0.1", port=0)
    try:
        with pytest.raises(ValueError, match="exact declared device"):
            await camera_regression(
                f"tcp://{host}:{number}",
                "readiness_cell",
                "camera",
                2,
                32,
                0.05,
                preview_device="other",
                preview_channel="frame",
            )
        assert camera.sequence == 0 and not cache.list_entries(category=CATEGORY)
    finally:
        await server.stop()
