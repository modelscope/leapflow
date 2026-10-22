"""Offline D405 geometry and fail-closed qualification; no physical hardware."""
from __future__ import annotations

import copy
import importlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from leapflow.hardware.rgbd import CameraIntrinsics
import piper_replay as replay
from piper_replay import load_script

INTRINSICS = {
    "width": 640, "height": 480, "fx": 393.31927, "fy": 392.90778,
    "ppx": 313.49576, "ppy": 243.16151, "model": "inverse_brown_conrady",
    "coeffs": [-.051584508, .055745274, .000540769, -.000608815, -.018499207],
}


@pytest.fixture(autouse=True)
def historical_imports(monkeypatch):
    with replay.imports(monkeypatch):
        yield


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    modules = {name: importlib.import_module(f"cl.{name}")
               for name in ("config", "eye", "plan", "wrist_verify", "bridge")}
    cfg = modules["config"].Cfg(copy.deepcopy(modules["config"].DEFAULTS))
    cfg["wrist"]["mount_digest"] = "measured-rigid-mount"
    monkeypatch.setattr(modules["config"], "load_profile", lambda: cfg)
    monkeypatch.setattr(modules["config"], "DATA", tmp_path)
    monkeypatch.setattr(modules["config"], "WRIST_HANDEYE", tmp_path / "wrist.json")
    record = {"geometry_revision": 2, "units": "mm", "qualified": True, "training_count": 8,
              "heldout": {"count": 4, "rms_mm": .2, "working_distance_passed": True},
              "serial_number": cfg.wrist.serial_number, "mount_digest": cfg.wrist.mount_digest,
              "intrinsics": copy.deepcopy(INTRINSICS), "T_flange_camera": np.eye(4).tolist()}
    modules["config"].dump_json(modules["config"].WRIST_HANDEYE, record)
    return SimpleNamespace(**modules, cfg=cfg, record=record,
                           grasp=load_script("grasp_execute"), jaw=load_script("c5_jaw_test"),
                           handeye=load_script("wrist_handeye"), depth=load_script("dual_camera_qualify"))


@pytest.mark.parametrize("pixel,point", [
    ((30., 40.), (-109.6575927734375, -78.78679656982422, 150.)),
    ((600., 430.), (111.05284118652344, 72.38529205322266, 150.)),
    ((313., 243.), (-.18906667828559875, -.06166122481226921, 150.)),
])
def test_d405_deprojection_matches_installed_librealsense(pixel, point):
    # Recorded from pyrealsense2 2.58.4's pure rs2_deproject_pixel_to_point on the node.
    camera = CameraIntrinsics.from_mapping(INTRINSICS)
    assert camera.deproject(*pixel, 150.) == pytest.approx(point, abs=3e-5)
    assert camera.project(camera.deproject(*pixel, 150.)) == pytest.approx(pixel, abs=1e-4)


@pytest.mark.parametrize("fault", ["mount", "rms", "training", "heldout", "distance", "rigid", "revision"])
def test_wrist_camera_rejects_uncertified_geometry(runtime, fault):
    record = copy.deepcopy(runtime.record)
    if fault == "mount":
        record["mount_digest"] = "remounted"
    elif fault == "rms":
        record["heldout"]["rms_mm"] = float("nan")
    elif fault == "training":
        record["training_count"] = 7
    elif fault == "heldout":
        record["heldout"]["count"] = 3
    elif fault == "distance":
        record["heldout"]["working_distance_passed"] = False
    elif fault == "revision":
        record["geometry_revision"] = 1
    else:
        record["T_flange_camera"][0][0] = -1
    with pytest.raises(ValueError):
        runtime.eye.WristCamera([0, 0, 150, -180, 0, 0], record)


def frame_metadata(runtime, number):
    return {"device_id": runtime.config.D405_DEVICE_ID, "sequence": number,
            "received_mono": time.monotonic(), "capture_duration_s": .01,
            "metadata": {"sdk_timestamp_ms": number * 33., "sdk_frame_number": number,
                         "session_id": "camera-session"},
            "camera_metadata": {"session_id": "camera-session", "serial_number": runtime.cfg.wrist.serial_number,
                                "color_intrinsics": copy.deepcopy(INTRINSICS),
                                "active_profile": {"color": {"width": 640, "height": 480}}}}


@pytest.mark.parametrize("fault", ["duplicate", "regression", "serial", "profile", "nan", "session", "missing_session", "metadata_race"])
def test_camera_source_progression_is_not_local_request_sequence(runtime, fault):
    old, new = frame_metadata(runtime, 1), frame_metadata(runtime, 2)
    if fault in ("duplicate", "regression"):
        new["metadata"]["sdk_frame_number"] = 1 if fault == "duplicate" else 0
    elif fault == "nan":
        new["metadata"]["sdk_timestamp_ms"] = float("nan")
    elif fault == "serial":
        new["camera_metadata"]["serial_number"] = "another-camera"
    elif fault in ("session", "missing_session"):
        session = "new-camera-session" if fault == "session" else ""
        new["metadata"]["session_id"] = new["camera_metadata"]["session_id"] = session
    elif fault == "metadata_race":
        new["camera_metadata"]["session_id"] = "reopened-after-preview"
    else:
        new["camera_metadata"]["active_profile"] = {}
    with pytest.raises(ValueError):
        runtime.eye.require_new_frame(new, old)


def synthetic_box(runtime, *, z=20., yaw=0., pose=None):
    camera = runtime.eye.WristCamera(pose or [0, 0, 170, -180, 0, 0], runtime.record)
    theta = np.radians(yaw)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = np.array([[-30, -15], [30, -15], [30, 15], [-30, 15]]) @ rotation.T
    pixels = np.array([camera.pixel_of(x, y, z) for x, y in corners])
    mask = np.zeros((480, 640), np.uint8)
    runtime.wrist_verify.cv2.fillConvexPoly(mask, np.rint(pixels).astype(np.int32), 1)
    image = np.zeros((480, 640, 3), np.uint8)
    image[mask.astype(bool)] = [0, 0, 180]
    plan = {"centre": [0, 0], "long_mm": 60., "short_mm": 30., "height_mm": 20.,
            "table_z": 0., "yaw_long_deg": yaw, "close_mm": 26., "closure": {"why": "effort_limit"}}
    return camera, mask.astype(bool), image, plan


def test_metric_box_pnp_resolves_height_without_assuming_support(runtime):
    for height in (20., 28.):
        camera, blob, _, plan = synthetic_box(runtime, z=height)
        measured = runtime.wrist_verify.box_top_in_base(camera, blob, plan, runtime.cfg)
        assert measured == pytest.approx([0, 0, height], abs=1.0)


@pytest.mark.parametrize("fault", ["age", "delay", "pose", "session", "serial", "source", "camera_session"])
def test_pregrasp_fails_closed_on_frame_correspondence(runtime, fault):
    _, _, image, plan = synthetic_box(runtime)
    pose = [0, 0, 170, -180, 0, 0]
    state = dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose), session_id="arm", gripper_mm=30.)
    before, after, metadata = dict(state), dict(state), frame_metadata(runtime, 1)
    if fault == "age":
        metadata["received_mono"] -= 10
    elif fault == "delay":
        metadata["capture_duration_s"] = float("nan")
    elif fault == "pose":
        before["x"] += 5
    elif fault == "session":
        after["session_id"] = "rebooted"
    elif fault == "serial":
        metadata["camera_metadata"]["serial_number"] = "wrong"
    elif fault == "camera_session":
        metadata["metadata"].pop("session_id")
    else:
        metadata["metadata"]["sdk_frame_number"] = float("nan")
    frame = runtime.wrist_verify.WristFrame(image, metadata, before, after)
    assert runtime.wrist_verify.verify_pre_grasp(pose, frame, plan).verdict == "unknown"


def test_pregrasp_alignment_and_distractor_gates(runtime):
    _, _, image, plan = synthetic_box(runtime)
    pose = [0, 0, 170, -180, 0, 0]
    state = dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose), session_id="arm")
    frame = runtime.wrist_verify.WristFrame(image, frame_metadata(runtime, 1), state, state)
    assert runtime.wrist_verify.verify_pre_grasp(pose, frame, plan).verdict == "aligned"
    shifted = dict(plan, centre=[5, 0])
    assert runtime.wrist_verify.verify_pre_grasp(pose, frame, shifted).verdict == "misaligned"
    assert runtime.wrist_verify.verify_pre_grasp(pose, frame, dict(plan, yaw_long_deg=20)).verdict == "obstructed"
    image[20:50, 20:50] = [0, 0, 180]
    assert runtime.wrist_verify.verify_pre_grasp(pose, frame, plan).verdict == "unknown"


def test_visual_absence_never_proves_empty_jaws(runtime):
    pose = [0, 0, 170, -180, 0, 0]
    state = dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose), session_id="arm", gripper_mm=70.)
    runtime.cfg["wrist"]["inner_pad_tool_mm"] = {"z_mm": 100., "width_bias_mm": 0., "evidence_ref": "calipers"}
    frame = runtime.wrist_verify.WristFrame(np.zeros((480, 640, 3), np.uint8), frame_metadata(runtime, 1), state, state)
    assert runtime.wrist_verify.verify_empty(pose, frame).verdict == "unknown"


def proof_record(z, tool_z):
    return {"pose": [0, 0, tool_z, -180, 0, 0],
            "verdict": {"evidence": {"object_top_base_mm": [0, 0, z]}}}


@pytest.mark.parametrize("object_z", [20., 23., float("nan")])
def test_arm_lift_alone_cannot_pass_proof(runtime, object_z):
    with pytest.raises(ValueError):
        runtime.grasp.verify_proof(proof_record(20, 130), proof_record(object_z, 138),
                                   {"table_z": 0, "height_mm": 20}, runtime.cfg)


def test_independently_measured_object_rise_passes_proof(runtime):
    runtime.grasp.verify_proof(proof_record(20, 130), proof_record(28, 138),
                               {"table_z": 0, "height_mm": 20}, runtime.cfg)


class JawArm:
    def __init__(self, *, fault=None):
        self.fault, self.calls, self.effects_attempted = fault, [], False
        self.hello = {"session_id": "arm"}
        self.width = 70.

    def latest(self, **kwargs):
        return {"z": 190., "tilt_deg": 0., "gripper_mm": self.width, "gripper_nm": -.1}

    def check(self):
        return {"ok": self.fault != "guard", "enabled": True, "halted": False}

    def call(self, name):
        self.calls.append(name)
        self.effects_attempted = True
        return {"verified": self.fault != "convention", "closing_sign": -1, "unloaded_nm": .103}

    def open_gripper(self, width):
        return self.grip(width)

    def grip(self, width, **kwargs):
        self.calls.append(width)
        self.effects_attempted = True
        self.width = width
        return {"why": "timeout" if self.fault == width else "reached"}

    def stop(self):
        self.calls.append("stop")
        return {"stopped": True}


@pytest.mark.parametrize("fault", ["guard", "convention", 10., 40., 70., None])
def test_jaw_test_stops_at_first_failure_and_persists(runtime, tmp_path, fault):
    arm = JawArm(fault=fault)
    path = tmp_path / "jaw.json"
    report = runtime.jaw.run_test(arm, path)
    assert json.loads(path.read_text())["passed"] is (fault is None)
    assert report["passed"] is (fault is None)
    if fault == "guard":
        assert arm.calls == []
        assert report["envelope"]["side_effect_state"] == "none"
    elif fault is not None:
        assert arm.calls[-1] == "stop"
        assert report["envelope"]["side_effect_state"] == "unknown"
        if fault == 10.:
            assert 40. not in arm.calls
    else:
        assert arm.calls == ["gripper_qualify", 70., 10., 40., 68., 70.]


def test_depth_qualification_requires_scale_distance_flatness_and_temporal_noise(runtime):
    stack = np.full((20, 30, 30), 1500, dtype=np.uint16)
    assert runtime.depth.depth_statistics(stack, .0001, [0, 0, 30, 30])["qualified"]
    assert not runtime.depth.depth_statistics(stack, .001, [0, 0, 30, 30])["qualified"]
    noisy = stack.copy()
    noisy[::2] += 30
    assert not runtime.depth.depth_statistics(noisy, .0001, [0, 0, 30, 30])["qualified"]
    stack[:, :25] = 0
    assert not runtime.depth.depth_statistics(stack, .0001, [0, 0, 30, 30])["qualified"]


def test_full_path_ik_requires_every_intermediate_sample(runtime):
    cfg = runtime.cfg
    cfg["qualification"]["planar_error_p95_mm"] = .5
    cfg["qualification"]["angular_error_p95_deg"] = 0
    observation = {"footprint": {"centre_mm": [170, 100], "long_mm": 60., "short_mm": 30.,
                                 "yaw_long_deg": 0.}, "height_mm": 20., "height_quality": "ok",
                                                    "projection_z": -15.5, "table": {"z": -15.5}}
    grasp = runtime.plan.grasp_plan(observation, {"contact_z": 112.92, "table_z": -15.5}, cfg)
    requested = []

    class IkArm:
        def ik(self, poses):
            requested.append(poses)
            return {"results": [{"feasible": True, "margins_deg": [20.] * 6}]}

    with pytest.raises(ValueError, match="incomplete"):
        runtime.grasp.validate_path(IkArm(), grasp, cfg, [150, 130, 190, -180, 0, -90])
    assert requested and all(len(poses) > 20 for poses in requested)


def test_bridge_heartbeat_survives_long_command_and_closes_threads(runtime, monkeypatch):
    daemon = load_script("cl_board/cl_armd")
    monkeypatch.setattr(daemon.ArmService, "_watch", lambda self: None)

    class Guard:
        def calibrate(self, sampler, seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                assert service.lease.alive(), "continuous heartbeat lost"
                time.sleep(.05)
            return {"done": True}

    service = daemon.ArmService(SimpleNamespace(), SimpleNamespace(rate=200), Guard())
    service.state = lambda args: {}
    server = daemon.build_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bridge = runtime.bridge.ArmBridge("127.0.0.1", server.server_address[1])
    try:
        with bridge:
            assert bridge.call("calibrate", seconds=3.3)["done"]
            assert service.lease.ttl == 3.0
        assert not bridge._reader_thread.is_alive()
        assert not bridge._beat_thread.is_alive()
    finally:
        bridge.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def control_server(runtime, monkeypatch):
    """Real localhost sockets, fake CAN writes, no physical hardware."""
    daemon = load_script("cl_board/cl_armd")
    monkeypatch.setattr(daemon.ArmService, "_watch", lambda self: None)
    writes = []
    snapshot = {"z": 190.}
    hardware = SimpleNamespace(EmergencyStop=lambda mode: writes.append(("stop", mode)),
                               EndPoseCtrl=lambda *pose: writes.append(("pose", pose)))
    sampler = SimpleNamespace(rate=200, latest=lambda: dict(snapshot))
    service = daemon.ArmService(hardware, sampler, SimpleNamespace())
    service.state = lambda args: dict(snapshot)
    service.check = lambda args: {"ok": True}
    service.ik = lambda args: {"results": []}
    started, release = threading.Event(), threading.Event()

    def move(args):
        service._enter()
        try:
            service.arm.EndPoseCtrl(1)
            started.set()
            if args.get("wait"):
                assert release.wait(5), "test command was not released"
                service.arm.EndPoseCtrl(2)
            return {"done": True}
        finally:
            service._leave()
    service.move = move
    server = daemon.build_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bridges = [runtime.bridge.ArmBridge("127.0.0.1", server.server_address[1]).connect() for _ in range(2)]
    try:
        yield SimpleNamespace(daemon=daemon, service=service, writes=writes, owner=bridges[0],
                              observer=bridges[1], started=started, release=release, server=server)
    finally:
        release.set()
        for bridge in bridges:
            bridge.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_observer_reads_and_heartbeats_cannot_renew_or_claim_control(runtime, control_server):
    test = control_server
    test.owner.call("move")
    # Freeze owner heartbeats without disconnecting its socket.
    test.owner._stop.set()
    test.owner._beat_thread.join(timeout=3)
    deadline = test.service.lease.deadline
    assert test.observer.call("heartbeat") == {"lease": False}
    for command in ("hello", "state", "check", "ik"):
        test.observer.call(command)
    assert test.service.lease.deadline == deadline
    with pytest.raises(runtime.bridge.BridgeError) as error:
        test.observer.call("move")
    assert error.value.code == "control_owned" and error.value.side_effect_state == "none"
    assert test.writes == [("pose", (1,))]
    # Observer disconnect must not halt the controller.
    test.observer.close()
    assert not test.service.halted


def test_expired_lease_cannot_be_resurrected_by_a_late_heartbeat(runtime, control_server):
    test = control_server
    test.owner.call("move")
    test.owner._stop.set()
    test.owner._beat_thread.join(timeout=3)
    test.service.lease.deadline = time.monotonic() - 1
    assert test.owner.call("heartbeat") == {"lease": False}
    assert not test.service.lease.alive()
    with pytest.raises(runtime.bridge.BridgeError):
        test.owner.call("move")
    assert test.writes == [("pose", (1,))]


@pytest.mark.parametrize("cause", ["disconnect", "observer_stop", "expiry"])
def test_cancelled_command_cannot_write_after_its_checkpoint(runtime, control_server, cause):
    test = control_server
    errors = []
    def run():
        try:
            test.owner.call("move", wait=True)
        except runtime.bridge.BridgeError as error:
            errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    assert test.started.wait(2) and test.writes == [("pose", (1,))]
    if cause == "disconnect":
        test.owner.close()
        deadline = time.monotonic() + 2
        while not test.service.halted and time.monotonic() < deadline:
            time.sleep(.01)
        assert test.service.halted and test.service.lease.owner is None
    elif cause == "observer_stop":
        test.observer.stop()
        assert test.service.halted
    else:
        test.owner._stop.set()
        test.owner._beat_thread.join(timeout=3)
        test.service.lease.deadline = time.monotonic() - 1
    if cause == "expiry":
        test.service._watch_once()
        assert test.service.halt_reason == "lease_expired"
    test.release.set()
    worker.join(timeout=3)
    assert not worker.is_alive() and errors
    assert not any(row == ("pose", (2,)) for row in test.writes)
    assert ("stop", 1) in test.writes


@pytest.mark.parametrize("command", ["move", "grip", "open", "enable", "recover", "chunk", "gripper_qualify"])
def test_bridge_refuses_legacy_daemon_before_any_control_send(runtime, command):
    bridge = runtime.bridge.ArmBridge("fake")
    bridge._sock = SimpleNamespace(sendall=lambda data: pytest.fail("control write must not be sent"))
    bridge.hello = {"version": "legacy", "session_id": "arm"}
    with pytest.raises(runtime.bridge.BridgeError) as error:
        bridge.call(command)
    assert error.value.code == "lease_contract_missing"
    assert error.value.side_effect_state == "none" and not bridge.effects_attempted and not bridge._pending


def test_legacy_daemon_still_accepts_protective_stop_and_observation(runtime):
    bridge = runtime.bridge.ArmBridge("fake")
    requests = []
    def send(data):
        request = json.loads(data)
        requests.append(request["cmd"])
        event, slot = bridge._pending[request["id"]]
        slot["reply"] = {"ok": True, "result": {}}
        event.set()
    bridge._sock = SimpleNamespace(sendall=send)
    bridge.call("state")
    assert not bridge.effects_attempted
    bridge.stop()
    assert requests == ["state", "stop"] and bridge.effects_attempted


@pytest.mark.parametrize("brake_fails", [False, True])
def test_queued_enable_cannot_clear_a_later_stop(control_server, monkeypatch, brake_fails):
    test = control_server
    conn = test.daemon.Conn(SimpleNamespace(), "offline")
    queued_epoch = test.service._halt_epoch
    if brake_fails:
        def fail(mode):
            test.writes.append(("stop", mode))
            raise OSError("fake emergency stop failure")
        monkeypatch.setattr(test.service.arm._arm, "EmergencyStop", fail)
        with pytest.raises(test.daemon.core.Trip, match="stop_failed"):
            test.service.stop({})
    else:
        test.service.stop({})
    response = test.server.RequestHandlerClass._run(
        test.service, "enable", {"confirm": "ENABLE"}, 42, conn, queued_epoch,
    )
    assert response["ok"] is False and response["error"]["code"] == "operator_stop"
    assert test.service.halted and test.writes == [("stop", 1)]


def test_controller_disconnect_halts_even_between_commands(control_server):
    test = control_server
    test.owner.call("move")
    assert not test.service.motion_active and test.service.control_started
    test.owner.close()
    deadline = time.monotonic() + 2
    while not test.service.halted and time.monotonic() < deadline:
        time.sleep(.01)
    assert test.service.halted and test.service.halt_reason == "controller_disconnected"
    assert test.writes == [("pose", (1,)), ("stop", 1)]


def handeye_samples(runtime):
    target = {"pattern_size": [4, 3], "square_size_mm": 5.,
              "mount_digest": runtime.cfg.wrist.mount_digest,
              "T_base_target": runtime.eye.flange_matrix([150, 100, 0, 0, 0, 0]).tolist()}
    mount = runtime.eye.flange_matrix([15, -5, 20, 5, -10, 8])
    samples = {"geometry_revision": 2, "calibration": [], "heldout": [], "serial_number": runtime.cfg.wrist.serial_number,
               "mount_digest": runtime.cfg.wrist.mount_digest, "intrinsics": copy.deepcopy(INTRINSICS)}
    for index in range(12):
        camera_target = runtime.eye.flange_matrix([index % 3, index % 4, 150,
                                                   (index % 3 - 1) * 8, (index % 4 - 1) * 7, index * 3])
        flange = np.asarray(target["T_base_target"]) @ np.linalg.inv(camera_target) @ np.linalg.inv(mount)
        row = {"geometry_revision": 2, "sample_id": f"sample-{index}", "T_base_flange": flange.tolist(),
               "T_camera_target": camera_target.tolist(), "distance_mm": float(np.linalg.norm(camera_target[:3, 3])),
               "pnp_rms_px": .1, "arm_session_id": "arm", "frame": frame_metadata(runtime, index + 1)}
        samples["calibration" if index < 8 else "heldout"].append(row)
    return samples, target, mount


def test_handeye_producer_matches_camera_admission_and_independent_holdout(runtime):
    samples, target, mount = handeye_samples(runtime)
    record = runtime.handeye.solve(samples, target)
    assert record["geometry_revision"] == 2 and record["qualified"]
    assert np.asarray(record["T_flange_camera"]) == pytest.approx(mount, abs=1e-6)
    runtime.eye.WristCamera([0, 0, 150, -180, 0, 0], record)
    # Independent target disagreement must invalidate held-out accuracy, even with a perfect AX=XB solve.
    target["T_base_target"][0][3] += 5
    assert not runtime.handeye.solve(samples, target)["qualified"]


@pytest.mark.parametrize("fault", ["mount", "first_session", "range", "overlap", "arm_session", "degenerate", "old_geometry"])
def test_handeye_refuses_invalid_or_degenerate_samples(runtime, fault):
    samples, target, _ = handeye_samples(runtime)
    if fault == "mount":
        samples["mount_digest"] = "remounted"
    elif fault == "first_session":
        samples["calibration"][0]["frame"]["metadata"].pop("session_id")
    elif fault == "range":
        samples["heldout"][0]["distance_mm"] = float("nan")
    elif fault == "overlap":
        samples["heldout"][0]["sample_id"] = samples["calibration"][0]["sample_id"]
    elif fault == "arm_session":
        samples["heldout"][0]["arm_session_id"] = "rebooted"
    elif fault == "old_geometry":
        samples["calibration"][0].pop("geometry_revision")
    else:
        for row in samples["calibration"]:
            transform = np.array(row["T_base_flange"])
            transform[:3, :3] = np.eye(3)
            row["T_base_flange"] = transform.tolist()
    with pytest.raises((ValueError, KeyError)):
        runtime.handeye.solve(samples, target)


def depth_evidence(runtime):
    evidence = runtime.depth.depth_statistics(np.full((20, 30, 30), 1500, np.uint16), .0001, [0, 0, 30, 30])
    evidence["source_frames"] = [
        {"session_id": "camera-session", "profile_digest": "measured-profile", "aligned_to_color": True,
         "depth_scale_m": .0001, "color_source": {"frame_number": i, "timestamp_ms": i * 33.},
         "depth_source": {"frame_number": i, "timestamp_ms": i * 33.}} for i in range(20)]
    return evidence


@pytest.mark.parametrize("fault", [None, "duplicate_depth", "profile", "missing", "metric"])
def test_depth_report_requires_distinct_raw_sources(runtime, fault):
    evidence = depth_evidence(runtime)
    if fault == "duplicate_depth":
        evidence["source_frames"][1]["depth_source"] = evidence["source_frames"][0]["depth_source"]
    elif fault == "profile":
        evidence["source_frames"][1]["profile_digest"] = "changed"
    elif fault == "missing":
        evidence.pop("source_frames")
    elif fault == "metric":
        evidence["temporal_p95_deviation_mm"] = float("nan")
    assert runtime.depth.qualified_depth(evidence) is (fault is None)


@pytest.mark.parametrize("fault", [None, "speed", "height", "orientation", "ik"])
def test_visual_abort_only_releases_from_a_bounded_settled_pose(runtime, tmp_path, fault):
    grasp = {"centre": [170., 100.], "grasp_z": 135., "retreat_z": 143., "pre_z": 158.,
             "approach_z": 190., "rz": -90., "open_mm": 68.}
    calls = []
    state = dict(zip(("x", "y", "z", "rx", "ry", "rz"), [170., 100., 135., -180., 0., -90.]),
                 speeds_rad_s=[0.] * 6)
    if fault == "speed":
        state["speeds_rad_s"][0] = .5
    elif fault == "height":
        state["z"] = 170.
    elif fault == "orientation":
        state["rz"] += 3

    class AbortArm:
        def latest(self, **kwargs):
            return state
        def check(self):
            return {"ok": True, "enabled": True, "halted": False}
        def ik(self, path):
            return {"results": [] if fault == "ik" else [{"feasible": True}] * len(path)}
        def open_gripper(self, width):
            calls.append("open")
            return {"why": "reached"}
        def move(self, *args, **kwargs):
            calls.append("retreat")
            return {"why": "arrived"}
    if fault:
        with pytest.raises(ValueError):
            runtime.grasp.abort_visual_failure(AbortArm(), grasp, runtime.cfg, tmp_path)
        assert calls == []
    else:
        runtime.grasp.abort_visual_failure(AbortArm(), grasp, runtime.cfg, tmp_path)
        assert calls == ["open", "retreat"]


def report_inputs(runtime, tmp_path):
    runtime.cfg["qualification"].update(independent_dims_mm=[60., 30., 20.],
                                        independent_dims_evidence_ref="calipers", planar_error_p95_mm=.5,
                                        planar_error_evidence_ref="independent-heldout")
    runtime.cfg["wrist"]["inner_pad_tool_mm"] = {"z_mm": 150., "width_bias_mm": 0., "evidence_ref": "pads"}
    report = load_script("qualification_report")
    jaw = runtime.jaw.run_test(JawArm(), tmp_path / "jaw.json")
    poses = [{"label": str(i), "pose": [150., 100., 190., -180., 0., -90.], "joints": [0.] * 6,
              "frame": {"width": 640, "height": 480, "size_bytes": 5000}} for i in range(18)]
    previews = {f"camera.{name}.preview": {"width": 640, "height": 480, "metadata": {"session_id": "camera-session"}}
                for name in ("color", "depth", "infrared_left", "infrared_right")}
    data = {"trajectory": {"phases": {"c4": {"all_arrived": True, "calibration_count": 12, "heldout_count": 6},
                                      "c6": {"repetitions": [{"rep": i, "arrived": True, "planar_err_mm": .1,
                                                              "z_err_mm": .1, "rz_err_deg": .1} for i in range(1, 11)]}}},
            "observations": {"calibration": poses[:12], "heldout": poses[12:]},
            "jaw_latest": jaw, "depth": {"depth": depth_evidence(runtime)}, "wrist": runtime.record,
            "commission": {"dual_camera_preview_passed": True,
                           "cameras": {device: {"previews": previews}
                                       for device in (runtime.config.D435_DEVICE_ID, runtime.config.D405_DEVICE_ID)}}}
    artifacts = {key: {"state": "recorded", "data": value} for key, value in data.items()}
    health = {"can": {"stdout": json.dumps([{"linkinfo": {"info_data": {"state": "ERROR-ACTIVE"}}}])},
              "arm_daemon": {"returncode": 0},
              "arm_feedback": {"returncode": 0, "stdout": "(1.0) can0 2A1#0100000000000000\n" + "\n".join(
                  f"(1.0) can0 26{i}#00E600241C400000" for i in range(1, 7))}}
    cycles = {"cycles": [], "qualified": True, "consecutive_successes": 3, "session_id": "arm"}
    for index in range(3):
        base = index * 10
        def stage(number, verdict, z, tool_z):
            return {"frame": frame_metadata(runtime, base + number), "pose": [0, 0, tool_z, -180, 0, 0],
                    "verdict": {"verdict": verdict, "evidence": {"object_top_base_mm": [0, 0, z]}}}
        cycles["cycles"].append({"success": True, "post_place_error_mm": .5,
                                 "grasp": {"table_z": 0., "height_mm": 20.},
                                 "pre": stage(1, "aligned", 20, 170), "held": stage(2, "held", 20, 130),
                                 "proof": [stage(i, "held", 28, 138) for i in (3, 4, 5)],
                                 "empty": stage(6, "empty", 20, 170)})
    return report, artifacts, health, cycles


@pytest.mark.parametrize("fault", [None, "missing", "descent", "wrist", "jaw", "malformed", "can", "daemon", "proof", "session", "arm_status", "missing_feedback"])
def test_reports_require_complete_independent_evidence(runtime, tmp_path, fault):
    report, artifacts, health, cycles = report_inputs(runtime, tmp_path)
    if fault == "missing":
        artifacts.pop("observations")
    elif fault == "descent":
        artifacts["trajectory"]["data"]["phases"]["c6"]["repetitions"][0]["z_err_mm"] = 20
    elif fault == "wrist":
        artifacts["wrist"]["data"]["geometry_revision"] = 1
    elif fault == "jaw":
        artifacts["jaw_latest"]["data"]["strokes"][-1]["actual_mm"] = 5
    elif fault == "malformed":
        artifacts["trajectory"]["data"] = ["not an evidence object"]
        cycles = {"cycles": [None]}
    elif fault == "can":
        health["can"]["stdout"] = "invalid CAN diagnostics"
    elif fault == "daemon":
        health["arm_daemon"]["returncode"] = 1
    elif fault == "proof":
        cycles["cycles"][0]["proof"][0]["verdict"]["evidence"]["object_top_base_mm"][2] = 20
    elif fault == "session":
        cycles["session_id"] = "restarted-arm"
    elif fault == "arm_status":
        health["arm_feedback"]["stdout"] = health["arm_feedback"]["stdout"].replace("0100000000000000", "0104000200000000")
    elif fault == "missing_feedback":
        health.pop("arm_feedback")
    result = report.summarize(artifacts, health, [], cycles)
    assert result["qualified"] is (fault is None)
    assert bool(result["phase_d"]["blockers"]) is (fault is not None)
    assert result["phase_c"]["qualified"] is (fault in (None, "can", "daemon", "proof", "session", "arm_status", "missing_feedback"))


def test_grasp_connection_failure_has_durable_no_effect_checkpoint(runtime, monkeypatch):
    class Unreachable:
        effects_attempted = False
        hello = {}
        def __init__(self, *args):
            pass
        def connect(self):
            raise runtime.bridge.BridgeError("unreachable", "offline test")
        def close(self):
            pass
    monkeypatch.setattr(runtime.grasp, "ArmBridge", Unreachable)
    monkeypatch.setattr(runtime.grasp, "preconditions", lambda cfg: [])
    monkeypatch.setattr(sys, "argv", ["grasp_execute.py", "--execute"])
    assert runtime.grasp.main() == 1
    paths = list(runtime.config.DATA.glob("dual_grasp/*/report.json"))
    assert len(paths) == 1
    stored = json.loads(paths[0].read_text())
    assert stored["envelope"]["side_effect_state"] == "none"
    assert stored["qualified"] is False and stored["cycles"] == []


def test_missing_handeye_backend_blocks_before_connection(runtime, monkeypatch):
    monkeypatch.delattr(runtime.handeye.cv2, "calibrateHandEye", raising=False)
    monkeypatch.setattr(runtime.handeye, "ArmBridge", lambda *args: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="calibrateHandEye unavailable"):
        runtime.handeye.collect({})


@pytest.mark.parametrize("yaw,expected", [(90., "held"), (0., "unknown")])
def test_hold_requires_pad_engagement_at_box_boundary_not_inside_mask(runtime, yaw, expected):
    pose = [0, 0, 170, -180, 0, yaw]
    _, _, image, plan = synthetic_box(runtime, pose=pose)
    runtime.cfg["wrist"]["inner_pad_tool_mm"] = {"z_mm": 150., "width_bias_mm": 0., "evidence_ref": "pads"}
    state = dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose), session_id="arm", gripper_mm=30.)
    frame = runtime.wrist_verify.WristFrame(image, frame_metadata(runtime, 1), state, state)
    verdict = runtime.wrist_verify.verify_hold(pose, frame, plan)
    assert verdict.verdict == expected
    assert verdict.evidence["contact_proven"] is False


@pytest.mark.parametrize("object_z,expected", [(20., "empty"), (50., "unknown")])
def test_release_requires_object_on_support_below_pads(runtime, object_z, expected):
    pose = [0, 0, 200, -180, 0, 90]
    _, _, image, plan = synthetic_box(runtime, z=object_z, pose=pose)
    runtime.cfg["wrist"]["inner_pad_tool_mm"] = {"z_mm": 150., "width_bias_mm": 0., "evidence_ref": "pads"}
    state = dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose), session_id="arm", gripper_mm=68.)
    frame = runtime.wrist_verify.WristFrame(image, frame_metadata(runtime, 1), state, state)
    assert runtime.wrist_verify.verify_empty(pose, frame, plan).verdict == expected


@pytest.mark.parametrize("fault", [None, "open", "close", "reopen", "nonfinite", "stop", "lease"])
def test_daemon_gripper_qualification_never_installs_a_partial_convention(runtime, monkeypatch, fault):
    daemon = load_script("cl_board/cl_armd")
    monkeypatch.setattr(daemon.ArmService, "_watch", lambda self: None)
    monkeypatch.setattr(daemon.time, "sleep", lambda seconds: None)
    snap = {"gripper_enabled": True, "gripper_mm": 70., "gripper_nm": .103}
    calls = []

    class Hardware:
        def GripperCtrl(self, width, *args):
            calls.append(width / daemon.core.SCALE)
            snap["gripper_mm"] = width / daemon.core.SCALE
            snap["gripper_nm"] = .103 if width / daemon.core.SCALE == 70. else -.12
            if fault == "open" and len(calls) == 1:
                snap["gripper_mm"] = 60.
            elif fault == "close" and len(calls) == 2:
                snap["gripper_mm"] = 70.
            elif fault == "reopen" and len(calls) == 3:
                snap["gripper_mm"] = 50.
            elif fault == "nonfinite" and len(calls) == 2:
                snap["gripper_nm"] = float("nan")
            elif fault == "stop" and len(calls) == 2:
                service.halted = True
            elif fault == "lease" and len(calls) == 2:
                service.lease.deadline = float("-inf")
        def EmergencyStop(self, mode):
            calls.append("stop")
    guard = SimpleNamespace(gripper_convention={"verified": True}, check=lambda snap, **kwargs: None)
    service = daemon.ArmService(Hardware(), SimpleNamespace(rate=200, latest=lambda: dict(snap)), guard)
    if fault:
        with pytest.raises(daemon.core.Trip):
            service.gripper_qualify({})
        assert service.halted and guard.gripper_convention is None
        assert calls[-1] == "stop"
        if fault in ("close", "nonfinite", "stop", "lease"):
            assert calls == [70., 50., "stop"]
    else:
        result = service.gripper_qualify({})
        assert guard.gripper_convention == result and result["verified"]
        assert result["closing_sign"] == -1 and calls == [70., 50., 70.]
    assert service.motion_active is False and not service.motion.locked()


def test_report_does_not_substitute_historical_pass_or_leak_raw_payload(runtime, tmp_path):
    report, artifacts, health, cycles = report_inputs(runtime, tmp_path)
    artifacts["jaw_previous"] = artifacts.pop("jaw_latest")
    artifacts["wrist_candidate"] = artifacts.pop("wrist")
    result = report.summarize(artifacts, health, [], cycles)
    assert not result["qualified"] and not result["phase_c"]["jaw"]["qualified"]
    assert not result["phase_c"]["wrist"]["qualified"]
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"frame_b64": "raw", "nested": {"token": "secret", "value": float("nan")}}))
    evidence = report.read_artifact(path)
    assert len(evidence["sha256"]) == 64
    assert evidence["data"] == {"nested": {"value": None}}


def test_passive_feedback_cannot_confuse_enablement_with_motion_health(runtime, tmp_path):
    report, _, health, _ = report_inputs(runtime, tmp_path)
    probe = health["arm_feedback"]
    assert report.passive_arm_health(health)["qualified"]
    for replacement in ("0104000200000000", "0200000000000000", "0100000000000001"):
        changed = copy.deepcopy(health)
        changed["arm_feedback"]["stdout"] = probe["stdout"].replace("0100000000000000", replacement)
        assert not report.passive_arm_health(changed)["qualified"]
    changed = copy.deepcopy(health)
    changed["arm_feedback"]["stdout"] = probe["stdout"].replace("00E600241C400000", "00E600241C000000")
    assert not report.passive_arm_health(changed)["qualified"]
    changed["arm_feedback"]["stdout"] = probe["stdout"].replace("00E600241C400000", "00E600241C600000")
    actual = report.passive_arm_health(changed)
    assert all(actual["motor_enabled"].values()) and not actual["qualified"]


@pytest.mark.parametrize("key", ["clearance_margin_mm", "proof_lift_mm", "proof_rise_error_mm", "post_place_error_mm"])
def test_nonfinite_qualification_budgets_fail_closed(runtime, key):
    runtime.cfg["qualification"][key] = float("nan")
    assert runtime.grasp.preconditions(runtime.cfg)

