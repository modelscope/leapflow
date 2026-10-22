"""Metric qualification and geometry using lossless synthetic RGB-D only."""
from dataclasses import asdict, replace
import time

import numpy as np
import pytest

from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.rgbd_quality import DepthLimits, RegionDepthProcessor, qualify_depth
from leapflow.hardware.rgbd_geometry import CuboidSeed, fit_cuboid_surfaces, inner_jaw_midpoint
from leapflow.hardware.rgbd_observation import ColorGeometryProcessor


LIMITS = DepthLimits(9, .6, 3, 2, 20, 5, 1, 3, 1)


def frames():
    now = time.monotonic()
    depth = np.full((12, 16), 700, dtype=np.uint16)
    depth[2:10, 2:10] = 500
    result = []
    for i in range(3):
        source = {"frame_number": i + 1, "timestamp_ms": 33 * i + 1,
                  "timestamp_domain": "hardware_clock"}
        result.append(RgbdFrame(
            device_id="camera", width=16, height=12, color=bytes(16 * 12 * 3),
            depth=depth.tobytes(), depth_scale_m=.001,
            intrinsics=CameraIntrinsics(16, 12, 100, 100, 8, 6),
            observed_at=time.time(), monotonic_at=now - .1 + i * .03, sequence=i + 1,
            metadata={"color_source": dict(source), "depth_source": dict(source),
                      "aligned_to_color": True, "session_id": "camera-session",
                      "profile_digest": "profile", "mount_digest": "mount", "calibration_digest": "cal"}))
    return result


def masks():
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:10, 2:10] = True
    back = np.zeros_like(mask)
    back[:, 12:16] = True
    return mask, back


def qualify(items, *, uncertainty=.5):
    return qualify_depth(items, *masks(), limits=LIMITS, now=time.monotonic(),
                         calibration_digest="cal", mount_digest="mount", uncertainty_mm=uncertainty)


def test_distinct_metric_bundle_qualifies_without_inferring_uncertainty():
    evidence = qualify(frames())
    assert evidence.qualified and evidence.distinct_frames == 3
    assert evidence.median_range_mm == pytest.approx(500)
    assert evidence.background_separation_mm == pytest.approx(200)
    assert not qualify(frames(), uncertainty=None).qualified


@pytest.mark.parametrize("fault,reason", [
    ("duplicate", "duplicate_frame"), ("depth_duplicate", "source_frame_not_advancing"),
    ("skew", "rgb_depth_skew"), ("domain", "source_clock_unqualified"),
    ("mount", "calibration_or_mount_mismatch"), ("profile", "profile_or_calibration_changed"),
    ("stale", "frame_stale"), ("background", "foreground_background_unresolved"),
    ("holes", "interior_depth_insufficient"),
])
def test_unqualified_bundle_is_never_repaired_from_backdrop(fault, reason):
    items = frames()
    metadata = dict(items[-1].metadata)
    if fault == "duplicate":
        items[-1] = items[-2]
    elif fault == "depth_duplicate":
        metadata["depth_source"] = dict(metadata["depth_source"], frame_number=2)
    elif fault == "skew":
        metadata["depth_source"] = dict(metadata["depth_source"], timestamp_ms=1000)
    elif fault == "domain":
        metadata["depth_source"] = dict(metadata["depth_source"], timestamp_domain="system_time")
    elif fault in ("mount", "profile"):
        metadata[fault + "_digest"] = "other"
    elif fault == "stale":
        items[-1] = replace(items[-1], monotonic_at=time.monotonic() - 2)
    elif fault in ("background", "holes"):
        depth = np.full((12, 16), 700, dtype=np.uint16)
        if fault == "holes":
            depth[2:10, 2:10] = 0
        items[-1] = replace(items[-1], depth=depth.tobytes())
    if fault != "duplicate":
        items[-1] = replace(items[-1], metadata=metadata)
    evidence = qualify(items)
    assert not evidence.qualified and reason in evidence.reasons


def test_landmarks_need_valid_native_center_and_foreground_support():
    frame = frames()[-1]
    support, _ = masks()
    kwargs = dict(independent=True, radius_px=1, min_samples=3, support_masks=(support, support))
    assert inner_jaw_midpoint(frame, [(4, 4), (7, 4)], **kwargs) is not None
    depth = np.array(frame.depth_units()).reshape(12, 16)
    depth[4, 4] = 0
    hole = replace(frame, depth=depth.tobytes())
    assert inner_jaw_midpoint(hole, [(4, 4), (7, 4)], **kwargs) is None
    assert inner_jaw_midpoint(frame, [(4, 4), (7, 4)], independent=False, radius_px=1, min_samples=3) is None
    depth[4, 4] = 700
    assert inner_jaw_midpoint(replace(frame, depth=depth.tobytes()), [(4, 4), (7, 4)], **kwargs) is None


def test_cuboid_preserves_hypotheses_and_binds_frame_mask():
    items = frames()
    mask, _ = masks()
    seeds = [CuboidSeed(name, (40, 40, 20), (-10, 0, 510), (0, 0, yaw))
             for name, yaw in (("face-a", 0), ("face-b", 90))]
    contour = [(x, 2) for x in range(2, 10)] + [(x, 9) for x in range(2, 10)]
    contour += [(2, y) for y in range(3, 9)] + [(9, y) for y in range(3, 9)]
    kwargs = dict(evidence=qualify(items), surface_sigma_mm=1, contour_sigma_px=1, max_iterations=2)
    fits = fit_cuboid_surfaces(items[-1], mask, contour, seeds, **kwargs)
    assert [f.hypothesis_id for f in fits] == ["face-a", "face-b"]
    assert all(np.isfinite(f.surface_rms_mm) for f in fits)
    with pytest.raises(ValueError, match="frame and mask"):
        fit_cuboid_surfaces(items[-1], ~mask, contour, seeds, **kwargs)
    with pytest.raises(ValueError, match="frame and mask"):
        fit_cuboid_surfaces(replace(items[-1], metadata=dict(items[-1].metadata, session_id="reboot")), mask, contour, seeds, **kwargs)


def depth_processor_config(**changes):
    config = {
        "limits": asdict(LIMITS),
        "roi": [2, 2, 10, 10],
        "background_roi": [12, 0, 16, 12],
        "calibration_digest": "cal",
        "mount_digest": "mount",
        "uncertainty_mm": .5,
        "max_window_bytes": 5000,
    }
    config.update(changes)
    return config


def geometry_processor_config(**changes):
    selector = lambda roi, bgr, lo=1, hi=200: {
        "roi": roi, "lower_bgr": [max(0, b - 1) for b in bgr],
        "upper_bgr": [min(255, b + 1) for b in bgr],
        "min_area_px": lo, "max_area_px": hi,
    }
    config = {
        "limits": asdict(LIMITS),
        "calibration_digest": "cal",
        "mount_digest": "mount",
        "uncertainty_mm": .5,
        "max_window_bytes": 50000,
        "scene_id": "synthetic-scene",
        "background_roi": [12, 0, 16, 12],
        "object": selector([0, 0, 16, 12], [255, 0, 0], 10, 200),
        "jaws": [selector([0, 0, 16, 12], [0, 255, 0], 1, 200),
                 selector([0, 0, 16, 12], [0, 0, 255], 1, 200)],
        "cuboid_seeds": [{"hypothesis_id": "standing", "dimensions_mm": [40, 40, 20],
                           "center_camera_mm": [-10, 0, 510], "rpy_deg": [0, 0, 0]}],
        "surface_sigma_mm": 1,
        "contour_sigma_px": 1,
        "jaw_dispersion_mm": 5,
        "jaw_radius_px": 1,
        "jaw_min_samples": 3,
    }
    config.update(changes)
    return config


def colored_frames():
    """16x12 synthetic BGR frames: blue box 2..10, green/red pad regions, background at depth 700."""
    result = []
    for frame in frames():
        # Frame depth has box region at 500 (depth 2:10, 2:10) and 700 elsewhere.
        image = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
        # Object is BGR [255, 0, 0] == blue channel only.
        image[2:10, 2:10] = [255, 0, 0]
        # Two jaw pads in disjoint sub-regions.
        image[3:6, 3:5] = [0, 255, 0]
        image[3:6, 7:9] = [0, 0, 255]
        result.append(replace(frame, color=image.tobytes()))
    return result


@pytest.mark.parametrize("field,value", [
    ("max_window_bytes", True), ("calibration_digest", ""), ("mount_digest", None),
    ("roi", [2.5, 2, 10, 10]),
])
def test_region_processor_refuses_untyped_or_unbound_configuration(field, value):
    with pytest.raises(ValueError):
        RegionDepthProcessor(depth_processor_config(**{field: value}))


def test_region_processor_resets_on_source_identity_and_enforces_byte_budget():
    items = frames()
    clock = lambda: items[-1].monotonic_at + .1
    processor = RegionDepthProcessor(depth_processor_config(), clock=clock)
    assert processor.measure(items[0]).distinct_frames == 1
    assert not processor.measure(items[1]).qualified
    assert processor.measure(items[2]).qualified

    rebooted = [replace(frame, metadata=dict(frame.metadata, session_id="reboot")) for frame in items]
    assert processor.measure(rebooted[0]).distinct_frames == 1
    assert not processor.measure(rebooted[1]).qualified
    assert processor.measure(rebooted[2]).qualified
    assert processor.measure(rebooted[1]).qualified is False  # Regressed source order stays unqualified.

    too_small = RegionDepthProcessor(depth_processor_config(max_window_bytes=959), clock=clock)
    with pytest.raises(ValueError, match="exceeds"):
        too_small.measure(items[0])
    assert too_small._frames == []


def test_geometry_processor_uses_lossless_depth_and_independent_visible_pads():
    items = colored_frames()
    processor = ColorGeometryProcessor(geometry_processor_config(),
                                       clock=lambda: items[-1].monotonic_at + .1)
    evidence = None
    for frame in items:
        evidence = processor.measure(frame)
    assert evidence is not None
    # The processor should at least produce depth evidence and proposals.
    assert len(evidence.proposals) >= 1
    assert [proposal.hypothesis_id for proposal in evidence.proposals] == ["standing"]
    assert evidence.qualified is False  # GeometryEvidence.qualified is always False.
    assert evidence.scene_id == "synthetic-scene"


def test_geometry_processor_refuses_oversize_frame_and_ambiguous_pad_selector():
    item = colored_frames()[0]
    processor = ColorGeometryProcessor(geometry_processor_config(max_window_bytes=959))
    with pytest.raises(ValueError, match="exceeds"):
        processor.measure(item)
    assert processor._frames == []

    ambiguous = geometry_processor_config()
    ambiguous["jaws"][1] = dict(ambiguous["jaws"][0])
    processor = ColorGeometryProcessor(ambiguous, clock=lambda: item.monotonic_at + .1)
    evidence = processor.measure(item)
    assert not evidence.independent_jaws and evidence.jaw_midpoint_camera_mm is None
