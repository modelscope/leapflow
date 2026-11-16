"""RGB-D value types: intrinsics geometry, frame validation, and encoding."""

from __future__ import annotations

from array import array

import pytest

from leapflow.hardware.rgbd import (
    DISTORTION_BROWN_CONRADY,
    DISTORTION_INVERSE_BROWN_CONRADY,
    DISTORTION_NONE,
    CameraIntrinsics,
    RgbdFrame,
)

_COEFFS = (0.12, -0.05, 0.001, -0.002, 0.01)


def _intrinsics(model: str = DISTORTION_NONE, coeffs: tuple[float, ...] = ()) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640,
        height=480,
        fx=610.0,
        fy=608.0,
        ppx=321.5,
        ppy=239.0,
        model=model,
        coeffs=coeffs or (0.0,) * 5,
    )


def test_principal_point_deprojects_onto_the_optical_axis() -> None:
    assert _intrinsics().deproject(321.5, 239.0, 500.0) == pytest.approx((0.0, 0.0, 500.0))


@pytest.mark.parametrize(
    "model", [DISTORTION_NONE, DISTORTION_BROWN_CONRADY, DISTORTION_INVERSE_BROWN_CONRADY]
)
@pytest.mark.parametrize("pixel", [(321.5, 239.0), (40.0, 30.0), (600.0, 450.0), (100.0, 400.0)])
def test_project_is_the_inverse_of_deproject(model: str, pixel: tuple[float, float]) -> None:
    intrinsics = _intrinsics(model, _COEFFS if model != DISTORTION_NONE else ())

    point = intrinsics.deproject(*pixel, 750.0)

    assert point[2] == pytest.approx(750.0)
    assert intrinsics.project(point) == pytest.approx(pixel, abs=1e-3)


def test_distortion_moves_off_axis_points() -> None:
    plain = _intrinsics().deproject(40.0, 30.0, 500.0)
    distorted = _intrinsics(DISTORTION_INVERSE_BROWN_CONRADY, _COEFFS).deproject(40.0, 30.0, 500.0)

    assert plain[:2] != pytest.approx(distorted[:2], abs=0.5)


def test_a_point_behind_the_camera_does_not_project() -> None:
    with pytest.raises(ValueError, match="behind"):
        _intrinsics().project((0.0, 0.0, -1.0))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"width": 0}, "image size"),
        ({"fx": 0.0}, "focal"),
        ({"model": "kannala_brandt4", "coeffs": (0.1, 0.0, 0.0, 0.0, 0.0)}, "not supported"),
    ],
)
def test_unusable_intrinsics_are_refused(overrides: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "width": 640,
        "height": 480,
        "fx": 610.0,
        "fy": 608.0,
        "ppx": 320.0,
        "ppy": 240.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        CameraIntrinsics(**values)  # type: ignore[arg-type]


def test_an_unknown_model_without_coefficients_is_a_pinhole() -> None:
    intrinsics = CameraIntrinsics(640, 480, 610.0, 608.0, 320.0, 240.0, model="kannala_brandt4")

    assert intrinsics.deproject(320.0, 240.0, 100.0) == pytest.approx((0.0, 0.0, 100.0))


def test_intrinsics_round_trip_through_a_mapping() -> None:
    intrinsics = _intrinsics(DISTORTION_BROWN_CONRADY, _COEFFS)

    assert CameraIntrinsics.from_mapping(intrinsics.to_dict()) == intrinsics


def test_differs_from_names_the_first_difference() -> None:
    base = _intrinsics()

    assert base.differs_from(_intrinsics()) == ""
    assert base.differs_from(CameraIntrinsics(1280, 720, 610.0, 608.0, 321.5, 239.0)).startswith(
        "resolution"
    )
    assert "ppx" in base.differs_from(CameraIntrinsics(640, 480, 610.0, 608.0, 330.0, 239.0))
    assert base.differs_from(CameraIntrinsics(640, 480, 610.2, 608.0, 321.5, 239.0)) == ""


def _frame(**overrides: object) -> RgbdFrame:
    values: dict[str, object] = {
        "device_id": "cam",
        "width": 2,
        "height": 2,
        "color": bytes(range(12)),
        "depth": array("H", [0, 1000, 250, 4000]).tobytes(),
        "depth_scale_m": 0.001,
        "intrinsics": CameraIntrinsics(2, 2, 2.0, 2.0, 1.0, 1.0),
        "observed_at": 1.0,
        "monotonic_at": 1.0,
    }
    values.update(overrides)
    return RgbdFrame(**values)  # type: ignore[arg-type]


def test_depth_is_read_in_millimetres_with_zero_meaning_no_measurement() -> None:
    frame = _frame()

    assert frame.depth_mm(0, 0) == 0.0
    assert frame.depth_mm(1, 0) == pytest.approx(1000.0)
    assert frame.depth_mm(0, 1) == pytest.approx(250.0)
    assert len(frame.depth_units()) == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"color": b"\x00" * 11}, "colour buffer"),
        ({"depth": b"\x00" * 6}, "aligned"),
        ({"intrinsics": CameraIntrinsics(4, 2, 2.0, 2.0, 1.0, 1.0)}, "different image size"),
        ({"depth_scale_m": 0.0}, "depth scale"),
    ],
)
def test_inconsistent_frames_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _frame(**overrides)


def test_colour_encodes_as_a_full_resolution_jpeg() -> None:
    pytest.importorskip("PIL")

    data = _frame().color_jpeg(quality=80)

    assert data.startswith(b"\xff\xd8")
