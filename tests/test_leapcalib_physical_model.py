# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic dual-model geometry; synthetic geometry is NOT physical qualification."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from leapcalib.contracts.identity import digest_of
from leapcalib.geometry.se3 import Transform as SiTransform
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry, check_dual_sweep
from leapflow.hardware.skills.adapt.ik import solve_ik
from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.geometry import WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain, DhLink, KinematicModel

pytestmark = pytest.mark.component


def model(offset: float = 0, tool_m: float = 0) -> PhysicalModel:
    chain = DhChain((DhLink(100, 0, 0),), "standard")
    return PhysicalModel(chain, (offset,), SiTransform(translation_m=(tool_m, 0, 0)), digest_of(chain.to_dict()), "synthetic-deps", "synthetic-certificate")


def geometry(**changes: object) -> SafetyGeometry:
    return SafetyGeometry(**{
        "link_radii_mm": (1, 1),
        "whole_link_workspace": WorkspaceBox(-500, 500, -500, 500, -500, 500),
        "keepouts": (WorkspaceBox(300, 310, 300, 310, 300, 310),),
        "provenance_digest": "synthetic-isolated-test",
        "uncertainty_mm": 0.1, "sensing_mm": 0.1, "timing_mm": 0.1, "arrival_mm": 0.1,
        **changes,
    })


def test_zero_applied_once_tool_si_and_roundtrip() -> None:
    corrected = model(math.pi / 2, 0.01)
    assert isinstance(corrected, KinematicModel)
    assert corrected.raw_to_physical_deg((0,)) == pytest.approx((90,))
    assert corrected.transform((0,)).translation == pytest.approx((0, 110, 0))
    assert corrected.nominal.transform((0,)).translation == pytest.approx((100, 0, 0))
    assert PhysicalModel.from_mapping(corrected.to_dict()) == corrected
    malformed = {**corrected.to_dict(), "composition": "subtract_zero_after_ik"}
    with pytest.raises(ValueError, match="coordinates/composition"):
        PhysicalModel.from_mapping(malformed)
    with pytest.raises(ValueError, match="coordinates/composition"):
        PhysicalModel.from_mapping({**corrected.to_dict(), "extra": True})


def test_corrected_ik_returns_raw_equivalent_to_physical_solve() -> None:
    corrected = model(math.radians(2))
    target = corrected.transform((5,))
    settings = IkSettings(tolerance_mm=0.001, tolerance_deg=0.001)
    raw = solve_ik(corrected, target, (0,), ((-30, 30),), settings)
    physical = solve_ik(corrected.nominal, target, (2,), ((-28, 32),), settings)
    assert raw == pytest.approx((5,), abs=0.001)
    assert raw == pytest.approx((physical[0] - 2,), abs=0.001)
    # Subtracting after a RAW-coordinate solve is a demonstrable double correction.
    double_applied = (raw[0] - 2,)
    assert corrected.transform(double_applied).to_pose().distance_mm(target.to_pose()) > 3
    assert corrected.nominal.agreement((5,), corrected.nominal.forward((5,))) == (0, 0)


def test_nominal_chain_is_defensively_frozen() -> None:
    links = [DhLink(100, 0, 0)]
    chain = DhChain(links, "standard")
    corrected = PhysicalModel(chain, (0,), SiTransform(), digest_of(chain.to_dict()), "deps", "certificate")
    links.append(DhLink(50, 0, 0))
    assert corrected.dof == 1
    assert corrected.transform((0,)).translation == (100, 0, 0)


def test_dual_sweep_clearance_uncertainty_and_path_sampling() -> None:
    nominal, corrected = model(), model(math.radians(1), 0.001)
    result = check_dual_sweep(nominal, corrected, (0,), (1,), geometry())
    assert result["samples_per_model"] == 3
    assert result["uncertainty_mm"] > 0.4
    assert result["minimum_keepout_clearance_mm"] > 30
    with pytest.raises(ValueError, match="boundary"):
        check_dual_sweep(nominal, corrected, (0,), (1,), geometry(whole_link_workspace=WorkspaceBox(-500, 121.5, -500, 500, -500, 500)))
    with pytest.raises(ValueError, match="clearance"):
        check_dual_sweep(nominal, corrected, (0,), (1,), geometry(keepouts=(WorkspaceBox(110, 120, -1, 1, -1, 1),)))
    with pytest.raises(ValueError, match="sampling"):
        check_dual_sweep(nominal, corrected, (0,), (5,), geometry(max_path_samples=2))
    with pytest.raises(ValueError, match="every link"):
        check_dual_sweep(nominal, corrected, (0,), (1,), geometry(link_radii_mm=(1,)))
    assert replace(geometry(), uncertainty_mm=10).uncertainty_total_mm > 10


def test_intermediate_sweep_is_checked_even_when_endpoints_clear() -> None:
    nominal = model()
    obstacle = WorkspaceBox(68, 72, 68, 72, -2, 2)
    bounds = geometry(keepouts=(obstacle,), minimum_clearance_mm=5)
    check_dual_sweep(nominal, nominal, (0,), (0,), bounds)
    check_dual_sweep(nominal, nominal, (90,), (90,), bounds)
    with pytest.raises(ValueError, match="clearance"):
        check_dual_sweep(nominal, nominal, (0,), (90,), bounds)


@pytest.mark.parametrize("changes", [{"uncertainty_mm": float("nan")}, {"arrival_mm": -1}, {"path_step_deg": 0}, {"max_path_samples": True}])
def test_geometry_rejects_nonfinite_and_invalid_budgets(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        geometry(**changes)
