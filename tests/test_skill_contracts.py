# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-skill contracts: dependency rule, digests, argument validation, geometry, wrap.

These are the invariants every skill pack relies on, so they are pinned here
independently of any one skill's behaviour.
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from leapflow.hardware.node_gate import NodeWriteGate, is_cyclic, rate_delta
from leapflow.hardware.skills.args import validate_args
from leapflow.hardware.skills.builtin.calibration import skills as calibration_skills
from leapflow.hardware.skills.builtin.camera_calibration import skills as camera_skills
from leapflow.hardware.skills.builtin.episode import skills as episode_skills
from leapflow.hardware.skills.builtin.grasp_cycle import skills as cycle_skills
from leapflow.hardware.skills.builtin.grasp_verify import skills as verify_skills
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.policy import skills as policy_skills
from leapflow.hardware.skills.builtin.scene import skills as scene_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.digest import plan_digest, run_permit_digest
from leapflow.hardware.skills.geometry import (
    Pose,
    Transform,
    WorkspaceBox,
    matrix_to_rpy,
    rpy_to_matrix,
    tool_cone_deg,
)
from leapflow.hardware.skills.ports import (
    NodeSkill,
)
from leapflow.hardware.skills.protocol import (
    EFFECT_ACTUATE,
    EFFECT_READ,
    FAILURE_INVALID_ARGS,
    SkillError,
    SkillPlan,
    SkillSpec,
)
from tests._harness.skill_stubs import ARM_ID, SkillNodeRegistry, piper_context

_SKILLS = Path(__file__).parents[1] / "src" / "leapflow" / "hardware" / "skills"
_BUILTIN = _SKILLS / "builtin"
_ALLOWED = {
    "leapflow.hardware.skills.protocol",
    "leapflow.hardware.skills.ports",
    "leapflow.hardware.skills.geometry",
    # Measurement maths only -- clouds and screws in, a camera mount out -- and the
    # same rule as geometry: no transport, no device, no registry.
    "leapflow.hardware.skills.screw",
}
# The adapter is pure embodiment math: declarations in, vectors out. It knows no
# workcell, transport, or binding, so training-side converters can import it alone.
# PIL is the node's existing lazy JPEG codec (observation images only).
_ADAPT_ALLOWED = {
    "leapflow.hardware.skills.geometry",
    "leapflow.hardware.skills.kinematics",
    "PIL",
}
# Routing words in a card steer the model toward one skill; cards state
# capability, arguments, and limits, and the model chooses.
_ROUTING_WORDS = ("prefer", "instead of", "rather than", "always use", "use this when")


# ── Dependency rule ──────────────────────────────────────────────────────────


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("path", sorted(_BUILTIN.glob("*.py")), ids=lambda p: p.name)
def test_builtin_skills_import_only_the_skill_contract_and_stdlib(path: Path) -> None:
    for module in _imports(path):
        top = module.split(".")[0]
        assert module in _ALLOWED or top in sys.stdlib_module_names or top == "__future__", (
            f"{path.name} imports {module}; skills reach devices only through SkillContext"
        )


@pytest.mark.parametrize("path", sorted((_SKILLS / "adapt").glob("*.py")), ids=lambda p: p.name)
def test_the_action_adapter_imports_only_math_and_stdlib(path: Path) -> None:
    for module in _imports(path):
        top = module.split(".")[0]
        assert (
            module in _ADAPT_ALLOWED
            or module.startswith("leapflow.hardware.skills.adapt")
            or top in sys.stdlib_module_names
            or top == "__future__"
        ), f"adapt/{path.name} imports {module}; the adapter takes declarations, not bindings"


def _builtin_skills() -> tuple[NodeSkill, ...]:
    return (
        *motion_skills(),
        *sequence_skills(),
        *calibration_skills(),
        *camera_skills(),
        *scene_skills(),
        *policy_skills(),
        *episode_skills(),
        *cycle_skills(),
        *verify_skills(),
    )


def test_builtin_skills_satisfy_the_protocol_with_unique_ids() -> None:
    found = _builtin_skills()
    assert all(isinstance(skill, NodeSkill) for skill in found)
    ids = [skill.spec.skill_id for skill in found]
    assert len(ids) == len(set(ids))
    reads = {skill.spec.skill_id for skill in found if skill.spec.effect == EFFECT_READ}
    assert reads == {
        "arm.state",
        "perceive.locate",
        "perceive.snapshot",
        "perceive.lift",
        "grasp.propose",
        "episode.record",
        "grasp.verify",
        "grasp.inspect",
    }
    assert all(skill.spec.effect in (EFFECT_ACTUATE, EFFECT_READ) for skill in found)


def test_builtin_cards_describe_capability_without_routing() -> None:
    for skill in _builtin_skills():
        card = skill.spec.card.lower()
        hits = [word for word in _ROUTING_WORDS if word in card]
        assert not hits, f"{skill.spec.skill_id} card routes the model: {hits}"


# ── Spec and digests ─────────────────────────────────────────────────────────


def test_spec_rejects_unknown_effects_and_empty_cards() -> None:
    with pytest.raises(ValueError, match="effect"):
        SkillSpec(skill_id="x", version="1", card="c", args_schema={}, effect="teleport")
    with pytest.raises(ValueError, match="card"):
        SkillSpec(skill_id="x", version="1", card="  ", args_schema={})


def _plan(**overrides: object) -> SkillPlan:
    base = SkillPlan(
        skill_id="motion.sequence",
        args={"steps": [{"op": "move", "relative": {"dz": 100.0}}]},
        summary="lift",
        targets=(Pose(250, 0, 350, 180, 0, 0),),
        limits={"speed_pct": 20.0},
        basis={"start": {"z": 250.0}},
        workcell="cell",
        version="1.0.0",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_plan_digest_covers_content_but_not_bookkeeping() -> None:
    plan = _plan()
    assert plan_digest(plan) == plan_digest(replace(plan, created_at=0.0, digest="x"))
    for change in (
        {"args": {"steps": [{"op": "move", "relative": {"dz": 101.0}}]}},
        {"targets": (Pose(250, 0, 351, 180, 0, 0),)},
        {"limits": {"speed_pct": 21.0}},
        {"basis": {"start": {"z": 251.0}}},
        {"workcell": "other"},
        {"version": "1.0.1"},
        {"summary": "lift more"},
    ):
        assert plan_digest(replace(plan, **change)) != plan_digest(plan), change  # type: ignore[arg-type]


def test_permit_digest_binds_plan_version_workcell_and_limits() -> None:
    base = dict(
        workcell="cell", skill_id="motion.sequence", version="1", plan_digest="d", limits={"a": 1.0}
    )
    digest = run_permit_digest(**base)  # type: ignore[arg-type]
    for key, value in (
        ("workcell", "other"),
        ("version", "2"),
        ("plan_digest", "e"),
        ("limits", {"a": 2.0}),
    ):
        assert run_permit_digest(**{**base, key: value}) != digest  # type: ignore[arg-type]


# ── Arguments ────────────────────────────────────────────────────────────────

_SCHEMA = {
    "type": "object",
    "properties": {
        "dz": {"type": "number", "minimum": -10, "maximum": 10, "default": 0.0},
        "frame": {"type": "string", "enum": ["base", "tool"], "default": "base"},
        "n": {"type": "integer"},
        "flag": {"type": "boolean"},
        "xs": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["n"],
}


def test_validate_args_applies_defaults_and_coerces_numbers() -> None:
    assert validate_args(_SCHEMA, {"n": 3, "xs": [1, 2]}) == {
        "dz": 0.0,
        "frame": "base",
        "n": 3,
        "xs": [1.0, 2.0],
    }
    assert validate_args(_SCHEMA, {"n": 2.0})["n"] == 2


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"n": 1, "dz": 11},
        {"n": 1, "dz": True},
        {"n": 1, "dz": math.nan},
        {"n": 1.5},
        {"n": 1, "frame": "world"},
        {"n": 1, "flag": 1},
        {"n": 1, "xs": [1]},
        {"n": 1, "xs": "12"},
        {"n": 1, "extra": 0},
    ],
)
def test_validate_args_refuses_with_invalid_args(args: dict[str, object]) -> None:
    with pytest.raises(SkillError) as error:
        validate_args(_SCHEMA, args)
    assert error.value.failure_code == FAILURE_INVALID_ARGS


def _example(schema: dict[str, object]) -> object:
    """The smallest value *schema* accepts, built from its declared shape."""
    if "enum" in schema:
        return schema["enum"][0]  # type: ignore[index]
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties") or {}
        return {name: _example(properties[name]) for name in schema.get("required") or ()}  # type: ignore[index]
    if kind == "array":
        item = _example(schema.get("items") or {})  # type: ignore[arg-type]
        return [item] * int(schema.get("minItems") or 1)  # type: ignore[call-overload]
    return {"boolean": False, "string": "x", "integer": schema.get("minimum", 0)}.get(kind, schema.get("minimum", 0.0))  # type: ignore[arg-type]


def test_every_builtin_schema_validates_its_own_example_shape() -> None:
    for skill in _builtin_skills():
        validate_args(skill.spec.args_schema, _example(skill.spec.args_schema))  # type: ignore[arg-type]


# ── Geometry ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rpy", [(180, 0, 0), (170, 10, -45), (0, 0, 90), (-30, 60, 120)])
def test_rpy_round_trips_through_the_matrix(rpy: tuple[float, float, float]) -> None:
    back = rpy_to_matrix(*matrix_to_rpy(rpy_to_matrix(*rpy)))
    for row, expected in zip(back, rpy_to_matrix(*rpy)):
        assert row == pytest.approx(expected, abs=1e-9)


def test_transform_compose_and_inverse_are_consistent() -> None:
    flange = Transform.from_pose(Pose(250, 0, 370, 180, 0, 0))
    tool = Transform.from_pose(Pose(0, 0, 120, 0, 0, 0))
    tcp = flange.compose(tool).to_pose()
    assert (tcp.x, tcp.y, tcp.z) == pytest.approx((250, 0, 250))
    back = Transform.from_pose(tcp).compose(tool.inverse()).to_pose()
    assert back.distance_mm(Pose(250, 0, 370, 180, 0, 0)) == pytest.approx(0, abs=1e-9)
    assert tool_cone_deg(tcp) == pytest.approx(0, abs=1e-9)


def test_workspace_box_reports_the_violating_axis() -> None:
    box = WorkspaceBox.from_mapping({"x": [0, 10], "y": [0, 10], "z": [0, 10]})
    assert box.contains((5, 5, 5))
    assert "z=11.0" in (box.violation((5, 5, 11)) or "")
    with pytest.raises(ValueError):
        WorkspaceBox(0, 0, 0, 1, 0, 1)


# ── Cyclic rate limiting ─────────────────────────────────────────────────────


def test_only_full_turn_degree_channels_are_cyclic() -> None:
    context = piper_context()
    assert is_cyclic(context.channel("ee.rz"))
    assert not is_cyclic(context.channel("ee.x"))
    assert not is_cyclic(context.channel("joint.j1.position"))  # +-150 deg is not a turn
    assert rate_delta(context.channel("ee.rx"), -179.0, 179.0) == pytest.approx(2.0)
    assert rate_delta(context.channel("ee.x"), -179.0, 179.0) == pytest.approx(-358.0)


def test_rate_limit_takes_the_short_way_round_at_the_seam() -> None:
    registry = SkillNodeRegistry()
    gate = NodeWriteGate(registry)
    channel = piper_context().channel("ee.rx")
    registry.record_command(ARM_ID, "ee.rx", 179.0)
    # 2 deg across the +-180 seam at 45 deg/s, not 358 deg.
    assert gate.rate_wait_s(ARM_ID, channel, -179.0) < 0.05
