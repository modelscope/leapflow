# Copyright (c) Alibaba, Inc. and its affiliates.
"""Conservative observation contracts: measured support is not statistical certainty."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from leapflow.hardware.skills.protocol import GraspCandidate, LocatedObject


@dataclass(frozen=True)
class ObservationSettings:
    """Workcell observation policy, independent of embodiment or camera vendor."""

    enabled: bool = False
    samples: int = 225
    min_support: float = 0.85
    depth_error_mm: float = 3.0
    clearance_mm: float = 2.0
    disagreement_mm: float = 10.0
    max_views: int = 3
    max_candidates: int = 8
    speed_pct: float = 10.0
    settle_s: float = 0.5
    max_travel_mm: float = 300.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ObservationSettings:
        if not isinstance(raw, Mapping):
            raise ValueError("observation must be a mapping")
        known = {f.name for f in fields(cls)}
        extras = {"observe", "cameras", "body_radii_mm", "planner", "planner_settings", "clearance"}
        unknown = set(raw) - known - extras
        if unknown:
            raise ValueError(f"unknown observation fields: {sorted(unknown)}")
        values = {name: raw[name] for name in known if name in raw}
        result = cls(**values)
        if not isinstance(result.enabled, bool):
            raise ValueError("observation.enabled must be boolean")
        for name in known - {"enabled"}:
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"observation.{name} must be finite and positive")
        if not 0 < result.min_support <= 1 or result.speed_pct > 100:
            raise ValueError("observation support or speed is out of range")
        for name, ceiling in (("samples", 4096), ("max_views", 16), ("max_candidates", 64)):
            value = getattr(result, name)
            if type(value) is not int or value > ceiling:
                raise ValueError(f"observation.{name} must be an integer <= {ceiling}")
        _validate_declarations(raw)
        return result


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_declarations(raw: Mapping[str, Any]) -> None:
    observe = raw.get('observe', {})
    if not isinstance(observe, Mapping) or any(not isinstance(k, str) or not k or
            type(v) not in (int, float) or not math.isfinite(v) for k, v in observe.items()):
        raise ValueError('observation.observe must contain finite named joint targets')
    radii = raw.get('body_radii_mm', [])
    if not isinstance(radii, (list, tuple)) or any(not _positive(r) for r in radii):
        raise ValueError('observation.body_radii_mm must contain positive link radii')
    cameras = raw.get('cameras', {})
    if not isinstance(cameras, Mapping):
        raise ValueError('observation.cameras must be a mapping')
    for role, values in cameras.items():
        if (not isinstance(role, str) or role.split('.')[0] != 'camera' or not isinstance(values, Mapping)
                or set(values) - {'near_mm', 'far_mm', 'depth_error_mm'}
                or not all(_positive(v) for v in values.values())
                or ('near_mm' in values) != ('far_mm' in values)
                or ('near_mm' in values and values['near_mm'] >= values['far_mm'])):
            raise ValueError('observation camera requires a valid declared depth interval')
    if not isinstance(raw.get('planner', 'ring'), str) or not raw.get('planner', 'ring'):
        raise ValueError('observation.planner must be a nonempty plugin name')
    if not isinstance(raw.get('planner_settings', {}), Mapping):
        raise ValueError('observation.planner_settings must be a mapping')
    clearance = raw.get('clearance', {})
    if not isinstance(clearance, Mapping) or set(clearance) - {'free_box', 'tool_radius_mm', 'joint_tracking_deg'}:
        raise ValueError('observation.clearance has unknown fields')
    if clearance:
        from leapflow.hardware.skills.geometry import WorkspaceBox
        if set(clearance) != {'free_box', 'tool_radius_mm', 'joint_tracking_deg'}:
            raise ValueError('survey clearance requires free_box, tool_radius_mm, and joint_tracking_deg')
        try:
            WorkspaceBox.from_mapping(clearance['free_box'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('survey free_box must contain finite x/y/z intervals') from exc
        if not all(_positive(clearance[k]) for k in ('tool_radius_mm', 'joint_tracking_deg')):
            raise ValueError('survey clearance radii and tracking bounds must be positive')


@dataclass(frozen=True)
class ViewEvidence:
    camera: str
    capture_id: str
    state: str
    support: float = 0.0
    truncated: bool = False
    nearer: float = 0.0
    missing: float = 0.0
    error_mm: float | None = None
    occluder: str = "unknown"
    reason: str = ""
    extent_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationReport:
    object_id: str
    views: tuple[ViewEvidence, ...] = ()
    conflict: bool = False
    attempts: tuple[ViewEvidence, ...] = ()

    def complete_views(self, settings: ObservationSettings) -> tuple[ViewEvidence, ...]:
        return tuple(view for view in self.views if view.state == "visible" and not view.truncated
                     and view.support >= settings.min_support and view.error_mm is not None
                                          and view.extent_supported and view.nearer == 0)

    def to_dict(self, settings: ObservationSettings) -> dict[str, Any]:
        complete = self.complete_views(settings)
        error = min((v.error_mm for v in complete), default=None)
        verdict = "sufficient" if complete and not self.conflict else "partial" if self.views else "insufficient"
        return {"object": self.object_id, "verdict": verdict, "views": [v.to_dict() for v in self.views],
                "conflict": self.conflict, "attempts": [v.to_dict() for v in self.attempts],
                                "position_bound_mm": error,
                "extent_is_lower_bound": not bool(complete),
                "brief": f"{self.object_id}: {verdict}; " + "; ".join(
                    f"{v.camera} {v.state} support {v.support:.0%}" for v in self.views)}

    def for_grasp(self, obj: LocatedObject, grasp: GraspCandidate,
                  settings: ObservationSettings) -> dict[str, Any]:
        """Compare conservative spatial bounds with this candidate's actual slack.

        No 1/sqrt(N) reduction: camera calibration errors are correlated. The
        bound covers translation, pixel resolution and declared depth error;
        truncated extents cannot be certified by opening the jaws wider.
        """
        report = self.to_dict(settings)
        error = report["position_bound_mm"]
        slack = (grasp.open_mm - grasp.width_mm) / 2 - settings.clearance_mm
        vertical = min(grasp.approach_mm, grasp.retreat_mm or grasp.approach_mm) - settings.clearance_mm
        # Opposing edges can each shift by the correlated metric bound. Treat
        # angular error as unknown when that displacement spans the short edge.
        short_edge = min(obj.length, obj.width)
        yaw_bound = (math.degrees(math.asin(2 * error / short_edge))
                     if error is not None and short_edge > 2 * error else None)
        corner = (math.hypot(obj.length, obj.width) * math.sin(math.radians(yaw_bound) / 2)
                  if yaw_bound is not None else None)
        closing_bound = error + corner if error is not None and corner is not None else None
        reasons = []
        if self.conflict:
            reasons.append("contradictory camera measurements")
        if report["extent_is_lower_bound"]:
            reasons.append("extent is truncated, unsupported, or has unknown metric accuracy")
        if obj.geometry == "point":
            reasons.append("a point does not establish closing width or height")
        if closing_bound is None or slack <= closing_bound:
            reasons.append("jaw slack does not exceed position and yaw displacement bounds")
        if error is None or vertical <= error:
            reasons.append("vertical clearance does not exceed the observation bound")
        return {**report, "sufficient": not reasons, "jaw_slack_mm": round(slack, 3),
                "vertical_clearance_mm": round(vertical, 3), "yaw_bound_deg": yaw_bound,
                                "closing_bound_mm": closing_bound, "reasons": reasons}
