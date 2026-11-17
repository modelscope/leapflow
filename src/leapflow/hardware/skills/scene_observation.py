# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-local passive multi-view measurements and observation evidence."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

from leapflow.hardware.skills.observability import ObservationReport, ObservationSettings, ViewEvidence
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT, FAILURE_OBSERVATION_INSUFFICIENT, FAILURE_TARGET_NOT_FOUND, LocatedObject, SkillError,
)
from leapflow.hardware.skills.scene_memory import SceneCapture
from leapflow.hardware.skills.table_view import as_seen, mask_box, rebased, table_plane
from leapflow.hardware.vision.group import CaptureGroup

if TYPE_CHECKING:
    from leapflow.hardware.skills.perception import WorkcellScene


class SceneObservations:
    """Delegate with a back-reference; never captures mutable registry/frame state."""

    def __init__(self, scene: WorkcellScene) -> None:
        self.scene = scene

    @property
    def settings(self) -> ObservationSettings:
        return ObservationSettings.from_mapping(self.scene._spec.observation)

    def roles(self) -> tuple[str, ...]:
        scene = self.scene
        return tuple(role for role in sorted(scene._spec.roles) if role.split('.')[0] == 'camera'
                     and scene._frames.transform(role) is not None)

    async def captures(self, roles: Sequence[str]) -> tuple[dict[str, SceneCapture], dict[str, Any]]:
        scene = self.scene
        if not roles:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, "no calibrated camera can measure this object")
        async def capture(role: str, after_monotonic: float) -> SceneCapture:
            return await scene._observe(role, after_monotonic=after_monotonic)
        sources = {role: (lambda after, role=role: capture(role, after)) for role in roles}
        sync = scene._spec.vision
        group = CaptureGroup(sources, tolerance_ms=sync['sync_tolerance_ms'], attempts=int(sync['sync_attempts']))
        result = await group.capture()
        for error in result.errors.values():
            if not isinstance(error, SkillError):
                raise error
        summary = result.summary()
        summary['errors'] = {role: {'failure_code': e.failure_code, 'message': e.message}
                             for role, e in result.errors.items()}
        if not result.frames:
            raise SkillError(FAILURE_DEVICE_FAULT, 'no camera captured a usable frame', result={'sync': summary})
        return dict(result.frames), summary

    def analyze(self, obj: LocatedObject, capture: SceneCapture) -> ViewEvidence:
        """Sample the projected top surface at a fixed normalized grid density."""
        frame = capture.frame
        mount, table = capture.placement()
        obj = as_seen(obj, frame, mount, table, self.scene._settings, camera=capture.camera)
        to_camera = mount.inverse()
        settings = self.settings
        side = max(2, int(math.sqrt(settings.samples)))
        yaw = math.radians(obj.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        supported = nearer = missing = outside = 0
        depth = frame.depth_units()
        scale = frame.depth_scale_m * 1000
        bound = self.scene._frames.diagnosis(capture.camera).achieved
        # Unknown angular calibration accuracy cannot establish a metric bound.
        error = None if bound is None or bound.rotation_deg is None else bound.translation_mm
        camera_settings = self.scene._spec.observation.get('cameras', {}).get(capture.camera, {})
        depth_error = camera_settings.get('depth_error_mm', settings.depth_error_mm)
        max_range = 0.0
        for row in range(side):
            for col in range(side):
                dx = obj.length * (col / (side - 1) - .5)
                dy = obj.width * (row / (side - 1) - .5)
                point = (obj.center[0] + dx*c - dy*s, obj.center[1] + dx*s + dy*c,
                         obj.center[2] + obj.height / 2)
                local = to_camera.apply(point)
                if local[2] <= 0:
                    outside += 1
                    continue
                max_range = max(max_range, math.sqrt(sum(v*v for v in local)))
                u, v = frame.intrinsics.project(local)
                if not 0 <= u < frame.width or not 0 <= v < frame.height:
                    outside += 1
                    continue
                measured = depth[int(v)*frame.width + int(u)] * scale
                tolerance = depth_error + (error or 0)
                if bound is not None and bound.rotation_deg is not None:
                    tolerance += math.sqrt(sum(v*v for v in local)) * 2 * math.sin(math.radians(min(180, bound.rotation_deg)) / 2)
                if not camera_settings.get('near_mm', 0) <= measured <= camera_settings.get('far_mm', math.inf):
                    measured = 0
                if measured <= 0:
                    missing += 1
                elif measured < local[2] - tolerance:
                    nearer += 1
                elif abs(measured - local[2]) <= tolerance:
                    supported += 1
        count = side * side
        if error is not None:
            error += max_range * 2 * math.sin(math.radians(min(180, bound.rotation_deg)) / 2)
            error += max_range / min(frame.intrinsics.fx, frame.intrinsics.fy) + depth_error
        state = 'visible' if supported/count >= settings.min_support else (
            'occluded' if nearer else 'out_of_view' if outside == count else 'no_depth' if missing else 'moved_or_gone')
        extent = (state == 'visible' and not nearer and not outside and error is not None
                  and self._boundary(obj, capture, error, side))
        return ViewEvidence(capture.camera, capture.capture_id, state, supported/count,
                            outside > 0, nearer/count, missing/count, error, extent_supported=extent)

    def _boundary(self, obj: LocatedObject, capture: SceneCapture, error: float, side: int) -> bool:
        """Require measured background outside all four edges; projection alone is insufficient.

        This certifies only the declared box measurement model, not hidden cavities
        or arbitrary shape semantics. Missing/background occlusion stays unknown.
        """
        if obj.geometry == 'point' or min(obj.length, obj.width, obj.height) <= 2 * error:
            return False
        inverse = capture.placement()[0].inverse()
        frame = capture.frame
        depth = frame.depth_units()
        yaw = math.radians(obj.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        margin = 2 * error
        for i in range(side):
            t = i / (side - 1) - .5
            for dx, dy in ((-obj.length/2-margin, t*obj.width), (obj.length/2+margin, t*obj.width),
                           (t*obj.length, -obj.width/2-margin), (t*obj.length, obj.width/2+margin)):
                local = inverse.apply((obj.center[0]+dx*c-dy*s, obj.center[1]+dx*s+dy*c,
                                       obj.center[2]+obj.height/2))
                if local[2] <= 0:
                    return False
                u, v = frame.intrinsics.project(local)
                if not 0 <= u < frame.width or not 0 <= v < frame.height:
                    return False
                measured = depth[int(v)*frame.width+int(u)]*frame.depth_scale_m*1000
                if measured <= local[2] + error:
                    return False
        return True

    async def evidence(self, obj: LocatedObject, capture: SceneCapture) -> ViewEvidence:
        evidence = self.analyze(obj, capture)
        radii = self.scene._spec.observation.get('body_radii_mm')
        if (evidence.nearer <= 0 or not radii or self.scene._motion is None
                or self.scene._frames.parent(capture.camera) != 'base'):
            return evidence
        from leapflow.hardware.skills.body_visibility import capsules
        geometry = self.scene._motion.geometry_snapshot()
        state = await self.scene._motion.state()
        try:
            geometry.check_kinematics(state)
            frames = geometry.kinematics().link_transforms([state.joints[n] for n in geometry.names])
            body = capsules([f.translation for f in frames], radii)
        except (SkillError, ValueError):
            return evidence
        origin = capture.placement()[0].translation
        x, y, z = obj.center
        top = (x, y, z + obj.height/2)
        blocked = any(link.blocks(origin, top) for link in body)
        return replace(evidence, occluder='arm_predicted' if blocked else 'unknown')

    def report(self, obj: LocatedObject) -> ObservationReport:
        found = self.scene._memory.observation(obj, self.scene._basis, self.scene._settings.memory_ttl_s)
        return found if isinstance(found, ObservationReport) else ObservationReport(obj.object_id)

    def record(self, obj: LocatedObject, views: Sequence[ViewEvidence], *, conflict: bool = False,
               attempts: Sequence[ViewEvidence] = ()) -> None:
        report = ObservationReport(obj.object_id, tuple(views), conflict, tuple(attempts))
        self.scene._memory.remember_observation(obj, report, self.scene._basis)
        self.scene._note('observation', report.to_dict(self.settings))

    async def relocate(self, obj: LocatedObject, *, commit: bool = True) -> LocatedObject:
        """Remeasure without grounding; read-only confirmation never refreshes memory."""
        from leapflow.hardware.skills.perception import lift_detection, reproject
        from leapflow.hardware.skills.scene_fusion import presence
        scene = self.scene
        roles = tuple(dict.fromkeys((obj.camera, *self.roles())))
        attempts = []
        for role in roles:
            try:
                capture = await scene.capture(role)
                evidence = await self.evidence(obj, capture)
                mount, table = capture.placement()
                projected = as_seen(obj, capture.frame, mount, table, scene._settings, camera=role)
                state, _, _, depths = presence(projected, capture.frame, mount.inverse(), scene._settings)
                attempts.append(replace(evidence, state=state,
                    reason=f"presence={state}; expected_mm={depths.get('expected_mm')}; measured_mm={depths.get('measured_mm')}"))
                if state != 'visible':
                    continue
                if self.settings.enabled and (evidence.nearer > 0 or evidence.support <= 0):
                    continue
                detection = reproject(projected, capture)
                seen = lift_detection(detection, capture, scene._settings)
                if seen.height > obj.height + self.settings.disagreement_mm:
                    attempts[-1] = replace(attempts[-1], reason='remeasured top exceeds association tolerance')
                    continue
            except (LookupError, ValueError) as exc:
                attempts.append(ViewEvidence(role, '', 'unavailable', reason=str(exc)))
                continue
            except SkillError as exc:
                if exc.failure_code not in {FAILURE_DEVICE_FAULT, 'calibration_missing', 'calibration_stale', 'skill_unavailable'}:
                    raise
                attempts.append(ViewEvidence(role, '', 'unavailable', reason=exc.message))
                continue
            moved = replace(seen, object_id=obj.object_id, source=obj.source, geometry=obj.geometry)
            if self.settings.enabled and math.dist(moved.center, obj.center) > self.settings.disagreement_mm:
                continue
            if commit:
                scene._memory.update(moved, scene._basis)
                # Earlier views describe the old geometry; never certify the new fit
                # with evidence measured against a different extent.
                self.record(moved, (await self.evidence(moved, capture),), attempts=attempts)
            return moved
        report = ObservationReport(obj.object_id, tuple(attempts), attempts=tuple(attempts))
        if commit:
            self.record(obj, attempts)
        gone = any(v.state == 'moved_or_gone' for v in attempts) and not any(v.state == 'visible' for v in attempts)
        detail = 'moved or disappeared from its measured position' if gone else 'cannot be measured reliably (occlusion, missing depth, or measurement disagreement)'
        raise SkillError(FAILURE_TARGET_NOT_FOUND if gone else FAILURE_OBSERVATION_INSUFFICIENT,
                         f'{obj.object_id} {detail}; observe again and replan',
                         result={'observation': report.to_dict(self.settings),
                                 'next': ['perceive.survey', 'perceive.snapshot with_cameras=[all]']})

    async def measure(self, obj: LocatedObject, captures: Sequence[SceneCapture], *,
                      latest: Sequence[SceneCapture] | None = None) -> LocatedObject:
        """Bounded geometric reacquisition and cloud fusion, never label re-grounding.

        An expanded footprint may recover a previously clipped edge only within
        the declared association tolerance. Neighbours and nearer top surfaces
        veto expansion; unsupported outer edges remain lower bounds.
        """
        from leapflow.hardware.skills.perception import describe, lift_mask, reproject
        scene, settings = self.scene, self.settings
        attempts, cloud, used = [], [], []
        latest = tuple(captures) if latest is None else tuple(latest)
        anchor_ids = {c.capture_id for c in latest}
        newest = max((c.monotonic_at for c in latest), default=0.)
        captures = tuple(c for c in captures if 0 <= newest-c.monotonic_at <= scene._settings.memory_ttl_s)
        margin = settings.disagreement_mm
        search = replace(obj, length=obj.length+2*margin, width=obj.width+2*margin)
        neighbours = scene._memory.objects(scene._basis, scene._settings.memory_ttl_s)
        ambiguous = any(n.object_id != obj.object_id and math.dist(n.center, obj.center)
                        <= math.hypot(search.length, search.width)/2 + math.hypot(n.length, n.width)/2
                        for n in neighbours)
        if ambiguous:
            search = obj
        for capture in captures:
            evidence = await self.evidence(obj, capture)
            attempts.append(evidence)
            if evidence.support <= 0 or evidence.nearer > 0:
                continue
            try:
                mount, table = capture.placement()
                projected = as_seen(search, capture.frame, mount, table, scene._settings, camera=capture.camera)
                detection = reproject(projected, capture)
                if detection.mask is None:
                    continue
                viewed = table_plane(mask_box(detection.mask), capture.frame, mount, table, scene._settings) or table
                points = rebased(lift_mask(detection.mask, capture.frame, mount, viewed, scene._settings), viewed, table)
                # Never turn a previously unknown higher/nearer surface into the object.
                top = obj.center[2] + obj.height/2
                allowance = evidence.error_mm or settings.depth_error_mm
                if any(p[2] > top + allowance for p in points):
                    continue
                if len(points) < scene._settings.min_points:
                    continue
                fitted = describe(points, table, label=obj.label, score=obj.score,
                                  observed_at=capture.frame.observed_at,
                                  top_quantile=scene._settings.top_quantile)
                if math.dist(fitted.center, obj.center) > margin or abs(fitted.height-obj.height) > margin:
                    continue
            except (LookupError, ValueError):
                continue
            cloud.extend(points)
            used.append(capture)
        if not cloud or not any(c.capture_id in anchor_ids for c in used):
            self.record(obj, [v for v in attempts if v.capture_id in anchor_ids],
                        conflict=ambiguous, attempts=attempts)
            return obj
        table = used[-1].placement()[1]
        fitted = describe(cloud, table, label=obj.label, score=obj.score,
                          observed_at=max(c.frame.observed_at for c in used),
                          top_quantile=scene._settings.top_quantile)
        if (math.dist(fitted.center, obj.center) > margin or abs(fitted.height-obj.height) > margin
                or fitted.length > search.length or fitted.width > search.width):
            self.record(obj, [v for v in attempts if v.capture_id in anchor_ids],
                        conflict=True, attempts=attempts)
            return obj
        fresh = replace(fitted, object_id=obj.object_id, source=obj.source,
                        camera=obj.camera, geometry=obj.geometry, table_offset_mm=obj.table_offset_mm)
        scene._memory.update(fresh, scene._basis, invalidate_grasps=True)
        # Historic captures improve the fit, but never authorize an absent live target.
        views = [await self.evidence(fresh, c) for c in latest]
        self.record(fresh, views, conflict=ambiguous, attempts=attempts)
        return fresh

    async def locate(self, queries: Sequence[str], floor: float) -> tuple[LocatedObject, ...]:
        """Fuse only mutually unique geometric associations; ambiguous objects stay distinct."""
        from leapflow.hardware.skills.perception import describe, lift_detection, lift_mask, lift_box
        scene = self.scene
        captures, sync = await self.captures(self.roles())
        if not sync.get('within_tolerance', True):
            raise SkillError(FAILURE_DEVICE_FAULT, 'multi-view capture exceeds synchronization tolerance', result={'sync': sync})
        groups: list[tuple[LocatedObject, list[Any], list[ViewEvidence], set[str]]] = []
        conflicts = []
        conflicted: set[int] = set()
        for role, capture in captures.items():
            result, model = await scene._ground(capture.frame, queries)
            detected = []
            for detection in result.detections:
                if detection.score < floor:
                    continue
                try:
                    obj = lift_detection(detection, capture, scene._settings)
                    mount, table = capture.placement()
                    box = mask_box(detection.mask) if detection.mask is not None else detection.box
                    viewed = table_plane(box, capture.frame, mount, table, scene._settings) if box is not None else None
                    viewed = viewed or table
                    points = rebased(lift_mask(detection.mask, capture.frame, mount, viewed, scene._settings)
                              if detection.mask is not None else lift_box(detection.box, capture.frame, mount, viewed, scene._settings)
                              if detection.box is not None else [], viewed, table)
                    detected.append((obj, points, await self.evidence(obj, capture)))
                except LookupError:
                    continue
                except ValueError as exc:
                    raise SkillError('model_unavailable', f'locator {model!r} returned unusable geometry: {exc}') from exc
            matches = [[i for i, (old, _, views, cameras) in enumerate(groups)
                        if role not in cameras and _associated(old, obj, self.settings.disagreement_mm)
                        and views[0].error_mm is not None]
                       for obj, _, _ in detected]
            for (obj, points, evidence), candidates in zip(detected, matches):
                if evidence.error_mm is not None and len(candidates) == 1 and sum(candidates[0] in m for m in matches) == 1:
                    i = candidates[0]
                    old, cloud, views, cameras = groups[i]
                    if abs(old.height - obj.height) > self.settings.disagreement_mm:
                        conflicts.append({'cameras': sorted(cameras | {role}), 'reason': 'height disagreement'})
                        conflicted.update((i, len(groups)))
                        groups.append((obj, points, [evidence], {role}))
                        continue
                    merged = cloud + points
                    if merged and capture.table is not None:
                        fitted = describe(merged, capture.table, label=old.label, score=max(old.score, obj.score),
                                          observed_at=max(old.observed_at, obj.observed_at), top_quantile=scene._settings.top_quantile)
                        old = replace(fitted, camera=old.camera, geometry=old.geometry, table_offset_mm=old.table_offset_mm)
                    groups[i] = (old, merged, views + [evidence], cameras | {role})
                else:
                    nearby = [i for i, (old, _, _, cameras) in enumerate(groups)
                              if role not in cameras and math.dist(old.center, obj.center) <= self.settings.disagreement_mm]
                    if candidates or nearby:
                        reason = 'ambiguous geometric association' if candidates else 'extent disagreement'
                        conflicts.append({'camera': role, 'reason': reason})
                        conflicted.update((*candidates, *nearby, len(groups)))
                    groups.append((obj, points, [evidence], {role}))
        named = scene._memory.remember([g[0] for g in groups], scene._basis)
        for i, (obj, (_, _, views, _)) in enumerate(zip(named, groups)):
            fresh = [await self.evidence(obj, captures[v.camera]) for v in views]
            self.record(obj, fresh, conflict=i in conflicted)
        scene._note('multi_view', {'sync': sync, 'conflicts': conflicts, 'cameras': list(captures)})
        return named


def _associated(a: LocatedObject, b: LocatedObject, tolerance: float) -> bool:
    # Associations use measured base-frame geometry, never textual labels.
    return (math.dist(a.center, b.center) <= tolerance
            and abs(a.length-b.length) <= tolerance and abs(a.width-b.width) <= tolerance)
