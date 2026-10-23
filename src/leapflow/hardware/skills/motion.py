# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed discrete motion: the only way a node skill moves anything.

``GovernedMotion`` implements ``MotionFacade`` on top of ``NodeWriteGate``. A
skill states *where* (a TCP pose in the base frame, joint targets, a gripper
width); this module owns *how*, in one place:

- **Frames.** A TCP target becomes a flange target through the calibrated
  ``T_flange_tcp``: ``T_base_flange = T_base_tcp · inv(T_flange_tcp)``.
- **Region.** Every TCP target must lie inside the calibrated workspace box --
  or, for ``calib.*`` runs, inside the human-declared bootstrap box when there
  is one (calibration never trusts the geometry it is about to re-measure) --
  with the tool inside the declared cone around straight down, and the flange
  target inside the arm's own declared envelopes.
- **Speed.** The commanded speed rides in the same batch as the target when the
  arm declares ``arm.speed_pct``, capped by the run's effective limit and
  quantized *down* to the channel's declared step, so it never exceeds the cap.
  A cap below the arm's declared minimum is a refusal, never a silent raise.
- **Path.** A pose target's path shape (``linear`` / ``point``) rides in the
  same batch when the arm declares ``arm.pose_path``; a move that names none
  uses the first declared value, so the shape never depends on the last run.
- **Rate.** A command the gate's slew check would refuse is waited out (within
  the run deadline) rather than turned into a failure.
- **Arrival.** A move ends when the arm reports arrival *and* the observed pose
  is within the declared tolerances -- position by distance, orientation by
  geodesic angle, because a vendor may report an equivalent Euler triple. A
  gripper has settled at its target, or once its width stayed within the
  declared tolerance for the declared settling time (closed on an object).
  No threshold is a constant here: admission refuses an arm or gripper that
  does not declare them.
- **Verdicts.** A refusal before anything was sent is ``write_refused`` (NONE);
  anything after the first send is at least PARTIAL. ``sent_any`` lets the
  runtime merge a skill's own verdict with what physically happened.
- **Dispatch.** A batch is registered as in flight *before* it enters the
  transport and runs shielded from the skill's cancellation, so a stop that
  lands mid-write still sees "something may have moved" and halts. A dispatch
  that ends without a verdict (cancelled, raised) makes the run ``uncertain``
  (side effect UNKNOWN); ``settle`` lets the runtime wait for stragglers before
  it releases the devices.
- **Record.** Every batch that reached a transport is reported to the run's
  ``EpisodeTap``, so the recorded actions are exactly what was commanded.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from leapflow.hardware.context import as_numeric
from leapflow.hardware.node_gate import NodeWriteGate, merged_batch_state, rate_delta
from leapflow.hardware.skills.episodes import NULL_TAP, EpisodeTap
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import (
    Pose,
    Transform,
    WorkspaceBox,
    tool_cone_deg,
)
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.ports import PATH_DEFAULT
from leapflow.hardware.skills.protocol import (
    FAILURE_ARM_NOT_READY,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_DEVICE_FAULT,
    FAILURE_INTERNAL_DEFECT,
    FAILURE_INVALID_ARGS,
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TIMEOUT,
    FAILURE_UNREACHABLE,
    FAILURE_WRITE_REFUSED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    STATUS_FAILED,
    STATUS_HALTED,
    ArmState,
    GripperResult,
    MotionResult,
    SkillError,
)
from leapflow.hardware.skills.tuning import DEFAULT_LIMITS
from leapflow.hardware.skills.workcell import (
    ARM_STATUS_CHANNEL,
    ARRIVED_CHANNEL,
    CTRL_MODE_CHANNEL,
    ENABLED_CHANNEL,
    GRIPPER_EFFORT,
    GRIPPER_POSITION,
    POSE_CHANNELS,
    POSE_PATH_CHANNEL,
    ROLE_ARM,
    ROLE_GRIPPER,
    SPEED_CHANNEL,
    WorkcellSpec,
    joint_channels,
)
from leapflow.hardware.transport import TransportError

POLL_S = 0.05

# Vendor ``arm_status`` codes that mean "this target was refused, nothing moved".
# They are a planning verdict the firmware delivers late -- after the command has
# been accepted -- so they must be read as a refusal, not as a slow arrival: the
# timeout path halts, and a halt engages the emergency stop, which releases the
# drives and drops the arm under gravity.
_ARM_STATUS_REFUSALS = {
    0x02: "no inverse-kinematics solution for the target",
    0x03: "the target lies on a singularity",
    0x04: "the target exceeds a joint limit",
}
# A refusing code can also be left over from an earlier target, so it only ends
# the move once the arm is demonstrably still where the command found it for
# long enough that a real move would have left the tolerance band.
_REFUSAL_DWELL_S = 1.5

Sleep = Callable[[float], Awaitable[None]]


def _within_gripper_tolerance(actual: float, target: float, tolerance: float) -> bool:
    """Honor the declared tolerance, accounting only for binary arithmetic roundoff."""
    if not all(math.isfinite(v) for v in (actual, target, tolerance)) or tolerance < 0:
        return False
    if tolerance == 0:
        return actual == target
    # SI conversion/subtraction can put a boundary sample a few ULPs outside.
    # This is a floating-point error budget, not sensor uncertainty or extra mm.
    roundoff = math.ulp(actual) + math.ulp(target) + math.ulp(tolerance)
    return abs(actual - target) <= tolerance + roundoff


class GovernedMotion:
    """``MotionFacade`` for one run in one workcell."""

    def __init__(
        self,
        *,
        registry: Any,
        gate: NodeWriteGate,
        spec: WorkcellSpec,
        frames: FrameGraph,
        limits: Mapping[str, float],
        cancel: asyncio.Event,
        deadline: float,
        allow_bootstrap: bool = False,
        allow_writes: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        poll_s: float = POLL_S,
        tap: EpisodeTap = NULL_TAP,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._spec = spec
        self._frames = frames
        self._limits = dict(limits)
        self._cancel = cancel
        self._deadline = float(deadline)
        self._allow_bootstrap = allow_bootstrap
        self._allow_writes = allow_writes
        self._clock = clock
        self._sleep = sleep
        self._poll_s = poll_s
        self._tap = tap
        self._landed = 0  # batches that provably reached a transport
        self._uncertain = False  # a dispatch ended without a verdict
        self._inflight: set[asyncio.Future[Any]] = set()
        self._arm_id = spec.role(ROLE_ARM)
        self._arm = registry.context(self._arm_id)
        self._joints = joint_channels(self._arm)
        self._dh_chain: DhChain | None = None
        self._chain_loaded = False
        tcp = frames.transform("tcp")
        self._flange_tcp = tcp if tcp is not None else Transform()

    @property
    def sent_any(self) -> bool:
        """Whether anything may have reached a transport: landed, unresolved, or in flight."""
        return self._landed > 0 or self.uncertain

    @property
    def uncertain(self) -> bool:
        """Whether some dispatch is still in flight or ended without a verdict."""
        return self._uncertain or bool(self._inflight)

    @property
    def landed(self) -> int:
        """How many batches provably reached a transport (grows on late landings)."""
        return self._landed

    async def settle(self, timeout_s: float) -> bool:
        """Wait up to *timeout_s* for in-flight dispatches; True when none is left."""
        if self._inflight:
            await asyncio.wait(set(self._inflight), timeout=max(0.0, timeout_s))
        return not self._inflight

    def when_settled(self, callback: Callable[[], None]) -> None:
        """Call *callback* once no dispatch is in flight (at once when none is)."""
        pending = set(self._inflight)
        if not pending:
            callback()
            return

        def _done(future: asyncio.Future[Any]) -> None:
            pending.discard(future)
            if not pending:
                callback()

        for future in pending:
            future.add_done_callback(_done)

    @property
    def deadline(self) -> float:
        """The run clock time at which every motion call raises ``timeout``."""
        return self._deadline

    @property
    def tap(self) -> EpisodeTap:
        """The run's episode recorder (a no-op outside recorded runs)."""
        return self._tap

    def now(self) -> float:
        """The run clock the deadline is measured on."""
        return self._clock()

    async def command(self, device_id: str, commands: Sequence[tuple[str, Any]]) -> None:
        """Send one governed batch and return without awaiting arrival (streaming).

        Same gate, slew spacing, checkpoint and verdicts as every other motion;
        the caller owns the targets' safety (``SafetyShield``) and the cadence.
        """
        await self._send(device_id, commands)

    async def pause(self, seconds: float) -> None:
        """Sleep up to *seconds*, raising at once when the run is stopped or out of time."""
        await self._pause(max(0.0, seconds))

    # ── State ──

    async def state(self) -> ArmState:
        readable = [ARRIVED_CHANNEL]
        if self._arm.channel(ENABLED_CHANNEL) is not None:
            readable.append(ENABLED_CHANNEL)
        if self._arm.channel(CTRL_MODE_CHANNEL) is not None:
            readable.append(CTRL_MODE_CHANNEL)
        if self._arm.channel(ARM_STATUS_CHANNEL) is not None:
            readable.append(ARM_STATUS_CHANNEL)
        gripper_id = self._spec.role(ROLE_GRIPPER)
        if gripper_id == self._arm_id:
            readable.append(GRIPPER_POSITION)
        values = await self._read(self._arm_id, (*POSE_CHANNELS, *self._joints, *readable))
        pose = [as_numeric(values.get(cid)) for cid in POSE_CHANNELS]
        if any(value is None for value in pose):
            raise SkillError(
                FAILURE_DEVICE_FAULT,
                f"arm {self._arm_id!r} did not report a complete end-effector pose",
                side_effect_state=self._failure_state(),
            )
        flange = Pose(*pose)  # type: ignore[arg-type]
        gripper_mm = as_numeric(values.get(GRIPPER_POSITION))
        if gripper_id and gripper_id != self._arm_id:
            gripper_mm = await self._gripper_width()
        enabled = values.get(ENABLED_CHANNEL)
        arrived = values.get(ARRIVED_CHANNEL)
        ctrl_mode = values.get(CTRL_MODE_CHANNEL)
        reported = values.get(ARM_STATUS_CHANNEL)
        arm_status = (
            as_numeric(reported.get("arm_status")) if isinstance(reported, Mapping) else None
        )
        return ArmState(
            flange=flange,
            tcp=self.flange_to_tcp(flange),
            joints={
                cid: value
                for cid in self._joints
                if (value := as_numeric(values.get(cid))) is not None
            },
            gripper_mm=gripper_mm,
            enabled=None if enabled is None else bool(enabled),
            arrived=None if arrived is None else bool(arrived),
            observed_at=time.time(),
            ctrl_mode=None if ctrl_mode in (None, "") else str(ctrl_mode),
            arm_status=None if arm_status is None else int(arm_status),
        )

    def flange_to_tcp(self, flange: Pose) -> Pose:
        return Transform.from_pose(flange).compose(self._flange_tcp).to_pose()

    def tcp_to_flange(self, tcp: Pose) -> Pose:
        return Transform.from_pose(tcp).compose(self._flange_tcp.inverse()).to_pose()

    def reached(self, pose: Pose, state: ArmState) -> bool:
        return self._within((), self.tcp_to_flange(pose), state)

    # ── Region ──

    def check_target(self, pose: Pose) -> None:
        box = self._region()
        violation = box.violation(pose.position)
        if violation is not None:
            raise SkillError(FAILURE_OUT_OF_WORKSPACE, f"target leaves the workspace: {violation}")
        cone = tool_cone_deg(pose)
        if cone > self._spec.tool_cone_deg:
            raise SkillError(
                FAILURE_UNREACHABLE,
                f"tool would tilt {cone:.1f} deg from straight down; "
                f"the workcell allows {self._spec.tool_cone_deg:.1f}",
            )
        flange = self.tcp_to_flange(pose)
        for channel_id, value in zip(POSE_CHANNELS, flange.as_tuple()):
            channel = self._arm.channel(channel_id)
            if not channel.envelope.contains(value):
                raise SkillError(
                    FAILURE_UNREACHABLE,
                    f"flange {channel_id}={value:.1f} is outside the arm's declared envelope",
                )

    def _region(self) -> WorkspaceBox:
        box = self._frames.workspace
        if self._allow_bootstrap and self._spec.bootstrap_box is not None:
            box = self._spec.bootstrap_box
        if box is None:
            raise SkillError(
                FAILURE_CALIBRATION_MISSING,
                "the workspace is not calibrated (run calib.workspace)",
            )
        return box

    def _chain(self) -> DhChain:
        if not self._chain_loaded:
            raw = getattr(getattr(self._arm, "kinematics", None), "dh", None) or {}
            try:
                chain = DhChain.from_mapping(raw) if raw else None
            except ValueError as exc:
                raise SkillError(
                    FAILURE_SKILL_UNAVAILABLE, f"the arm's kinematics.dh is invalid: {exc}"
                ) from exc
            if chain is not None and chain.dof != len(self._joints):
                raise SkillError(
                    FAILURE_SKILL_UNAVAILABLE,
                    f"kinematics.dh has {chain.dof} links for {len(self._joints)} joints",
                )
            self._dh_chain = chain
            self._chain_loaded = True
        if self._dh_chain is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "move_joints needs the arm's kinematics.dh to check where the tool goes",
            )
        return self._dh_chain

    def check_kinematics(self, state: ArmState) -> None:
        """Refuse a joint move unless declared FK agrees with the reported flange."""
        missing = [name for name in self._joints if name not in state.joints]
        if missing:
            raise SkillError(
                FAILURE_DEVICE_FAULT,
                f"arm {self._arm_id!r} did not report {', '.join(missing)}",
            )
        chain = self._chain()
        values = [state.joints[name] for name in self._joints]
        mm, deg = chain.agreement(values, state.flange)
        mm_limit = float(self._limits["fk_tolerance_mm"])
        deg_limit = float(self._limits["fk_tolerance_deg"])
        if mm > mm_limit or deg > deg_limit:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"the arm's kinematics.dh puts the flange {mm:.1f} mm / {deg:.1f} deg "
                f"from where the firmware reports it (tolerance {mm_limit:g} mm / "
                f"{deg_limit:g} deg); fix the table before moving joints",
            )

    def check_joints(
        self, targets: Mapping[str, float], current: Mapping[str, float] | None
    ) -> Pose:
        """Check every sampled FK pose of a joint-space move and return its endpoint TCP."""
        if not targets:
            raise SkillError(FAILURE_INVALID_ARGS, "joint targets are empty")
        unknown = sorted(set(targets) - set(self._joints))
        if unknown:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"{', '.join(unknown)} are not joint channels of {self._arm_id!r}",
            )
        numeric = {name: float(value) for name, value in targets.items()}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise SkillError(FAILURE_INVALID_ARGS, "joint targets must be finite")
        for name, value in numeric.items():
            if not self._arm.channel(name).envelope.contains(value):
                raise SkillError(
                    FAILURE_UNREACHABLE,
                    f"joint {name}={value:g} is outside the arm's declared envelope",
                )
        start = {} if current is None else dict(current)
        missing = [name for name in self._joints if name not in numeric and name not in start]
        if missing:
            code = FAILURE_INVALID_ARGS if current is None else FAILURE_DEVICE_FAULT
            detail = "must be targeted" if current is None else "was not reported by the arm"
            raise SkillError(code, f"joint {', '.join(missing)} {detail}")
        endpoint = {name: numeric[name] if name in numeric else start[name] for name in self._joints}
        chain = self._chain()
        if current is None:
            samples = (tuple(endpoint.values()),)
        else:
            begin = tuple(float(start[name]) for name in self._joints)
            end = tuple(endpoint.values())
            largest = max(abs(after - before) for before, after in zip(begin, end))
            count = max(1, math.ceil(largest / float(self._limits["joint_path_step_deg"])))
            samples = tuple(
                tuple(before + (after - before) * index / count for before, after in zip(begin, end))
                for index in range(1, count + 1)
            )
        target: Pose | None = None
        for sample in samples:
            target = self.flange_to_tcp(chain.forward(sample))
            self.check_target(target)
        assert target is not None
        return target

    def check_cartesian_path(self, state: ArmState, targets: Sequence[Pose]) -> tuple[Mapping[str, Any], ...]:
        from leapflow.hardware.skills.grasp_path import check_path
        return check_path(self, state, targets)

    # ── Motion ──

    @property
    def paths(self) -> tuple[str, ...]:
        channel = self._arm.channel(POSE_PATH_CHANNEL)
        return () if channel is None else tuple(map(str, channel.envelope.allowed_values))

    async def move_tcp(
        self, pose: Pose, *, speed_pct: float | None = None, path: str = PATH_DEFAULT
    ) -> MotionResult:
        self.check_target(pose)
        settings = self._path_command(path)
        flange = self.tcp_to_flange(pose)
        commands = tuple(zip(POSE_CHANNELS, flange.as_tuple()))
        return await self._move(commands, speed_pct, flange_target=flange, settings=settings)

    def _path_command(self, path: str) -> tuple[tuple[str, Any], ...]:
        paths = self.paths
        if path and path not in paths:
            declared = ", ".join(paths) if paths else "none"
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"arm {self._arm_id!r} cannot move along a {path!r} path (declared: {declared})",
            )
        return ((POSE_PATH_CHANNEL, path or paths[0]),) if paths else ()

    async def move_joints(
        self, targets: Mapping[str, float], *, speed_pct: float | None = None
    ) -> MotionResult:
        self._checkpoint()
        before = await self.state()
        self.check_kinematics(before)
        self.check_joints(targets, before.joints)
        commands = tuple((cid, float(value)) for cid, value in targets.items())
        return await self._move(commands, speed_pct, flange_target=None, before=before)

    async def _move(
        self,
        commands: Sequence[tuple[str, float]],
        speed_pct: float | None,
        *,
        flange_target: Pose | None,
        settings: Sequence[tuple[str, Any]] = (),
        before: ArmState | None = None,
    ) -> MotionResult:
        started = self._clock()
        self._checkpoint()
        before = before or await self.state()
        if before.enabled is False:
            raise SkillError(
                FAILURE_ARM_NOT_READY,
                f"arm {self._arm_id!r} is not enabled; energize it (arm.enable=1) first",
            )
        batch = [*settings, *commands]
        if self._arm.channel(SPEED_CHANNEL) is not None:
            batch.insert(0, (SPEED_CHANNEL, self.speed_for(speed_pct)))
        await self._send(self._arm_id, batch)
        state = await self._await_arrival(commands, flange_target, before=before)
        return MotionResult(state=state, elapsed_s=self._clock() - started)

    def speed_for(self, requested: float | None) -> float:
        """The speed value a move commands: requested or the run default, capped, quantized.

        Quantization rounds *down* to the channel's declared step so the command
        never exceeds the cap; a value that ends below the channel's declared
        minimum is refused rather than raised to it.
        """
        cap = float(self._limits["speed_pct_max"])
        value = float(self._limits.get("speed_pct", cap) if requested is None else requested)
        value = min(cap, value)
        channel = self._arm.channel(SPEED_CHANNEL)
        envelope = channel.envelope if channel is not None else None
        step = envelope.quantization if envelope is not None else None
        if step:
            value = math.floor(value / step + 1e-9) * step
        low = envelope.min_value if envelope is not None else None
        if value <= 0.0 or (low is not None and value < low):
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"speed {value:g} is below what {self._arm_id!r} accepts "
                f"(minimum {low if low is not None else 0:g}); raise the speed or the cap",
                side_effect_state=self._failure_state(),
            )
        return value

    async def _send(self, device_id: str, commands: Sequence[tuple[str, Any]]) -> None:
        if not self._allow_writes:
            raise SkillError(
                FAILURE_INTERNAL_DEFECT, "a skill tried to move during plan(), which is read-only"
            )
        await self._wait_rate(device_id, commands)
        self._checkpoint()
        batch = tuple(commands)
        # Registered before the transport sees it and shielded from the skill's
        # cancellation: a stop racing this write must still halt, and the write's
        # real verdict is reconciled when it lands (``_resolved``).
        dispatch = asyncio.ensure_future(self._gate.write_batch(device_id, batch))
        self._inflight.add(dispatch)
        dispatch.add_done_callback(lambda done: self._resolved(done, device_id, batch))
        try:
            ok, outcomes = await asyncio.shield(dispatch)
        except asyncio.CancelledError:
            raise
        except TransportError as exc:  # "could not attempt": nothing left the node
            raise SkillError(
                FAILURE_WRITE_REFUSED, str(exc), side_effect_state=self._failure_state()
            ) from exc
        except Exception as exc:  # noqa: BLE001 - a dispatch without a verdict is UNKNOWN
            raise SkillError(
                FAILURE_DEVICE_FAULT,
                f"dispatch to {device_id!r} ended without a verdict: {type(exc).__name__}: {exc}",
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                status=STATUS_HALTED,
            ) from exc
        if ok:
            return
        state = merged_batch_state(ok, outcomes)
        error = next((o.error for o in outcomes if not o.ok and o.error), "write failed")
        if state == SIDE_EFFECT_NONE:
            raise SkillError(FAILURE_WRITE_REFUSED, error, side_effect_state=self._failure_state())
        raise SkillError(FAILURE_DEVICE_FAULT, error, side_effect_state=state)

    def _resolved(
        self, dispatch: asyncio.Future[Any], device_id: str, commands: Sequence[tuple[str, Any]]
    ) -> None:
        """Account for one finished dispatch, whether or not anyone still awaits it."""
        self._inflight.discard(dispatch)
        if dispatch.cancelled():
            self._uncertain = True
            return
        error = dispatch.exception()
        if error is not None:
            if not isinstance(error, TransportError):
                self._uncertain = True
            return
        ok, outcomes = dispatch.result()
        state = merged_batch_state(ok, outcomes)
        if state == SIDE_EFFECT_UNKNOWN:
            self._uncertain = True
        if state != SIDE_EFFECT_NONE:
            self._landed += 1
            self._tap.command(device_id, commands)

    async def _wait_rate(self, device_id: str, commands: Sequence[tuple[str, Any]]) -> None:
        context = self._registry.context(device_id)
        wait_s = 0.0
        for channel_id, value in commands:
            channel = context.channel(channel_id) if context is not None else None
            if channel is not None:
                wait_s = max(wait_s, self._gate.rate_wait_s(device_id, channel, value))
        if wait_s <= 0.0:
            return
        if self._clock() + wait_s > self._deadline:
            raise SkillError(
                FAILURE_TIMEOUT,
                f"the move needs {wait_s:.1f}s of slew-rate spacing, beyond the run deadline",
                side_effect_state=self._failure_state(),
            )
        await self._pause(wait_s)

    async def _await_arrival(
        self,
        commands: Sequence[tuple[str, float]],
        flange_target: Pose | None,
        before: ArmState | None = None,
    ) -> ArmState:
        expires = self._expiry()
        state = await self.state()
        refused_since: float | None = None
        while not (state.arrived is not False and self._within(commands, flange_target, state)):
            refusal = _ARM_STATUS_REFUSALS.get(state.arm_status or 0)
            if refusal is None or not self._still_at(before, state):
                refused_since = None
            else:
                refused_since = refused_since if refused_since is not None else self._clock()
                if self._clock() - refused_since >= _REFUSAL_DWELL_S:
                    raise self._target_refused(refusal, state)
            if self._clock() >= expires:
                raise SkillError(
                    FAILURE_TIMEOUT,
                    "the arm did not reach the target in time",
                    side_effect_state=SIDE_EFFECT_PARTIAL,
                    status=STATUS_HALTED,
                    result={"state": state.to_dict()},
                )
            await self._pause(self._poll_s)
            state = await self.state()
        return state

    def _still_at(self, before: ArmState | None, state: ArmState) -> bool:
        """Whether *state* is still where *before* found the arm, within tolerance."""
        if before is None:
            return False
        return state.flange.distance_mm(before.flange) <= self._tolerance(
            POSE_CHANNELS[:3]
        ) and state.flange.rotation_to_deg(before.flange) <= self._tolerance(POSE_CHANNELS[3:])

    def _target_refused(self, refusal: str, state: ArmState) -> SkillError:
        """The arm's own kinematics refused the target without executing anything.

        A refusal is a planning verdict, not a late arrival: the arm is where the
        command found it and nothing was commanded. The status is what matters --
        a timeout halts the run, and a halt engages the emergency stop, which
        releases the drives and drops the arm. The side effect stays conservative:
        the frame did reach the transport, so reconciliation lifts the skill's
        NONE to PARTIAL on its own.
        """
        return SkillError(
            FAILURE_UNREACHABLE,
            f"the arm refused the target before moving ({refusal}); "
            "choose a target inside its reachable set",
            side_effect_state=SIDE_EFFECT_NONE,
            status=STATUS_FAILED,
            result={"state": state.to_dict(), "refusal": refusal},
        )

    def _within(
        self, commands: Sequence[tuple[str, float]], flange_target: Pose | None, state: ArmState
    ) -> bool:
        if flange_target is not None:
            position_ok = state.flange.distance_mm(flange_target) <= self._tolerance(
                POSE_CHANNELS[:3]
            )
            rotation_ok = state.flange.rotation_to_deg(flange_target) <= self._tolerance(
                POSE_CHANNELS[3:]
            )
            return position_ok and rotation_ok
        for channel_id, target in commands:
            observed = state.joints.get(channel_id)
            if observed is None:
                return False
            delta = rate_delta(self._arm.channel(channel_id), observed, target)
            if abs(delta) > self._tolerance((channel_id,)):
                return False
        return True

    def _tolerance(self, channel_ids: Sequence[str]) -> float:
        """The widest declared tolerance among *channel_ids* (admission requires one)."""
        return max(self._arm.channel(cid).envelope.tolerance for cid in channel_ids)

    # ── Gripper ──

    async def gripper(self, width_mm: float, *, effort: float | None = None) -> GripperResult:
        gripper_id = self._spec.role(ROLE_GRIPPER)
        if not gripper_id:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "this workcell binds no gripper")
        max_width = self._spec.gripper_spec.get("max_width_mm")
        if width_mm < 0.0 or (max_width is not None and width_mm > max_width):
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"gripper width {width_mm:.1f} mm is outside [0, {max_width}]",
            )
        started = self._clock()
        self._checkpoint()
        commands: list[tuple[str, Any]] = [(GRIPPER_POSITION, float(width_mm))]
        context = self._registry.context(gripper_id)
        if effort is not None and context.channel(GRIPPER_EFFORT) is not None:
            commands.append((GRIPPER_EFFORT, float(effort)))
        await self._send(gripper_id, commands)
        width = await self._await_gripper(width_mm, context)
        tolerance = context.channel(GRIPPER_POSITION).envelope.tolerance
        return GripperResult(
            width_mm=width,
            target_mm=float(width_mm),
            blocked=not _within_gripper_tolerance(width, width_mm, tolerance),
            elapsed_s=self._clock() - started,
        )

    async def _await_gripper(self, target: float, context: Any) -> float:
        envelope = context.channel(GRIPPER_POSITION).envelope
        tolerance = envelope.tolerance
        settle_s = envelope.effective_settling_s
        expires = self._expiry()
        # Settled: at the target, or unmoved (within tolerance of an anchor) for
        # the declared settling time -- fingers closed on an object stop short.
        # A gripper that has not started moving yet also looks unmoved, which is
        # why the window is the declared settling time and not one poll.
        anchor: float | None = None
        anchored_at = 0.0
        while True:
            width = await self._gripper_width()
            now = self._clock()
            if _within_gripper_tolerance(width, target, tolerance):
                return width
            if anchor is None or not _within_gripper_tolerance(width, anchor, tolerance):
                anchor, anchored_at = width, now
            elif now - anchored_at >= settle_s:
                return width
            if now >= expires:
                raise SkillError(
                    FAILURE_TIMEOUT,
                    "the gripper did not settle in time",
                    side_effect_state=SIDE_EFFECT_PARTIAL,
                    status=STATUS_HALTED,
                    result={"width_mm": width},
                )
            await self._pause(self._poll_s)

    async def _gripper_width(self) -> float:
        gripper_id = self._spec.role(ROLE_GRIPPER)
        values = await self._read(gripper_id, (GRIPPER_POSITION,))
        width = as_numeric(values.get(GRIPPER_POSITION))
        if width is None:
            raise SkillError(
                FAILURE_DEVICE_FAULT,
                f"gripper {gripper_id!r} did not report its width",
                side_effect_state=self._failure_state(),
            )
        return width

    # ── Plumbing ──

    async def _read(self, device_id: str, channel_ids: Sequence[str]) -> dict[str, Any]:
        batch = await self._registry.read_batch(device_id, tuple(channel_ids))
        return {
            reading.channel_id: reading.value
            for reading in batch.readings
            if getattr(reading, "is_trustworthy", True)
        }

    def _checkpoint(self) -> None:
        """Raise when the run was stopped or ran out of time."""
        if self._cancel.is_set():
            raise SkillError(
                FAILURE_STOPPED,
                "the run was stopped",
                side_effect_state=self._failure_state(),
                status=STATUS_HALTED,
            )
        if self._clock() >= self._deadline:
            raise SkillError(
                FAILURE_TIMEOUT,
                "the run exceeded its maximum duration",
                side_effect_state=self._failure_state(),
                status=STATUS_HALTED,
            )

    async def _pause(self, seconds: float) -> None:
        """Sleep, waking early for a stop; re-checks the run after waking."""
        end = self._clock() + seconds
        while (remaining := end - self._clock()) > 0:
            self._checkpoint()
            await self._sleep(min(remaining, self._poll_s))
        self._checkpoint()

    def _expiry(self) -> float:
        timeout = float(self._limits.get("move_timeout_s", DEFAULT_LIMITS["move_timeout_s"]))
        return min(self._clock() + timeout, self._deadline)

    def _failure_state(self) -> str:
        if self.uncertain:
            return SIDE_EFFECT_UNKNOWN
        return SIDE_EFFECT_PARTIAL if self._landed else SIDE_EFFECT_NONE


__all__ = ["GovernedMotion"]
