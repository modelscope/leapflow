#!/usr/bin/env python3
"""Piper X emergency stop, recovery and control-mode switching on pyAgxArm, for manual operator use.

Usage (on the board, conda env ``leap``); run without arguments for the full usage::

    python piper_estop.py stop      [--can can0] [--timeout 1.5]
    python piper_estop.py recover   [--can can0] [--speed 20] [--yes] [--monitor 2]
    python piper_estop.py mode {standby,can,teach,ethernet,wifi,offline} [--yes] ...
    python piper_estop.py status    [--can can0]

``stop``     Sends the damped electronic emergency stop (never a torque drop)
             and confirms the firmware latched it. It takes no lock and ignores
             any other bus owner: a stop must never wait behind anything.
``recover``  Same as ``mode can``: clears a latched stop or driver fault
             (``reset``), enables all six drivers, enters CAN control and
             watches the arm hold it.
``mode``     Switches the controller mode and confirms it from feedback.
             Leaving teach mode issues the vendor ``reset`` first. ``remote``
             (0x05) and ``linkage-teach`` (0x06) are only reported: they follow
             the remote/leader-follower configuration, not a mode command.
``status``   Read-only; sends no frame.

``recover`` and ``mode`` are refused while the LHP node or another bus owner
runs, because two writers race; the persistent node recovers through its own
governed ``arm.enable`` path instead.

Exit codes: 0 confirmed, 1 refused (nothing sent), 2 sent but unconfirmed
(inspect the arm; use the physical stop button), 130 interrupted.
"""
from __future__ import annotations

import argparse
import fcntl
import math
import os
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent
LIFECYCLE_LOCK = ROOT / ".piper-node.lock"  # shared with piper_ops node lifecycle

ARM_NORMAL, ARM_ESTOP = 0x00, 0x01
CTRL_STANDBY, CTRL_CAN, CTRL_TEACH = 0x00, 0x01, 0x02
# Modes a command can select; the names double as the ``mode`` argument.
MODES = {"standby": CTRL_STANDBY, "can": CTRL_CAN, "teach": CTRL_TEACH,
         "ethernet": 0x03, "wifi": 0x04, "offline": 0x07}
CTRL_NAMES = {code: name for name, code in MODES.items()} | {0x05: "remote", 0x06: "linkage-teach"}
MODE_WARNINGS = {
    "can": "re-energizes all six joints (reset first if a stop is latched); the arm may drop "
           "briefly before the drives hold.",
    "standby": "leaves CAN control; the drives keep their enable state but stop taking motion commands.",
    "teach": "enters drag-teach: the joints go compliant and can be moved by hand.",
    "ethernet": "hands control to the Ethernet interface, so CAN motion commands stop being obeyed. "
                "If the arm no longer answers `mode can`, power-cycle it.",
    "wifi": "hands control to the Wi-Fi interface, so CAN motion commands stop being obeyed. "
            "If the arm no longer answers `mode can`, power-cycle it.",
    "offline": "selects offline-trajectory mode, which can replay a stored trajectory.",
}
LEAVE_TEACH_WARNING = ("Leaving teach mode needs a reset: the arm may go limp and fall until the "
                       "drives re-engage.")
TEACH_START = 0x01  # MotionCtrl drag-teach byte: start teach recording (enters teach mode)
ARM_STATUS_NAMES = {
    0x00: "normal", 0x01: "emergency stop", 0x02: "no IK solution", 0x03: "singularity",
    0x04: "joint limit exceeded", 0x05: "joint communication error", 0x06: "brake not released",
    0x07: "collision", 0x08: "overspeed while dragging", 0x09: "joint status error",
    0x0A: "controller fault", 0x0E: "controller over-temperature",
    0x0F: "release-resistor over-temperature",
}
FAULT_FLAGS = ("driver_overcurrent", "driver_overheating", "driver_error_status",
               "collision_status", "stall_status", "motor_overheating", "voltage_too_low")
# Arm status (0x2A1) and the six driver low-speed frames (0x261-0x266) carry
# every field this tool decides on; all of them must be freshly received.
STATUS_GROUPS = (0x2A1, *range(0x261, 0x267))
FRESH_S = 0.25
POLL_S = 0.02
ENABLE_RETRY_S = 0.1
MODE_RETRY_S = 0.15  # longer than pyAgxArm's 0.1 s identical-mode-frame throttle
RESTART_PAUSE_S = 0.25


class Refused(Exception):
    """Nothing was sent to the arm."""


class Unconfirmed(Exception):
    """Frames were sent but the intended state was not observed."""


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _msg(getter: Callable[..., Any], *args: Any) -> Any:
    """The ``.msg`` body of a pyAgxArm feedback getter, or None when unavailable."""
    try:
        message = getter(*args)
    except Exception:  # noqa: BLE001 - an unavailable getter reads as no feedback
        return None
    return getattr(message, "msg", None) if message is not None else None


def _joints(code: int | None, shift: int) -> list[int]:
    mask = ((code or 0) >> shift) & 0xFF
    return [j for j in range(1, 7) if mask >> (j - 1) & 1]


def wait(predicate: Callable[[], bool], until: float) -> bool:
    """Poll ``predicate`` until it holds or the monotonic deadline passes."""
    while True:
        if predicate():
            return True
        if time.monotonic() >= until:
            return False
        time.sleep(POLL_S)


class Arm:
    """One pyAgxArm connection with per-frame receive stamps."""

    def __init__(self, can: str) -> None:
        # Imported lazily so argument errors never wait on the SDK import.
        from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

        self.profile = PiperFW.V189
        self._rx: dict[int, float] = {}
        self._lock = threading.Lock()
        config = create_agx_arm_config(robot=ArmModel.PIPER_X, comm="can",
                                       firmeware_version=self.profile, channel=can)
        self.robot = AgxArmFactory.create_arm(config)
        self.robot.get_context().register_parser_packet_fun(self._on_frame)
        self.robot.connect()

    def _on_frame(self, message: Any) -> None:
        # Runs on the SDK read thread after its parsers; must never raise.
        try:
            if getattr(message, "is_remote_frame", False) or int(getattr(message, "dlc", 0)) != 8:
                return
            with self._lock:
                self._rx[int(message.arbitration_id)] = time.monotonic()
        except Exception:  # noqa: BLE001 - an exception here would stop the bus reader
            return

    def fresh(self) -> bool:
        now = time.monotonic()
        with self._lock:
            return all(now - self._rx.get(g, -math.inf) <= FRESH_S for g in STATUS_GROUPS)

    def status(self) -> dict[str, Any]:
        msg = _msg(self.robot.get_arm_status)
        ctrl, code = _int(getattr(msg, "ctrl_mode", None)), _int(getattr(msg, "arm_status", None))
        err = _int(getattr(msg, "err_code", None))
        enabled: list[bool | None] = []
        faults: dict[int, list[str]] = {}
        for joint in range(1, 7):
            foc = getattr(_msg(self.robot.get_driver_states, joint), "foc_status", None)
            flag = getattr(foc, "driver_enable_status", None)
            enabled.append(flag if isinstance(flag, bool) else None)
            bad = [name for name in FAULT_FLAGS if getattr(foc, name, False) is True]
            if bad:
                faults[joint] = bad
        radians = _msg(self.robot.get_joint_angles)
        try:
            joints = [round(math.degrees(float(v)), 3) for v in radians] if radians is not None else None
        except (TypeError, ValueError):
            joints = None
        return {
            "fresh": self.fresh(),
            "ctrl_mode": CTRL_NAMES.get(ctrl, ctrl),
            "arm_status": ARM_STATUS_NAMES.get(code, code),
            "arm_status_code": code,
            "ctrl_mode_code": ctrl,
            "comm_error_joints": _joints(err, 0),
            "limit_error_joints": _joints(err, 8),
            "enabled": enabled,
            "driver_faults": faults,
            "joints_deg": joints,
        }

    def check_firmware(self, timeout_s: float = 2.0) -> None:
        """Refuse an arm whose firmware does not resolve to the V189 profile used here."""
        from pyAgxArm import ArmModel, resolve_firmware_profile

        deadline = time.monotonic() + timeout_s
        version = ""
        while not version and time.monotonic() < deadline:
            try:
                info = self.robot.get_firmware(timeout=0.25, min_interval=0.0)
            except Exception:  # noqa: BLE001 - retried until the deadline
                info = None
            version = str((info or {}).get("software_version") or "").strip()
        if not version:
            raise Refused("arm firmware version did not answer; check CAN reception")
        resolved = resolve_firmware_profile(ArmModel.PIPER_X, version)
        if str(resolved) != str(self.profile):
            raise Refused(f"firmware {version!r} resolves to {resolved!r}, expected {self.profile!r}")

    def send(self, frame: Any) -> None:
        # pyAgxArm has no public setter for these modes; _send_msg is its own packer and transport.
        self.robot._send_msg(frame)

    def close(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001 - closing must not mask the real outcome
            pass


def bus_owners() -> list[str]:
    """Same-host processes that write to the arm: the LHP node or the legacy daemon."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            names = [Path(os.fsdecode(a)).name for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if any(names[i:i + 2] == ["hw", "serve"] for i in range(len(names))):
            found.append(f"pid {entry.name} (leap hw serve)")
        elif any("cl_armd" in name for name in names):
            found.append(f"pid {entry.name} (legacy cl_armd)")
    return found


@contextmanager
def lifecycle_lock() -> Iterator[None]:
    """Exclude concurrent node start/restart; the lock inode is never removed."""
    with LIFECYCLE_LOCK.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused("a node lifecycle operation holds .piper-node.lock") from exc
        yield


def connect(can: str, *, sending: bool) -> Arm:
    try:
        return Arm(can)
    except ModuleNotFoundError as exc:
        raise Refused(f"{exc}; run with ~/miniconda3/envs/leap/bin/python") from exc
    except Exception as exc:  # noqa: BLE001 - any bring-up failure means no frame left
        hint = "; USE THE PHYSICAL EMERGENCY STOP" if sending else ""
        raise (Unconfirmed if sending else Refused)(f"cannot open {can}: {exc}{hint}") from exc


def _flags(enabled: list[bool | None]) -> str:
    marks = {True: "on", False: "off", None: "?"}
    count = sum(flag is True for flag in enabled)
    return f"{count}/6  " + "  ".join(f"J{j}:{marks[f]}" for j, f in enumerate(enabled, 1))


def _angles(joints: list[float] | None) -> str:
    if joints is None:
        return "unavailable"
    return "  ".join(f"J{j}:{v:8.3f}" for j, v in enumerate(joints, 1))


def _faults(state: dict[str, Any]) -> str:
    parts = [f"J{j}: {', '.join(flags)}" for j, flags in state["driver_faults"].items()]
    if state["comm_error_joints"]:
        parts.append(f"comm error joints {state['comm_error_joints']}")
    if state["limit_error_joints"]:
        parts.append(f"limit error joints {state['limit_error_joints']}")
    return "; ".join(parts) or "none"


def show(title: str, state: dict[str, Any]) -> None:
    """Print one state snapshot as an aligned block."""
    rows = (
        ("feedback", "fresh" if state["fresh"] else "STALE"),
        ("ctrl_mode", state["ctrl_mode"]),
        ("arm_status", state["arm_status"]),
        ("enabled", _flags(state["enabled"])),
        ("joints_deg", _angles(state["joints_deg"])),
        ("faults", _faults(state)),
    )
    print(f"[{title}]")
    for key, value in rows:
        print(f"  {key:<11}: {value}")
    sys.stdout.flush()


def show_line(title: str, state: dict[str, Any]) -> None:
    """Print one state snapshot on a single line, for periodic monitoring."""
    enabled = sum(flag is True for flag in state["enabled"])
    joints = state["joints_deg"]
    angles = " ".join(f"{v:.1f}" for v in joints) if joints is not None else "unavailable"
    print(f"[{title}] {'fresh' if state['fresh'] else 'STALE'}  ctrl={state['ctrl_mode']}  "
          f"status={state['arm_status']}  enabled={enabled}/6  faults={_faults(state)}  "
          f"joints=[{angles}]", flush=True)


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_stop(args: argparse.Namespace) -> int:
    arm = connect(args.can, sending=True)
    try:
        # Send first, verify afterwards: a stop never waits for feedback.
        deadline = time.monotonic() + args.timeout
        while True:
            arm.robot.electronic_emergency_stop()
            window = min(deadline, time.monotonic() + 0.1)
            if wait(lambda: arm.fresh() and arm.status()["arm_status_code"] == ARM_ESTOP, window):
                break
            if time.monotonic() >= deadline:
                show("state", arm.status())
                raise Unconfirmed("emergency stop not confirmed by the arm; USE THE PHYSICAL EMERGENCY STOP")
        show("stopped", arm.status())
        print("Damped emergency stop latched. The arm may sink slowly; support it if needed.")
        if bus_owners():
            print("Note: the LHP node is running; resume through its governed arm.enable "
                  "(piper_ops arm initialize), not this script.")
        return 0
    finally:
        arm.close()


def _confirm(skip: bool, before: dict[str, Any], target: str) -> None:
    if skip:
        return
    if not sys.stdin.isatty():
        raise Refused("non-interactive terminal; pass --yes after checking the workspace")
    print(f"Switching {before['ctrl_mode']} -> {target}: {MODE_WARNINGS[target]}")
    if before["ctrl_mode_code"] == CTRL_TEACH:
        print(LEAVE_TEACH_WARNING)
    print("Keep the workspace clear and support the arm.")
    if input("Type 'yes' to continue: ").strip().lower() != "yes":
        raise Refused("cancelled by operator")


def _mode_frame(code: int, speed: int) -> Any:
    from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgModeCtrl

    if code == CTRL_STANDBY:  # the proven standby handshake carries no motion selection
        return ArmMsgModeCtrl(ctrl_mode=code, move_mode=0x00, move_spd_rate_ctrl=0)
    return ArmMsgModeCtrl(ctrl_mode=code, move_mode=0x01, move_spd_rate_ctrl=speed)


def _teach_frame() -> Any:
    from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgMotionCtrl

    return ArmMsgMotionCtrl(grag_teach_ctrl=TEACH_START)


def _teach_hint(arm: Arm) -> str:
    if arm.status()["ctrl_mode_code"] == CTRL_TEACH:
        return "; still in teach mode, press the teach button on the arm base"
    return ""


def _repeat(send: Callable[[], None], done: Callable[[], bool], deadline: float,
            retry_s: float, failure: Callable[[], str]) -> None:
    """Resend until ``done`` holds; Unconfirmed with ``failure()`` once ``deadline`` passes."""
    while True:
        send()
        if wait(done, min(deadline, time.monotonic() + retry_s)):
            return
        if time.monotonic() >= deadline:
            raise Unconfirmed(failure())


def _reached(state: dict[str, Any], code: int) -> bool:
    """Whether ``state`` already is mode ``code``; CAN also needs a normal, fully enabled arm."""
    if state["ctrl_mode_code"] != code:
        return False
    return code != CTRL_CAN or (state["arm_status_code"] == ARM_NORMAL and not state["driver_faults"]
                                and all(flag is True for flag in state["enabled"]))


def _unstable(arm: Arm, settle_s: float, code: int) -> str:
    """Why the arm did not hold mode ``code`` for ``settle_s``; "" if it did.

    CAN control additionally requires a normal status and all six drivers enabled.
    """
    end = time.monotonic() + settle_s
    while True:
        state = arm.status()
        if not state["fresh"]:
            return "status feedback stopped arriving"
        if code == CTRL_CAN and state["arm_status_code"] != ARM_NORMAL:
            joints = state["comm_error_joints"]
            return f"arm reports {state['arm_status']}" + (f" (joints {joints})" if joints else "")
        if state["ctrl_mode_code"] != code:
            return f"it left {CTRL_NAMES[code]} ({state['ctrl_mode']})"
        if code == CTRL_CAN and not all(flag is True for flag in state["enabled"]):
            return f"a motor dropped its enable {state['enabled']}"
        if time.monotonic() >= end:
            return ""
        time.sleep(POLL_S)


def _clear_latch(arm: Arm, deadline: float) -> None:
    """``reset`` a latched stop or driver fault and wait for the latch to clear."""
    state = arm.status()
    if state["arm_status_code"] == ARM_NORMAL and not state["driver_faults"]:
        return
    print(f"reset: clearing {state['arm_status']} {state['driver_faults'] or ''}".rstrip(), flush=True)
    arm.robot.reset()
    if not wait(lambda: arm.fresh() and arm.status()["arm_status_code"] != ARM_ESTOP, deadline):
        raise Unconfirmed("stop still latched after reset; release the physical stop button")


def _energize(arm: Arm, speed: int, deadline: float, settle_s: float) -> None:
    """reset (if latched/faulted) -> enable all six -> CAN control -> observe it holds."""
    while True:
        _clear_latch(arm, deadline)
        print("enable: all six drivers", flush=True)
        _repeat(arm.robot.enable,
                lambda: arm.fresh() and all(f is True for f in arm.status()["enabled"]),
                deadline, ENABLE_RETRY_S,
                lambda: f"not every motor enabled {arm.status()['enabled']}; "
                        "check the stop button, power and motor cables")
        print(f"mode: CAN control at {speed}% speed", flush=True)
        _repeat(lambda: arm.robot.set_speed_percent(speed),
                lambda: arm.status()["ctrl_mode_code"] == CTRL_CAN,
                deadline, MODE_RETRY_S,
                lambda: f"arm did not enter CAN control (reports {arm.status()['ctrl_mode']})"
                        + _teach_hint(arm))
        reason = _unstable(arm, settle_s, CTRL_CAN)
        if not reason:
            return
        if time.monotonic() + settle_s >= deadline:
            raise Unconfirmed(f"arm would not stay energized: {reason}")
        print(f"retry: enable did not hold ({reason})", flush=True)
        time.sleep(RESTART_PAUSE_S)


def _enter(arm: Arm, target: str, speed: int, deadline: float, settle_s: float) -> None:
    """Send the frame that selects a non-CAN mode until feedback reports it, then observe it holds."""
    code = MODES[target]
    if code == CTRL_TEACH:
        _clear_latch(arm, deadline)  # drag-teach starts from a released stop
        frame = _teach_frame()
    else:
        frame = _mode_frame(code, speed)
    print(f"mode: {target}", flush=True)
    _repeat(lambda: arm.send(frame),
            lambda: arm.fresh() and arm.status()["ctrl_mode_code"] == code,
            deadline, MODE_RETRY_S,
            lambda: f"arm did not enter {target} (reports {arm.status()['ctrl_mode']})" + _teach_hint(arm))
    reason = _unstable(arm, settle_s, code)
    if reason:
        raise Unconfirmed(f"arm would not stay in {target}: {reason}")


def _switch(args: argparse.Namespace, target: str) -> int:
    code = MODES[target]
    owners = bus_owners()
    if owners:
        raise Refused(f"bus owner running: {', '.join(owners)}; stop it first or recover "
                      "through piper_ops arm initialize")
    with lifecycle_lock():
        arm = connect(args.can, sending=False)
        try:
            if not wait(arm.fresh, time.monotonic() + args.warmup):
                raise Refused("no fresh arm status/driver feedback; check arm power and CAN")
            arm.check_firmware()
            before = arm.status()
            show("before", before)
            if _reached(before, code):
                print(f"Already in {target}; nothing sent.")
                return 0
            _confirm(args.yes, before, target)
            deadline = time.monotonic() + args.timeout
            try:
                if before["ctrl_mode_code"] == CTRL_TEACH and code != CTRL_TEACH:
                    # Vendor rule: teach mode is left with one reset before any other mode.
                    print("reset: leaving teach mode", flush=True)
                    arm.robot.reset()
                if code == CTRL_CAN:
                    _energize(arm, args.speed, deadline, args.settle)
                else:
                    _enter(arm, target, args.speed, deadline, args.settle)
            except BaseException:
                if code == CTRL_CAN:
                    # A partly energized arm -- including after Ctrl-C -- ends in a damped stop.
                    try:
                        arm.robot.electronic_emergency_stop()
                        print("Recovery aborted: damped emergency stop sent.", file=sys.stderr)
                    except Exception as exc:  # noqa: BLE001 - report, then surface the original failure
                        print(f"Recovery aborted and stop send failed ({exc}); "
                              "USE THE PHYSICAL EMERGENCY STOP", file=sys.stderr)
                raise
            show("after", arm.status())
            end = time.monotonic() + args.monitor
            while time.monotonic() < end:
                time.sleep(0.5)
                show_line("monitor", arm.status())
            if code == CTRL_CAN:
                print("Done: all six enabled in CAN control. CAN control may lapse to standby "
                      "after this script disconnects; drive the arm through the LHP node.")
            else:
                print(f"Done: arm in {target} mode. Return with `mode can` or `mode standby`.")
            return 0
        finally:
            arm.close()


def cmd_recover(args: argparse.Namespace) -> int:
    return _switch(args, "can")


def cmd_mode(args: argparse.Namespace) -> int:
    return _switch(args, args.target)


def cmd_status(args: argparse.Namespace) -> int:
    arm = connect(args.can, sending=False)
    try:
        if not wait(arm.fresh, time.monotonic() + args.warmup):
            show("stale", arm.status())
            raise Refused("no fresh arm status/driver feedback; check arm power and CAN")
        show("status", arm.status())
        owners = bus_owners()
        print(f"  {'bus_owners':<11}: {', '.join(owners) if owners else 'none'}")
        return 0
    finally:
        arm.close()


def _speed(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError("speed must be 1-100")
    return value


OVERVIEW = """\
Piper X emergency stop, recovery and control-mode switching (pyAgxArm, PIPER_X, firmware V189).
Run on the board with the leap env: ~/miniconda3/envs/leap/bin/python piper_estop.py ...

commands:
  stop      Damped electronic emergency stop, confirmed from feedback. Takes no lock
            and ignores other bus owners: a stop never waits behind anything.
  recover   Same as `mode can`: reset a latched stop/fault, enable all six drivers,
            enter CAN control and watch the arm hold it.
  mode      Switch the controller mode: standby, can, teach, ethernet, wifi, offline.
  status    Read-only snapshot; sends no frame.

recover and mode are refused while the LHP node (leap hw serve) or cl_armd runs,
or while a node lifecycle operation holds .piper-node.lock. Both print the state,
explain the risk and ask you to type 'yes' (skip with --yes)."""

EXAMPLES = """\
examples:
  python piper_estop.py status                  read-only snapshot, sends nothing
  python piper_estop.py stop                    damped emergency stop
  python piper_estop.py recover                 clear stop/fault, enable, CAN control
  python piper_estop.py mode teach              enter drag-teach (joints movable by hand)
  python piper_estop.py mode can --speed 30     leave teach/standby for CAN control at 30%
  python piper_estop.py mode standby --yes      switch without the confirmation prompt
  python piper_estop.py status --can can1       use another CAN interface
  python piper_estop.py mode -h                 help for one command

exit codes:
  0    confirmed, or already in the requested mode
  1    refused: nothing was sent
  2    sent but not confirmed, or an SDK/bus error: inspect the arm, use the physical stop
  130  interrupted (Ctrl-C)"""

COMMAND_HELP = {
    "stop": """\
Send the damped electronic emergency stop (never a torque drop) and resend it until
the arm reports `emergency stop`, within --timeout. The arm may sink slowly after
the latch; resume with `recover`, or `piper_ops arm initialize` if the LHP node runs.""",
    "recover": """\
Same as `mode can`. Sequence: leave teach mode with one reset (if in teach) ->
reset a latched stop or driver fault -> enable all six drivers -> enter CAN control
at --speed -> require it to hold for --settle -> monitor for --monitor.
Any failure or Ctrl-C after the first frame ends in a damped emergency stop.
CAN control may lapse to standby after the script disconnects.""",
    "mode": """\
Switch the controller mode and confirm it from feedback, then require the mode to
hold for --settle and monitor for --monitor. If the arm already is in the target
mode, nothing is sent. Leaving teach mode always issues one vendor reset first.""",
    "status": """\
Read-only: wait up to --warmup for fresh status/driver feedback, then print control
mode, arm status, enable flags, joint angles, driver faults and bus owners.""",
}


def parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    root = argparse.ArgumentParser(prog="piper_estop.py", description=OVERVIEW, epilog=EXAMPLES,
                                   formatter_class=raw)
    sub = root.add_subparsers(dest="command", required=True, metavar="{stop,recover,mode,status}")
    bus = argparse.ArgumentParser(add_help=False)
    bus.add_argument("--can", default="can0", help="SocketCAN interface (default can0)")

    def command(name: str, parents: list[argparse.ArgumentParser], **kwargs: Any) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=parents, description=COMMAND_HELP[name],
                              formatter_class=raw, **kwargs)

    stop = command("stop", [bus])
    stop.add_argument("--timeout", type=float, default=1.5, help="confirmation budget, s (default 1.5)")
    switching = argparse.ArgumentParser(add_help=False)
    switching.add_argument("--speed", type=_speed, default=20,
                           help="speed percent sent with the mode, 1-100 (default 20)")
    switching.add_argument("--timeout", type=float, default=8.0, help="whole-sequence budget, s (default 8)")
    switching.add_argument("--settle", type=float, default=1.0,
                           help="the new mode must hold this long, s (default 1)")
    switching.add_argument("--warmup", type=float, default=3.0, help="wait for fresh feedback, s (default 3)")
    switching.add_argument("--monitor", type=float, default=2.0,
                           help="read-only watch after the switch, s (default 2)")
    switching.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    command("recover", [bus, switching])
    modes = "\n".join(textwrap.fill(MODE_WARNINGS[name], width=88, initial_indent=f"  {name:<10}0x{code:02X}  ",
                                      subsequent_indent=" " * 18) for name, code in MODES.items())
    mode = command("mode", [bus, switching],
                   epilog=f"modes:\n{modes}\n\nremote (0x05) and linkage-teach (0x06) are only reported "
                          "by status:\nthey follow the remote/leader-follower configuration, not a mode command.")
    mode.add_argument("target", choices=tuple(MODES), help="controller mode to switch to")
    status = command("status", [bus])
    status.add_argument("--warmup", type=float, default=3.0, help="wait for fresh feedback, s (default 3)")
    root.commands = sub.choices  # type: ignore[attr-defined]
    return root


def full_help(root: argparse.ArgumentParser) -> str:
    """Overview plus every command's own help, for a bare or ``-h`` invocation."""
    sections = [root.format_help()]
    for name, command in root.commands.items():  # type: ignore[attr-defined]
        sections.append(f"{'=' * 30} {name} {'=' * (40 - len(name))}\n{command.format_help()}")
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    root = parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv in (["-h"], ["--help"]):
        print(full_help(root))
        return 0
    args = root.parse_args(argv)
    command = {"stop": cmd_stop, "recover": cmd_recover, "mode": cmd_mode, "status": cmd_status}[args.command]
    try:
        return command(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except Unconfirmed as exc:
        print(f"UNCONFIRMED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - an SDK/bus error leaves the arm state unknown
        print(f"ERROR: {type(exc).__name__}: {exc}; arm state unknown, "
              "USE THE PHYSICAL EMERGENCY STOP if it is moving", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
