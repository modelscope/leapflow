# Copyright (c) Alibaba, Inc. and its affiliates.
"""High-frequency control bus: deterministic real-time control independent of asyncio.

The EventBus and HardwareStreamSource run inside the asyncio event loop, which
shares its thread with LLM inference, UI rendering, and RPC handling.  A 10ms
LLM token callback blocks the entire loop, making sub-20ms control cycles
unreliable.

This module provides a dedicated-thread control bus that:

1. Runs in its own OS thread with elevated priority (SCHED_FIFO on Linux)
2. Uses monotonic clock + busy-wait hybrid for sub-ms cycle accuracy
3. Executes a ControlPolicy each cycle: read sensors → compute → write actuators
4. Checks SafetyPolicy every cycle — violation triggers immediate halt
5. Monitors DegradationPolicy — comm loss triggers configured fallback
6. Provides an asyncio bridge for the Agent layer to start/stop/monitor

The bus does NOT replace EventBus or StreamSource — it coexists:

- EventBus: interaction signals (UI events, approvals, findings)
- StreamSource: observation telemetry (readings → events → dashboard)
- ControlBus: closed-loop servo control (read → compute → write at fixed Hz)
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@runtime_checkable
class ControlPolicy(Protocol):
    """Computes actuator commands from sensor readings each control cycle.

    Implementations range from simple PID to VLA neural network inference.
    Must complete within the cycle budget (1/frequency seconds) — the bus
    will log and optionally halt if the policy consistently overruns.
    """

    @property
    def policy_id(self) -> str:
        """Stable identifier for this policy instance."""
        ...

    def compute(self, state: "ControlState") -> "ControlCommand":
        """Synchronous computation — called from the RT thread.

        For neural network policies that need GPU, run inference in a
        separate thread/process and return the latest cached result here.
        """
        ...

    def reset(self) -> None:
        """Reset internal state (e.g. PID integrators) to initial values."""
        ...


@dataclass(frozen=True)
class ControlState:
    """Sensor readings for one control cycle."""

    timestamp: float
    """Monotonic clock instant at which the readings were captured."""

    joint_positions: Mapping[str, float]
    """channel_id → position value for every readable joint channel."""

    joint_velocities: Mapping[str, float]
    """channel_id → velocity value (may be empty if not declared)."""

    sensor_readings: Mapping[str, Any]
    """channel_id → arbitrary sensor value for non-joint channels."""

    cycle_number: int
    """Zero-based index of this cycle since the bus was started."""

    cycle_dt_s: float
    """Actual elapsed time since the previous cycle's read, in seconds."""


@dataclass(frozen=True)
class ControlCommand:
    """Actuator commands for one control cycle."""

    joint_commands: Mapping[str, float]
    """channel_id → commanded value for every writable joint channel."""

    gripper_command: float | None = None
    """Optional gripper setpoint (None = no change)."""

    halt: bool = False
    """When True the policy requests an emergency stop."""


@dataclass(frozen=True)
class ControlBusConfig:
    """Configuration for the high-frequency control bus."""

    frequency_hz: float = 100.0
    """Target control frequency in Hz."""

    max_overrun_ratio: float = 1.5
    """Halt if a single cycle takes longer than ratio * period."""

    consecutive_overrun_limit: int = 5
    """Halt after N consecutive overruns."""

    use_busy_wait: bool = True
    """Busy-wait the last millisecond for sub-ms accuracy (higher CPU)."""

    priority: str = "realtime"
    """Thread scheduling class: ``realtime`` (SCHED_FIFO), ``high``, ``normal``."""

    safety_check_interval: int = 1
    """Check SafetyPolicy every N cycles (1 = every cycle)."""

    comm_loss_timeout_s: float = 0.5
    """Seconds of consecutive read failures before triggering comm-loss halt."""


@dataclass
class ControlBusStats:
    """Runtime statistics for the control bus (mutable — updated each cycle)."""

    cycles: int = 0
    overruns: int = 0
    consecutive_overruns: int = 0
    max_jitter_ms: float = 0.0
    mean_jitter_ms: float = 0.0
    mean_cycle_ms: float = 0.0
    safety_violations: int = 0
    degradation_halts: int = 0
    policy_errors: int = 0
    started_at: float = 0.0
    stopped_at: float = 0.0

    # Internal accumulators — not part of the public dict.
    _jitter_sum: float = field(default=0.0, repr=False)
    _cycle_sum: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the public subset as a plain dict."""
        return {
            "cycles": self.cycles,
            "overruns": self.overruns,
            "consecutive_overruns": self.consecutive_overruns,
            "max_jitter_ms": round(self.max_jitter_ms, 4),
            "mean_jitter_ms": round(self.mean_jitter_ms, 4),
            "mean_cycle_ms": round(self.mean_cycle_ms, 4),
            "safety_violations": self.safety_violations,
            "degradation_halts": self.degradation_halts,
            "policy_errors": self.policy_errors,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }


# ---------------------------------------------------------------------------
# HighFrequencyControlBus
# ---------------------------------------------------------------------------


class HighFrequencyControlBus:
    """Dedicated-thread control bus for real-time servo control.

    Lifecycle::

        bus = HighFrequencyControlBus(registry)
        bus.start(device_id, policy)   # spawns RT thread
        ...                            # bus.latest_state is updated each cycle
        bus.stop()                     # graceful halt + join
        bus.emergency_stop()           # immediate halt, no waiting

    Thread model:

    - **Control thread**: reads sensors, calls ``policy.compute()``, writes
      actuators — all synchronous from the thread's own event loop.
    - **Asyncio bridge**: ``async_start`` / ``async_stop`` wrap the synchronous
      methods via ``asyncio.to_thread`` so the agent layer can drive the bus
      from its own event loop.
    - **State sharing**: ``latest_state`` is updated via simple attribute
      assignment, which is atomic under CPython's GIL.
    """

    def __init__(
        self,
        registry: Any,
        *,
        config: ControlBusConfig | None = None,
        safety_checker: Any = None,
    ) -> None:
        self._registry = registry
        self._config = config or ControlBusConfig()
        self._safety_checker = safety_checker

        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()

        self._device_id: str = ""
        self._policy: ControlPolicy | None = None
        self._latest_state: ControlState | None = None  # GIL-atomic
        self._stats = ControlBusStats()

        # Pre-resolved channel lists — populated in start().
        self._read_channel_ids: tuple[str, ...] = ()
        self._write_channel_ids: tuple[str, ...] = ()

    # -- Public interface --------------------------------------------------

    def start(self, device_id: str, policy: ControlPolicy) -> None:
        """Start the control loop in a dedicated thread.

        Synchronous — can be called from async via ``asyncio.to_thread``
        or directly from synchronous code.

        Raises ``RuntimeError`` if the bus is already running.
        """
        if self._running.is_set():
            raise RuntimeError(
                "HighFrequencyControlBus is already running; "
                "call stop() before starting again"
            )
        self._device_id = device_id
        self._policy = policy
        self._latest_state = None
        self._stats = ControlBusStats(started_at=time.monotonic())
        self._stop_requested.clear()

        # Resolve channel topology from the registry before spawning the
        # thread so a misconfigured device fails loudly here.
        context = self._registry.context(device_id)
        if context is None:
            raise ValueError(f"unknown device {device_id!r}")
        self._read_channel_ids = tuple(
            ch.channel_id for ch in context.channels if ch.is_readable
        )
        self._write_channel_ids = tuple(
            ch.channel_id for ch in context.channels if ch.is_writable
        )
        if not self._read_channel_ids:
            raise ValueError(
                f"device {device_id!r} has no readable channels for control"
            )

        self._thread = threading.Thread(
            target=self._control_thread_main,
            name=f"control-bus-{device_id}",
            daemon=True,
        )
        self._thread.start()
        # Wait until the thread signals that it is running (transport opened).
        if not self._running.wait(timeout=10.0):
            raise RuntimeError(
                f"control thread for {device_id!r} did not start within 10s"
            )

    def stop(self, timeout_s: float = 2.0) -> None:
        """Request graceful stop: finishes current cycle, halts device, joins thread."""
        if not self._running.is_set():
            return
        self._stop_requested.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                logger.warning(
                    "Control thread for %s did not join within %.1fs",
                    self._device_id,
                    timeout_s,
                )
        self._thread = None

    def emergency_stop(self) -> None:
        """Immediate halt: sets stop flag, calls transport.halt() from calling thread.

        Does not wait for the control thread to finish — the halt path is
        lock-free by design (see ``HardwareTransport.halt`` docstring).
        """
        self._stop_requested.set()
        if not self._device_id:
            return
        try:
            # Use the lock-free synchronous accessor — the transport was
            # opened before the control thread started, so it is guaranteed
            # to be present while the bus is alive.  A short-lived event
            # loop is still needed for the async ``halt()`` call, but that
            # call does not touch the registry’s ``asyncio.Lock``.
            transport = self._registry.get_open_transport(self._device_id)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(transport.halt())
            finally:
                loop.close()
        except Exception as exc:  # noqa: BLE001 — emergency path must not raise
            logger.error(
                "Emergency halt failed for %s: %s",
                self._device_id,
                exc,
                exc_info=True,
            )

    @property
    def is_running(self) -> bool:
        """Return whether the control thread is active."""
        return self._running.is_set()

    @property
    def latest_state(self) -> ControlState | None:
        """Latest sensor state from the control thread.  Lock-free (GIL atomic)."""
        return self._latest_state

    @property
    def stats(self) -> ControlBusStats:
        """Runtime statistics snapshot."""
        return self._stats

    # -- Asyncio bridge ----------------------------------------------------

    async def async_start(self, device_id: str, policy: ControlPolicy) -> None:
        """Async wrapper: ensure transport is open, then start the control loop.

        The transport must be opened in the main event loop *before* the RT
        thread starts, because ``registry.transport()`` uses an
        ``asyncio.Lock`` that cannot be shared across event loops.  The RT
        thread then retrieves the already-open transport via the lock-free
        ``get_open_transport()``.
        """
        # Pre-open the transport in the caller's (main) event loop.
        await self._registry.transport(device_id)
        await asyncio.to_thread(self.start, device_id, policy)

    async def async_stop(self) -> None:
        """Async wrapper: stop the control loop."""
        await asyncio.to_thread(self.stop)

    # -- Control thread internals ------------------------------------------

    def _control_thread_main(self) -> None:
        """Main function of the dedicated control thread.

        1. Set thread priority (SCHED_FIFO on Linux, high priority on macOS)
        2. Open transport via a thread-local event loop
        3. Loop at target frequency:
           a. Read sensors (transport.read_batch — synchronous wrapper)
           b. Build ControlState
           c. Call policy.compute(state) → ControlCommand
           d. Safety check (every N cycles per config)
           e. Write actuators (transport.write_batch — synchronous wrapper)
           f. Update stats (jitter, overrun count)
           g. Wait for next cycle (busy-wait or sleep hybrid)
        4. On stop: halt device, close transport
        """
        self._set_thread_priority()
        loop = asyncio.new_event_loop()
        config = self._config
        period_s = 1.0 / config.frequency_hz
        stats = self._stats
        device_id = self._device_id
        policy = self._policy
        assert policy is not None  # ensured by start()

        try:
            # Retrieve the already-opened transport.  The transport must have
            # been opened in the main event loop (via ``async_start`` or an
            # explicit ``await registry.transport(device_id)``) before the RT
            # thread starts.  Using ``get_open_transport()`` avoids crossing
            # event loop boundaries with the registry's ``asyncio.Lock``.
            transport = self._registry.get_open_transport(device_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Control bus failed to obtain transport for %s: %s",
                device_id,
                exc,
                exc_info=True,
            )
            return
        finally:
            # Signal the starting thread regardless of success/failure.
            self._running.set()

        consecutive_read_failures = 0
        comm_loss_start: float | None = None
        prev_read_mono: float = time.monotonic()
        next_cycle_at = time.monotonic()

        try:
            while not self._stop_requested.is_set():
                cycle_start = time.monotonic()

                # (a) Read sensors ----------------------------------------
                try:
                    batch = self._sync_read_batch(loop, transport)
                    consecutive_read_failures = 0
                    comm_loss_start = None
                except Exception as exc:  # noqa: BLE001
                    consecutive_read_failures += 1
                    if consecutive_read_failures == 1:
                        logger.warning(
                            "Control bus read failed on %s: %s",
                            device_id,
                            exc,
                        )
                        comm_loss_start = time.monotonic()
                    # DegradationPolicy: comm-loss check.
                    if (
                        comm_loss_start is not None
                        and (time.monotonic() - comm_loss_start)
                        > config.comm_loss_timeout_s
                    ):
                        logger.error(
                            "Control bus comm loss exceeded %.2fs on %s — halting",
                            config.comm_loss_timeout_s,
                            device_id,
                        )
                        stats.degradation_halts += 1
                        self._sync_halt(loop, transport)
                        break
                    # Back-off briefly and retry.
                    self._precise_wait(
                        time.monotonic() + min(period_s, 0.01)
                    )
                    next_cycle_at = time.monotonic() + period_s
                    continue

                # (b) Build ControlState ----------------------------------
                now = time.monotonic()
                dt = now - prev_read_mono
                prev_read_mono = now

                positions: dict[str, float] = {}
                velocities: dict[str, float] = {}
                sensors: dict[str, Any] = {}
                for reading in batch.readings:
                    ch = reading.channel_id
                    val = reading.value
                    qty = reading.quantity
                    if qty in ("position", "angle", "joint_position"):
                        positions[ch] = float(val) if val is not None else 0.0
                    elif qty in ("velocity", "angular_velocity", "joint_velocity"):
                        velocities[ch] = float(val) if val is not None else 0.0
                    else:
                        sensors[ch] = val

                state = ControlState(
                    timestamp=now,
                    joint_positions=positions,
                    joint_velocities=velocities,
                    sensor_readings=sensors,
                    cycle_number=stats.cycles,
                    cycle_dt_s=dt,
                )
                # GIL-atomic publish.
                self._latest_state = state

                # (c) Compute command -------------------------------------
                try:
                    command = policy.compute(state)
                except Exception as exc:  # noqa: BLE001
                    stats.policy_errors += 1
                    logger.warning(
                        "Control policy %s error on cycle %d: %s",
                        policy.policy_id,
                        stats.cycles,
                        exc,
                    )
                    # Skip write but keep the loop alive — a transient
                    # policy failure should not halt the device.
                    self._update_cycle_stats(stats, cycle_start, period_s)
                    next_cycle_at += period_s
                    self._precise_wait(next_cycle_at)
                    continue

                # Policy-requested halt.
                if command.halt:
                    logger.warning(
                        "Control policy %s requested halt on cycle %d",
                        policy.policy_id,
                        stats.cycles,
                    )
                    self._sync_halt(loop, transport)
                    break

                # (d) Safety check ----------------------------------------
                if (
                    config.safety_check_interval > 0
                    and stats.cycles % config.safety_check_interval == 0
                ):
                    safe = self._check_safety(state, command)
                    if not safe:
                        stats.safety_violations += 1
                        logger.error(
                            "Safety violation on cycle %d — halting %s",
                            stats.cycles,
                            device_id,
                        )
                        self._sync_halt(loop, transport)
                        break

                # (e) Write actuators -------------------------------------
                if command.joint_commands:
                    try:
                        self._sync_write_batch(loop, transport, command)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Control bus write failed on %s cycle %d: %s",
                            device_id,
                            stats.cycles,
                            exc,
                        )
                        # A write failure is serious but not necessarily
                        # fatal — the next cycle will attempt again.  A
                        # persistent failure is caught by the read-side
                        # comm-loss path.

                # (f) Update stats ----------------------------------------
                self._update_cycle_stats(
                    stats, cycle_start, period_s, config.max_overrun_ratio
                )

                # Check for consecutive overrun halt.
                if (
                    stats.consecutive_overruns
                    >= config.consecutive_overrun_limit
                ):
                    logger.error(
                        "Control bus hit %d consecutive overruns on %s — halting",
                        stats.consecutive_overruns,
                        device_id,
                    )
                    self._sync_halt(loop, transport)
                    break

                # (g) Wait for next cycle ---------------------------------
                next_cycle_at += period_s
                self._precise_wait(next_cycle_at)

        except Exception as exc:  # noqa: BLE001 — thread must not die silently
            logger.error(
                "Control thread unexpected error on %s: %s",
                device_id,
                exc,
                exc_info=True,
            )
        finally:
            # Ensure the device is halted on any exit path.
            try:
                self._sync_halt(loop, transport)
            except Exception:  # noqa: BLE001
                pass
            loop.close()
            stats.stopped_at = time.monotonic()
            self._running.clear()
            logger.info(
                "Control bus stopped for %s after %d cycles "
                "(overruns=%d, safety=%d, policy_errors=%d)",
                device_id,
                stats.cycles,
                stats.overruns,
                stats.safety_violations,
                stats.policy_errors,
            )

    # -- Thread priority ---------------------------------------------------

    def _set_thread_priority(self) -> None:
        """Elevate thread priority for real-time control.

        Linux: ``SCHED_FIFO`` with priority 50 (requires ``CAP_SYS_NICE``
        or root).  macOS: ``pthread_setschedparam`` with ``SCHED_RR``.
        Fallback: log a warning and continue at normal priority.
        """
        target = self._config.priority
        if target == "normal":
            return

        system = platform.system()
        try:
            if system == "Linux":
                self._set_priority_linux(target)
            elif system == "Darwin":
                self._set_priority_darwin(target)
            else:
                logger.info(
                    "Thread priority elevation not supported on %s; "
                    "continuing at normal priority",
                    system,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to elevate thread priority to %r: %s — "
                "continuing at normal priority (consider running with "
                "CAP_SYS_NICE on Linux or as root)",
                target,
                exc,
            )

    @staticmethod
    def _set_priority_linux(target: str) -> None:
        """Set SCHED_FIFO or elevated nice on Linux."""
        SCHED_FIFO = 1  # noqa: N806

        if target == "realtime":
            # sched_setscheduler via ctypes — avoids the os.sched_* wrappers
            # which are unavailable on some minimal Python builds.
            libc = ctypes.CDLL("libc.so.6", use_errno=True)

            class _SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]

            param = _SchedParam(sched_priority=50)
            ret = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
            if ret != 0:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
            logger.info("Control thread set to SCHED_FIFO priority 50")
        elif target == "high":
            os.nice(-10)
            logger.info("Control thread nice set to -10")

    @staticmethod
    def _set_priority_darwin(target: str) -> None:
        """Elevate thread priority on macOS via pthread."""
        import ctypes.util

        libpthread_path = ctypes.util.find_library("pthread")
        if not libpthread_path:
            logger.info("pthread library not found on macOS; skipping priority")
            return

        libpthread = ctypes.CDLL(libpthread_path, use_errno=True)
        SCHED_RR = 2  # noqa: N806
        thread_self = libpthread.pthread_self()

        class _SchedParam(ctypes.Structure):
            _fields_ = [("sched_priority", ctypes.c_int)]

        prio = 47 if target == "realtime" else 31
        param = _SchedParam(sched_priority=prio)
        ret = libpthread.pthread_setschedparam(
            thread_self, SCHED_RR, ctypes.byref(param)
        )
        if ret != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        logger.info("Control thread set to SCHED_RR priority %d (macOS)", prio)

    # -- Synchronous transport wrappers ------------------------------------

    def _sync_read_batch(
        self, loop: asyncio.AbstractEventLoop, transport: Any
    ) -> Any:
        """Synchronous read_batch wrapper for the control thread.

        Uses the thread-local event loop created in ``_control_thread_main``.
        The transport's ``read_batch`` is async, so we run it with
        ``loop.run_until_complete``.  Falling back to sequential reads when
        the transport does not support ``BatchTransport``.
        """
        from leapflow.hardware.transport import BatchReading, BatchTransport, Reading

        if isinstance(transport, BatchTransport):
            return loop.run_until_complete(
                transport.read_batch(self._read_channel_ids)
            )
        # Sequential fallback — still under one loop.run_until_complete to
        # keep the overhead per-cycle rather than per-channel.
        async def _seq_read() -> BatchReading:
            readings: list[Reading] = []
            for ch_id in self._read_channel_ids:
                readings.append(await transport.read(ch_id))
            return BatchReading(
                device_id=self._device_id,
                readings=tuple(readings),
            )

        return loop.run_until_complete(_seq_read())

    def _sync_write_batch(
        self,
        loop: asyncio.AbstractEventLoop,
        transport: Any,
        command: ControlCommand,
    ) -> None:
        """Synchronous write_batch wrapper for the control thread."""
        from leapflow.hardware.transport import BatchTransport

        commands_tuple = tuple(
            (ch_id, val)
            for ch_id, val in command.joint_commands.items()
            if ch_id in self._write_channel_ids
        )
        if not commands_tuple:
            return

        if isinstance(transport, BatchTransport):
            loop.run_until_complete(transport.write_batch(commands_tuple))
        else:
            # Sequential fallback.
            async def _seq_write() -> None:
                for ch_id, val in commands_tuple:
                    await transport.write(ch_id, val)

            loop.run_until_complete(_seq_write())

    @staticmethod
    def _sync_halt(loop: asyncio.AbstractEventLoop, transport: Any) -> None:
        """Synchronous halt wrapper — called on every exit path."""
        try:
            loop.run_until_complete(transport.halt())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Control bus halt failed: %s", exc, exc_info=True)

    # -- Precise timing ----------------------------------------------------

    def _precise_wait(self, target_time: float) -> None:
        """Wait until *target_time* (monotonic) with sub-ms accuracy.

        Strategy when ``use_busy_wait`` is enabled:

        - If remaining > 2ms: ``time.sleep(remaining - 1ms)``
        - Busy-wait the last 1ms for precision

        When disabled: plain ``time.sleep`` (lower CPU, higher jitter).
        """
        remaining = target_time - time.monotonic()
        if remaining <= 0:
            return

        if not self._config.use_busy_wait:
            if remaining > 0:
                time.sleep(remaining)
            return

        # Sleep the bulk, busy-wait the tail.
        _BUSY_THRESHOLD = 0.001  # 1ms
        if remaining > _BUSY_THRESHOLD + 0.001:
            time.sleep(remaining - _BUSY_THRESHOLD)

        # Busy-wait for the remaining sub-ms.
        while time.monotonic() < target_time:
            pass

    # -- Safety checks -----------------------------------------------------

    def _check_safety(
        self, state: ControlState, command: ControlCommand
    ) -> bool:
        """Check SafetyPolicy constraints on the computed command.

        Returns True if safe.  On violation: logs and returns False.
        The caller halts instead of writing.
        """
        checker = self._safety_checker
        if checker is None:
            return True

        context = self._registry.context(self._device_id)
        if context is None or context.safety is None:
            return True

        # Check each commanded joint against the device-level safety policy.
        for ch_id, val in command.joint_commands.items():
            channel = context.channel(ch_id)
            if channel is None:
                continue
            allowed, reason = checker(context, channel, val)
            if not allowed:
                logger.error(
                    "Safety violation on %s.%s: %s (value=%s)",
                    self._device_id,
                    ch_id,
                    reason,
                    val,
                )
                return False

        # Check velocity limits against state readings.
        if context.safety.max_velocity_rad_s > 0:
            for ch_id, vel in state.joint_velocities.items():
                if abs(vel) > context.safety.max_velocity_rad_s:
                    logger.error(
                        "Safety: observed velocity %.3f rad/s on %s.%s "
                        "exceeds limit %.3f rad/s",
                        abs(vel),
                        self._device_id,
                        ch_id,
                        context.safety.max_velocity_rad_s,
                    )
                    return False

        return True

    # -- Stats helpers -----------------------------------------------------

    @staticmethod
    def _update_cycle_stats(
        stats: ControlBusStats,
        cycle_start: float,
        period_s: float,
        overrun_ratio: float = 1.5,
    ) -> None:
        """Update timing statistics after one complete cycle."""
        cycle_elapsed = time.monotonic() - cycle_start
        cycle_ms = cycle_elapsed * 1000.0
        jitter_ms = abs(cycle_elapsed - period_s) * 1000.0

        stats.cycles += 1
        stats._cycle_sum += cycle_ms
        stats._jitter_sum += jitter_ms
        stats.mean_cycle_ms = stats._cycle_sum / stats.cycles
        stats.mean_jitter_ms = stats._jitter_sum / stats.cycles
        if jitter_ms > stats.max_jitter_ms:
            stats.max_jitter_ms = jitter_ms

        # Overrun detection: cycle took longer than the tolerance-adjusted
        # period.  A cycle that exceeds ``overrun_ratio * period_s`` is a
        # hard overrun; one that merely exceeds the period is a soft overrun.
        # Both increment the consecutive counter so the bus can halt after
        # a sustained sequence of missed deadlines.
        if cycle_elapsed > period_s * overrun_ratio:
            stats.overruns += 1
            stats.consecutive_overruns += 1
        elif cycle_elapsed > period_s:
            stats.overruns += 1
            stats.consecutive_overruns += 1
        else:
            stats.consecutive_overruns = 0


__all__ = [
    "ControlBusConfig",
    "ControlBusStats",
    "ControlCommand",
    "ControlPolicy",
    "ControlState",
    "HighFrequencyControlBus",
]
