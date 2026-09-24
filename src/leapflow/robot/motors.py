# Copyright (c) Alibaba, Inc. and its affiliates.
"""Motor bus abstractions for LeapRobot.

Defines the ``MotorsBus`` Protocol (replacing the upstream ABC),
supporting data types (``Motor``, ``MotorCalibration``,
``MotorNormMode``), and a ``SerialMotorsBus`` base implementation for
serial-protocol motor buses (Feetech, Dynamixel, …).

pyserial is an optional dependency — the module is importable without
it, but ``SerialMotorsBus.connect()`` requires it at runtime.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Value / NameOrID aliases
# ---------------------------------------------------------------------------

Value = int | float
"""Motor register value — raw int from hardware or normalised float."""

NameOrID = str | int
"""Motor identifier — either the human-readable name or the bus ID."""


# ---------------------------------------------------------------------------
# Motor data types
# ---------------------------------------------------------------------------

class MotorNormMode(str, Enum):
    """Normalisation modes for motor position values."""

    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class MotorCalibration:
    """Persisted calibration parameters for a single motor."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    """Descriptor for a motor connected to a bus."""

    id: int
    model: str
    norm_mode: MotorNormMode
    motor_type_str: str | None = None
    recv_id: int | None = None


# ---------------------------------------------------------------------------
# MotorsBus Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MotorsBus(Protocol):
    """Protocol for motor bus implementations.

    This is the minimal interface that all motor buses must satisfy,
    regardless of their communication protocol (serial, CAN, etc.).
    """

    def connect(self, handshake: bool = True) -> None:
        """Establish connection to the motors."""
        ...

    def disconnect(self, disable_torque: bool = True) -> None:
        """Disconnect from the motors."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check whether the bus is currently connected."""
        ...

    def read(self, data_name: str, motor: str, **kwargs: Any) -> Value:
        """Read a value from a single motor."""
        ...

    def write(self, data_name: str, motor: str, value: Value, **kwargs: Any) -> None:
        """Write a value to a single motor."""
        ...

    def sync_read(
        self, data_name: str, motors: str | list[str] | None = None, **kwargs: Any,
    ) -> dict[str, Value]:
        """Read a value from multiple motors at once."""
        ...

    def sync_write(
        self, data_name: str, values: Value | dict[str, Value], **kwargs: Any,
    ) -> None:
        """Write values to multiple motors at once."""
        ...

    def enable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0,
    ) -> None:
        """Enable torque on selected motors."""
        ...

    def disable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0,
    ) -> None:
        """Disable torque on selected motors."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ctrl_table(
    model_ctrl_table: dict[str, dict[str, tuple[int, int]]],
    model: str,
) -> dict[str, tuple[int, int]]:
    """Look up the control table for a motor model.

    Raises:
        KeyError: If *model* is not found.
    """
    ctrl_table = model_ctrl_table.get(model)
    if ctrl_table is None:
        raise KeyError(f"Control table for model={model!r} not found.")
    return ctrl_table


def get_address(
    model_ctrl_table: dict[str, dict[str, tuple[int, int]]],
    model: str,
    data_name: str,
) -> tuple[int, int]:
    """Return ``(address, byte_length)`` for a register on a motor model.

    Raises:
        KeyError: If *data_name* is not in the model's control table.
    """
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    addr_bytes = ctrl_table.get(data_name)
    if addr_bytes is None:
        raise KeyError(
            f"Address for '{data_name}' not found in {model} control table."
        )
    return addr_bytes


# ---------------------------------------------------------------------------
# SerialMotorsBus
# ---------------------------------------------------------------------------

class SerialMotorsBus:
    """Base implementation for serial-based motor buses.

    This class captures the common read/write, sync-read/write,
    normalisation, and lifecycle logic shared between Feetech and
    Dynamixel buses.  Subclasses must populate the class-level table
    attributes and implement the abstract hooks.

    pyserial (``serial``) is imported at *connection* time so the
    module can be imported in environments where hardware is absent.
    """

    # -- Subclasses MUST populate these class attributes --
    apply_drive_mode: bool = False
    available_baudrates: list[int] = []
    default_baudrate: int = 1_000_000
    default_timeout: int = 1000
    model_baudrate_table: dict[str, dict[int, int]] = {}
    model_ctrl_table: dict[str, dict[str, tuple[int, int]]] = {}
    model_encoding_table: dict[str, dict[str, int]] = {}
    model_number_table: dict[str, int] = {}
    model_resolution_table: dict[str, int] = {}
    normalized_data: list[str] = []

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        self.port = port
        self.motors = motors
        self.calibration: dict[str, MotorCalibration] = calibration or {}

        # Internal look-up tables
        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: name for name, m in self.motors.items()}
        self._model_nb_to_model_dict: dict[int, str] = (
            {v: k for k, v in self.model_number_table.items()}
            if self.model_number_table
            else {}
        )

        self._validate_motors()

        # Populated by _connect / subclass __init__
        self.port_handler: Any = None
        self.packet_handler: Any = None
        self.sync_reader: Any = None
        self.sync_writer: Any = None
        self._comm_success: int = 0
        self._no_error: int = 0

    # -- identification helpers -------------------------------------------

    @property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        return motor

    def _get_motor_model(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            return self.motors[motor].model
        return self._id_to_model_dict[motor]

    def _get_motors_list(
        self, motors: NameOrID | Sequence[NameOrID] | None,
    ) -> list[str]:
        if motors is None:
            return list(self.motors)
        if isinstance(motors, str):
            return [motors]
        if isinstance(motors, int):
            return [self._id_to_name(motors)]
        return [
            m if isinstance(m, str) else self._id_to_name(m) for m in motors
        ]

    def _get_ids_values_dict(
        self, values: Value | dict[str, Value],
    ) -> dict[int, Value]:
        if isinstance(values, (int, float)):
            return dict.fromkeys(self.ids, values)
        if isinstance(values, dict):
            return {
                self.motors[motor].id: val for motor, val in values.items()
            }
        raise TypeError(f"'values' must be a scalar or dict, got {type(values)}")

    # -- validation -------------------------------------------------------

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Duplicate motor IDs detected: {self.ids}")
        for model in self.models:
            if self.model_ctrl_table:
                get_ctrl_table(self.model_ctrl_table, model)

    # -- communication predicates -----------------------------------------

    def _is_comm_success(self, comm: int) -> bool:
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        return error != self._no_error

    # -- connection lifecycle ---------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` if the underlying serial port is open."""
        if self.port_handler is None:
            return False
        return getattr(self.port_handler, "is_open", False)

    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and initialise communication.

        Args:
            handshake: If ``True``, ping every motor and run
                subclass-specific integrity checks.
        """
        if self.is_connected:
            return
        self._connect(handshake)
        self._set_timeout()
        logger.debug("%s connected on %s.", type(self).__name__, self.port)

    def _connect(self, handshake: bool = True) -> None:
        """Low-level port open — subclasses override to set SDK handles."""
        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            if handshake:
                self._handshake()
        except Exception as exc:
            raise ConnectionError(
                f"Could not connect on port '{self.port}'."
            ) from exc

    def _handshake(self) -> None:
        """Subclass hook called after the port is opened."""

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port.

        Args:
            disable_torque: If ``True`` (default) torque is disabled on
                all motors before closing to avoid damage.
        """
        if not self.is_connected:
            return
        if disable_torque:
            try:
                self.disable_torque(num_retry=5)
            except Exception:
                logger.warning("Failed to disable torque during disconnect.")
        self.port_handler.closePort()
        logger.debug("%s disconnected.", type(self).__name__)

    def _set_timeout(self, timeout_ms: int | None = None) -> None:
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        if self.port_handler is not None:
            self.port_handler.setPacketTimeoutMillis(timeout_ms)

    # -- normalisation ----------------------------------------------------

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError("No calibration registered.")
        result: dict[int, float] = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            cal = self.calibration[motor]
            if cal.range_max == cal.range_min:
                raise ValueError(
                    f"Invalid calibration for motor '{motor}': "
                    f"min and max are equal."
                )
            bounded = min(cal.range_max, max(cal.range_min, val))
            drive = self.apply_drive_mode and cal.drive_mode
            nm = self.motors[motor].norm_mode
            if nm is MotorNormMode.RANGE_M100_100:
                norm = ((bounded - cal.range_min) / (cal.range_max - cal.range_min)) * 200 - 100
                result[id_] = -norm if drive else norm
            elif nm is MotorNormMode.RANGE_0_100:
                norm = ((bounded - cal.range_min) / (cal.range_max - cal.range_min)) * 100
                result[id_] = 100 - norm if drive else norm
            elif nm is MotorNormMode.DEGREES:
                mid = (cal.range_min + cal.range_max) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                result[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError(f"Unsupported norm mode: {nm}")
        return result

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError("No calibration registered.")
        result: dict[int, int] = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            cal = self.calibration[motor]
            if cal.range_max == cal.range_min:
                raise ValueError(
                    f"Invalid calibration for motor '{motor}': "
                    f"min and max are equal."
                )
            drive = self.apply_drive_mode and cal.drive_mode
            nm = self.motors[motor].norm_mode
            if nm is MotorNormMode.RANGE_M100_100:
                v = -val if drive else val
                bounded = min(100.0, max(-100.0, v))
                result[id_] = int(
                    ((bounded + 100) / 200) * (cal.range_max - cal.range_min) + cal.range_min
                )
            elif nm is MotorNormMode.RANGE_0_100:
                v = 100 - val if drive else val
                bounded = min(100.0, max(0.0, v))
                result[id_] = int(
                    (bounded / 100) * (cal.range_max - cal.range_min) + cal.range_min
                )
            elif nm is MotorNormMode.DEGREES:
                mid = (cal.range_min + cal.range_max) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                result[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError(f"Unsupported norm mode: {nm}")
        return result

    # -- serial data helpers ----------------------------------------------

    def _serialize_data(self, value: int, length: int) -> list[int]:
        """Convert an unsigned int to a list of byte-sized ints."""
        if value < 0:
            raise ValueError(f"Negative values not allowed: {value}")
        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {length}")
        if value > max_value:
            raise ValueError(f"Value {value} exceeds max for {length} bytes.")
        return self._split_into_byte_chunks(value, length)

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """Subclasses must override with endianness-specific logic."""
        raise NotImplementedError

    # -- read / write (single motor) --------------------------------------

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
        """Read a register from a single motor."""
        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        value, _, _ = self._read(
            addr, length, id_, num_retry=num_retry, raise_on_error=True,
        )
        decoded = self._decode_sign(data_name, {id_: value})
        if normalize and data_name in self.normalized_data:
            return self._normalize(decoded)[id_]
        return decoded[id_]

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
    ) -> tuple[int, int, int]:
        """Low-level register read — subclasses should override."""
        raise NotImplementedError

    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write a value to a single motor's register."""
        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        int_value = int(value)
        if normalize and data_name in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]
        int_value = self._encode_sign(data_name, {id_: int_value})[id_]
        self._write(
            addr, length, id_, int_value,
            num_retry=num_retry, raise_on_error=True,
        )

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
    ) -> tuple[int, int]:
        """Low-level register write — subclasses should override."""
        raise NotImplementedError

    # -- sync read / write ------------------------------------------------

    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """Read the same register from multiple motors."""
        names = self._get_motors_list(motors)
        ids = [self.motors[m].id for m in names]
        models = [self.motors[m].model for m in names]
        model = models[0]
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        raw, _ = self._sync_read(addr, length, ids, num_retry=num_retry)
        decoded = self._decode_sign(data_name, raw)
        if normalize and data_name in self.normalized_data:
            normed = self._normalize(decoded)
            return {self._id_to_name(id_): v for id_, v in normed.items()}
        return {self._id_to_name(id_): v for id_, v in decoded.items()}

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
    ) -> tuple[dict[int, int], int]:
        """Low-level sync read — subclasses should override."""
        raise NotImplementedError

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write the same register on multiple motors."""
        raw_ids = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in raw_ids]
        model = models[0]
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        int_ids = {id_: int(v) for id_, v in raw_ids.items()}
        if normalize and data_name in self.normalized_data:
            int_ids = self._unnormalize(raw_ids)
        int_ids = self._encode_sign(data_name, int_ids)
        self._sync_write(addr, length, int_ids, num_retry=num_retry)

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        *,
        num_retry: int = 0,
    ) -> int:
        """Low-level sync write — subclasses should override."""
        raise NotImplementedError

    # -- torque control ---------------------------------------------------

    def enable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0,
    ) -> None:
        """Enable torque on selected motors — subclasses should override."""
        raise NotImplementedError

    def disable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0,
    ) -> None:
        """Disable torque on selected motors — subclasses should override."""
        raise NotImplementedError

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """Context manager that guarantees torque is re-enabled."""
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    # -- sign encoding (subclass hooks) -----------------------------------

    def _encode_sign(
        self, data_name: str, ids_values: dict[int, int],
    ) -> dict[int, int]:
        """Encode signed values — subclasses override per protocol."""
        return ids_values

    def _decode_sign(
        self, data_name: str, ids_values: dict[int, int],
    ) -> dict[int, int]:
        """Decode signed values — subclasses override per protocol."""
        return ids_values

    # -- calibration (subclass hooks) -------------------------------------

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration from hardware — subclasses should override."""
        raise NotImplementedError

    def write_calibration(
        self,
        calibration_dict: dict[str, MotorCalibration],
        cache: bool = True,
    ) -> None:
        """Write calibration to hardware — subclasses should override."""
        raise NotImplementedError

    # -- repr / len -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.motors)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(port={self.port!r}, "
            f"motors={list(self.motors.keys())})"
        )


__all__ = [
    "Value",
    "NameOrID",
    "MotorNormMode",
    "MotorCalibration",
    "Motor",
    "MotorsBus",
    "SerialMotorsBus",
    "get_ctrl_table",
    "get_address",
]
