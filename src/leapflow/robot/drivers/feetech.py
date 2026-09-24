# Copyright (c) Alibaba, Inc. and its affiliates.
"""Feetech servo motor bus driver for LeapRobot.

Implements ``SerialMotorsBus`` for the Feetech STS / SMS / SCS series
servo motors.  The Feetech SDK (``scservo_sdk``) is an optional
dependency imported at connection time.

This module absorbs and adapts the inference-relevant subset of the
upstream open-source Feetech driver (see copyright notice), removing interactive
calibration UI and training-only utilities.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from enum import Enum
from pprint import pformat
from typing import TYPE_CHECKING, Any

from leapflow.robot.motors import (
    Motor,
    MotorCalibration,
    NameOrID,
    SerialMotorsBus,
    Value,
    get_address,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feetech SDK — optional dependency
# ---------------------------------------------------------------------------
try:
    import scservo_sdk as scs

    _SCS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCS_AVAILABLE = False
    scs = None  # type: ignore[assignment]


def _require_scs() -> None:
    if not _SCS_AVAILABLE:
        raise ImportError(
            "feetech-servo-sdk is required for Feetech motors.  "
            "Install with: pip install feetech-servo-sdk"
        )


# ---------------------------------------------------------------------------
# Sign-magnitude encoding helpers
# ---------------------------------------------------------------------------

def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    """Encode a signed integer to sign-magnitude representation."""
    if value < 0:
        return (-value) | (1 << sign_bit)
    return value


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    """Decode a sign-magnitude value to a signed integer."""
    if value & (1 << sign_bit):
        return -(value & ~(1 << sign_bit))
    return value


# ---------------------------------------------------------------------------
# Control tables
# ---------------------------------------------------------------------------

FIRMWARE_MAJOR_VERSION = (0, 1)
FIRMWARE_MINOR_VERSION = (1, 1)
MODEL_NUMBER_ADDR = (3, 2)

# STS / SMS series control table (address, byte_length)
STS_SMS_SERIES_CONTROL_TABLE: dict[str, tuple[int, int]] = {
    # EPROM
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,
    "Model_Number": MODEL_NUMBER_ADDR,
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Velocity_closed_loop_P": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I": (39, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    "Maximum_Acceleration": (85, 1),
}

# SCS series control table
SCS_SERIES_CONTROL_TABLE: dict[str, tuple[int, int]] = {
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,
    "Model_Number": MODEL_NUMBER_ADDR,
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protective_Torque": (37, 1),
    "Protection_Time": (38, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Running_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Lock": (48, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
}

STS_SMS_SERIES_BAUDRATE_TABLE: dict[int, int] = {
    1_000_000: 0, 500_000: 1, 250_000: 2, 128_000: 3,
    115_200: 4, 57_600: 5, 38_400: 6, 19_200: 7,
}

SCS_SERIES_BAUDRATE_TABLE: dict[int, int] = {
    1_000_000: 0, 500_000: 1, 250_000: 2, 128_000: 3,
    115_200: 4, 57_600: 5, 38_400: 6, 19_200: 7,
}

MODEL_CONTROL_TABLE: dict[str, dict[str, tuple[int, int]]] = {
    "sts_series": STS_SMS_SERIES_CONTROL_TABLE,
    "scs_series": SCS_SERIES_CONTROL_TABLE,
    "sms_series": STS_SMS_SERIES_CONTROL_TABLE,
    "sts3215": STS_SMS_SERIES_CONTROL_TABLE,
    "sts3250": STS_SMS_SERIES_CONTROL_TABLE,
    "scs0009": SCS_SERIES_CONTROL_TABLE,
    "sm8512bl": STS_SMS_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION: dict[str, int] = {
    "sts_series": 4096, "sms_series": 4096, "scs_series": 1024,
    "sts3215": 4096, "sts3250": 4096, "sm8512bl": 4096, "scs0009": 1024,
}

MODEL_BAUDRATE_TABLE: dict[str, dict[int, int]] = {
    "sts_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sms_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sm8512bl": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3215": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3250": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs0009": SCS_SERIES_BAUDRATE_TABLE,
}

STS_SMS_SERIES_ENCODINGS: dict[str, int] = {
    "Present_Load": 10, "Homing_Offset": 11,
    "Goal_Position": 15, "Goal_Velocity": 15, "Goal_Speed": 15,
    "Present_Position": 15, "Present_Velocity": 15, "Present_Speed": 15,
}

MODEL_ENCODING_TABLE: dict[str, dict[str, int]] = {
    "sts_series": STS_SMS_SERIES_ENCODINGS,
    "sms_series": STS_SMS_SERIES_ENCODINGS,
    "scs_series": {},
    "sts3215": STS_SMS_SERIES_ENCODINGS,
    "sts3250": STS_SMS_SERIES_ENCODINGS,
    "sm8512bl": STS_SMS_SERIES_ENCODINGS,
    "scs0009": {},
}

SCAN_BAUDRATES: list[int] = [
    4_800, 9_600, 14_400, 19_200, 38_400, 57_600,
    115_200, 128_000, 250_000, 500_000, 1_000_000,
]

MODEL_NUMBER_TABLE: dict[str, int] = {
    "sts3215": 777, "sts3250": 2825, "sm8512bl": 11272, "scs0009": 1284,
}

MODEL_PROTOCOL: dict[str, int] = {
    "sts_series": 0, "sms_series": 0, "scs_series": 1,
    "sts3215": 0, "sts3250": 0, "sm8512bl": 0, "scs0009": 1,
}

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000
NORMALIZED_DATA = ["Goal_Position", "Present_Position"]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OperatingMode(Enum):
    """Feetech motor operating modes."""
    POSITION = 0
    VELOCITY = 1
    PWM = 2
    STEP = 3


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


# ---------------------------------------------------------------------------
# SDK timeout patch
# ---------------------------------------------------------------------------

def _patch_set_packet_timeout(port_handler: Any, packet_length: int) -> None:
    """Fix scservo_sdk timeout calculation bug.

    The official Feetech SDK on PyPI has incorrect timeout maths;
    this patch matches the fix in the canonical Gitee repository.
    """
    port_handler.packet_start_time = port_handler.getCurrentTime()
    port_handler.packet_timeout = (
        (port_handler.tx_time_per_byte * packet_length)
        + (port_handler.tx_time_per_byte * 3.0)
        + 50
    )


# ---------------------------------------------------------------------------
# FeetechMotorsBus
# ---------------------------------------------------------------------------

class FeetechMotorsBus(SerialMotorsBus):
    """Feetech STS / SMS / SCS motor bus driver.

    Wraps the ``scservo_sdk`` library behind the ``SerialMotorsBus``
    interface.  The SDK is imported lazily at connection time.
    """

    apply_drive_mode = True
    available_baudrates = deepcopy(SCAN_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ) -> None:
        _require_scs()
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version

        # Validate protocol consistency
        for model in self.models:
            expected = MODEL_PROTOCOL.get(model, 0)
            if expected != self.protocol_version:
                raise ValueError(
                    f"Motor model {model!r} requires protocol {expected}, "
                    f"but bus uses {self.protocol_version}."
                )

        # SDK handles
        self.port_handler = scs.PortHandler(self.port)
        self.port_handler.setPacketTimeout = (
            _patch_set_packet_timeout.__get__(self.port_handler)
        )
        self.packet_handler = scs.PacketHandler(protocol_version)
        self.sync_reader = scs.GroupSyncRead(
            self.port_handler, self.packet_handler, 0, 0,
        )
        self.sync_writer = scs.GroupSyncWrite(
            self.port_handler, self.packet_handler, 0, 0,
        )
        self._comm_success = scs.COMM_SUCCESS
        self._no_error = 0x00

    # -- handshake --------------------------------------------------------

    def _handshake(self) -> None:
        self._assert_motors_exist()

    def _assert_motors_exist(self) -> None:
        """Ping every registered motor and verify model numbers."""
        expected = {m.id: self.model_number_table[m.model] for m in self.motors.values()}
        found: dict[int, int] = {}
        for id_ in self.ids:
            model_nb = self._ping(id_)
            if model_nb is not None:
                found[id_] = model_nb

        missing = [id_ for id_ in self.ids if id_ not in found]
        wrong = {
            id_: (expected[id_], found[id_])
            for id_ in found if expected.get(id_) != found[id_]
        }
        if missing or wrong:
            parts = [f"{type(self).__name__} check failed on '{self.port}':"]
            if missing:
                parts.append(f"  Missing IDs: {missing}")
            if wrong:
                parts.append(f"  Wrong models: {wrong}")
            raise RuntimeError("\n".join(parts))

    def _ping(self, motor_id: int, num_retry: int = 0) -> int | None:
        """Ping a motor and return its model number, or ``None``."""
        for _ in range(1 + num_retry):
            model_nb, comm, error = self.packet_handler.ping(
                self.port_handler, motor_id,
            )
            if self._is_comm_success(comm):
                return model_nb
        return None

    # -- protocol compatibility -------------------------------------------

    def _assert_protocol_compatible(self, instruction: str) -> None:
        if instruction == "sync_read" and self.protocol_version == 1:
            raise NotImplementedError(
                "Sync Read is not available with Feetech Protocol 1."
            )

    # -- low-level read / write -------------------------------------------

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
    ) -> tuple[int, int, int]:
        read_fn = {
            1: self.packet_handler.read1ByteTxRx,
            2: self.packet_handler.read2ByteTxRx,
            4: self.packet_handler.read4ByteTxRx,
        }.get(length)
        if read_fn is None:
            raise ValueError(f"Unsupported register length: {length}")

        comm = error = 0
        value = 0
        for _ in range(1 + num_retry):
            value, comm, error = read_fn(self.port_handler, motor_id, address)
            if self._is_comm_success(comm):
                break

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(
                f"Read @{address} (len={length}) failed on id={motor_id}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
        return value, comm, error

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
        data = self._serialize_data(value, length)
        comm = error = 0
        for _ in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(
                self.port_handler, motor_id, addr, length, data,
            )
            if self._is_comm_success(comm):
                break

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(
                f"Write @{addr} (len={length}) failed on id={motor_id}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
        return comm, error

    # -- sync read / write ------------------------------------------------

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
    ) -> tuple[dict[int, int], int]:
        self._assert_protocol_compatible("sync_read")
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)

        comm = 0
        for _ in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break

        values = {
            id_: self.sync_reader.getData(id_, addr, length)
            for id_ in motor_ids
        }
        return values, comm

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        *,
        num_retry: int = 0,
    ) -> int:
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for id_, value in ids_values.items():
            data = self._serialize_data(value, length)
            self.sync_writer.addParam(id_, data)

        comm = 0
        for _ in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break
        return comm

    # -- byte splitting (little-endian) -----------------------------------

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        if length == 1:
            return [value]
        elif length == 2:
            return [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        elif length == 4:
            return [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
            ]
        raise NotImplementedError(f"Unsupported length: {length}")

    # -- sign encoding / decoding -----------------------------------------

    def _encode_sign(
        self, data_name: str, ids_values: dict[int, int],
    ) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            enc_table = self.model_encoding_table.get(model)
            if enc_table and data_name in enc_table:
                ids_values[id_] = encode_sign_magnitude(
                    ids_values[id_], enc_table[data_name],
                )
        return ids_values

    def _decode_sign(
        self, data_name: str, ids_values: dict[int, int],
    ) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            enc_table = self.model_encoding_table.get(model)
            if enc_table and data_name in enc_table:
                ids_values[id_] = decode_sign_magnitude(
                    ids_values[id_], enc_table[data_name],
                )
        return ids_values

    # -- torque control ---------------------------------------------------

    def enable_torque(
        self, motors: int | str | list[str] | None = None, num_retry: int = 0,
    ) -> None:
        for motor in self._get_motors_list(motors):
            self.write(
                "Torque_Enable", motor,
                TorqueMode.ENABLED.value, num_retry=num_retry,
            )
            self.write("Lock", motor, 1, num_retry=num_retry)

    def disable_torque(
        self, motors: int | str | list[str] | None = None, num_retry: int = 0,
    ) -> None:
        for motor in self._get_motors_list(motors):
            self.write(
                "Torque_Enable", motor,
                TorqueMode.DISABLED.value, num_retry=num_retry,
            )
            self.write("Lock", motor, 0, num_retry=num_retry)

    # -- calibration ------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        if not self.calibration:
            return False
        hw_cal = self.read_calibration()
        if set(hw_cal) != set(self.calibration):
            return False
        return all(
            self.calibration[m].range_min == c.range_min
            and self.calibration[m].range_max == c.range_max
            for m, c in hw_cal.items()
        )

    def read_calibration(self) -> dict[str, MotorCalibration]:
        calibration: dict[str, MotorCalibration] = {}
        for motor, m in self.motors.items():
            min_pos = self.read("Min_Position_Limit", motor, normalize=False)
            max_pos = self.read("Max_Position_Limit", motor, normalize=False)
            offset = (
                self.read("Homing_Offset", motor, normalize=False)
                if self.protocol_version == 0
                else 0
            )
            calibration[motor] = MotorCalibration(
                id=m.id, drive_mode=0, homing_offset=int(offset),
                range_min=int(min_pos), range_max=int(max_pos),
            )
        return calibration

    def write_calibration(
        self,
        calibration_dict: dict[str, MotorCalibration],
        cache: bool = True,
    ) -> None:
        for motor, cal in calibration_dict.items():
            if self.protocol_version == 0:
                self.write("Homing_Offset", motor, cal.homing_offset)
            self.write("Min_Position_Limit", motor, cal.range_min)
            self.write("Max_Position_Limit", motor, cal.range_max)
        if cache:
            self.calibration = calibration_dict

    # -- motor configuration ----------------------------------------------

    def configure_motors(
        self,
        return_delay_time: int = 0,
        maximum_acceleration: int = 254,
        acceleration: int = 254,
    ) -> None:
        """Apply recommended runtime configuration to all motors."""
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)
            if self.protocol_version == 0:
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            self.write("Acceleration", motor, acceleration)


__all__ = [
    "FeetechMotorsBus",
    "OperatingMode",
    "DriveMode",
    "TorqueMode",
    "encode_sign_magnitude",
    "decode_sign_magnitude",
    # Tables
    "MODEL_CONTROL_TABLE",
    "MODEL_RESOLUTION",
    "MODEL_BAUDRATE_TABLE",
    "MODEL_ENCODING_TABLE",
    "MODEL_NUMBER_TABLE",
    "MODEL_PROTOCOL",
    "SCAN_BAUDRATES",
]
