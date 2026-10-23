"""Append-only run evidence and typed, non-replayable operational failures."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path
from typing import Any

from leapflow.engine.recovery.failure_envelope import (
    FailureEnvelope, FailureSource, Recoverability, SideEffectState,
)
from leapflow.hardware.transport import worst_side_effect_state


class Refusal(RuntimeError):
    """A structured refusal; never a signal to retry an actuator command."""

    def __init__(self, message: str, *, code: str = "piper_operations_refused",
                 state: SideEffectState = SideEffectState.NONE) -> None:
        super().__init__(message)
        internal = code == "internal_defect"
        self.envelope = FailureEnvelope.create(
            source=FailureSource.SYSTEM if internal else FailureSource.TOOL,
            category="internal_defect" if internal else "hardware", failure_class=type(self).__name__,
            failure_code=code, message=message,
            recoverability=Recoverability.NON_RECOVERABLE if internal else Recoverability.USER_FIXABLE,
            side_effect_state=state,
        )


class StaleFeedback(Refusal):
    """A momentary receive-quality gap: re-sample it, never re-command the arm.

    The receive contract certifies one sample, not the device: a CAN group that misses
    its freshness window leaves the servo exactly where it was. An observation may
    therefore read again inside a bounded window, and every decision still fails closed
    because the unqualified sample never enters a check as a valid reading.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code="piper_feedback_stale")


def json_default(value: Any) -> Any:
    """Serialize only known evidence types, without arbitrary repr leakage."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported evidence type {type(value).__name__}")


def encode(value: Any) -> str:
    return json.dumps(value, indent=2, default=json_default, allow_nan=False)


class Evidence:
    """Reserve new report and event paths before any side effect."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._side_effect_state = SideEffectState.NONE
        self.audit_path = path.with_suffix(".audit.ndjson")
        self.event_path = path.with_suffix(".events.ndjson")
        if any(p.exists() for p in (path, self.audit_path, self.event_path)):
            raise Refusal("evidence paths already exist; choose a new run name")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as report:
            report.write(encode({"ok": False, "status": "in_progress"}) + "\n")
        self.event_path.open("x").close()
        self.audit_path.open("x").close()

    def observe_effect(self, state: Any) -> None:
        """Keep the worst observed verdict; a later halt is not a rollback."""
        try:
            observed = SideEffectState(state)
        except (ValueError, TypeError):
            observed = SideEffectState.UNKNOWN
        self._side_effect_state = SideEffectState(worst_side_effect_state(
            (self._side_effect_state.value, observed.value)))

    def failure_envelope(self, exc: Refusal) -> FailureEnvelope:
        """Preserve failure identity while adding this run's observed effects."""
        state = SideEffectState(worst_side_effect_state(
            (exc.envelope.side_effect_state.value, self._side_effect_state.value)))
        return replace(exc.envelope, side_effect_state=state)

    def record(self, event: str, **details: Any) -> None:
        row = {"event": event, "monotonic": time.monotonic(), **details}
        if "side_effect_state" in details:
            self.observe_effect(details["side_effect_state"])
        result = details.get("result")
        if isinstance(result, dict) and "side_effect_state" in result:
            self.observe_effect(result["side_effect_state"])
        self.events.append(row)
        with self.event_path.open("a") as stream:
            stream.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, result: dict[str, Any]) -> dict[str, Any]:
        result = {**result, "events": self.events}
        with self.path.open("w") as stream:
            stream.write(encode(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result

    def failure(self, exc: Refusal) -> dict[str, Any]:
        return self.finish({"ok": False, "failure": asdict(self.failure_envelope(exc))})
