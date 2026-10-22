"""Append-only run evidence and typed, non-replayable operational failures."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from leapflow.engine.recovery.failure_envelope import (
    FailureEnvelope, FailureSource, Recoverability, SideEffectState,
)


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
        self.audit_path = path.with_suffix(".audit.ndjson")
        self.event_path = path.with_suffix(".events.ndjson")
        if any(p.exists() for p in (path, self.audit_path, self.event_path)):
            raise Refusal("evidence paths already exist; choose a new run name")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as report:
            report.write(encode({"ok": False, "status": "in_progress"}) + "\n")
        self.event_path.open("x").close()
        self.audit_path.open("x").close()

    def record(self, event: str, **details: Any) -> None:
        row = {"event": event, "monotonic": time.monotonic(), **details}
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
        return self.finish({"ok": False, "failure": asdict(exc.envelope)})
