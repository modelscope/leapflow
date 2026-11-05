# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapSpace adapter for daemon-native task-environment observations."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from leapflow.domain.environment_signal import EnvironmentObservation, InterfaceSnapshot
from leapflow.perception.environment_source import EnvironmentEmit

logger = logging.getLogger(__name__)


class LeapSpaceEnvironmentSource:
    """Poll LeapSpace's atomic state envelopes without importing its UI runtime."""

    def __init__(
        self,
        state_root: Path | str,
        *,
        workspace_id: str = "",
        session_id: str = "",
        poll_interval_s: float = 0.5,
    ) -> None:
        self._root = Path(state_root).expanduser().resolve()
        self._workspace_id = str(workspace_id)
        self._session_id = str(session_id)
        self._poll_interval_s = max(0.05, float(poll_interval_s))
        self._snapshots: dict[str, InterfaceSnapshot] = {}
        self._result_hashes: dict[str, str] = {}
        self._stopped = asyncio.Event()

    @property
    def source_id(self) -> str:
        return "leapspace"

    async def start(self, emit: EnvironmentEmit) -> None:
        self._stopped.clear()
        while not self._stopped.is_set():
            for observation in await asyncio.to_thread(self._scan):
                await emit(observation)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._poll_interval_s)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        self._stopped.set()

    def _scan(self) -> tuple[EnvironmentObservation, ...]:
        if not self._root.exists():
            return ()
        observations: list[EnvironmentObservation] = []
        for state_path in sorted(self._root.glob("*/state.json")):
            envelope = self._read_json(state_path)
            if not envelope:
                continue
            app_id = str(envelope.get("app_id") or state_path.parent.name)
            snapshot = InterfaceSnapshot.create(
                source_id=self.source_id,
                app_id=app_id,
                workspace_id=self._workspace_id,
                session_id=self._session_id,
                version=str(envelope.get("version") or ""),
                affordances=envelope.get("affordances") or (),
                elements=envelope.get("elements") or (),
                data=envelope.get("data") or {},
                observed_at=state_path.stat().st_mtime,
                provenance={"kind": "leapspace_state", "path": f"{app_id}/state.json"},
            )
            previous = self._snapshots.get(app_id)
            self._snapshots[app_id] = snapshot
            observation = (
                EnvironmentObservation.snapshot(snapshot)
                if previous is None
                else EnvironmentObservation.between(previous, snapshot)
            )
            if observation is not None:
                observations.append(observation)

        for result_path in sorted(self._root.glob("*/result.json")):
            payload = self._read_json(result_path)
            if not payload:
                continue
            task_id = str(payload.get("task_id") or result_path.parent.name)
            digest = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            if self._result_hashes.get(task_id) == digest:
                continue
            self._result_hashes[task_id] = digest
            observations.append(
                EnvironmentObservation.task_outcome(
                    source_id=self.source_id,
                    task_id=task_id,
                    outcome=str(payload.get("outcome") or "UNKNOWN"),
                    capability=str(payload.get("capability") or ""),
                    workspace_id=self._workspace_id,
                    session_id=self._session_id,
                    observed_at=result_path.stat().st_mtime,
                    provenance={
                        "kind": "leapspace_verdict",
                        "path": f"{task_id}/result.json",
                        "exit_code": payload.get("exit_code", ""),
                    },
                )
            )
        return tuple(observations)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}


__all__ = ["LeapSpaceEnvironmentSource"]
