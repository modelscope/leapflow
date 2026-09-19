# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contracts for daemon-native task-environment observation."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from leapflow.domain.environment_signal import EnvironmentObservation, InterfaceSnapshot
from leapflow.perception.environment_source import EnvironmentSourceManager
from leapflow.perception.leapspace_source import LeapSpaceEnvironmentSource


def _write_state(root: Path, *, version: str, affordances: list[str], enabled: bool = True) -> None:
    app_dir = root / "chat"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "state.json").write_text(
        json.dumps(
            {
                "app_id": "chat",
                "version": version,
                "affordances": affordances,
                "elements": [{"name": "send", "role": "QPushButton", "enabled": enabled}],
                "data": {"draft": "value that must not define structural drift"},
            }
        )
    )


def test_interface_delta_uses_structure_not_mutable_app_data() -> None:
    before = InterfaceSnapshot.create(
        source_id="leapspace",
        app_id="chat",
        affordances=("chat.send.v1",),
        elements=({"name": "send", "role": "button", "enabled": True},),
        data={"draft": "one"},
    )
    data_only = InterfaceSnapshot.create(
        source_id="leapspace",
        app_id="chat",
        affordances=("chat.send.v1",),
        elements=({"name": "send", "role": "button", "enabled": True},),
        data={"draft": "two"},
    )
    changed = InterfaceSnapshot.create(
        source_id="leapspace",
        app_id="chat",
        version="2",
        affordances=("chat.send.v2",),
        elements=({"name": "submit", "role": "button", "enabled": True},),
    )

    assert EnvironmentObservation.between(before, data_only) is None
    delta = EnvironmentObservation.between(before, changed)
    assert delta is not None and delta.is_structural
    assert delta.removed_affordances == ("chat.send.v1",)
    assert delta.capability_results()[0]["capability"] == "chat.send.v1"


def test_leapspace_source_emits_snapshot_delta_and_ground_truth_once(tmp_path: Path) -> None:
    _write_state(tmp_path, version="1", affordances=["chat.send.v1"])
    source = LeapSpaceEnvironmentSource(
        tmp_path,
        workspace_id="ws-1",
        session_id="session-1",
    )

    initial = source._scan()
    assert len(initial) == 1 and initial[0].kind == "snapshot"

    _write_state(tmp_path, version="2", affordances=["chat.send.v2"])
    delta = source._scan()
    assert len(delta) == 1 and delta[0].kind == "delta"
    assert delta[0].session_id == "session-1"

    result_dir = tmp_path / "task-1"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        json.dumps({"task_id": "task-1", "outcome": "FAIL", "capability": "chat.send.v2"})
    )
    outcome = source._scan()
    assert len(outcome) == 1 and outcome[0].outcome == "FAIL"
    assert outcome[0].capability_results()[0]["error_type"] == "task_outcome_failed"
    assert source._scan() == ()


@pytest.mark.asyncio
async def test_environment_source_manager_owns_source_lifecycle() -> None:
    received: list[EnvironmentObservation] = []

    class _Source:
        source_id = "test-source"

        def __init__(self) -> None:
            self.stopped = asyncio.Event()

        async def start(self, emit) -> None:
            await emit(
                EnvironmentObservation.task_outcome(
                    source_id=self.source_id,
                    task_id="task-1",
                    outcome="PASS",
                    session_id="session-1",
                )
            )
            await self.stopped.wait()

        async def stop(self) -> None:
            self.stopped.set()

    async def sink(observation: EnvironmentObservation) -> None:
        received.append(observation)

    source = _Source()
    manager = EnvironmentSourceManager(sink, shutdown_timeout_s=1)
    manager.register(source)
    await manager.start()
    for _ in range(20):
        if received:
            break
        await asyncio.sleep(0.01)
    await manager.close()

    assert manager.source_ids == ("test-source",)
    assert [item.outcome for item in received] == ["PASS"]
    assert source.stopped.is_set()
