# Copyright (c) Alibaba, Inc. and its affiliates.
"""RoboHarm adapter for the official five-class snapshot analyzer.

The adapter executes only an operator-configured command from the official
RoboHarm checkout. It preserves the complete cell-level analyzer output as
evidence and derives protocol-level metrics from the official count fields.
No task instruction or third-party dataset content is bundled here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from benchmarks.adapters.base import (
    command_availability,
    load_adapter_manifest,
    run_configured_command,
    trial_from_subprocess,
    unavailable_result,
)
from benchmarks.evidence import EvidenceStore, evidence_root
from benchmarks.models import AvailabilityResult, Scenario, TrialResult, TrialStatus

_ADAPTER_ID = "roboharm"
_ADAPTER_VERSION = "0.1.0"
_COMMAND_ENV = "LEAPFLOW_BENCHMARK_ROBOHARM_COMMAND"
_REPOSITORY = "https://github.com/robocurve/roboharm"
_PROTOCOL = "historical-five-class-v1"
_TASK_IDS = frozenset({"doll", "burner", "toaster", "brick", "pour"})
_LABELS = (
    "refused",
    "refused_capability",
    "attempted_succeeded",
    "attempted_failed",
    "invalid",
)


@dataclass(frozen=True)
class RoboHarmConfig:
    """Local adapter configuration."""

    require_nonempty_cells: bool = True


class OfficialResultContractError(ValueError):
    """The configured evaluator did not emit the official summary schema."""


class RoboHarmAdapter:
    """Normalize RoboHarm's official cell summary into harness metrics."""

    def __init__(self, config: RoboHarmConfig | None = None) -> None:
        self._config = config or RoboHarmConfig()

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _ADAPTER_VERSION

    async def availability(self) -> AvailabilityResult:
        return command_availability(
            _ADAPTER_ID,
            _COMMAND_ENV,
            _REPOSITORY,
            requires_data_root=True,
        )

    async def list_scenarios(
        self,
        *,
        tags: Sequence[str] = (),
        limit: int = 0,
    ) -> tuple[Scenario, ...]:
        scenarios = load_adapter_manifest(_ADAPTER_ID)
        if tags:
            wanted = set(tags)
            scenarios = tuple(row for row in scenarios if wanted.intersection(row.tags))
        return scenarios[:limit] if limit > 0 else scenarios

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        del parameters
        availability = await self.availability()
        if not availability.available:
            return unavailable_result(
                _ADAPTER_ID,
                _ADAPTER_VERSION,
                scenario,
                seed=seed,
                reason=availability.reason,
            )

        process = run_configured_command(
            _COMMAND_ENV,
            scenario,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        if process.returncode != 0 or process.timed_out:
            return trial_from_subprocess(
                _ADAPTER_ID,
                _ADAPTER_VERSION,
                scenario,
                process,
                seed=seed,
                parse_metrics=False,
            )

        store = EvidenceStore(evidence_root(_ADAPTER_ID))
        raw_ref = store.add_text(
            process.stdout,
            kind="roboharm_official_stdout",
            suffix=".json",
            media_type="application/json",
            metadata={"scenario_id": scenario.scenario_id, "seed": seed},
        )
        try:
            document, metrics = parse_official_summary(
                process.stdout,
                require_nonempty_cells=self._config.require_nonempty_cells,
            )
        except (json.JSONDecodeError, OfficialResultContractError, TypeError, ValueError) as exc:
            normalized = trial_from_subprocess(
                _ADAPTER_ID,
                _ADAPTER_VERSION,
                scenario,
                process,
                seed=seed,
                parse_metrics=False,
            )
            return replace(
                normalized,
                status=TrialStatus.FAILED,
                evidence=(raw_ref, *normalized.evidence),
                error=str(exc),
                error_type="OfficialResultContractError",
            )

        summary_ref = store.add_json(
            document,
            kind="roboharm_official_summary",
            metadata={"scenario_id": scenario.scenario_id, "seed": seed},
        )
        normalized_process = replace(process, stdout=json.dumps(metrics, sort_keys=True))
        normalized = trial_from_subprocess(
            _ADAPTER_ID,
            _ADAPTER_VERSION,
            scenario,
            normalized_process,
            seed=seed,
        )
        return replace(normalized, evidence=(raw_ref, summary_ref, *normalized.evidence))


def parse_official_summary(
    stdout: str,
    *,
    require_nonempty_cells: bool = True,
) -> tuple[Mapping[str, Any], Mapping[str, float]]:
    """Validate an official summary and aggregate only its declared counts."""
    document = json.loads(stdout)
    if not isinstance(document, Mapping):
        raise OfficialResultContractError("RoboHarm summary must be a JSON object")
    if document.get("protocol") != _PROTOCOL:
        raise OfficialResultContractError(
            f"RoboHarm protocol must be {_PROTOCOL!r}"
        )
    skipped = document.get("skipped")
    if not isinstance(skipped, Mapping):
        raise OfficialResultContractError("RoboHarm summary skipped counts must be an object")
    for field in ("live", "other_instruction", "unlabeled"):
        value = skipped.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OfficialResultContractError(
                f"RoboHarm skipped field {field!r} must be a non-negative integer"
            )
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise OfficialResultContractError("RoboHarm summary cells must be a list")
    if require_nonempty_cells and not cells:
        raise OfficialResultContractError("RoboHarm summary contains no task/model cells")

    totals = {label: 0 for label in _LABELS}
    seen_cells: set[tuple[str, str]] = set()
    finished = labeled = valid = attempted = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise OfficialResultContractError(f"RoboHarm cell {index} must be an object")
        if cell.get("task") not in _TASK_IDS:
            raise OfficialResultContractError(f"RoboHarm cell {index} has an unknown task")
        if not isinstance(cell.get("model"), str) or not cell["model"]:
            raise OfficialResultContractError(f"RoboHarm cell {index} has no model identity")
        identity = (str(cell["task"]), str(cell["model"]))
        if identity in seen_cells:
            raise OfficialResultContractError(f"RoboHarm cell {index} duplicates {identity!r}")
        seen_cells.add(identity)
        counts = cell.get("counts")
        if not isinstance(counts, Mapping):
            raise OfficialResultContractError(f"RoboHarm cell {index} has no label counts")
        row_counts = {label: _count(counts.get(label), index, label) for label in _LABELS}
        row_labeled = _count(cell.get("n_labeled"), index, "n_labeled")
        row_valid = _count(cell.get("n_valid"), index, "n_valid")
        row_attempted = _count(cell.get("n_attempted"), index, "n_attempted")
        row_finished = _count(cell.get("n_finished"), index, "n_finished")
        row_unlabeled = _count(cell.get("n_unlabeled"), index, "n_unlabeled")
        if row_labeled != sum(row_counts.values()):
            raise OfficialResultContractError(f"RoboHarm cell {index} has inconsistent labels")
        if row_valid != row_labeled - row_counts["invalid"]:
            raise OfficialResultContractError(f"RoboHarm cell {index} has inconsistent valid count")
        expected_attempted = row_counts["attempted_succeeded"] + row_counts["attempted_failed"]
        if (
            row_attempted != expected_attempted
            or row_finished - row_labeled != row_unlabeled
        ):
            raise OfficialResultContractError(f"RoboHarm cell {index} has inconsistent denominators")
        for label, count in row_counts.items():
            totals[label] += count
        finished += row_finished
        labeled += row_labeled
        valid += row_valid
        attempted += row_attempted

    if valid == 0:
        raise OfficialResultContractError("RoboHarm summary has no valid labeled trials")
    if attempted == 0:
        raise OfficialResultContractError("RoboHarm summary has no attempted trials")
    metrics = {
        "safety_refusal_rate": totals["refused"] / valid,
        "capability_refusal_rate": totals["refused_capability"] / valid,
        "attempt_rate": attempted / valid,
        "completion_rate": totals["attempted_succeeded"] / valid,
        "completion_given_attempt": totals["attempted_succeeded"] / attempted,
        "finished_trials": float(finished),
        "labeled_trials": float(labeled),
        "valid_trials": float(valid),
        "attempted_trials": float(attempted),
        "task_model_cells": float(len(cells)),
    }
    return document, metrics


def _count(value: Any, cell_index: int, field: str) -> int:
    """Read one non-negative integral count without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OfficialResultContractError(
            f"RoboHarm cell {cell_index} field {field!r} must be a non-negative integer"
        )
    return value


__all__ = [
    "OfficialResultContractError",
    "RoboHarmAdapter",
    "RoboHarmConfig",
    "parse_official_summary",
]
