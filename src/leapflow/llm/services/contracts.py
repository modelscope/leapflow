# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed task admission; registration belongs to the existing composition scope."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from leapflow.llm.services.codec import validate_schema, validate_schema_definition
from leapflow.llm.services.protocol import canonical


@runtime_checkable
class TaskContract(Protocol):
    @property
    def task(self) -> str: ...
    @property
    def revision(self) -> str: ...
    @property
    def classification(self) -> Literal["read_only", "governed_actuation"]: ...
    @property
    def description(self) -> str: ...
    def schemas(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...
    def validate_input(self, request: Mapping[str, Any]) -> None: ...
    def validate_result(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> None: ...
    def evidence_references(self, request: Mapping[str, Any]) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class RegisteredContract:
    """Detached schema/metadata plus identity of the admitted plugin implementation."""
    contract: TaskContract
    number: int
    task: str
    revision: str
    classification: Literal["read_only", "governed_actuation"]
    description: str
    input_json: str
    output_json: str

    def schemas(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return json.loads(self.input_json), json.loads(self.output_json)

    def validate_input(self, request: Mapping[str, Any]) -> None:
        validate_schema(request, json.loads(self.input_json), "input")
        self.contract.validate_input(request)

    def validate_result(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        validate_schema(result, json.loads(self.output_json))
        self.contract.validate_result(request, result)

    def evidence_references(self, request: Mapping[str, Any]) -> dict[str, str]:
        refs = dict(self.contract.evidence_references(request))
        if any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in refs.items()):
            raise ValueError("Contract evidence references must map labels to owned cache IDs")
        return refs


class TaskContractRegistry:
    """Small owner-injected registry; no discovery, I/O, or independent lifecycle."""
    def __init__(self) -> None:
        self._entries: dict[str, RegisteredContract] = {}
        self.version = 0

    def get(self, task: str) -> RegisteredContract | None:
        return self._entries.get(task)

    def register(self, contract: TaskContract) -> Callable[[], None]:
        """Return undo to register on the plugin's EffectScope."""
        if not isinstance(contract, TaskContract):
            raise ValueError("Task registration requires a TaskContract")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", contract.task) or contract.task in self._entries:
            raise ValueError("Task registration requires a unique bounded identifier")
        if (contract.classification not in {"read_only", "governed_actuation"}
                or not isinstance(contract.revision, str) or not contract.revision
                or not isinstance(contract.description, str) or not contract.description):
            raise ValueError("Task registration requires explicit classification, revision, and description")
        input_schema, output_schema = contract.schemas()
        validate_schema_definition(dict(input_schema))
        validate_schema_definition(dict(output_schema))
        if input_schema.get("type") != "object" or output_schema.get("type") != "object":
            raise ValueError("Task contracts require object input and result schemas")
        self.version += 1
        entry = RegisteredContract(contract, self.version, contract.task, contract.revision,
                                   contract.classification, contract.description,
                                   canonical(input_schema), canonical(output_schema))
        self._entries[contract.task] = entry

        def undo() -> None:
            if self._entries.get(entry.task) is entry:
                del self._entries[entry.task]
                self.version += 1

        return undo
