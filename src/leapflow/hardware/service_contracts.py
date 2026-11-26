# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hardware-owned query contracts; native tensor/identity adapters remain qualified."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from leapflow.hardware.model_contract import Detection, ProposedGrasp, parse_response, provider_result_schema


def text(maximum: int = 8192) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def array(items: Mapping[str, Any], *, minimum: int = 0, maximum: int = 64) -> dict[str, Any]:
    return {"type": "array", "items": dict(items), "minItems": minimum, "maxItems": maximum}


def obj(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


GEOMETRY_INPUTS = {
    "grounding": obj({"queries": array(text(256), minimum=1, maximum=16)}, ("queries",)),
    "grasp_proposal": obj({"depth": obj({"scale_m": {"type": "number", "minimum": 0.000000001}},
                                        ("scale_m",)),
                           "intrinsics": obj({
                               "width": {"type": "integer", "minimum": 1},
                               "height": {"type": "integer", "minimum": 1},
                               "fx": {"type": "number", "minimum": 0.000000001},
                               "fy": {"type": "number", "minimum": 0.000000001},
                               "ppx": {"type": "number"}, "ppy": {"type": "number"},
                               "model": text(128), "coeffs": array({"type": "number"}, maximum=5),
                           }, ("width", "height", "fx", "fy", "ppx", "ppy")),
                           "object": obj({"label": text(256),
                               "center": array({"type": "number"}, minimum=3, maximum=3),
                               "pixel": array({"type": "number", "minimum": 0}, minimum=2, maximum=2),
                               "size": obj({key: {"type": "number", "minimum": 0} for key in
                                            ("length", "width", "height")}, ("length", "width", "height")),
                           }, ("center",)),
                           "gripper": obj({"max_width_mm": {"type": "number", "minimum": 0.000000001}},
                                          ("max_width_mm",)),
                           "top_k": {"type": "integer", "minimum": 1, "maximum": 64}},
                          ("depth", "intrinsics", "object", "gripper", "top_k")),
}
GEOMETRY_OUTPUTS = {
    "grounding": obj({"detections": array({"type": "object"}, maximum=256)}, ("detections",)),
    "grasp_proposal": obj({"grasps": array({"type": "object"}, maximum=64)}, ("grasps",)),
}
_ACTION_INPUT = obj({"state": array({"type": "number"}, minimum=1, maximum=65536),
                     "task": text(), "pending": {"type": "object"}}, ("state", "task"))


@dataclass(frozen=True)
class HardwareTaskContract:
    task: str
    revision: str = "hardware.model.v1.1"

    @property
    def classification(self) -> Literal["read_only", "governed_actuation"]:
        return "governed_actuation" if self.task == "action_chunk" else "read_only"

    @property
    def description(self) -> str:
        return {"grounding": "Image geometry candidates; metric lifting requires calibrated observations.",
                "grasp_proposal": "Grasp proposals, not permission to move; depth and embodiment qualification required.",
                "action_chunk": "Native action tensors; only qualified adapters and governed policy execution may consume them."}[self.task]

    def schemas(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if self.task == "action_chunk":
            return _ACTION_INPUT, provider_result_schema(self.task)
        return GEOMETRY_INPUTS[self.task], GEOMETRY_OUTPUTS[self.task]

    def validate_input(self, request: Mapping[str, Any]) -> None:
        # Embedded media/tensor validation belongs to the qualified adapter, after consent.
        schema, _ = self.schemas()
        from leapflow.llm.services.codec import validate_schema
        validate_schema(request, schema, "input")

    def validate_result(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        if self.task == "action_chunk":
            parse_response(self.task, result)
            return
        parser, field = (Detection, "detections") if self.task == "grounding" else (ProposedGrasp, "grasps")
        for item in result[field]:
            parser.from_mapping(item)

    def evidence_references(self, request: Mapping[str, Any]) -> Mapping[str, str]:
        # Hardware query artifacts are explicitly supplied, never inferred from model text.
        return {}
