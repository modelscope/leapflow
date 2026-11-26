# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded advisory contracts; model judgments never represent measured outcomes."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, Mapping


def text(maximum: int = 8192) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def array(items: Mapping[str, Any], *, minimum: int = 0, maximum: int = 64) -> dict[str, Any]:
    return {"type": "array", "items": dict(items), "minItems": minimum, "maxItems": maximum}


def obj(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


SCORE = {"type": "number"}
UNIT = {"type": "number", "minimum": 0, "maximum": 1}
CANDIDATE = obj({"id": text(128), "description": text()}, ("id", "description"))
# Structured task state (poses, gripper, progress) stays typed instead of being
# flattened into prose; bounded here because JSON schema cannot bound open objects.
STATE_MAX_BYTES = 8192
STATE_MAX_DEPTH = 4
INPUT_SCHEMAS = {
    "decision": obj({"goal": text(), "observations": array(text()), "state": {"type": "object"},
                     "candidates": array(CANDIDATE, minimum=1)}, ("goal", "candidates")),
    "prediction": obj({"observation_summary": text(), "observation_refs": array(text(128)),
                       "candidates": array(CANDIDATE, minimum=1),
                       "horizon_s": {"type": "number", "minimum": 0.000001, "maximum": 86400}},
                      ("observation_summary", "candidates", "horizon_s")),
    "evaluation": obj({"episode_ref": text(128), "outcome_summary": text(), "rubric": text(),
                       "evidence": array(CANDIDATE)},
                      ("episode_ref", "outcome_summary", "rubric", "evidence")),
}
OUTPUT_SCHEMAS = {
    # Rationale is optional: classifier heads score candidates but cannot explain them,
    # and an adapter-written explanation would be indistinguishable from model reasoning.
    "decision": obj({"selected_id": text(128), "ranking": array(text(128)),
                     "candidate_scores": array(obj({"id": text(128), "score": UNIT}, ("id", "score"))),
                     "rationale": text(), "model_score": SCORE}, ("selected_id",)),
    "prediction": obj({"predictions": array(obj({
        "candidate_id": text(128), "outcome_summary": text(),
        "horizon_s": {"type": "number", "minimum": 0.000001, "maximum": 86400},
        "predicted_state": {"type": "object"}, "model_score": SCORE,
    }, ("candidate_id", "outcome_summary", "horizon_s")), minimum=1)}, ("predictions",)),
    "evaluation": obj({"assessment": {"type": "string", "enum": ["success", "failure", "uncertain"]},
                       "rationale": text(), "model_score": SCORE,
                       "evidence_ids": array(text(128))}, ("assessment", "rationale")),
}
REVISIONS = {"decision": "advisory.v2", "prediction": "advisory.v1", "evaluation": "advisory.v1"}


@dataclass(frozen=True)
class AdvisoryDecision:
    selected_id: str
    ranking: tuple[str, ...]
    rationale: str = ""
    model_score: float | None = None
    candidate_scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class HypotheticalPrediction:
    candidate_id: str
    outcome_summary: str
    horizon_s: float
    model_score: float | None = None


@dataclass(frozen=True)
class TeacherAssessment:
    assessment: str
    rationale: str
    evidence_ids: tuple[str, ...]
    model_score: float | None = None


@dataclass(frozen=True)
class AdvisoryTaskContract:
    """Domain-owned registrations; descriptions do not assert model qualification."""
    task: str
    classification: Literal["read_only", "governed_actuation"] = "read_only"

    @property
    def revision(self) -> str:
        return REVISIONS[self.task]

    @property
    def description(self) -> str:
        return {"decision": "Advisory choice among caller-eligible alternatives, optionally with per-candidate "
                             "scores; no physical authority.",
                "prediction": "Hypothetical outcomes, not verified observations or measured success.",
                "evaluation": "Teacher assessment, not independently measured task success."}[self.task]

    def schemas(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return INPUT_SCHEMAS[self.task], OUTPUT_SCHEMAS[self.task]

    def validate_input(self, request: Mapping[str, Any]) -> None:
        eligible_ids(request, self.task)
        if "state" in request:
            _bounded_state(request["state"])

    def validate_result(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        validate_result(self.task, request, result)

    def evidence_references(self, request: Mapping[str, Any]) -> Mapping[str, str]:
        references = {ref: ref for ref in request.get("observation_refs", [])}
        if self.task == "evaluation":
            references["episode"] = request["episode_ref"]
        return references


def eligible_ids(request: Mapping[str, Any], task: str) -> set[str]:
    """Caller-admitted identity set; no model text can resolve a new candidate."""
    entries = request.get("evidence" if task == "evaluation" else "candidates", [])
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eligible identifiers")
    refs = request.get("observation_refs", [])
    if len(refs) != len(set(refs)):
        raise ValueError("duplicate observation references")
    return set(ids)


def _bounded_state(state: Any) -> None:
    def depth(value: Any, level: int) -> None:
        if level > STATE_MAX_DEPTH:
            raise ValueError(f"state nesting exceeds {STATE_MAX_DEPTH} levels")
        for item in value.values() if isinstance(value, dict) else value if isinstance(value, list) else ():
            depth(item, level + 1)

    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    depth(state, 1)
    if len(json.dumps(state, separators=(",", ":"), allow_nan=False).encode()) > STATE_MAX_BYTES:
        raise ValueError(f"state exceeds {STATE_MAX_BYTES} bytes; summarize it")


def validate_result(task: str, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Check relations that JSON schema cannot express, after structural validation."""
    ids = eligible_ids(request, task)
    if task == "decision":
        ranking = result.get("ranking", [])
        scored = [entry["id"] for entry in result.get("candidate_scores", [])]
        if (result["selected_id"] not in ids or set(ranking) - ids or len(ranking) != len(set(ranking))
                or set(scored) - ids or len(scored) != len(set(scored))):
            raise ValueError("decision contains an unknown or duplicate eligible identifier")
    elif task == "prediction":
        predictions = result["predictions"]
        returned = [entry["candidate_id"] for entry in predictions]
        if set(returned) != ids or len(returned) != len(ids):
            raise ValueError("prediction must contain exactly one result per candidate")
        if any(entry["horizon_s"] != request["horizon_s"] for entry in predictions):
            raise ValueError("prediction horizon differs from the requested horizon")
    elif task == "evaluation":
        references = result.get("evidence_ids", [])
        if set(references) - ids or len(references) != len(set(references)):
            raise ValueError("evaluation contains unknown or duplicate evidence identifiers")
    else:
        raise ValueError("unknown advisory task")
