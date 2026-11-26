# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable inference accounting, scoped to the caller rather than a shared client.

Attempt records contain metadata only, never prompts, responses, credentials, or
image bytes. Capture is synchronous bookkeeping; failures never become provider
failures. The owning engine admits records through its normal signal boundary.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterator, Literal, Mapping, Protocol, runtime_checkable

from leapflow.domain.events import RecordAdmission, RecordScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceAttempt:
    call_id: str
    attempt_id: str
    source: Literal["primary", "specialist"]
    provider: str
    model: str
    status: Literal["returned", "failed", "cancelled"]
    elapsed_ms: float
    usage_json: str = "{}"
    service_id: str = ""
    generation: int = 0
    declaration_digest: str = ""
    contract_revision: str = ""
    codec_revision: int = 0
    scope: RecordScope = RecordScope("", "", "")
    task_id: str = ""
    tool_call_id: str = ""
    error_type: str = ""
    accounting_level: Literal["provider_attempt", "call_boundary"] = "provider_attempt"

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["usage"] = json.loads(record.pop("usage_json"))
        record["usage_known"] = all(
            type(record["usage"].get(key)) is int
            for key in ("prompt_tokens", "completion_tokens")
        )
        return record


@dataclass(frozen=True)
class ConsultationRecord:
    """Detached advisory evidence; linkage never implies permission or success."""

    call_id: str
    scope: RecordScope
    task_id: str
    tool_call_id: str
    service_id: str
    status: Literal["eligible", "refused", "cancelled", "failed"]
    elapsed_ms: float
    failure_code: str = ""
    provenance_json: str = "{}"
    attempts: tuple[InferenceAttempt, ...] = ()
    result_digest: str = ""
    timing_json: str = "{}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id, "scope": asdict(self.scope),
            "task_id": self.task_id, "tool_call_id": self.tool_call_id,
            "service_id": self.service_id, "status": self.status,
            "failure_code": self.failure_code, "adoption": "unlinked",
            "elapsed_ms": self.elapsed_ms, "result_digest": self.result_digest,
            "timing": json.loads(self.timing_json),
            "provenance": json.loads(self.provenance_json),
            "inference_call_ids": list(dict.fromkeys(item.call_id for item in self.attempts)),
            "attempts": [item.to_dict() for item in self.attempts],
            "usage": usage_totals(list(self.attempts)),
        }


@dataclass(frozen=True)
class ConsultationReference:
    call_id: str
    disposition: Literal["adopted", "overridden", "unused"]


@dataclass(frozen=True)
class GovernedPlanRecord:
    """Explicit LLM attribution plus verified execution identity, never task success."""

    record_id: str
    scope: RecordScope
    task_id: str
    tool_call_id: str
    workcell: str
    skill_id: str
    plan_digest: str
    permit_digest: str
    references_json: str
    outcome_json: str
    origin_record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["references"] = json.loads(value.pop("references_json"))
        value["outcome"] = json.loads(value.pop("outcome_json"))
        value.update(attribution_source="llm_explicit_reference", plan_verified=True,
                     measured_task_outcome="unverified", physical_authority=False)
        return value


@runtime_checkable
class ConsultationLookup(Protocol):
    def consultation_reference(self, scope: RecordScope, task_id: str,
                               call_id: str) -> dict[str, Any] | None: ...


def record_governed_plan(*, workcell: str, skill_id: str, plan_digest: str,
                         permit_digest: str, references: tuple[ConsultationReference, ...],
                         result: Mapping[str, Any]) -> GovernedPlanRecord | None:
    """Called only after host plan verification; optional bookkeeping never executes."""
    capture = current_capture()
    if capture is None or not references:
        return
    try:
        sink = capture.usage_sink()
        linked = []
        for reference in references:
            consultation = (sink.consultation_reference(capture.scope, capture.task_id, reference.call_id)
                            if isinstance(sink, ConsultationLookup) else None)
            status = "linked" if consultation is not None else "unknown_or_foreign"
            if (consultation is not None and reference.disposition == "adopted"
                    and consultation.get("status") != "eligible"):
                status = "ineligible_advice"
            linked.append({**asdict(reference), "link_status": status,
                           "consultation_result_digest": consultation.get("result_digest", "") if consultation else "",
                           "consultation_tool_call_id": consultation.get("tool_call_id", "") if consultation else ""})
        record = GovernedPlanRecord(new_call_id(), capture.scope, capture.task_id, capture.tool_call_id,
            workcell, skill_id, plan_digest, permit_digest,
            json.dumps(linked, sort_keys=True, separators=(",", ":")), _execution_outcome(result))
        capture.record_plan(record)
        return record
    except Exception:  # noqa: BLE001 - lineage is not a physical or provider gate
        logger.warning("Governed plan lineage recording failed", exc_info=True)
    return None


def _execution_outcome(result: Mapping[str, Any]) -> str:
    # These are node observations, not a model-authored full-task verdict.
    outcome = {key: result[key] for key in (
        "run_id", "status", "failure_code", "side_effect_state", "dry_run", "timed_out",
        "started_at", "finished_at", "elapsed_s") if key in result}
    outcome["evidence_level"] = "node_execution"
    verification = result.get("operation_verification")
    if isinstance(verification, Mapping):
        outcome["operation_verification"] = verification
    evidence = result.get("evidence")
    if evidence:
        body = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
        outcome["evidence_digest"] = hashlib.sha256(body.encode()).hexdigest()
    return json.dumps(outcome, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_governed_outcome(plan: GovernedPlanRecord, result: Mapping[str, Any]) -> None:
    """Append a node-status observation to an immutable, exactly owned plan link."""
    capture = current_capture()
    if capture is None or capture.scope != plan.scope or not capture.task_id or capture.task_id != plan.task_id:
        return
    try:
        run_id = json.loads(plan.outcome_json).get("run_id")
        if (not run_id or result.get("run_id") != run_id or result.get("workcell") != plan.workcell
                or result.get("plan_digest") != plan.plan_digest):
            return
        capture.record_plan(replace(plan, record_id=new_call_id(),
            origin_record_id=plan.origin_record_id or plan.record_id,
            tool_call_id=capture.tool_call_id, outcome_json=_execution_outcome(result)))
    except Exception:  # noqa: BLE001 - status telemetry never changes run governance
        logger.warning("Governed outcome lineage recording failed", exc_info=True)


def admitted_record_value(receipt: RecordAdmission, scope: RecordScope, *,
                          event_type: str, identity: str = "") -> dict[str, Any] | None:
    """Validate detached private receipts before retaining or retrieving lineage."""
    if (not isinstance(receipt, RecordAdmission) or receipt.admitted is not True
            or receipt.scope != scope or receipt.event_type != event_type):
        return None
    try:
        payload = json.loads(receipt.payload_json)
        value = payload.get("value") if isinstance(payload, dict) else None
        if not isinstance(value, dict) or value.get("scope") != asdict(scope):
            return None
        key = {"model.inference_attempt": "attempt_id", "model.plan_link": "record_id"}.get(event_type, "call_id")
        if not isinstance(value.get(key), str) or not value[key] or identity and value[key] != identity:
            return None
        return value
    except (TypeError, ValueError):
        return None


@runtime_checkable
class InferenceUsageSink(Protocol):
    def record_inference(self, attempt: InferenceAttempt) -> None: ...


def freeze_usage(usage: Mapping[str, Any] | None) -> str:
    """Retain only finite, nonnegative numeric usage; absence remains unknown."""
    try:
        values = {key: value for key, value in (usage or {}).items()
                  if isinstance(key, str) and type(value) in (int, float) and value >= 0}
        return json.dumps(values, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001 - malformed telemetry cannot fail an inference
        logger.warning("Inference usage could not be retained", exc_info=True)
        return "{}"


@dataclass
class InferenceCapture:
    scope: RecordScope = RecordScope("", "", "")
    task_id: str = ""
    tool_call_id: str = ""
    sink: InferenceUsageSink | None = None
    parent: InferenceCapture | None = field(default=None, repr=False)
    attempts: list[InferenceAttempt] = field(default_factory=list)
    consultations: list[ConsultationRecord] = field(default_factory=list)
    plans: list[GovernedPlanRecord] = field(default_factory=list)
    phase_timings: dict[str, float] = field(default_factory=dict)
    provenance_json: str = "{}"
    provider_accounting: bool = True

    def usage_sink(self) -> InferenceUsageSink | None:
        return self.sink if self.sink is not None else self.parent.usage_sink() if self.parent else None

    def record_phase(self, name: str, elapsed_ms: float) -> None:
        self.phase_timings[name] = self.phase_timings.get(name, 0.0) + elapsed_ms
        if self.parent is not None:
            self.parent.record_phase(name, elapsed_ms)

    def record_consultation(self, record: ConsultationRecord) -> None:
        if record.scope != self.scope:
            raise ValueError("Consultation ownership mismatch")
        self.consultations.append(record)
        if self.parent is not None:
            self.parent.record_consultation(record)

    def record_plan(self, record: GovernedPlanRecord) -> None:
        if record.scope != self.scope:
            raise ValueError("Plan ownership mismatch")
        self.plans.append(record)
        if self.parent is not None:
            self.parent.record_plan(record)

    def record(self, attempt: InferenceAttempt) -> None:
        if attempt.scope != self.scope:
            raise ValueError("Inference ownership mismatch")
        self.attempts.append(attempt)
        if self.parent is not None:
            self.parent.record(attempt)
        elif self.sink is not None:
            self.sink.record_inference(attempt)


_capture: contextvars.ContextVar[InferenceCapture | None] = contextvars.ContextVar(
    "leapflow_inference_capture", default=None,
)


def current_capture() -> InferenceCapture | None:
    return _capture.get()


def record_phase(name: str, elapsed_ms: float) -> None:
    """Retain measured phase intervals; missing phases remain absent, not zero."""
    capture = current_capture()
    if capture is None:
        return
    try:
        capture.record_phase(name, elapsed_ms)
    except Exception:  # noqa: BLE001 - phase telemetry cannot replace execution failures
        logger.warning("Inference phase timing unavailable", exc_info=True)


@contextmanager
def measure_phase(name: str) -> Iterator[None]:
    """Measure one owned interval, including refusal, cancellation, and cleanup."""
    started = time.monotonic()
    try:
        yield
    finally:
        record_phase(name, (time.monotonic() - started) * 1000)


@contextmanager
def capture_inference(*, scope: RecordScope | None = None, task_id: str = "",
                      tool_call_id: str = "", sink: InferenceUsageSink | None = None,
                      provider_accounting: bool | None = None) -> Iterator[InferenceCapture]:
    parent = current_capture()
    inherited = parent if parent is not None and (scope is None or parent.scope == scope) else None
    capture = InferenceCapture(
        scope=scope or (inherited.scope if inherited else RecordScope("", "", "")),
        task_id=task_id or (inherited.task_id if inherited else ""),
        tool_call_id=tool_call_id or (inherited.tool_call_id if inherited else ""),
        sink=sink, parent=inherited if sink is None else None,
        provider_accounting=(provider_accounting if provider_accounting is not None
                             else inherited.provider_accounting if inherited else True),
    )
    with use_inference_capture(capture):
        yield capture


@contextmanager
def use_inference_capture(capture: InferenceCapture) -> Iterator[InferenceCapture]:
    """Install an owner for one await; never retain ContextVar tokens across yields."""
    token = _capture.set(capture)
    try:
        yield capture
    finally:
        _capture.reset(token)


class ProviderAttempt:
    """Metadata-only attempt recorder; ownership is captured before streaming yields."""

    def __init__(self, call_id: str, number: int, provider: str, model: str, *,
                 accounting_level: Literal["provider_attempt", "call_boundary"] = "provider_attempt") -> None:
        self.owner = current_capture()
        self.accounting_level = accounting_level
        self.call_id = call_id
        self.number = number
        self.provider = provider
        self.model = model
        self.usage_json = "{}"
        self.started = time.monotonic()

    def observe(self, usage: Mapping[str, Any] | None, model: str | None = None) -> None:
        """Preserve provider usage before payload extraction, including partial streams."""
        try:
            prior = json.loads(self.usage_json)
            prior.update(json.loads(freeze_usage(usage)))
            self.usage_json = freeze_usage(prior)
            if isinstance(model, str) and model:
                self.model = model
        except Exception:  # noqa: BLE001 - observations never fail the provider boundary
            logger.warning("Provider usage observation failed", exc_info=True)

    def response(self, response: Any) -> Any:
        try:
            self.observe(response.usage, response.model)
            response.inference_call_id = self.call_id
        except Exception:  # noqa: BLE001 - legacy response objects may reject metadata
            logger.warning("Provider response accounting failed", exc_info=True)
        return response

    def __enter__(self) -> ProviderAttempt:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        owner = self.owner
        if owner is None or not owner.provider_accounting:
            return
        try:
            status = "returned" if exc is None else (
                "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "failed")
            owner.record(InferenceAttempt(self.call_id, f"{self.call_id}:{self.number}", "primary",
                self.provider, self.model, status, (time.monotonic() - self.started) * 1000,
                usage_json=self.usage_json, scope=owner.scope, task_id=owner.task_id,
                tool_call_id=owner.tool_call_id, error_type=type(exc).__name__ if exc else "",
                accounting_level=self.accounting_level))
        except Exception:  # noqa: BLE001 - sink defects never enter provider retry handling
            logger.warning("Provider attempt recording failed", exc_info=True)


def new_call_id() -> str:
    return uuid.uuid4().hex


def record_attempt(attempt: InferenceAttempt) -> None:
    """Deliver metadata to the active owner without crossing the provider boundary."""
    capture = current_capture()
    if capture is None:
        return
    try:
        capture.record(attempt)
    except Exception:  # noqa: BLE001 - plugin telemetry must not affect inference
        logger.warning("Inference attempt recording failed", exc_info=True)


def usage_totals(attempts: list[InferenceAttempt]) -> dict[str, Any]:
    """Deduplicate attempts; report known subtotals separately from missing usage."""
    records = {attempt.attempt_id: attempt for attempt in attempts}
    totals: dict[str, Any] = {}
    for source in ("primary", "specialist", "combined"):
        selected = [record for record in records.values()
                    if source == "combined" or record.source == source]
        usages = [json.loads(record.usage_json) for record in selected]
        totals[source] = {
            "calls": len({record.call_id for record in selected}),
            "attempts": len(selected),  # Lower bound when opaque call boundaries are present.
            "provider_attempts": sum(record.accounting_level == "provider_attempt" for record in selected),
            "call_boundaries": sum(record.accounting_level == "call_boundary" for record in selected),
            "attempt_count_complete": all(record.accounting_level == "provider_attempt" for record in selected),
            "failed_attempts": sum(record.status != "returned" for record in selected),
            "unknown_usage_attempts": sum(not all(type(usage.get(key)) is int
                for key in ("prompt_tokens", "completion_tokens")) for usage in usages),
            "known_prompt_tokens": sum(usage.get("prompt_tokens", 0) +
                usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                for usage in usages),
            "known_completion_tokens": sum(usage.get("completion_tokens", 0) for usage in usages),
            "known_cached_tokens": sum(usage.get("cached_tokens", 0) for usage in usages),
            "elapsed_ms": sum(record.elapsed_ms for record in selected),
        }
    return totals
