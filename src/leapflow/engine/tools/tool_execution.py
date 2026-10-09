# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool execution identity, policy, and idempotency ledger."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

ExecutionPolicy = Literal["read_only", "mutating_idempotent", "mutating_once", "external_side_effect"]
ExecutionStatus = Literal["reserved", "running", "completed", "failed_retryable", "failed_final"]

# Policies whose failure leaves the effect's fate unknown: the call may already
# have landed (a delivered message, a committed write) even though it reported an
# error, so a blind retry can duplicate it. ``mutating_idempotent`` is absent on
# purpose — re-applying it converges, so flagging it would stall safe retries.
UNCERTAIN_EFFECT_POLICIES: frozenset[str] = frozenset({"external_side_effect", "mutating_once"})


def effect_is_uncertain_on_failure(policy: str) -> bool:
    """Return whether a failed call under ``policy`` may still have taken effect."""
    return str(policy or "") in UNCERTAIN_EFFECT_POLICIES


def exit_code_from(result: Any) -> int | None:
    """Return a process exit code from a tool result under either key name.

    Shell-shaped tools mirror ``subprocess``' own ``returncode`` attribute, while
    the model-facing evidence and the TUI read ``exit_code``. Reading only one
    name silently dropped the code from both surfaces, so the mapping lives here
    once instead of being re-guessed per consumer.
    """
    if not isinstance(result, Mapping):
        return None
    for key in ("exit_code", "returncode"):
        value = result.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None

EXECUTION_POLICIES: frozenset[str] = frozenset(
    {"read_only", "mutating_idempotent", "mutating_once", "external_side_effect"}
)


def normalize_execution_policy(
    value: Any,
    *,
    default: ExecutionPolicy = "external_side_effect",
) -> ExecutionPolicy:
    """Validate a declared policy, using a conservative fallback when absent."""
    candidate = str(value or "")
    if candidate in EXECUTION_POLICIES:
        return cast(ExecutionPolicy, candidate)
    return default


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for execution identity keys."""
    return json.dumps(value or {}, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def execution_policy_for(tool_name: str, spec: Any | None = None) -> ExecutionPolicy:
    """Resolve execution policy exclusively from declared registry metadata.

    ``tool_name`` remains part of the API for diagnostics, but never influences
    classification. Missing or contradictory metadata fails safe as an external
    side effect instead of guessing from a vendor or verb embedded in the name.
    """
    del tool_name
    if spec is None:
        return "external_side_effect"
    declared = str(getattr(spec, "execution_policy", "") or "")
    if declared in EXECUTION_POLICIES:
        return cast(ExecutionPolicy, declared)
    risk_level = str(getattr(spec, "risk_level", "") or "")
    mutates_state = bool(getattr(spec, "mutates_state", False))
    idempotency_scope = str(getattr(spec, "idempotency_scope", "") or "")
    effect_scope = str(getattr(spec, "effect_scope", "") or "")
    if risk_level == "read_only" and not mutates_state and effect_scope != "external":
        return "read_only"
    if effect_scope == "external" or risk_level == "external":
        return "external_side_effect"
    if idempotency_scope == "session":
        return "mutating_once"
    if mutates_state:
        return "external_side_effect"
    return "external_side_effect"


def build_idempotency_key(
    *,
    session_id: str,
    turn_id: str,
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    policy: ExecutionPolicy,
) -> str:
    """Build a stable system-owned idempotency key for a tool execution."""
    scope = "turn" if policy in {"read_only", "mutating_idempotent"} else "session"
    payload = {
        "session_id": session_id,
        "turn_id": turn_id if scope == "turn" else "",
        "scope": scope,
        "tool_name": str(tool_name or "").removeprefix("gp_"),
        "arguments": arguments or {},
        "policy": policy,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolExecutionRecord:
    """Immutable execution ledger row."""

    execution_id: str
    session_id: str
    turn_id: str
    command_id: str
    tool_call_id: str
    tool_name: str
    idempotency_key: str
    arguments: dict[str, Any]
    policy: ExecutionPolicy
    status: ExecutionStatus
    result: Any = None
    created_at: float = 0.0
    completed_at: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        turn_id: str,
        command_id: str,
        tool_call_id: str,
        tool_name: str,
        idempotency_key: str,
        arguments: Mapping[str, Any] | None,
        policy: ExecutionPolicy,
    ) -> "ToolExecutionRecord":
        return cls(
            execution_id=uuid.uuid4().hex,
            session_id=session_id,
            turn_id=turn_id,
            command_id=command_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            idempotency_key=idempotency_key,
            arguments=dict(arguments or {}),
            policy=policy,
            status="running",
            created_at=time.time(),
        )

    def mark_completed(self, result: Any) -> "ToolExecutionRecord":
        return replace(self, status="completed", result=result, completed_at=time.time())

    def mark_failed(self, result: Any, *, retryable: bool) -> "ToolExecutionRecord":
        return replace(
            self,
            status="failed_retryable" if retryable else "failed_final",
            result=result,
            completed_at=time.time(),
        )


@dataclass(frozen=True)
class TaskCompletionEvidence:
    """Immutable, per-step proof that a requested artifact or effect completed."""

    step_id: str
    artifact_uri: str
    artifact_hash: str
    effect_confirmed: bool
    opened_paths: tuple[str, ...]
    source: str
    observed_at: float

    def to_metadata(self) -> dict[str, Any]:
        """Return the safe structured form carried in tool/stream metadata."""
        return {
            "step_id": self.step_id,
            "artifact_uri": self.artifact_uri,
            "artifact_hash": self.artifact_hash,
            "effect_confirmed": self.effect_confirmed,
            "opened_paths": list(self.opened_paths),
            "source": self.source,
            "observed_at": self.observed_at,
        }


def _canonical_artifact_path(value: Any) -> str:
    """Normalize a local artifact path without treating URLs as filesystem paths."""
    text = str(value or "").strip()
    if not text or "://" in text and not text.startswith("file://"):
        return ""
    if text.startswith("file://"):
        text = text[7:]
    try:
        return str(Path(text).expanduser().resolve())
    except (OSError, ValueError):
        return ""


def _artifact_digest(path: str) -> str:
    """Hash an existing artifact so equal content is idempotent within a step."""
    if not path:
        return ""
    try:
        candidate = Path(path)
        if not candidate.is_file():
            return ""
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65_536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _result_artifact_paths(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract local artifact paths from the common result contract fields."""
    candidates: list[Any] = [
        result.get("artifact_uri"),
        result.get("path"),
        result.get("file_path"),
        result.get("screenshot_file_path"),
    ]
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, Mapping):
                candidates.append(artifact.get("uri") or artifact.get("path"))
            else:
                candidates.append(artifact)
    paths = [_canonical_artifact_path(candidate) for candidate in candidates]
    return tuple(dict.fromkeys(path for path in paths if path))


def _opened_paths(tool_name: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract explicit or shell-requested local file opens from a successful call."""
    candidates: list[Any] = []
    for key in ("opened_path", "opened_file", "path_opened"):
        candidates.append(result.get(key))
    raw_paths = result.get("opened_paths")
    if isinstance(raw_paths, (list, tuple)):
        candidates.extend(raw_paths)
    if bool(result.get("opened")):
        candidates.extend((result.get("path"), result.get("file_path")))
    lowered_name = str(tool_name or "").lower()
    if lowered_name.startswith("open"):
        candidates.extend((arguments.get("path"), arguments.get("file_path")))
    if lowered_name == "shell_run":
        command = str(arguments.get("command") or "")
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if len(tokens) >= 2 and tokens[0].lower() in {"open", "xdg-open", "start"}:
            candidates.append(tokens[-1])
    paths = [_canonical_artifact_path(candidate) for candidate in candidates]
    return tuple(dict.fromkeys(path for path in paths if path))


class TaskCompletionTracker:
    """Per-frame evidence ledger that turns completed screenshot work into a finalization signal."""

    def __init__(self) -> None:
        self._evidence: dict[tuple[str, str, str], TaskCompletionEvidence] = {}

    @property
    def evidence(self) -> tuple[TaskCompletionEvidence, ...]:
        return tuple(self._evidence.values())

    def record(
        self,
        *,
        step_id: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        result: Mapping[str, Any],
    ) -> tuple[TaskCompletionEvidence, ...]:
        """Record success evidence and return only records newly added this call."""
        if result.get("ok") is not True or result.get("duplicate_suppressed"):
            return ()
        args = arguments or {}
        source = "screenshot" if (
            tool_name == "screenshot"
            or bool(result.get("captured"))
            or bool(result.get("screenshot_file_path"))
        ) else "tool_result"
        effect_confirmed = bool(result.get("effect_confirmed", result.get("completed", True)))
        opened = _opened_paths(tool_name, args, result)
        added: list[TaskCompletionEvidence] = []
        for artifact_path in _result_artifact_paths(result):
            artifact_hash = _artifact_digest(artifact_path)
            # Screenshot completion needs a real file, not only the driver saying
            # where it intended to write one.
            if source == "screenshot" and not artifact_hash:
                continue
            identity = artifact_hash or artifact_path
            key = (step_id, source, identity)
            if key in self._evidence:
                continue
            evidence = TaskCompletionEvidence(
                step_id=step_id,
                artifact_uri=artifact_path,
                artifact_hash=artifact_hash,
                effect_confirmed=effect_confirmed,
                opened_paths=opened,
                source=source,
                observed_at=time.time(),
            )
            self._evidence[key] = evidence
            added.append(evidence)
        if opened:
            key = (step_id, "open_request", "|".join(opened))
            if key not in self._evidence:
                evidence = TaskCompletionEvidence(
                    step_id=step_id,
                    artifact_uri="",
                    artifact_hash="",
                    effect_confirmed=effect_confirmed,
                    opened_paths=opened,
                    source="open_request",
                    observed_at=time.time(),
                )
                self._evidence[key] = evidence
                added.append(evidence)
        return tuple(added)

    def ready_for_final_response(self) -> bool:
        """Return true once two real screenshots have each been requested open."""
        screenshots: dict[str, TaskCompletionEvidence] = {}
        opened: set[str] = set()
        for evidence in self._evidence.values():
            opened.update(evidence.opened_paths)
            if evidence.source == "screenshot" and evidence.effect_confirmed and evidence.artifact_uri:
                identity = evidence.artifact_hash or evidence.artifact_uri
                screenshots[identity] = evidence
        if len(screenshots) < 2:
            return False
        screenshot_paths = {evidence.artifact_uri for evidence in screenshots.values()}
        return screenshot_paths.issubset(opened)


class ToolExecutionLedger:
    """Turn-scope idempotency ledger with optional durable backing store."""

    def __init__(self, *, store: Any | None = None) -> None:
        self._records: dict[str, ToolExecutionRecord] = {}
        self._inflight: dict[str, asyncio.Future[ToolExecutionRecord]] = {}
        self._store = store

    def reset(self, *, store: Any | None = None) -> None:
        self._records.clear()
        self._inflight.clear()
        self._store = store

    def reserve(
        self,
        *,
        session_id: str,
        turn_id: str,
        command_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        policy: ExecutionPolicy,
    ) -> tuple[ToolExecutionRecord, ToolExecutionRecord | None]:
        """Reserve an execution or return the original record for duplicates."""
        key = build_idempotency_key(
            session_id=session_id,
            turn_id=turn_id,
            tool_name=tool_name,
            arguments=arguments,
            policy=policy,
        )
        if policy != "read_only":
            existing = self._records.get(key) or self._get_durable(session_id, key)
            if existing is not None:
                return existing, existing
        record = ToolExecutionRecord.create(
            session_id=session_id,
            turn_id=turn_id,
            command_id=command_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            idempotency_key=key,
            arguments=arguments,
            policy=policy,
        )
        self._records[key] = record
        if policy != "read_only":
            self._inflight[key] = asyncio.get_running_loop().create_future()
        self._reserve_durable(record)
        return record, None

    async def wait_for_completion(
        self,
        record: ToolExecutionRecord,
        *,
        timeout_s: float,
    ) -> ToolExecutionRecord:
        """Wait for an in-flight duplicate's original execution to finish."""
        future = self._inflight.get(record.idempotency_key)
        if future is None:
            return self._records.get(record.idempotency_key, record)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=max(1.0, timeout_s))
        except TimeoutError:
            current = self._records.get(record.idempotency_key, record)
            return current.mark_failed(
                {
                    "ok": False,
                    "error": "Original tool execution is still running.",
                    "retryable": True,
                    "already_executed": True,
                    "duplicate_suppressed": True,
                    "counts_as_failure": False,
                    "counts_as_tool_attempt": False,
                    "ui_hidden": True,
                },
                retryable=True,
            )

    def complete(self, record: ToolExecutionRecord, result: Any) -> ToolExecutionRecord:
        retryable = bool(result.get("retryable")) if isinstance(result, dict) else False
        ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
        updated = record.mark_completed(result) if ok else record.mark_failed(result, retryable=retryable)
        self._records[record.idempotency_key] = updated
        future = self._inflight.pop(record.idempotency_key, None)
        if future is not None and not future.done():
            future.set_result(updated)
        self._complete_durable(updated)
        return updated

    @staticmethod
    def duplicate_result(record: ToolExecutionRecord) -> dict[str, Any]:
        """Return a structured payload for a suppressed duplicate side effect."""
        completed = record.status == "completed"
        payload: dict[str, Any] = {
            "ok": completed,
            "already_executed": True,
            "duplicate_suppressed": True,
            "execution_reused": completed,
            "execution_skipped": not completed,
            "counts_as_failure": False,
            "counts_as_tool_attempt": False,
            "ui_hidden": True,
            "execution_id": record.execution_id,
            "idempotency_key": record.idempotency_key,
            "execution_status": record.status,
            "original_result": record.result,
            "retryable": record.status == "failed_retryable",
        }
        if not completed:
            payload["error"] = "An identical side-effect attempt is already recorded. Review the original result before retrying."
        # Preserve the original attempt's uncertainty verdict: if that failure may
        # already have taken effect, the suppressed duplicate must say so too, or
        # the model loses exactly the warning that told it to verify first.
        original = record.result if isinstance(record.result, Mapping) else {}
        if original.get("side_effect_uncertain"):
            payload["side_effect_uncertain"] = True
            if original.get("retry_guidance"):
                payload["retry_guidance"] = original["retry_guidance"]
        return payload

    def _get_durable(self, session_id: str, key: str) -> ToolExecutionRecord | None:
        if self._store is None or not hasattr(self._store, "get_tool_execution_by_key"):
            return None
        try:
            return self._store.get_tool_execution_by_key(session_id, key)
        except Exception:
            return None

    def _reserve_durable(self, record: ToolExecutionRecord) -> None:
        if self._store is None or not hasattr(self._store, "reserve_tool_execution"):
            return
        try:
            self._store.reserve_tool_execution(record)
        except Exception:
            pass

    def _complete_durable(self, record: ToolExecutionRecord) -> None:
        if self._store is None or not hasattr(self._store, "complete_tool_execution"):
            return
        try:
            self._store.complete_tool_execution(record)
        except Exception:
            pass
