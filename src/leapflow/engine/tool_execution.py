"""Tool execution identity, policy, and idempotency ledger."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

ExecutionPolicy = Literal["read_only", "mutating_idempotent", "mutating_once", "external_side_effect"]
ExecutionStatus = Literal["reserved", "running", "completed", "failed_retryable", "failed_final"]

_EXTERNAL_TOOLS = frozenset({
    "shell_run",
    "scm_sync",
    "gateway_send",
    "gateway_connect",
    "platform_action",
    "platform_connect",
    "hub_push",
    "hub_pull",
    "hub_sync",
})


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for execution identity keys."""
    return json.dumps(value or {}, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def execution_policy_for(tool_name: str, spec: Any | None = None) -> ExecutionPolicy:
    """Classify a tool into an idempotency policy using registry metadata."""
    name = str(tool_name or "").removeprefix("gp_")
    risk_level = str(getattr(spec, "risk_level", "") or "")
    mutates_state = bool(getattr(spec, "mutates_state", False))
    idempotency_scope = str(getattr(spec, "idempotency_scope", "") or "")
    effect_scope = str(getattr(spec, "effect_scope", "") or "")
    if risk_level == "read_only" and not mutates_state:
        return "read_only"
    if name in _EXTERNAL_TOOLS or risk_level == "external" or effect_scope == "external":
        return "external_side_effect"
    if idempotency_scope == "session":
        return "mutating_once"
    return "mutating_idempotent"


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
