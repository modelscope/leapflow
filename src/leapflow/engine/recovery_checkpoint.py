"""Recovery checkpoint system for cross-turn state persistence and safe resumption.

Enables the agent loop to save execution state when halting with HALT_WITH_CHECKPOINT,
and resume safely in a subsequent turn after user action or external fix.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class CheckpointState(Enum):
    """Lifecycle states of a recovery checkpoint."""

    PENDING = "pending"       # Created, waiting for resume
    CONSUMED = "consumed"     # Successfully resumed (CAS consumed)
    EXPIRED = "expired"       # TTL exceeded, auto-cleaned
    CANCELLED = "cancelled"   # User cancelled recovery


@dataclass(frozen=True)
class Precondition:
    """A verifiable condition that must hold for safe resumption.

    Examples: session_id match, workspace_root unchanged, credential hash valid.
    """

    key: str                    # e.g. "session_id", "workspace_root", "credential_hash"
    expected_value: str         # Value at checkpoint creation time
    description: str = ""      # Human-readable explanation
    critical: bool = True      # If True, drift blocks resume; if False, warn only


@dataclass(frozen=True)
class StepRecord:
    """Record of a completed execution step."""

    step_id: str
    tool_name: str = ""
    action: str = ""
    result_summary: str = ""
    timestamp: float = 0.0
    has_side_effect: bool = False


@dataclass
class RecoveryCheckpoint:
    """Persistent checkpoint for cross-turn recovery.

    Created when RecoveryCoordinator returns HALT_WITH_CHECKPOINT.
    Enables the loop to resume from a known-safe state after user fixes the issue.

    Design decisions:
    - Mutable (state transitions: PENDING -> CONSUMED/EXPIRED/CANCELLED)
    - version field for optimistic locking (CAS semantics)
    - TTL-based expiry with configurable default
    - preconditions verified before resume to detect state drift
    """

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    turn_id: int = 0
    version: int = 1                              # Optimistic lock version
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0                   # Default 1 hour
    state: CheckpointState = CheckpointState.PENDING

    # Failure context
    failure_envelope_data: dict[str, Any] = field(default_factory=dict)
    interaction_request_id: str = ""

    # Execution state snapshot
    completed_steps: tuple[StepRecord, ...] = ()
    pending_steps: tuple[str, ...] = ()           # Descriptions of what remains
    messages_snapshot: list[dict[str, Any]] = field(default_factory=list)

    # Preconditions for safe resume
    preconditions: tuple[Precondition, ...] = ()

    # Context for resumption
    context_data: dict[str, Any] = field(default_factory=dict)

    # Recovery tracking
    resume_attempts: int = 0
    max_resume_attempts: int = 3
    consumed_at: float | None = None

    @property
    def is_expired(self) -> bool:
        """Whether the checkpoint has exceeded its TTL."""
        return time.time() > (self.created_at + self.ttl_seconds)

    @property
    def is_consumable(self) -> bool:
        """Whether this checkpoint can still be consumed (resumed)."""
        return (
            self.state == CheckpointState.PENDING
            and not self.is_expired
            and self.resume_attempts < self.max_resume_attempts
        )

    @property
    def expires_at(self) -> float:
        """Absolute timestamp when the checkpoint expires."""
        return self.created_at + self.ttl_seconds


@dataclass(frozen=True)
class ResumeResult:
    """Result of attempting to resume from a checkpoint."""

    success: bool
    reason: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    context_data: dict[str, Any] = field(default_factory=dict)
    drift_warnings: tuple[str, ...] = ()  # Non-critical precondition drifts


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint persistence backends.

    Implementations must provide CAS (Compare-And-Swap) semantics
    for load_and_consume to prevent concurrent resumption.
    """

    def save(self, checkpoint: RecoveryCheckpoint) -> None:
        """Persist a checkpoint. Overwrites if same checkpoint_id exists."""
        ...

    def load(self, checkpoint_id: str) -> RecoveryCheckpoint | None:
        """Load a checkpoint by ID. Returns None if not found or expired."""
        ...

    def consume(self, checkpoint_id: str) -> bool:
        """Atomically mark a PENDING checkpoint as CONSUMED. Returns True if successful."""
        ...

    def load_and_consume(self, checkpoint_id: str) -> RecoveryCheckpoint | None:
        """Atomically load and mark as CONSUMED (CAS semantics).

        Returns the checkpoint if successfully consumed, None if:
        - Not found
        - Already consumed
        - Expired
        - Max resume attempts exceeded

        This must be atomic — concurrent calls for the same ID must
        result in exactly one success.
        """
        ...

    def list_pending(self, session_id: str = "") -> list[RecoveryCheckpoint]:
        """List all pending (resumable) checkpoints, optionally filtered by session."""
        ...

    def cancel(self, checkpoint_id: str) -> bool:
        """Mark a checkpoint as cancelled. Returns True if it was pending."""
        ...

    def cleanup_expired(self) -> int:
        """Remove expired checkpoints. Returns count of removed entries."""
        ...


class InMemoryCheckpointStore:
    """In-memory checkpoint store for testing and single-session use.

    Thread-safe via simple dict operations (GIL-protected for CPython).
    For production multi-process use, implement DuckDB-backed store.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, RecoveryCheckpoint] = {}

    def save(self, checkpoint: RecoveryCheckpoint) -> None:
        """Persist a checkpoint. Overwrites if same checkpoint_id exists."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    def load(self, checkpoint_id: str) -> RecoveryCheckpoint | None:
        """Load a checkpoint by ID. Returns None if not found or expired."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return None
        if cp.is_expired:
            self._checkpoints.pop(checkpoint_id, None)
            return None
        return cp

    def consume(self, checkpoint_id: str) -> bool:
        """Atomically mark a PENDING checkpoint as CONSUMED. Returns True if successful."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return False
        if cp.is_expired:
            cp.state = CheckpointState.EXPIRED
            return False
        if cp.state != CheckpointState.PENDING:
            return False
        if cp.resume_attempts >= cp.max_resume_attempts:
            return False
        cp.state = CheckpointState.CONSUMED
        cp.consumed_at = time.time()
        cp.version += 1
        return True

    def load_and_consume(self, checkpoint_id: str) -> RecoveryCheckpoint | None:
        """Atomically load and mark as CONSUMED (CAS semantics).

        Returns the checkpoint if successfully consumed, None if not consumable.
        Increments resume_attempts on each call regardless of success.
        """
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return None

        # Auto-expire if past TTL
        if cp.is_expired:
            cp.state = CheckpointState.EXPIRED
            return None

        # Must be PENDING and within attempt limits
        if cp.state != CheckpointState.PENDING:
            return None
        if cp.resume_attempts >= cp.max_resume_attempts:
            return None

        # CAS: atomically transition to CONSUMED and bump version
        cp.resume_attempts += 1
        cp.state = CheckpointState.CONSUMED
        cp.consumed_at = time.time()
        cp.version += 1
        return cp

    def list_pending(self, session_id: str = "") -> list[RecoveryCheckpoint]:
        """List all pending (resumable) checkpoints, optionally filtered by session."""
        result: list[RecoveryCheckpoint] = []
        for cp in self._checkpoints.values():
            # Auto-expire stale entries
            if cp.is_expired and cp.state == CheckpointState.PENDING:
                cp.state = CheckpointState.EXPIRED
                continue
            if cp.state != CheckpointState.PENDING:
                continue
            if session_id and cp.session_id != session_id:
                continue
            result.append(cp)
        return result

    def cancel(self, checkpoint_id: str) -> bool:
        """Mark a checkpoint as cancelled. Returns True if it was pending."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return False
        if cp.state != CheckpointState.PENDING:
            return False
        cp.state = CheckpointState.CANCELLED
        cp.version += 1
        return True

    def cleanup_expired(self) -> int:
        """Remove expired checkpoints. Returns count of removed entries."""
        expired_ids: list[str] = []
        for cid, cp in self._checkpoints.items():
            if cp.is_expired:
                expired_ids.append(cid)
        for cid in expired_ids:
            del self._checkpoints[cid]
        return len(expired_ids)


@runtime_checkable
class DriftVerifier(Protocol):
    """Protocol for custom drift verification logic."""

    def get_current_value(self, key: str) -> str:
        """Retrieve the current value for a precondition key."""
        ...


class DefaultDriftVerifier:
    """Default drift verifier — returns empty string for all keys."""

    def get_current_value(self, key: str) -> str:
        """Returns empty string (no environment introspection by default)."""
        return ""


class CheckpointResumer:
    """Orchestrates safe resumption from a checkpoint.

    Responsibilities:
    1. Load checkpoint via CAS (prevents double-resume)
    2. Verify all preconditions (detect state drift)
    3. Build resume context (messages + injection)
    4. Return ResumeResult for the engine loop to act on
    """

    def __init__(
        self,
        store: CheckpointStore,
        drift_verifier: DriftVerifier | None = None,
    ) -> None:
        self._store = store
        self._drift_verifier = drift_verifier or DefaultDriftVerifier()

    def resume(
        self,
        checkpoint_id: str,
        current_context: dict[str, Any] | None = None,
    ) -> ResumeResult:
        """Attempt to resume from a checkpoint.

        Args:
            checkpoint_id: The checkpoint to resume from
            current_context: Current environment values for drift detection
                           (e.g. {"session_id": "...", "workspace_root": "..."})
        """
        # 1. Load checkpoint (without consuming)
        checkpoint = self._store.load(checkpoint_id)
        if checkpoint is None or not checkpoint.is_consumable:
            return ResumeResult(
                success=False,
                reason="Checkpoint not available (expired, consumed, or not found)",
            )

        # 2. Verify preconditions
        context = current_context or {}
        drift_errors: list[str] = []
        drift_warnings: list[str] = []

        for precond in checkpoint.preconditions:
            actual = context.get(precond.key, "")
            if str(actual) != precond.expected_value:
                msg = f"{precond.key}: expected '{precond.expected_value}', got '{actual}'"
                if precond.critical:
                    drift_errors.append(msg)
                else:
                    drift_warnings.append(msg)

        if drift_errors:
            # Preconditions failed: increment resume_attempts, keep PENDING
            checkpoint.resume_attempts += 1
            if checkpoint.resume_attempts >= checkpoint.max_resume_attempts:
                checkpoint.state = CheckpointState.EXPIRED
            self._store.save(checkpoint)
            return ResumeResult(
                success=False,
                reason=f"Critical precondition drift: {'; '.join(drift_errors)}",
                drift_warnings=tuple(drift_warnings),
            )

        # 3. Preconditions passed — atomically consume
        if not self._store.consume(checkpoint_id):
            return ResumeResult(
                success=False,
                reason="Checkpoint not available (expired, consumed, or not found)",
            )

        # 4. Build resume messages
        messages = list(checkpoint.messages_snapshot)
        failure_msg = checkpoint.failure_envelope_data.get("message", "unknown")
        messages.append({
            "role": "system",
            "content": (
                f"[Recovery] Resuming from checkpoint. "
                f"Previous failure: {failure_msg}. "
                f"User has resolved the issue. Continue from where you left off."
            ),
        })

        return ResumeResult(
            success=True,
            messages=messages,
            context_data=checkpoint.context_data,
            drift_warnings=tuple(drift_warnings),
        )
