# Copyright (c) Alibaba, Inc. and its affiliates.
"""Core type definitions for the long-horizon async task scheduler."""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    """Lifecycle states for a scheduled task."""

    ARMED = "armed"
    WATCHING = "watching"
    DUE = "due"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    SUSPENDED = "suspended"
    PAUSED = "paused"


class ExecutionTier(str, Enum):
    """Where the task should be executed."""

    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"


class SchedulerExecutionMode(str, Enum):
    """How a scheduled task runs its skill on each trigger.

    Both modes are dispatched through the single :class:`SkillExecutor`
    contract; the mode travels in a task's ``parameters['execution_mode']`` and
    is routed transparently by the coordinator's routing executor. ``SCRIPT`` is
    the default and runs the skill directly; ``AGENT`` runs an isolated,
    bounded LLM tool loop (see ``AgentSkillExecutor``).

    The two modes share the full ``execute(skill_name, parameters) -> dict``
    interface and the same retry / error-handling path in ``LocalScheduler``;
    the only difference is *which* executor the router selects.
    """

    SCRIPT = "script"
    AGENT = "agent"

    @classmethod
    def from_value(cls, value: object) -> "SchedulerExecutionMode":
        """Coerce an arbitrary value to a mode, defaulting to ``SCRIPT``.

        Any value that is not a recognized mode (``None``, ``""``, or an
        unknown string) maps to ``SCRIPT`` — the same fall-through the router
        applies, so a task with a malformed mode still runs rather than error.
        """
        try:
            return cls(str(value))
        except ValueError:
            return cls.SCRIPT


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArmedTask:
    """Mutable representation of a scheduled task with full lifecycle state."""

    skill_name: str
    trigger_type: str
    trigger_config: dict
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "armed"
    execution_tier: str = "auto"
    context_snapshot: dict = field(default_factory=dict)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    next_due_at: float = 0.0
    last_run_at: float = 0.0
    run_count: int = 0
    max_runs: int = -1  # -1 = unlimited
    grace_seconds: float = 120.0
    parameters: dict = field(default_factory=dict)
    cloud_worker_id: str = ""
    metadata: dict = field(default_factory=dict)
    max_retries: int = 0
    retry_count: int = 0
    retry_backoff_s: float = 60.0


@dataclass
class TaskStatus:
    """Runtime status snapshot for a task."""

    task: ArmedTask
    is_running: bool = False
    logs_tail: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Trigger(Protocol):
    """Protocol that all trigger implementations must satisfy.

    Triggers determine *when* a task becomes due for execution.
    All triggers must be serializable to/from plain dicts for persistence.
    """

    @property
    def trigger_type(self) -> str:
        """Unique string identifier for the trigger kind."""
        ...

    def is_due(self, now: float) -> bool:
        """Return True if the trigger condition is currently met."""
        ...

    def advance(self, now: float) -> None:
        """Advance internal state to the next trigger point after firing."""
        ...

    @property
    def next_due_at(self) -> float:
        """Timestamp (epoch seconds) of the next expected trigger."""
        ...

    def serialize(self) -> dict:
        """Serialize trigger state to a JSON-compatible dict."""
        ...


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol for cloud/remote compute backends.

    Backends manage the lifecycle of remote workers that execute skills.
    """

    @property
    def backend_type(self) -> str:
        """Identifier string for this backend (e.g. 'cloudflare', 'fly')."""
        ...

    async def create_worker(
        self, worker_id: str, package_path: Path, visibility: str
    ) -> str:
        """Create a new remote worker. Returns the worker URL or identifier."""
        ...

    async def inject_secrets(
        self, worker_id: str, secrets: Dict[str, str]
    ) -> None:
        """Inject environment secrets into a worker."""
        ...

    async def deploy(self, worker_id: str) -> None:
        """Deploy (or redeploy) the worker."""
        ...

    async def stop(self, worker_id: str) -> None:
        """Stop a running worker without destroying it."""
        ...

    async def get_status(self, worker_id: str) -> str:
        """Get current status string of the worker."""
        ...

    async def get_logs(self, worker_id: str, tail: int) -> List[str]:
        """Retrieve recent log lines from the worker."""
        ...

    async def destroy(self, worker_id: str) -> None:
        """Permanently destroy the worker and its resources."""
        ...


class SkillExecutor(Protocol):
    """Unified execution-mode contract for the scheduler.

    This is the single interface every scheduler execution mode implements:
    the default script executor, the ``AgentSkillExecutor`` (agent mode), and
    the ``_RoutingExecutor`` that dispatches between them all satisfy it via
    structural subtyping. Because the contract is identical across modes, the
    ``LocalScheduler`` tick, retry logic, and result handling never branch on
    the mode — they call ``execute`` and interpret the returned dict.

    Contract:
    - ``skill_name``: the skill/instruction identifier being executed.
    - ``parameters``: the task payload; ``execution_mode`` (see
      :class:`SchedulerExecutionMode`) selects the mode when routing.
    - returns a dict with ``ok`` (bool) and, on success, ``output``; on
      failure, ``error`` (str). Implementations must never raise: a scheduler
      tick must survive any skill failure.
    """

    async def execute(self, skill_name: str, parameters: dict) -> dict:
        """Execute a skill. Returns ``{"ok": bool, "output"|"error": ...}``."""
        ...
