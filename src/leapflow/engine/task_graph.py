"""Task DAG (Directed Acyclic Graph) data model for complex task orchestration.

Supports:
- Dependency-based execution ordering (topological sort)
- Parallel execution of independent nodes
- Template parameter resolution (${task_id.output} references)
- Conditional execution (skip nodes when condition is false)
- Retry policies with exponential backoff and fallback actions
- Graph validation (cycle detection, reference integrity)
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Template reference pattern: ${node_id.output}, ${node_id.result.key}, ${graph.goal}, ${env.key}
_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")


class TaskStatus(Enum):
    """Lifecycle status of a task node."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RetryPolicy:
    """Retry strategy for failed task nodes."""

    max_retries: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    fallback_action: Optional[str] = None
    retryable_errors: List[str] = field(default_factory=list)

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay before retry attempt (exponential backoff)."""
        return self.backoff_seconds * (self.backoff_multiplier ** attempt)

    def is_retryable(self, error: str) -> bool:
        """Check if the given error is eligible for retry."""
        if not self.retryable_errors:
            return True
        return any(pattern in error for pattern in self.retryable_errors)


@dataclass
class TaskNode:
    """A single executable node in the task graph."""

    id: str
    name: str
    action: str
    action_type: str = "skill"
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    expected_effect: Optional[str] = None
    retry_policy: Optional[RetryPolicy] = None
    timeout_seconds: float = 300.0
    repeat_count: int = 1
    repeat_until: Optional[str] = None

    # Runtime state (mutated during execution)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    attempt_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_terminal(self) -> bool:
        """Node has reached a final state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)


class GraphValidationError(Exception):
    """Raised when graph structure is invalid."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Graph validation failed: {'; '.join(errors)}")


@dataclass
class TaskGraph:
    """A DAG of task nodes representing a complex multi-step plan.

    Invariants:
    - No cycles (validated on construction)
    - All depends_on references point to existing nodes
    - Template references (${id.output}) point to upstream nodes only
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === Construction ===

    def add_node(self, node: TaskNode) -> None:
        """Add a node, validating dependencies exist."""
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        for dep_id in node.depends_on:
            if dep_id not in self.nodes:
                raise ValueError(
                    f"Node '{node.id}' depends on unknown node '{dep_id}'"
                )
            if dep_id == node.id:
                raise ValueError(f"Node '{node.id}' cannot depend on itself")
        self.nodes[node.id] = node

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """Construct from a serialized dict (e.g. LLM JSON output)."""
        graph = cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            goal=data.get("goal", ""),
            metadata=data.get("metadata", {}),
        )
        if "created_at" in data:
            graph.created_at = data["created_at"]

        for node_data in data.get("nodes", []):
            retry_data = node_data.get("retry_policy")
            retry_policy = None
            if retry_data:
                retry_policy = RetryPolicy(
                    max_retries=retry_data.get("max_retries", 3),
                    backoff_seconds=retry_data.get("backoff_seconds", 1.0),
                    backoff_multiplier=retry_data.get("backoff_multiplier", 2.0),
                    fallback_action=retry_data.get("fallback_action"),
                    retryable_errors=retry_data.get("retryable_errors", []),
                )
            node = TaskNode(
                id=node_data["id"],
                name=node_data.get("name", node_data["id"]),
                action=node_data["action"],
                action_type=node_data.get("action_type", "skill"),
                params=node_data.get("params", {}),
                depends_on=node_data.get("depends_on", []),
                condition=node_data.get("condition"),
                expected_effect=node_data.get("expected_effect"),
                retry_policy=retry_policy,
                timeout_seconds=node_data.get("timeout_seconds", 300.0),
                repeat_count=int(node_data.get("repeat_count", 1)),
                repeat_until=node_data.get("repeat_until"),
            )
            graph.nodes[node.id] = node

        errors = graph.validate()
        if errors:
            raise GraphValidationError(errors)
        return graph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence or display."""
        nodes_list = []
        for node in self.nodes.values():
            node_dict: Dict[str, Any] = {
                "id": node.id,
                "name": node.name,
                "action": node.action,
                "action_type": node.action_type,
                "params": node.params,
                "depends_on": node.depends_on,
                "condition": node.condition,
                "expected_effect": node.expected_effect,
                "timeout_seconds": node.timeout_seconds,
                "repeat_count": node.repeat_count,
                "repeat_until": node.repeat_until,
                "status": node.status.value,
                "result": node.result,
                "error": node.error,
                "attempt_count": node.attempt_count,
                "started_at": node.started_at,
                "completed_at": node.completed_at,
            }
            if node.retry_policy:
                node_dict["retry_policy"] = {
                    "max_retries": node.retry_policy.max_retries,
                    "backoff_seconds": node.retry_policy.backoff_seconds,
                    "backoff_multiplier": node.retry_policy.backoff_multiplier,
                    "fallback_action": node.retry_policy.fallback_action,
                    "retryable_errors": node.retry_policy.retryable_errors,
                }
            nodes_list.append(node_dict)

        return {
            "id": self.id,
            "goal": self.goal,
            "nodes": nodes_list,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    # === Query ===

    def ready_nodes(self) -> List[TaskNode]:
        """Return all PENDING nodes whose dependencies are all COMPLETED/SKIPPED."""
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                self.nodes[dep_id].status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
                for dep_id in node.depends_on
            )
            if deps_satisfied:
                ready.append(node)
        return ready

    def topological_order(self) -> List[str]:
        """Return node IDs in valid execution order (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                # dep_id -> node (node depends on dep_id)
                # in_degree counts incoming edges in dependency direction
                in_degree[node.id] += 0  # already counted below

        # Rebuild in-degree: an edge dep -> node means node has in-degree +1
        in_degree = {nid: 0 for nid in self.nodes}
        adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                adjacency[dep_id].append(node.id)
                in_degree[node.id] += 1

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: List[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for downstream_id in adjacency[nid]:
                in_degree[downstream_id] -= 1
                if in_degree[downstream_id] == 0:
                    queue.append(downstream_id)

        if len(order) != len(self.nodes):
            raise GraphValidationError(["Cycle detected in task graph"])
        return order

    @property
    def is_complete(self) -> bool:
        """All nodes reached terminal state."""
        return bool(self.nodes) and all(n.is_terminal for n in self.nodes.values())

    @property
    def has_failed(self) -> bool:
        """At least one node FAILED (and no fallback succeeded)."""
        return any(n.status == TaskStatus.FAILED for n in self.nodes.values())

    @property
    def progress(self) -> float:
        """Completion ratio (0.0 to 1.0)."""
        if not self.nodes:
            return 1.0
        terminal = sum(1 for n in self.nodes.values() if n.is_terminal)
        return terminal / len(self.nodes)

    # === Parameter Resolution ===

    def resolve_params(self, node: TaskNode) -> Dict[str, Any]:
        """Resolve ${task_id.output} and ${task_id.result.field} template references.

        Template syntax:
        - ${node_id.output}  -> upstream node's result (full)
        - ${node_id.result.key} -> specific field from upstream result dict
        - ${graph.goal} -> the original user goal text
        - ${env.key} -> from graph.metadata["env"]
        """
        resolved: Dict[str, Any] = {}
        for key, value in node.params.items():
            result = self._resolve_value(value)
            if result is None:
                logger.warning(
                    "resolve_params: key=%s resolved to None, using empty string",
                    key,
                )
                result = ""
            resolved[key] = result
        return resolved

    def _resolve_value(self, value: Any) -> Any:
        """Recursively resolve templates in a value (str, dict, list)."""
        if isinstance(value, str):
            # If the entire string is a single template, return the resolved value directly
            # (preserving non-string types from upstream results)
            match = _TEMPLATE_PATTERN.fullmatch(value)
            if match:
                return self._resolve_reference(match.group(1))
            # Otherwise, do string interpolation for embedded templates
            def replacer(m: re.Match) -> str:
                resolved = self._resolve_reference(m.group(1))
                return str(resolved) if resolved is not None else ""
            return _TEMPLATE_PATTERN.sub(replacer, value)
        elif isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        return value

    def _resolve_reference(self, ref: str) -> Any:
        """Parse and resolve a single template reference path."""
        parts = ref.split(".")
        if not parts:
            logger.warning("resolve_reference: empty ref")
            return ""

        root = parts[0]

        # ${graph.goal}
        if root == "graph":
            if len(parts) >= 2 and parts[1] == "goal":
                return self.goal
            return ""

        # ${env.key}
        if root == "env":
            env = self.metadata.get("env", {})
            result = _nested_get(env, parts[1:])
            if result is None:
                logger.warning("resolve_reference: env key not found ref=%s", ref)
                return ""
            return result

        # ${node_id.output} or ${node_id.result.field}
        if root not in self.nodes:
            logger.warning("resolve_reference: node not found ref=%s", ref)
            return ""
        upstream = self.nodes[root]
        if len(parts) < 2:
            if upstream.result is None:
                logger.warning("resolve_reference: node result is None ref=%s", ref)
                return ""
            return upstream.result

        accessor = parts[1]
        if accessor in ("output", "result"):
            if len(parts) == 2:
                if upstream.result is None:
                    logger.warning("resolve_reference: node result is None ref=%s", ref)
                    return ""
                return upstream.result
            # Nested field access: ${node_id.result.key.subkey}
            result = _nested_get(upstream.result, parts[2:])
            if result is None:
                logger.warning("resolve_reference: nested access returned None ref=%s", ref)
                return ""
            return result
        logger.warning("resolve_reference: unknown accessor ref=%s", ref)
        return ""

    # === Validation ===

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid).

        Checks:
        1. No cycles (topological sort succeeds)
        2. All depends_on reference existing node IDs
        3. All ${ref} templates reference upstream nodes
        4. No self-dependencies
        """
        errors: List[str] = []

        # Check self-dependencies and missing references
        for node in self.nodes.values():
            if node.id in node.depends_on:
                errors.append(f"Node '{node.id}' depends on itself")
            for dep_id in node.depends_on:
                if dep_id not in self.nodes:
                    errors.append(
                        f"Node '{node.id}' depends on unknown node '{dep_id}'"
                    )

        # Cycle detection
        if not errors and self._detect_cycle():
            errors.append("Graph contains a cycle")

        # Template reference validation
        if not errors:
            upstream_map = self._build_upstream_map()
            for node in self.nodes.values():
                template_errors = self._validate_templates(node, upstream_map)
                errors.extend(template_errors)

        return errors

    def _detect_cycle(self) -> bool:
        """Return True if graph contains a cycle (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                if dep_id in self.nodes:
                    in_degree[node.id] += 1

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        visited = 0
        while queue:
            nid = queue.popleft()
            visited += 1
            for other in self.nodes.values():
                if nid in other.depends_on:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        queue.append(other.id)

        return visited != len(self.nodes)

    def _build_upstream_map(self) -> Dict[str, Set[str]]:
        """Build transitive upstream set for each node."""
        upstream: Dict[str, Set[str]] = {nid: set() for nid in self.nodes}
        for nid in self.topological_order():
            node = self.nodes[nid]
            for dep_id in node.depends_on:
                upstream[nid].add(dep_id)
                upstream[nid].update(upstream[dep_id])
        return upstream

    def _validate_templates(
        self, node: TaskNode, upstream_map: Dict[str, Set[str]]
    ) -> List[str]:
        """Validate that template references in params point to upstream nodes."""
        errors: List[str] = []
        refs = self._extract_refs(node.params)
        upstream_ids = upstream_map.get(node.id, set())

        for ref in refs:
            parts = ref.split(".")
            root = parts[0]
            if root in ("graph", "env"):
                continue
            if root not in self.nodes:
                errors.append(
                    f"Node '{node.id}' references unknown node '{root}' in template"
                )
            elif root not in upstream_ids:
                errors.append(
                    f"Node '{node.id}' references non-upstream node '{root}' in template"
                )
        return errors

    def _extract_refs(self, value: Any) -> List[str]:
        """Extract all ${...} references from a value recursively."""
        refs: List[str] = []
        if isinstance(value, str):
            refs.extend(m.group(1) for m in _TEMPLATE_PATTERN.finditer(value))
        elif isinstance(value, dict):
            for v in value.values():
                refs.extend(self._extract_refs(v))
        elif isinstance(value, list):
            for item in value:
                refs.extend(self._extract_refs(item))
        return refs

    # === State Management ===

    def mark_running(self, node_id: str) -> None:
        """Transition node to RUNNING state."""
        node = self._get_node(node_id)
        node.status = TaskStatus.RUNNING
        node.started_at = time.time()
        node.attempt_count += 1

    def mark_completed(self, node_id: str, result: Any) -> None:
        """Transition node to COMPLETED with result."""
        node = self._get_node(node_id)
        node.status = TaskStatus.COMPLETED
        node.result = result
        node.error = None
        node.completed_at = time.time()

    def mark_failed(self, node_id: str, error: str) -> None:
        """Transition node to FAILED with error."""
        node = self._get_node(node_id)
        node.status = TaskStatus.FAILED
        node.error = error
        node.completed_at = time.time()

    def mark_skipped(self, node_id: str, reason: str = "") -> None:
        """Transition node to SKIPPED."""
        node = self._get_node(node_id)
        node.status = TaskStatus.SKIPPED
        node.error = reason or None
        node.completed_at = time.time()

    def reset_node(self, node_id: str) -> None:
        """Reset node to PENDING (for retry)."""
        node = self._get_node(node_id)
        node.status = TaskStatus.PENDING
        node.result = None
        node.error = None
        node.started_at = None
        node.completed_at = None

    # === Utilities ===

    def summary(self) -> str:
        """Human-readable execution summary."""
        lines = [f"TaskGraph '{self.id}' — {self.goal}"]
        lines.append(f"Progress: {self.progress:.0%} ({len(self.nodes)} nodes)")
        lines.append("-" * 40)
        for nid in self._safe_topological_order():
            node = self.nodes[nid]
            duration = f" ({node.duration_s:.1f}s)" if node.duration_s else ""
            status_icon = _STATUS_ICONS.get(node.status, "?")
            lines.append(f"  {status_icon} {node.name} [{node.status.value}]{duration}")
            if node.error:
                lines.append(f"      error: {node.error}")
        return "\n".join(lines)

    def downstream_of(self, node_id: str) -> Set[str]:
        """All nodes transitively depending on given node."""
        self._get_node(node_id)  # validate existence
        result: Set[str] = set()
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for other in self.nodes.values():
                if current in other.depends_on and other.id not in result:
                    result.add(other.id)
                    queue.append(other.id)
        return result

    def upstream_of(self, node_id: str) -> Set[str]:
        """All nodes that given node transitively depends on."""
        self._get_node(node_id)  # validate existence
        result: Set[str] = set()
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for dep_id in self.nodes[current].depends_on:
                if dep_id not in result:
                    result.add(dep_id)
                    queue.append(dep_id)
        return result

    # === Private Helpers ===

    def _get_node(self, node_id: str) -> TaskNode:
        """Retrieve node by ID or raise KeyError."""
        if node_id not in self.nodes:
            raise KeyError(f"No node with ID '{node_id}'")
        return self.nodes[node_id]

    def _safe_topological_order(self) -> List[str]:
        """Topological order that won't raise on invalid graphs."""
        try:
            return self.topological_order()
        except GraphValidationError:
            return list(self.nodes.keys())


# === Module-level helpers ===

_STATUS_ICONS = {
    TaskStatus.PENDING: "○",
    TaskStatus.READY: "◎",
    TaskStatus.RUNNING: "◉",
    TaskStatus.COMPLETED: "●",
    TaskStatus.FAILED: "✗",
    TaskStatus.SKIPPED: "⊘",
}


def _nested_get(obj: Any, keys: List[str]) -> Any:
    """Safely traverse nested dicts/objects by key path."""
    current = obj
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return None
    return current
