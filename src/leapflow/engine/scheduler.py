"""DAG-based task scheduler with parallel execution, retry, and fault tolerance.

Executes a TaskGraph by:
1. Finding ready nodes (all dependencies completed)
2. Executing them in parallel (bounded by max_concurrency)
3. Handling failures via RetryPolicy (exponential backoff + fallback)
4. Propagating results to downstream nodes via template resolution
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Dict, Optional, Set

from .task_graph import TaskGraph, TaskNode, TaskStatus, RetryPolicy
from leapflow.skills.registry import SkillRegistry, SkillResult
from leapflow.platform.protocol import HostRpc

logger = logging.getLogger(__name__)

# Callback type for progress reporting
NodeCallback = Callable[[TaskNode, TaskGraph], None]


class SchedulerError(Exception):
    """Raised when the scheduler encounters an unrecoverable issue."""


class DeadlockError(SchedulerError):
    """Raised when no progress can be made (no ready nodes, graph incomplete)."""


class TaskScheduler:
    """Async DAG scheduler with bounded parallelism and fault tolerance.

    Design principles:
    - Single-responsibility: only orchestrates execution order and concurrency
    - Open/closed: extensible dispatch via action_type routing
    - Dependency inversion: depends on SkillRegistry and HostRpc abstractions
    """

    def __init__(
        self,
        registry: SkillRegistry,
        rpc: HostRpc,
        *,
        max_concurrency: int = 3,
        on_node_complete: Optional[NodeCallback] = None,
        on_node_failed: Optional[NodeCallback] = None,
        graph_planner: Optional[Any] = None,
    ) -> None:
        self._registry = registry
        self._rpc = rpc
        self._max_concurrency = max_concurrency
        self._on_node_complete = on_node_complete
        self._on_node_failed = on_node_failed
        self._graph_planner = graph_planner
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ═══ Public API ═══

    async def execute_graph(self, graph: TaskGraph) -> TaskGraph:
        """Execute the entire task graph, returning the updated graph.

        Algorithm:
        1. Validate graph structure
        2. Initialize concurrency semaphore
        3. Loop: find ready nodes → execute in parallel → update states
        4. Continue until graph.is_complete or deadlock detected
        5. Return graph with all results populated

        Raises:
            ValueError: If graph fails validation.
            DeadlockError: If no progress is possible.
        """
        errors = graph.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._semaphore = asyncio.Semaphore(self._max_concurrency)

        while not graph.is_complete:
            ready = graph.ready_nodes()
            if not ready:
                self._handle_deadlock(graph)
                break

            tasks = [self._execute_with_semaphore(node, graph) for node in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    node = ready[i]
                    logger.error(
                        "scheduler.unhandled_exception node=%s error=%s",
                        node.id, result, exc_info=result
                    )
                    # If the node hasn't been marked failed yet (exception outside _execute_node)
                    if node.status == TaskStatus.RUNNING:
                        graph.mark_failed(node.id, f"Unhandled: {result}")

        return graph

    async def execute_node_isolated(
        self, node: TaskNode, graph: TaskGraph
    ) -> TaskNode:
        """Execute a single node (useful for testing or manual re-execution).

        Returns the node with updated status/result.
        """
        await self._execute_node(node, graph)
        return node

    # ═══ Internal Execution ═══

    async def _execute_with_semaphore(
        self, node: TaskNode, graph: TaskGraph
    ) -> None:
        """Execute a single node with concurrency limiting."""
        async with self._semaphore:  # type: ignore[union-attr]
            await self._execute_node(node, graph)

    async def _execute_node(self, node: TaskNode, graph: TaskGraph) -> None:
        """Execute a single task node with retry and error handling.

        Steps:
        1. Evaluate condition (skip if false)
        2. Resolve params (template references from upstream results)
        3. Execute action (skill or bridge) with retry loop
        4. Handle success/failure state transitions
        """
        # Condition check — skip node if condition evaluates to false
        if node.condition:
            if not self._evaluate_condition(node.condition, graph):
                graph.mark_skipped(node.id, reason=f"Condition false: {node.condition}")
                return

        graph.mark_running(node.id)
        resolved_params = graph.resolve_params(node)

        total_iterations = max(1, node.repeat_count)
        last_result: Any = None
        policy = node.retry_policy or RetryPolicy(max_retries=0)
        last_error: Optional[str] = None
        success = False

        for iteration in range(total_iterations):
            last_error = None
            success = False

            for attempt in range(policy.max_retries + 1):
                if attempt > 0:
                    node.attempt_count += 1
                try:
                    last_result = await self._dispatch_action(node, resolved_params)
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        "scheduler.node_attempt_failed node=%s attempt=%d error=%s",
                        node.id, attempt + 1, last_error,
                    )
                    if attempt < policy.max_retries:
                        if not policy.is_retryable(last_error):
                            logger.info(
                                "scheduler.non_retryable node=%s error=%s",
                                node.id, last_error,
                            )
                            break
                        delay = policy.delay_for_attempt(attempt)
                        await asyncio.sleep(delay)

            if not success:
                break

            node.result = last_result

            if node.repeat_until and iteration < total_iterations - 1:
                if self._evaluate_condition(node.repeat_until, graph):
                    break

        if success:
            graph.mark_completed(node.id, last_result)
            if self._on_node_complete:
                self._on_node_complete(node, graph)
            return

        # All retries exhausted — try fallback
        if policy.fallback_action:
            try:
                result = await self._dispatch_fallback(
                    node, policy.fallback_action, resolved_params
                )
                graph.mark_completed(node.id, result)
                if self._on_node_complete:
                    self._on_node_complete(node, graph)
                return
            except Exception as e:
                last_error = f"Fallback '{policy.fallback_action}' also failed: {e}"
                logger.error("scheduler.fallback_failed node=%s error=%s", node.id, e)

        # Final failure
        graph.mark_failed(node.id, last_error or "Unknown error")
        if self._on_node_failed:
            self._on_node_failed(node, graph)

        # Attempt LLM re-planning before cascading skip
        if self._graph_planner is not None:
            try:
                graph = await self._graph_planner.replan_on_failure(graph, node)
                if not graph.downstream_of(node.id):
                    return
            except Exception:
                logger.debug("scheduler.replan_on_failure failed", exc_info=True)

        # Skip downstream nodes that depend on this failed node
        self._cascade_skip(node.id, graph)

    # ═══ Action Dispatch ═══

    async def _dispatch_action(
        self, node: TaskNode, params: Dict[str, Any]
    ) -> Any:
        """Route execution to skill registry or RPC bridge based on action_type."""
        if node.action_type == "skill":
            result: SkillResult = await self._registry.invoke(
                node.action,
                user_goal=node.expected_effect or "",
                **params,
            )
            if not result.ok:
                raise RuntimeError(result.error or f"Skill '{node.action}' failed")
            return result.output
        elif node.action_type == "bridge":
            return await self._dispatch_bridge(node, params)
        else:
            raise ValueError(f"Unknown action_type: '{node.action_type}'")

    async def _dispatch_bridge(
        self, node: TaskNode, params: Dict[str, Any]
    ) -> Any:
        """Execute a bridge action, optionally wrapping with prediction loop."""
        pl = self._registry.prediction_loop
        if pl is not None and pl.enabled and node.expected_effect:
            prediction = pl.create_from_react_prediction(
                action_desc=f"dag_bridge:{node.action}",
                predicted_effect=node.expected_effect,
            )
            await pl.capture_pre_snapshot()
            result = await self._rpc.call(node.action, params or None)
            await pl.verify_prediction(prediction)
            return result
        return await self._rpc.call(node.action, params or None)

    async def _dispatch_fallback(
        self, node: TaskNode, fallback_action: str, params: Dict[str, Any]
    ) -> Any:
        """Execute fallback action on final failure.

        Attempts skill registry first, falls back to bridge call.
        Both paths are observable by the prediction loop when available.
        """
        skill = self._registry.get(fallback_action)
        if skill is not None:
            result = await self._registry.invoke(
                fallback_action, user_goal=node.expected_effect or "", **params,
            )
            if not result.ok:
                raise RuntimeError(result.error or f"Fallback skill '{fallback_action}' failed")
            return result.output
        # Try as bridge method (with prediction wrap if available)
        pl = self._registry.prediction_loop
        if pl is not None and pl.enabled and node.expected_effect:
            prediction = pl.create_from_react_prediction(
                action_desc=f"dag_fallback:{fallback_action}",
                predicted_effect=node.expected_effect,
            )
            await pl.capture_pre_snapshot()
            result = await self._rpc.call(fallback_action, params or None)
            await pl.verify_prediction(prediction)
            return result
        return await self._rpc.call(fallback_action, params or None)

    # ═══ Condition Evaluation ═══

    _COND_REFERENCE = re.compile(r"\$\{([^}]+)\}")

    def _evaluate_condition(self, condition: str, graph: TaskGraph) -> bool:
        """Safely evaluate a condition expression against graph state.

        Supports:
        - "${node_id.status}" == "completed"
        - "${node_id.output}" is not None
        - Simple comparison expressions

        Uses restricted evaluation with only graph-derived context.
        """
        resolved = condition
        for match in self._COND_REFERENCE.finditer(condition):
            ref = match.group(1)
            value = self._resolve_condition_ref(ref, graph)
            # Replace with repr for safe evaluation
            resolved = resolved.replace(match.group(0), repr(value))

        try:
            # Restricted eval: only allow comparison operators
            return bool(eval(resolved, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception:
            logger.warning("scheduler.condition_eval_failed condition=%s", condition)
            return True  # Default to executing on eval failure

    @staticmethod
    def _resolve_condition_ref(ref: str, graph: TaskGraph) -> Any:
        """Resolve a single ${...} reference in a condition string."""
        parts = ref.split(".")
        if not parts:
            return None

        root = parts[0]
        if root not in graph.nodes:
            return None

        node = graph.nodes[root]
        if len(parts) < 2:
            return node.result

        accessor = parts[1]
        if accessor == "status":
            return node.status.value
        elif accessor in ("output", "result"):
            if len(parts) == 2:
                return node.result
            # Nested access
            current = node.result
            for key in parts[2:]:
                if isinstance(current, dict):
                    current = current.get(key)
                elif hasattr(current, key):
                    current = getattr(current, key)
                else:
                    return None
            return current
        return None

    # ═══ Deadlock & Cascade ═══

    def _handle_deadlock(self, graph: TaskGraph) -> None:
        """Handle situation where no nodes are ready but graph is not complete.

        This occurs when all remaining nodes have failed dependencies.
        Log the state and mark unresolvable nodes as skipped.
        """
        unfinished = [
            n for n in graph.nodes.values() if not n.is_terminal
        ]
        if not unfinished:
            return

        # Check if all unfinished nodes are blocked by failed nodes
        failed_ids: Set[str] = {
            n.id for n in graph.nodes.values() if n.status == TaskStatus.FAILED
        }
        for node in unfinished:
            blocked_by_failure = any(
                dep_id in failed_ids for dep_id in node.depends_on
            )
            if blocked_by_failure:
                graph.mark_skipped(
                    node.id,
                    reason="Skipped: upstream dependency failed",
                )
            else:
                # True deadlock — should not happen with valid DAG
                logger.error(
                    "scheduler.deadlock node=%s deps=%s", node.id, node.depends_on
                )
                graph.mark_failed(node.id, "Deadlock: no progress possible")

    def _cascade_skip(self, failed_node_id: str, graph: TaskGraph) -> None:
        """Skip all downstream nodes that transitively depend on the failed node."""
        downstream = graph.downstream_of(failed_node_id)
        for nid in downstream:
            node = graph.nodes[nid]
            if not node.is_terminal:
                graph.mark_skipped(
                    nid,
                    reason=f"Skipped: upstream '{failed_node_id}' failed",
                )
