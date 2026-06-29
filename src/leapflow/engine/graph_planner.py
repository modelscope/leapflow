"""LLM-driven Task DAG planner.

Generates structured task graphs from natural language goals,
leveraging the skill catalog and execution context for informed planning.
Supports dynamic re-planning on failure.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .task_graph import TaskGraph, TaskNode, RetryPolicy
from ..skills.registry import SkillRegistry
from ..llm.base import LLMProvider
from ..llm.message_builder import build_system_message, build_user_message_text

logger = logging.getLogger(__name__)

# ── JSON extraction pattern ──
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)

# ── Planning prompt template ──
_PLANNING_SYSTEM_PROMPT = """\
You are a task planner for LeapFlow. Given a user goal, generate a structured \
execution plan as a JSON DAG (directed acyclic graph).

## Available Skills
{skill_catalog}

## Available Bridge Methods
Standard OS operations via RPC: file.list, file.move, file.copy, file.delete, \
app.launch, app.activate, clipboard.get, ax.tree, ax.perform, system.info.

## Output Format
Return a JSON object with a "nodes" array. Each node:
{{
  "id": "unique_short_id",
  "name": "Human-readable step name",
  "action": "skill_name or bridge_method",
  "action_type": "skill" or "bridge",
  "params": {{"key": "value"}},
  "depends_on": ["id_of_upstream_node"],
  "condition": null or "${{node_id.status}} == 'completed'",
  "expected_effect": "one sentence describing what this step will change",
  "retry": null or {{"max_retries": 2, "backoff_seconds": 1.0}}
}}

## Rules
1. Use template references ${{node_id.output}} to pass results between nodes.
2. Maximize parallelism: only add depends_on when data or ordering is required.
3. Keep plans concise (3-8 nodes for typical tasks).
4. Include retry policies for unreliable operations (network, file I/O).
5. Use conditions sparingly — only when a branch truly depends on runtime state.

## Context
{context_section}

Return ONLY the JSON object, no explanations.
"""

_REPLAN_SYSTEM_PROMPT = """\
You are replanning a failed task execution. A node in the DAG has failed after \
exhausting retries. Suggest an alternative execution path.

## Original Goal
{goal}

## Current Graph State
{graph_state}

## Failed Node
- ID: {failed_id}
- Action: {failed_action}
- Error: {failed_error}

## Available Skills
{skill_catalog}

Generate a replacement sub-graph (JSON "nodes" array) that achieves the same \
result via an alternative approach. The replacement nodes should have new IDs \
and may reference any completed upstream node results via ${{node_id.output}}.

Return ONLY the JSON object with "nodes" array.
"""


class GraphPlanner:
    """LLM-powered planner that generates TaskGraph from user goals.

    Responsibilities:
    - Constructs rich planning prompts with skill catalog
    - Parses structured JSON responses from LLM
    - Validates and auto-fixes generated graphs
    - Supports dynamic re-planning on node failure
    """

    def __init__(self, llm: LLMProvider, registry: SkillRegistry) -> None:
        self._llm = llm
        self._registry = registry

    # ═══ Public API ═══

    async def plan(
        self, user_goal: str, context: Optional[Dict[str, Any]] = None
    ) -> TaskGraph:
        """Generate a TaskGraph from a user goal.

        Steps:
        1. Build planning prompt (skill catalog + context + format spec)
        2. Call LLM to generate structured JSON plan
        3. Parse and validate the graph
        4. Return TaskGraph ready for execution

        Raises:
            ValueError: If LLM cannot produce a valid plan after auto-fix.
        """
        prompt = self._build_planning_prompt(user_goal, context)
        messages = [
            build_system_message(prompt),
            build_user_message_text(user_goal),
        ]

        resp = await self._llm.achat(messages, stream=False, enable_thinking=False)
        graph = self._parse_plan_response(resp.content or "", user_goal)

        errors = graph.validate()
        if errors:
            graph = self._auto_fix_graph(graph, errors)
            remaining_errors = graph.validate()
            if remaining_errors:
                raise ValueError(
                    f"Could not generate valid plan: {'; '.join(remaining_errors)}"
                )

        logger.info(
            "graph_planner.plan_generated goal=%s nodes=%d",
            user_goal[:50], len(graph.nodes),
        )
        return graph

    async def replan_on_failure(
        self, graph: TaskGraph, failed_node: TaskNode
    ) -> TaskGraph:
        """Dynamically adjust plan when a node fails.

        Strategies:
        1. If fallback exists → already handled by scheduler (no-op here)
        2. If downstream nodes exist → ask LLM for alternative path
        3. Return updated graph with replacement nodes

        Returns:
            Updated TaskGraph with alternative execution path,
            or the original graph unchanged if re-planning is not feasible.
        """
        if failed_node.retry_policy and failed_node.retry_policy.fallback_action:
            return graph  # Scheduler handles fallback

        downstream = graph.downstream_of(failed_node.id)
        if not downstream:
            return graph  # Terminal failure, nothing to replan

        prompt = _REPLAN_SYSTEM_PROMPT.format(
            goal=graph.goal,
            graph_state=graph.summary(),
            failed_id=failed_node.id,
            failed_action=failed_node.action,
            failed_error=failed_node.error or "Unknown",
            skill_catalog=self._registry.describe_with_params(),
        )
        messages = [
            build_system_message(prompt),
            build_user_message_text(
                f"Replan around failed node '{failed_node.id}' "
                f"to achieve: {graph.goal}"
            ),
        ]

        try:
            resp = await self._llm.achat(messages, stream=False, enable_thinking=False)
            replacement_nodes = self._parse_replacement_nodes(resp.content or "")
            if replacement_nodes:
                graph = self._apply_replacement(graph, failed_node.id, replacement_nodes)
                logger.info(
                    "graph_planner.replan_applied failed=%s new_nodes=%d",
                    failed_node.id, len(replacement_nodes),
                )
        except Exception as e:
            logger.warning("graph_planner.replan_failed error=%s", e)

        return graph

    # ═══ Prompt Construction ═══

    def _build_planning_prompt(
        self, user_goal: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """Construct the planning system prompt with skill catalog and context."""
        skill_catalog = self._registry.describe_with_params()
        if not skill_catalog.strip():
            skill_catalog = self._registry.describe()

        context_section = "No additional context."
        if context:
            context_section = json.dumps(context, ensure_ascii=False, indent=2)

        return _PLANNING_SYSTEM_PROMPT.format(
            skill_catalog=skill_catalog,
            context_section=context_section,
        )

    # ═══ Response Parsing ═══

    def _parse_plan_response(self, response: str, goal: str) -> TaskGraph:
        """Parse LLM response into TaskGraph.

        Extracts JSON from response (handles markdown code blocks),
        constructs TaskNode instances, builds TaskGraph with ordered insertion.
        """
        json_str = self._extract_json(response)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

        # Accept both "nodes" and "steps" keys for flexibility
        nodes_data = data.get("nodes") or data.get("steps") or []
        if not nodes_data:
            raise ValueError("LLM plan contains no nodes/steps")

        graph = TaskGraph(goal=goal)

        # First pass: create all nodes without dependency validation
        # (TaskGraph.add_node validates deps exist, so we insert in order)
        node_objects: List[TaskNode] = []
        for node_data in nodes_data:
            node = TaskNode(
                id=str(node_data.get("id", "")),
                name=node_data.get("name", node_data.get("id", "unnamed")),
                action=node_data.get("action", ""),
                action_type=node_data.get("action_type", "skill"),
                params=node_data.get("params") or {},
                depends_on=node_data.get("depends_on") or [],
                condition=node_data.get("condition"),
                retry_policy=self._parse_retry(node_data.get("retry")),
                timeout_seconds=node_data.get("timeout_seconds", 300.0),
            )
            node_objects.append(node)

        # Insert nodes directly (bypassing add_node's strict dep check for now)
        for node in node_objects:
            graph.nodes[node.id] = node

        return graph

    def _parse_replacement_nodes(self, response: str) -> List[TaskNode]:
        """Parse replacement nodes from replan LLM response."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            nodes_data = data.get("nodes") or data.get("steps") or []
            nodes: List[TaskNode] = []
            for nd in nodes_data:
                nodes.append(TaskNode(
                    id=str(nd.get("id", "")),
                    name=nd.get("name", nd.get("id", "unnamed")),
                    action=nd.get("action", ""),
                    action_type=nd.get("action_type", "skill"),
                    params=nd.get("params") or {},
                    depends_on=nd.get("depends_on") or [],
                    condition=nd.get("condition"),
                    retry_policy=self._parse_retry(nd.get("retry")),
                ))
            return nodes
        except Exception as e:
            logger.warning("graph_planner.parse_replacement_failed error=%s", e)
            return []

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from LLM response (handles ```json blocks)."""
        # Try markdown code block first
        match = _JSON_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()

        # Try to find raw JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return text[start: end + 1]

        raise ValueError("No JSON object found in LLM response")

    def _parse_retry(self, data: Optional[Dict[str, Any]]) -> Optional[RetryPolicy]:
        """Parse retry policy from JSON dict."""
        if not data:
            return None
        return RetryPolicy(
            max_retries=data.get("max_retries", 3),
            backoff_seconds=data.get("backoff_seconds", 1.0),
            backoff_multiplier=data.get("backoff_multiplier", 2.0),
            fallback_action=data.get("fallback_action"),
            retryable_errors=data.get("retryable_errors") or [],
        )

    # ═══ Auto-fix ═══

    def _auto_fix_graph(self, graph: TaskGraph, errors: List[str]) -> TaskGraph:
        """Attempt automatic fixes for common graph issues.

        Fixes:
        - Remove depends_on references to non-existent nodes
        - Remove self-dependencies
        """
        existing_ids = set(graph.nodes.keys())

        for node in graph.nodes.values():
            # Remove invalid dependency references
            node.depends_on = [
                dep_id for dep_id in node.depends_on
                if dep_id in existing_ids and dep_id != node.id
            ]

        return graph

    def _apply_replacement(
        self, graph: TaskGraph, failed_id: str, new_nodes: List[TaskNode]
    ) -> TaskGraph:
        """Apply replacement nodes to graph, rewiring dependencies.

        Downstream nodes that depended on the failed node will instead
        depend on the last new node.
        """
        downstream = graph.downstream_of(failed_id)
        new_ids = {n.id for n in new_nodes}

        # Insert new nodes (they may reference existing completed nodes)
        for node in new_nodes:
            # Filter depends_on to valid existing nodes + other new nodes
            valid_deps = [
                d for d in node.depends_on
                if d in graph.nodes or d in new_ids
            ]
            node.depends_on = valid_deps
            graph.nodes[node.id] = node

        # Rewire downstream: replace dependency on failed_id with last new node
        if new_nodes:
            last_new_id = new_nodes[-1].id
            for nid in downstream:
                node = graph.nodes[nid]
                if failed_id in node.depends_on:
                    node.depends_on = [
                        last_new_id if d == failed_id else d
                        for d in node.depends_on
                    ]
                # Reset downstream nodes so they can be re-executed
                if not node.is_terminal:
                    continue  # Already pending/ready
                graph.reset_node(nid)

        return graph
