"""Task planning utilities.

Contains:
- propose_subtasks: Legacy flat-list planner (deprecated, retained for backward compat)
- propose_task_graph: DAG-based planner via GraphPlanner
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text

if TYPE_CHECKING:
    from leapflow.engine.graph_planner import GraphPlanner
    from leapflow.engine.task_graph import TaskGraph

logger = logging.getLogger(__name__)


async def propose_subtasks(llm: LLMProvider, user_goal: str, skills_catalog: str) -> List[str]:
    """Ask the model for 3-7 concrete steps (best-effort).

    .. deprecated::
        Use :func:`propose_task_graph` or :class:`GraphPlanner.plan()` for
        DAG-based planning with dependency ordering, parallelism, and retry.
    """
    warnings.warn(
        "propose_subtasks is deprecated, use GraphPlanner.plan() for DAG-based planning",
        DeprecationWarning,
        stacklevel=2,
    )

    messages = [
        build_system_message(
            "Return STRICT JSON: {\"steps\":[\"...\", ...]} with 3-7 steps for the goal. "
            f"Available skills:\n{skills_catalog}"
        ),
        build_user_message_text(user_goal),
    ]
    try:
        resp = await llm.achat(messages, stream=False, enable_thinking=False)
        raw = (resp.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        blob = raw[start : end + 1] if start != -1 and end != -1 else raw
        data = json.loads(blob)
        steps = [str(x) for x in list(data.get("steps") or [])]
        return steps[:10]
    except Exception:
        logger.debug("planner failed", exc_info=True)
        return [user_goal]


async def propose_task_graph(
    planner: "GraphPlanner",
    user_goal: str,
    context: Optional[Dict] = None,
) -> "TaskGraph":
    """Generate a TaskGraph for complex multi-step goals.

    Modern replacement for :func:`propose_subtasks`, supporting:
    - Dependency-based ordering (DAG)
    - Parallel execution of independent branches
    - Conditional branches
    - Retry policies with exponential backoff

    Args:
        planner: A configured GraphPlanner instance.
        user_goal: Natural language description of the user's objective.
        context: Optional context dict (recent observations, env state, etc.)

    Returns:
        A validated TaskGraph ready for execution by TaskScheduler.
    """
    return await planner.plan(user_goal, context)
