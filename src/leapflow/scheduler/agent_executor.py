# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent-mode skill executor for the scheduler — runs an LLM tool loop in isolation.

Design:
- Reuses the ``DefaultSubagentExecutor`` pattern from ``engine.subagent`` for
  isolated, bounded LLM→tool execution.  This avoids duplicating the loop and
  shares the same governance / budget machinery.
- Adapts the scheduler ``SkillExecutor`` Protocol (``execute(skill_name, parameters)``)
  to the subagent interface (``SubagentConfig → SubagentResult``).
- All exceptions are contained: the caller always gets a ``dict`` result with
  ``ok`` and ``output`` / ``error`` — a scheduler tick must NEVER crash.
- Config-driven: iteration budget and tool blocklist come from Settings; no
  hardcoded limits.
- Cold-path only: scheduler ticks are infrequent, so construction cost is
  acceptable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List

logger = logging.getLogger(__name__)


class AgentSkillExecutor:
    """SkillExecutor that runs an isolated LLM agent loop to complete a task.

    Satisfies the ``SkillExecutor`` Protocol declared in
    ``leapflow.scheduler.types`` (structural subtyping via ``execute``).

    Constructor dependencies mirror ``DefaultSubagentExecutor`` from
    ``engine.subagent``: an LLM client, tool handlers/definitions, and
    settings.  These are injected by the coordinator at construction time.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tool_handlers: Dict[str, Any],
        tool_definitions: List[dict],
        settings: Any = None,
    ) -> None:
        self._llm = llm
        self._tool_handlers = dict(tool_handlers)
        self._tool_definitions = list(tool_definitions)
        self._settings = settings

    # ------------------------------------------------------------------
    # SkillExecutor Protocol
    # ------------------------------------------------------------------

    async def execute(self, skill_name: str, parameters: dict) -> dict:
        """Execute a skill by running an isolated agent loop.

        Parameters
        ----------
        skill_name:
            Name of the skill (used in the system prompt framing).
        parameters:
            Must contain ``instruction`` (str).  Optional keys:
            - ``tool_blocklist``: comma-separated tool names to block (overrides
              the ``scheduler_agent_tool_blocklist`` setting).
            - ``context``: additional context string for the agent.

        Returns
        -------
        dict with ``ok`` (bool), ``output`` (str), and on failure ``error`` (str).
        """
        try:
            return await self._execute_inner(skill_name, parameters)
        except Exception as exc:
            logger.error(
                "AgentSkillExecutor caught unhandled error for skill=%s: %s",
                skill_name, exc, exc_info=True,
            )
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute_inner(self, skill_name: str, parameters: dict) -> dict:
        """Construct and run the subagent; may raise."""
        # Lazy imports keep the module importable without engine dependencies.
        from leapflow.engine.subagent import (
            DefaultSubagentExecutor,
            SubagentConfig,
            SubagentManager,
        )

        instruction = parameters.get("instruction", "")
        if not instruction:
            return {"ok": False, "error": "Missing 'instruction' in task parameters."}

        context = parameters.get("context", "")

        # Resolve iteration budget from settings.
        max_iterations = 25
        if self._settings is not None:
            max_iterations = getattr(
                self._settings, "scheduler_agent_max_iterations", 25,
            )

        # Resolve tool blocklist: payload override > settings.
        blocklist_raw = parameters.get("tool_blocklist", "")
        if not blocklist_raw and self._settings is not None:
            blocklist_raw = getattr(
                self._settings, "scheduler_agent_tool_blocklist", "",
            )
        blocked_tools: FrozenSet[str] = frozenset(
            name.strip() for name in str(blocklist_raw).split(",") if name.strip()
        )

        # Build the concrete executor (same pattern as engine wiring).
        # Attempt to inject the shared tool pipeline for approval gating.
        # If unavailable, the executor degrades fail-closed (read_only only).
        tool_pipeline = None
        try:
            from leapflow.plugins import get_registry
            tool_pipeline = get_registry().tool_pipeline
        except Exception:
            logger.debug("scheduler: tool_pipeline unavailable, fail-closed mode")
        subagent_executor = DefaultSubagentExecutor(
            llm=self._llm,
            tool_handlers=self._tool_handlers,
            tool_definitions=self._tool_definitions,
            settings=self._settings,
            tool_pipeline=tool_pipeline,
        )

        # Wrap with the manager for lifecycle, depth-gating, and trimming.
        manager = SubagentManager(executor=subagent_executor, max_depth=1)

        config = SubagentConfig(
            goal=instruction,
            context=context,
            blocked_tools=blocked_tools,
            max_iterations=max_iterations,
            depth=0,
        )

        result = await manager.delegate(config)

        # Build tool summary from the subagent result.
        tool_summary = f"tool_calls={result.tool_calls}"
        output_text = result.summary or "(no output)"
        if result.tool_calls > 0:
            output_text = f"{output_text}\n\n[{tool_summary}]"

        if result.status == "completed":
            return {"ok": True, "output": output_text}

        return {
            "ok": False,
            "output": output_text,
            "error": result.error or result.status,
            "context": f"{skill_name}: {instruction[:80]}",
        }
