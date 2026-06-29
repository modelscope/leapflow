"""ReAct-style tool-use executor for SKILL.md skills.

Gives the LLM access to real system tools (file ops, shell, UI) via
ExecutionPort. Each SKILL.md instruction is executed as a bounded
observe → reason → act loop.

Architecture:
    ToolDefinition → describes available tools for LLM prompt
    ToolCall       → parsed from LLM JSON output
    ToolBridge     → dispatches ToolCalls to ExecutionPort (SRP: only routing)
    ToolUseSkillExecutor → orchestrates the ReAct loop per instruction
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.engine.budget import BudgetConfig, BudgetStatus, IterationBudget
from leapflow.engine.context_compressor import CompressorConfig, ContextCompressor
from leapflow.engine.message_healer import MessageHealer

if TYPE_CHECKING:
    from leapflow.engine.confirmation import IOProvider
    from leapflow.llm.base import LLMProvider
    from leapflow.skills.action_policy import PolicyContext, PolicyEngine
    from leapflow.storage.bundle_writer import BundleContext

logger = logging.getLogger(__name__)

_DRIFT_THRESHOLD = 2
_MUTATING_SHELL_PREFIXES = (
    "open", "rm", "mv", "cp", "mkdir", "touch", "chmod", "chown",
    "kill", "killall", "pkill", "launchctl",
    "defaults write", "xattr",
    "brew", "pip", "npm",
    "curl", "wget",
    "dd", "diskutil", "hdiutil",
    "sed", "awk", "tee",
)
_SHELL_CHAIN_SPLIT = re.compile(r"\s*(?:&&|\|\||;)\s*")
_TOOL_CALL_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
)
_INLINE_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:', re.DOTALL
)


def _is_shell_mutating(command: str) -> bool:
    """Classify a shell command as state-mutating or readonly.

    Uses a closed set of mutating prefixes. Unknown commands default to
    readonly — correct for the open set of query tools (pgrep, ps, stat, …).
    Handles chained commands (&&, ||, ;): mutating if ANY segment mutates.
    """
    for segment in _SHELL_CHAIN_SPLIT.split(command.strip()):
        seg = segment.strip()
        if any(seg.startswith(p) for p in _MUTATING_SHELL_PREFIXES):
            return True
    return False


@dataclass(frozen=True)
class ToolDefinition:
    """Schema for one available tool — injected into the LLM system prompt.

    Traits:
        mutates_state: Tool changes observable state → clears dedup cache.
        counts_as_progress: Tool represents forward progress toward the goal
            → triggers completion HINT. Defaults to mutates_state.
            Set False for timing/polling tools (wait, wait_until_stable).
    """

    name: str
    description: str
    parameters: Dict[str, str]
    mutates_state: bool = False
    counts_as_progress: bool | None = None

    @property
    def is_progress(self) -> bool:
        if self.counts_as_progress is not None:
            return self.counts_as_progress
        return self.mutates_state


@dataclass(frozen=True)
class ToolCall:
    """Parsed tool invocation from LLM output."""

    name: str
    params: Dict[str, Any]


@dataclass
class StepOutput:
    """Result of executing one instruction step."""

    ok: bool
    result: str = ""
    error: str = ""
    tool_calls_made: int = 0
    goal_complete: bool = False


@runtime_checkable
class ExecutionPort(Protocol):
    """Minimal execution interface (matches vsi.ports.ExecutionPort)."""

    async def perform_file_op(self, op: str, params: Dict[str, Any]) -> Dict[str, Any]: ...
    async def exec_shell(self, command: str) -> Dict[str, Any]: ...
    async def launch_app(self, app_id: str) -> Dict[str, Any]: ...
    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════════════
# ToolBridge — dispatch layer
# ═══════════════════════════════════════════════════════════════════════


class _ToolHandler:
    """A single tool's definition + dispatch logic."""

    __slots__ = ("definition", "handler")

    def __init__(
        self,
        definition: ToolDefinition,
        handler: Any,
    ) -> None:
        self.definition = definition
        self.handler = handler


class ToolBridge:
    """Maps tool call names to ExecutionPort methods via a handler registry.

    Open/Closed: new tools can be added via register() without modifying dispatch.
    """

    def __init__(
        self,
        execution: ExecutionPort,
        *,
        policy: Optional["PolicyEngine"] = None,
        io: Optional["IOProvider"] = None,
    ) -> None:
        self._execution = execution
        self.policy = policy
        self.io = io
        self._handlers: Dict[str, _ToolHandler] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in tools derived from ExecutionPort capabilities."""
        ex = self._execution

        self._register(
            "file_list", "List files and directories at a path",
            {"path": "string (required) — directory path to list"},
            lambda p: ex.perform_file_op("list", p),
        )
        self._register(
            "file_move", "Move or rename a file/directory",
            {"source": "string (required) — source path",
             "destination": "string (required) — destination path"},
            lambda p: ex.perform_file_op("move", p),
            mutates_state=True,
        )
        self._register(
            "file_copy", "Copy a file/directory",
            {"source": "string (required) — source path",
             "destination": "string (required) — destination path"},
            lambda p: ex.perform_file_op("copy", p),
            mutates_state=True,
        )
        self._register(
            "file_delete", "Delete a file or directory",
            {"path": "string (required) — path to delete"},
            lambda p: ex.perform_file_op("delete", p),
            mutates_state=True,
        )
        self._register(
            "mkdir", "Create a directory (including parent directories)",
            {"path": "string (required) — directory path to create"},
            lambda p: ex.exec_shell(f"mkdir -p {shlex.quote(p.get('path', ''))}"),
            mutates_state=True,
        )
        self._register(
            "shell", "Run a shell command and return stdout/stderr",
            {"command": "string (required) — shell command to execute"},
            lambda p: ex.exec_shell(p.get("command", "")),
        )
        self._register(
            "launch_app", "Launch an application by bundle ID",
            {"app_id": "string (required) — application bundle ID"},
            lambda p: ex.launch_app(p.get("app_id", "")),
            mutates_state=True,
        )
        self._register(
            "ui_action", "Perform a UI action on an accessibility element",
            {"node_id": "string (required) — target UI element ID",
             "action": "string (required) — action name (e.g. AXPress)"},
            lambda p: ex.perform_ui_action(
                p.get("node_id", ""), p.get("action", ""),
                {k: v for k, v in p.items() if k not in ("node_id", "action")} or None,
            ),
            mutates_state=True,
        )
        self._register(
            "done", "Signal that the current instruction is complete",
            {"result": "string — summary of what was accomplished"},
            None,
        )

    def _register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        handler: Any,
        *,
        mutates_state: bool = False,
        counts_as_progress: bool | None = None,
    ) -> None:
        defn = ToolDefinition(
            name=name, description=description, parameters=parameters,
            mutates_state=mutates_state, counts_as_progress=counts_as_progress,
        )
        self._handlers[name] = _ToolHandler(definition=defn, handler=handler)

    def register(
        self, name: str, description: str, parameters: Dict[str, str], handler: Any,
        *, mutates_state: bool = False, counts_as_progress: bool | None = None,
    ) -> None:
        """Register a custom tool (extensibility point)."""
        self._register(
            name, description, parameters, handler,
            mutates_state=mutates_state, counts_as_progress=counts_as_progress,
        )

    def is_mutating(self, name: str) -> bool:
        """Check if a tool is declared as state-mutating (clears dedup cache)."""
        entry = self._handlers.get(name)
        return entry is not None and entry.definition.mutates_state

    def is_progress(self, name: str) -> bool:
        """Check if a tool counts as forward progress toward the goal."""
        entry = self._handlers.get(name)
        return entry is not None and entry.definition.is_progress

    async def dispatch(self, call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call and return the raw result dict."""
        entry = self._handlers.get(call.name)
        if entry is None or entry.handler is None:
            return {"ok": False, "error": f"unknown_tool: {call.name}"}
        return await entry.handler(call.params)

    async def dispatch_guarded(
        self, call: ToolCall, context: "PolicyContext"
    ) -> Dict[str, Any]:
        """Dispatch with policy gate. Falls back to direct dispatch when no policy."""
        if self.policy is None:
            return await self.dispatch(call)

        from leapflow.skills.action_policy import Verdict

        decision = await self.policy.evaluate(call, context)

        if decision.verdict == Verdict.ALLOW:
            return await self.dispatch(call)

        if decision.verdict == Verdict.DENY:
            return {"ok": False, "error": f"policy_denied: {decision.reason}"}

        # ASK: prompt user for confirmation
        if self.io is None:
            return await self.dispatch(call)

        summary = decision.summary or f"{call.name}({call.params})"
        await self.io.display(f"\n  [Policy] {summary}")
        response = await self.io.prompt("  Allow? (yes/no) ")
        if response.strip().lower() in ("y", "yes", "确认", "好"):
            return await self.dispatch(call)
        return {"ok": False, "error": "user_denied", "reason": decision.reason}

    def tool_definitions(self) -> List[ToolDefinition]:
        """Return all available tool schemas for prompt injection."""
        return [h.definition for h in self._handlers.values()]


# ═══════════════════════════════════════════════════════════════════════
# Early Stop — signal detection & budget enforcement
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _EarlyStopState:
    """Per-step state for early stop signal tracking."""

    consecutive_readonly: int = 0
    hint_sent: bool = False
    call_cache: Dict[str, Any] = field(default_factory=dict)
    tool_calls_budget: int = 6


def _check_goal_likely_met(
    tool_call: ToolCall, result: Dict[str, Any], *, is_mutating: bool = False,
) -> bool:
    """P0 Layer 1: Structural heuristic — successful state-changing action."""
    if not result.get("ok", False):
        return False
    if tool_call.name == "shell":
        return _is_shell_mutating(tool_call.params.get("command", ""))
    return is_mutating


def _check_exploration_drift(
    tool_call: ToolCall, state: _EarlyStopState, *, is_mutating: bool = False,
) -> bool:
    """P0 Layer 1: Detect consecutive non-mutating calls indicating drift."""
    if is_mutating:
        state.consecutive_readonly = 0
    elif tool_call.name == "shell":
        if _is_shell_mutating(tool_call.params.get("command", "")):
            state.consecutive_readonly = 0
        else:
            state.consecutive_readonly += 1
    else:
        state.consecutive_readonly += 1
    return state.consecutive_readonly >= _DRIFT_THRESHOLD


def _make_call_key(call: ToolCall) -> str:
    """P2: Deterministic key for duplicate detection."""
    return f"{call.name}:{json.dumps(call.params, sort_keys=True)}"


def _demote_stderr(result: Dict[str, Any]) -> Dict[str, Any]:
    """P2: If shell exit_code=0 but stderr is non-empty, prefix with (info)."""
    if (
        result.get("ok")
        and result.get("exit_code") == 0
        and result.get("stderr", "").strip()
    ):
        result = dict(result)
        result["stderr"] = f"(info) {result['stderr']}"
    return result


# ═══════════════════════════════════════════════════════════════════════
# ToolUseSkillExecutor — ReAct loop
# ═══════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_TEMPLATE = """\
You are a desktop automation agent executing a learned skill.
You MUST use tools to perform actual operations — do NOT just describe what you would do.

Available tools (call ONE at a time by outputting a JSON code block):
{tool_definitions}

To call a tool, output EXACTLY one JSON code block in this format:
```json
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
```

IMPORTANT: Use EXACTLY this format. The JSON must have "name" (string) and "arguments" (object) keys.
Example:
```json
{{"name": "shell", "arguments": {{"command": "ls -la /tmp"}}}}
```

When done with the instruction, call:
```json
{{"name": "done", "arguments": {{"result": "summary of what was accomplished"}}}}
```

Execution rules:
- Shell commands are STATELESS: each "shell" call starts fresh. Use absolute paths or chain with &&.
- Do NOT call "launch_app" for Terminal — use the "shell" tool directly.
- Use selectors EXACTLY as returned by observe_ui. Never construct or guess selectors.
- After a successful action, check the tool result before calling observe_ui again — the result often includes the new UI state.
- Use screenshot only when observe_ui cannot provide the information you need.

Failure recovery:
- If click fails, switch to keyboard-based interaction: use shortcut (Tab, Enter, arrow keys) or type_text.
- If an element is not found, call observe_ui to refresh your view of the current UI.
- If stuck after 2 failed attempts with the same approach, try a completely different strategy.

After the tool executes, you will receive its result. Then call the next tool or signal completion.

Skill context:
{skill_content}
"""


class ToolUseSkillExecutor:
    """Executes SKILL.md instructions via bounded ReAct loop with real tools."""

    def __init__(
        self,
        llm: Any,
        bridge: ToolBridge,
        skill_content: str,
        instructions: List[str],
        *,
        vlm: Optional["LLMProvider"] = None,
        bundle_context: Optional["BundleContext"] = None,
        skill_name: str = "",
        budget_config: Optional[BudgetConfig] = None,
        compressor_config: Optional[CompressorConfig] = None,
        step_timeout_s: float = 30.0,
    ) -> None:
        self._llm = llm
        self._vlm = vlm
        self._bridge = bridge
        self._skill_content = skill_content
        self._instructions = instructions
        self._bundle_context = bundle_context
        self._skill_name = skill_name
        self._budget_config = budget_config or BudgetConfig(
            max_iterations=30, soft_limit=24, warning_threshold=20,
        )
        self._compressor = ContextCompressor(compressor_config or CompressorConfig())
        self._healer = MessageHealer()
        self._step_timeout_s = step_timeout_s

    async def run(
        self,
        *,
        user_goal: str = "",
        instruction_idx: Optional[int] = None,
        _policy: Optional["PolicyEngine"] = None,
        _io: Optional["IOProvider"] = None,
        **params: Any,
    ) -> str:
        """Execute skill instructions and return result summary.

        Args:
            user_goal: High-level user intent
            instruction_idx: If provided, execute only this step (0-based)
            _policy: Optional runtime policy engine (set by session for HITL)
            _io: Optional IOProvider for user prompts
            **params: Skill parameters (target_directory, etc.)
        """
        if _policy is not None:
            self._bridge.policy = _policy
        if _io is not None:
            self._bridge.io = _io

        if instruction_idx is not None:
            if 0 <= instruction_idx < len(self._instructions):
                step = self._instructions[instruction_idx]
                output = await self._execute_instruction(
                    step, params, user_goal, step_num=instruction_idx + 1,
                )
                return self._format_output(output)
            return f"Invalid instruction index: {instruction_idx}"

        import sys as _sys
        _DIM = "\033[2m"
        _RESET = "\033[0m"
        total = len(self._instructions)

        results: List[StepOutput] = []
        for step_idx, step in enumerate(self._instructions):
            _sys.stderr.write(
                f"{_DIM}→ Step [{step_idx + 1}/{total}]: {step[:80]}{_RESET}\n"
            )
            _sys.stderr.flush()
            output = await self._execute_instruction(
                step, params, user_goal, step_num=step_idx + 1,
            )
            results.append(output)
            if not output.ok:
                _sys.stderr.write(f"{_DIM}  ✗ Step failed: {output.error}{_RESET}\n")
                _sys.stderr.flush()
                break
            _sys.stderr.write(f"{_DIM}  ✓ Done{_RESET}\n")
            _sys.stderr.flush()
            if output.goal_complete:
                _sys.stderr.write(
                    f"{_DIM}  ▶ Skill goal fully achieved at step {step_idx + 1}/{total}, "
                    f"skipping remaining steps{_RESET}\n"
                )
                _sys.stderr.flush()
                break
            await self._run_post_step_verification(step_idx + 1)

        return self._format_results(results)

    async def _execute_instruction(
        self, instruction: str, params: Dict[str, Any], user_goal: str,
        *, step_num: int = 0,
    ) -> StepOutput:
        """Run a bounded ReAct loop for one instruction with early stop."""
        import sys
        import time as _time

        from leapflow.utils.stream_progress import StreamProgressWriter
        from leapflow.llm.message_builder import (
            build_assistant_message,
            build_system_message,
            build_user_message_text,
        )

        _DIM = "\033[2m"
        _RESET = "\033[0m"

        tool_defs = self._bridge.tool_definitions()
        tool_defs_text = json.dumps(
            [{"name": t.name, "description": t.description, "parameters": t.parameters}
             for t in tool_defs],
            indent=2,
        )

        bundle_section = self._build_bundle_section(step_num=step_num)
        system = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_definitions=tool_defs_text,
            skill_content=self._skill_content,
        )
        if bundle_section:
            system += f"\n{bundle_section}"

        params_text = "\n".join(f"  {k} = {v}" for k, v in params.items()) if params else "(none)"
        user_msg = (
            f"Goal: {user_goal}\n"
            f"Parameters:\n{params_text}\n\n"
            f"Execute this instruction:\n{instruction}"
        )

        messages = [
            build_system_message(system),
            build_user_message_text(user_msg),
        ]

        tool_calls_made = 0
        stop_state = _EarlyStopState(
            tool_calls_budget=self._budget_config.max_iterations,
        )
        step_start = _time.monotonic()
        budget = IterationBudget.for_tool_execution(
            max_calls=self._budget_config.max_iterations,
            soft=self._budget_config.soft_limit,
        )

        while not budget.exhausted:
            # P1: Step timeout check
            elapsed = _time.monotonic() - step_start
            if elapsed > self._step_timeout_s:
                sys.stderr.write(f"{_DIM}  ⏱ Step timeout ({self._step_timeout_s:.0f}s){_RESET}\n")
                sys.stderr.flush()
                return StepOutput(
                    ok=True,
                    result="step_timeout: partial progress",
                    tool_calls_made=tool_calls_made,
                )

            # Consume budget tick
            status = budget.consume()
            if status == BudgetStatus.EXHAUSTED:
                break
            if status == BudgetStatus.SOFT_LIMIT:
                messages.append(build_user_message_text(
                    "SYSTEM: Tool call budget nearly exhausted. "
                    "Call `done` now with your best result."
                ))

            # Heal + compress messages
            messages = self._healer.heal(messages)
            messages = self._compressor.compress(messages)
            writer = StreamProgressWriter(prefix="  │ ")
            try:
                response = await asyncio.wait_for(
                    self._llm.achat(
                        messages, stream=True, enable_thinking=False,
                        on_chunk=writer,
                    ),
                    timeout=self._step_timeout_s,
                )
            except asyncio.TimeoutError:
                writer.finish()
                return StepOutput(
                    ok=False,
                    error=f"LLM response timeout ({self._step_timeout_s:.0f}s)",
                    tool_calls_made=tool_calls_made,
                )
            except Exception as e:
                return StepOutput(ok=False, error=f"LLM call failed: {e}", tool_calls_made=tool_calls_made)
            finally:
                writer.finish()

            content = response.content or ""

            # Repetition detection — abort if LLM is stuck
            if _detect_repetition(content):
                return StepOutput(
                    ok=False,
                    error="LLM response stuck in repetitive pattern",
                    tool_calls_made=tool_calls_made,
                )

            tool_call = _parse_tool_call(content)

            if tool_call is None:
                return StepOutput(ok=True, result=content, tool_calls_made=tool_calls_made)

            if tool_call.name == "done":
                return StepOutput(
                    ok=True,
                    result=tool_call.params.get("result", "completed"),
                    tool_calls_made=tool_calls_made,
                )

            # P1: Budget exhausted — force final iteration
            if tool_calls_made >= stop_state.tool_calls_budget:
                sys.stderr.write(f"{_DIM}  ⚡ Budget exhausted ({stop_state.tool_calls_budget} calls){_RESET}\n")
                sys.stderr.flush()
                messages.append(build_assistant_message(content))
                messages.append(build_user_message_text(
                    "SYSTEM: Tool call budget exhausted. Call `done` now with your best result summary."
                ))
                continue

            # P2: Duplicate call detection
            call_key = _make_call_key(tool_call)
            if call_key in stop_state.call_cache:
                cached = stop_state.call_cache[call_key]
                sys.stderr.write(f"{_DIM}  │  ↺ duplicate (cached){_RESET}\n")
                sys.stderr.flush()
                messages.append(build_assistant_message(content))
                messages.append(build_user_message_text(
                    f"Tool result ({tool_call.name}) [cached - identical call]:\n"
                    f"```json\n{json.dumps(cached, default=str)}\n```\n"
                    "NOTE: You already made this exact call. Proceed to the next action or call `done`."
                ))
                continue

            sys.stderr.write(
                f"{_DIM}  ├─ tool: {tool_call.name}"
                f"({', '.join(f'{k}={v!r}' for k, v in tool_call.params.items())}){_RESET}\n"
            )
            sys.stderr.flush()

            try:
                from leapflow.skills.action_policy import PolicyContext

                _ctx = PolicyContext(
                    skill_name=self._skill_name,
                    iteration=budget.used,
                    history=[],
                )
                result = await self._dispatch_with_retry(tool_call, _ctx)
                tool_calls_made += 1
            except Exception as e:
                result = {"ok": False, "error": str(e)}

            # P2: Stderr demotion for successful shell calls
            if tool_call.name == "shell":
                result = _demote_stderr(result)

            # VLM: enrich screenshot results with visual description
            if tool_call.name == "screenshot":
                result = await self._process_screenshot(result)

            _is_mut = self._bridge.is_mutating(tool_call.name)
            _is_progress = self._bridge.is_progress(tool_call.name)

            # P2: Cache result for duplicate detection
            if _is_mut:
                stop_state.call_cache.clear()
            else:
                stop_state.call_cache[call_key] = result

            ok_str = "ok" if result.get("ok", True) else "err"
            sys.stderr.write(f"{_DIM}  │  → {ok_str}{_RESET}\n")
            sys.stderr.flush()

            logger.debug(
                "tool_executor.call tool=%s result_ok=%s",
                tool_call.name, result.get("ok", "?"),
            )

            # Signal detection — _is_progress for goal HINT,
            # _is_mut for drift reset (wait tools reset drift but don't count as progress)
            goal_met = _check_goal_likely_met(tool_call, result, is_mutating=_is_progress)
            drifting = _check_exploration_drift(tool_call, stop_state, is_mutating=_is_mut)

            result_text = (
                f"Tool result ({tool_call.name}):\n```json\n{json.dumps(result, default=str)}\n```"
            )
            if goal_met and not stop_state.hint_sent:
                stop_state.hint_sent = True
                result_text += (
                    "\n\nHINT: The operation succeeded and likely fulfills the goal. "
                    "If complete, call `done` with a result summary."
                )
            elif drifting:
                result_text += (
                    "\n\nHINT: Multiple consecutive read-only operations detected. "
                    "Focus on completing the task — call `done` if the goal is met, "
                    "or take a concrete action."
                )

            messages.append(build_assistant_message(content))
            messages.append(build_user_message_text(result_text))

        return StepOutput(
            ok=False,
            error="max_iterations_exceeded",
            tool_calls_made=tool_calls_made,
        )

    async def _process_screenshot(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Strip base64 from screenshot result, add VLM description if available."""
        base64_data = result.get("base64", "")
        result = {k: v for k, v in result.items() if k != "base64"}
        if base64_data and self._vlm:
            description = await self._describe_screenshot(base64_data)
            if description:
                result["visual_description"] = description
        return result

    async def _describe_screenshot(self, base64_data: str) -> str:
        """Use VLM to produce a text description of a screenshot."""
        from leapflow.llm.message_builder import build_user_message_multimodal

        try:
            msg = build_user_message_multimodal(
                "Describe the current UI state in this screenshot. "
                "Focus on: app name, visible interactive elements (buttons, text fields, menus), "
                "their labels, and the overall layout. Be concise (2-3 sentences).",
                images_base64=[base64_data],
                image_mime="image/jpeg",
            )
            resp = await self._vlm.achat(
                [msg], stream=False, enable_thinking=False, max_tokens=200,
            )
            return resp.content or ""
        except Exception:
            logger.debug("vlm_describe_screenshot_failed", exc_info=True)
            return ""

    async def _dispatch_with_retry(
        self, call: "ToolCall", ctx: Any, *, max_attempts: int = 2, retry_delay: float = 0.5,
    ) -> Dict[str, Any]:
        """Dispatch a tool call with one retry on transient connection errors."""
        for attempt in range(max_attempts):
            try:
                return await self._bridge.dispatch_guarded(call, ctx)
            except (asyncio.TimeoutError, OSError, ConnectionError) as e:
                if attempt < max_attempts - 1:
                    logger.debug("tool_call_retry tool=%s attempt=%d error=%s", call.name, attempt + 1, e)
                    await asyncio.sleep(retry_delay)
                else:
                    return {"ok": False, "error": f"connection_failed: {e}"}
        return {"ok": False, "error": "unreachable"}

    def _build_bundle_section(self, *, step_num: int = 0) -> str:
        """Build additional prompt context from bundle files.

        When step_num > 0, filters anchors to only include those relevant
        to the current step to prevent context bloat.
        """
        ctx = self._bundle_context
        if ctx is None or not ctx.has_bundle:
            return ""

        parts: List[str] = []
        if ctx.anchors_yaml:
            anchors_text = self._filter_anchors_for_step(ctx.anchors_yaml, step_num)
            if anchors_text:
                parts.append(f"UI Anchors (use these to locate elements):\n{anchors_text}")
        if ctx.recovery_scripts:
            script_names = [p.name for p in ctx.recovery_scripts]
            parts.append(
                "Recovery scripts available (run via shell tool):\n"
                + "\n".join(f"  - {n}" for n in script_names)
            )
        if ctx.verification_tests:
            test_names = [p.name for p in ctx.verification_tests]
            parts.append(
                "Verification tests (run after completing each step):\n"
                + "\n".join(f"  - {n}" for n in test_names)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _filter_anchors_for_step(anchors_yaml: str, step_num: int) -> str:
        """Filter anchors YAML to only include entries relevant to a step.

        Returns the full YAML if step_num is 0 (no filtering) or if
        parsing fails.
        """
        if step_num <= 0:
            return anchors_yaml
        try:
            import yaml
            data = yaml.safe_load(anchors_yaml)
            if not isinstance(data, dict) or "anchors" not in data:
                return anchors_yaml
            all_anchors = data["anchors"]
            if not isinstance(all_anchors, dict):
                return anchors_yaml
            filtered = {
                k: v for k, v in all_anchors.items()
                if not isinstance(v, dict) or v.get("step") in (step_num, None)
            }
            if not filtered:
                return ""
            return yaml.dump(
                {"anchors": filtered},
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        except Exception:
            return anchors_yaml

    async def _run_post_step_verification(self, step_num: int) -> None:
        """Run verification test for a completed step if one exists.

        Verification is informational — failures are logged but don't block execution.
        """
        ctx = self._bundle_context
        if ctx is None or not ctx.verification_tests:
            return

        target_name = f"verify_step_{step_num}"
        script = next(
            (p for p in ctx.verification_tests if target_name in p.name), None
        )
        if script is None:
            return

        try:
            result = await self._bridge.dispatch(
                ToolCall(name="shell", params={"command": str(script)})
            )
            ok = result.get("ok", True)
            status = "passed" if ok else "FAILED"
            logger.info("post_step_verify step=%d script=%s result=%s", step_num, script.name, status)
            if not ok:
                logger.warning(
                    "Verification failed for step %d: %s",
                    step_num, result.get("stderr", result.get("error", "")),
                )
        except Exception:
            logger.debug("post_step_verify step=%d error", step_num, exc_info=True)

    def _format_output(self, output: StepOutput) -> str:
        if output.ok:
            return output.result
        return f"Error: {output.error}"

    def _format_results(self, results: List[StepOutput]) -> str:
        parts: List[str] = []
        total_tools = sum(r.tool_calls_made for r in results)
        ok_count = sum(1 for r in results if r.ok)
        parts.append(f"Executed {ok_count}/{len(results)} steps ({total_tools} tool calls)")
        for i, r in enumerate(results, 1):
            status = "OK" if r.ok else "FAILED"
            detail = r.result if r.ok else r.error
            parts.append(f"  Step {i}: [{status}] {detail}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Tool call parsing
# ═══════════════════════════════════════════════════════════════════════


def _detect_repetition(content: str, threshold: int = 10) -> bool:
    """Detect if LLM output is stuck in a repetitive pattern."""
    if len(content) < 100:
        return False
    # Check for repeated closing tags (common failure mode)
    repeated_patterns = ["</invoke>", "</tool_call>", "```\n```"]
    for pattern in repeated_patterns:
        if content.count(pattern) >= threshold:
            return True
    # Check last 200 chars for character-level repetition
    tail = content[-200:]
    if len(set(tail.split())) <= 3 and len(tail) > 50:
        return True
    return False


def _parse_tool_call(content: str) -> Optional[ToolCall]:
    """Extract a tool call JSON from LLM response text.

    Supports:
    - ```json {"name": ..., "arguments": {...}} ``` (primary)
    - Inline {"name": ...} patterns
    - <tool_call>{"name": ..., "arguments": {...}}</tool_call> patterns
    - Legacy {"tool": ..., "params": {...}} format
    """
    # 1. Standard markdown code block
    match = _TOOL_CALL_PATTERN.search(content)
    if match:
        result = _try_parse_json(match.group(1))
        if result:
            return result

    # 2. <tool_call> XML-style wrapper
    tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|</invoke>)', content, re.DOTALL)
    if tc_match:
        result = _try_parse_json(tc_match.group(1))
        if result:
            return result

    # 3. Inline JSON with "name" or "tool" key
    idx = -1
    for pattern_str in ['"name"', '"tool"']:
        search = content.find('{')
        while search != -1:
            # Check if this { starts a valid tool call JSON
            if pattern_str in content[search:search + 50]:
                idx = search
                break
            search = content.find('{', search + 1)
        if idx != -1:
            break

    if idx == -1:
        match2 = _INLINE_JSON_PATTERN.search(content)
        if match2:
            idx = match2.start()

    if idx != -1:
        depth = 0
        end = idx
        for i in range(idx, min(len(content), idx + 2000)):  # limit scan to 2000 chars
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth == 0:
            return _try_parse_json(content[idx:end])

    return None


def _try_parse_json(text: str) -> Optional[ToolCall]:
    """Parse a JSON string into a ToolCall (OpenAI function calling format)."""
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        # Primary: OpenAI function calling format {"name": ..., "arguments": {...}}
        if "name" in data:
            name = data["name"]
            params = data.get("arguments", data.get("params", data.get("parameters", {})))
            if isinstance(params, str):
                # Sometimes arguments is a JSON string
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {"raw": params}
            return ToolCall(name=str(name), params=params if isinstance(params, dict) else {})

        # Fallback: legacy {"tool": ..., "params": {...}} format
        if "tool" in data:
            return ToolCall(
                name=data["tool"],
                params=data.get("params", data.get("arguments", data.get("parameters", {}))),
            )

    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None
