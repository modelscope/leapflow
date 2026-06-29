"""Run subcommand — execute a skill by trigger match or explicit name."""

from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional

from leapflow.cli.helpers import require_initialized
from leapflow.engine.situational_assessor import Assessment, AssessmentVerdict

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.skills.registry import Skill

logger = logging.getLogger(__name__)

_PARAM_PATH_RE = re.compile(r"(?:/[\w.@~-]+)+(?:/[\w.@~-]*)*|~(?:/[\w.@~-]+)+")


async def _extract_params_from_prompt(
    prompt: str,
    skill: "Skill",
    llm: Any,
) -> Dict[str, Any]:
    """Extract declared skill parameters from a natural-language prompt."""
    params = skill.parameters
    if not params:
        return {}

    if llm is not None:
        extracted = await _llm_extract_params(prompt, skill, params, llm)
        if extracted:
            return extracted

    return _heuristic_extract_params(prompt, params)


async def _llm_extract_params(
    prompt: str, skill: "Skill", params: list, llm: Any,
) -> Dict[str, Any]:
    """Semantic parameter extraction via LLM (streamed)."""
    try:
        from leapflow.llm.message_builder import build_system_message, build_user_message_text
        from leapflow.utils.stream_progress import StreamProgressWriter

        param_desc = "\n".join(
            f"- {p.name} ({p.type}, {'required' if p.required else 'optional'}): "
            f"{p.description or p.name}"
            for p in params
        )
        writer = StreamProgressWriter(prefix="  │ ")
        resp = await llm.achat(
            [
                build_system_message(
                    "Extract parameter values from the user's request. "
                    "Return ONLY a JSON object mapping parameter names to values. "
                    "If a value cannot be determined, omit it."
                ),
                build_user_message_text(
                    f"User request: {prompt}\n\n"
                    f"Skill: {skill.name} — {skill.description}\n\n"
                    f"Parameters to extract:\n{param_desc}"
                ),
            ],
            stream=True,
            enable_thinking=False,
            on_chunk=writer,
        )
        writer.finish()
        raw = resp.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            import json as _json
            data = _json.loads(raw[start : end + 1])
            return {
                p.name: data[p.name]
                for p in params
                if p.name in data and data[p.name] is not None
            }
    except Exception:
        logger.debug("param_extraction.llm_failed", exc_info=True)
    return {}


def _heuristic_extract_params(
    prompt: str, params: list,
) -> Dict[str, Any]:
    """Regex-based fallback for path/number extraction when LLM is unavailable."""
    extracted: Dict[str, Any] = {}
    path_matches = _PARAM_PATH_RE.findall(prompt)

    for p in params:
        if p.type in ("path", "str") and path_matches:
            extracted[p.name] = path_matches.pop(0)

    return extracted


_VERDICT_LABELS = {
    AssessmentVerdict.ALREADY_DONE: ("ALREADY DONE", "\033[33m"),
    AssessmentVerdict.BLOCKED: ("BLOCKED", "\033[31m"),
    AssessmentVerdict.RISKY: ("CAUTION", "\033[33m"),
}


async def _run_assessment(
    ctx: "Context", prompt: str, skill: "Skill", params: Dict[str, Any],
) -> bool:
    """Run situational assessment. Returns True if execution should proceed."""
    if not hasattr(ctx, "assessor") or ctx.assessor is None:
        return True

    sys.stderr.write("\033[2m→ Assessing situation...\033[0m\n")
    sys.stderr.flush()

    assessment = await ctx.assessor.assess(prompt, skill, params)

    if assessment.verdict == AssessmentVerdict.READY:
        return True

    label, color = _VERDICT_LABELS.get(
        assessment.verdict, ("UNKNOWN", "\033[2m")
    )
    _RESET = "\033[0m"
    sys.stderr.write(f"{color}  [{label}] {assessment.reason}{_RESET}\n")
    if assessment.suggestions:
        for s in assessment.suggestions:
            sys.stderr.write(f"\033[2m  → {s}{_RESET}\n")
    sys.stderr.flush()

    try:
        response = input("  Proceed anyway? (yes/no) ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = "no"

    return response in ("y", "yes", "确认", "好")


def _print_execution_result(result: Any) -> None:
    """Print execution result as formatted JSON."""
    import json as _json
    from leapflow.engine.confirmation import StepResult

    output_data: Any
    if isinstance(result.output, list) and result.output and isinstance(result.output[0], StepResult):
        output_data = [
            {
                "step": r.step_idx + 1,
                "total": r.total,
                "instruction": r.description,
                "status": r.status,
                "output": r.output,
            }
            for r in result.output
        ]
    else:
        output_data = result.output

    payload = {
        "ok": result.ok,
        "skill": result.skill_name,
        "output": output_data,
    }
    if result.error:
        payload["error"] = result.error
    if result.duration_s:
        payload["duration_s"] = round(result.duration_s, 1)
    if result.steps_total:
        payload["steps_executed"] = result.steps_executed
        payload["steps_total"] = result.steps_total

    print(_json.dumps(payload, indent=2, ensure_ascii=False, default=str))


async def cmd_run(ctx: "Context", prompt: str, skill_name: Optional[str], step: bool, auto: bool = False) -> int:
    require_initialized(ctx)
    from leapflow.utils.terminal_io import TerminalIOProvider
    from leapflow.engine.confirmation import ConfirmLevel

    io = TerminalIOProvider()
    if step:
        confirm_override = ConfirmLevel.STEP
    else:
        confirm_override = ConfirmLevel.AUTO

    if skill_name:
        sys.stderr.write(f"\033[2m→ Executing skill: {skill_name}\033[0m\n")
        sys.stderr.flush()
        skill = ctx.registry.get(skill_name)
        params: Dict[str, Any] = {}
        if skill and prompt:
            params = await _extract_params_from_prompt(prompt, skill, ctx.llm)
        if prompt:
            params["user_goal"] = prompt
        if skill and not await _run_assessment(ctx, prompt or skill_name, skill, params):
            return 0
        result = await ctx.session.execute_skill(
            skill_name, params or None, io=io, confirm_override=confirm_override,
        )
        _print_execution_result(result)
        return 0 if result.ok else 1

    match = ctx.session.find_skill_match(prompt, threshold=0.3)

    if match is None:
        intent_label = "unknown"
        if ctx.intent_classifier is not None:
            try:
                intent = await ctx.intent_classifier.classify(prompt)
                intent_label = (
                    getattr(intent, "label", None)
                    or getattr(intent, "intent", None)
                    or str(intent)
                )
            except Exception as e:
                logger.debug("Intent classification failed: %s", e)
        sys.stderr.write(
            f"\033[2m→ No skill matched (intent: {intent_label})\n"
            f"  Falling back to LLM agent...\033[0m\n"
        )
        sys.stderr.flush()
        result = await ctx.engine.run(prompt)
        print(result)
        return 0

    skill = match.skill
    m = skill.metadata
    print("[ MATCHED SKILL ]")
    print(f"  Name:        {skill.name} (v{m.version})")
    print(
        f'  Confidence:  {match.score:.0%}  (matched trigger: "{match.matched_trigger}")'
    )
    print(f"  Description: {skill.description}")
    print(f"  Source:      {m.source}")
    preconds = (
        getattr(skill, "preconditions", None)
        or getattr(skill, "pre_conditions", None)
    )
    if preconds:
        print(f"  Preconds:    {', '.join(preconds)}")
    print()

    sys.stderr.write("\033[2m→ Extracting parameters...\033[0m\n")
    sys.stderr.flush()
    params = await _extract_params_from_prompt(prompt, skill, ctx.llm)
    params["user_goal"] = prompt
    if params:
        param_display = ", ".join(f"{k}={v!r}" for k, v in params.items() if k != "user_goal")
        if param_display:
            sys.stderr.write(f"\033[2m  params: {param_display}\033[0m\n")
            sys.stderr.flush()

    if not await _run_assessment(ctx, prompt, skill, params):
        return 0

    result = await ctx.session.execute_skill(
        skill.name, params, io=io, confirm_override=confirm_override,
    )
    _print_execution_result(result)
    return 0 if result.ok else 1
