"""Skill document generation — transform DistillationCandidates into SKILL.md.

Two strategies (same pattern as codegen.py):
- LLMSkillDocGenerator: LLM generates generalized instructions grounded in user's demonstration
- TemplateSkillDocGenerator: fallback for known patterns when LLM is unavailable (zero LLM)

Strategy order: LLM-first (uses actual demonstration), template-fallback (generic).
"""

from __future__ import annotations

import json
import logging
import os.path
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.learning.document import (
    ErrorHandlingEntry,
    ExampleDoc,
    ParameterDoc,
    SkillDocument,
    title_to_kebab,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocGenContext:
    """Context for skill document generation."""

    existing_skill_names: List[str] = field(default_factory=list)
    platform: str = "macOS"
    episode: Any = None


@runtime_checkable
class SkillDocGenerator(Protocol):
    async def generate(
        self, candidate: Any, context: DocGenContext, *, on_chunk: Any = None,
    ) -> Optional[SkillDocument]: ...


_ACTION_TOOL_RULES: List[tuple[List[str], List[str]]] = [
    (["file.", "file_", "create", "move", "rename", "delete"],
     ["Bash(find:*)", "Bash(mv:*)", "Bash(cp:*)", "Bash(mkdir:*)", "Bash(rm:*)"]),
    (["launch", "app.", "app_", "open"],
     ["leap-vsi(launch_app:*)"]),
    (["click", "scroll", "ui.", "ui_"],
     ["leap-vsi(ui_action:*)"]),
    (["clipboard", "paste", "pbcopy", "pbpaste"],
     ["Bash(pbcopy:*)"]),
    (["type", "input", "keystroke"],
     ["leap-vsi(ui_action:*)"]),
    (["shell", "command", "terminal"],
     ["Bash(*)"]),
]


def infer_allowed_tools(episode: Any) -> str:
    """Derive allowed-tools declaration from episode action types.

    Uses a declarative rule table (_ACTION_TOOL_RULES) mapping action
    keywords to tool declarations. Returns empty string if no semantic
    actions are available.
    """
    if not hasattr(episode, "semantic_actions"):
        return ""

    tools: set[str] = set()
    for action in episode.semantic_actions:
        name = action.action_name.lower()
        for keywords, tool_decls in _ACTION_TOOL_RULES:
            if any(kw in name for kw in keywords):
                tools.update(tool_decls)

    return " ".join(sorted(tools))


# ═══════════════════════════════════════════════════════════════════════
# Template Generator
# ═══════════════════════════════════════════════════════════════════════

_TEMPLATE_INSTRUCTIONS: Dict[str, List[str]] = {
    "file_organize": [
        "List all files in the source directory, identifying their extensions and types.",
        "Present the organization plan to the user: which files will move where, and whether any naming conflicts exist. Wait for confirmation before proceeding.",
        "Create any missing destination subdirectories.",
        "Move each file to its designated subdirectory. If a file with the same name exists at the destination, append a numeric suffix to avoid overwriting.",
        "Report results: how many files were moved, to which subdirectories, and any files that were skipped.",
    ],
    "batch_rename": [
        "List files in the target directory that match the given pattern.",
        "Show the user a preview of proposed renames (old name → new name) and ask for confirmation.",
        "Rename each matching file according to the replacement pattern.",
        "Report results: number of files renamed, any errors encountered.",
    ],
    "cross_app_transfer": [
        "Focus the source application and navigate to the target content.",
        "Copy the content to the clipboard (Cmd+C or appropriate shortcut).",
        "Switch to the target application.",
        "Paste the content (Cmd+V or appropriate shortcut) at the designated location.",
        "Verify the transfer succeeded by checking the target application state.",
    ],
    "clipboard_transform": [
        "Read the current clipboard content.",
        "Apply the specified transformation to the text.",
        "Write the transformed result back to the clipboard.",
        "Inform the user of the transformation applied and a preview of the result.",
    ],
    "web_download": [
        "Open the specified URL in the default browser.",
        "Wait for the page to load and identify the download target.",
        "Initiate the download to the specified save path.",
        "Verify the file was downloaded successfully by checking the destination.",
    ],
}

_TEMPLATE_PARAMETERS: Dict[str, List[ParameterDoc]] = {
    "file_organize": [
        ParameterDoc(name="source_dir", type="path", required=True, description="Directory to organize"),
        ParameterDoc(name="rules", type="dict", required=False, default="{}", description="Mapping of file extension to target subdirectory"),
    ],
    "batch_rename": [
        ParameterDoc(name="directory", type="path", required=True, description="Directory containing files to rename"),
        ParameterDoc(name="pattern", type="str", required=True, description="Regex pattern to match in filenames"),
        ParameterDoc(name="replacement", type="str", required=True, description="Replacement string for matched pattern"),
    ],
    "cross_app_transfer": [
        ParameterDoc(name="source_app", type="str", required=True, description="Source application name or bundle ID"),
        ParameterDoc(name="target_app", type="str", required=True, description="Target application name or bundle ID"),
    ],
    "clipboard_transform": [
        ParameterDoc(name="transform", type="str", required=True, description="Transformation to apply: upper, lower, strip, title, or custom"),
    ],
    "web_download": [
        ParameterDoc(name="url", type="str", required=True, description="URL to download from"),
        ParameterDoc(name="save_path", type="path", required=False, default="~/Downloads", description="Destination path for downloaded file"),
    ],
}

_TEMPLATE_TOOLS: Dict[str, str] = {
    "file_organize": "Bash(find:*) Bash(ls:*) Bash(mkdir:*) Bash(mv:*)",
    "batch_rename": "Bash(find:*) Bash(ls:*) Bash(mv:*)",
    "cross_app_transfer": "leap-vsi(launch_app:*) leap-vsi(ui_action:*)",
    "clipboard_transform": "Bash(pbcopy:*) Bash(pbpaste:*)",
    "web_download": "Bash(curl:*) Bash(open:*)",
}

_PATTERN_KEYWORDS: Dict[str, List[str]] = {
    "file_organize": ["organize", "sort", "classify", "整理", "move files", "by extension", "分类"],
    "batch_rename": ["rename", "batch", "pattern", "重命名", "批量", "regex"],
    "cross_app_transfer": ["transfer", "copy", "paste", "between apps", "跨应用", "clipboard"],
    "clipboard_transform": ["clipboard", "transform", "convert", "剪贴板", "format", "text"],
    "web_download": ["download", "url", "browser", "save", "下载", "fetch"],
}


class TemplateSkillDocGenerator:
    """Template-based document generation for known patterns (zero LLM cost)."""

    def _match_pattern(self, candidate: Any) -> Optional[str]:
        title_lower = candidate.title.lower()
        steps_text = " ".join(candidate.steps).lower()
        combined = f"{title_lower} {steps_text}"

        best_pattern: Optional[str] = None
        best_score = 0

        for pattern, keywords in _PATTERN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_pattern = pattern

        return best_pattern if best_score >= 2 else None

    async def generate(
        self, candidate: Any, context: DocGenContext, *, on_chunk: Any = None,
    ) -> Optional[SkillDocument]:
        pattern = self._match_pattern(candidate)
        if pattern is None:
            return None

        name = title_to_kebab(candidate.title)
        triggers = candidate.trigger_phrases[:3] if candidate.trigger_phrases else []
        trigger_text = ", ".join(f'"{t}"' for t in triggers)

        description = (
            f"{candidate.title}. "
            f"Use when user says {trigger_text}."
            if triggers else candidate.title
        )

        episode_tools = infer_allowed_tools(context.episode) if context.episode else ""
        template_tools = _TEMPLATE_TOOLS.get(pattern, "")
        all_tools = sorted(set(template_tools.split()) | set(episode_tools.split()) - {""})
        allowed_tools = " ".join(all_tools) if all_tools else template_tools

        return SkillDocument(
            name=name,
            description=description[:1024],
            goal=candidate.title,
            allowed_tools=allowed_tools,
            parameters=list(_TEMPLATE_PARAMETERS.get(pattern, [])),
            instructions=list(_TEMPLATE_INSTRUCTIONS[pattern]),
            preconditions=list(candidate.pre_conditions),
            postconditions=list(getattr(candidate, "post_conditions", [])),
            error_handling=_safe_parse_error_handling(
                getattr(candidate, "error_handling", None)
            ),
            procedure_graph=getattr(candidate, "procedure_graph", ""),
            examples=[ExampleDoc(trigger=t, actions=candidate.steps[:3]) for t in triggers[:2]],
            metadata={
                "author": "leapflow",
                "version": 1,
                "source": "learned",
                "confidence": candidate.confidence,
                "pattern": pattern,
            },
            source_trajectory_id=getattr(candidate, "source_trajectory_id", ""),
            source_episode_id=getattr(candidate, "source_episode_id", ""),
            learned_pattern=pattern,
        )


# ═══════════════════════════════════════════════════════════════════════
# LLM Generator
# ═══════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a skill document generator for desktop automation.
    Given a user's ACTUAL demonstrated workflow (recorded operations), generate a
    structured skill document that faithfully reflects their workflow pattern.

    EXECUTION MODEL — how instructions will be executed at runtime:
    - Each instruction runs in an INDEPENDENT ReAct loop with its own LLM context
    - Shell commands are STATELESS — working directory does NOT persist between steps
    - The executor has direct shell access — do NOT include "open Terminal" steps just to run commands
    - Each instruction must be SELF-CONTAINED: use absolute paths or `{parameter}` references, not relative paths from prior steps
    - Merge related operations into a single instruction (e.g., mkdir + mv = one step)
    - Fewer semantically complete instructions are better than many atomic ones

    CRITICAL RULES:
    - Instructions MUST reflect the user's actual demonstrated workflow
    - Parameterize specific paths, filenames, and patterns for reuse
    - Preserve structural relationships between steps (e.g., create dir then move into it)
    - Write generalizable, environment-agnostic instructions describing INTENT
    - Reference parameters by name using `{parameter_name}` syntax
    - Include verification/confirmation steps where appropriate

    PROCEDURE GRAPH:
    If the workflow has conditional branches, loops, or error recovery paths,
    include a "procedure_graph" field with Mermaid flowchart syntax (graph TD).

    ERROR HANDLING:
    Return "error_handling" as structured objects with pattern, signal, and recovery fields.

    Return a JSON object with these fields:
    {
      "goal": "one-line description of what this skill achieves",
      "instructions": ["step 1 text", "step 2 text", ...],
      "parameters": [{"name": "...", "type": "str|path|int|dict|list", "required": true/false, "default": null, "description": "..."}],
      "preconditions": ["condition that must be true before execution"],
      "postconditions": ["condition that should be true after execution"],
      "error_handling": [{"pattern": "error type", "signal": "detection signal", "recovery": "recovery action"}],
      "procedure_graph": "(optional) Mermaid graph TD flowchart if workflow has branches/loops",
      "examples": [{"trigger": "user phrase", "actions": ["action 1", "action 2"], "result": "expected outcome"}]
    }

    Return ONLY the JSON object, no extra text.
""")


class LLMSkillDocGenerator:
    """LLM-driven generation of generalized skill instructions."""

    def __init__(self, llm: Any, *, llm_timeout: float = 50.0) -> None:
        self._llm = llm
        self._llm_timeout = llm_timeout

    async def generate(
        self, candidate: Any, context: DocGenContext, *, on_chunk: Any = None,
    ) -> Optional[SkillDocument]:
        import asyncio

        from leapflow.llm.message_builder import build_system_message, build_user_message_text

        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(candidate.steps))
        params_text = "\n".join(
            f"  - {p.get('name', '?')}: {p.get('description', '')} (e.g. {p.get('example', '')})"
            for p in candidate.parameters
        ) or "  (none)"

        # Build rich episode context if available
        action_details = steps_text
        structural_notes = ""
        if context.episode is not None:
            action_details = _format_episode_actions(context.episode)
            structural_notes = _analyze_structure(context.episode)

        trigger_text = ", ".join(candidate.trigger_phrases[:3]) if candidate.trigger_phrases else ""

        prompt = textwrap.dedent(f"""\
            Generate a skill document for this demonstrated workflow:

            Goal: {candidate.title}
            Trigger phrases: {trigger_text}

            Demonstrated operations (from user recording):
            {action_details}

            Parameters detected:
            {params_text}

            Structural observations:
            {structural_notes or "  (none)"}

            Preconditions: {candidate.pre_conditions}
            Platform: {context.platform}

            IMPORTANT: Instructions must reflect the user's ACTUAL demonstrated pattern,
            parameterized for reuse. Do not invent a generic workflow — describe what the
            user actually did, generalized with parameters.
        """)

        try:
            coro = self._llm.achat(
                [
                    build_system_message(_SYSTEM_PROMPT),
                    build_user_message_text(prompt),
                ],
                stream=True,
                enable_thinking=False,
                on_chunk=on_chunk,
            )
            resp = await asyncio.wait_for(coro, timeout=self._llm_timeout)
        except asyncio.TimeoutError:
            logger.warning("LLM skill doc generation timed out after %.0fs", self._llm_timeout)
            return None
        except Exception:
            logger.warning("LLM skill doc generation failed", exc_info=True)
            return None

        return self._parse_response(resp.content or "", candidate, context)

    def _parse_response(self, raw: str, candidate: Any, context: DocGenContext) -> Optional[SkillDocument]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return None

        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        name = title_to_kebab(candidate.title)
        triggers = candidate.trigger_phrases[:3] if candidate.trigger_phrases else []
        trigger_text = ", ".join(f'"{t}"' for t in triggers)
        description = (
            f"{data.get('goal', candidate.title)}. "
            f"Use when user says {trigger_text}."
            if triggers else data.get("goal", candidate.title)
        )

        parameters = _safe_parse_parameters(data.get("parameters"))
        examples = _safe_parse_examples(data.get("examples"))

        allowed_tools = infer_allowed_tools(context.episode) if context.episode else ""

        instructions = data.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            instructions = list(candidate.steps)

        preconditions = data.get("preconditions")
        if not isinstance(preconditions, list):
            preconditions = list(candidate.pre_conditions)

        procedure_graph = data.get("procedure_graph", "")
        if not isinstance(procedure_graph, str):
            procedure_graph = ""

        return SkillDocument(
            name=name,
            description=description[:1024],
            goal=data.get("goal", candidate.title),
            allowed_tools=allowed_tools,
            parameters=parameters,
            instructions=instructions,
            preconditions=preconditions,
            postconditions=data.get("postconditions") if isinstance(data.get("postconditions"), list) else [],
            error_handling=_safe_parse_error_handling(data.get("error_handling")),
            examples=examples,
            procedure_graph=procedure_graph,
            metadata={
                "author": "leapflow",
                "version": 1,
                "source": "learned",
                "confidence": candidate.confidence,
            },
            source_trajectory_id=getattr(candidate, "source_trajectory_id", ""),
            source_episode_id=getattr(candidate, "source_episode_id", ""),
        )


# ═══════════════════════════════════════════════════════════════════════
def _safe_parse_parameters(raw: Any) -> List[ParameterDoc]:
    """Parse parameters from LLM response, tolerating malformed data."""
    if not isinstance(raw, list):
        return []
    result: List[ParameterDoc] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        result.append(ParameterDoc(
            name=str(p.get("name", "")),
            type=str(p.get("type", "str")),
            required=bool(p.get("required", False)),
            default=p.get("default"),
            description=str(p.get("description", "")),
        ))
    return result


def _safe_parse_examples(raw: Any) -> List[ExampleDoc]:
    """Parse examples from LLM response, tolerating malformed data."""
    if not isinstance(raw, list):
        return []
    result: List[ExampleDoc] = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        actions = ex.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        result.append(ExampleDoc(
            trigger=str(ex.get("trigger", "")),
            actions=actions,
            result=str(ex.get("result", "")),
        ))
    return result


def _safe_parse_error_handling(raw: Any) -> List[ErrorHandlingEntry]:
    """Parse error_handling from LLM response — accepts str list or dict list."""
    if not isinstance(raw, list):
        return []
    result: List[ErrorHandlingEntry] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(ErrorHandlingEntry(
                pattern=str(item.get("pattern", "")),
                signal=str(item.get("signal", "")),
                recovery=str(item.get("recovery", "")),
                script=str(item.get("script", "")),
            ))
        elif isinstance(item, str):
            result.append(ErrorHandlingEntry(pattern=item))
    return result


# ═══════════════════════════════════════════════════════════════════════
# Composite Generator
# ═══════════════════════════════════════════════════════════════════════


class CompositeSkillDocGenerator:
    """LLM-first, template-fallback (Strategy pattern).

    The LLM generator produces instructions grounded in the user's actual
    demonstrated workflow. Template is used only when LLM is unavailable.
    """

    def __init__(
        self,
        llm_generator: Optional[LLMSkillDocGenerator] = None,
        template_generator: Optional[TemplateSkillDocGenerator] = None,
    ) -> None:
        self._template = template_generator or TemplateSkillDocGenerator()
        self._llm = llm_generator

    async def generate(
        self, candidate: Any, context: DocGenContext, *, on_chunk: Any = None,
    ) -> Optional[SkillDocument]:
        # LLM-first: generates instructions from actual demonstration
        if self._llm is not None:
            doc = await self._llm.generate(candidate, context, on_chunk=on_chunk)
            if doc is not None:
                logger.info("doc_generator.llm generated name=%s", doc.name)
                return self._deduplicate_name(doc, context)
            logger.warning("doc_generator.llm_failed, falling back to template")

        # Template fallback (with user-facing warning)
        result = await self._template.generate(candidate, context)
        if result is not None:
            logger.warning(
                "doc_generator.template_fallback name=%s — "
                "skill document uses generic template, not learned workflow",
                result.name,
            )
            print(
                "Warning: LLM unavailable — using generic template for skill document. "
                "For workflow-specific instructions, configure LEAPFLOW_LLM_API_KEY.",
                file=sys.stderr,
            )
            return self._deduplicate_name(result, context)

        return self._deduplicate_name(self._build_minimal(candidate), context)

    @staticmethod
    def _deduplicate_name(doc: SkillDocument, context: DocGenContext) -> SkillDocument:
        """Append numeric suffix if name collides with existing skills."""
        if doc.name not in context.existing_skill_names:
            return doc
        base = doc.name
        for i in range(2, 100):
            candidate_name = f"{base}-{i}"
            if candidate_name not in context.existing_skill_names:
                doc.name = candidate_name
                return doc
        doc.name = f"{base}-dup"
        return doc

    def _build_minimal(self, candidate: Any) -> SkillDocument:
        """Fallback: construct minimal document from candidate fields directly."""
        name = title_to_kebab(candidate.title)
        triggers = candidate.trigger_phrases[:3] if candidate.trigger_phrases else []
        trigger_text = ", ".join(f'"{t}"' for t in triggers)
        description = (
            f"{candidate.title}. Use when user says {trigger_text}."
            if triggers else candidate.title
        )

        return SkillDocument(
            name=name,
            description=description[:1024],
            goal=candidate.title,
            allowed_tools="Bash(*)",
            parameters=[
                ParameterDoc(name=p.get("name", ""), description=p.get("description", ""))
                for p in candidate.parameters
            ],
            instructions=list(candidate.steps),
            preconditions=list(candidate.pre_conditions),
            error_handling=_safe_parse_error_handling(
                getattr(candidate, "error_handling", None)
            ),
            procedure_graph=getattr(candidate, "procedure_graph", ""),
            examples=[ExampleDoc(trigger=t) for t in triggers[:2]],
            metadata={
                "author": "leapflow",
                "version": 1,
                "source": "learned",
                "confidence": candidate.confidence,
            },
            source_trajectory_id=getattr(candidate, "source_trajectory_id", ""),
            source_episode_id=getattr(candidate, "source_episode_id", ""),
        )


# ═══════════════════════════════════════════════════════════════════════
# Episode Context Helpers
# ═══════════════════════════════════════════════════════════════════════


def _format_episode_actions(episode: Any) -> str:
    """Format episode semantic actions with parameters for the LLM prompt."""
    if not hasattr(episode, "semantic_actions") or not episode.semantic_actions:
        return "  (no actions recorded)"

    lines: List[str] = []
    for i, action in enumerate(episode.semantic_actions, 1):
        params_str = ""
        if action.parameters:
            relevant = {
                k: v for k, v in action.parameters.items()
                if k not in ("_noise", "_merged_count") and v
            }
            if relevant:
                params_str = " | " + ", ".join(
                    f"{k}={_truncate(str(v), 80)}" for k, v in relevant.items()
                )
        lines.append(f"  {i}. {action.action_name}: {action.description}{params_str}")
    return "\n".join(lines)


def _analyze_structure(episode: Any) -> str:
    """Detect structural relationships between semantic actions for richer LLM context."""
    if not hasattr(episode, "semantic_actions") or not episode.semantic_actions:
        return ""

    actions = episode.semantic_actions
    notes: List[str] = []

    # Detect: directory creation followed by file moves into that directory
    created_dirs: List[str] = []
    for a in actions:
        if "create" in a.action_name:
            path = a.parameters.get("path") or a.parameters.get("target_dir", "")
            if path:
                created_dirs.append(path)

    move_targets: List[str] = []
    for a in actions:
        if any(kw in a.action_name for kw in ("rename", "move", "batch")):
            sources = a.parameters.get("sources")
            if isinstance(sources, list):
                move_targets.extend(s for s in sources if s)
            else:
                path = (
                    a.parameters.get("destination")
                    or a.parameters.get("path")
                    or a.parameters.get("first_file", "")
                )
                if path:
                    move_targets.append(path)

    for d in created_dirs:
        related = [t for t in move_targets if t.startswith(d + "/") or os.path.dirname(t) == d]
        if related:
            notes.append(
                f"  - Directory '{os.path.basename(d)}' was created as target, "
                f"then {len(related)} file(s) moved into it"
            )

    # Detect: common file extension among moved/renamed files
    if move_targets:
        extensions = [os.path.splitext(t)[1].lower() for t in move_targets if os.path.splitext(t)[1]]
        if extensions:
            common = Counter(extensions).most_common(1)
            if common and common[0][1] > 1:
                notes.append(
                    f"  - Moved files share extension '{common[0][0]}' — likely filtering by type"
                )

    # Detect: batch operations
    batch_actions = [a for a in actions if "batch" in a.action_name]
    if batch_actions:
        for a in batch_actions:
            count = a.parameters.get("_merged_count") or a.parameters.get("count", "")
            if count:
                notes.append(f"  - Batch operation: {a.action_name} ({count} items)")

    return "\n".join(notes) if notes else ""


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
