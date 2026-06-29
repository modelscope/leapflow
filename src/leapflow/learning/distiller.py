"""Skill distillation — extract reusable skills from trajectories and transcripts.

Supports two pathways:
  1. Heuristic (transcript-based): fast keyword matching, no LLM cost
  2. Trajectory-based: converts Episode semantic actions into parameterized skills
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from leapflow.domain.skill_types import (
    AnchorCandidate,
    DistillationCandidate,
    RecoveryEvent,
)

if TYPE_CHECKING:
    from leapflow.domain.trajectory import Episode
    from leapflow.learning.codegen import CodeGenContext, GeneratedSkill, SkillCodeGenerator

logger = logging.getLogger(__name__)


class SkillDistiller:
    """Extract reusable patterns from trajectories (best-effort, offline-quality)."""

    def propose_from_transcript(self, transcript: str) -> List[DistillationCandidate]:
        """Derive trivial candidates using lightweight heuristics.

        This is intentionally conservative: real distillation should call an LLM offline.
        """
        out: List[DistillationCandidate] = []
        if re.search(r"\bmove\b|\b整理\b|\bmv\b", transcript, re.I):
            out.append(
                DistillationCandidate(
                    title="Move files by rule",
                    trigger_phrases=["organize", "整理", "move pdfs"],
                    steps=["List directory", "Classify", "Move in batches"],
                )
            )
        if "clipboard" in transcript.lower() or "剪贴板" in transcript:
            out.append(
                DistillationCandidate(
                    title="Clipboard digest",
                    trigger_phrases=["clipboard", "剪贴板"],
                    steps=["Observe change", "Summarize", "Store memory"],
                )
            )
        return out

    def propose_from_episode(self, episode: "Episode") -> Optional[DistillationCandidate]:
        """Extract a skill candidate from a segmented, abstracted episode.

        Uses the episode's semantic actions to build a parameterized skill
        without requiring LLM calls.  For LLM-enhanced distillation, see
        LLMSkillDistiller.
        """
        from leapflow.analysis.causal import CausalChainAnalyzer

        if not episode.semantic_actions:
            return None
        if len(episode.semantic_actions) < 2:
            return None

        actions = CausalChainAnalyzer().extract_causal_chain(episode.semantic_actions)
        steps = [a.description for a in actions]
        params = _extract_variable_params(episode)
        title = episode.inferred_goal or _generate_title(episode)
        triggers = _generate_triggers(episode)

        return DistillationCandidate(
            title=title,
            trigger_phrases=triggers,
            steps=steps,
            parameters=params,
            pre_conditions=_infer_preconditions(episode),
            source_trajectory_id=episode.trajectory_id,
            source_episode_id=episode.episode_id,
            confidence=episode.confidence or _estimate_confidence(episode),
            recovery_events=_extract_recovery_events(episode),
            anchor_candidates=_extract_anchor_candidates(episode),
        )

    def persist_stub(self, candidate: DistillationCandidate, path: str) -> None:
        """Write a JSON artifact for humans to promote into a real skill module."""
        payload: Dict[str, Any] = {
            "title": candidate.title,
            "trigger_phrases": candidate.trigger_phrases,
            "steps": candidate.steps,
            "parameters": candidate.parameters,
            "pre_conditions": candidate.pre_conditions,
            "post_conditions": candidate.post_conditions,
            "source": {
                "trajectory_id": candidate.source_trajectory_id,
                "episode_id": candidate.source_episode_id,
            },
            "confidence": candidate.confidence,
        }
        logger.info("Distillation stub:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))

    async def distill_to_executable(
        self,
        episode: "Episode",
        codegen: "SkillCodeGenerator",
        context: "CodeGenContext",
    ) -> Optional["GeneratedSkill"]:
        """Complete pipeline: episode → candidate → generated code.

        1. propose_from_episode(episode) → DistillationCandidate
        2. codegen.generate(candidate, context) → GeneratedSkill
        3. Return GeneratedSkill (caller decides whether to register)
        """
        candidate = self.propose_from_episode(episode)
        if candidate is None:
            return None
        return await codegen.generate(candidate, context)


# ── LLM-enhanced distiller ──


_PATH_OPTIMIZATION_INSTRUCTION = """
IMPORTANT: The action sequence may contain:
- Error corrections (action → undo → retry): extract only the final correct action
- Redundant operations (multiple saves): keep only one
- Suboptimal paths (menu navigation instead of shortcuts): use the most efficient method
- Irrelevant detours (checking notifications): remove entirely

Generate the MINIMAL set of steps needed to achieve the goal.
For each step, prefer keyboard shortcuts over menu navigation where applicable.
If the user navigated A→B→C→B→D, the optimal path is A→B→D.

STEP MERGING — each step should be a complete semantic outcome, not a single command:
- Merge causally related operations into one step (e.g., create directory + move files into it = one step)
- Do NOT include "open Terminal" or "launch shell" steps just to run commands — shell access is built in
- Do NOT include bare "cd" steps — use absolute paths or inline cd with &&
- Parameterize paths, patterns, and names so the skill is reusable across different inputs
"""


class LLMSkillDistiller(SkillDistiller):
    """Extended distiller that uses LLM for richer skill extraction."""

    def __init__(self, llm: Any) -> None:
        super().__init__()
        self._llm = llm

    async def propose_from_episode_async(
        self, episode: "Episode", *, on_chunk: "Any" = None,
    ) -> Optional[DistillationCandidate]:
        """LLM-powered skill extraction from semantic episode."""
        from leapflow.analysis.causal import CausalChainAnalyzer
        from leapflow.llm.message_builder import build_system_message, build_user_message_text

        if not episode.semantic_actions:
            return None

        logger.info(
            "distill: LLM extraction for episode %s (%d actions)",
            episode.episode_id, len(episode.semantic_actions),
        )

        actions = CausalChainAnalyzer().extract_causal_chain(episode.semantic_actions)
        event_actions = [a for a in actions if a.parameters.get("_source") != "visual"]
        visual_actions = [a for a in actions if a.parameters.get("_source") == "visual"]

        action_lines = "\n".join(
            f"{i + 1}. {a.description} (params: {a.parameters})"
            for i, a in enumerate(event_actions)
        )

        visual_lines = ""
        if visual_actions:
            visual_items = "\n".join(
                f"- {a.description} [evidence: {a.parameters.get('evidence', '')}]"
                for a in visual_actions
            )
            visual_lines = (
                f"\nVisual observations (from screen capture analysis):\n{visual_items}\n"
            )

        goal_line = ""
        if episode.inferred_goal:
            goal_line = f"User's stated goal: {episode.inferred_goal}\n"

        prompt = (
            "Analyze this user operation sequence and extract a reusable skill.\n\n"
            f"{goal_line}"
            f"Apps involved: {', '.join(episode.app_sequence)}\n"
            f"Actions:\n{action_lines}\n"
            f"{visual_lines}\n"
            f"{_PATH_OPTIMIZATION_INSTRUCTION}\n"
            "Generate a JSON response with:\n"
            '- "title": concise skill name (use the user\'s goal as basis if provided)\n'
            '- "trigger_phrases": list of phrases that would invoke this skill\n'
            '- "steps": list of parameterized step descriptions\n'
            '- "parameters": list of {{"name": "...", "description": "..."}}\n'
            '- "pre_conditions": list of required conditions\n'
            '- "confidence": 0.0-1.0\n'
            '- "procedure_graph": (optional) Mermaid flowchart (graph TD) if the workflow has branches or loops\n'
            '- "error_handling": (optional) list of {{"pattern": "...", "signal": "...", "recovery": "..."}}\n\n'
            "Return ONLY the JSON object."
        )

        try:
            resp = await self._llm.achat(
                [
                    build_system_message("You are a skill extraction engine for desktop automation."),
                    build_user_message_text(prompt),
                ],
                stream=True,
                enable_thinking=False,
                on_chunk=on_chunk,
            )
            candidate = self._parse_llm_response(resp.content or "", episode)
            logger.info(
                "distill: LLM extraction complete for episode %s, candidate=%s",
                episode.episode_id, candidate.title if candidate else None,
            )
            return candidate
        except Exception:
            logger.debug("LLM skill distillation failed; falling back", exc_info=True)
            return self.propose_from_episode(episode)

    async def distill_to_executable(
        self,
        episode: "Episode",
        codegen: "SkillCodeGenerator",
        context: "CodeGenContext",
    ) -> Optional["GeneratedSkill"]:
        """Complete pipeline: episode → candidate → generated code (LLM-enhanced).

        Uses LLM-powered proposal for richer extraction, then generates code.
        Falls back to heuristic proposal if LLM fails.
        """
        candidate = await self.propose_from_episode_async(episode)
        if candidate is None:
            return None
        return await codegen.generate(candidate, context)

    def _parse_llm_response(
        self, raw: str, episode: "Episode"
    ) -> Optional[DistillationCandidate]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return self.propose_from_episode(episode)
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return self.propose_from_episode(episode)

        error_handling = data.get("error_handling", [])
        if not isinstance(error_handling, list):
            error_handling = []

        return DistillationCandidate(
            title=data.get("title", ""),
            trigger_phrases=data.get("trigger_phrases", []),
            steps=data.get("steps", []),
            parameters=data.get("parameters", []),
            pre_conditions=data.get("pre_conditions", []),
            confidence=data.get("confidence", 0.5),
            source_trajectory_id=episode.trajectory_id,
            source_episode_id=episode.episode_id,
            recovery_events=_extract_recovery_events(episode),
            anchor_candidates=_extract_anchor_candidates(episode),
            procedure_graph=data.get("procedure_graph", ""),
            error_handling=error_handling,
        )


# ── Helpers ──


def _extract_variable_params(episode: "Episode") -> List[Dict[str, str]]:
    """Identify parameters that vary across actions (likely user-configurable)."""
    import os.path

    params: List[Dict[str, str]] = []
    seen_names: set[str] = set()

    # Basic extraction: surface-level variable fields
    for action in episode.semantic_actions:
        for key, val in action.parameters.items():
            if key in seen_names:
                continue
            if key in ("target", "path", "text_preview", "target_label"):
                seen_names.add(key)
                params.append({"name": key, "description": f"Variable: {key}", "example": str(val)[:100]})

    # Structural inference: detect target_dir from create→move relationship
    _infer_structural_params(episode, params, seen_names)

    return params


def _infer_structural_params(
    episode: "Episode", params: List[Dict[str, str]], seen_names: set[str],
) -> None:
    """Detect structural relationships between actions and infer higher-level params."""
    import os.path

    _CREATE_ACTIONS = ("file.create", "batch_move_to_folder", "move_to_new_folder")
    _MOVE_ACTIONS = (
        "file.rename", "file.move", "batch_move",
        "batch_rename", "batch_move_to_folder", "move_to_new_folder",
    )

    created_paths: List[str] = []
    moved_paths: List[str] = []

    for action in episode.semantic_actions:
        if action.action_name in _CREATE_ACTIONS:
            path = action.parameters.get("path") or action.parameters.get("target_dir", "")
            if path:
                created_paths.append(path)
        if action.action_name in _MOVE_ACTIONS:
            # batch_move carries a list of source paths
            sources = action.parameters.get("sources")
            if isinstance(sources, list):
                moved_paths.extend(s for s in sources if s)
            else:
                path = (
                    action.parameters.get("destination")
                    or action.parameters.get("path")
                    or action.parameters.get("first_file", "")
                )
                if path:
                    moved_paths.append(path)

    # Detect: target directory that contains moved files
    if created_paths and moved_paths and "target_dir" not in seen_names:
        for d in created_paths:
            if any(p.startswith(d + "/") or os.path.dirname(p) == d for p in moved_paths):
                seen_names.add("target_dir")
                params.append({
                    "name": "target_dir",
                    "description": "Target directory for organized files",
                    "example": d,
                })
                break

    # Detect: common file extension among moved files
    if moved_paths and "file_pattern" not in seen_names:
        extensions = {
            os.path.splitext(p)[1].lower()
            for p in moved_paths if os.path.splitext(p)[1]
        }
        if len(extensions) == 1:
            ext = extensions.pop()
            seen_names.add("file_pattern")
            params.append({
                "name": "file_pattern",
                "description": "File pattern to match",
                "example": f"*{ext}",
            })


def _generate_title(episode: "Episode") -> str:
    """Synthesize a title from the episode's action sequence."""
    if not episode.semantic_actions:
        return "Untitled skill"
    verbs: List[str] = []
    for a in episode.semantic_actions[:3]:
        name = a.action_name.replace("_", " ").replace(".", " ")
        verbs.append(name)
    apps = ", ".join(episode.app_sequence[:2]) if episode.app_sequence else ""
    suffix = f" in {apps}" if apps else ""
    return " → ".join(verbs) + suffix


def _generate_triggers(episode: "Episode") -> List[str]:
    """Generate plausible trigger phrases from episode content."""
    triggers: List[str] = []
    if episode.inferred_goal:
        goal_lower = episode.inferred_goal.lower()
        triggers.append(goal_lower)
        words = [w for w in goal_lower.split() if len(w) > 2]
        if len(words) >= 2:
            triggers.append(" ".join(words))
    for app in episode.app_sequence[:2]:
        short = app.split(".")[-1].lower()
        triggers.append(short)
    return triggers or ["automated task"]


def _infer_preconditions(episode: "Episode") -> List[str]:
    """Infer preconditions from the episode's action sequence."""
    conditions: List[str] = []
    for app in episode.app_sequence:
        conditions.append(f"{app} available")
    return conditions


def _estimate_confidence(episode: "Episode") -> float:
    """Heuristic confidence based on episode characteristics."""
    if not episode.semantic_actions:
        return 0.0
    base = 0.5
    if len(episode.semantic_actions) >= 3:
        base += 0.1
    if len(episode.app_sequence) >= 2:
        base += 0.1
    if episode.inferred_goal:
        base += 0.15
    return min(base, 1.0)


_UNDO_KEYWORDS = ("undo", "back", "cancel", "dismiss", "close", "撤销", "返回")


def _extract_recovery_events(episode: "Episode") -> List[RecoveryEvent]:
    """Detect error-recovery patterns in the action sequence.

    Patterns detected:
    - Undo/back: action followed by undo or navigation back
    - Retry: consecutive actions with the same action_name
    """
    events: List[RecoveryEvent] = []
    actions = episode.semantic_actions
    if not actions:
        return events

    for i in range(1, len(actions)):
        prev, curr = actions[i - 1], actions[i]
        curr_name = curr.action_name.lower()

        if any(kw in curr_name for kw in _UNDO_KEYWORDS):
            events.append(RecoveryEvent(
                pattern="error_correction",
                trigger_action=prev.action_name,
                recovery_action=curr.action_name,
                confidence=0.6,
            ))

        if curr.action_name == prev.action_name:
            events.append(RecoveryEvent(
                pattern="retry",
                trigger_action=prev.action_name,
                recovery_action=curr.action_name,
                confidence=0.5,
            ))

    return events


def _extract_anchor_candidates(episode: "Episode") -> List[AnchorCandidate]:
    """Extract UI element identifiers from raw actions for grounding anchors."""
    anchors: List[AnchorCandidate] = []
    seen: set[str] = set()

    for step_idx, action in enumerate(episode.semantic_actions):
        label = action.parameters.get("target_label", "")
        role = action.parameters.get("target_role", "")
        app = action.parameters.get("app_bundle_id", "")
        if not app and episode.app_sequence:
            app = episode.app_sequence[0] if len(episode.app_sequence) == 1 else ""

        if label and label not in seen:
            seen.add(label)
            anchors.append(AnchorCandidate(
                step_index=step_idx,
                element_label=label,
                element_role=role,
                app_bundle_id=app,
            ))

    return anchors
