# Copyright (c) Alibaba, Inc. and its affiliates.
"""Skill / intent dispatch and action-execution helpers for :class:`AgentEngine`.

Extracted from ``engine.py`` (Phase 3 refactor). This component owns trigger
matching for learned skills, memory-recent question answering, recording/learn
intent handling, skill list/execute/review/approve commands, and the no-LLM
action-execution boundary (``execute_action`` and its per-type dispatch). It
holds a back-reference to the owning engine so every access reads the engine's
*live* mutable state (stores/session injected at runtime via ``set_*`` methods),
preserving exact runtime semantics.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from leapflow.engine.intent_classifier import Intent
from leapflow.engine.tools.action_executor import ActionInvocation
from leapflow.engine.tools.tool_execution import ExecutionPolicy, normalize_execution_policy
from leapflow.llm.message_builder import build_system_message, build_user_message_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.engine.engine import AgentEngine

logger = logging.getLogger(__name__)


class SkillDispatcher:
    """Skill/intent dispatch and action execution, held by composition."""

    # Patterns that indicate a genuine teach session command.
    # Uses regex word-boundary checks to avoid false positives like
    # "teaching methods for math".
    _TEACH_COMMAND_RE = re.compile(
        r"^(?:"
        r"(?:start\s+)?teach(?:ing)?(?:\s+(?:this|that|it|me|now))?$"
        r"|stop\s+teach(?:ing)?"
        r"|pause\s+teach(?:ing)?"
        r"|resume\s+teach(?:ing)?"
        r"|done\s+teach(?:ing)?"
        r"|finish\s+teach(?:ing)?"
        r"|end\s+teach(?:ing)?"
        r"|教(?:我|一下)?$"
        r"|开始教学"
        r"|停止教学|暂停教学|继续教学|结束教学"
        r"|watch\s+me"
        r")",
        re.IGNORECASE,
    )

    def __init__(self, engine: "AgentEngine") -> None:
        self._engine = engine

    async def _try_trigger_match(self, user_text: str) -> Optional[str]:
        """Check if a learned skill directly matches the user's request.

        Returns the skill output if a high-confidence match is found,
        or None to fall through to the ReAct/DAG path.

        Enforces Progressive Trust: the ConfirmationHandler determines
        whether the skill requires user confirmation before execution.
        """
        matches = self._engine._registry.find_by_trigger(user_text, threshold=0.5)
        if not matches:
            return None

        best = matches[0]
        if best.metadata.source not in ("distilled", "template"):
            return None
        if best.metadata.confidence < 0.6:
            return None

        from leapflow.engine.confirmation import ConfirmationHandler, ConfirmLevel

        handler = ConfirmationHandler(skill_store=self._engine._skill_library)
        level = handler.determine_level(best)

        if level in (ConfirmLevel.STEP, ConfirmLevel.CONFIRM):
            logger.info(
                "audit.trigger_match_deferred skill=%s tier=%s (requires confirmation)",
                best.name,
                best.metadata.tier.name,
            )
            return None

        logger.info(
            "audit.trigger_match skill=%s confidence=%.2f level=%s",
            best.name,
            best.metadata.confidence,
            level.value,
        )
        result = await self.execute_action(
            {
                "type": "skill",
                "name": best.name,
                "payload": {},
                "execution_policy": best.metadata.execution_policy,
            },
            user_text,
        )
        if bool(result.get("ok", True)):
            return str(result.get("result", ""))
        logger.warning(
            "audit.trigger_match_failed skill=%s error=%s",
            best.name,
            result.get("error"),
        )
        return None

    async def _handle_memory_recent(self, user_text: str) -> str:
        """Answer questions about recent activity using memory + optional LLM."""
        events = self._collect_recent_events()

        if not events:
            return "No recent activity records in memory."

        for f in self._engine._imm.recent(limit=50):
            self._engine._imm.touch(f.fragment_id)

        if self._engine._settings.has_llm_credentials:
            return await self._synthesize_memory_answer(user_text, events)

        return self._format_recent_events(events)

    def _collect_recent_events(self) -> List[Dict[str, Any]]:
        """Gather events from immediate memory, dedup by (path, action)."""
        frags = self._engine._imm.recent(limit=50)
        if not frags:
            hits = self._engine._lt.recent_file_events(within_seconds=3600)
            return [
                {
                    "ts": h.created_at,
                    "time": datetime.fromtimestamp(h.created_at).strftime("%H:%M:%S"),
                    "type": h.kind,
                    "content": h.content,
                    "path": h.path or "",
                }
                for h in hits[:30]
            ]

        seen: Dict[str, Dict[str, Any]] = {}
        for f in frags:
            key = f"{f.event_type}:{f.path or f.content}"
            if key not in seen or f.created_at > seen[key]["ts"]:
                seen[key] = {
                    "ts": f.created_at,
                    "time": datetime.fromtimestamp(f.created_at).strftime("%H:%M:%S"),
                    "type": f.event_type,
                    "content": f.content,
                    "path": f.path or "",
                }
        result = sorted(seen.values(), key=lambda e: e["ts"], reverse=True)
        return result

    async def _synthesize_memory_answer(self, user_text: str, events: List[Dict[str, Any]]) -> str:
        """Use LLM to answer the user's question based on collected events."""
        events_json = json.dumps(events, ensure_ascii=False)
        messages = [
            build_system_message(
                "You are LeapFlow's memory assistant. "
                "Given a list of recent system events (file changes, clipboard, app focus, etc.), "
                "answer the user's question accurately and concisely.\n"
                "Rules:\n"
                "- Filter events relevant to the user's question (time range, file type, etc.)\n"
                "- Skip obvious system/background noise (databases, caches, logs)\n"
                "- Include timestamps when the user asks for them\n"
                "- If no relevant events match, say so clearly\n"
                "- Answer in the same language as the user's question"
            ),
            build_user_message_text(
                f"Question: {user_text}\n\nRecent events ({len(events)} total):\n{events_json}"
            ),
        ]
        try:
            resp = await self._engine._llm.achat(messages, stream=False, enable_thinking=False)
            answer = (resp.content or "").strip()
            if answer:
                return answer
        except Exception:
            logger.warning("LLM synthesis failed for memory_recent", exc_info=True)
        return self._format_recent_events(events)

    @staticmethod
    def _format_recent_events(events: List[Dict[str, Any]]) -> str:
        """Fallback formatting when LLM is unavailable."""
        lines = [f"Recent activity ({len(events)} events):\n"]
        for e in events[:30]:
            lines.append(f"- {e['time']} [{e['type']}] {e['content']}")
        return "\n".join(lines)

    async def _handle_recording_intent(self, intent: Intent, user_text: str) -> str:
        """Handle recording-related intents (start/stop/analyze)."""
        if self._engine._imitation is None:
            return "Imitation learning is not configured."

        if intent.label == "recording_start":
            tid = await self._engine._imitation.start_recording()
            return f"Recording started. Trajectory ID: {tid}"

        if intent.label == "recording_stop":
            traj = await self._engine._imitation.stop_recording()
            if traj is None:
                return "No active recording to stop."
            return (
                f"Recording stopped. Trajectory: {traj.trajectory_id}\n"
                f"Steps: {traj.step_count} | Duration: {traj.duration:.1f}s\n"
                f"Apps: {', '.join(traj.app_sequence) or 'none'}"
            )

        if intent.label == "recording_analyze":
            trajs = self._engine._imitation.list_trajectories(limit=1)
            if not trajs:
                return "No trajectories found. Start a recording first."
            tid = trajs[0]["id"]
            candidates = await self._engine._imitation.distill(tid)
            if not candidates:
                replay = self._engine._imitation.format_trajectory(tid)
                return f"No skill candidates found.\n\nTrajectory replay:\n{replay}"
            lines = [f"Distilled {len(candidates)} skill candidate(s) from trajectory {tid}:\n"]
            for c in candidates:
                lines.append(f"  - {c.title} (confidence: {c.confidence:.2f})")
                lines.append(f"    Steps: {' → '.join(c.steps[:5])}")
                if c.trigger_phrases:
                    lines.append(f"    Triggers: {', '.join(c.trigger_phrases[:3])}")
            return "\n".join(lines)

        return "Unknown recording command."

    async def _handle_learn_intent(self, intent: Intent, user_text: str) -> str:
        if self._engine._session is None:
            return "Session controller is not configured."

        if intent.label == "learn_start":
            try:
                session = await self._engine._session.enter_learning(goal=user_text)
                return (
                    f"Learning started. Session: {session.session_id}\n"
                    f"Trajectory: {session.trajectory_id}\n"
                    "Perform the task you want me to learn. Say 'stop learning' when done."
                )
            except Exception as e:
                return f"Cannot start learning: {e}"

        if intent.label == "learn_stop":
            try:
                result = await self._engine._session.exit_learning()
                lines = [
                    f"Learning stopped. Trajectory: {result.trajectory_id}",
                    f"Steps: {result.step_count} | Duration: {result.duration:.1f}s",
                ]
                if result.new_skills:
                    lines.append(f"New skills learned: {', '.join(result.new_skills)}")
                if result.suggestions > 0:
                    lines.append(f"Suggestions pending: {result.suggestions}")
                return "\n".join(lines)
            except Exception as e:
                return f"Cannot stop learning: {e}"

        if intent.label == "learn_pause":
            self._engine._session.pause_learning()
            return "Learning paused. Say 'resume learning' to continue."

        if intent.label == "learn_resume":
            self._engine._session.resume_learning()
            return "Learning resumed."

        if intent.label == "learn_annotate":
            self._engine._session.annotate(user_text)
            return "Annotation added."

        return "Unknown learning command."

    def _handle_skill_list(self) -> str:
        skills = self._engine._registry.list_all()
        if not skills:
            return "No skills registered."
        lines = [f"Registered skills ({len(skills)}):\n"]
        for s in skills:
            meta = s.metadata
            lines.append(
                f"  - {s.name} (v{meta.version}, {meta.confidence:.0%}) — {s.description[:60]}"
            )
        return "\n".join(lines)

    async def _handle_skill_execute(self, user_text: str) -> str:
        if self._engine._session is None:
            triggered = await self._try_trigger_match(user_text)
            return triggered or "No matching skill found."

        skill_name = self._engine._session.find_skill(user_text)
        if skill_name is None:
            return "No matching skill found for your request."

        result = await self._engine._session.execute_skill(skill_name)
        if result.ok:
            return f"Skill '{result.skill_name}' executed successfully.\n{result.output or ''}"
        return f"Skill '{result.skill_name}' failed: {result.error}"

    def _is_teach_command(self, text: str) -> bool:
        """Check if text is a teach command that needs special session handling.

        Uses regex matching to avoid false positives like 'teach me how to cook'
        which should go through the unified tool loop.
        """
        stripped = text.strip()
        return bool(self._TEACH_COMMAND_RE.match(stripped))

    async def _handle_learn_command(self, user_text: str) -> str:
        """Route learn/teach commands through intent classifier for sub-intent dispatch."""
        intent = await self._engine._classifier.classify(user_text)
        logger.debug("learn.classify label=%s reason=%s", intent.label, intent.reason)

        if intent.label in (
            "learn_start",
            "learn_stop",
            "learn_pause",
            "learn_resume",
            "learn_annotate",
        ):
            return await self._handle_learn_intent(intent, user_text)

        # Not actually a learn command after classification — fall through to unified loop
        return await self._engine._unified_tool_loop(user_text)

    def _inject_pending_skill_reminder(self) -> None:
        if self._engine._skill_library is None:
            return
        n = self._engine._skill_library.count_pending()
        if n > 0:
            self._engine._wm.remember_event(
                "skill_suggestion_reminder",
                f"[{n} skill update suggestion(s) pending review — say 'review skill suggestions']",
            )

    def _handle_skill_review(self) -> str:
        if self._engine._skill_library is None:
            return "Skill library is not configured."
        suggestions = self._engine._skill_library.load_pending_suggestions(limit=10)
        if not suggestions:
            return "No pending skill update suggestions."
        lines = [f"Pending skill suggestions ({len(suggestions)}):\n"]
        for i, s in enumerate(suggestions, 1):
            details = s.similarity_details
            rationale = details.get("llm_rationale", "")
            changes = s.proposed_changes
            lines.append(
                f'  {i}. "{s.existing_skill_title}" (similarity: {s.similarity_score:.0%})'
            )
            if rationale:
                lines.append(f"     LLM: {rationale}")
            new_steps = changes.get("new_steps", [])
            new_triggers = changes.get("new_triggers", [])
            if new_steps:
                lines.append(f"     +steps: {', '.join(new_steps[:3])}")
            if new_triggers:
                lines.append(f"     +triggers: {', '.join(new_triggers[:3])}")
        lines.append("\nSay 'approve <number>' or 'reject <number>' to act.")
        return "\n".join(lines)

    async def _handle_skill_approve(self, user_text: str) -> str:
        if self._engine._skill_library is None:
            return "Skill library is not configured."
        suggestions = self._engine._skill_library.load_pending_suggestions(limit=20)
        if not suggestions:
            return "No pending suggestions to approve or reject."

        action, indices = await self._parse_approval(user_text, suggestions)

        results: list[str] = []
        for idx in indices:
            if idx < 0 or idx >= len(suggestions):
                results.append(f"Index {idx + 1} out of range.")
                continue
            s = suggestions[idx]
            if action == "approve":
                merged = self._engine._skill_merger.apply(s, self._engine._skill_library)
                results.append(f'Approved: "{s.existing_skill_title}" → v{merged.version}')
            else:
                self._engine._skill_library.resolve_suggestion(s.suggestion_id, "rejected")
                results.append(f'Rejected: "{s.existing_skill_title}"')
        return "\n".join(results)

    async def _parse_approval(self, user_text: str, suggestions: list) -> tuple[str, list[int]]:
        text_lower = user_text.lower()
        is_approve = any(w in text_lower for w in ("approve", "accept", "yes", "批准", "接受"))
        is_reject = any(w in text_lower for w in ("reject", "deny", "no", "拒绝"))
        action = "approve" if is_approve else ("reject" if is_reject else "approve")

        if "all" in text_lower or "全部" in text_lower:
            return action, list(range(len(suggestions)))

        nums = re.findall(r"\d+", user_text)
        indices = [int(n) - 1 for n in nums if 0 < int(n) <= len(suggestions)]
        if not indices:
            indices = [0]
        return action, indices

    def _evolution_action_context(self, action_id: str) -> Any:
        """Build causal identity for one action from the active session/frame.

        Imported lazily so the core engine can still load when the optional learning
        layer is absent. Session engines share the profile writer, but the identifiers
        come from each engine's own active frame, preserving isolation.
        """
        from leapflow.domain.evolution_event import EvolutionContext
        from leapflow.layout import workspace_id_for_path

        frame = self._engine._active_frame
        session_id = str(
            getattr(frame, "session_id", "") or self._engine._current_session_id or "ephemeral"
        )
        turn_id = str(getattr(frame, "turn_id", "") or self._engine._current_turn_id or "")
        command_id = str(
            getattr(frame, "command_id", "") or self._engine._current_command_id or turn_id
        )
        profile_layout = getattr(self._engine._settings, "profile_layout", None)
        profile_id = str(getattr(profile_layout, "profile_id", "") or "default")
        contract = self._engine._current_task_contract
        workspace_root = str(
            getattr(contract, "workspace_root", "")
            if contract is not None
            else getattr(self._engine._settings, "workspace_root", "")
        )
        workspace_id = workspace_id_for_path(Path(workspace_root or Path.cwd()))
        correlation_id = f"session:{profile_id}:{session_id}"
        return EvolutionContext(
            profile_id=profile_id,
            workspace_id=workspace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=command_id,
            action_id=str(action_id),
            correlation_id=correlation_id,
        )

    async def _execute_action_boundary(
        self,
        *,
        action_type: str,
        action_name: str,
        arguments: Dict[str, Any],
        execution_id: str,
        execution_policy: ExecutionPolicy,
        execute: Any,
    ) -> Any:
        """Delegate one operation to the shared no-LLM action executor."""
        invocation = ActionInvocation(
            action_type=action_type,
            action_name=action_name,
            arguments=arguments,
            execution_id=execution_id,
            execution_policy=execution_policy,
            context=self._evolution_action_context(execution_id),
            goal=str(getattr(self._engine._active_frame, "user_text", "") or ""),
        )
        return await self._engine._action_executor.execute(invocation, execute)

    async def execute_action(self, action: Dict[str, Any], user_goal: str) -> Any:
        a_type = str(action.get("type", "")).strip()
        name = str(action.get("name", "")).strip()
        payload = dict(action.get("payload") or {})

        # Memory tool interception: route memory_* calls to MemoryManager.
        if (a_type == "memory" or name.startswith("memory_")) and self._engine._memory_manager:
            tool_name = name if name.startswith("memory_") else f"memory_{name}"
            workspace_root = (
                self._engine._current_task_contract.workspace_root
                if self._engine._current_task_contract
                else ""
            )

            async def _memory_action() -> Dict[str, Any]:
                try:
                    result = await self._engine._memory_manager.handle_tool_call(
                        tool_name, payload, workspace_root=workspace_root
                    )
                    logger.info("audit.memory_tool name=%s", tool_name)
                    return {"ok": True, "result": result}
                except Exception as exc:
                    return {"ok": False, "error": f"memory_tool_failed: {exc}"}

            return await self._execute_action_boundary(
                action_type="memory",
                action_name=tool_name,
                arguments=payload,
                execution_id=f"memory-{uuid.uuid4().hex}",
                execution_policy=normalize_execution_policy(
                    action.get("execution_policy"),
                    default="mutating_idempotent",
                ),
                execute=_memory_action,
            )

        if a_type == "skill":
            async def _skill_action() -> Dict[str, Any]:
                result = await self._engine._registry.invoke(
                    name,
                    user_goal=user_goal,
                    **payload,
                )
                if not result.ok:
                    return {"ok": False, "error": result.error}
                logger.info("audit.skill name=%s ok", name)
                return {"ok": True, "result": result.output}

            skill = self._engine._registry.get(name)
            skill_policy = normalize_execution_policy(
                getattr(getattr(skill, "metadata", None), "execution_policy", "")
            )
            return await self._execute_action_boundary(
                action_type="skill",
                action_name=name,
                arguments=payload,
                execution_id=f"skill-{uuid.uuid4().hex}",
                execution_policy=skill_policy,
                execute=_skill_action,
            )

        if a_type == "bridge":
            method = str(payload.pop("method", "")).strip()
            if not method:
                return {"ok": False, "error": "missing_method"}

            async def _bridge_action() -> Dict[str, Any]:
                result = await self._engine._rpc.call(method, payload or None)
                logger.info("audit.bridge method=%s", method)
                return {"ok": True, "result": result}

            return await self._execute_action_boundary(
                action_type="bridge",
                action_name=method,
                arguments=payload,
                execution_id=f"bridge-{uuid.uuid4().hex}",
                execution_policy=normalize_execution_policy(action.get("execution_policy")),
                execute=_bridge_action,
            )

        if a_type == "tool":
            tool_call_dict = {"name": name, "arguments": payload}
            result = await self._engine._tool_dispatch._execute_tool_with_ledger(
                tool_call_dict,
                self._engine._tool_dispatch._unified_tool_handlers(),
                tool_call_id=f"action-{name}",
            )
            logger.info("audit.tool name=%s ok=%s", name, result.get("ok"))
            return result

        return {"ok": False, "error": f"unsupported_action:{a_type}"}
