"""Episode intent inference using LLM or rule-based fallback.

Responsible for determining the user's high-level goal from
a sequence of semantic actions within an episode.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.domain.trajectory import ActionType, Episode, SemanticAction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Result of intent inference."""

    goal: str  # Concise intent description (<=30 chars ideally)
    confidence: float  # 0.0 - 1.0
    reasoning: str = ""  # Optional explanation
    category: str = ""  # Optional: "data_transfer", "file_ops", etc.


@runtime_checkable
class IntentInferrer(Protocol):
    """Interface for episode intent inference (DIP)."""

    async def infer(
        self, episode: Episode, context: Optional[Dict[str, Any]] = None
    ) -> InferenceResult: ...

    async def infer_batch(
        self, episodes: List[Episode], context: Optional[Dict[str, Any]] = None
    ) -> List[InferenceResult]: ...


# ── LLM-based inferrer ──


class LLMIntentInferrer:
    """LLM-based intent inference for episodes.

    Constructs a prompt from the episode's semantic actions and app sequence,
    calls LLM to generate a concise intent description.
    Supports batch inference to reduce API costs.
    """

    _SYSTEM_PROMPT = (
        "你是一个操作意图分析系统。分析用户的操作序列，为每个片段推断用户的高级意图。\n"
        "要求:\n"
        "- 每个意图描述不超过20个字\n"
        "- 返回严格JSON数组: [{\"goal\": \"...\", \"confidence\": 0.0-1.0, \"category\": \"...\"}]\n"
        "- category 可选值: file_ops, data_transfer, text_editing, web_browsing, "
        "app_management, communication, coding, media, system, other\n"
        "- 不要输出任何额外内容，只输出JSON"
    )

    def __init__(
        self,
        llm: Any,
        *,
        max_actions_per_episode: int = 20,
        language: str = "zh",
    ) -> None:
        """
        Args:
            llm: LLMProvider instance (must support achat()).
            max_actions_per_episode: Max actions to include in prompt.
            language: Response language ("zh" or "en").
        """
        self._llm = llm
        self._max_actions = max_actions_per_episode
        self._language = language

    async def infer(
        self, episode: Episode, context: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """Infer intent for a single episode."""
        results = await self.infer_batch([episode], context)
        return results[0]

    async def infer_batch(
        self,
        episodes: List[Episode],
        context: Optional[Dict[str, Any]] = None,
        *,
        on_chunk: Any = None,
    ) -> List[InferenceResult]:
        """Batch inference for multiple episodes (single LLM call)."""
        if not episodes:
            return []

        prompt = self._build_prompt(episodes, context)
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._llm.achat(
                messages, stream=True, on_chunk=on_chunk,
            )
            return self._parse_response(response.content, len(episodes))
        except Exception as e:
            logger.warning("LLM intent inference failed: %s", e)
            return [
                InferenceResult(goal="", confidence=0.0, reasoning=f"LLM error: {e}")
                for _ in episodes
            ]

    def _build_prompt(
        self, episodes: List[Episode], context: Optional[Dict[str, Any]]
    ) -> str:
        """Build inference prompt from episodes."""
        parts: List[str] = ["分析以下用户操作序列，为每个片段推断用户的高级意图。\n"]

        for idx, ep in enumerate(episodes, 1):
            parts.append(self._format_episode(ep, idx))

        if context:
            parts.append(f"\n附加上下文: {json.dumps(context, ensure_ascii=False)}")

        parts.append(
            f"\n请为以上{len(episodes)}个片段各返回一条推断结果，"
            "严格返回JSON数组格式。"
        )
        return "\n".join(parts)

    def _format_episode(self, episode: Episode, idx: int) -> str:
        """Format a single episode for the prompt."""
        lines = [f"片段{idx}:"]

        if episode.app_sequence:
            lines.append(f"  应用序列: {episode.app_sequence}")

        actions = episode.semantic_actions[: self._max_actions]
        if actions:
            action_descs = [
                f"    - {a.action_name}: {a.description}" for a in actions
            ]
            lines.append("  操作:")
            lines.extend(action_descs)

        if len(episode.semantic_actions) > self._max_actions:
            lines.append(
                f"    ... (共{len(episode.semantic_actions)}个操作，已截断)"
            )

        lines.append(f"  步骤数: {episode.action_count}")
        return "\n".join(lines)

    def _parse_response(self, response: str, count: int) -> List[InferenceResult]:
        """Parse LLM response into InferenceResults.

        Handles JSON extraction from potentially noisy LLM output.
        """
        # Try to extract JSON array from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in LLM response: %.100s...", response)
            return [InferenceResult(goal="", confidence=0.0) for _ in range(count)]

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s", e)
            return [InferenceResult(goal="", confidence=0.0) for _ in range(count)]

        results: List[InferenceResult] = []
        for i in range(count):
            if i < len(data) and isinstance(data[i], dict):
                item = data[i]
                results.append(
                    InferenceResult(
                        goal=str(item.get("goal", ""))[:30],
                        confidence=float(
                            min(1.0, max(0.0, item.get("confidence", 0.5)))
                        ),
                        category=str(item.get("category", "")),
                    )
                )
            else:
                results.append(InferenceResult(goal="", confidence=0.0))
        return results


# ── Rule-based fallback ──

# Action type to semantic domain mapping
_ACTION_DOMAIN: Dict[str, str] = {
    ActionType.FILE_CREATE.value: "file_ops",
    ActionType.FILE_MODIFY.value: "file_ops",
    ActionType.FILE_DELETE.value: "file_ops",
    ActionType.FILE_RENAME.value: "file_ops",
    ActionType.CLIPBOARD_COPY.value: "data_transfer",
    ActionType.APP_SWITCH.value: "navigation",
    ActionType.UI_CLICK.value: "ui_interaction",
    ActionType.UI_TYPE.value: "text_editing",
    ActionType.UI_SHORTCUT.value: "ui_interaction",
    ActionType.UI_SCROLL.value: "browsing",
}

# Domain to Chinese label
_DOMAIN_LABEL: Dict[str, str] = {
    "file_ops": "文件操作",
    "data_transfer": "数据传输",
    "navigation": "应用切换",
    "ui_interaction": "界面交互",
    "text_editing": "文本编辑",
    "browsing": "浏览",
}


class RuleBasedIntentInferrer:
    """Rule-based fallback when LLM is unavailable.

    Generates intent descriptions from:
    - App sequence pattern
    - Dominant action types
    - Known composite patterns (e.g., copy+switch+paste = transfer)
    """

    async def infer(
        self, episode: Episode, context: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """Heuristic intent inference."""
        goal = self._infer_from_actions(episode)
        confidence = self._compute_confidence(episode)
        category = self._determine_category(episode)
        return InferenceResult(goal=goal, confidence=confidence, category=category)

    async def infer_batch(
        self, episodes: List[Episode], context: Optional[Dict[str, Any]] = None
    ) -> List[InferenceResult]:
        """Batch heuristic inference — sync logic, no gather needed."""
        results: List[InferenceResult] = []
        for ep in episodes:
            goal = self._infer_from_actions(ep)
            confidence = self._compute_confidence(ep)
            category = self._determine_category(ep)
            results.append(InferenceResult(goal=goal, confidence=confidence, category=category))
        return results

    def _infer_from_actions(self, episode: Episode) -> str:
        """Generate description from action patterns.

        Rules applied in priority order:
        1. clipboard.copy + app.switch → "跨应用数据传输"
        2. Dominant file actions → "文件操作: {detail}"
        3. Multiple ui.type → "文本编辑: {target app}"
        4. Web-like patterns (scroll + click) → "网页浏览"
        5. Default → "{primary_app} 操作"
        """
        actions = episode.semantic_actions
        if not actions:
            return self._fallback_from_apps(episode)

        action_names = [a.action_name for a in actions]
        domain_counts = self._count_domains(action_names)

        # Rule 1: Cross-app data transfer
        if self._is_data_transfer(action_names, episode):
            return "跨应用数据传输"

        # Rule 2: Dominant file operations
        if domain_counts.get("file_ops", 0) >= len(actions) * 0.4:
            detail = self._file_detail(actions)
            return f"文件操作: {detail}" if detail else "文件操作"

        # Rule 3: Text editing
        if domain_counts.get("text_editing", 0) >= len(actions) * 0.3:
            app = episode.app_sequence[0] if episode.app_sequence else ""
            return f"文本编辑: {app}" if app else "文本编辑"

        # Rule 4: Browsing (lots of scroll + click, few type)
        browsing = domain_counts.get("browsing", 0) + domain_counts.get(
            "ui_interaction", 0
        )
        if browsing >= len(actions) * 0.5 and domain_counts.get("text_editing", 0) < 2:
            return "网页浏览/搜索"

        # Rule 5: Fallback
        return self._fallback_from_apps(episode)

    def _compute_confidence(self, episode: Episode) -> float:
        """Heuristic confidence based on pattern clarity.

        Capped at 0.5 — rule-based inference is pattern-matching only,
        never semantic understanding. This prevents downstream activation
        of low-quality candidates.
        """
        action_count = max(1, len(episode.semantic_actions))
        app_count = max(1, len(episode.app_sequence))

        base = 0.7 / (1.0 + 0.05 * action_count)
        focus_bonus = 0.1 if app_count == 1 else 0.0
        return min(0.5, base + focus_bonus)

    def _determine_category(self, episode: Episode) -> str:
        """Determine episode category from dominant domain."""
        actions = episode.semantic_actions
        if not actions:
            return "other"
        domain_counts = self._count_domains([a.action_name for a in actions])
        if not domain_counts:
            return "other"
        dominant = max(domain_counts, key=domain_counts.get)  # type: ignore[arg-type]
        return dominant

    @staticmethod
    def _count_domains(action_names: List[str]) -> Dict[str, int]:
        """Count occurrences of each semantic domain."""
        counts: Dict[str, int] = {}
        for name in action_names:
            domain = _ACTION_DOMAIN.get(name, "other")
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    @staticmethod
    def _is_data_transfer(action_names: List[str], episode: Episode) -> bool:
        """Detect cross-app data transfer pattern."""
        has_copy = any(
            ActionType.CLIPBOARD_COPY.value in n for n in action_names
        )
        has_switch = any(ActionType.APP_SWITCH.value in n for n in action_names)
        multi_app = len(episode.app_sequence) >= 2
        return has_copy and (has_switch or multi_app)

    @staticmethod
    def _file_detail(actions: List[SemanticAction]) -> str:
        """Describe file operation pattern (type-based, not filename-based)."""
        op_counts: Dict[str, int] = {}
        for a in actions:
            name = a.action_name.lower()
            if "rename" in name:
                op_counts["重命名"] = op_counts.get("重命名", 0) + 1
            elif "create" in name:
                op_counts["创建"] = op_counts.get("创建", 0) + 1
            elif "delete" in name:
                op_counts["删除"] = op_counts.get("删除", 0) + 1
            elif "modify" in name:
                op_counts["修改"] = op_counts.get("修改", 0) + 1
            elif "move" in name:
                op_counts["移动"] = op_counts.get("移动", 0) + 1
        if not op_counts:
            return ""
        primary = max(op_counts, key=op_counts.get)
        count = op_counts[primary]
        if count > 1:
            return f"批量{primary}"
        return primary

    @staticmethod
    def _fallback_from_apps(episode: Episode) -> str:
        """Generate fallback description from app sequence."""
        if episode.app_sequence:
            primary = episode.app_sequence[0]
            # Strip bundle ID prefix for readability
            short = primary.rsplit(".", 1)[-1] if "." in primary else primary
            return f"{short} 操作"
        return "未知操作"
