"""LLM-based intent classification module.

Design:
- IntentClassifier Protocol (DIP: engine depends on this abstraction)
- LLMIntentClassifier uses a single LLM call with declarative intent specs
- Intent specs are data (OCP: add new intents without modifying logic)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text

logger = logging.getLogger(__name__)

IntentLabel = Literal[
    "conversational",
    "memory_recent",
    "file_organize",
    "clipboard",
    "app_automation",
    "file_search",
    "recording_start",
    "recording_stop",
    "recording_analyze",
    "learn_start",
    "learn_stop",
    "learn_pause",
    "learn_resume",
    "learn_annotate",
    "skill_list",
    "skill_execute",
    "skill_review",
    "skill_approve",
    "execute_confirm",
    "execute_skip",
    "execute_stop",
    "desktop_action",
    "complex",
]


@dataclass(frozen=True)
class Intent:
    """Classified intent with rationale."""

    label: IntentLabel
    reason: str


@dataclass(frozen=True)
class IntentSpec:
    """Declarative definition of a single intent category."""

    label: IntentLabel
    description: str


INTENT_SPECS: Sequence[IntentSpec] = (
    IntentSpec("conversational", "greetings, identity questions, thanks, help requests, general chat"),
    IntentSpec("memory_recent", "recent changes, what just happened, recent file/clipboard/app activity"),
    IntentSpec("file_organize", "organize, sort, rename, or categorize files and folders"),
    IntentSpec("clipboard", "read, summarize, or act on clipboard content"),
    IntentSpec("app_automation", "launch, open, activate, or control applications"),
    IntentSpec("file_search", "find or search files by content, name, or time range"),
    IntentSpec("recording_start", "start recording, begin observation, 开始录制, start demonstration"),
    IntentSpec("recording_stop", "stop recording, end observation, 停止录制, finish demonstration"),
    IntentSpec("recording_analyze", "analyze recording, learn skill, distill, 分析录制, 学习技能, replay trajectory"),
    IntentSpec("learn_start", "start learning, begin learning, learn this, 开始学习, start demonstration, watch me"),
    IntentSpec("learn_stop", "stop learning, end learning, done learning, 停止学习, 结束学习, finish learning"),
    IntentSpec("learn_pause", "pause learning, pause recording, 暂停学习, 暂停"),
    IntentSpec("learn_resume", "resume learning, continue learning, 继续学习, 恢复学习"),
    IntentSpec("learn_annotate", "annotate step, mark step, add note, 标注, 注释"),
    IntentSpec("skill_list", "list skills, show skills, what can you do, 列出技能, 技能列表"),
    IntentSpec("skill_execute", "run skill, execute skill, do task, perform, 执行技能, 运行"),
    IntentSpec("skill_review", "review skill suggestions, pending updates, skill changes, 查看技能建议, 审核建议, review skills"),
    IntentSpec("skill_approve", "approve/reject skill update, accept/reject suggestion, 接受建议, 拒绝建议, approve skill, reject skill"),
    IntentSpec("execute_confirm", "yes, confirm, continue, next, 确认, 继续, 好, proceed, go ahead"),
    IntentSpec("execute_skip", "skip, skip this step, 跳过"),
    IntentSpec("execute_stop", "stop, cancel, abort, halt, 停止, 取消, 中止"),
    IntentSpec(
        "desktop_action",
        "operate apps, type text, click UI elements, open URLs/tabs, switch windows, "
        "send keyboard shortcuts, or any task requiring direct interaction with the desktop GUI",
    ),
    IntentSpec("complex", "multi-step tasks requiring planning and multiple tool calls"),
)

_VALID_LABELS: frozenset[str] = frozenset(spec.label for spec in INTENT_SPECS)


def _build_classifier_prompt(specs: Sequence[IntentSpec]) -> str:
    lines = [f"- {s.label}: {s.description}" for s in specs]
    return (
        "Classify the user message into exactly one intent label.\n\n"
        "Labels:\n" + "\n".join(lines) + "\n\n"
        "Return ONLY a JSON object: {\"label\":\"...\",\"reason\":\"...\"}\n"
        "The reason should be one short phrase. No other text."
    )


_CLASSIFIER_PROMPT = _build_classifier_prompt(INTENT_SPECS)


@runtime_checkable
class IntentClassifier(Protocol):
    """Abstract intent classifier (DIP: engine depends on this)."""

    async def classify(self, user_text: str) -> Intent: ...


class LLMIntentClassifier:
    """Classifies user intent via a single LLM call."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def classify(self, user_text: str) -> Intent:
        messages = [
            build_system_message(_CLASSIFIER_PROMPT),
            build_user_message_text(user_text),
        ]
        try:
            resp = await self._llm.achat(messages, stream=False, enable_thinking=False)
            return self._parse_response(resp.content or "")
        except Exception:
            logger.debug("Intent classification failed; defaulting to complex", exc_info=True)
            return Intent(label="complex", reason="classifier_error")

    def _parse_response(self, raw: str) -> Intent:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return Intent(label="complex", reason="no_json_in_response")
        data = json.loads(raw[start : end + 1])
        label = str(data.get("label", "complex")).strip()
        reason = str(data.get("reason", "llm")).strip()
        if label not in _VALID_LABELS:
            return Intent(label="complex", reason=f"unknown_label:{label}")
        return Intent(label=label, reason=reason)  # type: ignore[arg-type]


class FallbackClassifier:
    """Returns 'complex' for every input (used when LLM is unavailable)."""

    async def classify(self, user_text: str) -> Intent:
        return Intent(label="complex", reason="no_llm_available")
