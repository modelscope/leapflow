"""Progressive context disclosure for unified agent turns.

This module decides how much runtime context a turn should disclose before the
provider call. It does not answer the user, execute tools, or split the agent
runtime into multiple loops.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class DisclosureLevel(str, Enum):
    """Stable disclosure levels understood by the unified loop."""

    LIGHT = "light"
    INDEXED_CAPABILITIES = "indexed_capabilities"
    SELECTED_TOOLS = "selected_tools"
    FULL = "full"


class MemoryDisclosure(str, Enum):
    """Memory disclosure policy for a turn."""

    NONE = "none"
    SESSION_SUMMARY = "session_summary"
    QUERY_RETRIEVAL = "query_retrieval"
    TASK_RETRIEVAL = "task_retrieval"


class HistoryDisclosure(str, Enum):
    """Prior-turn disclosure policy for a turn."""

    SHORT = "short"
    RECENT = "recent"
    FULL_WINDOW = "full_window"


class ReasoningDisclosure(str, Enum):
    """Provider reasoning mode requested by the plan."""

    OFF = "off"
    AUTO = "auto"
    ON = "on"


@dataclass(frozen=True)
class CapabilityManifest:
    """Compact runtime-facing capability metadata derived from tool schemas."""

    name: str
    category: str
    summary: str
    input_signals: tuple[str, ...] = ()
    risk_level: str = "read_only"
    requires_approval: bool = False
    schema_cost: str = "medium"

    @classmethod
    def from_tool_definition(cls, tool_definition: Mapping[str, Any]) -> "CapabilityManifest":
        """Derive a manifest from an OpenAI-style tool definition."""
        function = tool_definition.get("function", {}) if isinstance(tool_definition, Mapping) else {}
        name = str(function.get("name") or tool_definition.get("name") or "")
        description = str(function.get("description") or "")
        raw_metadata = (
            tool_definition.get("x_leapflow")
            or tool_definition.get("x-leapflow")
            or function.get("x_leapflow")
            or function.get("x-leapflow")
            or {}
        )
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        category = str(metadata.get("category") or _infer_category(name, description))
        signals = _metadata_signals(metadata.get("input_signals")) or _signals_for_category(category)
        risk_level = str(metadata.get("risk_level") or _risk_for_category(category))
        requires_approval = bool(metadata.get("requires_approval", category in {"write", "shell", "gateway"}))
        schema_cost = str(metadata.get("schema_cost") or ("high" if category in {"hub", "gateway", "delegate"} else "medium"))
        return cls(
            name=name,
            category=category,
            summary=str(metadata.get("summary") or description),
            input_signals=signals,
            risk_level=risk_level,
            requires_approval=requires_approval,
            schema_cost=schema_cost,
        )


@dataclass(frozen=True)
class PromptAssemblyPlan:
    """Provider payload plan consumed by the unified loop."""

    level: DisclosureLevel
    tool_definitions: tuple[Mapping[str, Any], ...] = ()
    catalog_definitions: tuple[Mapping[str, Any], ...] = ()
    memory: MemoryDisclosure = MemoryDisclosure.NONE
    history: HistoryDisclosure = HistoryDisclosure.SHORT
    reasoning: ReasoningDisclosure = ReasoningDisclosure.OFF
    native_tools: bool = False
    stream_mode: str = "direct"
    risk_level: str = "none"
    reason: str = ""
    selected_tool_names: tuple[str, ...] = ()
    max_prior_turns: int = 2

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable disclosure summary."""
        return {
            "level": self.level.value,
            "reason": self.reason,
            "tools": list(self.selected_tool_names),
            "tool_count": len(self.selected_tool_names),
            "memory": self.memory.value,
            "history": self.history.value,
            "reasoning": self.reasoning.value,
            "native_tools": self.native_tools,
            "stream_mode": self.stream_mode,
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True)
class DisclosureRuntimeState:
    """Low-cost signals available before assembling the provider payload."""

    enable_thinking: bool = False
    native_tools_enabled: bool = False
    slash_command: bool = False
    context_posture: str = "baseline"
    recent_failure: bool = False


@dataclass(frozen=True)
class DisclosurePlanner:
    """Manifest-driven planner for progressive context disclosure."""

    manifests: tuple[CapabilityManifest, ...] = field(default_factory=tuple)

    def plan(
        self,
        user_text: str,
        tool_definitions: Sequence[Mapping[str, Any]],
        runtime: DisclosureRuntimeState,
    ) -> PromptAssemblyPlan:
        """Build a prompt assembly plan without invoking another LLM."""
        manifests = self.manifests or tuple(CapabilityManifest.from_tool_definition(td) for td in tool_definitions)
        if runtime.slash_command or runtime.context_posture in {"research", "finalizing"}:
            return self.full_plan(user_text, tool_definitions, runtime, "runtime posture requires full agent context")

        if _asks_about_capabilities(user_text):
            return PromptAssemblyPlan(
                level=DisclosureLevel.INDEXED_CAPABILITIES,
                catalog_definitions=tuple(tool_definitions),
                memory=MemoryDisclosure.SESSION_SUMMARY,
                history=HistoryDisclosure.SHORT,
                reasoning=ReasoningDisclosure.OFF,
                native_tools=False,
                stream_mode="direct",
                risk_level="none",
                reason="capability question needs compact index but no executable schemas",
                selected_tool_names=tuple(_tool_name(td) for td in tool_definitions if _tool_name(td)),
                max_prior_turns=2,
            )

        selected = self._select_capabilities(user_text, manifests)
        if self._needs_full_context(user_text, selected, runtime):
            return self.full_plan(user_text, tool_definitions, runtime, "task requires broad execution context")

        if selected:
            selected_names = {manifest.name for manifest in selected}
            selected_defs = tuple(td for td in tool_definitions if _tool_name(td) in selected_names)
            risk = _highest_risk(selected)
            return PromptAssemblyPlan(
                level=DisclosureLevel.SELECTED_TOOLS,
                tool_definitions=selected_defs,
                catalog_definitions=selected_defs,
                memory=MemoryDisclosure.QUERY_RETRIEVAL if _needs_memory(user_text, selected) else MemoryDisclosure.NONE,
                history=HistoryDisclosure.RECENT,
                reasoning=ReasoningDisclosure.AUTO if runtime.enable_thinking else ReasoningDisclosure.OFF,
                native_tools=runtime.native_tools_enabled and bool(selected_defs),
                stream_mode="tool_aware",
                risk_level=risk,
                reason="selected capabilities matched observable task signals",
                selected_tool_names=tuple(sorted(selected_names)),
                max_prior_turns=6,
            )

        return PromptAssemblyPlan(
            level=DisclosureLevel.LIGHT,
            memory=MemoryDisclosure.NONE,
            history=HistoryDisclosure.SHORT,
            reasoning=ReasoningDisclosure.OFF,
            native_tools=False,
            stream_mode="direct",
            risk_level="none",
            reason="no external capability signal detected",
            max_prior_turns=2,
        )

    def full_plan(
        self,
        user_text: str,
        tool_definitions: Sequence[Mapping[str, Any]],
        runtime: DisclosureRuntimeState,
        reason: str,
    ) -> PromptAssemblyPlan:
        """Build a full-disclosure fallback plan for safety and compatibility."""
        names = tuple(_tool_name(td) for td in tool_definitions if _tool_name(td))
        return PromptAssemblyPlan(
            level=DisclosureLevel.FULL,
            tool_definitions=tuple(tool_definitions),
            catalog_definitions=tuple(tool_definitions),
            memory=MemoryDisclosure.TASK_RETRIEVAL,
            history=HistoryDisclosure.FULL_WINDOW,
            reasoning=ReasoningDisclosure.AUTO if runtime.enable_thinking else ReasoningDisclosure.OFF,
            native_tools=runtime.native_tools_enabled and bool(tool_definitions),
            stream_mode="tool_aware",
            risk_level="high" if _has_mutation_signal(user_text) else "medium",
            reason=reason,
            selected_tool_names=names,
            max_prior_turns=10,
        )

    def _select_capabilities(
        self,
        user_text: str,
        manifests: Sequence[CapabilityManifest],
    ) -> tuple[CapabilityManifest, ...]:
        normalized = _normalize(user_text)
        selected: list[CapabilityManifest] = []
        for manifest in manifests:
            haystack = " ".join((manifest.name, manifest.category, manifest.summary, *manifest.input_signals)).lower()
            if any(signal and signal in normalized for signal in manifest.input_signals):
                selected.append(manifest)
                continue
            if any(token and token in normalized for token in _name_tokens(manifest.name)):
                selected.append(manifest)
                continue
            if manifest.category in {"hub", "gateway"} and any(token in normalized for token in ("hub", "gateway", "message", "send")):
                selected.append(manifest)
                continue
            if manifest.category == "delegate" and any(token in normalized for token in ("delegate", "subagent", "parallel", "子任务")):
                selected.append(manifest)
                continue
            if any(token in normalized for token in _description_tokens(haystack)):
                selected.append(manifest)
        return tuple(_dedupe_by_name(selected))

    def _needs_full_context(
        self,
        user_text: str,
        selected: Sequence[CapabilityManifest],
        runtime: DisclosureRuntimeState,
    ) -> bool:
        normalized = _normalize(user_text)
        if runtime.recent_failure:
            return True
        if _has_complexity_signal(normalized):
            return True
        risky_selected = any(item.requires_approval for item in selected)
        return risky_selected and _has_mutation_signal(normalized)


def build_capability_manifests(tool_definitions: Sequence[Mapping[str, Any]]) -> tuple[CapabilityManifest, ...]:
    """Build manifests for all known tools."""
    return tuple(CapabilityManifest.from_tool_definition(tool) for tool in tool_definitions)


def _tool_name(tool_definition: Mapping[str, Any]) -> str:
    function = tool_definition.get("function", {}) if isinstance(tool_definition, Mapping) else {}
    return str(function.get("name") or tool_definition.get("name") or "")


def _metadata_signals(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item).lower().strip() for item in value if str(item).strip())


def _infer_category(name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    if any(token in text for token in ("write", "replace", "delete", "store", "add")):
        return "write"
    if any(token in text for token in ("shell", "command", "execute")):
        return "shell"
    if any(token in text for token in ("file", "directory", "path")):
        return "file"
    if "memory" in text:
        return "memory"
    if "skill" in text:
        return "skill"
    if "delegate" in text or "subagent" in text:
        return "delegate"
    if "hub" in text:
        return "hub"
    if "gateway" in text or "message" in text:
        return "gateway"
    if "time" in text or "environment" in text or "date" in text:
        return "system"
    return "general"


def _signals_for_category(category: str) -> tuple[str, ...]:
    signals = {
        "file": ("file", "path", "read", "list", "文件", "目录", "路径", ".py", ".md"),
        "write": ("write", "edit", "modify", "replace", "save", "store", "写", "改", "保存", "记住"),
        "shell": ("run", "command", "terminal", "execute", "pytest", "命令", "执行", "测试"),
        "memory": ("memory", "remember", "recall", "history", "记忆", "回忆"),
        "skill": ("skill", "capability", "能力", "技能"),
        "system": ("time", "date", "env", "environment", "时间", "环境"),
        "delegate": ("delegate", "subagent", "parallel", "子任务", "并行"),
        "hub": ("hub", "publish", "install", "marketplace"),
        "gateway": ("gateway", "message", "send", "notify", "发送", "通知"),
    }
    return signals.get(category, ())


def _risk_for_category(category: str) -> str:
    if category in {"write", "shell", "gateway"}:
        return "high"
    if category in {"delegate", "hub"}:
        return "medium"
    return "read_only"


def _normalize(text: str) -> str:
    return text.lower().strip()


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[_\W]+", name.lower()) if len(token) >= 3)


def _description_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"\W+", text.lower()) if len(token) >= 8)


def _dedupe_by_name(manifests: Iterable[CapabilityManifest]) -> list[CapabilityManifest]:
    seen: set[str] = set()
    result: list[CapabilityManifest] = []
    for manifest in manifests:
        if manifest.name and manifest.name not in seen:
            seen.add(manifest.name)
            result.append(manifest)
    return result


def _highest_risk(manifests: Sequence[CapabilityManifest]) -> str:
    order = {"none": 0, "read_only": 1, "medium": 2, "high": 3}
    highest = "none"
    for manifest in manifests:
        if order.get(manifest.risk_level, 0) > order.get(highest, 0):
            highest = manifest.risk_level
    return highest


def _asks_about_capabilities(text: str) -> bool:
    normalized = _normalize(text)
    return any(token in normalized for token in ("what can you do", "capabilities", "skills", "tools", "你能做", "有哪些能力", "技能", "工具"))


def _needs_memory(text: str, manifests: Sequence[CapabilityManifest]) -> bool:
    normalized = _normalize(text)
    return any(manifest.category == "memory" for manifest in manifests) or any(token in normalized for token in ("memory", "remember", "recall", "history", "记忆", "之前"))


def _has_complexity_signal(normalized_text: str) -> bool:
    return any(
        token in normalized_text
        for token in (
            "implement", "refactor", "debug", "root cause", "architecture", "design", "analyze", "test",
            "实现", "重构", "调试", "根因", "架构", "设计", "深入分析", "执行", "代码", "测试",
        )
    )


def _has_mutation_signal(text: str) -> bool:
    normalized = _normalize(text)
    return any(token in normalized for token in ("write", "edit", "modify", "delete", "send", "execute", "写", "改", "删", "发送", "执行"))
