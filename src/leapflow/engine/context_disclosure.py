"""Progressive context disclosure for unified agent turns.

This module decides how much runtime context a turn should disclose before the
provider call. It does not answer the user, execute tools, or split the agent
runtime into multiple loops.

Design contract (do not regress): disclosure decisions are derived *only* from
structural, deterministic runtime facts — capability manifests (static tool
metadata) and stable runtime gates (slash commands, context posture, recent
failures, prior-turn tool-category continuity). Reading the user's free-form
text to guess which tools "sound relevant" is explicitly forbidden here: that
approach has unbounded coverage gaps and was the root cause of LLMs inventing
non-existent tool names when their real need fell outside a keyword table.
The compact tool index (Tier 0) is therefore always disclosed in full, and a
static low-risk tool whitelist (Tier 0.5) is always native-callable, so a turn
never presents an empty or contradictory tool contract to the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class DisclosureLevel(str, Enum):
    """Stable disclosure levels understood by the unified loop.

    CORE: Tier 0 compact index + Tier 0.5 static low-risk tool whitelist.
        This is the floor — it is never empty and never omitted.
    EXPANDED: CORE plus Tier 1 categories opened by a structural gate
        (prior-turn continuity, or a model-initiated capability_expand call
        already reflected in this turn's native tool schema).
    FULL: All registered tool schemas, for high-stakes/broad-context turns
        (slash commands, research/converging/finalizing posture, recent
        failure recovery).
    """

    CORE = "core"
    EXPANDED = "expanded"
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
    """Compact runtime-facing capability metadata derived from tool schemas.

    All fields here are *static* tool properties, declared once at tool
    registration time (via the optional ``x_leapflow`` schema extension) or
    inferred from the tool's own name/description. None of them are derived
    from a given turn's user text.
    """

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
        signals = _metadata_signals(metadata.get("input_signals"))
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

    @property
    def is_core(self) -> bool:
        """Return whether this tool belongs in the always-on Tier 0.5 whitelist.

        Core tools are read-only and cheap to disclose (small schema). This is
        a static property of the tool, evaluated once — never a per-turn guess.
        """
        return self.risk_level == "read_only" and self.schema_cost != "high"


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
    expanded_categories: tuple[str, ...] = ()
    max_prior_turns: int = 2

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable disclosure summary."""
        return {
            "level": self.level.value,
            "reason": self.reason,
            "tools": list(self.selected_tool_names),
            "tool_count": len(self.selected_tool_names),
            "expanded_categories": list(self.expanded_categories),
            "memory": self.memory.value,
            "history": self.history.value,
            "reasoning": self.reasoning.value,
            "native_tools": self.native_tools,
            "stream_mode": self.stream_mode,
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True)
class DisclosureRuntimeState:
    """Low-cost, structural signals available before assembling the provider payload.

    Every field here is a deterministic runtime fact, never a text-matching
    verdict: enable_thinking is a user/runtime toggle, native_tools_enabled is
    a settings flag, slash_command/context_posture/recent_failure are
    structured session state, and last_turn_tool_categories is derived from
    the prior turn's actual tool_calls (not from re-reading its text).
    """

    enable_thinking: bool = False
    native_tools_enabled: bool = False
    slash_command: bool = False
    context_posture: str = "baseline"
    recent_failure: bool = False
    last_turn_tool_categories: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DisclosurePlanner:
    """Manifest-driven planner for progressive context disclosure.

    No natural-language fitting of any kind is performed here. Every branch
    decides purely from ``DisclosureRuntimeState`` (structural facts) and
    ``CapabilityManifest`` (static tool metadata).
    """

    manifests: tuple[CapabilityManifest, ...] = field(default_factory=tuple)

    def plan(
        self,
        tool_definitions: Sequence[Mapping[str, Any]],
        runtime: DisclosureRuntimeState,
    ) -> PromptAssemblyPlan:
        """Build a prompt assembly plan from structural runtime facts only."""
        manifests = self.manifests or build_capability_manifests(tool_definitions)
        manifest_by_name = {m.name: m for m in manifests if m.name}

        if runtime.slash_command or runtime.context_posture in {"research", "expanding", "converging", "finalizing"} or runtime.recent_failure:
            return self.full_plan(tool_definitions, runtime, _full_reason(runtime))

        core_defs, core_names = _core_whitelist(tool_definitions, manifest_by_name)
        expanded_defs: list[Mapping[str, Any]] = list(core_defs)
        expanded_names: set[str] = set(core_names)
        expanded_categories: list[str] = []

        for category in sorted(runtime.last_turn_tool_categories):
            if not category:
                continue
            matched = False
            for tool_definition in tool_definitions:
                name = _tool_name(tool_definition)
                manifest = manifest_by_name.get(name)
                if manifest and manifest.category == category and name not in expanded_names:
                    expanded_defs.append(tool_definition)
                    expanded_names.add(name)
                    matched = True
            if matched:
                expanded_categories.append(category)

        level = DisclosureLevel.EXPANDED if expanded_categories else DisclosureLevel.CORE
        reason = (
            f"tier1: continuity({', '.join(expanded_categories)})"
            if expanded_categories
            else "tier0/0.5: static core whitelist"
        )
        scoped_manifests = [manifest_by_name[name] for name in expanded_names if name in manifest_by_name]
        return PromptAssemblyPlan(
            level=level,
            tool_definitions=tuple(expanded_defs),
            catalog_definitions=tuple(tool_definitions),
            memory=MemoryDisclosure.QUERY_RETRIEVAL if expanded_categories else MemoryDisclosure.NONE,
            history=HistoryDisclosure.RECENT if expanded_categories else HistoryDisclosure.SHORT,
            # At the CORE floor (no Tier 1 category opened) skip reasoning entirely: a
            # turn that only needs the static low-risk whitelist is, by construction, not
            # complex enough to justify the added latency of provider-side reasoning.
            reasoning=(
                ReasoningDisclosure.AUTO
                if runtime.enable_thinking and expanded_categories
                else ReasoningDisclosure.OFF
            ),
            native_tools=runtime.native_tools_enabled and bool(expanded_defs),
            stream_mode="tool_aware" if expanded_categories else "direct",
            risk_level=_highest_risk(scoped_manifests),
            reason=reason,
            selected_tool_names=tuple(sorted(expanded_names)),
            expanded_categories=tuple(expanded_categories),
            max_prior_turns=6 if expanded_categories else 2,
        )

    def full_plan(
        self,
        tool_definitions: Sequence[Mapping[str, Any]],
        runtime: DisclosureRuntimeState,
        reason: str,
    ) -> PromptAssemblyPlan:
        """Build a full-disclosure fallback plan for safety and compatibility."""
        manifests = self.manifests or build_capability_manifests(tool_definitions)
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
            risk_level=_highest_risk(manifests),
            reason=reason,
            selected_tool_names=names,
            expanded_categories=tuple(sorted({m.category for m in manifests if m.category})),
            max_prior_turns=10,
        )


def build_capability_manifests(tool_definitions: Sequence[Mapping[str, Any]]) -> tuple[CapabilityManifest, ...]:
    """Build manifests for all known tools."""
    return tuple(CapabilityManifest.from_tool_definition(tool) for tool in tool_definitions)


def _core_whitelist(
    tool_definitions: Sequence[Mapping[str, Any]],
    manifest_by_name: Mapping[str, CapabilityManifest],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Return the static Tier 0.5 whitelist: always-on, low-risk, cheap-schema tools."""
    defs: list[Mapping[str, Any]] = []
    names: list[str] = []
    for tool_definition in tool_definitions:
        name = _tool_name(tool_definition)
        manifest = manifest_by_name.get(name)
        if manifest is not None and manifest.is_core:
            defs.append(tool_definition)
            names.append(name)
    return defs, names


def _full_reason(runtime: DisclosureRuntimeState) -> str:
    if runtime.slash_command:
        return "gate: slash_command"
    if runtime.recent_failure:
        return "gate: recent_failure"
    return f"gate: posture({runtime.context_posture})"


def _tool_name(tool_definition: Mapping[str, Any]) -> str:
    function = tool_definition.get("function", {}) if isinstance(tool_definition, Mapping) else {}
    return str(function.get("name") or tool_definition.get("name") or "")


def _metadata_signals(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item).lower().strip() for item in value if str(item).strip())


def _infer_category(name: str, description: str) -> str:
    """Best-effort category guess for tools that declare no explicit x_leapflow.

    This is purely a *bootstrap convenience* for a handful of well-known,
    already-audited built-in tools — it must never be the mechanism that
    grants a brand-new, unaudited tool core-whitelist eligibility. Anything
    that does not match one of the recognized safe keyword patterns below
    falls through to "unclassified", which `_risk_for_category` deliberately
    treats as non-core by default (fail-closed): a future tool added without
    explicit metadata must be reviewed and opted in, not silently trusted.
    """
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
    return "unclassified"


def _risk_for_category(category: str) -> str:
    """Map a category to its default risk level.

    ``unclassified`` deliberately does *not* fall through to "read_only": a
    tool that could not be matched against any recognized safe keyword
    pattern (and declared no explicit ``x_leapflow`` metadata) must not be
    silently granted Tier 0.5 core-whitelist eligibility. Only categories
    that have been explicitly reviewed as safe reach "read_only" here.
    """
    if category in {"write", "shell", "gateway"}:
        return "high"
    if category in {"delegate", "hub", "unclassified"}:
        return "medium"
    return "read_only"


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
