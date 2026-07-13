"""Tool registry and name-resolution primitives.

This module centralizes the Tool Capability Contract: the set of canonical
tool names the runtime can actually dispatch, plus structured unknown-tool
feedback when a proposed name is not part of that contract. There is no
alias table and no argument-shape guessing here — a proposed tool name is
either the exact canonical name (optionally normalized for case/separator
formatting or an internal bridge prefix), or it is reported as unknown with
discovery suggestions. It is intentionally execution-free: callers resolve a
tool name here, then execute through their existing dispatch path.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Sequence

ResolutionStatus = Literal["exact", "normalized", "unknown"]
ResolutionConfidence = Literal["high", "medium", "low"]
RiskLevel = Literal["read_only", "mutating", "external"]

_READ_ONLY_TOOLS = {
    "file_list",
    "file_read",
    "time_get",
    "env_info",
    "text_search",
    "skills_list",
    "skill_view",
    "memory_search",
}
_MUTATING_NAME_SIGNALS = (
    "write",
    "replace",
    "delete",
    "move",
    "copy",
    "create",
    "add",
    "send",
    "post",
    "run",
    "shell",
    "delegate",
)

def tool_lookup_key(tool_name: str) -> str:
    """Return a stable lookup key for exact tool-name identity matching.

    This only normalizes formatting (case, hyphen/space vs underscore) of the
    *same* canonical identifier. It never maps one tool name onto another.
    """
    return tool_name.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class ToolSpec:
    """Canonical metadata for a registered tool."""

    name: str
    description: str = ""
    parameters: frozenset[str] = field(default_factory=frozenset)
    required: frozenset[str] = field(default_factory=frozenset)
    risk_level: RiskLevel = "read_only"


@dataclass(frozen=True)
class ToolResolution:
    """Result of resolving an LLM-proposed tool name."""

    original_name: str
    normalized_name: str | None
    status: ResolutionStatus
    confidence: ResolutionConfidence
    reason: str
    suggestions: tuple[str, ...] = ()
    auto_executable: bool = False
    risk_level: RiskLevel = "read_only"

    @property
    def is_resolved(self) -> bool:
        """Return whether the resolution has a canonical target."""
        return self.normalized_name is not None and self.status != "unknown"

    def to_metadata(self) -> Dict[str, Any]:
        """Return compact metadata suitable for TUI logs and trace events."""
        metadata: Dict[str, Any] = {
            "original_tool_name": self.original_name,
            "tool_resolution_status": self.status,
            "tool_resolution_confidence": self.confidence,
            "tool_resolution_reason": self.reason,
            "tool_risk_level": self.risk_level,
            "tool_auto_executable": self.auto_executable,
        }
        if self.normalized_name is not None:
            metadata["normalized_tool_name"] = self.normalized_name
            if self.normalized_name != self.original_name:
                metadata["resolved_from"] = self.original_name
        if self.suggestions:
            metadata["tool_suggestions"] = list(self.suggestions)
        return metadata


@dataclass(frozen=True)
class ToolRegistry:
    """Single source of truth for tool identity and resolution metadata.

    The registry enforces a strict Tool Capability Contract: a proposed tool
    name is only ever executed when it is the exact canonical name, or a
    formatting-only variant of it (case, separators, or the internal ``gp_``
    bridge prefix). There is no alias table and no argument-shape guessing;
    anything else is reported as unknown so the caller can retry with an
    exact name from the disclosed capability list.
    """

    specs: Mapping[str, ToolSpec]

    @classmethod
    def from_definitions(
        cls,
        tool_definitions: Sequence[Mapping[str, Any]],
        handlers: Mapping[str, Any],
        *,
        bridge_tools: Sequence[Mapping[str, Any]] = (),
    ) -> "ToolRegistry":
        """Build a registry from OpenAI schemas, dispatch handlers, and bridge metadata."""
        bridge_mutates = {
            str(tool.get("name", "")).removeprefix("gp_"): bool(tool.get("mutates_state", False))
            for tool in bridge_tools
        }
        specs: dict[str, ToolSpec] = {}
        for definition in tool_definitions:
            function = definition.get("function", {})
            name = str(function.get("name") or definition.get("name") or "")
            if not name:
                continue
            parameters_schema = function.get("parameters", {}) or {}
            properties = parameters_schema.get("properties", {}) or {}
            required = parameters_schema.get("required", []) or []
            specs[name] = ToolSpec(
                name=name,
                description=str(function.get("description") or definition.get("description") or ""),
                parameters=frozenset(str(key) for key in properties.keys()),
                required=frozenset(str(key) for key in required),
                risk_level=_infer_risk_level(name, bridge_mutates.get(name, False)),
            )
        for name in handlers.keys():
            canonical = str(name).removeprefix("gp_")
            if canonical and canonical not in specs:
                specs[canonical] = ToolSpec(
                    name=canonical,
                    risk_level=_infer_risk_level(canonical, bridge_mutates.get(canonical, False)),
                )
        return cls(specs=specs)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return all canonical tool names sorted for stable feedback."""
        return tuple(sorted(self.specs.keys()))

    def normalize_name(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """Return the canonical name when resolution is safely executable."""
        resolution = self.resolve(tool_name, arguments or {})
        return resolution.normalized_name if resolution.auto_executable and resolution.normalized_name else tool_name

    def resolve(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> ToolResolution:
        """Resolve a proposed tool call against the canonical tool contract."""
        original_name = str(tool_name or "")
        args = dict(arguments or {})
        key = tool_lookup_key(original_name)
        if original_name in self.specs:
            return self._resolution(original_name, original_name, "exact", "high", "canonical tool name")
        if key in self.specs:
            return self._resolution(original_name, key, "normalized", "high", "case or separator normalization")
        if key.startswith("gp_") and key[3:] in self.specs:
            return self._resolution(original_name, key[3:], "normalized", "high", "internal bridge prefix")

        suggestions = self._suggestions(original_name, args)
        return ToolResolution(
            original_name=original_name,
            normalized_name=None,
            status="unknown",
            confidence="low",
            reason="no exact canonical tool name match",
            suggestions=suggestions,
            auto_executable=False,
            risk_level="read_only",
        )

    def unknown_result(self, resolution: ToolResolution) -> Dict[str, Any]:
        """Build structured feedback for an unresolved tool name."""
        available = self.tool_names[:16]
        suggestions = resolution.suggestions or available[:5]
        recovery_hint = (
            f"'{resolution.original_name}' is not a registered tool. Retry once with an exact "
            f"name from: {', '.join(suggestions)}."
            if suggestions
            else f"'{resolution.original_name}' is not a registered tool and no close match was found; "
            "ask for clarification instead of guessing."
        )
        return {
            "ok": False,
            "error_type": "unknown_tool",
            "error": f"Unknown tool: {resolution.original_name}",
            "original_tool_name": resolution.original_name,
            "normalized_tool_name": resolution.normalized_name,
            "resolution_status": resolution.status,
            "resolution_confidence": resolution.confidence,
            "resolution_reason": resolution.reason,
            "suggestions": list(suggestions),
            "available_tools": list(available),
            "recovery_hint": recovery_hint,
            "retryable": True,
        }

    def _resolution(
        self,
        original_name: str,
        canonical: str,
        status: ResolutionStatus,
        confidence: ResolutionConfidence,
        reason: str,
    ) -> ToolResolution:
        spec = self.specs[canonical]
        return ToolResolution(
            original_name=original_name,
            normalized_name=canonical,
            status=status,
            confidence=confidence,
            reason=reason,
            suggestions=(canonical,) if canonical != original_name else (),
            auto_executable=True,
            risk_level=spec.risk_level,
        )

    def _suggestions(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        """Return non-executing discovery hints for an unknown tool name.

        These are surfaced to the caller for a retry with an exact name; they
        are never used to auto-execute a different tool.
        """
        names = list(self.tool_names)
        close = difflib.get_close_matches(tool_lookup_key(tool_name), names, n=5, cutoff=0.55)
        if close:
            return tuple(close)
        keys = {str(key) for key in arguments.keys()}
        shape_matches = [
            spec.name
            for spec in self.specs.values()
            if keys and keys.issubset(spec.parameters | spec.required)
        ]
        return tuple(shape_matches[:5])


def _infer_risk_level(name: str, bridge_mutates: bool) -> RiskLevel:
    if name.startswith("gateway_") or name.startswith("hub_"):
        return "external"
    if name in _READ_ONLY_TOOLS and not bridge_mutates:
        return "read_only"
    if bridge_mutates or any(signal in name for signal in _MUTATING_NAME_SIGNALS):
        return "mutating"
    return "read_only"
