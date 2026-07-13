"""Tool registry and name-resolution primitives.

This module centralizes tool identity, safe aliases, risk metadata, and
structured unknown-tool feedback. It is intentionally execution-free: callers
resolve a tool name here, then execute through their existing dispatch path.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Sequence

ResolutionStatus = Literal["exact", "alias", "parameter_match", "unknown", "ambiguous"]
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

_DEFAULT_ALIASES: Mapping[str, str] = {
    "list_dir": "file_list",
    "list_directory": "file_list",
    "list_files": "file_list",
    "directory_list": "file_list",
    "file_ls": "file_list",
    "ls": "file_list",
    "dir": "file_list",
    "read_file": "file_read",
    "open_file": "file_read",
    "cat_file": "file_read",
    "view_file": "file_read",
    "write_file": "file_write",
    "save_file": "file_write",
    "search_text": "text_search",
    "grep_text": "text_search",
    "replace_text": "text_replace",
    "execute_command": "shell_run",
    "execute_shell": "shell_run",
    "execute_shell_command": "shell_run",
    "run_command": "shell_run",
    "run_shell": "shell_run",
    "shell": "shell_run",
    "bash": "shell_run",
    "terminal_command": "shell_run",
}


def tool_alias_key(tool_name: str) -> str:
    """Return a stable lookup key for tool-name matching."""
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
        return self.normalized_name is not None and self.status not in {"unknown", "ambiguous"}

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
                metadata["alias"] = self.original_name
        if self.suggestions:
            metadata["tool_suggestions"] = list(self.suggestions)
        return metadata


@dataclass(frozen=True)
class ToolRegistry:
    """Single source of truth for tool identity and resolution metadata."""

    specs: Mapping[str, ToolSpec]
    aliases: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_definitions(
        cls,
        tool_definitions: Sequence[Mapping[str, Any]],
        handlers: Mapping[str, Any],
        *,
        bridge_tools: Sequence[Mapping[str, Any]] = (),
        aliases: Mapping[str, str] | None = None,
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
        alias_source = aliases or _DEFAULT_ALIASES
        normalized_aliases = {
            tool_alias_key(alias): canonical
            for alias, canonical in alias_source.items()
            if canonical in specs
        }
        return cls(specs=specs, aliases=normalized_aliases)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return all canonical tool names sorted for stable feedback."""
        return tuple(sorted(self.specs.keys()))

    def normalize_name(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """Return the canonical name when resolution is safely executable."""
        resolution = self.resolve(tool_name, arguments or {})
        return resolution.normalized_name if resolution.auto_executable and resolution.normalized_name else tool_name

    def resolve(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> ToolResolution:
        """Resolve a proposed tool call against canonical tool metadata."""
        original_name = str(tool_name or "")
        args = dict(arguments or {})
        key = tool_alias_key(original_name)
        if original_name in self.specs:
            return self._resolution(original_name, original_name, "exact", "high", "canonical tool name")
        if key in self.specs:
            canonical = key
            return self._resolution(original_name, canonical, "alias", "high", "case or separator normalization")
        if key.startswith("gp_") and key[3:] in self.specs:
            canonical = key[3:]
            return self._resolution(original_name, canonical, "alias", "high", "gp-prefixed bridge alias")
        alias_target = self.aliases.get(key)
        if alias_target:
            return self._resolution(original_name, alias_target, "alias", "high", "declared tool alias")

        parameter_candidates = self._parameter_candidates(original_name, args)
        if len(parameter_candidates) == 1:
            canonical, confidence, reason = parameter_candidates[0]
            spec = self.specs[canonical]
            auto = spec.risk_level == "read_only" and confidence in {"high", "medium"}
            return ToolResolution(
                original_name=original_name,
                normalized_name=canonical,
                status="parameter_match",
                confidence=confidence,
                reason=reason,
                suggestions=(canonical,),
                auto_executable=auto,
                risk_level=spec.risk_level,
            )
        if len(parameter_candidates) > 1:
            suggestions = tuple(candidate[0] for candidate in parameter_candidates[:5])
            return ToolResolution(
                original_name=original_name,
                normalized_name=None,
                status="ambiguous",
                confidence="low",
                reason="arguments match multiple tool shapes",
                suggestions=suggestions,
                auto_executable=False,
                risk_level="read_only",
            )

        suggestions = self._suggestions(original_name, args)
        return ToolResolution(
            original_name=original_name,
            normalized_name=None,
            status="unknown",
            confidence="low",
            reason="no canonical tool, alias, or safe parameter match",
            suggestions=suggestions,
            auto_executable=False,
            risk_level="read_only",
        )

    def unknown_result(self, resolution: ToolResolution) -> Dict[str, Any]:
        """Build structured feedback for unknown or ambiguous tool names."""
        available = self.tool_names[:16]
        suggestions = resolution.suggestions or available[:5]
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
        auto = confidence == "high" or spec.risk_level == "read_only"
        return ToolResolution(
            original_name=original_name,
            normalized_name=canonical,
            status=status,
            confidence=confidence,
            reason=reason,
            suggestions=(canonical,) if canonical != original_name else (),
            auto_executable=auto,
            risk_level=spec.risk_level,
        )

    def _parameter_candidates(
        self, tool_name: str, arguments: Mapping[str, Any]) -> list[tuple[str, ResolutionConfidence, str]]:
            keys = {str(key) for key in arguments.keys()}
            name_key = tool_alias_key(tool_name)
            candidates: list[tuple[str, ResolutionConfidence, str]] = []
            if {"text", "old", "new"}.issubset(keys) and "text_replace" in self.specs:
                candidates.append(("text_replace", "high", "text replacement argument shape"))
            if {"text", "pattern"}.issubset(keys) and "text_search" in self.specs:
                candidates.append(("text_search", "high", "text search argument shape"))
            if "command" in keys and "shell_run" in self.specs:
                confidence: ResolutionConfidence = "high" if any(token in name_key for token in ("shell", "command", "exec", "bash", "terminal")) else "medium"
                candidates.append(("shell_run", confidence, "shell command argument shape"))
            if "path" in keys:
                if any(token in name_key for token in ("list", "dir", "directory", "ls", "files", "scan")) and "file_list" in self.specs:
                    candidates.append(("file_list", "high", "directory listing argument shape"))
                if any(token in name_key for token in ("read", "open", "view", "cat")) and "file_read" in self.specs:
                    candidates.append(("file_read", "high", "file read argument shape"))
                if "content" in keys and any(token in name_key for token in ("write", "save", "create")) and "file_write" in self.specs:
                    candidates.append(("file_write", "high", "file write argument shape"))
            return candidates

    def _suggestions(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        names = list(self.tool_names)
        close = difflib.get_close_matches(tool_alias_key(tool_name), names, n=5, cutoff=0.55)
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
