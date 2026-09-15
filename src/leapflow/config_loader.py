# Copyright (c) Alibaba, Inc. and its affiliates.
"""Structured YAML configuration loader for LeapFlow.

The loader treats YAML files as the long-lived configuration source and process
environment variables as ephemeral overrides. It returns environment-style
values as an integration boundary while the existing Settings dataclass is being
split into typed nested settings.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from leapflow.layout import PathLayout, ProfileLayout
from leapflow.security.secrets import FernetSecretVault

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigSource:
    """One configuration document in the load chain."""

    path: Path
    scope: str
    required: bool = False


@dataclass(frozen=True)
class ConfigBundle:
    """Loaded configuration values and source metadata."""

    values: dict[str, Any]
    sources: tuple[ConfigSource, ...]
    watched_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()

    @property
    def env(self) -> dict[str, str]:
        """Return config flattened to LEAPFLOW_* keys for current Settings builder."""
        env_vars: dict[str, str] = {}
        _flatten_yaml(self.values, prefix="LEAPFLOW", env_vars=env_vars)
        return env_vars


class ConfigLoader:
    """Load user, profile, workspace, and environment configuration."""

    def __init__(self, layout: PathLayout, profile_layout: ProfileLayout, workspace_root: Path) -> None:
        self._layout = layout
        self._profile_layout = profile_layout
        self._workspace_root = workspace_root

    def sources(self) -> tuple[ConfigSource, ...]:
        workspace_config = self._layout.workspace_config_path(self._workspace_root)
        return (
            ConfigSource(self._layout.user_config_path, "user"),
            ConfigSource(self._layout.policy_config_path, "policy"),
            ConfigSource(self._profile_layout.runtime_config_path, "profile.runtime"),
            ConfigSource(self._profile_layout.llm_config_path, "profile.llm"),
            ConfigSource(self._profile_layout.perception_config_path, "profile.perception"),
            ConfigSource(self._profile_layout.gateway_config_path, "profile.gateway"),
            ConfigSource(self._profile_layout.hub_config_path, "profile.hub"),
            ConfigSource(self._profile_layout.privacy_config_path, "profile.privacy"),
            ConfigSource(self._profile_layout.approval_config_path, "profile.approval"),
            ConfigSource(self._profile_layout.cache_config_path, "profile.cache"),
            ConfigSource(workspace_config, "workspace"),
        )

    def load(self) -> ConfigBundle:
        values: dict[str, Any] = {}
        warnings: list[str] = []
        sources = self.sources()
        for source in sources:
            loaded = _read_yaml(source.path, warnings=warnings)
            if loaded:
                values = _deep_merge(values, loaded)
        values = _resolve_secret_refs(values, self._layout, self._profile_layout, warnings=warnings)
        values = _deep_merge(values, _env_overrides())
        watched = [source.path for source in sources]
        watched.append(self._layout.mcp_servers_path)
        return ConfigBundle(
            values=values,
            sources=sources,
            watched_paths=tuple(watched),
            warnings=tuple(warnings),
        )


def load_config_bundle(
    layout: PathLayout,
    profile_layout: ProfileLayout,
    workspace_root: Path,
) -> ConfigBundle:
    """Load a ConfigBundle for the active runtime."""
    return ConfigLoader(layout, profile_layout, workspace_root).load()


def config_signature(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    """Return a stable signature for hot-reloadable config files."""
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            signature.append((str(path), 0, 0))
        else:
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _read_yaml(path: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(parsed, dict):
            raise ValueError("YAML root must be a mapping")
        _validate_known_sections(path, parsed, warnings)
        return parsed
    except (OSError, yaml.YAMLError, ValueError) as exc:
        backup = path.with_suffix(path.suffix + ".corrupt.bak")
        logger.warning("config parse error for %s (%s), backing up to %s", path, exc, backup.name)
        if warnings is not None:
            warnings.append(f"{path}: {exc}")
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
        return {}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _flatten_yaml(node: Any, *, prefix: str, env_vars: dict[str, str]) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            child_prefix = f"{prefix}_{key}".upper()
            _flatten_yaml(value, prefix=child_prefix, env_vars=env_vars)
    elif isinstance(node, (list, tuple)):
        env_vars[prefix] = ",".join(str(item) for item in node)
    elif isinstance(node, bool):
        env_vars[prefix] = "true" if node else "false"
    elif node is not None:
        env_vars[prefix] = str(node)


def _resolve_secret_refs(
    values: Mapping[str, Any],
    layout: PathLayout,
    profile_layout: ProfileLayout,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve secret:// refs into in-memory config values without writing YAML."""
    profile_vault = FernetSecretVault(
        profile_layout.secrets.vault_path,
        profile_layout.secrets.key_path,
    )
    global_vault = FernetSecretVault(
        layout.global_secrets.vault_path,
        layout.global_secrets.key_path,
    )

    def resolve_ref(ref: str) -> str | None:
        try:
            if ref.startswith("secret://global/"):
                resolved = global_vault.get(ref)
                if resolved is None and warnings is not None:
                    warnings.append(f"Missing secret ref: {ref}")
                return resolved
            if ref.startswith("secret://profile/"):
                resolved = profile_vault.get(ref)
                if resolved is None and warnings is not None:
                    warnings.append(f"Missing secret ref: {ref}")
                return resolved
            logger.warning("Unsupported secret scope in config ref: %s", ref)
        except Exception as exc:
            logger.warning("Failed to resolve config secret ref %s: %s", ref, type(exc).__name__)
            if warnings is not None:
                warnings.append(f"Failed to resolve secret ref {ref}: {type(exc).__name__}")
        return None

    def visit(node: Any) -> Any:
        if isinstance(node, Mapping):
            resolved: dict[str, Any] = {}
            for key, value in node.items():
                if key.endswith("_ref") and isinstance(value, str) and value.startswith("secret://"):
                    resolved[key] = value
                    target_key = key[:-4]
                    secret_value = resolve_ref(value)
                    if secret_value is not None:
                        resolved[target_key] = secret_value
                else:
                    resolved[key] = visit(value)
            return resolved
        if isinstance(node, list):
            return [visit(item) for item in node]
        if isinstance(node, str) and node.startswith("secret://"):
            return resolve_ref(node) or node
        return node

    result = visit(values)
    return result if isinstance(result, dict) else {}


def _validate_known_sections(
    path: Path,
    parsed: Mapping[str, Any],
    warnings: list[str] | None,
) -> None:
    if warnings is None:
        return
    known_sections = {
        "user", "paths", "version",
        "runtime", "mock", "log",
        "llm", "vlm",
        "memory", "pattern", "snapshot", "codegen", "dag", "intent",
        "learn", "confirm", "skill",
        "perception", "visual", "text", "clipboard", "perceptual", "attention",
        "recording", "video", "learnability",
        "causal", "heuristic", "prediction", "curiosity", "replay", "semantic",
        "budget", "ast", "mhms", "surprise", "signal",
        "react", "tool", "native", "stream", "verbose", "compress", "max",
        "context", "error", "session", "stale", "default", "circuit",
        "rpc", "use", "cua", "copilot", "notification", "live",
        "gateway", "hub", "privacy", "approval", "cache", "logging",
        "observer", "scheduler", "web",
    }
    for key, value in parsed.items():
        if key in known_sections and key != "version" and not isinstance(value, Mapping):
            warnings.append(f"{path}: section '{key}' must be a mapping")


# Known compound keys whose internal underscores are NOT level separators.
# Used by _env_overrides to avoid splitting e.g. LEAPFLOW_LLM_API_KEY into
# {"llm": {"api": {"key": ...}}} instead of the intended {"llm": {"api_key": ...}}.
_COMPOUND_KEYS: frozenset[str] = frozenset({
    "api_key", "base_url", "max_retries", "context_length",
    "max_tokens", "timeout_s", "idle_ttl_s", "max_live_sessions",
    "track_enabled", "hard_limit_ratio", "warning_ratio",
    "result_budget", "tool_budget",
})


def _env_overrides() -> dict[str, Any]:
    """Collect process LEAPFLOW_* overrides as nested config values.

    The strategy is greedy: after the LEAPFLOW_ prefix, the first segment is the
    section. The remaining segments are re-joined and checked against known
    compound keys from longest to shortest, so that LEAPFLOW_LLM_API_KEY maps to
    {"llm": {"api_key": "..."}} rather than {"llm": {"api": {"key": "..."}}}.
    """
    values: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("LEAPFLOW_"):
            continue
        remainder = key[len("LEAPFLOW_"):].lower()
        path = _split_env_key(remainder)
        cursor = values
        for part in path[:-1]:
            next_cursor = cursor.setdefault(part, {})
            if not isinstance(next_cursor, dict):
                next_cursor = {}
                cursor[part] = next_cursor
            cursor = next_cursor
        cursor[path[-1]] = value
    return values


def _split_env_key(remainder: str) -> list[str]:
    """Split a lowercased env suffix into config path segments.

    Uses greedy matching against _COMPOUND_KEYS so that multi-word leaf keys
    (api_key, base_url, ...) are not split into extra nesting levels.
    """
    parts = remainder.split("_")
    if len(parts) <= 1:
        return parts
    # Try greedy: from the end, find the longest tail that matches a compound key.
    for tail_start in range(1, len(parts)):
        candidate = "_".join(parts[tail_start:])
        if candidate in _COMPOUND_KEYS:
            return parts[:tail_start] + [candidate]
    # No compound match — fall back to original per-segment splitting.
    return parts
