"""Canonical filesystem layout for LeapFlow runtime data.

This module is the single source of truth for paths under the LeapFlow data
root. Runtime modules should depend on these immutable layout objects instead
of constructing profile-relative paths locally.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


_PROFILE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_CONFIG_NAMES = (
    "runtime.yaml",
    "llm.yaml",
    "perception.yaml",
    "gateway.yaml",
    "hub.yaml",
    "privacy.yaml",
    "approval.yaml",
    "cache.yaml",
)


def validate_profile_name(profile: str) -> str:
    """Return a filesystem-safe profile name or raise ValueError."""
    normalized = (profile or "default").strip()
    if not normalized or any(char not in _PROFILE_NAME_CHARS for char in normalized):
        raise ValueError(
            "Invalid LEAPFLOW_PROFILE; use only letters, numbers, underscores, or dashes"
        )
    return normalized


def workspace_id_for_path(path: Path) -> str:
    """Return a stable workspace id from a canonical path."""
    canonical = str(path.expanduser().resolve())
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"ws-{digest}"


@dataclass(frozen=True)
class SecretsLayout:
    """Secret vault paths for either global or profile scope."""

    root: Path

    @property
    def vault_path(self) -> Path:
        return self.root / "vault.json"

    @property
    def key_path(self) -> Path:
        return self.root / "vault.key"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CacheLayout:
    """Profile cache layout with profile/workspace/session scopes."""

    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / "cache.yaml"

    @property
    def index_path(self) -> Path:
        return self.root / "index.duckdb"

    @property
    def profile_dir(self) -> Path:
        return self.root / "profile"

    @property
    def workspaces_dir(self) -> Path:
        return self.root / "workspaces"

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.workspaces_dir / workspace_id

    def workspace_shared_dir(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "shared"

    def session_dir(self, workspace_id: str, session_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "sessions" / session_id

    def category_dir(
        self,
        *,
        scope: str,
        category: str,
        workspace_id: str = "",
        session_id: str = "",
    ) -> Path:
        if scope == "profile":
            return self.profile_dir / category
        if scope == "workspace":
            if not workspace_id:
                raise ValueError("workspace_id is required for workspace cache scope")
            return self.workspace_shared_dir(workspace_id) / category
        if scope == "session":
            if not workspace_id or not session_id:
                raise ValueError("workspace_id and session_id are required for session cache scope")
            return self.session_dir(workspace_id, session_id) / category
        raise ValueError(f"Unsupported cache scope: {scope}")

    def ensure(self) -> None:
        for path in (self.root, self.profile_dir, self.workspaces_dir):
            path.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            _write_yaml_if_missing(self.config_path, _default_cache_config())


@dataclass(frozen=True)
class GatewayLayout:
    """Gateway configuration, manifest, and state paths."""

    root: Path
    config_path: Path

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def platform_state_path(self) -> Path:
        return self.root / "platform_state.yaml"

    def ensure(self) -> None:
        for path in (self.root, self.manifests_dir, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)
        _write_yaml_if_missing(self.config_path, _default_gateway_config())


@dataclass(frozen=True)
class ApprovalLayout:
    """Approval grant and audit paths."""

    root: Path

    @property
    def grants_path(self) -> Path:
        return self.root / "grants.json"

    @property
    def audit_path(self) -> Path:
        return self.root / "audit.jsonl"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProfileManifest:
    """First-class profile metadata."""

    version: int
    profile_id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    owner: dict[str, str] = field(default_factory=dict)
    workspace_policy: dict[str, Any] = field(default_factory=dict)
    cache_policy_ref: str = "config/cache.yaml"
    secrets_scope: str = "profile"
    runtime: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls, profile_id: str) -> "ProfileManifest":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            version=1,
            profile_id=profile_id,
            name=profile_id.title() if profile_id != "default" else "Default",
            description="Local LeapFlow profile",
            created_at=now,
            updated_at=now,
            owner={"user_id": "local", "tenant_id": "personal"},
            workspace_policy={"id_strategy": "canonical_path_hash", "default_workspace_root": None},
            runtime={"daemon_enabled": True, "profile_isolation": "strict"},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, fallback_id: str) -> "ProfileManifest":
        return cls(
            version=int(data.get("version") or 1),
            profile_id=str(data.get("id") or fallback_id),
            name=str(data.get("name") or fallback_id),
            description=str(data.get("description") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            owner=dict(data.get("owner") or {}),
            workspace_policy=dict(data.get("workspace_policy") or {}),
            cache_policy_ref=str(data.get("cache_policy_ref") or "config/cache.yaml"),
            secrets_scope=str(data.get("secrets_scope") or "profile"),
            runtime=dict(data.get("runtime") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner": dict(self.owner),
            "workspace_policy": dict(self.workspace_policy),
            "cache_policy_ref": self.cache_policy_ref,
            "secrets_scope": self.secrets_scope,
            "runtime": dict(self.runtime),
        }


@dataclass(frozen=True)
class ProfileLayout:
    """Canonical layout for one LeapFlow profile."""

    root: Path
    profile_id: str

    @property
    def manifest_path(self) -> Path:
        return self.root / "profile.yaml"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def runtime_config_path(self) -> Path:
        return self.config_dir / "runtime.yaml"

    @property
    def llm_config_path(self) -> Path:
        return self.config_dir / "llm.yaml"

    @property
    def perception_config_path(self) -> Path:
        return self.config_dir / "perception.yaml"

    @property
    def gateway_config_path(self) -> Path:
        return self.config_dir / "gateway.yaml"

    @property
    def hub_config_path(self) -> Path:
        return self.config_dir / "hub.yaml"

    @property
    def privacy_config_path(self) -> Path:
        return self.config_dir / "privacy.yaml"

    @property
    def approval_config_path(self) -> Path:
        return self.config_dir / "approval.yaml"

    @property
    def cache_config_path(self) -> Path:
        return self.config_dir / "cache.yaml"

    @property
    def secrets(self) -> SecretsLayout:
        return SecretsLayout(self.root / "secrets")

    @property
    def db_dir(self) -> Path:
        return self.root / "db"

    @property
    def duckdb_path(self) -> Path:
        return self.db_dir / "leap.duckdb"

    @property
    def skill_library_path(self) -> Path:
        return self.db_dir / "skill_library.duckdb"

    @property
    def conversation_db_path(self) -> Path:
        return self.db_dir / "conversation.duckdb"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def global_memory_dir(self) -> Path:
        return self.memory_dir / "global"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @property
    def gateway(self) -> GatewayLayout:
        return GatewayLayout(self.root / "gateway", self.gateway_config_path)

    @property
    def approval(self) -> ApprovalLayout:
        return ApprovalLayout(self.root / "approval")

    @property
    def audit_dir(self) -> Path:
        return self.root / "audit"

    @property
    def audit_log_path(self) -> Path:
        return self.audit_dir / "runtime.jsonl"

    @property
    def cache(self) -> CacheLayout:
        return CacheLayout(self.root / "cache")

    @property
    def runtime_dir(self) -> Path:
        return self.root / "runtime"

    @property
    def observation_pid_path(self) -> Path:
        return self.runtime_dir / "observation_daemon.pid"

    @property
    def observation_log_path(self) -> Path:
        return self.audit_dir / "observation_daemon.log"

    def config_paths(self) -> tuple[Path, ...]:
        return tuple(self.config_dir / name for name in _CONFIG_NAMES)

    def ensure(self) -> None:
        for path in (
            self.root,
            self.config_dir,
            self.db_dir,
            self.memory_dir,
            self.global_memory_dir,
            self.skills_dir,
            self.audit_dir,
            self.runtime_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.secrets.ensure()
        self.gateway.ensure()
        self.approval.ensure()
        self.cache.ensure()
        _write_yaml_if_missing(self.manifest_path, ProfileManifest.default(self.profile_id).to_dict())
        for path in self.config_paths():
            _write_yaml_if_missing(path, _default_profile_config(path.name))

    def load_manifest(self) -> ProfileManifest:
        if not self.manifest_path.exists():
            return ProfileManifest.default(self.profile_id)
        try:
            data = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError, ValueError):
            return ProfileManifest.default(self.profile_id)
        if not isinstance(data, dict):
            return ProfileManifest.default(self.profile_id)
        return ProfileManifest.from_dict(data, fallback_id=self.profile_id)


@dataclass(frozen=True)
class PathLayout:
    """Canonical layout for a LeapFlow data root."""

    root: Path

    @property
    def global_config_dir(self) -> Path:
        return self.root / "config"

    @property
    def user_config_path(self) -> Path:
        return self.global_config_dir / "user.yaml"

    @property
    def policy_config_path(self) -> Path:
        return self.global_config_dir / "policy.yaml"

    @property
    def mcp_servers_path(self) -> Path:
        return self.global_config_dir / "mcp_servers.json"

    @property
    def defaults_lock_path(self) -> Path:
        return self.global_config_dir / "defaults.lock.yaml"

    @property
    def global_secrets(self) -> SecretsLayout:
        return SecretsLayout(self.root / "secrets")

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    def profile(self, profile_id: str) -> ProfileLayout:
        safe_profile = validate_profile_name(profile_id)
        return ProfileLayout(self.profiles_dir / safe_profile, safe_profile)

    def ensure(self, *, profile_id: str = "default") -> ProfileLayout:
        for path in (self.root, self.global_config_dir, self.logs_dir, self.profiles_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.global_secrets.ensure()
        _write_yaml_if_missing(self.user_config_path, _default_user_config(profile_id))
        _write_yaml_if_missing(self.policy_config_path, _default_policy_config())
        _write_yaml_if_missing(self.defaults_lock_path, _default_defaults_lock())
        profile_layout = self.profile(profile_id)
        profile_layout.ensure()
        return profile_layout

    def watched_config_paths(self, profile_id: str, workspace_root: Path | None = None) -> tuple[Path, ...]:
        profile_layout = self.profile(profile_id)
        paths: list[Path] = [self.user_config_path, self.policy_config_path, *profile_layout.config_paths()]
        if workspace_root is not None:
            paths.append(workspace_root / ".leapflow" / "config.yaml")
        return tuple(paths)


def build_layout(data_dir: str | Path) -> PathLayout:
    """Create a PathLayout from a root path."""
    return PathLayout(Path(data_dir).expanduser())


def _write_yaml_if_missing(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _default_user_config(profile_id: str) -> dict[str, Any]:
    return {
        "version": 1,
        "user": {"default_profile": profile_id},
        "logging": {"level": "INFO"},
    }


def _default_policy_config() -> dict[str, Any]:
    return {"version": 1, "paths": {"policy": "layout-derived"}}


def _default_defaults_lock() -> dict[str, Any]:
    return {"version": 1, "managed_by": "leapflow"}


def _default_gateway_config() -> dict[str, Any]:
    return {"version": 1, "platforms": {}, "auto_connect": []}


def _default_cache_config() -> dict[str, Any]:
    return {
        "version": 1,
        "default_ttl_s": 604800,
        "profile_quota_mb": 1024,
        "workspace_quota_mb": 2048,
        "session_quota_mb": 512,
        "sensitive_session_ttl_s": 86400,
    }


def _default_profile_config(name: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "runtime.yaml": {"runtime": {}},
        "llm.yaml": {"llm": {"primary": {}}},
        "perception.yaml": {"perception": {}},
        "gateway.yaml": _default_gateway_config(),
        "hub.yaml": {"hub": {}},
        "privacy.yaml": {"privacy": {}},
        "approval.yaml": {"approval": {}},
        "cache.yaml": _default_cache_config(),
    }
    return {"version": 1, **defaults.get(name, {})}


def existing_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Return paths that currently exist, preserving order."""
    return tuple(path for path in paths if path.exists())
