"""Path sensitivity classification for local file access governance.

The classifier is intentionally policy-oriented and tool-agnostic: it maps a
path to a stable sensitivity category that file tools and risk assessment can
use before performing reads or writes. It does not execute file operations and
it does not know about UI or approval rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from leapflow.layout import ManagedPathDescriptor, build_layout


@dataclass(frozen=True)
class PathSensitivity:
    """Structured sensitivity decision for a local path."""

    category: str = "normal"
    level: str = "low"
    hardline: bool = False
    readable: bool = True
    writable: bool = True
    requires_approval: bool = False
    redact_on_read: bool = False
    reason: str = "ordinary_file"
    scope: str = ""
    owner_component: str = ""
    syncable: bool = True

    @property
    def is_sensitive(self) -> bool:
        return self.category != "normal" or self.requires_approval or self.hardline


_SYSTEM_WRITE_PREFIXES = (
    "/System", "/usr", "/bin", "/sbin", "/etc",
    "/var/root", "/Library/System",
)

_CREDENTIAL_NAMES = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging",
    "credentials.json", "service_account.json", "token.json",
    "secrets.yaml", "secrets.yml", ".netrc", ".npmrc", ".pypirc",
})

_CREDENTIAL_PATTERNS = (
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/id_ecdsa", ".ssh/id_dsa",
    ".ssh/authorized_keys", ".ssh/known_hosts",
    ".aws/credentials", ".aws/config", ".kube/config",
    ".gnupg/", ".gpg",
)

_STORAGE_SUFFIXES = frozenset({".duckdb", ".sqlite", ".db"})
_RUNTIME_CONTROL_NAMES = frozenset({
    "leapd.sock", "leapd.pid", "leapd.lock",
})
_BINARY_EXTENSIONS = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".class", ".o", ".obj", ".wasm",
})
_LEAPFLOW_DATA_ROOTS: tuple[Path, ...] = (Path("~/.leapflow").expanduser(),)


def configure_path_sensitivity_roots(roots: Iterable[Path]) -> None:
    """Set canonical LeapFlow data roots used by path sensitivity classification."""
    global _LEAPFLOW_DATA_ROOTS
    normalized: list[Path] = []
    for root in roots:
        try:
            normalized.append(root.expanduser().resolve())
        except OSError:
            normalized.append(root.expanduser())
    if normalized:
        _LEAPFLOW_DATA_ROOTS = tuple(normalized)


def configured_path_sensitivity_roots() -> tuple[Path, ...]:
    """Return canonical LeapFlow data roots used by path sensitivity classification."""
    return _LEAPFLOW_DATA_ROOTS


def _is_under(path: str, root: Path) -> bool:
    root_text = str(root.expanduser()).replace("\\", "/").rstrip("/")
    return path == root_text or path.startswith(root_text + "/")


def _from_descriptor(descriptor: ManagedPathDescriptor) -> PathSensitivity:
    category = descriptor.category
    name = descriptor.path.name.lower()
    hardline = category == "runtime_database" or (category == "secret_vault" and name.endswith(".key"))
    writable = category not in {"secret_vault", "runtime_database", "runtime_state", "approval_state"}
    readable = not hardline
    requires_approval = category not in {"cache_profile"}
    redact_on_read = category in {
        "secret_vault", "approval_state", "memory_store", "runtime_state", "config",
        "mcp_config", "workspace_manifest", "history", "cache_sensitive", "leapflow_profile_data",
    }
    return PathSensitivity(
        category=category,
        level=descriptor.sensitivity,
        hardline=hardline,
        readable=readable,
        writable=writable,
        requires_approval=requires_approval,
        redact_on_read=redact_on_read,
        reason=f"{category}_file",
        scope=descriptor.scope,
        owner_component=descriptor.owner_component,
        syncable=descriptor.syncable,
    )


def classify_path_sensitivity(path: Path) -> PathSensitivity:
    """Classify a path before file read/write execution."""
    expanded = path.expanduser()
    normalized = str(expanded).replace("\\", "/")
    lowered = normalized.lower()
    name = expanded.name.lower()

    if lowered.startswith("/dev/") or lowered.startswith("/proc/"):
        return PathSensitivity(
            category="device_path",
            level="critical",
            hardline=True,
            readable=False,
            writable=False,
            reason="device_path_access",
        )

    if any(normalized.startswith(prefix) for prefix in _SYSTEM_WRITE_PREFIXES):
        return PathSensitivity(
            category="system_path",
            level="critical",
            hardline=True,
            readable=True,
            writable=False,
            reason="system_path_write",
        )

    if name in _RUNTIME_CONTROL_NAMES or name.endswith(".sock") or name.endswith(".lock"):
        return PathSensitivity(
            category="runtime_control",
            level="critical",
            hardline=True,
            readable=False,
            writable=False,
            reason="runtime_control_file",
        )

    if name.endswith(".duckdb.wal") or expanded.suffix.lower() in _STORAGE_SUFFIXES:
        return PathSensitivity(
            category="runtime_database",
            level="critical",
            hardline=True,
            readable=False,
            writable=False,
            reason="runtime_database_file",
        )

    if name in _CREDENTIAL_NAMES or any(pattern in normalized for pattern in _CREDENTIAL_PATTERNS):
        return PathSensitivity(
            category="credential",
            level="high",
            hardline=False,
            readable=True,
            writable=True,
            requires_approval=True,
            redact_on_read=True,
            reason="credential_or_secret_file",
        )

    if "/.leapflow/" in lowered and name in {"config.yaml", "workspace.yaml"}:
        category = "workspace_manifest" if name == "workspace.yaml" else "config"
        return PathSensitivity(
            category=category,
            level="high",
            readable=True,
            writable=True,
            requires_approval=True,
            redact_on_read=True,
            reason=f"{category}_file",
            scope="workspace",
            owner_component="config",
            syncable=True,
        )

    leapflow_root = next(
        (root for root in _LEAPFLOW_DATA_ROOTS if _is_under(normalized, root)),
        None,
    )
    if leapflow_root is not None:
        return _from_descriptor(build_layout(leapflow_root).describe_path(expanded))

    if expanded.suffix.lower() in _BINARY_EXTENSIONS:
        return PathSensitivity(
            category="binary_file",
            level="medium",
            readable=False,
            writable=True,
            reason="binary_file_content",
        )

    return PathSensitivity()
