"""Path sensitivity classification for local file access governance.

The classifier is intentionally policy-oriented and tool-agnostic: it maps a
path to a stable sensitivity category that file tools and risk assessment can
use before performing reads or writes. It does not execute file operations and
it does not know about UI or approval rendering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
    "gateway.yaml", ".credential_key",
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

    data_dir = os.getenv("LEAPFLOW_DATA_DIR", "~/.leapflow")
    leapflow_dir = str(Path(data_dir).expanduser()).replace("\\", "/")
    if normalized.startswith(leapflow_dir):
        if "/approval/" in lowered or name == "audit.jsonl":
            return PathSensitivity(
                category="audit_log",
                level="high",
                readable=True,
                writable=False,
                requires_approval=True,
                redact_on_read=True,
                reason="approval_or_runtime_audit_log",
            )
        if "/memory/" in lowered:
            return PathSensitivity(
                category="memory_store",
                level="high",
                readable=True,
                writable=True,
                requires_approval=True,
                redact_on_read=True,
                reason="user_memory_store",
            )
        if "/run/" in lowered:
            return PathSensitivity(
                category="runtime_state",
                level="medium",
                readable=True,
                writable=False,
                requires_approval=True,
                redact_on_read=True,
                reason="runtime_state_file",
            )
        return PathSensitivity(
            category="leapflow_profile_data",
            level="medium",
            readable=True,
            writable=True,
            requires_approval=True,
            redact_on_read=True,
            reason="leapflow_profile_data",
        )

    if expanded.suffix.lower() in _BINARY_EXTENSIONS:
        return PathSensitivity(
            category="binary_file",
            level="medium",
            readable=False,
            writable=True,
            reason="binary_file_content",
        )

    return PathSensitivity()
