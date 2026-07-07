"""File system operations — list, read, write.

All handlers follow the ToolBridge convention: receive params dict, return result dict.
Safety layers:
1. Sensitive path block: credential files, private keys, auth tokens
2. System path block: OS system directories for writes
3. Binary extension block: prevent reading non-text files
4. Output redaction: secrets stripped from file content before returning
5. Character limit: prevent context overflow from large reads
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet

logger = logging.getLogger(__name__)

_BLOCKED_WRITE_PREFIXES = (
    "/System", "/usr", "/bin", "/sbin", "/etc",
    "/var/root", "/Library/System",
)

# Sensitive paths that should NEVER be read and returned to an LLM
_SENSITIVE_READ_PATTERNS: FrozenSet[str] = frozenset({
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/id_ecdsa", ".ssh/id_dsa",
    ".ssh/authorized_keys", ".ssh/known_hosts",
    ".gnupg/", ".gpg",
    ".aws/credentials", ".aws/config",
    ".env", ".env.local", ".env.production",
    "credentials.json", "service_account.json",
    ".netrc", ".npmrc", ".pypirc",
    "token.json", "secrets.yaml", "secrets.yml",
    ".kube/config",
    "id_rsa", "id_ed25519",
})

_SENSITIVE_EXACT_NAMES: FrozenSet[str] = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging",
    "credentials.json", "service_account.json", "token.json",
    "secrets.yaml", "secrets.yml", ".netrc", ".npmrc", ".pypirc",
})

# Binary extensions that should not be read as text
_BINARY_EXTENSIONS: FrozenSet[str] = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".class", ".o", ".obj",
    ".wasm", ".sqlite", ".db", ".duckdb",
})

_MAX_READ_CHARS = 100_000


def _is_write_blocked(path: Path) -> bool:
    resolved = str(path)
    return any(resolved.startswith(prefix) for prefix in _BLOCKED_WRITE_PREFIXES)


def _is_sensitive_read(path: Path) -> bool:
    """Check if reading the file would expose credentials or secrets."""
    resolved = str(path)
    name = path.name
    if name in _SENSITIVE_EXACT_NAMES:
        return True
    return any(pattern in resolved for pattern in _SENSITIVE_READ_PATTERNS)


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _is_device_path(path: Path) -> bool:
    resolved = str(path)
    return resolved.startswith("/dev/") or resolved.startswith("/proc/")


async def file_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """List directory contents with optional glob pattern."""
    path = params.get("path", ".")
    pattern = params.get("pattern", "*")

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {path}"}
    if not target.is_dir():
        return {"ok": False, "error": f"Not a directory: {path}"}

    entries = []
    for item in sorted(target.glob(pattern)):
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
        })

    return {"ok": True, "path": str(target), "entries": entries[:100]}


async def file_read(params: Dict[str, Any]) -> Dict[str, Any]:
    """Read text file content with line limit and security guards."""
    path = params.get("path", "")
    max_lines = int(params.get("max_lines", 200))

    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}

    target = Path(path).expanduser().resolve()

    if _is_device_path(target):
        return {"ok": False, "error": f"Device path reads are blocked: {path}"}

    if not target.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    if not target.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    if _is_sensitive_read(target):
        return {"ok": False, "error": f"Sensitive file blocked by security policy: {target.name}"}

    if _is_binary(target):
        return {"ok": False, "error": f"Binary file cannot be read as text: {target.name}"}

    try:
        raw = target.read_text(errors="replace")
        if len(raw) > _MAX_READ_CHARS:
            raw = raw[:_MAX_READ_CHARS]
            logger.debug("file_read: truncated to %d chars", _MAX_READ_CHARS)

        lines = raw.splitlines()
        truncated = len(lines) > max_lines
        content = "\n".join(lines[:max_lines])

        try:
            from leapflow.security.redact import redact_sensitive_text
            content = redact_sensitive_text(content)
        except ImportError:
            pass

        try:
            from leapflow.security.threat_patterns import scan_for_threats, ThreatScope
            threats = scan_for_threats(content, scope=ThreatScope.CONTEXT, max_results=3)
            if threats:
                threat_names = [t.pattern_name for t in threats]
                logger.warning("file_read: threat patterns in %s: %s", target.name, threat_names)
        except ImportError:
            pass

        return {
            "ok": True,
            "path": str(target),
            "content": content,
            "lines": len(lines),
            "truncated": truncated,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def file_write(params: Dict[str, Any]) -> Dict[str, Any]:
    """Write content to a file. Supports overwrite and append modes."""
    path = params.get("path", "")
    content = params.get("content", "")
    mode = params.get("mode", "overwrite")

    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}

    target = Path(path).expanduser().resolve()

    if _is_write_blocked(target):
        return {"ok": False, "error": f"Write blocked by safety policy: {target}"}

    if _is_sensitive_read(target):
        return {"ok": False, "error": f"Sensitive file write blocked by security policy: {target.name}"}

    try:
        from leapflow.tools.registry_bootstrap import get_file_write_gate
        gate = get_file_write_gate()
        if gate is not None:
            approved = await gate.check(str(target), content)
            if not approved:
                return {"ok": False, "error": f"File write denied by approval gate: {target.name}"}
    except ImportError:
        pass

    try:
        from leapflow.security.threat_patterns import scan_for_threats, ThreatScope
        threats = scan_for_threats(content, scope=ThreatScope.ALL, max_results=3)
        high_threats = [t for t in threats if t.severity >= 0.8]
        if high_threats:
            logger.warning("file_write: high-severity threats in content for %s: %s",
                           target.name, [t.pattern_name for t in high_threats])
    except ImportError:
        pass

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with target.open("a") as f:
                f.write(content)
        else:
            target.write_text(content)
        return {"ok": True, "path": str(target), "bytes_written": len(content.encode())}
    except Exception as e:
        return {"ok": False, "error": str(e)}
