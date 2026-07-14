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
from typing import Any, Dict, Iterable

from leapflow.security.path_sensitivity import PathSensitivity, classify_path_sensitivity

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 100_000
_FILE_LIST_LIMIT = 100
_FILE_READ_MODES = frozenset({"raw", "outline", "symbols"})
_SYMBOL_PREFIXES = (
    "class ", "def ", "async def ", "function ", "const ", "let ", "var ",
    "interface ", "type ", "enum ", "struct ", "trait ", "impl ",
)


def _safe_int(value: Any, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _line_window(lines: list[str], *, start_line: int, max_lines: int) -> tuple[list[str], int, int]:
    start = max(0, start_line - 1)
    end = min(len(lines), start + max_lines)
    return lines[start:end], start + 1, end


def _outline_lines(lines: Iterable[str], *, limit: int) -> list[tuple[int, str]]:
    outline: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(('#', '##', '###', '- ', '* ')):
            outline.append((index, f"{index}: {stripped}"))
        elif stripped.endswith((':', '{')) and len(stripped) < 140:
            outline.append((index, f"{index}: {stripped}"))
        if len(outline) >= limit:
            break
    return outline


def _symbol_lines(lines: Iterable[str], *, limit: int) -> list[tuple[int, str]]:
    symbols: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_SYMBOL_PREFIXES) or stripped.startswith("@dataclass"):
            symbols.append((index, f"{index}: {stripped}"))
        if len(symbols) >= limit:
            break
    return symbols


def _read_text_window(path: Path, *, max_chars: int) -> tuple[str, bool]:
    """Read at most max_chars characters plus one sentinel without loading huge files."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read(max_chars + 1)
    if len(raw) <= max_chars:
        return raw, False
    logger.debug("file_read: truncated to %d chars", max_chars)
    return raw[:max_chars], True


def _read_block_message(path: Path, sensitivity: PathSensitivity) -> str:
    if sensitivity.category == "binary_file":
        return f"Binary file cannot be read as text: {path.name}"
    if sensitivity.category == "runtime_database":
        return f"Runtime database files cannot be read as text: {path.name}"
    if sensitivity.category == "runtime_control":
        return f"Runtime control files cannot be read through file_read: {path.name}"
    if sensitivity.category == "device_path":
        return f"Device path reads are blocked: {path}"
    return f"File read blocked by safety policy: {path}"


def _write_block_message(path: Path, sensitivity: PathSensitivity) -> str:
    if sensitivity.category == "system_path":
        return f"Write blocked by safety policy: {path}"
    if sensitivity.category == "runtime_database":
        return f"Runtime database files cannot be written through file_write: {path.name}"
    if sensitivity.category == "runtime_control":
        return f"Runtime control files cannot be written through file_write: {path.name}"
    if sensitivity.category == "audit_log":
        return f"Audit logs cannot be modified through file_write: {path.name}"
    return f"File write blocked by safety policy: {path}"


def _sensitivity_metadata(sensitivity: PathSensitivity) -> dict[str, Any]:
    return {
        "sensitivity_category": sensitivity.category,
        "sensitivity_level": sensitivity.level,
        "sensitivity_reason": sensitivity.reason,
        "redact_on_read": sensitivity.redact_on_read,
    }


def _is_unsupported_leapflow_config_probe(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 2 and parts[-2:] == (".leapflow", "config.json")


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

    visible = entries[:_FILE_LIST_LIMIT]
    return {
        "ok": True,
        "path": str(target),
        "entries": visible,
        "entry_count": len(entries),
        "truncated": len(entries) > len(visible),
    }


async def file_read(params: Dict[str, Any]) -> Dict[str, Any]:
    """Read text file content with context-aware modes and security guards."""
    path = params.get("path", "")
    max_lines = _safe_int(params.get("max_lines", 200), 200, minimum=1, maximum=2000)
    start_line = _safe_int(params.get("start_line", 1), 1, minimum=1)
    max_chars = _safe_int(params.get("max_chars", _MAX_READ_CHARS), _MAX_READ_CHARS, minimum=200, maximum=_MAX_READ_CHARS)
    mode = str(params.get("mode", "raw") or "raw").strip().lower()
    if mode not in _FILE_READ_MODES:
        mode = "raw"

    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}

    target = Path(path).expanduser().resolve()
    sensitivity = classify_path_sensitivity(target)

    if not sensitivity.readable:
        return {"ok": False, "error": _read_block_message(target, sensitivity)}

    if not target.exists():
        if _is_unsupported_leapflow_config_probe(target):
            return {
                "ok": False,
                "error": (
                    "LeapFlow does not use <workspace>/.leapflow/config.json. "
                    "Use ~/.leapflow/.env for global runtime config, or an existing ./.env "
                    "for project overrides."
                ),
                "error_type": "unsupported_config_probe",
                "retryable": False,
                "config_locations": ["~/.leapflow/.env", "./.env"],
            }
        return {"ok": False, "error": f"File not found: {path}"}
    if not target.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    if sensitivity.requires_approval:
        try:
            from leapflow.tools.registry_bootstrap import get_file_read_gate
            gate = get_file_read_gate()
            if gate is None:
                return {
                    "ok": False,
                    "error": f"Sensitive file read requires approval: {target.name}",
                    "requires_approval": True,
                    **_sensitivity_metadata(sensitivity),
                }
            approved = await gate.check(str(target), mode, _sensitivity_metadata(sensitivity))
            if not approved:
                message = str(getattr(gate, "denial_message", "") or f"File read denied by approval gate: {target.name}")
                return {"ok": False, "error": message}
        except ImportError:
            return {
                "ok": False,
                "error": f"Sensitive file read requires approval: {target.name}",
                "requires_approval": True,
                **_sensitivity_metadata(sensitivity),
            }

    try:
        raw, raw_truncated = _read_text_window(target, max_chars=max_chars)

        lines = raw.splitlines()
        selected_lines, selected_start, selected_end = _line_window(
            lines,
            start_line=start_line,
            max_lines=max_lines,
        )
        line_truncated = selected_end < len(lines)

        if mode == "outline":
            outline = _outline_lines(lines, limit=max_lines)
            content = "\n".join(text for _, text in outline)
            selected_start = outline[0][0] if outline else 1
            selected_end = outline[-1][0] if outline else 0
            line_truncated = raw_truncated or len(outline) >= max_lines
        elif mode == "symbols":
            symbols = _symbol_lines(lines, limit=max_lines)
            content = "\n".join(text for _, text in symbols)
            selected_start = symbols[0][0] if symbols else 1
            selected_end = symbols[-1][0] if symbols else 0
            line_truncated = raw_truncated or len(symbols) >= max_lines
        else:
            content = "\n".join(selected_lines)

        try:
            from leapflow.security.redact import redact_sensitive_text
            content = redact_sensitive_text(content, file_read=sensitivity.redact_on_read)
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
            "start_line": selected_start,
            "end_line": selected_end,
            "selected_lines": len(content.splitlines()) if content else 0,
            "mode": mode,
            "truncated": raw_truncated or line_truncated,
            **_sensitivity_metadata(sensitivity),
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
    sensitivity = classify_path_sensitivity(target)

    if sensitivity.hardline or not sensitivity.writable:
        return {"ok": False, "error": _write_block_message(target, sensitivity)}

    try:
        from leapflow.tools.registry_bootstrap import get_file_write_gate
        gate = get_file_write_gate()
        if gate is not None:
            approved = await gate.check(str(target), content, mode, _sensitivity_metadata(sensitivity))
            if not approved:
                message = str(getattr(gate, "denial_message", "") or f"File write denied by approval gate: {target.name}")
                return {"ok": False, "error": message}
        elif sensitivity.requires_approval:
            return {
                "ok": False,
                "error": f"Sensitive file write requires approval: {target.name}",
                "requires_approval": True,
                **_sensitivity_metadata(sensitivity),
            }
    except ImportError:
        if sensitivity.requires_approval:
            return {
                "ok": False,
                "error": f"Sensitive file write requires approval: {target.name}",
                "requires_approval": True,
                **_sensitivity_metadata(sensitivity),
            }

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
        return {
            "ok": True,
            "path": str(target),
            "bytes_written": len(content.encode()),
            **_sensitivity_metadata(sensitivity),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
