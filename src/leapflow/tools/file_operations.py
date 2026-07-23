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

import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from leapflow.security.path_sensitivity import PathSensitivity, classify_path_sensitivity

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 100_000
_FILE_LIST_LIMIT = 100
_FILE_READ_MODES = frozenset({"raw", "outline", "symbols"})
# Directories skipped by code_search / file_find (VCS, deps, build, caches).
_SEARCH_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".ruff_cache", ".pytest_cache", ".mypy_cache", "dist", "build", ".idea",
})
_CODE_SEARCH_MAX_RESULTS = 200
_CODE_SEARCH_LINE_CHARS = 500
_FILE_FIND_MAX_RESULTS = 500
_SEARCH_TIMEOUT_S = 30
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
        "sensitivity_scope": sensitivity.scope,
        "owner_component": sensitivity.owner_component,
        "syncable": sensitivity.syncable,
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
                    "Use ~/.leapflow/config/user.yaml, "
                    "~/.leapflow/profiles/<profile>/config/*.yaml, or "
                    "<workspace>/.leapflow/config.yaml for structured configuration."
                ),
                "error_type": "unsupported_config_probe",
                "retryable": False,
                "config_locations": [
                    "~/.leapflow/config/user.yaml",
                    "~/.leapflow/profiles/<profile>/config/*.yaml",
                    "<workspace>/.leapflow/config.yaml",
                ],
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
            try:
                approved = await gate.check(str(target), mode, _sensitivity_metadata(sensitivity))
            except TypeError:
                approved = await gate.check(str(target), mode)
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
            try:
                approved = await gate.check(str(target), content, mode, _sensitivity_metadata(sensitivity))
            except TypeError:
                approved = await gate.check(str(target), content, mode)
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


# ── ripgrep provisioning: seamless Python fallback + best-effort auto-install ──

# Cached provision outcome so the (potentially slow) install is attempted at most
# once per process and never blocks repeated searches.
_RG_PROVISION: Dict[str, Any] = {"done": False, "available": False}


def ripgrep_path() -> str | None:
    """Return the ripgrep executable path if on PATH, else None (fast lookup)."""
    return shutil.which("rg")


def ripgrep_install_hint() -> str:
    """Platform-appropriate manual install command for ripgrep."""
    if sys.platform == "darwin":
        return "brew install ripgrep"
    if shutil.which("apt-get"):
        return "sudo apt-get install -y ripgrep"
    if shutil.which("dnf"):
        return "sudo dnf install -y ripgrep"
    if shutil.which("pacman"):
        return "sudo pacman -S ripgrep"
    if shutil.which("cargo"):
        return "cargo install ripgrep"
    return "install ripgrep — see https://github.com/BurntSushi/ripgrep#installation"


def ensure_ripgrep_available(*, autoinstall: bool = True, timeout: float = 180.0) -> bool:
    """Best-effort, cached, non-fatal provision of ripgrep. Never raises.

    ripgrep is only an *accelerator*: ``code_search`` always works via the pure
    Python fallback (zero install). This attempts a seamless, no-sudo install
    when ripgrep is missing — currently only via Homebrew on macOS (which needs
    no elevated privileges). Other platforms fall back to the Python search plus
    a manual-install hint. Intended to run once in the background at startup so
    it never blocks a search. Returns whether ripgrep is available afterward.
    """
    if _RG_PROVISION["done"]:
        return _RG_PROVISION["available"]
    if ripgrep_path() is not None:
        _RG_PROVISION.update(done=True, available=True)
        return True
    available = False
    if autoinstall and sys.platform == "darwin" and shutil.which("brew"):
        try:
            logger.info("code_search: ripgrep missing — attempting 'brew install ripgrep' (best-effort)")
            subprocess.run(
                ["brew", "install", "ripgrep"],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            available = ripgrep_path() is not None
        except Exception:  # noqa: BLE001 - best-effort, never fatal
            logger.debug("code_search: ripgrep auto-install failed; using Python fallback", exc_info=True)
    _RG_PROVISION.update(done=True, available=available)
    return available


# ── code_search: regex search across a directory tree (ripgrep-backed) ──

def _iter_search_files(base: Path, glob: str | None) -> Iterable[Path]:
    """Yield candidate files under ``base``, skipping VCS/dep/build/cache dirs."""
    if base.is_file():
        yield base
        return
    for root, dirs, filenames in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SEARCH_SKIP_DIRS]
        for name in filenames:
            if glob and not fnmatch.fnmatch(name, glob):
                continue
            yield Path(root) / name


def _code_search_rg(
    pattern: str, base: Path, glob: str | None, ignore_case: bool, multiline: bool, max_results: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    args = ["rg", "--json"]
    if ignore_case:
        args.append("-i")
    if multiline:
        args += ["-U", "--multiline-dotall"]
    if glob:
        args += ["-g", glob]
    # Skip VCS/dependency/build/cache dirs consistently with the Python fallback
    # (ripgrep honors .gitignore but not these when no ignore file is present).
    for skip in _SEARCH_SKIP_DIRS:
        args += ["-g", f"!{skip}"]
    args += ["--", pattern, str(base)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT_S)
    results: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if len(results) >= max_results:
            return results, True
        try:
            evt = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if evt.get("type") != "match":
            continue
        data = evt.get("data", {})
        subs = data.get("submatches", []) or []
        text = (data.get("lines", {}).get("text", "") or "").rstrip("\n")
        results.append({
            "path": data.get("path", {}).get("text", ""),
            "line": data.get("line_number"),
            "column": (subs[0].get("start", 0) + 1) if subs else None,
            "text": text[:_CODE_SEARCH_LINE_CHARS],
        })
    return results, False


def _code_search_python(
    pattern: str, base: Path, glob: str | None, ignore_case: bool, multiline: bool, max_results: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL
    rx = re.compile(pattern, flags)
    results: List[Dict[str, Any]] = []
    for fp in _iter_search_files(base, glob):
        if len(results) >= max_results:
            return results, True
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = rx.search(line)
            if match:
                results.append({
                    "path": str(fp),
                    "line": lineno,
                    "column": match.start() + 1,
                    "text": line[:_CODE_SEARCH_LINE_CHARS],
                })
                if len(results) >= max_results:
                    return results, True
    return results, False


async def code_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search file *contents* by regex across a directory tree (ripgrep-backed,
    Python fallback). Read-only; VCS/dependency/build/cache dirs are skipped and
    results are redacted."""
    pattern = str(params.get("pattern", "") or "")
    if not pattern:
        return {"ok": False, "error": "Missing required parameter: pattern"}
    base = Path(str(params.get("path", ".") or ".")).expanduser().resolve()
    if not base.exists():
        return {"ok": False, "error": f"Path not found: {base}"}
    glob = params.get("glob") or None
    ignore_case = bool(params.get("ignore_case", False))
    multiline = bool(params.get("multiline", False))
    max_results = _safe_int(
        params.get("max_results", _CODE_SEARCH_MAX_RESULTS),
        _CODE_SEARCH_MAX_RESULTS, minimum=1, maximum=2000,
    )
    rg = ripgrep_path()
    try:
        if rg:
            results, truncated = _code_search_rg(pattern, base, glob, ignore_case, multiline, max_results)
        else:
            results, truncated = _code_search_python(pattern, base, glob, ignore_case, multiline, max_results)
    except re.error as exc:
        return {"ok": False, "error": f"Invalid regex: {exc}", "error_type": "invalid_regex"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Search timed out after {_SEARCH_TIMEOUT_S}s"}
    except Exception as exc:  # noqa: BLE001 - surface as structured error
        return {"ok": False, "error": str(exc)}
    try:
        from leapflow.security.redact import redact_sensitive_text
        for item in results:
            item["text"] = redact_sensitive_text(item.get("text", ""))
    except ImportError:
        pass
    payload: Dict[str, Any] = {
        "ok": True,
        "pattern": pattern,
        "path": str(base),
        "matches": results,
        "match_count": len(results),
        "truncated": truncated,
        "backend": "ripgrep" if rg else "python",
    }
    if not rg:
        # Seamless fallback already served the search; surface the manual path so
        # users can enable the faster ripgrep backend if they want it.
        payload["install_hint"] = (
            f"ripgrep not found — used the Python fallback (slower). For speed: {ripgrep_install_hint()}"
        )
    return payload


# ── file_find: locate files by recursive glob under a base path ──

async def file_find(params: Dict[str, Any]) -> Dict[str, Any]:
    """Find files by a (recursive) glob pattern under a base path. Read-only;
    VCS/dependency/build/cache dirs are skipped."""
    pattern = str(params.get("glob", params.get("pattern", "")) or "")
    if not pattern:
        return {"ok": False, "error": "Missing required parameter: glob"}
    base = Path(str(params.get("path", ".") or ".")).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return {"ok": False, "error": f"Not a directory: {base}"}
    max_results = _safe_int(
        params.get("max_results", _FILE_FIND_MAX_RESULTS),
        _FILE_FIND_MAX_RESULTS, minimum=1, maximum=5000,
    )
    matches: List[str] = []
    truncated = False
    try:
        for item in base.rglob(pattern):
            if any(part in _SEARCH_SKIP_DIRS for part in item.parts):
                continue
            matches.append(str(item))
            if len(matches) >= max_results:
                truncated = True
                break
    except Exception as exc:  # noqa: BLE001 - surface as structured error
        return {"ok": False, "error": str(exc)}
    matches.sort()
    return {
        "ok": True,
        "glob": pattern,
        "path": str(base),
        "files": matches,
        "file_count": len(matches),
        "truncated": truncated,
    }


# ── edit_file: targeted, anchored search-replace edits ──

async def edit_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply targeted, anchored search-replace edits to an existing text file.

    Each edit is ``{original_text, new_text, replace_all?}``. ``original_text``
    must match exactly; a non-unique anchor is rejected (unless ``replace_all``)
    so edits never corrupt the file silently. ``dry_run`` previews without
    writing. Mutating: flows through the same path-sensitivity guard, approval
    gate, and threat scan as ``file_write`` (use ``file_write`` to create files).
    """
    path = str(params.get("path", "") or "")
    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}
    edits = params.get("edits")
    if not edits and (params.get("original_text") is not None or params.get("old_text") is not None):
        edits = [{
            "original_text": params.get("original_text", params.get("old_text", "")),
            "new_text": params.get("new_text", params.get("new", "")),
            "replace_all": bool(params.get("replace_all", False)),
        }]
    if not isinstance(edits, list) or not edits:
        return {"ok": False, "error": "Missing required parameter: edits (list of {original_text, new_text, replace_all?})"}
    dry_run = bool(params.get("dry_run", False))

    target = Path(path).expanduser().resolve()
    sensitivity = classify_path_sensitivity(target)
    if sensitivity.hardline or not sensitivity.writable:
        return {"ok": False, "error": _write_block_message(target, sensitivity)}
    if not sensitivity.readable:
        return {"ok": False, "error": _read_block_message(target, sensitivity)}
    if not target.exists():
        return {"ok": False, "error": f"File not found (use file_write to create): {path}", "error_type": "file_not_found"}
    if not target.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    try:
        original = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Cannot read file: {exc}"}

    content = original
    total_replacements = 0
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return {"ok": False, "error": f"edit[{idx}] must be an object"}
        old = str(edit.get("original_text", edit.get("old_text", "")) or "")
        new = str(edit.get("new_text", edit.get("new", "")) or "")
        replace_all = bool(edit.get("replace_all", False))
        if old == "":
            return {"ok": False, "error": f"edit[{idx}]: original_text must be non-empty", "error_type": "empty_anchor"}
        if old == new:
            return {"ok": False, "error": f"edit[{idx}]: original_text equals new_text (no-op)", "error_type": "noop_edit"}
        occurrences = content.count(old)
        if occurrences == 0:
            return {"ok": False, "error": f"edit[{idx}]: original_text not found in file", "error_type": "anchor_not_found"}
        if occurrences > 1 and not replace_all:
            return {
                "ok": False,
                "error": f"edit[{idx}]: original_text is not unique ({occurrences} matches); add surrounding context or set replace_all=true",
                "error_type": "anchor_not_unique",
                "match_count": occurrences,
            }
        content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        total_replacements += occurrences if replace_all else 1

    if content == original:
        return {"ok": True, "path": str(target), "changed": False, "edits_applied": 0, "note": "no change"}

    if dry_run:
        return {
            "ok": True, "path": str(target), "changed": True, "dry_run": True,
            "edits_applied": len(edits), "replacements": total_replacements,
            "preview_bytes": len(content.encode()),
        }

    try:
        from leapflow.tools.registry_bootstrap import get_file_write_gate
        gate = get_file_write_gate()
        if gate is not None:
            try:
                approved = await gate.check(str(target), content, "overwrite", _sensitivity_metadata(sensitivity))
            except TypeError:
                approved = await gate.check(str(target), content, "overwrite")
            if not approved:
                message = str(getattr(gate, "denial_message", "") or f"File edit denied by approval gate: {target.name}")
                return {"ok": False, "error": message}
        elif sensitivity.requires_approval:
            return {"ok": False, "error": f"Sensitive file edit requires approval: {target.name}", "requires_approval": True, **_sensitivity_metadata(sensitivity)}
    except ImportError:
        if sensitivity.requires_approval:
            return {"ok": False, "error": f"Sensitive file edit requires approval: {target.name}", "requires_approval": True, **_sensitivity_metadata(sensitivity)}

    try:
        from leapflow.security.threat_patterns import scan_for_threats, ThreatScope
        threats = scan_for_threats(content, scope=ThreatScope.ALL, max_results=3)
        high_threats = [t for t in threats if t.severity >= 0.8]
        if high_threats:
            logger.warning("edit_file: high-severity threats in %s: %s", target.name, [t.pattern_name for t in high_threats])
    except ImportError:
        pass

    try:
        target.write_text(content)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Write failed: {exc}"}
    return {
        "ok": True,
        "path": str(target),
        "changed": True,
        "edits_applied": len(edits),
        "replacements": total_replacements,
        "bytes_written": len(content.encode()),
        **_sensitivity_metadata(sensitivity),
    }
