"""Precise code intelligence for the agent loop.

Currently provides ``symbols`` (document outline): for Python files an ``ast``
walk yields precise classes / functions / methods with line ranges and the def
header (far better than a line-prefix heuristic); other languages fall back to a
language-agnostic keyword-prefix scan. Read-only; respects path sensitivity.

Definition / reference lookups (cross-file) are intentionally out of scope here
and belong to a later LSP / tree-sitter backend.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List

from leapflow.security.path_sensitivity import classify_path_sensitivity

logger = logging.getLogger(__name__)

_CODE_INTEL_OPS = frozenset({"symbols"})
_MAX_INTEL_CHARS = 400_000
_HEURISTIC_PREFIXES = (
    "class ", "def ", "async def ", "function ", "func ", "fn ", "const ",
    "let ", "var ", "interface ", "type ", "enum ", "struct ", "trait ",
    "impl ", "public ", "private ", "protected ", "export ",
)


def _def_header(lines: List[str], lineno: int) -> str:
    """Return the (possibly multi-line) def/class header, stripped, ending at ':'."""
    start = max(0, lineno - 1)
    buf: List[str] = []
    for i in range(start, min(len(lines), start + 5)):
        stripped = lines[i].strip()
        buf.append(stripped)
        if lines[i].rstrip().endswith(":"):
            break
    return " ".join(buf)[:200]


def _python_symbols(text: str) -> List[Dict[str, Any]]:
    """Precise Python symbols via ast (raises SyntaxError on unparseable input)."""
    tree = ast.parse(text)
    lines = text.splitlines()
    symbols: List[Dict[str, Any]] = []

    def walk(node: ast.AST, parent: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({
                    "kind": "method" if parent else "function",
                    "name": child.name,
                    "line": child.lineno,
                    "end_line": getattr(child, "end_lineno", None),
                    "parent": parent,
                    "signature": _def_header(lines, child.lineno),
                })
                walk(child, child.name)
            elif isinstance(child, ast.ClassDef):
                symbols.append({
                    "kind": "class",
                    "name": child.name,
                    "line": child.lineno,
                    "end_line": getattr(child, "end_lineno", None),
                    "parent": parent,
                    "signature": _def_header(lines, child.lineno),
                })
                walk(child, child.name)

    walk(tree, "")
    return symbols


def _heuristic_symbols(text: str) -> List[Dict[str, Any]]:
    """Language-agnostic keyword-prefix symbol scan (fallback for non-Python)."""
    symbols: List[Dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(_HEURISTIC_PREFIXES) or stripped.startswith("@"):
            symbols.append({
                "kind": "symbol",
                "name": "",
                "line": index,
                "end_line": None,
                "parent": "",
                "signature": stripped[:200],
            })
    return symbols


async def code_intel(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return document symbols for a source file (Python: precise ast; else heuristic).

    Read-only. ``operation`` currently supports only ``symbols``.
    """
    path = str(params.get("path", "") or "")
    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}
    operation = str(params.get("operation", "symbols") or "symbols").strip().lower()
    if operation not in _CODE_INTEL_OPS:
        return {"ok": False, "error": f"Unsupported operation: {operation} (supported: symbols)"}

    target = Path(path).expanduser().resolve()
    sensitivity = classify_path_sensitivity(target)
    if not sensitivity.readable:
        return {"ok": False, "error": f"Read blocked by safety policy: {target.name}"}
    if not target.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    if not target.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    try:
        text = target.read_text(encoding="utf-8", errors="replace")[:_MAX_INTEL_CHARS]
    except Exception as exc:  # noqa: BLE001 - surface as structured error
        return {"ok": False, "error": f"Cannot read file: {exc}"}

    language = (target.suffix.lstrip(".").lower() or "text")
    engine = "heuristic"
    symbols: List[Dict[str, Any]] = []
    if target.suffix == ".py":
        try:
            symbols = _python_symbols(text)
            engine = "ast"
        except SyntaxError:
            # File may be mid-edit / not yet valid — fall back rather than fail.
            symbols = _heuristic_symbols(text)
            engine = "heuristic-fallback"
    else:
        symbols = _heuristic_symbols(text)

    return {
        "ok": True,
        "path": str(target),
        "operation": "symbols",
        "language": language,
        "engine": engine,
        "symbols": symbols,
        "symbol_count": len(symbols),
    }
