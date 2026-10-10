# Copyright (c) Alibaba, Inc. and its affiliates.
"""Architectural invariants of the benchmarks package.

- ``src/leapflow`` never imports ``benchmarks``: adapters depend on
  LeapFlow's public API, not vice versa.
- Adapter modules never use ``shell=True`` or trigger implicit downloads
  (``pip install``, ``git clone``, ``curl``, ``wget``, dataset
  autodownload hooks).  Allowed subprocess use goes through the
  ``run_subprocess`` helper, which forces ``shell=False``.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
LEAPFLOW_ROOT = SRC_ROOT / "leapflow"
BENCHMARKS_ROOT = SRC_ROOT / "benchmarks"

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shell=True", re.compile(r"\bshell\s*=\s*True\b")),
    ("pip install", re.compile(r"\bpip\s+install\b")),
    ("git clone", re.compile(r"\bgit\s+clone\b")),
    ("curl invocation", re.compile(r"(?<![\w./-])curl(?![\w./-])")),
    ("wget invocation", re.compile(r"(?<![\w./-])wget(?![\w./-])")),
    ("urllib.request.urlretrieve", re.compile(r"\burlretrieve\b")),
    ("urllib.request.urlopen", re.compile(r"\burlopen\b")),
    ("requests download", re.compile(r"\brequests\.(?:get|post)\b")),
    ("auto-download hook", re.compile(r"\bauto[_-]?download\b", re.IGNORECASE)),
    ("huggingface snapshot_download", re.compile(r"\bsnapshot_download\b")),
    ("torch hub.load", re.compile(r"\btorch\.hub\.load\b")),
)


def _python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_leapflow_does_not_import_benchmarks() -> None:
    offenders: list[str] = []
    for path in _python_files(LEAPFLOW_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "benchmarks" or alias.name.startswith("benchmarks."):
                        offenders.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and (
                    module == "benchmarks" or module.startswith("benchmarks.")
                ):
                    offenders.append(f"{path}: from {module} import ...")
    assert offenders == [], (
        "src/leapflow must not depend on the benchmarks package:\n"
        + "\n".join(offenders)
    )


def _strip_strings_and_comments(text: str) -> dict[int, str]:
    """Return per-line code with STRING / COMMENT tokens blanked out.

    Docstrings and comments must not count as forbidden usage — they
    frequently *document* the ban rather than violate it.
    """
    lines = text.splitlines()
    kept = {i + 1: line for i, line in enumerate(lines)}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenizeError:
        return kept
    for tok in tokens:
        if tok.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        (start_row, start_col), (end_row, end_col) = tok.start, tok.end
        for row in range(start_row, end_row + 1):
            if row not in kept:
                continue
            original = kept[row]
            begin = start_col if row == start_row else 0
            finish = end_col if row == end_row else len(original)
            begin = min(begin, len(original))
            finish = min(finish, len(original))
            kept[row] = original[:begin] + (" " * (finish - begin)) + original[finish:]
    return kept


def test_benchmark_sources_do_not_use_forbidden_shell_or_downloads() -> None:
    offenders: list[str] = []
    for path in _python_files(BENCHMARKS_ROOT):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(BENCHMARKS_ROOT.parent)
        for line_no, raw in _strip_strings_and_comments(text).items():
            for label, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(raw):
                    offenders.append(
                        f"{rel}:{line_no} forbids `{label}`: {raw.strip()}"
                    )
    assert offenders == [], (
        "benchmark sources must not shell out or auto-download:\n"
        + "\n".join(offenders)
    )


def test_adapter_subprocess_calls_route_through_run_subprocess() -> None:
    adapters_dir = BENCHMARKS_ROOT / "adapters"
    offenders: list[str] = []
    for path in _python_files(adapters_dir):
        if path.name == "base.py":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bsubprocess\.(run|Popen|call|check_output|check_call)\b", text):
            offenders.append(str(path.relative_to(BENCHMARKS_ROOT.parent)))
    assert offenders == [], (
        "adapters must call run_subprocess instead of invoking subprocess "
        "directly:\n" + "\n".join(offenders)
    )


def test_native_adapters_do_not_depend_on_private_leapflow_internals() -> None:
    native_dir = BENCHMARKS_ROOT / "native"
    offenders: list[str] = []
    private_patterns = (
        re.compile(r"from\s+leapflow\.[\w.]*\._"),
        re.compile(r"import\s+leapflow\.[\w.]*\._"),
    )
    for path in _python_files(native_dir):
        text = path.read_text(encoding="utf-8")
        for pattern in private_patterns:
            if pattern.search(text):
                offenders.append(str(path.relative_to(BENCHMARKS_ROOT.parent)))
                break
    assert offenders == [], (
        "native adapters must use public leapflow APIs only:\n"
        + "\n".join(offenders)
    )
