# Copyright (c) Alibaba, Inc. and its affiliates.
"""Change-scoped test selection.

Two separate jobs, with different economics:

**Mock layer** (``select_from_paths``) — available but *not wired into CI*. The
always-on tiers (real journeys, regression ledger, architecture contracts) can
never be selected away and already account for ~14s of an ~18s full run, so
selecting the mock layer can save at most a few seconds however precise it gets.
Kept for local use (``make test-impact``), where a single-module change narrows
to 2-3 test files, and for when the suite outgrows its feedback budget.

**Live journeys** (``select_journeys``) — wired in, because here each journey costs
real tokens and real minutes against a real provider, so the arithmetic comes out
the other way. Journeys declare their own ``SUBJECT_PATHS`` and ``LIVE_SIGNAL``, so
the selection lives next to the assertions it describes. The *offline* journey
lanes never select: replay is cheap and a suite that can be skipped will be
skipped.

Selection sources for the mock layer, in order of precedence:

1. **Escalation rules** (``tests/.impact/escalate.yaml``) — a change to shared
   foundations means everything runs. No cleverness is worth a missed regression
   in ``config.py``.
2. **Coverage map** (``tests/.impact/coverage_map.json``) — generated from a full
   run with ``--cov``. This is the only source that sees *runtime* coupling
   through EventBus and Protocol indirection, which a static import graph misses
   entirely.
3. **Static import closure** — the fallback for files the coverage map does not
   know yet (new modules, or a stale map). Over-selects, never under-selects.

The always-on tiers (architecture contracts, the regression ledger, the real
journeys) are never selected away; they are appended unconditionally.

Usage::

    python tools/impact.py --base origin/main            # print the selection
    python tools/impact.py --base origin/main --run      # run it
    python tools/impact.py --base origin/main --live-journeys
    python tools/impact.py --build-map                   # refresh the coverage map
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"
IMPACT_DIR = TESTS_ROOT / ".impact"
COVERAGE_MAP = IMPACT_DIR / "coverage_map.json"
ESCALATE_FILE = IMPACT_DIR / "escalate.yaml"

# Directories whose tests never take part in selection.
ALWAYS_ON_PATHS = ("tests/regression", "tests/journeys")
ALWAYS_ON_FILES = ("tests/test_architecture_contracts.py",)


class FullRun(Exception):
    """Raised when the change requires the whole mock layer."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ── Git ──────────────────────────────────────────────────────────────────


def changed_files(base: str) -> list[str]:
    """Return repo-relative paths changed since ``base``, including uncommitted work."""
    merge_base = base
    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", base],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() or base
    except subprocess.CalledProcessError:
        # No shared history (shallow clone, unfetched ref): fall back to the ref.
        pass

    paths: set[str] = set()
    for args in (
        ["git", "diff", "--name-only", merge_base, "--"],
        ["git", "diff", "--name-only", "HEAD", "--"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            raise FullRun(f"cannot determine changes ({' '.join(args)} failed); running all")
        paths.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return sorted(paths)


# ── Escalation ───────────────────────────────────────────────────────────


def escalation_patterns() -> list[str]:
    """Load glob patterns that force a full run.

    Parsed with a deliberately tiny reader rather than PyYAML: this runs before
    dependencies are guaranteed to be installed, and the file is a flat list.
    """
    if not ESCALATE_FILE.is_file():
        return []
    patterns: list[str] = []
    for line in ESCALATE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("- "):
            patterns.append(line[2:].strip().strip("\"'"))
    return patterns


def check_escalation(paths: list[str], patterns: list[str] | None = None) -> None:
    """Raise :class:`FullRun` when any changed path matches an escalation rule."""
    rules = escalation_patterns() if patterns is None else patterns
    for path in paths:
        candidate = Path(path)
        for pattern in rules:
            if candidate.match(pattern):
                raise FullRun(f"{path} matches escalation rule {pattern!r}")


# ── Static import graph ──────────────────────────────────────────────────


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build_import_graph() -> dict[str, set[str]]:
    """Return module -> set of leapflow modules that import it."""
    reverse: dict[str, set[str]] = {}
    for path in SRC_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        importer = _module_name(path)
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                targets.append(node.module)
            elif isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            for target in targets:
                if target.startswith("leapflow"):
                    reverse.setdefault(target, set()).add(importer)
    return reverse


def dependents_closure(modules: set[str], reverse: dict[str, set[str]]) -> set[str]:
    """Return ``modules`` plus everything that transitively imports them."""
    seen = set(modules)
    queue = list(modules)
    while queue:
        current = queue.pop()
        for importer in reverse.get(current, ()):
            if importer not in seen:
                seen.add(importer)
                queue.append(importer)
    return seen


def test_modules_importing(modules: set[str]) -> set[str]:
    """Return test files that import any module in ``modules``."""
    selected: set[str] = set()
    for path in TESTS_ROOT.rglob("test_*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            if any(
                name == module or name.startswith(f"{module}.")
                for name in names
                for module in modules
            ):
                selected.add(path.relative_to(REPO_ROOT).as_posix())
                break
    return selected


# ── Coverage map ─────────────────────────────────────────────────────────


def load_coverage_map() -> dict[str, list[str]]:
    """Return the committed test -> source-files map, or {} when absent."""
    if not COVERAGE_MAP.is_file():
        return {}
    try:
        payload = json.loads(COVERAGE_MAP.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload.get("tests") or {})


def tests_touching(paths: set[str], coverage: dict[str, list[str]]) -> tuple[set[str], set[str]]:
    """Return (selected tests, source paths the map did not know about)."""
    selected: set[str] = set()
    known: set[str] = set()
    for test_file, sources in coverage.items():
        source_set = set(sources)
        known |= source_set
        if source_set & paths:
            selected.add(test_file)
    return selected, paths - known


# ── Selection ────────────────────────────────────────────────────────────


def select_from_paths(
    paths: list[str],
    *,
    coverage: dict[str, list[str]] | None = None,
    patterns: list[str] | None = None,
) -> tuple[list[str], str]:
    """Decide the selection for an explicit set of changed paths.

    Separated from git so the decision itself is testable: this function is what
    determines whether a regression gets a chance to fail, and a defect in it
    would silently shrink the suite.
    """
    try:
        check_escalation(paths, patterns)
    except FullRun as escalate:
        return [], f"full mock layer: {escalate.reason}"

    if not paths:
        return [], "full mock layer: no changes detected relative to the base"

    source_paths = {p for p in paths if p.startswith("src/") and p.endswith(".py")}
    touched_tests = {
        p for p in paths if p.startswith("tests/") and Path(p).name.startswith("test_")
    }

    if not source_paths and not touched_tests:
        return [], "full mock layer: change touches neither src/ nor tests/"

    resolved_coverage = load_coverage_map() if coverage is None else coverage
    from_coverage, unknown = tests_touching(source_paths, resolved_coverage)

    from_static: set[str] = set()
    if unknown:
        modules = set()
        for path in unknown:
            full = REPO_ROOT / path
            if full.is_file():
                modules.add(_module_name(full))
        if modules:
            impacted = dependents_closure(modules, build_import_graph())
            from_static = test_modules_importing(impacted)

    selected = sorted(from_coverage | from_static | touched_tests)
    reason = (
        f"{len(selected)} test file(s) selected from {len(paths)} changed path(s): "
        f"{len(from_coverage)} via coverage map, {len(from_static)} via import graph "
        f"({len(unknown)} source file(s) not in the map), "
        f"{len(touched_tests)} directly edited"
    )
    if not selected:
        return [], f"full mock layer: nothing matched ({reason})"
    return selected, reason


def select(base: str) -> tuple[list[str], str]:
    """Return (pytest targets, human-readable explanation) for changes since ``base``."""
    try:
        paths = changed_files(base)
    except FullRun as escalate:
        return [], f"full mock layer: {escalate.reason}"
    return select_from_paths(paths)


def always_on_targets() -> list[str]:
    """Return the tiers that are never selected away."""
    targets = [path for path in ALWAYS_ON_PATHS if (REPO_ROOT / path).is_dir()]
    targets += [path for path in ALWAYS_ON_FILES if (REPO_ROOT / path).is_file()]
    return targets


# ── Journey selection (the live lane only) ────────────────────────────


@dataclass(frozen=True)
class JourneyMeta:
    """Declared metadata for one real end-to-end journey."""

    path: str
    subject_paths: tuple[str, ...]
    live_signal: bool


def journey_metadata() -> list[JourneyMeta]:
    """Read ``SUBJECT_PATHS`` / ``LIVE_SIGNAL`` from each journey module.

    Parsed rather than imported: importing a test module pulls in pytest fixtures
    and the harness, and this runs before a test session exists. Keeping the
    declaration inside the journey — instead of in a table here — means it moves
    with the assertions it describes and cannot silently go stale.
    """
    directory = TESTS_ROOT / "journeys"
    found: list[JourneyMeta] = []
    for path in sorted(directory.glob("test_*.py")):
        subjects: tuple[str, ...] = ()
        live = True
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                if target.id == "SUBJECT_PATHS":
                    subjects = tuple(str(item) for item in ast.literal_eval(node.value))
                elif target.id == "LIVE_SIGNAL":
                    live = bool(ast.literal_eval(node.value))
            except ValueError:
                continue
        found.append(
            JourneyMeta(
                path=path.relative_to(REPO_ROOT).as_posix(),
                subject_paths=subjects,
                live_signal=live,
            )
        )
    return found


def select_journeys(
    paths: list[str],
    *,
    live_only: bool = False,
    patterns: list[str] | None = None,
) -> tuple[list[str], str]:
    """Return the journeys a change could plausibly break.

    Used for the *live* lane, where each journey costs real tokens and real
    minutes. The offline lanes never call this — they run every journey, because
    replay is cheap and a suite that can be skipped will be skipped.

    A journey with no declared subjects is always selected: that is the safe
    direction for a missing declaration.
    """
    journeys = journey_metadata()
    if live_only:
        journeys = [item for item in journeys if item.live_signal]
    if not journeys:
        return [], "no journeys are eligible"

    try:
        check_escalation(paths, patterns)
    except FullRun as escalate:
        return (
            [item.path for item in journeys],
            f"all eligible journeys: {escalate.reason}",
        )

    source_paths = [p for p in paths if p.startswith("src/")]
    touched_journeys = {p for p in paths if p.startswith("tests/journeys/")}
    if not source_paths and not touched_journeys:
        return (
            [item.path for item in journeys],
            "all eligible journeys: change touches neither src/ nor tests/journeys/",
        )

    selected: list[str] = []
    for item in journeys:
        if item.path in touched_journeys or not item.subject_paths:
            selected.append(item.path)
            continue
        if any(p.startswith(prefix) for p in source_paths for prefix in item.subject_paths):
            selected.append(item.path)

    if not selected:
        return [], (
            f"no eligible journey declares a subject touched by these "
            f"{len(source_paths)} source path(s)"
        )
    return selected, (
        f"{len(selected)}/{len(journeys)} eligible journey(s) selected from "
        f"{len(source_paths)} changed source path(s)"
    )


def build_map() -> int:
    """Run the mock layer with coverage and write the test -> sources map."""
    IMPACT_DIR.mkdir(parents=True, exist_ok=True)
    print("running the mock layer with per-test coverage; this takes a while...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-m",
            "not e2e",
            "-p",
            "no:cacheprovider",
            "--cov=leapflow",
            "--cov-context=test",
            "--cov-report=",
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("the suite failed; not writing a coverage map from a red run")
        return result.returncode

    import sqlite3

    data_file = REPO_ROOT / ".coverage"
    if not data_file.is_file():
        print("coverage produced no data file")
        return 1

    mapping: dict[str, set[str]] = {}
    with sqlite3.connect(data_file) as connection:
        rows = connection.execute(
            """
            SELECT context.context, file.path
            FROM line_bits
            JOIN context ON context.id = line_bits.context_id
            JOIN file ON file.id = line_bits.file_id
            """
        ).fetchall()
    for context, file_path in rows:
        if not context:
            continue
        test_file = context.split("::", 1)[0]
        if not test_file.endswith(".py"):
            continue
        try:
            relative = Path(file_path).resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        mapping.setdefault(test_file, set()).add(relative)

    payload = {
        "_comment": (
            "Generated by tools/impact.py --build-map. Maps each test file to the "
            "source files it actually executed, so change-scoped selection sees "
            "runtime coupling that a static import graph cannot."
        ),
        "tests": {key: sorted(value) for key, value in sorted(mapping.items())},
    }
    COVERAGE_MAP.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {COVERAGE_MAP.relative_to(REPO_ROOT)} for {len(mapping)} test files")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Print or run the change-scoped selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git ref to compare against")
    parser.add_argument("--run", action="store_true", help="Run the selection with pytest")
    parser.add_argument(
        "--build-map", action="store_true", help="Refresh the coverage-derived impact map"
    )
    parser.add_argument(
        "--live-journeys",
        action="store_true",
        help="Print the live-capable journeys a change could break, one per line",
    )
    parser.add_argument("--jobs", default="auto", help="pytest-xdist parallelism")
    args = parser.parse_args(argv)

    if args.build_map:
        return build_map()

    if args.live_journeys:
        # Emitted on stdout as a bare list so a workflow can pass it straight to
        # pytest; the explanation goes to stderr so it stays out of that list.
        try:
            paths = changed_files(args.base)
        except FullRun as escalate:
            paths = []
            print(f"impact: {escalate.reason}", file=sys.stderr)
        selected, reason = select_journeys(paths, live_only=True)
        print(f"impact: {reason}", file=sys.stderr)
        for path in selected:
            print(path)
        return 0

    selected, reason = select(args.base)
    always_on = always_on_targets()
    print(f"impact: {reason}")

    # The always-on tiers are appended, never filtered: whichever mock tests were
    # selected, the real journeys and the incident ledger still run. No marker
    # expression is applied, because any filter here could exclude them.
    targets = (selected + always_on) if selected else ["tests/"]

    command = [sys.executable, "-m", "pytest", *targets, "-q", "-n", args.jobs]
    print("impact: " + " ".join(command[2:]))

    if not args.run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
