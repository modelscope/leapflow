"""Static checks for a task's action.py (structure + import safety).

Dual use: CLI (``python -m leapspace.app_space.action_lint <task_dir>``) and harness
import (``lint_task(task_dir) -> list[str]``). All checks are AST-only —
action.py is never imported: importing has side effects and needs the
runtime environment, while linting must work offline.

A lint problem is a task-authoring bug: the CLI prints each problem and
exits non-zero; linting never raises for a broken task file.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from leapspace.app_space.config import AppTaskConfig

# Top-level packages importable inside the sandbox app process, beyond the
# stdlib. Host-only libraries (cua_sandbox, mcp, ...) must be imported
# lazily inside the functions that use them: the app loads action.py whole
# as hooks.py at registration time, so module-level imports execute there.
SAFE_ROOTS = frozenset({"PyQt6", "leapspace"})


def lint_task(task_dir: str | Path) -> list[str]:
    """Check a task directory's action.py against its config.yaml.

    Returns the list of problems found (empty = clean); a broken config or
    action file is reported as problems rather than raised.
    """
    task_dir = Path(task_dir)
    try:
        config = AppTaskConfig.load(task_dir)
    except Exception as exc:  # any load failure is a lint problem, not a crash
        return [f"config.yaml: {exc}"]

    action_path = task_dir / "action.py"
    if not action_path.exists():
        return [f"{action_path}: not found"]
    try:
        tree = ast.parse(action_path.read_text())
    except SyntaxError as exc:
        return [f"{action_path}:{exc.lineno}: syntax error: {exc.msg}"]

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return [
        *_check_imports(tree),
        *_check_hooks(config, functions),
        *_check_reference(functions),
        *_check_expect(functions),
    ]


def _positional_count(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return len(fn.args.posonlyargs + fn.args.args)


def _check_imports(tree: ast.Module) -> list[str]:
    """Module-level imports must stay inside the in-sandbox safe set."""
    problems = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(
                    f"line {node.lineno}: relative import breaks when action.py "
                    "is loaded as hooks.py; import leapspace.* absolutely"
                )
                continue
            roots = [node.module.split(".")[0]]
        else:
            continue
        for root in roots:
            if root not in sys.stdlib_module_names and root not in SAFE_ROOTS:
                problems.append(
                    f"line {node.lineno}: module-level import {root!r} is not "
                    "in-sandbox safe; move it into the function that uses it"
                )
    return problems


def _check_hooks(
    config: AppTaskConfig, functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
) -> list[str]:
    """Every config-wired hook must exist with the fn(app) signature."""
    problems = []
    for app, wiring in config.hooks.items():
        for point, fn_name in wiring.items():
            label = f"hook {fn_name!r} ({app}.{point})"
            fn = functions.get(fn_name)
            if fn is None:
                problems.append(f"{label}: not defined in action.py")
            elif isinstance(fn, ast.AsyncFunctionDef):
                problems.append(f"{label}: must be sync — the app never awaits hooks")
            elif _positional_count(fn) != 1:
                problems.append(f"{label}: must take exactly one parameter (app)")
    return problems


def _check_reference(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[str]:
    fn = functions.get("reference")
    if fn is None:
        return ["reference: missing (async def reference(actor))"]
    if not isinstance(fn, ast.AsyncFunctionDef):
        return ["reference: must be async def — it awaits actor actions"]
    if _positional_count(fn) != 1:
        return ["reference: must take exactly one parameter (actor)"]
    return []


def _check_expect(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[str]:
    fn = functions.get("expect")
    if fn is None:
        return ["expect: missing (def expect(state_root='/tmp/leapspace'))"]
    if isinstance(fn, ast.AsyncFunctionDef):
        return ["expect: must be sync — the harness runs it as a plain script"]
    required = _positional_count(fn) - len(fn.args.defaults)
    required += sum(default is None for default in fn.args.kw_defaults)
    if required:
        return ["expect: all parameters must have defaults (the harness calls expect())"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="action_lint",
        description="Statically check a task's action.py against its config.yaml.",
    )
    parser.add_argument(
        "task_dir", type=Path, help="task directory holding config.yaml + action.py"
    )
    args = parser.parse_args(argv)
    problems = lint_task(args.task_dir)
    for problem in problems:
        print(f"{args.task_dir}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
