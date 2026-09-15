# Copyright (c) Alibaba, Inc. and its affiliates.
"""Static checks for a task's action.py (structure + import safety).

Dual use: CLI (``python -m leapspace.app_space.action_lint <config_path>``) and harness
import (``lint_task(config) -> list[str]`` — the caller owns config loading).
All checks are AST-only —
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

from leapspace.app_space.apps import APP_MODULES
from leapspace.app_space.config import AppTaskConfig

# Top-level packages importable inside the sandbox app process, beyond the
# stdlib. Host-only libraries (cua_sandbox, mcp, ...) must be imported
# lazily inside the functions that use them: the app loads action.py whole
# as hooks.py at registration time, so module-level imports execute there.
SAFE_ROOTS = frozenset({"PyQt6", "leapspace"})


def lint_task(config: AppTaskConfig) -> list[str]:
    """Check a task's action file (config.action_path) against its config.

    Returns the list of problems found (empty = clean); a broken action
    file is reported as problems rather than raised.
    """
    action_path = config.action_path
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
        *_check_app_ids(config),
        *_check_imports(tree),
        *_check_hooks(config, functions),
        *_check_reference(functions),
        *_check_expect(functions),
        *_check_main(tree),
    ]


def _check_app_ids(config: AppTaskConfig) -> list[str]:
    """Every configured app must be a registered app module."""
    return [
        f"app_id {app_id!r}: not registered in APP_MODULES"
        for app_id in config.app_ids
        if app_id not in APP_MODULES
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


def _is_main_guard(node: ast.If) -> bool:
    """True for `if __name__ == "__main__":` regardless of operand order."""
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    sides = [test.left, *test.comparators]
    has_name = any(
        isinstance(side, ast.Name) and side.id == "__name__" for side in sides
    )
    has_main = any(
        isinstance(side, ast.Constant) and side.value == "__main__" for side in sides
    )
    return has_name and has_main


def _check_main(tree: ast.Module) -> list[str]:
    """The in-box verdict entry is `python hooks.py`; __main__ must launch expect."""
    guard = next(
        (node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node)),
        None,
    )
    if guard is None:
        return [
            '__main__: missing — the in-box verdict entry is `python hooks.py`, '
            'whose guard must launch expect()'
        ]
    launches_expect = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "expect"
        for stmt in guard.body
        for node in ast.walk(stmt)
    )
    if not launches_expect:
        return [
            "__main__: never launches expect() — the in-box verdict entry "
            "`python hooks.py` would do nothing"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="action_lint",
        description="Statically check a task's action.py against its config.yaml.",
    )
    parser.add_argument(
        "config_path", type=Path, help="path to the task's config.yaml"
    )
    args = parser.parse_args(argv)
    try:
        config = AppTaskConfig.load(args.config_path)
    except Exception as exc:  # a load failure is a lint problem, not a crash
        print(f"{args.config_path}: {exc}", file=sys.stderr)
        return 1
    problems = lint_task(config)
    for problem in problems:
        print(f"{args.config_path}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
