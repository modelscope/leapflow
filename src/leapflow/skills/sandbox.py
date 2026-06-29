"""Sandboxed execution namespace for distilled skill code.

Restricts the runtime environment of exec()'d skill code to prevent
escape from the port-based execution model. Skills can only interact
with the system through their bound ExecutionPort/PerceptionPort.

Two-layer safety model:
  Layer 1 (static): ASTPreCheck — compile-time dangerous-pattern detection
  Layer 2 (runtime): SandboxedNamespace — restricted builtins + import whitelist
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import json
import logging
import re
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

_ALLOWED_IMPORTS = frozenset({
    "asyncio", "json", "re", "typing", "collections",
    "dataclasses", "enum", "functools", "itertools",
    "math", "datetime", "copy", "textwrap", "string",
})

_real_import = builtins.__import__


def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Only allow importing from the approved safe module set."""
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed in sandboxed skills")
    return _real_import(name, *args, **kwargs)


_SAFE_BUILTINS: Dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
    # Import (restricted)
    "__import__": _restricted_import,
    # Types
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "type": type,
    "object": object,
    # Functional
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "all": all,
    "any": any,
    # Type checks
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "hasattr": hasattr,
    # Conversion
    "repr": repr,
    "format": format,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "id": id,
    "hash": hash,
    # String
    "print": lambda *a, **kw: None,
    # Errors (allow raising)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "NotImplementedError": NotImplementedError,
    "ImportError": ImportError,
}


class SecurityError(RuntimeError):
    """Raised when skill code fails static security analysis."""


class ASTPreCheck(ast.NodeVisitor):
    """Static analysis layer for skill code security.

    Detects dangerous patterns that could escape the runtime sandbox:
    - Dunder attribute access (__subclasses__, __globals__, etc.)
    - Meta-programming calls (exec, eval, compile)
    - Star imports (from x import *)

    Runs before compile() as a fast-fail gate; the runtime sandbox
    (SandboxedNamespace) remains the authoritative enforcement layer.
    """

    _DANGEROUS_ATTRS: frozenset = frozenset({
        "__subclasses__", "__bases__", "__mro__", "__class__",
        "__globals__", "__code__", "__closure__", "__func__",
        "__self__", "__dict__", "__init_subclass__",
        "__import__", "__loader__", "__spec__",
    })

    _DANGEROUS_CALLS: frozenset = frozenset({
        "exec", "eval", "compile", "breakpoint",
        "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr",
    })

    def __init__(self) -> None:
        self._issues: List[str] = []

    def check(self, source: str) -> List[str]:
        """Parse and scan source code. Returns list of issue descriptions."""
        self._issues = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return [f"Syntax error at line {e.lineno}: {e.msg}"]
        self.visit(tree)
        return self._issues

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self._DANGEROUS_ATTRS:
            self._issues.append(
                f"line {node.lineno}: dangerous attribute '{node.attr}'"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self._DANGEROUS_CALLS:
            self._issues.append(
                f"line {node.lineno}: forbidden call '{node.func.id}()'"
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.names and any(alias.name == "*" for alias in node.names):
            self._issues.append(
                f"line {node.lineno}: wildcard import 'from {node.module} import *'"
            )
        self.generic_visit(node)


class SandboxedNamespace:
    """Factory for restricted execution namespaces.

    Prevents access to: filesystem (open), module loading (__import__),
    attribute manipulation (getattr/setattr/delattr), code generation
    (eval/exec/compile), and process control (os/subprocess).
    """

    @staticmethod
    def create() -> Dict[str, Any]:
        """Return a namespace dict with restricted __builtins__ and safe stdlib."""
        ns: Dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
        ns["asyncio"] = asyncio
        ns["json"] = json
        ns["re"] = re
        return ns

    @staticmethod
    def compile_skill(
        code: str,
        func_name: str,
        *,
        ast_precheck: bool = True,
    ) -> Callable:
        """Compile code and extract the named async function in a sandboxed namespace.

        Args:
            code: Python source for the skill.
            func_name: Name of the async function to extract.
            ast_precheck: Run static analysis before compilation (default True).

        Raises:
            SecurityError: If AST pre-check detects dangerous patterns.
            SyntaxError: If code is not valid Python.
            ValueError: If the named function is not found after compilation.
        """
        if ast_precheck:
            issues = ASTPreCheck().check(code)
            if issues:
                raise SecurityError(
                    f"AST safety check failed for '{func_name}': "
                    + "; ".join(issues)
                )

        ns = SandboxedNamespace.create()
        compiled = compile(code, f"<skill:{func_name}>", "exec")
        exec(compiled, ns)  # noqa: S102
        fn = ns.get(func_name)
        if fn is None:
            raise ValueError(f"Function '{func_name}' not found in compiled code")
        if not asyncio.iscoroutinefunction(fn):
            raise ValueError(f"Function '{func_name}' must be async")
        return fn
