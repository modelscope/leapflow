# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-run execution authority for opt-in benchmark capabilities."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class BenchmarkRuntimeContext:
    """Explicit authority granted to benchmark adapters for one run."""

    live_llm_enabled: bool = False
    require_live_llm: bool = False
    hardware_enabled: bool = False


_RUNTIME_CONTEXT: ContextVar[BenchmarkRuntimeContext] = ContextVar(
    "benchmark_runtime_context",
    default=BenchmarkRuntimeContext(),
)


def current_runtime_context() -> BenchmarkRuntimeContext:
    """Return the authority available to the currently executing benchmark."""
    return _RUNTIME_CONTEXT.get()


def install_runtime_context(context: BenchmarkRuntimeContext) -> Token[BenchmarkRuntimeContext]:
    """Install an explicit context and return its reset token."""
    return _RUNTIME_CONTEXT.set(context)


def reset_runtime_context(token: Token[BenchmarkRuntimeContext]) -> None:
    """Restore the parent benchmark authority context."""
    _RUNTIME_CONTEXT.reset(token)


@contextmanager
def runtime_context(context: BenchmarkRuntimeContext) -> Iterator[None]:
    """Scope explicit benchmark authority to a single execution flow."""
    token = install_runtime_context(context)
    try:
        yield
    finally:
        reset_runtime_context(token)


__all__ = [
    "BenchmarkRuntimeContext",
    "current_runtime_context",
    "install_runtime_context",
    "reset_runtime_context",
    "runtime_context",
]
