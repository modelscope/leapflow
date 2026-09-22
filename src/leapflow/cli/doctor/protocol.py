# Copyright (c) Alibaba, Inc. and its affiliates.
"""Diagnostic check protocol and finding aggregate for ``leap doctor``.

The ``DiagnosticCheck`` Protocol defines the contract every health check must
satisfy.  ``Finding`` is a value object that accumulates pass/warning/error
counts and supports merging so the orchestrator can present a single summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Finding:
    """Accumulator for diagnostic results."""

    passed: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fixed: int = 0

    # ── Mutation helpers ────────────────────────────────────────────

    def pass_(self, message: str = "") -> None:
        """Record a passing check."""
        self.passed += 1

    def warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(message)

    def error(self, message: str) -> None:
        """Record an error."""
        self.errors.append(message)

    def fix(self, message: str) -> None:
        """Record an auto-fix action (also counts as passed)."""
        self.fixed += 1
        self.passed += 1

    # ── Aggregation ─────────────────────────────────────────────────

    def merge(self, other: Finding) -> Finding:
        """Return a new Finding combining *self* and *other*."""
        return Finding(
            passed=self.passed + other.passed,
            warnings=[*self.warnings, *other.warnings],
            errors=[*self.errors, *other.errors],
            fixed=self.fixed + other.fixed,
        )

    @property
    def ok(self) -> bool:
        """True when no errors were recorded."""
        return len(self.errors) == 0

    @property
    def total(self) -> int:
        """Total number of individual check assertions."""
        return self.passed + len(self.warnings) + len(self.errors)


@runtime_checkable
class DiagnosticCheck(Protocol):
    """Contract for a single diagnostic check.

    Implementations must expose ``name`` and ``section`` as instance
    attributes and implement an async ``check`` method.
    """

    name: str
    section: str  # platform / config / connectivity / state / tools

    async def check(self, should_fix: bool = False) -> Finding:
        """Run the diagnostic and return a Finding."""
        ...
