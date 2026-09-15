# Copyright (c) Alibaba, Inc. and its affiliates.
"""Budget guard for the real end-to-end layer.

The real layer earns the right to run on *every* push — never skipped by impact
selection — only by staying small. That is the whole anti-seesaw mechanism: a
suite that can be skipped is a suite that will be skipped, and then a change to
one module breaks another with a green run. So the size of this layer is a hard,
executable constraint rather than a convention.

When this test fails, the fix is to *merge* a journey into an existing one, not
to raise the ceiling.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]
JOURNEYS_DIR = TESTS_ROOT / "journeys"
HARNESS_DIR = TESTS_ROOT / "_harness"

# Ceilings from the testing plan. Raising either needs a design decision, not a
# quick edit: every added journey is paid for on every push, by every developer.
MAX_JOURNEY_CASES = 8
MAX_JOURNEY_MODULES = 8


def _journey_modules() -> list[pathlib.Path]:
    """Return journey test modules (excluding conftest)."""
    if not JOURNEYS_DIR.is_dir():
        return []
    return sorted(p for p in JOURNEYS_DIR.glob("test_*.py"))


def _test_functions(path: pathlib.Path) -> list[str]:
    """Return top-level test function names defined in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def test_journey_case_count_within_budget() -> None:
    """The real layer stays within its case ceiling."""
    per_module = {path.name: _test_functions(path) for path in _journey_modules()}
    total = sum(len(names) for names in per_module.values())
    breakdown = "\n  ".join(
        f"{name}: {len(names)} ({', '.join(names)})" for name, names in sorted(per_module.items())
    )
    assert total <= MAX_JOURNEY_CASES, (
        f"real end-to-end layer has {total} cases, over the {MAX_JOURNEY_CASES} ceiling.\n"
        f"Merge phases into an existing journey instead of adding a case:\n  {breakdown}"
    )
    assert len(per_module) <= MAX_JOURNEY_MODULES, (
        f"{len(per_module)} journey modules, over the {MAX_JOURNEY_MODULES} ceiling"
    )


def test_journeys_are_not_parameterized() -> None:
    """One journey is one case.

    Parameterization is how a coarse layer silently becomes a fine one: the case
    count multiplies without a single new ``def test_``, and the per-case budget
    stops meaning anything.
    """
    offenders: list[str] = []
    for path in _journey_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr != "parametrize":
                continue
            offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        "journeys must not be parameterized — express variation as phases inside "
        f"one journey: {offenders}"
    )


def test_every_journey_declares_a_deadline_and_finishes() -> None:
    """Each journey must call ``finish()`` so its budget and misses are asserted.

    ``finish()`` is where "no cassette miss" and "within the time budget" are
    checked. A journey that forgets it can pass while replaying nothing, which
    would make the whole layer decorative.
    """
    missing: list[str] = []
    for path in _journey_modules():
        source = path.read_text(encoding="utf-8")
        if ".finish()" not in source:
            missing.append(path.name)
    assert missing == [], (
        f"these journeys never call journey.finish(), so cassette misses and the "
        f"time budget go unchecked: {missing}"
    )


def test_every_journey_declares_both_cost_ceilings() -> None:
    """A journey must bound its provider calls *and* its tokens.

    The two catch different failures. The call ceiling stops a turn that never
    converges; the token ceiling stops prompt growth, which raises cost without
    adding a single round. Either one alone leaves a way for the live lane to get
    slower or more expensive without anything turning red.
    """
    missing: list[str] = []
    for path in _journey_modules():
        source = path.read_text(encoding="utf-8")
        for setting in ("max_llm_calls=", "max_llm_tokens="):
            if setting not in source:
                missing.append(f"{path.name} is missing {setting.rstrip('=')}")
    assert missing == [], (
        "these journeys do not bound their own cost, so a regression in "
        f"convergence or prompt size would go unnoticed: {missing}"
    )


def test_harness_is_not_collected_as_tests() -> None:
    """Harness modules are infrastructure and must not define test functions."""
    offenders: list[str] = []
    for path in sorted(HARNESS_DIR.glob("*.py")):
        for name in _test_functions(path):
            offenders.append(f"{path.name}::{name}")
    assert offenders == [], (
        f"move these out of tests/_harness/ — harness code must stay infrastructure: {offenders}"
    )


@pytest.mark.parametrize("required", ["cassette.py", "cassette_proxy.py", "leapd.py", "journey.py"])
def test_harness_modules_present(required: str) -> None:
    """The four harness pieces the journeys depend on exist."""
    assert (HARNESS_DIR / required).is_file(), f"missing harness module {required}"
