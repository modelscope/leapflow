# Copyright (c) Alibaba, Inc. and its affiliates.
"""The incident ledger: one entry per outage that shipped with a green suite.

Every entry below is a real regression that reached users. What they had in
common was that the suite agreed with the defect — so the ledger's job is not to
re-test the fix, but to make sure the coverage that now catches it cannot
disappear quietly. A ledger entry fails when its home is deleted, renamed, or
moved somewhere that change-scoped selection could skip.

The ledger also carries structural guards for contracts that have no natural
home in a single module's tests, because that is exactly where these defects hid.
Where a behavioral guard exists, it is preferred: an end-to-end assertion that a
session-scoped status reports real usage is stronger and more durable than any
rule about which attribute the reporting code happens to read.

Adding an entry is how a post-mortem ends. Removing one requires arguing that the
failure mode is now impossible, not merely unlikely.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass

import pytest

TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT.parent / "src" / "leapflow"


@dataclass(frozen=True)
class Incident:
    """One past outage and the coverage that now catches it."""

    key: str
    symptom: str
    home: str
    required_tests: tuple[str, ...]
    journeys: tuple[str, ...] = ()


LEDGER: tuple[Incident, ...] = (
    Incident(
        key="status-bar-frozen-at-zero",
        symptom=(
            "The status bar showed 0/<limit> forever because reporting code read "
            "ctx.engine — the template sessions are cloned from, which never "
            "accumulates turns — instead of the caller's session engine."
        ),
        home="test_runtime_metadata_and_wrapping.py",
        required_tests=(),
        # The behavioral guard: a session-scoped status must report real usage,
        # and an unscoped one must report none. That is a stronger check than any
        # structural rule about which attribute the reporting code reads.
        journeys=("test_r1_conversation.py", "test_r2_isolation.py"),
    ),
    Incident(
        key="session-identity-adopted-from-another-client",
        symptom=(
            "A second TUI adopted the first client's session id from an unscoped "
            "status(), sent it with its own workspace, and was rejected on every "
            "turn — with advice that could not work, because a fresh client "
            "re-adopted the same id on its first poll."
        ),
        home="test_multi_client_session_isolation.py",
        required_tests=(
            "test_status_without_a_session_reports_no_identity",
            "test_client_adopts_a_session_only_when_it_has_none",
            "test_cross_client_fallback_is_named_for_what_it_does",
            "test_reusing_a_session_from_another_workspace_is_refused",
        ),
        journeys=("test_r2_isolation.py",),
    ),
    Incident(
        key="local-defect-classified-as-provider-condition",
        symptom=(
            "A mistyped attribute whose name contained 'context' raised "
            "AttributeError inside the provider call's try block; the message-"
            "matching provider classifier read it as a context overflow and drove "
            "every turn through three compressions, a failover, and a credential "
            "rotation before halting."
        ),
        home="test_internal_defect_reporting.py",
        required_tests=(
            "test_defect_types_are_matched_by_type_not_message",
            "test_internal_defect_is_not_classified_as_a_provider_condition",
            "test_real_provider_conditions_still_use_the_provider_taxonomy",
            "test_terminal_decision_carries_an_actionable_interaction",
        ),
        journeys=("test_r4_recovery.py",),
    ),
    Incident(
        key="long-answers-lost-their-tail",
        symptom=(
            "prompt_toolkit's renderer clips at the window edge rather than "
            "reflowing, so enabling soft_wrap on the shared console silently "
            "truncated the end of every long answer."
        ),
        home="test_runtime_metadata_and_wrapping.py",
        required_tests=(),
    ),
    Incident(
        key="calibration-wiring-faked-by-tests",
        symptom=(
            "Calibration tests built the engine with object.__new__ and assigned "
            "the attributes the method reads, so they agreed with a wrong "
            "attribute name and stayed green while every real turn raised "
            "AttributeError."
        ),
        home="test_budget_calibration.py",
        required_tests=(
            "test_real_engine_calibrates_on_the_production_path",
            "test_telemetry_helper_absorbs_its_own_defects",
        ),
    ),
)


def _test_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


@pytest.mark.parametrize("incident", LEDGER, ids=lambda item: item.key)
def test_incident_coverage_still_exists(incident: Incident) -> None:
    """Each recorded outage still has a home, with the tests that catch it.

    A deleted or renamed test is how a fixed defect becomes an unfixed one again.
    """
    home = TESTS_ROOT / incident.home
    assert home.is_file(), (
        f"incident {incident.key!r} has no coverage left: {incident.home} is gone.\n"
        f"Symptom that will return: {incident.symptom}"
    )
    if incident.required_tests:
        present = _test_names(home)
        missing = sorted(set(incident.required_tests) - present)
        assert missing == [], (
            f"incident {incident.key!r} lost these guards from {incident.home}: {missing}\n"
            f"Symptom that will return: {incident.symptom}"
        )

    for journey in incident.journeys:
        path = TESTS_ROOT / "journeys" / journey
        assert path.is_file(), (
            f"incident {incident.key!r} lost its end-to-end guard: "
            f"journeys/{journey} is gone.\n"
            f"Symptom that will return: {incident.symptom}"
        )


def test_ledger_covers_every_known_incident() -> None:
    """The ledger is the index of past outages; keep it complete.

    AGENTS.md records five failures that shipped green. If the ledger holds fewer,
    one of them has no permanent home.
    """
    assert len(LEDGER) >= 5, f"the ledger has shrunk to {len(LEDGER)} entries"
    assert len({item.key for item in LEDGER}) == len(LEDGER), "duplicate ledger keys"
    for item in LEDGER:
        assert item.symptom.strip(), f"{item.key} has no symptom description"


# ── Structural guards for contracts with no single natural home ───────────


def _keyword_values(path: pathlib.Path, keyword: str) -> list[tuple[int, object]]:
    """Return (line, value) for every ``keyword=<constant>`` argument in ``path``.

    Parsed rather than grepped: the incident that motivated this guard is
    described in a comment inside the very file being checked, so a text search
    finds the explanation and reports it as the defect.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kwarg in node.keywords:
            if kwarg.arg == keyword and isinstance(kwarg.value, ast.Constant):
                found.append((kwarg.lineno, kwarg.value.value))
    return found


def _reporting_modules() -> list[pathlib.Path]:
    """Modules that report runtime state to a client."""
    candidates = [
        SRC_ROOT / "daemon" / "service.py",
        SRC_ROOT / "daemon" / "session_coordinator.py",
    ]
    return [path for path in candidates if path.is_file()]


def test_session_engine_resolution_has_a_single_entry_point() -> None:
    """The resolver that reporting code must use has to keep existing.

    A structural ban on reading ``ctx.engine`` is the wrong guard: the same
    expression is legitimate when fetching the *template* to clone session engines
    from, and telling those two uses apart from the syntax alone needs a brittle
    rule. What can be checked cheaply is that the single sanctioned entry point
    still exists — and the behavioral proof lives in R1/R2, which assert that a
    session-scoped status reports real usage while an unscoped one reports none.
    """
    names = {"resolve_session_engine": False, "_active_engine": False}
    for path in _reporting_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                names[node.name] = True

    missing = sorted(name for name, found in names.items() if not found)
    assert missing == [], (
        f"the sanctioned session-engine resolver(s) {missing} no longer exist; "
        "without them reporting code resolves the engine template and silently "
        "reports zeros, which froze the status bar at 0/<limit>"
    )


def test_cross_session_fallback_is_named_for_what_it_does() -> None:
    """An aggregate resolver must say so in its name.

    "the current session" invites the misuse that leaked one client's identity to
    another; ``most_recent_any_client`` cannot be mistaken for a per-caller lookup.
    """
    registry = SRC_ROOT / "daemon" / "session_registry.py"
    # A moved home is exactly the quiet disappearance this ledger guards against,
    # so a relocation must fail loudly (forcing this path to be re-pointed at the
    # module's new home) rather than skip. As of this writing the module still
    # lives at daemon/session_registry.py and defines most_recent_any_client.
    assert registry.is_file(), (
        "session_registry.py is no longer at daemon/session_registry.py; the "
        "cross-session fallback guard lost its home. Update this ledger entry to "
        "the module's new path instead of letting the check skip."
    )
    tree = ast.parse(registry.read_text(encoding="utf-8"))
    method_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "most_recent_any_client" in method_names, (
        "the cross-session fallback was renamed; a resolver that ignores workspace "
        "and client identity must keep saying so in its name"
    )
    for banned in ("current_session", "get_current", "active_session"):
        assert banned not in method_names, (
            f"{registry.name} defines {banned!r} — a friendly name for an aggregate "
            "lookup is what let one client adopt another's session"
        )


def test_shared_console_does_not_enable_soft_wrap() -> None:
    """Wrapping belongs to the console layer, with ``soft_wrap`` off.

    prompt_toolkit's renderer clips at the window edge instead of reflowing, so
    ``soft_wrap=True`` on the shared console drops the tail of long answers.
    """
    console_path = SRC_ROOT / "cli" / "tui_app" / "console.py"
    # As with the session-registry guard above, a relocated console module must
    # fail loudly rather than skip: a silent skip is the quiet disappearance the
    # ledger exists to prevent. The module still lives at cli/tui_app/console.py
    # and states soft_wrap=False explicitly.
    assert console_path.is_file(), (
        "console.py is no longer at cli/tui_app/console.py; the soft_wrap guard "
        "lost its home. Update this ledger entry to the module's new path instead "
        "of letting the check skip."
    )

    settings = _keyword_values(console_path, "soft_wrap")
    enabled = [line for line, value in settings if value is True]
    assert enabled == [], (
        f"{console_path.name} enables soft_wrap at line(s) {enabled}; long answers "
        "will lose their tail under prompt_toolkit's renderer"
    )
    assert any(value is False for _, value in settings), (
        f"{console_path.name} no longer states soft_wrap explicitly — the wrapping "
        "contract must stay visible at the call site"
    )
