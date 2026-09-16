# Copyright (c) Alibaba, Inc. and its affiliates.
"""Guards for change-scoped test selection.

Selection decides which tests get a chance to fail, so a defect here silently
shrinks the suite — the exact failure mode the impact map exists to prevent.
These tests pin the properties that matter: shared foundations escalate to a full
run, a leaf change stays narrow, an unknown file falls back to the import graph
rather than being skipped, and the always-on tiers are never selected away.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import impact  # noqa: E402


def test_escalation_file_exists_and_declares_foundations() -> None:
    """The escalation list must exist and name the shared foundations.

    Without it every change would take the narrow path, including a change to
    ``config.py`` — whose blast radius is the whole codebase.
    """
    patterns = impact.escalation_patterns()
    assert patterns, f"no escalation rules parsed from {impact.ESCALATE_FILE}"
    required = (
        "src/leapflow/config.py",
        "src/leapflow/layout.py",
        "src/leapflow/engine/engine.py",
        "src/leapflow/daemon/service.py",
        "tests/conftest.py",
        "pyproject.toml",
    )
    missing = [rule for rule in required if rule not in patterns]
    assert missing == [], f"escalation rules do not cover: {missing}"


@pytest.mark.parametrize(
    "changed",
    [
        "src/leapflow/config.py",
        "src/leapflow/layout.py",
        "src/leapflow/domain/events.py",
        "src/leapflow/engine/engine.py",
        "src/leapflow/daemon/session_registry.py",
        "tests/conftest.py",
        "tests/_harness/journey.py",
        "pyproject.toml",
        ".github/workflows/ci.yaml",
    ],
)
def test_foundation_changes_force_a_full_run(changed: str) -> None:
    """A change to any shared foundation selects nothing, meaning "run everything"."""
    selected, reason = impact.select_from_paths([changed], coverage={})
    assert selected == [], f"{changed} should escalate but selected {selected}"
    assert "full mock layer" in reason
    assert changed in reason, f"the reason must name the file that escalated: {reason}"


def test_leaf_change_selects_only_related_tests_via_coverage_map() -> None:
    """A coverage map hit narrows the run without falling back to everything."""
    coverage = {
        "tests/test_web_fetch.py": ["src/leapflow/tools/web_fetch.py"],
        "tests/test_repo_map.py": ["src/leapflow/tools/repo_map.py"],
    }
    selected, reason = impact.select_from_paths(
        ["src/leapflow/tools/web_fetch.py"], coverage=coverage, patterns=[]
    )
    assert selected == ["tests/test_web_fetch.py"], selected
    assert "1 via coverage map" in reason


def test_unknown_source_file_falls_back_to_the_import_graph() -> None:
    """A file the map has never seen must still select its dependents.

    A new module is precisely the case where the coverage map is empty, so
    trusting the map alone would run nothing for brand-new code.
    """
    target = "src/leapflow/llm/openai_provider.py"
    assert (REPO_ROOT / target).is_file(), "fixture path no longer exists"

    selected, reason = impact.select_from_paths([target], coverage={}, patterns=[])
    assert selected, "an unknown source file selected no tests at all"
    assert "via import graph" in reason
    assert "1 source file(s) not in the map" in reason


def test_editing_a_test_file_always_runs_it() -> None:
    """A directly edited test must run even when no source changed."""
    selected, _ = impact.select_from_paths(
        ["tests/test_pure_algorithms.py"], coverage={}, patterns=[]
    )
    assert "tests/test_pure_algorithms.py" in selected


def test_documentation_only_change_does_not_narrow_the_run() -> None:
    """A change outside src/ and tests/ cannot be reasoned about, so run everything.

    Being conservative here costs one full run; being clever risks missing the
    case where a doc change accompanied something else.
    """
    selected, reason = impact.select_from_paths(["README.md"], coverage={}, patterns=[])
    assert selected == []
    assert "neither src/ nor tests/" in reason


def test_no_changes_runs_everything() -> None:
    """An empty diff must not be read as "nothing to test"."""
    selected, reason = impact.select_from_paths([], coverage={}, patterns=[])
    assert selected == []
    assert "no changes detected" in reason


def test_always_on_tiers_are_never_selected_away() -> None:
    """The real layer and the ledger run regardless of what changed.

    This is the anti-seesaw guarantee: a suite that can be skipped will be
    skipped, and then a change to one module breaks another with a green run.
    """
    targets = impact.always_on_targets()
    assert "tests/journeys" in targets, "the real journeys must always run"
    assert "tests/regression" in targets, "the incident ledger must always run"
    assert "tests/test_architecture_contracts.py" in targets


def test_selected_targets_exist_on_disk() -> None:
    """Every selectable path resolves, so pytest cannot fail on a stale name."""
    for target in impact.always_on_targets():
        assert (REPO_ROOT / target).exists(), f"always-on target {target} does not exist"


# ── Live journey selection ────────────────────────────────────────
#
# This selection *is* wired into CI, unlike the mock-layer one: each live journey
# costs real tokens and real minutes, so picking the wrong set is expensive in one
# direction and blind in the other.


def test_every_journey_declares_its_metadata() -> None:
    """A journey must state which sources it exercises and whether live adds signal.

    A missing declaration defaults to "always run live", which is the safe
    direction but also the expensive one — so it has to be a conscious choice
    rather than an omission.
    """
    metadata = impact.journey_metadata()
    assert metadata, "no journeys found"
    undeclared = [item.path for item in metadata if not item.subject_paths]
    assert undeclared == [], (
        f"these journeys declare no SUBJECT_PATHS, so every live run pays for them: "
        f"{undeclared}"
    )


def test_declared_subject_paths_exist() -> None:
    """Subject declarations must point at real source paths.

    A renamed module leaves a dead prefix behind, and a dead prefix silently stops
    matching — the journey would quietly drop out of live selection.
    """
    stale: list[str] = []
    for item in impact.journey_metadata():
        for prefix in item.subject_paths:
            if not (REPO_ROOT / prefix).exists():
                stale.append(f"{item.path} -> {prefix}")
    assert stale == [], (
        f"these SUBJECT_PATHS no longer exist, so the journey has silently dropped "
        f"out of live selection: {stale}"
    )


def test_scheduled_run_takes_every_live_capable_journey() -> None:
    """With no diff, the live lane runs everything that has live value."""
    selected, reason = impact.select_journeys([], live_only=True)
    live_capable = [item.path for item in impact.journey_metadata() if item.live_signal]
    assert sorted(selected) == sorted(live_capable), selected
    assert "all eligible journeys" in reason


def test_journeys_without_live_signal_are_never_selected_live() -> None:
    """Control-plane and lifecycle journeys must not spend tokens.

    They assert config layering, the vault, and process lifecycle — none of which
    a real provider can influence. R4 is excluded too: it asserts on injected
    failures a forwarding mode cannot produce.
    """
    excluded = {item.path for item in impact.journey_metadata() if not item.live_signal}
    assert excluded, "expected some journeys to opt out of the live lane"
    for paths in ([], ["src/leapflow/config.py"], ["src/leapflow/daemon/service.py"]):
        selected, _ = impact.select_journeys(paths, live_only=True)
        leaked = excluded.intersection(selected)
        assert leaked == set(), f"{leaked} would spend tokens for no signal"


def test_change_scoped_live_selection_narrows_to_declared_subjects() -> None:
    """A labelled pull request runs only the journeys the change could break."""
    selected, reason = impact.select_journeys(
        ["src/leapflow/recording/recorder.py"], live_only=True, patterns=[]
    )
    assert selected == ["tests/journeys/test_r5_learning.py"], selected
    assert "1/3" in reason


def test_foundation_change_takes_every_live_journey() -> None:
    """An escalating change cannot be narrowed — not even in the live lane."""
    selected, reason = impact.select_journeys(["src/leapflow/config.py"], live_only=True)
    live_capable = [item.path for item in impact.journey_metadata() if item.live_signal]
    assert sorted(selected) == sorted(live_capable)
    assert "escalation rule" in reason


def test_unrelated_change_selects_no_live_journey() -> None:
    """A change no journey claims must not trigger a paid run.

    The offline lanes still run every journey, so nothing goes unverified; this
    only declines to pay a provider for a change none of them exercises.
    """
    selected, reason = impact.select_journeys(
        ["src/leapflow/gateway/adapters/feishu.py"], live_only=True, patterns=[]
    )
    assert selected == []
    assert "no eligible journey declares a subject" in reason


def test_editing_a_journey_selects_it_live() -> None:
    """Changing a journey's own assertions must exercise them against a provider."""
    target = "tests/journeys/test_r1_conversation.py"
    selected, _ = impact.select_journeys([target], live_only=True, patterns=[])
    assert target in selected
