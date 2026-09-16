# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fitness functions for the test suite itself.

The mock layer is an asset — 1400-plus cases of branch coverage that no
end-to-end journey could afford. What makes it a liability is when a mock
encodes an assumption the product does not hold, because then it agrees with a
defect instead of catching it. These guards keep the mock layer honest without
rewriting it:

1. faked construction (``object.__new__``) must be backed by a real-instance test;
2. LLM response bodies come from recorded traffic, not from hand-written literals;
3. mocks stay on process/network/OS boundaries and never replace internal logic.

Existing debt is listed explicitly and the lists may only shrink: a stale entry
fails the test, so paying the debt down is rewarded and quietly adding to it is
not possible.
"""

from __future__ import annotations

import ast
import pathlib

TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── Guard 1: faked wiring needs real-instance cover ──────────────────────

# ``object.__new__(X)`` plus assignment of the private attributes X reads cannot
# detect a wrong attribute *name* — the test simply agrees with the typo. That is
# how the calibration feature raised AttributeError on every real turn while the
# suite stayed green. The technique is still legitimate for pinning down ordering
# contracts, but only when the same file also builds the class for real and drives
# the production path. This guard enforces that pairing.
#
# Files here predate the rule and still lack real-instance cover. The list must
# only ever shrink: a stale entry fails the test, so paying the debt is rewarded
# and adding to it is not possible without an explicit edit.
_FAKED_WIRING_DEBT = frozenset(
    {
        "test_context_budget_scaling.py",
        "test_daemon_isolation.py",
    }
)

# ── Guard 2: no hand-written LLM response bodies ──────────────────────────

# Markers of a hand-authored OpenAI-shaped body. Real shapes come from
# tests/_fixtures/cassettes (see tools/sync_fixtures.py), so a provider changing
# its payload shows up as a fixture diff instead of passing forever against a
# body nobody has verified.
_RESPONSE_BODY_MARKERS = ("chat.completion", "chatcmpl-", "prompt_tokens_details")

_HAND_WRITTEN_BODY_ALLOWLIST = frozenset(
    {
        # The harness authors bodies on purpose: it is the component that defines
        # what a recorded response looks like.
        "_harness/cassette.py",
        # The drift guard names the fields production reads so it can assert that
        # recorded traffic still carries them. Naming a field is the opposite of
        # hand-writing a body — it is what makes a missing field fail.
        "regression/test_provider_shape_drift.py",
    }
)

# Predates the cassette store. Must only shrink — a stale entry fails the test.
_HAND_WRITTEN_BODY_DEBT = frozenset(
    {
        "test_adaptive_depth.py",
        "test_gateway_adapters.py",
    }
)

# ── Guard 3: mocks only at boundaries ─────────────────────────────────────

# Patch targets that replace LeapFlow's own decision-making rather than an
# external boundary. Mocking these proves the test's own arrangement, not the
# behavior under test.
_INTERNAL_PATCH_TARGETS = (
    "leapflow.engine.recovery_coordinator.RecoveryCoordinator.evaluate",
    "leapflow.engine.unified_classifier",
    "leapflow.daemon.session_registry.SessionRegistry.acquire",
    "leapflow.config_service.ConfigService.set",
)


def _test_modules() -> list[pathlib.Path]:
    """Return every test module, excluding this file."""
    return sorted(
        path
        for path in TESTS_ROOT.rglob("*.py")
        if path.name.startswith(("test_", "conftest"))
        and path.resolve() != pathlib.Path(__file__).resolve()
    )


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(TESTS_ROOT).as_posix()


def _harness_modules() -> list[pathlib.Path]:
    """Return harness modules, which are infrastructure rather than tests."""
    return sorted((TESTS_ROOT / "_harness").glob("*.py"))


def test_faked_wiring_is_backed_by_a_real_instance_test() -> None:
    """A file that fakes construction must also build the class for real.

    Faking is acceptable for ordering contracts, but on its own it can only
    confirm the names the test itself chose. Constructing the same class properly
    somewhere in the file is what makes a mistyped attribute fail.
    """
    offenders: list[str] = []
    paid_off: list[str] = []
    for path in _test_modules():
        relative = _relative(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        faked: dict[str, int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "__new__":
                continue
            if not (isinstance(node.value, ast.Name) and node.value.id == "object"):
                continue
            parent_call = _enclosing_call(tree, node)
            target = _first_name_argument(parent_call)
            if target:
                faked.setdefault(target, node.lineno)

        uncovered = [
            f"{relative}:{lineno} fakes {name} but never constructs {name}(...)"
            for name, lineno in sorted(faked.items())
            if f"{name}(" not in source.replace(f"object.__new__({name})", "")
        ]
        if relative in _FAKED_WIRING_DEBT:
            if not uncovered:
                paid_off.append(relative)
            continue
        offenders.extend(uncovered)

    assert offenders == [], (
        "these files fake construction without ever driving the real one, so a "
        "wrong attribute name cannot fail them — add a test that builds the class "
        "and calls the production path:\n  " + "\n  ".join(offenders)
    )
    assert paid_off == [], (
        "these files now have real-instance cover; remove them from "
        f"_FAKED_WIRING_DEBT so the list keeps shrinking: {paid_off}"
    )


def _enclosing_call(tree: ast.AST, needle: ast.AST) -> ast.Call | None:
    """Return the Call node whose func is ``needle``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.func is needle:
            return node
    return None


def _first_name_argument(call: ast.Call | None) -> str:
    """Return the first positional argument's name, when it is a bare name."""
    if call is None or not call.args:
        return ""
    first = call.args[0]
    return first.id if isinstance(first, ast.Name) else ""


def test_llm_response_bodies_come_from_recorded_traffic() -> None:
    """Provider payload shapes are recorded, never hand-written.

    A hand-authored body freezes one developer's belief about the wire format. It
    keeps passing after the provider changes, which is precisely the failure mode
    the cassette store exists to remove.
    """
    offenders: list[str] = []
    paid_off: list[str] = []
    for path in list(_test_modules()) + _harness_modules():
        relative = _relative(path)
        if relative in _HAND_WRITTEN_BODY_ALLOWLIST:
            continue
        source = path.read_text(encoding="utf-8")
        found = next((m for m in _RESPONSE_BODY_MARKERS if m in source), "")
        if relative in _HAND_WRITTEN_BODY_DEBT:
            if not found:
                paid_off.append(relative)
            continue
        if found:
            offenders.append(f"{relative} (contains {found!r})")

    assert offenders == [], (
        "these files hand-write an OpenAI-shaped response body; use a recorded "
        "cassette or a fixture derived from one (make sync-fixtures):\n  "
        + "\n  ".join(offenders)
    )
    assert paid_off == [], (
        "these files no longer hand-write response bodies; remove them from "
        f"_HAND_WRITTEN_BODY_DEBT: {paid_off}"
    )


def test_mocks_stay_on_external_boundaries() -> None:
    """Mock external IO, never LeapFlow's own decision points.

    Replacing the recovery coordinator, the classifier, or the session registry
    with a double means the test asserts its own stub. These are exactly the
    components whose real behavior the suite is supposed to pin down.
    """
    offenders: list[str] = []
    for path in _test_modules():
        source = path.read_text(encoding="utf-8")
        for target in _INTERNAL_PATCH_TARGETS:
            if f'"{target}"' in source or f"'{target}'" in source:
                offenders.append(f"{_relative(path)} patches {target}")

    assert offenders == [], (
        "these tests replace internal logic with a double; mock only process, "
        "network, or OS boundaries:\n  " + "\n  ".join(offenders)
    )


def test_journeys_do_not_mock_anything() -> None:
    """The real layer earns its cost only by staying real.

    A journey that reaches for ``unittest.mock`` has stopped being an end-to-end
    check and has become an expensive unit test: it pays for a daemon subprocess
    and then stubs out the thing it was meant to exercise.
    """
    journeys_dir = TESTS_ROOT / "journeys"
    offenders: list[str] = []
    for path in sorted(journeys_dir.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for marker in ("unittest.mock", "MagicMock", "AsyncMock", "monkeypatch"):
            if marker in source:
                offenders.append(f"{_relative(path)} uses {marker}")

    assert offenders == [], (
        "journeys must exercise the real system end to end; move anything that "
        "needs a double down to the mock layer:\n  " + "\n  ".join(offenders)
    )


def test_committed_cassettes_are_readable_and_non_empty() -> None:
    """Replay-lane inputs must be present and loadable.

    The PR and main lanes run offline against these files. A corrupt or missing
    store turns every journey into a cassette-miss failure, so it is worth
    failing fast with a clear reason.
    """
    from tests._harness.cassette import CassetteStore

    root = TESTS_ROOT / "_fixtures" / "cassettes"
    assert root.is_dir(), (
        f"no committed cassette store at {root} — run `make seed-cassettes`"
    )
    journeys = sorted(p for p in root.iterdir() if p.is_dir())
    assert journeys, f"cassette store {root} has no journey directories"

    for directory in journeys:
        store = CassetteStore(directory)  # raises on unreadable content
        assert len(store) > 0, f"cassette directory {directory.name} is empty"
        for key in store.keys():
            record = store.get(key)
            assert record is not None and record.responses, (
                f"cassette {key} in {directory.name} carries no response"
            )
