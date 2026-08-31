"""Provider response-shape drift guard.

``tools/sync_fixtures.py`` distils every recorded cassette into the *shapes* the
provider actually sends. This guard asserts that those shapes still contain the
fields LeapFlow's parsers read. It is the drift detector the mock layer never
had: previously a provider could drop or rename a field and the suite would keep
passing, because every mock returned a body written from memory.

When this fails, the fix is not to relax the assertion. It is to look at the
diff in ``response_shapes.json`` and update the parser that depended on the
field which disappeared.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "_fixtures"
    / "llm_responses"
    / "response_shapes.json"
)
RECORDING_ROOT = pathlib.Path(__file__).resolve().parents[1] / "_fixtures" / "recordings"

# Fields the production parser in leapflow/llm/openai_provider.py reads from a
# successful non-streamed completion.
_REQUIRED_COMPLETION_PATHS = (
    ("choices", 0, "message", "role"),
    ("choices", 0, "message", "content"),
    ("choices", 0, "finish_reason"),
    ("usage", "prompt_tokens"),
    ("usage", "completion_tokens"),
    ("usage", "total_tokens"),
)

# Fields read when the model asks for a tool.
_REQUIRED_TOOL_CALL_PATHS = (
    ("choices", 0, "message", "tool_calls", 0, "id"),
    ("choices", 0, "message", "tool_calls", 0, "function", "name"),
    ("choices", 0, "message", "tool_calls", 0, "function", "arguments"),
)

# Fields the recovery classifier reads from an error body.
_REQUIRED_ERROR_PATHS = (
    ("error", "message"),
    ("error", "type"),
    ("error", "code"),
)

# Usage keys the effective-cost accounting depends on.
_REQUIRED_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _shapes() -> dict[str, Any]:
    if not FIXTURE.is_file():
        pytest.skip(f"no derived fixtures at {FIXTURE}; run `make sync-fixtures`")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _resolve(shape: Any, path: tuple[Any, ...]) -> Any:
    """Walk a derived shape by path, returning None when a step is absent."""
    node = shape
    for step in path:
        if isinstance(step, int):
            if not isinstance(node, list) or not node:
                return None
            node = node[0]
            continue
        if not isinstance(node, dict) or step not in node:
            return None
        node = node[step]
    return node


def _has(shapes: list[Any], path: tuple[Any, ...]) -> bool:
    return any(_resolve(shape, path) is not None for shape in shapes)


def test_derived_fixtures_are_present_and_non_trivial() -> None:
    """The distilled shapes must actually describe recorded traffic.

    Checked against the shapes themselves rather than a count of exchanges scanned.
    That count used to live in the fixture and was asserted here, but it also made
    ``sync_fixtures --check`` fail every time a journey was added -- identical shapes,
    red build -- so it was removed from the contract. Asserting the structure is the
    stronger test anyway: a corpus of two hundred stubs would satisfy a count and fail
    this.
    """
    shapes = _shapes()
    completions = shapes.get("completion_shapes") or []
    assert completions, "no successful completion shape recorded"
    for shape in completions:
        missing = {"choices", "usage"} - set(shape)
        assert not missing, (
            f"a completion body without {sorted(missing)} is not provider traffic; "
            "re-run `make seed-cassettes && make sync-fixtures`"
        )
    errors = shapes.get("error_shapes") or []
    assert errors, (
        "no error shape recorded — the recovery classifier's inputs are unverified"
    )
    for shape in errors:
        assert "error" in shape, f"an error body must carry an error object: {sorted(shape)}"
    assert shapes.get("usage_fields"), "no usage fields seen, so token accounting is unverified"


def test_recorded_traffic_carries_the_optional_fields_production_reads() -> None:
    """Real recordings must still expose the fields the provider layer opts into.

    These were the concrete proof that hand-written bodies drift: the seeded
    cassettes carried neither ``reasoning_content`` (read to surface thinking on
    dashscope/deepseek profiles) nor ``prompt_tokens_details`` (read by
    ``_extract_cached_tokens`` for prefix-cache accounting). Both appeared only
    once real traffic was recorded. Losing them again would silently disable two
    production features.
    """
    shapes = _shapes()
    if not RECORDING_ROOT.is_dir() or not any(RECORDING_ROOT.iterdir()):
        pytest.skip(
            "no real recordings committed; these fields only appear in provider "
            "traffic (run `make record-traffic` with credentials)"
        )

    completions = shapes.get("completion_shapes") or []
    assert _has(completions, ("choices", 0, "message", "reasoning_content")), (
        "no recorded response carries reasoning_content; thinking extraction is "
        "now asserted only against bodies nobody has verified"
    )
    reported = set(shapes.get("usage_fields") or ())
    assert "prompt_tokens_details" in reported, (
        "no recorded response reports prompt_tokens_details; prefix-cache "
        "accounting in _extract_cached_tokens is unverified"
    )


def test_completion_shape_carries_every_field_the_parser_reads() -> None:
    """A successful body must still expose the fields the provider layer reads."""
    completions = _shapes().get("completion_shapes") or []
    missing = [
        ".".join(str(step) for step in path)
        for path in _REQUIRED_COMPLETION_PATHS
        if not _has(completions, path)
    ]
    assert missing == [], (
        "recorded provider responses no longer carry these fields, which "
        f"leapflow/llm/openai_provider.py reads: {missing}"
    )


def test_tool_call_shape_carries_every_field_the_engine_reads() -> None:
    """Native tool calls must still expose id, name, and arguments."""
    completions = _shapes().get("completion_shapes") or []
    if not _has(completions, ("choices", 0, "message", "tool_calls")):
        pytest.skip("no tool-calling response recorded yet")
    missing = [
        ".".join(str(step) for step in path)
        for path in _REQUIRED_TOOL_CALL_PATHS
        if not _has(completions, path)
    ]
    assert missing == [], (
        f"recorded tool calls are missing fields the engine dispatches on: {missing}"
    )


def test_error_shape_carries_every_field_the_classifier_reads() -> None:
    """Error bodies must still expose message, type, and code."""
    errors = _shapes().get("error_shapes") or []
    missing = [
        ".".join(str(step) for step in path)
        for path in _REQUIRED_ERROR_PATHS
        if not _has(errors, path)
    ]
    assert missing == [], (
        "recorded provider errors no longer carry these fields, which the "
        f"recovery classifier reads: {missing}"
    )


def test_usage_accounting_fields_are_present() -> None:
    """Token accounting depends on these keys; their loss is silent otherwise."""
    reported = set(_shapes().get("usage_fields") or ())
    missing = [field for field in _REQUIRED_USAGE_FIELDS if field not in reported]
    assert missing == [], (
        f"providers no longer report {missing}; token accounting and the context "
        "status bar read them"
    )


def test_recorded_error_codes_cover_the_recovery_categories() -> None:
    """The failure classes the recovery journey depends on must be recorded.

    Without a recorded example of each, the classifier's handling of that class
    is asserted only against a body somebody wrote by hand.
    """
    codes = set(_shapes().get("error_codes") or ())
    expected = {"rate_limit_exceeded", "context_length_exceeded"}
    missing = sorted(expected - codes)
    assert missing == [], (
        f"no recorded provider error for {missing}; the recovery journey's "
        "injected failures are what keep these classes honest"
    )
