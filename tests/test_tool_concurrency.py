"""Tests for the metadata-driven tool concurrency policy (TC-P0).

Parallel-safety is derived from registry ToolSpec metadata via
execution_policy_for (the same source as idempotency / batch-stop), not a
hardcoded name list. These tests pin: read-only -> parallel; path-scoped
idempotent writes -> parallel iff non-overlapping; once/external/unknown ->
sequential; and the conservative default when no metadata is available.
"""
from __future__ import annotations

from leapflow.engine.tool_concurrency import DefaultConcurrencyPolicy, ToolCall
from leapflow.tools.name_resolver import ToolSpec


def _policy(specs: dict) -> DefaultConcurrencyPolicy:
    return DefaultConcurrencyPolicy(spec_lookup=lambda name: specs.get(name))


def _tc(name: str, **args) -> ToolCall:
    return ToolCall(id=name, name=name, arguments=dict(args))


def test_read_only_tools_run_concurrently() -> None:
    specs = {
        "file_read": ToolSpec(name="file_read", risk_level="read_only", mutates_state=False),
        "text_search": ToolSpec(name="text_search", risk_level="read_only", mutates_state=False),
    }
    concurrent, sequential = _policy(specs).partition(
        [_tc("file_read", path="a.py"), _tc("text_search", query="x")]
    )
    assert {tc.name for tc in concurrent} == {"file_read", "text_search"}
    assert sequential == []


def test_external_side_effect_tools_run_sequentially() -> None:
    specs = {
        "shell_run": ToolSpec(name="shell_run", risk_level="high", mutates_state=True, effect_scope="external"),
        "file_read": ToolSpec(name="file_read", risk_level="read_only"),
    }
    concurrent, sequential = _policy(specs).partition([_tc("shell_run"), _tc("file_read", path="a")])
    assert [tc.name for tc in concurrent] == ["file_read"]
    assert [tc.name for tc in sequential] == ["shell_run"]


def test_mutating_once_session_scoped_tool_runs_sequentially() -> None:
    specs = {
        "custom_commit": ToolSpec(name="custom_commit", risk_level="high", mutates_state=True, idempotency_scope="session"),
        "file_read": ToolSpec(name="file_read", risk_level="read_only"),
    }
    concurrent, sequential = _policy(specs).partition([_tc("file_read", path="a"), _tc("custom_commit")])
    assert [tc.name for tc in concurrent] == ["file_read"]
    assert [tc.name for tc in sequential] == ["custom_commit"]


def test_path_scoped_writes_parallel_iff_non_overlapping() -> None:
    specs = {"file_write": ToolSpec(name="file_write", risk_level="mutating", mutates_state=True)}
    policy = _policy(specs)

    concurrent, sequential = policy.partition(
        [_tc("file_write", path="/repo/a/x.py"), _tc("file_write", path="/repo/b/y.py")]
    )
    assert len(concurrent) == 2 and sequential == []

    concurrent, sequential = policy.partition(
        [_tc("file_write", path="/repo/a"), _tc("file_write", path="/repo/a/x.py")]
    )
    assert [tc.arguments["path"] for tc in concurrent] == ["/repo/a"]
    assert [tc.arguments["path"] for tc in sequential] == ["/repo/a/x.py"]


def test_mutating_idempotent_without_path_is_sequential() -> None:
    specs = {
        "file_write": ToolSpec(name="file_write", risk_level="mutating", mutates_state=True),
        "file_read": ToolSpec(name="file_read", risk_level="read_only"),
    }
    concurrent, sequential = _policy(specs).partition([_tc("file_read", path="a"), _tc("file_write")])
    assert [tc.name for tc in concurrent] == ["file_read"]
    assert [tc.name for tc in sequential] == ["file_write"]


def test_unknown_tool_defaults_to_sequential() -> None:
    specs = {"file_read": ToolSpec(name="file_read", risk_level="read_only")}
    concurrent, sequential = _policy(specs).partition(
        [_tc("file_read", path="a"), _tc("mystery_tool", x=1)]
    )
    assert [tc.name for tc in concurrent] == ["file_read"]
    assert [tc.name for tc in sequential] == ["mystery_tool"]


def test_gp_prefixed_name_resolves_via_lookup_fallback() -> None:
    # The engine's real spec_lookup strips a gp_ prefix; emulate a lookup that
    # only knows the plain name and confirm classification still works.
    specs = {"file_read": ToolSpec(name="file_read", risk_level="read_only")}

    def lookup(name: str):
        return specs.get(name) or specs.get(name.removeprefix("gp_"))

    policy = DefaultConcurrencyPolicy(spec_lookup=lookup)
    concurrent, sequential = policy.partition([_tc("gp_file_read", path="a"), _tc("file_read", path="b")])
    assert len(concurrent) == 2 and sequential == []


def test_no_spec_lookup_defaults_batch_to_sequential() -> None:
    policy = DefaultConcurrencyPolicy(spec_lookup=None)
    concurrent, sequential = policy.partition([_tc("file_read", path="a"), _tc("file_read", path="b")])
    assert concurrent == [] and len(sequential) == 2


def test_single_tool_is_trivially_concurrent() -> None:
    policy = DefaultConcurrencyPolicy(spec_lookup=None)
    concurrent, sequential = policy.partition([_tc("anything")])
    assert len(concurrent) == 1 and sequential == []
