# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for PromptAssembler — the engine's per-turn prompt/context assembly."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List


from leapflow.engine._stream_helpers import TaskContract
from leapflow.engine.prompt_assembler import PromptAssembler


# ── Minimal engine stub ──────────────────────────────────────────────


def _stub_engine(
    *,
    workspace_root: str = "/tmp/test-workspace",
    turn_count: int = 1,
    task_contract: TaskContract | None = None,
    skill_index: Any = None,
    wm_messages: List[Dict[str, Any]] | None = None,
    knowledge_store: Any = None,
    environment_fingerprint_id: str = "",
    last_turn_tool_categories: frozenset[str] = frozenset(),
) -> SimpleNamespace:
    """Build a minimal engine-like object with only the attributes PromptAssembler reads."""
    settings = SimpleNamespace(
        workspace_root=workspace_root,
        research_protocol_length_threshold=120,
    )
    wm = SimpleNamespace(
        as_chat_messages=lambda: list(wm_messages or []),
    )
    engine = SimpleNamespace(
        _settings=settings,
        _session_turn_count=turn_count,
        _current_task_contract=task_contract,
        _skill_index=skill_index,
        _wm=wm,
        _knowledge_store=knowledge_store,
        _knowledge_store_unavailable=False,
        _environment_fingerprint_id=environment_fingerprint_id,
        _last_turn_tool_categories=last_turn_tool_categories,
        _manifests_by_name=None,
        _focus_state=SimpleNamespace(
            render_prompt_context=lambda _res: "",
        ),
        _reference_resolver=SimpleNamespace(
            resolve=lambda _txt, _fs: SimpleNamespace(
                target_id=None, needs_clarification=False
            ),
        ),
        _last_reference_resolution=None,
    )
    return engine


# ── TaskContract building ────────────────────────────────────────────


class TestBuildTaskContract:
    def test_basic_contract_fields(self) -> None:
        engine = _stub_engine(workspace_root="/home/user/project", turn_count=3)
        assembler = PromptAssembler(engine)
        contract = assembler._build_task_contract("do something")
        assert contract.task_id == "turn-3"
        assert contract.original_request == "do something"
        assert "/home/user/project" in contract.workspace_root
        assert contract.allowed_roots == (contract.workspace_root,)

    def test_whitespace_stripped_from_request(self) -> None:
        engine = _stub_engine()
        assembler = PromptAssembler(engine)
        contract = assembler._build_task_contract("  hello world  ")
        assert contract.original_request == "hello world"

    def test_short_text_no_research_protocol(self) -> None:
        protocol = PromptAssembler._research_protocol_for("short text")
        assert protocol == ()

    def test_long_text_gets_research_protocol(self) -> None:
        long_text = "x" * 200
        protocol = PromptAssembler._research_protocol_for(long_text)
        assert len(protocol) > 0
        assert any("DECOMPOSE" in line for line in protocol)


# ── Task contract block rendering ────────────────────────────────────


class TestTaskContractBlock:
    def test_no_contract_returns_empty(self) -> None:
        engine = _stub_engine(task_contract=None)
        assembler = PromptAssembler(engine)
        assert assembler._task_contract_block() == ""

    def test_contract_renders_heading(self) -> None:
        contract = TaskContract(
            task_id="turn-1",
            original_request="test",
            workspace_root="/tmp",
            allowed_roots=("/tmp",),
        )
        engine = _stub_engine(task_contract=contract)
        assembler = PromptAssembler(engine)
        block = assembler._task_contract_block()
        assert block.startswith("## Task Contract")
        assert "turn-1" in block


# ── System prompt with task contract ─────────────────────────────────


class TestAppendTaskContract:
    def test_appends_to_existing_system(self) -> None:
        contract = TaskContract(
            task_id="turn-1",
            original_request="hello",
            workspace_root="/tmp",
            allowed_roots=("/tmp",),
        )
        engine = _stub_engine(task_contract=contract)
        assembler = PromptAssembler(engine)
        result = assembler._append_task_contract_to_system("You are a helpful assistant.")
        assert "You are a helpful assistant." in result
        assert "## Task Contract" in result

    def test_strips_old_contract_before_appending(self) -> None:
        old_system = "You are an assistant.\n\n## Task Contract\n- Task ID: turn-0"
        contract = TaskContract(
            task_id="turn-1",
            original_request="new task",
            workspace_root="/tmp",
            allowed_roots=("/tmp",),
        )
        engine = _stub_engine(task_contract=contract)
        assembler = PromptAssembler(engine)
        result = assembler._append_task_contract_to_system(old_system)
        assert "turn-0" not in result
        assert "turn-1" in result

    def test_no_contract_returns_original(self) -> None:
        engine = _stub_engine(task_contract=None)
        assembler = PromptAssembler(engine)
        result = assembler._append_task_contract_to_system("system text")
        assert result == "system text"


# ── Strip task contract block ────────────────────────────────────────


class TestStripTaskContractBlock:
    def test_removes_trailing_contract(self) -> None:
        content = "Preamble text.\n\n## Task Contract\n- Task ID: turn-1"
        result = PromptAssembler._strip_task_contract_block(content)
        assert "## Task Contract" not in result
        assert "Preamble text." in result

    def test_content_only_contract(self) -> None:
        result = PromptAssembler._strip_task_contract_block("## Task Contract\n- stuff")
        assert result == ""

    def test_no_contract_unchanged(self) -> None:
        content = "Just normal text."
        assert PromptAssembler._strip_task_contract_block(content) == content


# ── Session summary context ──────────────────────────────────────────


class TestBuildSessionSummaryContext:
    def test_empty_messages(self) -> None:
        engine = _stub_engine(wm_messages=[])
        assembler = PromptAssembler(engine)
        result = assembler._build_session_summary_context(max_messages=10)
        assert result == ""

    def test_user_turn_preserved(self) -> None:
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        engine = _stub_engine(wm_messages=msgs)
        assembler = PromptAssembler(engine)
        result = assembler._build_session_summary_context(max_messages=10)
        assert "[user]" in result
        assert "What is Python?" in result
        assert "[assistant]" in result

    def test_tool_call_turn_extracted(self) -> None:
        msgs = [
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": "[Called: shell, read_file]"},
        ]
        engine = _stub_engine(wm_messages=msgs)
        assembler = PromptAssembler(engine)
        result = assembler._build_session_summary_context(max_messages=10)
        assert "called:" in result
        assert "shell" in result

    def test_max_messages_respected(self) -> None:
        msgs = [
            {"role": "user", "content": f"Question {i}"} for i in range(20)
        ]
        engine = _stub_engine(wm_messages=msgs)
        assembler = PromptAssembler(engine)
        result = assembler._build_session_summary_context(max_messages=3)
        # Should only include the last 3
        assert "Question 17" in result
        assert "Question 0" not in result


# ── Skill section ────────────────────────────────────────────────────


class TestBuildSkillSection:
    def test_no_skill_index(self) -> None:
        engine = _stub_engine(skill_index=None)
        assembler = PromptAssembler(engine)
        assert assembler._build_skill_section(include_skills=True) == ""

    def test_skills_excluded_by_plan(self) -> None:
        engine = _stub_engine(skill_index=SimpleNamespace(
            get_entries=lambda: [{"name": "test"}],
            compact_index_text=lambda entries: "test skill",
        ))
        assembler = PromptAssembler(engine)
        assert assembler._build_skill_section(include_skills=False) == ""

    def test_empty_entries(self) -> None:
        engine = _stub_engine(skill_index=SimpleNamespace(
            get_entries=lambda: [],
            compact_index_text=lambda entries: "",
        ))
        assembler = PromptAssembler(engine)
        assert assembler._build_skill_section(include_skills=True) == ""

    def test_populated_skills(self) -> None:
        entries = [{"name": "deploy", "description": "Deploy app"}]
        engine = _stub_engine(skill_index=SimpleNamespace(
            get_entries=lambda: entries,
            compact_index_text=lambda ents: "- deploy: Deploy app",
        ))
        assembler = PromptAssembler(engine)
        result = assembler._build_skill_section(include_skills=True)
        assert "Learned Skills" in result
        assert "deploy" in result


# ── Tool category recording ──────────────────────────────────────────


class TestRecordToolCallCategories:
    def test_records_categories_from_manifest(self) -> None:
        manifest = SimpleNamespace(name="shell", category="execution", is_core=True)
        engine = _stub_engine(last_turn_tool_categories=frozenset())
        engine._manifests_by_name = {"shell": manifest}
        assembler = PromptAssembler(engine)
        call = SimpleNamespace(name="shell")
        assembler._record_tool_call_categories([call])
        assert "execution" in engine._last_turn_tool_categories

    def test_skips_system_and_general_categories(self) -> None:
        manifest = SimpleNamespace(name="internal", category="system", is_core=True)
        engine = _stub_engine(last_turn_tool_categories=frozenset())
        engine._manifests_by_name = {"internal": manifest}
        assembler = PromptAssembler(engine)
        call = SimpleNamespace(name="internal")
        assembler._record_tool_call_categories([call])
        assert len(engine._last_turn_tool_categories) == 0

    def test_accumulates_across_calls(self) -> None:
        m1 = SimpleNamespace(name="shell", category="execution", is_core=False)
        m2 = SimpleNamespace(name="read_file", category="filesystem", is_core=False)
        engine = _stub_engine(last_turn_tool_categories=frozenset({"execution"}))
        engine._manifests_by_name = {"shell": m1, "read_file": m2}
        assembler = PromptAssembler(engine)
        call = SimpleNamespace(name="read_file")
        assembler._record_tool_call_categories([call])
        assert "execution" in engine._last_turn_tool_categories
        assert "filesystem" in engine._last_turn_tool_categories


# ── Task scope keywords ──────────────────────────────────────────────


class TestTaskScopeKeywords:
    def test_includes_workspace_name(self) -> None:
        contract = TaskContract(
            task_id="turn-1",
            original_request="hello",
            workspace_root="/home/user/myproject",
            allowed_roots=("/home/user/myproject",),
        )
        engine = _stub_engine(task_contract=contract, workspace_root="/home/user/myproject")
        assembler = PromptAssembler(engine)
        kw = assembler._task_scope_keywords("find files")
        assert "myproject" in kw

    def test_deduplicates(self) -> None:
        engine = _stub_engine()
        assembler = PromptAssembler(engine)
        kw = assembler._task_scope_keywords("test test test")
        assert kw.count("test") == 1


# ── Auto-extract findings ────────────────────────────────────────────


class TestAutoExtractFindings:
    def test_extracts_from_tool_result(self) -> None:
        messages = [
            {
                "role": "tool",
                "content": "/path/to/file.py\n" + "x" * 500,
            }
        ]
        findings = PromptAssembler._auto_extract_findings(messages)
        assert len(findings) == 1
        assert "[auto-extracted]" in findings[0]

    def test_skips_short_content(self) -> None:
        messages = [{"role": "tool", "content": "short"}]
        assert PromptAssembler._auto_extract_findings(messages) == []

    def test_skips_non_tool_messages(self) -> None:
        messages = [{"role": "user", "content": "x" * 500}]
        assert PromptAssembler._auto_extract_findings(messages) == []

    def test_skips_error_json(self) -> None:
        payload = {"ok": False, "error": "something went wrong" + "x" * 400}
        messages = [{"role": "tool", "content": json.dumps(payload)}]
        assert PromptAssembler._auto_extract_findings(messages) == []


import json
