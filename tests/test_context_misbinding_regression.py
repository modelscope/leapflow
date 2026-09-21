# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from conftest import StubLLM, make_settings

from leapflow.engine.context.context_focus import ContextPlane, FocusEntity
from leapflow.engine.engine import AgentEngine
from leapflow.engine import build_default_registry
from leapflow.engine.intent_classifier import Intent
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.plugins import get_registry
_tool_reg = get_registry()

TOOL_DEFINITIONS = _tool_reg.tool_definitions


class _Classifier:
    async def classify(self, user_text: str) -> Intent:
        return Intent(label="complex", reason="test")


class _CaptureLLM(StubLLM):
    def __init__(self, replies=None) -> None:
        super().__init__(replies or ["ok"])
        self.calls: list[list[dict]] = []

    async def achat(self, messages, *, stream=True, enable_thinking=False, **kwargs):
        self.calls.append(list(messages))
        return await super().achat(
            messages,
            stream=stream,
            enable_thinking=enable_thinking,
            **kwargs,
        )


def _engine(tmp_path, *, llm=None):
    from leapflow.platform.mock import MockBridge

    settings = make_settings(str(tmp_path))
    rpc = MockBridge()
    llm = llm or StubLLM(["ok"])
    wm = WorkingMemoryProvider(max_tokens=2048)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    reg = build_default_registry(rpc, llm, wm, lt)
    return AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _Classifier()), lt


async def test_prompt_assembly_keeps_paper_focus_across_model_config_events(tmp_path) -> None:
    engine, lt = _engine(tmp_path)
    try:
        engine._focus_state.record_focus(FocusEntity(
            entity_id="paper:minicpm",
            kind="paper",
            canonical_name="MiniCPM-O 4.5 Technical Report",
            plane=ContextPlane.TASK_SEMANTIC,
            evidence_refs=("web_fetch:https://arxiv.org/abs/2601.21337",),
            first_turn=1,
            last_task_turn=1,
            last_mentioned_turn=1,
        ))
        engine._focus_state.record_tool_result(
            "config_set",
            {"key": "llm.model", "value": "qwen3.8-max"},
            {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
            turn_id=2,
        )

        assembly = await engine._prompt_assembler._assemble_unified_prompt(
            "上面的 paper 需要更深层次解读",
            tool_definitions=TOOL_DEFINITIONS,
            enable_thinking=False,
        )

        # Focus context is assembled into volatile_context (separated from
        # the byte-stable system prompt to preserve prefix-cache hits).
        ctx = assembly.volatile_context
        assert "## Semantic Focus Plane" in ctx
        assert "Current task focus" in ctx
        assert "MiniCPM-O 4.5 Technical Report" in ctx
        assert "llm.model -> qwen3.8-max" in ctx
        assert "Resolved user reference" in ctx
        assert engine._last_disclosure_metadata["reference_resolution"]["target_name"] == "MiniCPM-O 4.5 Technical Report"
    finally:
        lt.close()


async def test_full_turn_exposes_task_focus_not_control_model_to_provider(tmp_path) -> None:
    llm = _CaptureLLM(["MiniCPM deployment answer"])
    engine, lt = _engine(tmp_path, llm=llm)
    try:
        engine._focus_state.record_focus(FocusEntity(
            entity_id="paper:minicpm",
            kind="paper",
            canonical_name="MiniCPM-O 4.5 Technical Report",
            plane=ContextPlane.TASK_SEMANTIC,
            evidence_refs=("web_fetch:https://arxiv.org/abs/2601.21337",),
            first_turn=1,
            last_task_turn=1,
            last_mentioned_turn=1,
        ))
        engine._focus_state.record_tool_result(
            "config_set",
            {"key": "llm.model", "value": "qwen3.8-max"},
            {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
            turn_id=2,
        )

        answer = await engine.run("上面的 paper 需要更深层次解读，以及如何处理 mac 本地部署问题")
        provider_context = "\n".join(
            str(message.get("content") or "")
            for call in llm.calls
            for message in call
        )

        assert answer == "MiniCPM deployment answer"
        assert "## Semantic Focus Plane" in provider_context
        assert "MiniCPM-O 4.5 Technical Report" in provider_context
        assert "Recent control-plane events" in provider_context
        assert "llm.model -> qwen3.8-max" in provider_context
        assert "Use Current task focus for deictic task references" in provider_context
    finally:
        lt.close()


async def test_focus_block_survives_provider_message_preparation(tmp_path) -> None:
    engine, lt = _engine(tmp_path)
    try:
        engine._focus_state.record_focus(FocusEntity(
            entity_id="paper:minicpm",
            kind="paper",
            canonical_name="MiniCPM-O 4.5 Technical Report",
            plane=ContextPlane.TASK_SEMANTIC,
            evidence_refs=("web_fetch:https://arxiv.org/abs/2601.21337",),
            first_turn=1,
            last_task_turn=1,
            last_mentioned_turn=1,
        ))
        assembly = await engine._prompt_assembler._assemble_unified_prompt(
            "上面的 paper 需要更深层次解读",
            tool_definitions=TOOL_DEFINITIONS,
            enable_thinking=False,
        )
        messages = [
            build_system_message(assembly.system),
        ]
        # volatile_context carries semantic focus; the engine injects it as a
        # separate system message between the stable system prompt and history.
        if assembly.volatile_context:
            messages.append(build_system_message(assembly.volatile_context))
        messages += [
            build_user_message_text("older unrelated history " * 200),
            build_user_message_text("上面的 paper 需要更深层次解读"),
        ]

        prepared = engine._prompt_assembler._prepare_llm_messages(messages, tools=None)
        joined = "\n".join(str(msg.get("content") or "") for msg in prepared)

        assert "## Semantic Focus Plane" in joined
        assert "MiniCPM-O 4.5 Technical Report" in joined
    finally:
        lt.close()


async def test_prompt_assembly_resolves_to_task_entity_even_with_control_events(tmp_path) -> None:
    """When task entities exist, they always take priority over control events.

    The resolver no longer parses user text for keywords — it relies on
    structured focus state priority: task_semantic > control_plane.
    """
    engine, lt = _engine(tmp_path)
    try:
        engine._focus_state.record_focus(FocusEntity(
            entity_id="paper:minicpm",
            kind="paper",
            canonical_name="MiniCPM-O 4.5 Technical Report",
            plane=ContextPlane.TASK_SEMANTIC,
            first_turn=1,
            last_task_turn=1,
            last_mentioned_turn=1,
        ))
        engine._focus_state.record_tool_result(
            "config_set",
            {"key": "llm.model", "value": "qwen3.8-max"},
            {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
            turn_id=2,
        )

        await engine._prompt_assembler._assemble_unified_prompt(
            "刚才设置的默认模型是什么？",
            tool_definitions=TOOL_DEFINITIONS,
            enable_thinking=False,
        )

        resolution = engine._last_disclosure_metadata["reference_resolution"]
        # Task entity takes priority; control events are surfaced in prompt
        # context but not as the resolution target when task entities exist.
        assert resolution["plane"] == ContextPlane.TASK_SEMANTIC.value
        assert resolution["target_name"] == "MiniCPM-O 4.5 Technical Report"
    finally:
        lt.close()
