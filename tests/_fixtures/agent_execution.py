# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared helpers extracted from test_agent_execution.py during the P1-a split.

These helpers are used by multiple test files after the split, so they live here
to avoid cross-imports between test modules (which cause pytest collection order
issues).
"""

from __future__ import annotations

from leapflow.engine.intent_classifier import Intent


class _FixedClassifier:
    """Deterministic intent classifier for routing tests."""

    def __init__(self, label: str) -> None:
        self._intent = Intent(label=label, reason="test")

    async def classify(self, user_text: str) -> Intent:
        return self._intent


def _activate_desktop_plugin(monkeypatch) -> list:
    """Activate the global desktop_semantic plugin with recording fake tools.

    Mirrors the production wiring: cli/context.py calls
    registry.bind_runtime(perception=..., execution=...) and the engine reads
    schemas/handlers from the plugin. Returns the shared call log so tests can
    assert handler dispatch actually reached the semantic tools.
    """
    import leapflow.plugins.tool_plugins.desktop_semantic as ds
    from leapflow.plugins import get_registry

    calls: list = []

    def _fake_entries(adapter):
        async def _observe(params):
            calls.append(("observe_ui", dict(params)))
            return {"ok": True, "tree": "app:Browser"}

        async def _click(params):
            calls.append(("click", dict(params)))
            return {"ok": True, "clicked": params.get("selector")}

        return [
            ds.SemanticToolEntry(
                name="observe_ui",
                description="Observe the current UI state",
                parameters={"app": "string (optional) — application name"},
                handler=_observe,
            ),
            ds.SemanticToolEntry(
                name="click",
                description="Click a UI element",
                parameters={"selector": "string (required) — element selector"},
                handler=_click,
                mutates_state=True,
            ),
        ]

    monkeypatch.setattr(ds, "build_semantic_tool_entries", _fake_entries)
    get_registry().bind_runtime(perception=object(), execution=object())
    return calls


def _deactivate_desktop_plugin() -> None:
    from leapflow.plugins import get_registry

    get_registry().bind_runtime(perception=None, execution=None)


def _build_desktop_engine(td: str, llm=None, **settings_overrides):
    from conftest import StubLLM, make_settings
    from leapflow.engine._tool_helpers import build_default_registry
    from leapflow.engine.engine import AgentEngine
    from leapflow.memory import (
        EpisodicMemoryProvider,
        SemanticMemoryProvider,
        WorkingMemoryProvider,
    )
    from leapflow.platform.mock import MockBridge

    settings = make_settings(td)
    settings = settings.__class__(
        **{**settings.__dict__, "native_tool_calling_enabled": True, **settings_overrides}
    )
    rpc = MockBridge()
    llm = llm or StubLLM(["ok"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    reg = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(
        settings, rpc, llm, wm, lt, imm, reg,
        _FixedClassifier("chat"),
    )
    return engine, lt
