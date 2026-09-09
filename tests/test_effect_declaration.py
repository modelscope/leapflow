"""C-1: an acquisition can now be *confirmed*, not only refuted.

Before this, the only outcome channel was the usage sink, which receives just ``ok``.
A successful call therefore graded ``unverifiable`` -- verification could refute an
acquisition but never confirm one, which made "verified by observed effect" half a
mechanism.

C-1 closes it by recording outcomes where the **full result payload** is visible (the
engine's result-observation path) and by establishing the declaration channel: a
handler reports what it observably did under an ``effect`` key. Tools that say nothing
stay ``unverifiable`` -- silence must never be read as success.
"""

from __future__ import annotations

import asyncio

from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.evolution.observations import (
    CoevolutionObservations,
    install_observations,
)
from leapflow.evolution.sweep import CoevolutionSweep
from leapflow.learning.capability_effect_verifier import (
    OBSERVED_EFFECT_KEYS,
    EFFECT_UNREPORTED,
    VERIFIED,
    observed_effect_from_result,
)


def _requirement(expected: str = "the reply was delivered to the thread"):
    return EvolutionIntent.create(
        "chat.reply", "send path no-ops", expected_effect=expected
    ).to_requirement()


# ── the declaration channel ───────────────────────────────────────────────────


def test_both_accepted_spellings_are_read():
    assert OBSERVED_EFFECT_KEYS == ("observed_effect", "effect")
    assert observed_effect_from_result({"effect": "sent"}) == "sent"
    assert observed_effect_from_result({"observed_effect": "sent"}) == "sent"


def test_observed_effect_wins_when_both_are_present():
    result = {"observed_effect": "explicit", "effect": "shorthand"}
    assert observed_effect_from_result(result) == "explicit"


def test_silence_is_silence_and_is_never_synthesised():
    """An invented description could confirm an acquisition that never worked."""
    for result in ({}, {"ok": True}, {"ok": True, "effect": ""}, {"effect": "   "},
                   {"effect": 42}, None, "not-a-dict", []):
        assert observed_effect_from_result(result) == ""


def test_whitespace_is_trimmed():
    assert observed_effect_from_result({"effect": "  sent to thread \n"}) == "sent to thread"


# ── the engine records outcomes with the payload (drive the real method) ───────


def _drive_engine_outcome(item):
    from leapflow.engine.engine import AgentEngine

    AgentEngine._record_coevolution_outcome(item)


def _registry_with_tool(tool_name: str, plugin_id: str):
    """Install a real registry whose arbitration maps tool -> plugin."""
    from leapflow.plugins import registry as registry_module

    class _Reg:
        tool_owners = {tool_name: plugin_id}

    original = registry_module._REGISTRY if hasattr(registry_module, "_REGISTRY") else None
    return _Reg(), original


def test_engine_confirms_an_acquisition_from_the_declared_effect(monkeypatch):
    """The C-1 payoff: a successful call with a declared effect now verifies."""
    buf = CoevolutionObservations()
    install_observations(buf)
    reg, _ = _registry_with_tool("chat_reply", "gen_reply")
    monkeypatch.setattr("leapflow.plugins.get_registry", lambda: reg)
    try:
        buf.record_acquisition("gen_reply")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_reply")

        _drive_engine_outcome({
            "name": "chat_reply",
            "result": {"ok": True, "effect": "the reply was delivered to the thread"},
        })

        outcome = asyncio.run(CoevolutionSweep().run(
            verifications=buf.drain_verifications()
        ))
        assert outcome.verified == 1
        assert outcome.verdicts[0].reason == VERIFIED
    finally:
        install_observations(None)


def test_success_without_a_declared_effect_stays_unverifiable(monkeypatch):
    """Silence must not be promoted to confirmation."""
    buf = CoevolutionObservations()
    install_observations(buf)
    reg, _ = _registry_with_tool("chat_reply", "gen_reply")
    monkeypatch.setattr("leapflow.plugins.get_registry", lambda: reg)
    try:
        buf.record_acquisition("gen_reply")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_reply")

        _drive_engine_outcome({"name": "chat_reply", "result": {"ok": True}})

        outcome = asyncio.run(CoevolutionSweep().run(
            verifications=buf.drain_verifications()
        ))
        assert outcome.unverifiable == 1
        assert outcome.verdicts[0].reason == EFFECT_UNREPORTED
    finally:
        install_observations(None)


def test_engine_reads_error_payloads_as_failure(monkeypatch):
    buf = CoevolutionObservations()
    install_observations(buf)
    reg, _ = _registry_with_tool("chat_reply", "gen_reply")
    monkeypatch.setattr("leapflow.plugins.get_registry", lambda: reg)
    try:
        buf.record_acquisition("gen_reply")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_reply")

        _drive_engine_outcome({
            "name": "chat_reply", "result": {"ok": True, "error": "transport refused"},
        })

        outcome = asyncio.run(CoevolutionSweep().run(
            verifications=buf.drain_verifications()
        ))
        assert outcome.refuted == 1
    finally:
        install_observations(None)


def test_wrong_effect_refutes_even_with_ok_true(monkeypatch):
    """A structurally perfect adapter targeting the wrong thing must not pass."""
    buf = CoevolutionObservations()
    install_observations(buf)
    reg, _ = _registry_with_tool("chat_reply", "gen_reply")
    monkeypatch.setattr("leapflow.plugins.get_registry", lambda: reg)
    try:
        buf.record_acquisition("gen_reply")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_reply")

        _drive_engine_outcome({
            "name": "chat_reply",
            "result": {"ok": True, "effect": "opened a settings pane"},
        })

        outcome = asyncio.run(CoevolutionSweep().run(
            verifications=buf.drain_verifications()
        ))
        assert outcome.refuted == 1
    finally:
        install_observations(None)


def test_unacquired_plugins_are_not_paired(monkeypatch):
    """Ordinary traffic must leave the verification buffer untouched."""
    buf = CoevolutionObservations()
    install_observations(buf)
    reg, _ = _registry_with_tool("list_dir", "builtin_fs")
    monkeypatch.setattr("leapflow.plugins.get_registry", lambda: reg)
    try:
        for _ in range(20):
            _drive_engine_outcome({
                "name": "list_dir", "result": {"ok": True, "effect": "listed 3 files"},
            })
        assert buf.drain_verifications() == ()
    finally:
        install_observations(None)


def test_malformed_items_and_unowned_tools_are_ignored(monkeypatch):
    buf = CoevolutionObservations()
    install_observations(buf)
    reg, _ = _registry_with_tool("known", "p")
    monkeypatch.setattr("leapflow.plugins.get_registry", lambda: reg)
    try:
        for item in (None, "str", [], {}, {"name": ""}, {"name": "orphan"}):
            _drive_engine_outcome(item)            # must not raise
        assert buf.drain_verifications() == ()
    finally:
        install_observations(None)


def test_recording_survives_a_broken_registry(monkeypatch):
    """Observation must never affect execution."""
    def _boom():
        raise RuntimeError("registry down")

    monkeypatch.setattr("leapflow.plugins.get_registry", _boom)
    _drive_engine_outcome({"name": "t", "result": {"ok": True}})   # must not raise


# ── the generator asks for it, or nothing will ever be confirmable ────────────


def test_generation_prompt_requires_handlers_to_report_their_effect():
    """The contract has to reach the code that gets written, not just the verifier."""
    from leapflow.learning.plugin_generator import (
        PluginGenerationRequest,
        PluginGenerator,
    )

    prompt = PluginGenerator().build_generation_prompt(
        PluginGenerationRequest(plugin_id="gen_x", description="reply in a thread")
    )
    assert '"effect"' in prompt
    assert "can never be verified, only refuted" in prompt
    # The skeleton must model it, since that is what gets copied.
    assert '"effect": "<what observably changed>"' in prompt


def test_usage_sink_no_longer_records_outcomes():
    """Recording in two places would double-count and grade successes unverifiable."""
    import inspect

    from leapflow.learning.plugin_stats import PluginUsageTracker

    source = inspect.getsource(PluginUsageTracker.record)
    assert "record_tool_outcome" not in source
    assert "tracker.record(plugin_id, tool_name, ok)" in source   # streak feed stays
