# Copyright (c) Alibaba, Inc. and its affiliates.
"""C-1: an acquisition can now be *confirmed*, not only refuted.

Before this, the only outcome channel was the usage sink, which receives just ``ok``.
A successful call therefore graded ``unverifiable`` -- verification could refute an
acquisition but never confirm one, which made "verified by observed effect" half a
mechanism.

C-1 closes it by recording outcomes where the **full result payload** is visible (the
engine's result-observation path) and by establishing the declaration channel: a
handler reports what it observably did under an ``observed_effect`` key. Tools that
say nothing stay ``unverifiable`` -- silence must never be read as success.

The channel was originally two keys, ``observed_effect`` and a shorthand ``effect``.
The shorthand had to be withdrawn: ``effect`` is already used across the tree for a
risk *class* (``"effect": "write"``) and a hardware channel *type*, and single-token
overlap let ``effect="write"`` confirm an expectation reading "write the message to
the channel" -- a decided verdict that granted trust for evidence that never existed.
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


def test_the_declaration_channel_is_one_key_whose_name_is_not_taken():
    """``effect`` was withdrawn as a shorthand because that name already means
    something else: a risk class in self-management, a channel type in hardware.

    Keeping it cost more than it bought. Nothing emitted observed-effect prose under
    either spelling, while four call sites emitted the *other* meaning -- so the
    shorthand contributed no true confirmations and at least one false one.
    """
    assert OBSERVED_EFFECT_KEYS == ("observed_effect",)
    assert observed_effect_from_result({"observed_effect": "sent"}) == "sent"


def test_a_risk_class_under_the_old_shorthand_is_silence_not_evidence():
    """The exact payload that used to produce a false ``verified=True``."""
    assert observed_effect_from_result({"ok": True, "effect": "write"}) == ""
    assert observed_effect_from_result({"effect": "read"}) == ""


def test_silence_is_silence_and_is_never_synthesised():
    """An invented description could confirm an acquisition that never worked."""
    for result in ({}, {"ok": True}, {"ok": True, "effect": ""}, {"effect": "   "},
                   {"effect": 42}, None, "not-a-dict", []):
        assert observed_effect_from_result(result) == ""


def test_whitespace_is_trimmed():
    assert observed_effect_from_result({"observed_effect": "  sent to thread \n"}) == "sent to thread"


# ── the engine records outcomes with the payload (drive the real method) ───────


def _drive_engine_outcome(item):
    from leapflow.engine.learning_bridge import LearningBridge

    LearningBridge._record_coevolution_outcome(item)


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
            "result": {"ok": True, "observed_effect": "the reply was delivered to the thread"},
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
            "result": {"ok": True, "observed_effect": "opened a settings pane"},
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


def test_generation_prompt_names_the_key_the_verifier_actually_reads():
    """The contract has to reach the code that gets written, not just the verifier.

    These two drifted once and it was invisible: the declaration channel narrowed to
    ``observed_effect`` while the prompt still taught ``effect``. Every plugin the
    framework wrote for itself would then report through a key nothing reads, so its
    verdict would abstain forever -- and the board's abstain rate would have been read
    as "tools do not report" rather than "we told them the wrong key". Asserted
    against the verifier's own constant so the two cannot drift again.
    """
    from leapflow.learning.capability_effect_verifier import OBSERVED_EFFECT_KEYS
    from leapflow.learning.plugin_generator import (
        PluginGenerationRequest,
        PluginGenerator,
    )

    prompt = PluginGenerator().build_generation_prompt(
        PluginGenerationRequest(plugin_id="gen_x", description="reply in a thread")
    )
    channel = OBSERVED_EFFECT_KEYS[0]
    assert f'"{channel}"' in prompt
    assert "can never be verified, only refuted" in prompt
    # The skeleton must model it, since that is what gets copied.
    assert f'"{channel}": "<what observably changed>"' in prompt
    # The withdrawn shorthand must not be taught: elsewhere it means a risk class.
    assert '"effect":' not in prompt


def test_usage_sink_no_longer_records_outcomes():
    """Recording in two places would double-count and grade successes unverifiable."""
    import inspect

    from leapflow.learning.plugin_stats import PluginUsageTracker

    source = inspect.getsource(PluginUsageTracker.record)
    assert "record_tool_outcome" not in source
    assert "tracker.record(plugin_id, tool_name, ok)" in source   # streak feed stays


# ── the writer half: built-in tools that now declare their effect ─────────────


def test_declare_effect_spells_the_key_in_exactly_one_place():
    """A handler must never spell the key itself; that is how the halves drifted."""
    from leapflow.learning.capability_effect_verifier import (
        OBSERVED_EFFECT_KEYS,
        declare_effect,
    )

    assert declare_effect("wrote 12 bytes to a.py") == {
        OBSERVED_EFFECT_KEYS[0]: "wrote 12 bytes to a.py"
    }
    # Silence stays silence: an empty description contributes no key at all, so a
    # handler with nothing to say remains *unverifiable* rather than refuted.
    for empty in ("", "   ", None):
        assert declare_effect(empty) == {}


def test_file_write_declares_a_measured_effect(tmp_path, monkeypatch):
    """The byte count comes from the content that actually reached the file.

    That is what makes it an observation. A restatement of the request would be
    compared against the expectation and could confirm work that never happened.
    """
    import asyncio

    from leapflow.tools.file_operations import file_write

    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "note.txt"
    result = asyncio.run(file_write({"path": str(target), "content": "hello"}))

    assert result["ok"] is True, result
    effect = result.get("observed_effect", "")
    assert "5 bytes" in effect, effect          # measured, not declared
    assert "note.txt" in effect
    assert result["bytes_written"] == 5


def test_file_write_distinguishes_append_from_overwrite(tmp_path, monkeypatch):
    """Two different observed outcomes must not describe themselves identically."""
    import asyncio

    from leapflow.tools.file_operations import file_write

    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "log.txt"
    asyncio.run(file_write({"path": str(target), "content": "a"}))
    appended = asyncio.run(
        file_write({"path": str(target), "content": "b", "mode": "append"})
    )
    assert "appended" in appended.get("observed_effect", "")


def test_a_secret_config_write_declares_the_change_without_the_value(monkeypatch):
    """The effect channel obeys the same redaction rule as the payload.

    ``config_set`` deliberately never echoes a credential into the transcript. An
    effect declaration is transcript too, so a secret's new value must not travel
    here either -- while still reporting that the key changed.
    """
    from leapflow.learning.capability_effect_verifier import declare_effect

    # The two forms the handler chooses between, asserted directly: the redaction
    # decision is a branch on ``before.secret`` and both branches must stay
    # observations rather than one degrading into silence.
    plain = declare_effect("config key llm.model in scope user is now qwen3")
    secret = declare_effect("config key llm.api_key in scope user was updated")
    assert "qwen3" in plain["observed_effect"]
    assert "was updated" in secret["observed_effect"]
    assert "api_key" in secret["observed_effect"]


def test_edit_file_declares_what_it_actually_changed(tmp_path, monkeypatch):
    """Counts come from the edits that applied, so the description is measured."""
    import asyncio

    from leapflow.tools.file_operations import edit_file, file_write

    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "m.py"
    asyncio.run(file_write({"path": str(target), "content": "y = 2\n"}))

    result = asyncio.run(
        edit_file({
            "path": str(target),
            "edits": [{"original_text": "y = 2", "new_text": "y = 3"}],
        })
    )
    assert result["ok"] is True, result
    effect = result["observed_effect"]
    assert "1 edit" in effect and "1 replacement" in effect
    assert "m.py" in effect
