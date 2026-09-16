# Copyright (c) Alibaba, Inc. and its affiliates.
"""The self-evolution switch: one setting, off by default, stated on the first screen.

The world model is not what this gates. It reviews every session, records what it learned
about the environment for the next one, and recommends which installed provider to prefer
-- none of which writes code or changes what the agent is able to do. Switching that off
would cost adaptation and reduce no risk, so it runs unconditionally.

What is gated is the single branch that writes code: an ``acquire`` verdict becoming a
queued proposal for a new plugin. Off by default because acquiring a capability is the most
expensive and least reversible decision the system makes, and because a user should choose
it rather than discover it after the fact.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from leapflow.learning.capability_observation import (
    DEFAULT_ACCEPTED_EVIDENCE,
    CapabilityEvidenceClassifier,
)


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {"evolution_enabled": False, "accepted_evidence_kinds": ()}
    base.update(overrides)
    return SimpleNamespace(**base)


# ── the switch itself ─────────────────────────────────────────────────────────


def test_self_evolution_is_off_by_default():
    """Acquiring a capability is the least reversible thing the system decides."""
    from leapflow.config import get_settings

    assert get_settings().evolution_enabled is False


def test_the_switch_is_discoverable_through_the_config_control_plane():
    """Every durable, user-writable setting must be reachable via ``leap config``.

    And this one gets its own category rather than being folded into Learning or Plugins:
    it is the switch a user is most likely to go looking for, so burying it among tuning
    knobs would make the most consequential setting the hardest to find.
    """
    from leapflow.config_service import _build_field_specs

    spec = _build_field_specs()["evolution.enabled"]
    assert spec.value_type is bool
    assert spec.category == "Self-Evolution"
    assert spec.description and "Off by default" in spec.description
    # The description has to say what it does *not* gate, or a user turning it off would
    # reasonably expect the world model to stop too.
    assert "world model runs either way" in spec.description


# ── configuration unification ─────────────────────────────────────────────────


def test_one_switch_admits_world_model_evidence():
    """Two knobs were one too many.

    ``accepted_evidence_kinds`` surfaces as ``accepted.evidence_kinds`` -- a section that
    names nothing -- so a user who turned self-evolution on and saw nothing happen had no
    way to guess that a second, differently-named setting also had to list
    ``world_model_intent``.
    """
    off = CapabilityEvidenceClassifier.from_settings(_settings())
    on = CapabilityEvidenceClassifier.from_settings(_settings(evolution_enabled=True))

    assert "world_model_intent" not in off.accepted
    assert "world_model_intent" in on.accepted


def test_enabling_the_switch_does_not_drop_the_trigger_already_working():
    """``from_kinds`` treats a non-empty tuple as the whole accepted set.

    So appending alone would have *removed* ``unknown_tool``: turning self-evolution on
    would have silently disabled the trigger that already worked, and the chain would have
    looked more capable while covering less.
    """
    for enabled in (False, True):
        accepted = CapabilityEvidenceClassifier.from_settings(
            _settings(evolution_enabled=enabled)
        ).accepted
        assert DEFAULT_ACCEPTED_EVIDENCE <= accepted, enabled


def test_the_finer_grained_tuple_still_widens_the_set():
    """Structural kinds come from an environment probe, not the world model.

    They stay opted into separately, so the switch is the common path and not a ceiling.
    """
    accepted = CapabilityEvidenceClassifier.from_settings(
        _settings(evolution_enabled=True, accepted_evidence_kinds=("interface_drift",))
    ).accepted

    assert {"interface_drift", "world_model_intent", "unknown_tool"} <= accepted


def test_the_tuple_alone_still_works_without_the_switch():
    """An operator who set only the tuple must not lose that behaviour."""
    accepted = CapabilityEvidenceClassifier.from_settings(
        _settings(accepted_evidence_kinds=("world_model_intent",))
    ).accepted
    assert "world_model_intent" in accepted


def test_a_settings_object_without_the_field_behaves_as_off():
    """Absence must read as off, not as an error: the daemon may predate the field."""
    accepted = CapabilityEvidenceClassifier.from_settings(
        SimpleNamespace(accepted_evidence_kinds=())
    ).accepted
    assert "world_model_intent" not in accepted


# ── stated on the first screen, in both directions ────────────────────────────


class _Console:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def print(self, text: str, **kwargs: Any) -> None:
        self.lines.append(("emphasis", text))

    def system(self, text: str) -> None:
        self.lines.append(("plain", text))


def test_the_mode_is_announced_when_it_is_off():
    """Silence would make the quiet default indistinguishable from a build without it.

    A user who cannot tell which they have cannot reason about either.
    """
    from leapflow.cli.commands.interactive import _announce_self_evolution

    console = _Console()
    _announce_self_evolution(console, _settings())

    kind, text = console.lines[0]
    assert kind == "plain", "the default is not a warning"
    assert "Self-evolution off" in text
    # And it must say how to change it, or "off" is a dead end.
    assert "leap config set evolution.enabled true" in text


def test_the_mode_is_emphasised_when_it_is_on():
    """The same class of fact as the approval-bypass notice, in the same place."""
    from leapflow.cli.commands.interactive import _announce_self_evolution

    console = _Console()
    _announce_self_evolution(console, _settings(evolution_enabled=True))

    kind, text = console.lines[0]
    assert kind == "emphasis"
    assert "Self-evolution on" in text
    # Being on is not being unguarded, and the line must not imply otherwise.
    assert "needs your approval" in text


def test_both_startup_paths_announce_it():
    """In-process and daemon-backed startup must not differ on a safety-relevant mode."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "src" / "leapflow" / "cli" / "commands" / "interactive.py"
    ).read_text(encoding="utf-8")

    assert source.count("_announce_self_evolution(console,") == 2
