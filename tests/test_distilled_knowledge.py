# Copyright (c) Alibaba, Inc. and its affiliates.
"""C1-C3: the distillation channel, from teacher's conclusion to student's context.

This is the cheapest way the system adapts, and until now the only edge from teacher to
student was a single bit -- which tools are visible. Everything the teacher concluded
about *why* the environment behaved as it did was graded, traced, and thrown away, so a
correct judgement bought nothing and the next session repeated the same mistake.

Retirement is tested as heavily as recording, because stale knowledge does not merely go
unused: the acting agent cannot tell a current fact from one that expired three upgrades
ago, so telling it "the send control is labelled Dispatch" after another rename is worse
than telling it nothing.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest
from leapflow.engine.engine import AgentEngine
from leapflow.learning.world_model_driver import WorldModelEvolutionDriver
from leapflow.storage.distilled_knowledge_store import (
    DistilledKnowledge,
    JsonDistilledKnowledgeStore,
)
from leapflow.world_model.trajectory_grader import TeacherVerdict


def _env(os_version: str = "15.0", extra: bool = False) -> EnvironmentFingerprint:
    names = sorted(c.name for c in Capability)
    caps = {Capability[names[0]]}
    if extra:
        caps.add(Capability[names[1]])
    return EnvironmentFingerprint.from_platform_manifest(
        PlatformManifest(PlatformID.DARWIN_15, os_version, frozenset(caps))
    )


def _verdict(action: str, capability: str, knowledge: str, **kw: Any) -> AdaptationVerdict:
    return AdaptationVerdict.create(action, capability, knowledge, **kw)


def _reader(store: Any, *, fingerprint: str = "", limit: int = 12) -> AgentEngine:
    """An engine with only what this context layer reads, bound directly.

    Bound rather than resolved, so the test exercises the rendering rather than the lazy
    lookup -- which has its own test below.
    """
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = store
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = fingerprint
    engine._settings = SimpleNamespace(distilled_knowledge_limit=limit)
    return engine


# ── recording ─────────────────────────────────────────────────────────────────


def test_a_verdict_becomes_a_durable_fact(tmp_path):
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    entry = store.record(
        _verdict("rebind", "chat.reply", "the send control is now labelled Dispatch"),
        environment=_env().to_dict(),
    )

    assert isinstance(entry, DistilledKnowledge)
    assert entry.capability == "chat.reply"
    assert entry.environment_id == _env().fingerprint_id

    # And it survives a fresh reader, which is the whole point of persisting it.
    assert JsonDistilledKnowledgeStore(tmp_path / "dk.json").count() == 1


def test_a_verdict_without_knowledge_is_refused(tmp_path):
    """The parser already rejects this shape, so reaching here means something changed."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    assert store.record(SimpleNamespace(capability="chat.reply", knowledge="")) is None
    assert store.record(SimpleNamespace(capability="", knowledge="something")) is None
    assert store.count() == 0


def test_a_batch_is_one_write_and_the_last_word_wins(tmp_path):
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    stored = store.record_all(
        [
            _verdict("absorb", "chat.reply", "first conclusion"),
            _verdict("rebind", "chat.reply", "second conclusion"),
            _verdict("absorb", "chat.react", "unrelated"),
        ]
    )

    assert len(stored) == 2, "one entry per capability, even within a batch"
    assert store.for_capability("chat.reply").knowledge == "second conclusion"


# ── retirement: three ways out, and only three ─────────────────────────────────


def test_a_newer_verdict_supersedes_the_older_one(tmp_path):
    """Keeping both live would show the agent a capability's contradictory past."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(_verdict("rebind", "chat.reply", "v3: the control is Dispatch"))
    store.record(_verdict("absorb", "chat.reply", "v4: the control is Send again"))

    assert store.count() == 1
    assert store.for_capability("chat.reply").knowledge.startswith("v4")


def test_knowledge_expires_and_expiry_is_applied_on_read(tmp_path):
    """A sweep would leave stale facts disclosed until some unrelated write happened."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json", ttl_seconds=1.0)
    store.record(_verdict("absorb", "mail.send", "the sent folder was renamed"))

    assert store.count() == 1
    assert store.live(now=time.time() + 5) == ()


def test_ttl_zero_disables_expiry(tmp_path):
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json", ttl_seconds=0.0)
    store.record(_verdict("absorb", "mail.send", "still true"))
    assert len(store.live(now=time.time() + 10_000_000)) == 1


def test_knowledge_can_be_retracted_when_it_is_obsolete(tmp_path):
    """The only retirement a caller drives, and the only one that covers recovery.

    A capability observed working again makes knowledge describing its failure
    misleading, and neither supersession nor expiry removes it -- no newer verdict is
    coming precisely because there is no longer anything wrong.
    """
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(_verdict("absorb", "chat.react", "the control is missing"))
    store.record(_verdict("absorb", "chat.reply", "unrelated"))

    assert store.retract("chat.react", reason="observed working") is True
    assert [e.capability for e in store.live()] == ["chat.reply"]
    assert store.retract("chat.react") is False, "retracting twice is a no-op"
    assert store.retract("") is False


# ── the environment is disclosed, never used to filter ─────────────────────────


def test_a_fact_from_another_environment_is_disclosed_as_such(tmp_path):
    """Filtering on fingerprint mismatch would discard good knowledge on any upgrade.

    Whether an OS point release invalidates "the send control is labelled Dispatch" is a
    judgement about meaning. A predicate here would be a hard rule with no ability to
    generalise, so the mismatch is surfaced and the reader weighs it.
    """
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(
        _verdict("rebind", "chat.reply", "the control is Dispatch"),
        environment=_env().to_dict(),
    )

    same = _reader(store, fingerprint=_env().fingerprint_id)._distilled_knowledge_context()
    other = _reader(
        store, fingerprint=_env("15.7", extra=True).fingerprint_id
    )._distilled_knowledge_context()

    assert "different environment" not in same
    assert "chat.reply" in other, "knowledge is not dropped for a changed environment"
    assert "learned in a different environment" in other


# ── the student's context ──────────────────────────────────────────────────────


def test_the_student_sees_distilled_knowledge_as_observations(tmp_path):
    """The C1 payload, rendered. Framed as observations because it is not an order."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(_verdict("rebind", "chat.reply", "the control is now Dispatch"))

    block = _reader(store)._distilled_knowledge_context()

    assert "What is known about this environment" in block
    assert "- chat.reply: the control is now Dispatch" in block
    assert "not instructions" in block, "it must not read as a command to the framework"


def test_the_disclosed_set_is_bounded(tmp_path):
    """The channel meant to improve context must not come to dominate it."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    for i in range(20):
        store.record(_verdict("absorb", f"cap{i:02d}.thing", f"fact {i}"))

    lines = [
        line
        for line in _reader(store, limit=3)._distilled_knowledge_context().splitlines()
        if line.startswith("- ")
    ]
    assert len(lines) == 3


def test_an_empty_or_absent_store_produces_no_block(tmp_path):
    assert _reader(JsonDistilledKnowledgeStore(tmp_path / "e.json"))._distilled_knowledge_context() == ""

    bare = AgentEngine.__new__(AgentEngine)
    bare._knowledge_store = None
    bare._knowledge_store_unavailable = False
    bare._environment_fingerprint_id = ""
    bare._settings = SimpleNamespace(distilled_knowledge_limit=12, profile_layout=None)
    assert bare._distilled_knowledge_context() == "", "no profile layout must degrade, not fail"


def test_a_corrupt_store_degrades_context_rather_than_failing_a_turn(tmp_path):
    """Distilled knowledge is an improvement to context, so absence costs quality only."""
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    store = JsonDistilledKnowledgeStore(path)

    assert store.count() == 0
    assert _reader(store)._distilled_knowledge_context() == ""


def test_the_reader_binds_itself_rather_than_depending_on_another_path(tmp_path):
    """The adaptive loop builds an equivalent store, but only when resolving a capability.

    Relying on it would make knowledge appear or vanish for reasons unrelated to
    knowledge -- so this layer resolves its own reader.
    """
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = None
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = ""
    engine._settings = SimpleNamespace(
        distilled_knowledge_limit=12,
        distilled_knowledge_ttl_s=0.0,
        workspace_root=str(tmp_path),
        profile_layout=SimpleNamespace(distilled_knowledge_path=tmp_path / "dk.json"),
    )

    store = engine._resolve_knowledge_store()
    assert store is not None
    assert engine._environment_fingerprint_id, "the current environment must be known"
    assert engine._resolve_knowledge_store() is store, "bound once, not per turn"


# ── the driver writes it, and writes it whatever else happened ──────────────────


class _Teacher:
    def __init__(self, verdicts) -> None:
        self._verdict = TeacherVerdict(grades=(), verdicts=tuple(verdicts))

    async def grade_and_propose(self, trajectory, goal="", **kwargs):
        return self._verdict


class _Intake:
    def observe_result(self, result, **kwargs):
        return None

    def requirements(self, *, min_count: int = 1, limit: int = 50):
        return ()


def test_a_session_that_only_absorbed_still_taught_the_next_one(tmp_path):
    """The cheapest adaptation, and the one that used to leave no trace at all."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher(
            [
                _verdict("absorb", "chat.react", "the control moved to the overflow menu"),
                _verdict("rebind", "chat.reply", "the app is v3", target="chat_reply_v3"),
            ]
        ),
        intake=_Intake(),
        knowledge_store=store,
    )

    result = asyncio.run(driver.drive([{"action": "a"}], "reply", environment=_env()))

    assert set(result.distilled) == {"chat.react", "chat.reply"}
    assert result.queued_proposal_ids == (), "no code was written"
    assert store.count() == 2
    assert result.to_dict()["distilled"] == list(result.distilled)
    # The environment travelled with it, so a later session can see where it came from.
    assert store.for_capability("chat.reply").environment_id == _env().fingerprint_id


def test_a_failing_store_does_not_fail_the_session(tmp_path):
    """Distillation improves the next session; it must never break this one."""

    class _Broken:
        def record_all(self, verdicts, **kwargs):
            raise OSError("disk full")

    driver = WorldModelEvolutionDriver(
        teacher=_Teacher([_verdict("absorb", "chat.react", "moved")]),
        intake=_Intake(),
        knowledge_store=_Broken(),
    )

    result = asyncio.run(driver.drive([{"action": "a"}], "reply"))
    assert result.distilled == ()
    assert len(result.verdicts) == 1, "the verdict is still reported"


def test_a_driver_without_a_store_still_grades(tmp_path):
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher([_verdict("absorb", "chat.react", "moved")]), intake=_Intake()
    )
    result = asyncio.run(driver.drive([{"action": "a"}], "reply"))
    assert result.distilled == () and len(result.verdicts) == 1


# ── review findings: the invariants that keep the whitelist and the set honest ─


def test_the_persisted_fields_and_the_record_agree_exactly(tmp_path):
    """Bound in both directions, because each direction has its own failure.

    A field the dataclass has and the whitelist lacks is the historical silent drop: the
    value is written nowhere and reads back as a default, with nothing raised. An entry in
    the whitelist with no matching field is the mirror image -- it claims to persist
    something that never existed, which is how a whitelist stops being trustworthy.
    ``"environment"`` was exactly that, left over from considering whether to store the
    whole fingerprint.
    """
    from leapflow.storage.distilled_knowledge_store import _ENTRY_FIELDS

    assert set(DistilledKnowledge("chat.reply", "k").to_dict()) == _ENTRY_FIELDS

    # And the binding has to hold through a real round trip, not just in the abstract.
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(
        _verdict("rebind", "chat.reply", "the control is Dispatch", target="v3"),
        environment=_env().to_dict(),
    )
    reloaded = JsonDistilledKnowledgeStore(tmp_path / "dk.json").for_capability("chat.reply")
    assert reloaded.target == "v3"
    assert reloaded.environment_id == _env().fingerprint_id
    assert reloaded.action == "rebind"


def test_supersession_bounds_the_store_by_the_capability_count(tmp_path):
    """The answer to "does this grow unbounded on the hot path": it cannot.

    One live entry per capability, so the ceiling is the number of capabilities the
    system has -- not the number of sessions it has run. Measured at 200 entries,
    ``live()`` costs single-digit microseconds, which is why no sweep or index is needed.
    """
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    for round_number in range(50):
        store.record(_verdict("absorb", "chat.reply", f"conclusion {round_number}"))

    assert store.count() == 1
    assert store.for_capability("chat.reply").knowledge == "conclusion 49"


def test_every_retry_owned_class_is_one_a_classifier_emits():
    """A member with no producer claims to filter something never seen.

    ``"rate_limit"`` was in the set with no producer anywhere in the engine. Harmless in
    effect, and corrosive in meaning: the set is supposed to read as a statement about
    which failures the retry layer owns, and an invented member makes it fiction.
    """
    import re
    from pathlib import Path as _Path

    from leapflow.learning.capability_observation import RETRY_OWNED_FAILURE_CLASSES

    engine_dir = _Path(__file__).resolve().parent.parent / "src" / "leapflow" / "engine"
    emitted = set()
    for path in engine_dir.rglob("*.py"):
        emitted.update(re.findall(r'"([a-z_]+)"', path.read_text(encoding="utf-8")))

    missing = RETRY_OWNED_FAILURE_CLASSES - emitted
    assert not missing, f"no classifier emits: {sorted(missing)}"


def test_an_unknown_action_still_discloses_its_recommendation(tmp_path):
    """A phrase table, so a fifth action loses nothing while its phrase is missing."""
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(
        SimpleNamespace(
            capability="chat.reply",
            knowledge="something changed",
            action="a_future_action",
            verdict_id="adv-x",
            confidence=0.5,
            rationale="",
            target="do the thing",
            created_at=1.0,
        )
    )

    block = _reader(store)._distilled_knowledge_context()
    assert "do the thing" in block, "an unmapped action must not drop its target"


def test_a_persistent_lookup_failure_costs_one_attempt_not_one_per_turn(tmp_path):
    """The context layer runs every turn, so a failing lookup must not retry every turn."""
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = None
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = ""
    engine._settings = SimpleNamespace(
        distilled_knowledge_limit=12,
        distilled_knowledge_ttl_s=0.0,
        workspace_root=str(tmp_path),
        profile_layout=SimpleNamespace(
            distilled_knowledge_path=property(lambda self: 1 / 0)  # raises on access
        ),
    )

    assert engine._resolve_knowledge_store() is None
    assert engine._knowledge_store_unavailable is True
    assert engine._distilled_knowledge_context() == ""
