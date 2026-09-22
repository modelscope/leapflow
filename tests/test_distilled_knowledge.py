# Copyright (c) Alibaba, Inc. and its affiliates.
"""C1: the distillation channel, from the teacher's committed verdict to the student.

This is the cheapest way the system adapts. The durable teacher worker commits a
``TEACHER_VERDICT_RECORDED`` fact per verdict; ``EvolutionDistilledKnowledgeStore``
projects those facts into the read model the acting agent consults. There is no second
durable store — the projection is the single source, rebuilt from events.

Retirement is tested as heavily as recording, because stale knowledge does not merely go
unused: the acting agent cannot tell a current fact from one that expired three upgrades
ago, so telling it "the send control is labelled Dispatch" after another rename is worse
than telling it nothing.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent
from leapflow.engine.engine import AgentEngine
from leapflow.engine.prompt_assembler import PromptAssembler
from leapflow.storage.distilled_knowledge_store import (
    DistilledKnowledge,
    EvolutionDistilledKnowledgeStore,
)
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore


def _verdict(action: str, capability: str, knowledge: str, **kw: Any) -> AdaptationVerdict:
    return AdaptationVerdict.create(action, capability, knowledge, **kw)


def _events(tmp_path) -> DuckDBEvolutionEventStore:
    return DuckDBEvolutionEventStore(tmp_path / "events.duckdb")


def _seed(store: DuckDBEvolutionEventStore, verdict: AdaptationVerdict, *, seq: int = 0) -> None:
    """Commit one verdict as a durable fact, exactly as the teacher worker would."""
    payload = verdict.to_dict()
    store.append(
        EvolutionEvent.create(
            EvolutionEventType.TEACHER_VERDICT_RECORDED,
            context=EvolutionContext(profile_id="p", decision_id=verdict.verdict_id),
            payload=payload,
            producer="test",
            dedup_key=f"teacher.verdict_recorded:{verdict.verdict_id}:{seq}",
        )
    )


def _knowledge(tmp_path, *verdicts: AdaptationVerdict, ttl_seconds: float = 0.0):
    events = _events(tmp_path)
    for index, verdict in enumerate(verdicts):
        _seed(events, verdict, seq=index)
    store = EvolutionDistilledKnowledgeStore(events, profile_id="p", ttl_seconds=ttl_seconds)
    store.refresh()
    return events, store


def _reader(store: Any, *, fingerprint: str = "", limit: int = 12) -> AgentEngine:
    """An engine with only what this context layer reads, bound directly."""
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = store
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = fingerprint
    engine._settings = SimpleNamespace(distilled_knowledge_limit=limit)
    engine._prompt_assembler = PromptAssembler(engine)
    return engine


# ── recording: a committed verdict becomes durable, replayable knowledge ────────


def test_a_committed_verdict_becomes_durable_knowledge(tmp_path):
    events, store = _knowledge(
        tmp_path,
        _verdict("rebind", "chat.reply", "the send control is now labelled Dispatch",
                 target="chat_v2"),
    )

    entry = store.for_capability("chat.reply")
    assert isinstance(entry, DistilledKnowledge)
    assert entry.capability == "chat.reply"
    assert store.count() == 1

    # A fresh projection over the same events reconstructs the entry: the events are
    # the source, not any in-memory state.
    replayed = EvolutionDistilledKnowledgeStore(events, profile_id="p")
    replayed.refresh()
    assert replayed.count() == 1
    events.close()


def test_all_four_actions_are_disclosed_as_knowledge(tmp_path):
    """Every verdict carries mandatory knowledge; the student sees all four actions."""
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("absorb", "chat.react", "the control moved to the overflow menu"),
        _verdict("rebind", "chat.reply", "the app is v3", target="chat_reply_v3"),
        _verdict("acquire", "mail.send", "nothing installed sends mail"),
        _verdict("escalate", "drive.upload", "a person must grant the drive scope",
                 target="grant drive.file"),
    )

    assert store.count() == 4
    _events_store.close()


# ── retirement: three ways out, and only three ─────────────────────────────────


def test_a_newer_verdict_supersedes_the_older_one(tmp_path):
    """Keeping both live would show the agent a capability's contradictory past."""
    events, store = _knowledge(
        tmp_path,
        _verdict("rebind", "chat.reply", "v3: the control is Dispatch"),
    )
    later = AdaptationVerdict.create(
        "absorb", "chat.reply", "v4: the control is Send again",
        created_at=time.time() + 10,
    )
    _seed(events, later, seq=1)
    store.refresh()

    assert store.count() == 1
    assert store.for_capability("chat.reply").knowledge.startswith("v4")
    events.close()


def test_knowledge_expires_and_expiry_is_applied_on_read(tmp_path):
    """A sweep would leave stale facts disclosed until some unrelated write happened."""
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("absorb", "mail.send", "the sent folder was renamed"),
        ttl_seconds=1.0,
    )

    assert store.count() == 1
    assert store.live(now=time.time() + 5) == ()
    _events_store.close()


def test_ttl_zero_disables_expiry(tmp_path):
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("absorb", "mail.send", "still true"),
        ttl_seconds=0.0,
    )
    assert len(store.live(now=time.time() + 10_000_000)) == 1
    _events_store.close()


def test_knowledge_can_be_retracted_when_it_is_obsolete(tmp_path):
    """The only retirement a caller drives, and the only one that covers recovery."""
    events, store = _knowledge(
        tmp_path,
        _verdict("absorb", "chat.react", "the control is missing"),
        _verdict("absorb", "chat.reply", "unrelated"),
    )

    assert store.retract("chat.react", reason="observed working") is True
    assert [e.capability for e in store.live()] == ["chat.reply"]
    assert store.retract("chat.react") is False, "retracting twice is a no-op"
    assert store.retract("") is False
    events.close()


# ── the student's context ──────────────────────────────────────────────────────


def test_the_student_sees_distilled_knowledge_as_observations(tmp_path):
    """The C1 payload, rendered. Framed as observations because it is not an order."""
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("rebind", "chat.reply", "the control is now Dispatch", target="chat_v2"),
    )

    block = _reader(store)._prompt_assembler._distilled_knowledge_context()

    assert "What is known about this environment" in block
    assert "- chat.reply: the control is now Dispatch" in block
    assert "not instructions" in block, "it must not read as a command to the framework"
    _events_store.close()


def test_a_rebind_target_tells_the_student_what_to_prefer(tmp_path):
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("rebind", "chat.reply", "the app is now v3", target="chat_reply_v3"),
    )
    block = _reader(store)._prompt_assembler._distilled_knowledge_context()
    assert "Prefer chat_reply_v3." in block
    _events_store.close()


def test_an_escalation_target_names_what_a_person_must_do(tmp_path):
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("escalate", "drive.upload", "the scope was revoked",
                 target="grant the drive.file scope"),
    )
    block = _reader(store)._prompt_assembler._distilled_knowledge_context()
    assert "This needs a person to: grant the drive.file scope." in block
    _events_store.close()


def test_a_verdict_without_a_target_adds_no_hint(tmp_path):
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("absorb", "chat.react", "it moved"),
    )
    block = _reader(store)._prompt_assembler._distilled_knowledge_context()
    assert "- chat.react: it moved" in block
    assert "Prefer" not in block and "needs a person" not in block
    _events_store.close()


def test_the_disclosed_set_is_bounded(tmp_path):
    """The channel meant to improve context must not come to dominate it."""
    verdicts = [_verdict("absorb", f"cap{i:02d}.thing", f"fact {i}") for i in range(20)]
    _events_store, store = _knowledge(tmp_path, *verdicts)

    lines = [
        line
        for line in _reader(store, limit=3)._prompt_assembler._distilled_knowledge_context().splitlines()
        if line.startswith("- ")
    ]
    assert len(lines) == 3
    _events_store.close()


def test_an_empty_or_absent_store_produces_no_block(tmp_path):
    _events_store, store = _knowledge(tmp_path)
    assert _reader(store)._prompt_assembler._distilled_knowledge_context() == ""

    bare = AgentEngine.__new__(AgentEngine)
    bare._knowledge_store = None
    bare._knowledge_store_unavailable = True
    bare._environment_fingerprint_id = ""
    bare._settings = SimpleNamespace(distilled_knowledge_limit=12)
    bare._prompt_assembler = PromptAssembler(bare)
    assert bare._prompt_assembler._distilled_knowledge_context() == "", "no store must degrade, not fail"
    _events_store.close()


def test_the_reader_uses_the_injected_event_projection(tmp_path):
    """The hot path reads a shared projection and never opens a parallel JSON store."""
    _events_store, store = _knowledge(
        tmp_path,
        _verdict("rebind", "chat.reply", "the control is Dispatch", target="chat_v2"),
    )
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = None
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = ""
    engine._settings = SimpleNamespace(
        distilled_knowledge_limit=12,
        workspace_root=str(tmp_path),
    )

    engine.set_distilled_knowledge_store(store)
    engine._prompt_assembler = PromptAssembler(engine)

    assert engine._prompt_assembler._resolve_knowledge_store() is store
    assert "chat.reply" in engine._prompt_assembler._distilled_knowledge_context()
    assert engine._prompt_assembler._rebind_preferences() == (("chat.reply", "chat_v2"),)
    _events_store.close()


def test_an_unknown_action_still_discloses_its_recommendation(tmp_path):
    """A phrase table, so a fifth action loses nothing while its phrase is missing."""
    events = _events(tmp_path)
    events.append(
        EvolutionEvent.create(
            EvolutionEventType.TEACHER_VERDICT_RECORDED,
            context=EvolutionContext(profile_id="p", decision_id="adv-x"),
            payload={
                "capability": "chat.reply",
                "knowledge": "something changed",
                "action": "a_future_action",
                "verdict_id": "adv-x",
                "confidence": 0.5,
                "target": "do the thing",
                "created_at": 1.0,
            },
            producer="test",
            dedup_key="teacher.verdict_recorded:adv-x",
        )
    )
    store = EvolutionDistilledKnowledgeStore(events, profile_id="p")
    store.refresh()

    block = _reader(store)._prompt_assembler._distilled_knowledge_context()
    assert "do the thing" in block, "an unmapped action must not drop its target"
    events.close()
