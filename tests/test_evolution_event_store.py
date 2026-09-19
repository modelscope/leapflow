# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contract tests for the append-only evolution evidence foundation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent
from leapflow.evolution.action_recorder import ActionRecorder, sanitize_evidence
from leapflow.evolution.artifact_store import ArtifactIntegrityError, ContentAddressedArtifactStore
from leapflow.evolution.outbox import (
    EvolutionEventOutbox,
    EvolutionOutboxClosed,
    EvolutionOutboxWriteError,
)
from leapflow.layout import ProfileLayout
from leapflow.storage.connection import LocalConnectionHolder
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore
from leapflow.storage.schema import CURRENT_SCHEMA_VERSION, ensure_schema


def _context(*, session: str = "s1", action: str = "a1") -> EvolutionContext:
    return EvolutionContext.create(
        profile_id="p1",
        workspace_id="w1",
        session_id=session,
        turn_id="t1",
        frame_id="f1",
        action_id=action,
    )


def _event(kind: str = "action.started", *, session: str = "s1", action: str = "a1"):
    return EvolutionEvent.create(
        kind,
        context=_context(session=session, action=action),
        payload={"ok": True},
        producer="test",
        dedup_key=f"{kind}:{session}:{action}",
    )


def test_context_mints_one_correlation_id_and_preserves_it_on_updates() -> None:
    context = _context()
    changed = context.with_ids(causation_id="evt-parent", proposal_id="prop-1")

    assert context.correlation_id.startswith("evo-")
    assert changed.correlation_id == context.correlation_id
    assert changed.causation_id == "evt-parent"
    assert changed.proposal_id == "prop-1"
    assert context.causation_id == ""


def test_event_payload_hash_and_dedup_are_deterministic() -> None:
    context = EvolutionContext(profile_id="p", session_id="s", correlation_id="c")
    first = EvolutionEvent.create(
        "environment.observed", context=context, payload={"b": 2, "a": 1}, producer="test"
    )
    second = EvolutionEvent.create(
        "environment.observed", context=context, payload={"a": 1, "b": 2}, producer="test"
    )

    assert first.payload_hash == second.payload_hash
    assert first.dedup_key == second.dedup_key
    assert first.event_id != second.event_id


def test_event_round_trip_preserves_context_and_payload() -> None:
    event = _event()
    restored = EvolutionEvent.from_dict(event.to_dict())
    assert restored == event


def test_store_appends_and_deduplicates(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "events.duckdb")
    store = DuckDBEvolutionEventStore(holder)
    event = _event()

    assert store.append(event) is True
    assert store.append(event) is False
    assert store.count(profile_id="p1") == 1
    assert store.latest_sequence(profile_id="p1") == 1
    records = store.read(profile_id="p1")
    assert [record.sequence for record in records] == [1]
    assert [record.event for record in records] == [event]
    holder.close()


def test_store_batches_atomically_and_keeps_causal_order(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "events.duckdb")
    store = DuckDBEvolutionEventStore(holder)
    events = [
        _event("action.started", action="a1"),
        _event("action.completed", action="a1"),
        _event("action.started", session="s2", action="a2"),
    ]

    assert store.append_many(events) == 3
    assert store.append_many(events) == 0
    assert [
        record.event.event_type
        for record in store.read(profile_id="p1", session_id="s1")
    ] == ["action.started", "action.completed"]
    assert store.latest_sequence(profile_id="p1") == 3
    holder.close()


def test_store_filters_by_correlation_and_type(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    first = _event("action.started")
    second = EvolutionEvent.create(
        "teacher.verdict_recorded",
        context=first.context,
        payload={"action": "absorb"},
        producer="teacher",
    )
    store.append_many([first, second])

    assert [
        record.event for record in store.read(correlation_id=first.context.correlation_id)
    ] == [first, second]
    assert [
        record.event for record in store.read(event_type="teacher.verdict_recorded")
    ] == [second]
    store.close()


def test_artifact_store_is_immutable_and_content_addressed(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    first = store.put_text("same", privacy_class="session")
    second = store.put_text("same", privacy_class="session")
    other = store.put_text("different", privacy_class="session")

    assert first.digest == second.digest
    assert first.digest != other.digest
    assert store.get_text(first) == "same"
    assert first.relative_path.startswith(first.digest[:2] + "/")


def test_artifact_store_rejects_paths_disguised_as_refs(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="SHA-256"):
        store.resolve("../../secret")


def test_artifact_store_rejects_corrupted_content(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    ref = store.put_text("trusted")
    store.resolve(ref).write_text("tampered")

    with pytest.raises(ArtifactIntegrityError, match="digest mismatch"):
        store.get_bytes(ref)
    with pytest.raises(ArtifactIntegrityError, match="digest mismatch"):
        store.put_text("trusted")


def test_profile_layout_owns_the_evolution_cas_path(tmp_path: Path) -> None:
    layout = ProfileLayout(root=tmp_path / "profile", profile_id="p1")
    assert layout.evolution_artifacts_dir == layout.root / "artifacts" / "sha256"


def test_sanitizer_redacts_secrets_and_is_structurally_bounded() -> None:
    value = {
        "api_key": "sk-secret-value",
        "nested": {"password": "hunter2", "safe": "visible"},
        "items": list(range(40)),
        "long": "x" * 1200,
    }
    sanitized = sanitize_evidence(value)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["password"] == "[REDACTED]"
    assert sanitized["nested"]["safe"] == "visible"
    assert sanitized["items"][-1] == {"items_omitted": 8}
    assert len(sanitized["long"]) == 1001


@pytest.mark.asyncio
async def test_action_recorder_emits_linked_start_and_completion(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    outbox = EvolutionEventOutbox(store, flush_interval_s=0.001)
    recorder = ActionRecorder(outbox, producer_version="test")
    context = _context()

    started = await recorder.started(
        context=context,
        action_type="tool",
        action_name="file.read",
        arguments={"file_path": "/tmp/x", "token": "secret"},
        execution_policy="read_only",
        critical=False,
    )
    await recorder.completed(
        context=context,
        started_event=started,
        action_type="tool",
        action_name="file.read",
        result={"ok": True, "content": "hello"},
        duration_ms=1.5,
    )
    await outbox.flush()

    events = [record.event for record in store.read(session_id="s1")]
    assert [event.event_type for event in events] == ["action.started", "action.completed"]
    assert events[0].payload["arguments"]["token"] == "[REDACTED]"
    assert events[1].context.causation_id == events[0].event_id
    assert events[1].payload["ok"] is True
    assert recorder.metrics.started_latency.count == 1
    assert recorder.metrics.completed_latency.count == 1
    assert outbox.metrics.publish_latency.count == 2
    assert outbox.metrics.write_latency.count >= 1
    await outbox.close()
    store.close()


@pytest.mark.asyncio
async def test_critical_event_is_durable_before_publish_returns(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    outbox = EvolutionEventOutbox(store, critical_timeout_s=1.0)
    event = _event()

    await outbox.publish(event, critical=True)

    assert store.count(profile_id="p1") == 1
    assert outbox.metrics.queued == 0
    await outbox.close()
    store.close()


@pytest.mark.asyncio
async def test_outbox_rejects_publication_after_close(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    outbox = EvolutionEventOutbox(store)
    await outbox.close()

    with pytest.raises(EvolutionOutboxClosed):
        await outbox.publish(_event())
    store.close()


def test_schema_contains_long_running_evolution_tables(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "schema.duckdb")
    DuckDBEvolutionEventStore(holder)
    tables = {row[0] for row in holder.connection.execute("SHOW TABLES").fetchall()}
    assert {
        "evolution_events",
        "evolution_teacher_jobs",
        "evolution_projections",
    } <= tables
    assert "evolution_proposal_work" not in tables
    versions = [
        row[0]
        for row in holder.connection.execute(
            "SELECT version FROM _schema_version ORDER BY version"
        ).fetchall()
    ]
    assert versions == list(range(1, CURRENT_SCHEMA_VERSION + 1))
    holder.close()


def test_schema_upgrades_a_v1_database_in_order(tmp_path: Path) -> None:
    connection = duckdb.connect(str(tmp_path / "old.duckdb"))
    connection.execute(
        "CREATE TABLE _schema_version (version INTEGER NOT NULL, applied_at DOUBLE NOT NULL)"
    )
    connection.execute("INSERT INTO _schema_version VALUES (1, 1.0)")

    assert ensure_schema(connection) == CURRENT_SCHEMA_VERSION
    assert [
        row[0]
        for row in connection.execute(
            "SELECT version FROM _schema_version ORDER BY version"
        ).fetchall()
    ] == list(range(1, CURRENT_SCHEMA_VERSION + 1))
    assert connection.execute("SELECT nextval('evolution_event_sequence')").fetchone() == (1,)
    connection.close()


def test_event_payload_is_deeply_immutable_and_hash_checked() -> None:
    source = {"nested": {"items": [1, 2]}}
    event = EvolutionEvent.create(
        "environment.observed",
        context=_context(),
        payload=source,
        producer="test",
    )
    source["nested"]["items"].append(3)

    assert event.to_dict()["payload"] == {"nested": {"items": [1, 2]}}
    with pytest.raises(TypeError):
        event.payload["new"] = True
    with pytest.raises(AttributeError):
        event.payload["nested"]["items"].append(3)
    with pytest.raises(ValueError, match="payload_hash"):
        EvolutionEvent(
            event_id="evt-bad",
            event_type="environment.observed",
            context=_context(),
            payload={"ok": True},
            occurred_at=1.0,
            producer="test",
            dedup_key="bad",
            payload_hash="incorrect",
        )


def test_database_sequence_is_unique_across_store_instances(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "events.duckdb")
    first = DuckDBEvolutionEventStore(holder)
    second = DuckDBEvolutionEventStore(holder)

    def append_one(index: int) -> None:
        store = first if index % 2 else second
        assert store.append(_event(action=f"parallel-{index}")) is True

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append_one, range(40)))

    records = first.read(profile_id="p1", limit=100)
    sequences = [record.sequence for record in records]
    assert len(sequences) == 40
    assert len(set(sequences)) == 40
    assert sequences == sorted(sequences)
    holder.close()


@pytest.mark.asyncio
async def test_outbox_saturation_uses_direct_fallback_without_loss() -> None:
    class _Store:
        def __init__(self) -> None:
            self.events = []

        def append(self, event: EvolutionEvent) -> bool:
            self.events.append(event)
            return True

        def append_many(self, events) -> int:
            self.events.extend(events)
            return len(events)

    store = _Store()
    outbox = EvolutionEventOutbox(store, max_events=1, flush_interval_s=0.001)

    await outbox.publish(_event(action="queued"))
    await outbox.publish(_event(action="fallback"))
    await outbox.flush()

    assert outbox.metrics.direct_fallbacks == 1
    assert outbox.metrics.published == 2
    await outbox.close()


@pytest.mark.asyncio
async def test_outbox_write_failure_is_bounded_and_visible() -> None:
    class _FailingStore:
        def append(self, event: EvolutionEvent) -> bool:
            raise OSError("disk unavailable")

        def append_many(self, events) -> int:
            raise OSError("disk unavailable")

    outbox = EvolutionEventOutbox(
        _FailingStore(),
        max_write_attempts=2,
        retry_backoff_s=0.001,
        write_timeout_s=0.05,
        flush_timeout_s=0.5,
    )
    await outbox.publish(_event())

    with pytest.raises(EvolutionOutboxWriteError, match="failed to persist"):
        await outbox.flush()
    assert outbox.metrics.failures == 1
    with pytest.raises(EvolutionOutboxWriteError):
        await outbox.close()
