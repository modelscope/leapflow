# Copyright (c) Alibaba, Inc. and its affiliates.
"""Additive owner-only candidate/generation transactions on the shared instrument DB.

This store supplies atomic persistence, not qualification or authority. The owner
must recompute method-specific admission and reserve resources before committing.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import digest_of
from leapflow.storage.connection import ConnectionHolder

SCHEMA = "leapflow.calibration_generation.v1"
RECORD_DECISIONS = frozenset({"candidate", "rejected", "no_op"})
POINTER_DECISIONS = frozenset({"activated", "rolled_back", "invalidated"})


def _json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("calibration record must be a mapping")
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 32 or count > 65536:
            raise ValueError("bounded calibration record structure required")
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("calibration JSON keys must be strings at every depth")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
    digest_of(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 2 * 1024 * 1024:
        raise ValueError("derived calibration record exceeds persistence bound")
    return encoded


@dataclass(frozen=True)
class ModelGeneration:
    workcell_id: str
    generation: int
    model_digest: str
    dependency_digest: str
    model_json: str
    previous_generation: int | None
    request_id: str

    @property
    def model(self) -> dict[str, Any]:
        value = json.loads(self.model_json)
        if digest_of(value) != self.model_digest:
            raise ValueError("committed model generation digest mismatch")
        return value

    def basis(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "generation": self.generation, "model_digest": self.model_digest, "dependency_digest": self.dependency_digest}


@dataclass(frozen=True)
class GenerationState:
    """Atomic stop-use/active view; persisted consistency never establishes qualification."""

    workcell_id: str
    generation: ModelGeneration | None = None
    invalidation_json: str = ""
    stopped_generation: ModelGeneration | None = None

    @property
    def invalidation(self) -> dict[str, Any] | None:
        return json.loads(self.invalidation_json) if self.invalidation_json else None

    def basis(self) -> dict[str, Any]:
        basis = self.generation.basis() if self.generation is not None else {
            "schema": SCHEMA, "generation": None, "model_digest": None, "dependency_digest": None,
        }
        if self.invalidation_json:
            basis["invalidation_digest"] = self.invalidation["event_id"]
        return basis


class GenerationStore:
    """Shared-holder transactions; no competing instrument connection, worker, or lease."""
    def __init__(self, holder: ConnectionHolder, *, cache_manager: Any = None) -> None:
        if not isinstance(holder, ConnectionHolder):
            raise ValueError("generation store requires registry-owned ConnectionHolder")
        self._holder = holder
        self._lock = threading.RLock()
        self._ready = False
        self._cache = cache_manager
        self._registered_cache: Any = None

    def bind_cache(self, cache_manager: Any) -> None:
        """Late registry cache binding does not create a second database owner."""
        # An event-loop binding never waits on a transaction's thread lock. The
        # schema path registers the exact captured cache object, so a concurrent
        # rebind is recognized on the next operation rather than marked registered.
        self._cache = cache_manager

    def _schema(self, connection: Any) -> None:
        cache = self._cache
        if cache is not None and self._registered_cache is not cache:
            cache.register(path=self._holder.db_path, scope="profile", category="calibration",
                           source=self._holder.db_path.name, sensitive=True, syncable=False,
                           owner_component="hardware")
            self._registered_cache = cache
        if self._ready:
            return
        self._has_state(connection)
        statements = (
            "CREATE TABLE IF NOT EXISTS calib_candidates (candidate_digest VARCHAR PRIMARY KEY, workcell_id VARCHAR NOT NULL, evidence_digest VARCHAR NOT NULL, candidate_json VARCHAR NOT NULL, created_at DOUBLE NOT NULL)",
            "CREATE TABLE IF NOT EXISTS calib_assessments (assessment_digest VARCHAR PRIMARY KEY, candidate_digest VARCHAR NOT NULL, assessment_json VARCHAR NOT NULL, created_at DOUBLE NOT NULL)",
            "CREATE TABLE IF NOT EXISTS calib_generations (workcell_id VARCHAR, generation BIGINT, model_digest VARCHAR NOT NULL, dependency_digest VARCHAR NOT NULL, model_json VARCHAR NOT NULL, previous_generation BIGINT, request_id VARCHAR NOT NULL, PRIMARY KEY(workcell_id,generation), UNIQUE(workcell_id,request_id))",
            "CREATE TABLE IF NOT EXISTS calib_active (workcell_id VARCHAR PRIMARY KEY, generation BIGINT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS calib_blocks (workcell_id VARCHAR, generation BIGINT, block_id VARCHAR, record_json VARCHAR NOT NULL, PRIMARY KEY(workcell_id,generation,block_id))",
            "CREATE TABLE IF NOT EXISTS calib_decisions (event_id VARCHAR PRIMARY KEY, workcell_id VARCHAR NOT NULL, request_id VARCHAR NOT NULL, kind VARCHAR NOT NULL, decision_json VARCHAR NOT NULL, created_at DOUBLE NOT NULL, UNIQUE(workcell_id,request_id))",
            "CREATE TABLE IF NOT EXISTS calib_outbox (event_id VARCHAR PRIMARY KEY, event_json VARCHAR NOT NULL, delivered BOOLEAN NOT NULL DEFAULT FALSE)",
            "CREATE TABLE IF NOT EXISTS calib_pointer_journal (workcell_id VARCHAR, revision BIGINT, event_id VARCHAR NOT NULL UNIQUE, PRIMARY KEY(workcell_id,revision))",
        )
        legacy = not connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_pointer_journal'"
        ).fetchone()[0]
        connection.execute("BEGIN TRANSACTION")
        try:
            for statement in statements:
                connection.execute(statement)
            self._migrate_pointer_journal(connection, legacy=legacy)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        self._ready = True

    @staticmethod
    def _migrate_pointer_journal(connection: Any, *, legacy: bool) -> None:
        if legacy:
            # Legacy pointer decisions have only a wall timestamp. Refuse ambiguous
            # ordering rather than guessing which physical model was last committed.
            ambiguous = connection.execute(
                "SELECT 1 FROM calib_decisions d WHERE kind IN ('activated','rolled_back','invalidated') "
                "AND NOT EXISTS (SELECT 1 FROM calib_pointer_journal j WHERE j.workcell_id=d.workcell_id) "
                "GROUP BY workcell_id,created_at HAVING COUNT(*)>1 LIMIT 1"
            ).fetchone()
            if ambiguous:
                raise ValueError("legacy physical pointer decision ordering is ambiguous")
            connection.execute(
                "INSERT INTO calib_pointer_journal SELECT d.workcell_id, "
                "ROW_NUMBER() OVER (PARTITION BY d.workcell_id ORDER BY d.created_at,d.event_id),d.event_id "
                "FROM calib_decisions d WHERE kind IN ('activated','rolled_back','invalidated') "
                "AND NOT EXISTS (SELECT 1 FROM calib_pointer_journal j WHERE j.workcell_id=d.workcell_id)"
            )
        # An existing but truncated journal is corruption, never legacy state to
        # reconstruct by wall clock. Validate it before permitting any new record.
        workcells = connection.execute(
            "SELECT workcell_id FROM calib_decisions UNION SELECT workcell_id FROM calib_pointer_journal"
        ).fetchall()
        for (workcell_id,) in workcells:
            GenerationStore._pointer_decision(connection, workcell_id)

    def record_candidate(self, *, workcell_id: str, evidence_digest: str, candidate: Mapping[str, Any]) -> str:
        if not workcell_id or not evidence_digest:
            raise ValueError("candidate needs workcell and immutable evidence provenance")
        encoded = _json(candidate)
        identity = digest_of(json.loads(encoded))
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            row = connection.execute("SELECT workcell_id,evidence_digest,candidate_json FROM calib_candidates WHERE candidate_digest=?", [identity]).fetchone()
            if row is not None and row != (workcell_id, evidence_digest, encoded):
                raise ValueError("candidate digest rebound to different evidence/workcell")
            connection.execute("INSERT INTO calib_candidates VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING", [identity, workcell_id, evidence_digest, encoded, time.time()])
        return identity

    def candidate(self, identity: str, *, workcell_id: str) -> dict[str, Any]:
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            row = connection.execute("SELECT candidate_json FROM calib_candidates WHERE candidate_digest=? AND workcell_id=?", [identity, workcell_id]).fetchone()
            if row is None:
                raise ValueError("immutable candidate not indexed for this workcell")
            value = json.loads(row[0])
            if digest_of(value) != identity:
                raise ValueError("indexed candidate digest mismatch")
            return value

    def record_assessment(self, candidate_digest: str, assessment: Mapping[str, Any]) -> str:
        encoded = _json(assessment)
        identity = digest_of({"candidate_digest": candidate_digest, "assessment": json.loads(encoded)})
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            if connection.execute("SELECT 1 FROM calib_candidates WHERE candidate_digest=?", [candidate_digest]).fetchone() is None:
                raise ValueError("assessment has no immutable indexed candidate")
            connection.execute("INSERT INTO calib_assessments VALUES (?,?,?,?) ON CONFLICT DO NOTHING", [identity, candidate_digest, encoded, time.time()])
        return identity

    @staticmethod
    def _generation(connection: Any, workcell_id: str, generation: int) -> ModelGeneration:
        row = connection.execute("SELECT workcell_id,generation,model_digest,dependency_digest,model_json,previous_generation,request_id FROM calib_generations WHERE workcell_id=? AND generation=?", [workcell_id, generation]).fetchone()
        if row is None:
            raise ValueError("active pointer references a missing committed generation")
        result = ModelGeneration(*row)
        model = result.model
        decision_row = connection.execute(
            "SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?",
            [workcell_id, result.request_id],
        ).fetchone()
        if decision_row is None:
            raise ValueError("committed generation lacks durable activation causality")
        decision = GenerationStore._read_decision(decision_row[0])
        if (
            decision["kind"] != "activated" or decision.get("generation") != generation
            or decision["workcell_id"] != workcell_id or decision["request_id"] != result.request_id
            or decision.get("model_digest") != result.model_digest
            or decision.get("dependency_digest") != result.dependency_digest
            or decision.get("previous_generation") != result.previous_generation
        ):
            raise ValueError("committed generation differs from activation causality")
        blocks = {identity: json.loads(content) for identity, content in connection.execute(
            "SELECT block_id,record_json FROM calib_blocks WHERE workcell_id=? AND generation=?",
            [workcell_id, generation],
        ).fetchall()}
        fingerprint = digest_of({"model": model, "blocks": blocks, "dependencies": result.dependency_digest,
                                 "expected": result.previous_generation, "causal": decision["causal"]})
        if not blocks or fingerprint != decision.get("fingerprint"):
            raise ValueError("committed blocks or causal fingerprint mismatch")
        return result

    @staticmethod
    def _read_decision(encoded: str) -> dict[str, Any]:
        value = json.loads(encoded)
        if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("event_class") != "record":
            raise ValueError("durable calibration decision schema/classification mismatch")
        if (
            value.get("kind") not in RECORD_DECISIONS | POINTER_DECISIONS
            or not all(isinstance(value.get(key), str) and value[key] for key in ("workcell_id", "request_id"))
            or not isinstance(value.get("causal"), dict)
        ):
            raise ValueError("durable calibration decision identity/kind/causality mismatch")
        if value["kind"] in POINTER_DECISIONS:
            generation = value.get("generation")
            previous = value.get("previous_generation")
            if (
                (value["kind"] == "invalidated" and generation is not None)
                or (value["kind"] != "invalidated" and (type(generation) is not int or generation < 1))
                or (previous is not None and (type(previous) is not int or previous < 1))
                or (value["kind"] != "activated" and previous is None)
                or not isinstance(value.get("fingerprint"), str) or not value["fingerprint"]
            ):
                raise ValueError("durable physical pointer decision generation mismatch")
        core = {key: item for key, item in value.items() if key not in ("event_id", "event_class")}
        if value.get("event_id") != digest_of(core):
            raise ValueError("durable calibration decision digest mismatch")
        return value

    def snapshot(self, workcell_id: str) -> ModelGeneration | None:
        """Read a committed snapshot without creating schema or refreshing evidence.

        Plans use this off-loop. A corrupt/missing committed pointer fails closed;
        absence of any generation tables denotes the unchanged nominal model.
        """
        return self.state(workcell_id).generation

    @staticmethod
    def _has_state(connection: Any) -> bool:
        tables = {row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name IN "
            "('calib_active','calib_generations','calib_blocks','calib_decisions','calib_pointer_journal')"
        ).fetchall()}
        if not tables:
            return False
        if not {'calib_active', 'calib_generations', 'calib_blocks', 'calib_decisions'} <= tables:
            raise ValueError("incomplete durable physical model schema")
        return True

    def state(self, workcell_id: str) -> GenerationState:
        """Read pointer and stop-use together under the shared owner lock, without migration."""
        if not isinstance(workcell_id, str) or not workcell_id:
            raise ValueError("snapshot needs exact workcell identity")
        with self._lock:
            connection = self._holder.connection
            if not self._has_state(connection):
                return GenerationState(workcell_id)
            row = connection.execute("SELECT generation FROM calib_active WHERE workcell_id=?", [workcell_id]).fetchone()
            generation = self._generation(connection, workcell_id, row[0]) if row else None
            decision = self._pointer_decision(connection, workcell_id)
            expected = decision["generation"] if decision is not None else None
            if (row[0] if row else None) != expected:
                raise ValueError("active pointer differs from durable calibration decision")
            if decision is not None and decision["kind"] == "invalidated":
                stopped = self._generation(connection, workcell_id, decision["previous_generation"])
                return GenerationState(workcell_id, None, _json(decision), stopped)
            return GenerationState(workcell_id, generation)

    def states(self) -> tuple[GenerationState, ...]:
        """Startup read-only inventory; every persisted workcell is checked before dispatch."""
        with self._lock:
            connection = self._holder.connection
            if not self._has_state(connection):
                return ()
            workcells = connection.execute(
                "SELECT workcell_id FROM calib_active UNION SELECT workcell_id FROM calib_generations "
                "UNION SELECT workcell_id FROM calib_decisions"
            ).fetchall()
            journal = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_pointer_journal'"
            ).fetchone()[0]
            if journal:
                workcells = sorted(set(workcells) | set(connection.execute(
                    "SELECT DISTINCT workcell_id FROM calib_pointer_journal"
                ).fetchall()))
            return tuple(self.state(row[0]) for row in workcells)

    @staticmethod
    def _pointer_decision(connection: Any, workcell_id: str) -> dict[str, Any] | None:
        journal = connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_pointer_journal'").fetchone()[0]
        if journal:
            count, first, last = connection.execute(
                "SELECT COUNT(*),MIN(revision),MAX(revision) FROM calib_pointer_journal WHERE workcell_id=?",
                [workcell_id],
            ).fetchone()
            decisions = connection.execute(
                "SELECT COUNT(*) FROM calib_decisions WHERE workcell_id=? AND kind IN ('activated','rolled_back','invalidated')",
                [workcell_id],
            ).fetchone()[0]
            missing = connection.execute(
                "SELECT COUNT(*) FROM calib_pointer_journal j LEFT JOIN calib_decisions d "
                "ON j.event_id=d.event_id AND j.workcell_id=d.workcell_id WHERE j.workcell_id=? "
                "AND (d.event_id IS NULL OR d.kind NOT IN ('activated','rolled_back','invalidated'))",
                [workcell_id],
            ).fetchone()[0]
            if count != decisions or missing or (count and (first != 1 or last != count)):
                raise ValueError("physical pointer journal is incomplete or references a missing decision")
            row = connection.execute(
                "SELECT j.event_id,d.decision_json,d.request_id,d.kind FROM calib_pointer_journal j LEFT JOIN calib_decisions d "
                "ON j.event_id=d.event_id AND j.workcell_id=d.workcell_id WHERE j.workcell_id=? ORDER BY j.revision DESC LIMIT 1",
                [workcell_id],
            ).fetchone()
            if row:
                if row[1] is None:
                    raise ValueError("physical pointer journal references a missing decision")
                decision = GenerationStore._read_decision(row[1])
                if (decision["event_id"] != row[0] or decision["kind"] not in POINTER_DECISIONS
                    or decision["workcell_id"] != workcell_id or decision["request_id"] != row[2] or decision["kind"] != row[3]):
                    raise ValueError("physical pointer journal decision mismatch")
                return decision
            return None
        rows = connection.execute(
            "SELECT decision_json,created_at,event_id,request_id,kind FROM calib_decisions WHERE workcell_id=? "
            "AND kind IN ('activated','rolled_back','invalidated') ORDER BY created_at DESC,event_id DESC LIMIT 2",
            [workcell_id],
        ).fetchall()
        if len(rows) == 2 and rows[0][1] == rows[1][1]:
            raise ValueError("legacy physical pointer decision ordering is ambiguous")
        if not rows:
            return None
        decision = GenerationStore._read_decision(rows[0][0])
        if (decision["workcell_id"] != workcell_id or decision["event_id"] != rows[0][2]
            or decision["request_id"] != rows[0][3] or decision["kind"] != rows[0][4]):
            raise ValueError("legacy physical pointer decision identity mismatch")
        return decision

    def invalidation(self, workcell_id: str) -> dict[str, Any] | None:
        """Restore a persisted stop-use verdict, never turn deactivation into nominal trust."""
        return self.state(workcell_id).invalidation

    def generation(self, workcell_id: str, generation: int) -> ModelGeneration:
        """Read one immutable generation and its causal/blocks integrity, without mutation."""
        if not isinstance(workcell_id, str) or not workcell_id or type(generation) is not int or generation < 1:
            raise ValueError("exact workcell and positive committed generation required")
        with self._lock:
            connection = self._holder.connection
            if not self._has_state(connection):
                raise ValueError("no committed physical model generation")
            return self._generation(connection, workcell_id, generation)

    def active(self, workcell_id: str) -> ModelGeneration | None:
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            return self.snapshot(workcell_id)

    def committed_request(self, workcell_id: str, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            row = connection.execute("SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?", [workcell_id, request_id]).fetchone()
            return self._read_decision(row[0]) if row else None

    def record_decision(self, *, workcell_id: str, request_id: str, kind: str,
                        causal: Mapping[str, Any]) -> dict[str, Any]:
        """Durably retain rejected/no-op/candidate branches without mutating model authority."""
        if not all(isinstance(v, str) and v and len(v) <= 256 for v in (workcell_id, request_id)) or kind not in RECORD_DECISIONS:
            raise ValueError("bounded exact record-only calibration decision required")
        content = json.loads(_json(causal))
        decision = {"schema": SCHEMA, "kind": kind, "workcell_id": workcell_id,
                    "request_id": request_id, "causal": content}
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                row = connection.execute("SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?",
                                         [workcell_id, request_id]).fetchone()
                if row is not None:
                    previous = self._read_decision(row[0])
                    if {key: value for key, value in previous.items() if key not in ("event_id", "event_class")} != decision:
                        raise ValueError("request ID rebound to different calibration decision")
                else:
                    self._decision(connection, decision)
                encoded = connection.execute("SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?",
                                             [workcell_id, request_id]).fetchone()[0]
                result = self._read_decision(encoded)
                connection.execute("COMMIT")
                return result
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def commit_generation(self, *, workcell_id: str, request_id: str, expected_generation: int | None,
                          dependency_digest: str, model: Mapping[str, Any], accepted_blocks: Mapping[str, Mapping[str, Any]],
                          causal: Mapping[str, Any]) -> ModelGeneration:
        """Owner-internal atomic commit, called ONLY after admission and resource reservation.

        This primitive intentionally does not expose an activation RPC/skill and does
        not interpret caller flags as qualification. Physical admission is upstream.
        """
        if not all(isinstance(v, str) and v for v in (workcell_id, request_id, dependency_digest)) or not accepted_blocks:
            raise ValueError("activation needs exact workcell/request/dependencies and admitted blocks")
        if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 1):
            raise ValueError("expected generation must be a positive integer or nominal None")
        encoded = _json(model)
        model = json.loads(encoded)
        accepted_blocks = json.loads(_json(accepted_blocks))
        causal = json.loads(_json(causal))
        model_digest = digest_of(model)
        fingerprint = digest_of({"model": model, "blocks": accepted_blocks, "dependencies": dependency_digest, "expected": expected_generation, "causal": causal})
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                duplicate = connection.execute("SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?", [workcell_id, request_id]).fetchone()
                if duplicate:
                    decision = self._read_decision(duplicate[0])
                    if decision.get("fingerprint") != fingerprint or decision.get("kind") != "activated":
                        raise ValueError("request ID was already committed with different activation content")
                    result = self._generation(connection, workcell_id, decision["generation"])
                    connection.execute("COMMIT")
                    return result
                active = self.snapshot(workcell_id)
                previous = active.generation if active is not None else None
                if previous != expected_generation:
                    raise ValueError("active generation compare-and-swap conflict")
                if previous is not None:
                    prior = self._generation(connection, workcell_id, previous)
                    if prior.dependency_digest != dependency_digest:
                        raise ValueError("active model dependencies changed; invalidate before activation")
                last = connection.execute("SELECT MAX(generation) FROM calib_generations WHERE workcell_id=?", [workcell_id]).fetchone()[0]
                generation = (last or 0) + 1
                connection.execute("INSERT INTO calib_generations VALUES (?,?,?,?,?,?,?)", [workcell_id, generation, model_digest, dependency_digest, encoded, previous, request_id])
                for block_id, record in accepted_blocks.items():
                    if not isinstance(block_id, str) or not block_id:
                        raise ValueError("accepted block identity cannot be empty")
                    connection.execute("INSERT INTO calib_blocks VALUES (?,?,?,?)", [workcell_id, generation, block_id, _json(record)])
                decision = {"schema": SCHEMA, "kind": "activated", "workcell_id": workcell_id, "request_id": request_id, "generation": generation, "previous_generation": previous, "model_digest": model_digest, "dependency_digest": dependency_digest, "fingerprint": fingerprint, "causal": dict(causal)}
                self._decision(connection, decision)
                connection.execute("INSERT INTO calib_active VALUES (?,?) ON CONFLICT(workcell_id) DO UPDATE SET generation=excluded.generation", [workcell_id, generation])
                connection.execute("COMMIT")
                return ModelGeneration(workcell_id, generation, model_digest, dependency_digest, encoded, previous, request_id)
            except Exception:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _decision(connection: Any, decision: Mapping[str, Any]) -> None:
        event_id = digest_of(decision)
        encoded = _json({**decision, "event_id": event_id, "event_class": "record"})
        connection.execute("INSERT INTO calib_decisions VALUES (?,?,?,?,?,?)", [event_id, decision["workcell_id"], decision["request_id"], decision["kind"], encoded, time.time()])
        connection.execute("INSERT INTO calib_outbox(event_id,event_json) VALUES (?,?)", [event_id, encoded])
        if decision["kind"] in POINTER_DECISIONS:
            last = connection.execute("SELECT MAX(revision) FROM calib_pointer_journal WHERE workcell_id=?", [decision["workcell_id"]]).fetchone()[0]
            connection.execute("INSERT INTO calib_pointer_journal VALUES (?,?,?)", [decision["workcell_id"], (last or 0) + 1, event_id])

    def invalidate(self, *, workcell_id: str, request_id: str, expected_generation: int,
                   dependency_digest: str, causal: Mapping[str, Any]) -> None:
        """CAS stop-use transaction; no prior model is restored and no motion is commanded."""
        if not all(isinstance(v, str) and v for v in (workcell_id, request_id, dependency_digest)) or type(expected_generation) is not int or expected_generation < 1:
            raise ValueError("invalidation needs exact workcell/request/dependencies/generation")
        causal = json.loads(_json(causal))
        fingerprint = digest_of({"expected": expected_generation, "dependencies": dependency_digest, "causal": causal, "operation": "invalidate"})
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                duplicate = connection.execute("SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?", [workcell_id, request_id]).fetchone()
                if duplicate:
                    decision = self._read_decision(duplicate[0])
                    if decision.get("fingerprint") != fingerprint or decision.get("kind") != "invalidated":
                        raise ValueError("request ID rebound to different invalidation")
                else:
                    current = self.snapshot(workcell_id)
                    if current is None or current.generation != expected_generation:
                        raise ValueError("invalidation generation compare-and-swap conflict")
                    connection.execute("DELETE FROM calib_active WHERE workcell_id=?", [workcell_id])
                    self._decision(connection, {"schema": SCHEMA, "kind": "invalidated", "workcell_id": workcell_id,
                        "request_id": request_id, "generation": None, "previous_generation": current.generation,
                        "dependency_digest": dependency_digest, "fingerprint": fingerprint, "causal": causal})
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def rollback(self, *, workcell_id: str, request_id: str, expected_generation: int, dependency_digest: str, causal: Mapping[str, Any], compatible_model_digest: str | None = None) -> ModelGeneration | None:
        """Between-run owner primitive; incompatible prior dependencies deactivate, never move.

        The owner can additionally restrict restoration to its revalidated model
        digest. An empty digest forces stop-use in this SAME transaction when a
        prior safety projection cannot be requalified (including after restart).
        None preserves the dependency-only internal persistence primitive.
        """
        if not all(isinstance(v, str) and v for v in (workcell_id, request_id, dependency_digest)) or type(expected_generation) is not int or expected_generation < 1:
            raise ValueError("rollback needs exact workcell/request/dependencies/generation")
        causal = json.loads(_json(causal))
        if compatible_model_digest is not None and (
            not isinstance(compatible_model_digest, str)
            or (compatible_model_digest and (len(compatible_model_digest) != 64 or any(c not in "0123456789abcdef" for c in compatible_model_digest)))
        ):
            raise ValueError("compatible prior model must be a canonical digest or explicit stop-use")
        identity = {"expected": expected_generation, "dependencies": dependency_digest, "causal": causal}
        if compatible_model_digest is not None:
            identity["compatible_model_digest"] = compatible_model_digest
        fingerprint = digest_of(identity)
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                duplicate = connection.execute("SELECT decision_json FROM calib_decisions WHERE workcell_id=? AND request_id=?", [workcell_id, request_id]).fetchone()
                if duplicate:
                    decision = self._read_decision(duplicate[0])
                    if decision.get("fingerprint") != fingerprint or decision.get("kind") not in ("rolled_back", "invalidated"):
                        raise ValueError("request ID rebound to different rollback")
                    result = self._generation(connection, workcell_id, decision["generation"]) if decision["generation"] is not None else None
                    connection.execute("COMMIT")
                    return result
                current = self.snapshot(workcell_id)
                if current is None or current.generation != expected_generation:
                    raise ValueError("rollback generation compare-and-swap conflict")
                prior = self._generation(connection, workcell_id, current.previous_generation) if current.previous_generation is not None else None
                compatible = (
                    prior is not None and prior.dependency_digest == dependency_digest
                    and (compatible_model_digest is None or prior.model_digest == compatible_model_digest)
                )
                if compatible:
                    connection.execute("UPDATE calib_active SET generation=? WHERE workcell_id=?", [prior.generation, workcell_id])
                else:
                    connection.execute("DELETE FROM calib_active WHERE workcell_id=?", [workcell_id])
                self._decision(connection, {"schema": SCHEMA, "kind": "rolled_back" if compatible else "invalidated", "workcell_id": workcell_id, "request_id": request_id, "generation": prior.generation if compatible else None, "previous_generation": current.generation, "fingerprint": fingerprint, "causal": dict(causal)})
                connection.execute("COMMIT")
                return prior if compatible else None
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def pending_events(self, *, limit: int = 100, device_id: str = "") -> tuple[dict[str, Any], ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("calibration outbox limit must be bounded")
        with self._lock:
            connection = self._holder.connection
            if not self._has_state(connection):
                return ()
            if not connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_outbox'").fetchone()[0]:
                raise ValueError("durable calibration outbox missing from committed schema")
            if device_id:
                if not isinstance(device_id, str) or len(device_id) > 256:
                    raise ValueError("bounded exact lifecycle device required")
                # Filter before LIMIT: unrelated pending records cannot starve a
                # device's probe. Pointer events route through their persisted model,
                # record-only events through their explicitly bound causal device.
                rows = connection.execute(
                    "SELECT o.event_json FROM calib_outbox o JOIN calib_decisions d ON o.event_id=d.event_id "
                    "WHERE NOT o.delivered AND (json_extract_string(d.decision_json,'$.causal.device_id')=? "
                    "OR EXISTS (SELECT 1 FROM calib_generations g WHERE g.workcell_id=d.workcell_id "
                    "AND g.generation=COALESCE(TRY_CAST(json_extract_string(d.decision_json,'$.generation') AS BIGINT), "
                    "TRY_CAST(json_extract_string(d.decision_json,'$.previous_generation') AS BIGINT)) "
                    "AND json_extract_string(g.model_json,'$.device_id')=?)) ORDER BY o.event_id LIMIT ?",
                    [device_id, device_id, limit],
                ).fetchall()
            else:
                rows = connection.execute("SELECT event_json FROM calib_outbox WHERE NOT delivered ORDER BY event_id LIMIT ?", [limit]).fetchall()
            return tuple(self._read_decision(row[0]) for row in rows)

    def acknowledge_event(self, event_id: str) -> None:
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            connection.execute("UPDATE calib_outbox SET delivered=TRUE WHERE event_id=?", [event_id])
