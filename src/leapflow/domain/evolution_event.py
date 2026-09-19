# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable causal events for the long-running self-evolution pipeline.

The event log is the one durable vocabulary shared by execution, perception,
teaching, governance, and presentation.  Facts are appended; state is projected.
No consumer is allowed to infer missing stages as success.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

SCHEMA_VERSION = 1


class ActionEvidenceUnavailable(RuntimeError):
    """A mutating action cannot cross its durable evidence barrier."""


def _normalize_json(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy or reject ambiguous values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evolution event JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in normalized:
                raise ValueError(f"duplicate JSON key after string conversion: {name!r}")
            normalized[name] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalize_json(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    raise TypeError(f"unsupported evolution event JSON value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return strict deterministic JSON used for hashes and deduplication."""
    return json.dumps(
        _normalize_json(value if value is not None else {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Return a SHA-256 digest for deterministic JSON-compatible content."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvolutionContext:
    """Causal identity carried across the complete evolution lifecycle.

    Empty values are explicit unknowns, never permission to join records by guess.
    ``correlation_id`` identifies one evolution episode and ``causation_id`` points
    to the direct predecessor event.
    """

    profile_id: str = ""
    workspace_id: str = ""
    session_id: str = ""
    session_generation: int = 0
    turn_id: str = ""
    frame_id: str = ""
    action_id: str = ""
    observation_id: str = ""
    requirement_id: str = ""
    decision_id: str = ""
    proposal_id: str = ""
    artifact_id: str = ""
    plugin_id: str = ""
    version_id: str = ""
    correlation_id: str = ""
    causation_id: str = ""

    def __post_init__(self) -> None:
        if int(self.session_generation) < 0:
            raise ValueError("session_generation must be non-negative")
        object.__setattr__(self, "session_generation", int(self.session_generation))
        for name in self.__dataclass_fields__:
            if name != "session_generation":
                object.__setattr__(self, name, str(getattr(self, name) or ""))

    @classmethod
    def create(cls, **values: Any) -> "EvolutionContext":
        """Create a context, minting one episode correlation id when absent."""
        normalized = {
            name: (int(value) if name == "session_generation" else str(value or ""))
            for name, value in values.items()
            if name in cls.__dataclass_fields__
        }
        if not normalized.get("correlation_id"):
            normalized["correlation_id"] = f"evo-{uuid.uuid4().hex}"
        return cls(**normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EvolutionContext":
        """Build from a mapping while ignoring unknown future fields."""
        raw = dict(value or {})
        return cls(
            **{
                name: (
                    int(raw.get(name) or 0)
                    if name == "session_generation"
                    else str(raw.get(name) or "")
                )
                for name in cls.__dataclass_fields__
            }
        )

    def with_ids(self, **values: Any) -> "EvolutionContext":
        """Return a copy with selected identifiers updated."""
        allowed = {
            key: (int(value) if key == "session_generation" else str(value or ""))
            for key, value in values.items()
            if key in self.__dataclass_fields__
        }
        return replace(self, **allowed)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class EvolutionEvent:
    """One append-only fact in the evolution event stream."""

    event_id: str
    event_type: str
    context: EvolutionContext
    payload: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: float = 0.0
    producer: str = ""
    producer_version: str = ""
    privacy_class: str = "system"
    schema_version: int = SCHEMA_VERSION
    dedup_key: str = ""
    payload_hash: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.event_type:
            raise ValueError("event_type is required")
        if not self.producer:
            raise ValueError("producer is required")
        if not self.dedup_key:
            raise ValueError("dedup_key is required")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not math.isfinite(float(self.occurred_at)):
            raise ValueError("occurred_at must be finite")
        normalized = _normalize_json(dict(self.payload))
        digest = content_hash(normalized)
        if self.payload_hash and self.payload_hash != digest:
            raise ValueError("payload_hash does not match the event payload")
        object.__setattr__(self, "payload", _freeze_json(normalized))
        object.__setattr__(self, "payload_hash", digest)
        object.__setattr__(self, "occurred_at", float(self.occurred_at))

    @classmethod
    def create(
        cls,
        event_type: str,
        *,
        context: EvolutionContext,
        payload: Mapping[str, Any] | None = None,
        producer: str,
        producer_version: str = "",
        privacy_class: str = "system",
        occurred_at: float | None = None,
        dedup_key: str = "",
        event_id: str = "",
    ) -> "EvolutionEvent":
        """Create an event with stable payload hash and idempotency identity."""
        normalized_payload = dict(payload or {})
        payload_digest = content_hash(normalized_payload)
        identity = dedup_key or content_hash(
            {
                "event_type": str(event_type),
                "context": context.to_dict(),
                "payload_hash": payload_digest,
                "producer": str(producer),
            }
        )
        return cls(
            event_id=str(event_id or f"evt-{uuid.uuid4().hex}"),
            event_type=str(event_type),
            context=context,
            payload=normalized_payload,
            occurred_at=float(occurred_at if occurred_at is not None else time.time()),
            producer=str(producer),
            producer_version=str(producer_version),
            privacy_class=str(privacy_class or "system"),
            dedup_key=identity,
            payload_hash=payload_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "context": self.context.to_dict(),
            "payload": _thaw_json(self.payload),
            "occurred_at": self.occurred_at,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "privacy_class": self.privacy_class,
            "schema_version": self.schema_version,
            "dedup_key": self.dedup_key,
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvolutionEvent":
        return cls(
            event_id=str(value.get("event_id") or ""),
            event_type=str(value.get("event_type") or ""),
            context=EvolutionContext.from_mapping(value.get("context")),
            payload=_normalize_json(value.get("payload") or {}),
            occurred_at=float(value.get("occurred_at") or 0.0),
            producer=str(value.get("producer") or ""),
            producer_version=str(value.get("producer_version") or ""),
            privacy_class=str(value.get("privacy_class") or "system"),
            schema_version=int(value.get("schema_version") or SCHEMA_VERSION),
            dedup_key=str(value.get("dedup_key") or ""),
            payload_hash=str(value.get("payload_hash") or ""),
        )


@dataclass(frozen=True)
class EvolutionEventRecord:
    """One persisted event paired with its monotonic stream cursor."""

    sequence: int
    event: EvolutionEvent

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("event sequence must be positive")


@runtime_checkable
class EvolutionEventStore(Protocol):
    """Storage boundary for the append-only evolution fact stream."""

    def append(self, event: EvolutionEvent) -> bool: ...

    def append_many(self, events: Sequence[EvolutionEvent]) -> int: ...

    def read(
        self,
        *,
        profile_id: str = "",
        session_id: str = "",
        session_generation: int | None = None,
        correlation_id: str = "",
        proposal_id: str = "",
        proposal_events_only: bool = False,
        event_type: str = "",
        after_sequence: int = 0,
        through_sequence: int = 0,
        limit: int = 500,
    ) -> list[EvolutionEventRecord]: ...

    def latest_sequence(self, *, profile_id: str = "", session_id: str = "") -> int: ...

    def latest_evidence_sequence(
        self,
        *,
        profile_id: str,
        session_id: str,
        session_generation: int | None = None,
    ) -> int: ...

    def evidence_sessions(self, *, profile_id: str = "") -> list[dict[str, Any]]: ...

    def last_finalized_sequence(
        self,
        *,
        profile_id: str,
        session_id: str,
        session_generation: int | None = None,
    ) -> int: ...

    def finalize_session(
        self,
        *,
        profile_id: str,
        workspace_id: str,
        session_id: str,
        session_generation: int,
        from_sequence: int,
        through_sequence: int,
        reason: str,
        goal: str = "",
        model: str = "",
    ) -> tuple[str, str]: ...


__all__ = [
    "SCHEMA_VERSION",
    "ActionEvidenceUnavailable",
    "EvolutionContext",
    "EvolutionEvent",
    "EvolutionEventRecord",
    "EvolutionEventStore",
    "canonical_json",
    "content_hash",
]
