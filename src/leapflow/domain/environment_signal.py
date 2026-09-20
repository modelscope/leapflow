# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed task-environment snapshots and structural deltas."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from leapflow.domain.evolution_event import content_hash

EnvironmentObservationKind = Literal["snapshot", "delta", "outcome"]
EnvironmentOutcome = Literal["PASS", "FAIL", "UNKNOWN"]


@dataclass(frozen=True, order=True)
class InterfaceElement:
    """Stable, value-free description of one application affordance."""

    name: str
    role: str = "widget"
    enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InterfaceElement":
        return cls(
            name=str(value.get("name") or value.get("label") or ""),
            role=str(value.get("role") or "widget"),
            enabled=bool(value.get("enabled", True)),
        )


@dataclass(frozen=True)
class InterfaceSnapshot:
    """Immutable structural view of one task application."""

    source_id: str
    app_id: str
    workspace_id: str = ""
    session_id: str = ""
    version: str = ""
    affordances: tuple[str, ...] = ()
    elements: tuple[InterfaceElement, ...] = ()
    data_hash: str = ""
    observed_at: float = 0.0
    provenance: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        app_id: str,
        workspace_id: str = "",
        session_id: str = "",
        version: str = "",
        affordances: Any = (),
        elements: Any = (),
        data: Any = None,
        observed_at: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "InterfaceSnapshot":
        normalized_elements = tuple(
            sorted(
                (
                    item
                    if isinstance(item, InterfaceElement)
                    else InterfaceElement.from_mapping(item)
                )
                for item in (elements or ())
                if isinstance(item, (InterfaceElement, Mapping))
            )
        )
        return cls(
            source_id=str(source_id),
            app_id=str(app_id),
            workspace_id=str(workspace_id),
            session_id=str(session_id),
            version=str(version or ""),
            affordances=tuple(sorted({str(item) for item in (affordances or ()) if str(item)})),
            elements=normalized_elements,
            data_hash=content_hash(data or {}),
            observed_at=time.time() if observed_at is None else float(observed_at),
            provenance=tuple(
                sorted((str(key), str(value)) for key, value in (provenance or {}).items())
            ),
        )

    @property
    def snapshot_id(self) -> str:
        return "env-" + content_hash(
            {
                "source_id": self.source_id,
                "app_id": self.app_id,
                "workspace_id": self.workspace_id,
                "session_id": self.session_id,
                "version": self.version,
                "affordances": self.affordances,
                "elements": [asdict(item) for item in self.elements],
                "provenance": self.provenance,
            }
        )[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_id": self.source_id,
            "app_id": self.app_id,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "version": self.version,
            "affordances": list(self.affordances),
            "elements": [asdict(item) for item in self.elements],
            "data_hash": self.data_hash,
            "observed_at": self.observed_at,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class EnvironmentObservation:
    """A snapshot, structural delta, or ground-truth task outcome."""

    observation_id: str
    source_id: str
    kind: EnvironmentObservationKind
    app_id: str = ""
    workspace_id: str = ""
    session_id: str = ""
    before_snapshot_id: str = ""
    after_snapshot_id: str = ""
    version_before: str = ""
    version_after: str = ""
    added_affordances: tuple[str, ...] = ()
    removed_affordances: tuple[str, ...] = ()
    added_elements: tuple[InterfaceElement, ...] = ()
    removed_elements: tuple[InterfaceElement, ...] = ()
    changed_elements: tuple[InterfaceElement, ...] = ()
    outcome: EnvironmentOutcome = "UNKNOWN"
    capability: str = ""
    observed_at: float = 0.0
    provenance: tuple[tuple[str, str], ...] = ()

    @classmethod
    def snapshot(cls, snapshot: InterfaceSnapshot) -> "EnvironmentObservation":
        identity = {"kind": "snapshot", "snapshot_id": snapshot.snapshot_id}
        return cls(
            observation_id="obs-" + content_hash(identity)[:32],
            source_id=snapshot.source_id,
            kind="snapshot",
            app_id=snapshot.app_id,
            workspace_id=snapshot.workspace_id,
            session_id=snapshot.session_id,
            after_snapshot_id=snapshot.snapshot_id,
            version_after=snapshot.version,
            added_affordances=snapshot.affordances,
            added_elements=snapshot.elements,
            observed_at=snapshot.observed_at,
            provenance=snapshot.provenance,
        )

    @classmethod
    def between(
        cls,
        before: InterfaceSnapshot,
        after: InterfaceSnapshot,
    ) -> "EnvironmentObservation | None":
        if before.source_id != after.source_id or before.app_id != after.app_id:
            raise ValueError("environment snapshots must describe the same source and app")
        before_elements = {(item.name, item.role): item for item in before.elements}
        after_elements = {(item.name, item.role): item for item in after.elements}
        added_keys = after_elements.keys() - before_elements.keys()
        removed_keys = before_elements.keys() - after_elements.keys()
        changed = tuple(
            after_elements[key]
            for key in sorted(before_elements.keys() & after_elements.keys())
            if before_elements[key].enabled != after_elements[key].enabled
        )
        added_affordances = tuple(sorted(set(after.affordances) - set(before.affordances)))
        removed_affordances = tuple(sorted(set(before.affordances) - set(after.affordances)))
        added_elements = tuple(after_elements[key] for key in sorted(added_keys))
        removed_elements = tuple(before_elements[key] for key in sorted(removed_keys))
        if not any(
            (
                before.version != after.version,
                added_affordances,
                removed_affordances,
                added_elements,
                removed_elements,
                changed,
            )
        ):
            return None
        identity = {
            "kind": "delta",
            "before": before.snapshot_id,
            "after": after.snapshot_id,
        }
        return cls(
            observation_id="obs-" + content_hash(identity)[:32],
            source_id=after.source_id,
            kind="delta",
            app_id=after.app_id,
            workspace_id=after.workspace_id,
            session_id=after.session_id,
            before_snapshot_id=before.snapshot_id,
            after_snapshot_id=after.snapshot_id,
            version_before=before.version,
            version_after=after.version,
            added_affordances=added_affordances,
            removed_affordances=removed_affordances,
            added_elements=added_elements,
            removed_elements=removed_elements,
            changed_elements=changed,
            observed_at=after.observed_at,
            provenance=after.provenance,
        )

    @classmethod
    def task_outcome(
        cls,
        *,
        source_id: str,
        task_id: str,
        outcome: str,
        capability: str = "",
        workspace_id: str = "",
        session_id: str = "",
        observed_at: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "EnvironmentObservation":
        normalized = str(outcome or "UNKNOWN").upper()
        if normalized not in {"PASS", "FAIL", "UNKNOWN"}:
            normalized = "UNKNOWN"
        identity = {
            "kind": "outcome",
            "source_id": source_id,
            "task_id": task_id,
            "outcome": normalized,
            "capability": capability,
            "provenance": dict(provenance or {}),
        }
        return cls(
            observation_id="obs-" + content_hash(identity)[:32],
            source_id=str(source_id),
            kind="outcome",
            app_id=str(task_id),
            workspace_id=str(workspace_id),
            session_id=str(session_id),
            outcome=normalized,  # type: ignore[arg-type]
            capability=str(capability),
            observed_at=time.time() if observed_at is None else float(observed_at),
            provenance=tuple(
                sorted((str(key), str(value)) for key, value in (provenance or {}).items())
            ),
        )

    @property
    def is_structural(self) -> bool:
        return bool(
            self.version_before != self.version_after
            or self.added_affordances
            or self.removed_affordances
            or self.added_elements
            or self.removed_elements
            or self.changed_elements
        )

    def capability_results(self) -> tuple[dict[str, Any], ...]:
        """Return only explicitly named capability evidence; never infer from labels."""
        results = [
            {
                "ok": False,
                "error_type": "affordance_removed",
                "capability": capability,
                "observation_id": self.observation_id,
                "app_id": self.app_id,
            }
            for capability in self.removed_affordances
        ]
        if self.kind == "outcome" and self.outcome == "FAIL" and self.capability:
            results.append(
                {
                    "ok": False,
                    "error_type": "task_outcome_failed",
                    "capability": self.capability,
                    "observation_id": self.observation_id,
                    "task_id": self.app_id,
                }
            )
        return tuple(results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "kind": self.kind,
            "app_id": self.app_id,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "before_snapshot_id": self.before_snapshot_id,
            "after_snapshot_id": self.after_snapshot_id,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "added_affordances": list(self.added_affordances),
            "removed_affordances": list(self.removed_affordances),
            "added_elements": [asdict(item) for item in self.added_elements],
            "removed_elements": [asdict(item) for item in self.removed_elements],
            "changed_elements": [asdict(item) for item in self.changed_elements],
            "outcome": self.outcome,
            "capability": self.capability,
            "observed_at": self.observed_at,
            "provenance": dict(self.provenance),
        }


__all__ = [
    "EnvironmentObservation",
    "EnvironmentObservationKind",
    "EnvironmentOutcome",
    "InterfaceElement",
    "InterfaceSnapshot",
]
