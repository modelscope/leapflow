# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded durable calibration telemetry through ordinary owner probes.

Raw evidence and permit proofs never cross this boundary. Acknowledgment means
acceptance at the existing unified signal entry, not physical qualification or
authority. Records remain in the owner outbox until that acceptance is reported.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Mapping

from leapcalib.runtime.protocol import freeze_json
from leapflow.hardware.stream import HardwareEvent, HardwareEventClass

logger = logging.getLogger(__name__)
SCHEMA = "leapflow.calibration_lifecycle.v1"
MAX_EVENTS = 16
MAX_BYTES = 32 * 1024
KINDS = frozenset({"candidate", "rejected", "no_op", "activated", "rolled_back", "invalidated"})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"event_id", "kind", "workcell_id", "device_id", "request_digest", "causal_digest",
           "generation", "previous_generation", "model_digest", "dependency_digest"}


def project_decision(decision: Mapping[str, Any], device_id: str) -> dict[str, Any]:
    """Allowlist only redacted identities; never forward arbitrary causal payloads."""
    from leapcalib.contracts.identity import digest_of

    return {
        "event_id": decision["event_id"], "kind": decision["kind"],
        "workcell_id": decision["workcell_id"], "device_id": device_id,
        "request_digest": digest_of({"request_id": decision["request_id"]}),
        "causal_digest": digest_of(decision["causal"]),
        **{key: decision.get(key) for key in
           ("generation", "previous_generation", "model_digest", "dependency_digest")},
    }


def validate_packet(packet: Mapping[str, Any], device_id: str) -> dict[str, Any]:
    """Validate counts/bytes and exact same-device scope before signal admission."""
    if (not isinstance(packet, Mapping) or set(packet) != {"schema", "events"}
        or packet.get("schema") != SCHEMA or not isinstance(packet.get("events"), list)
        or len(packet["events"]) > MAX_EVENTS):
        raise ValueError("bounded versioned calibration telemetry required")
    value = freeze_json(packet, MAX_BYTES)
    seen = set()
    for event in value["events"]:
        if not isinstance(event, dict) or set(event) != _FIELDS:
            raise ValueError("exact redacted lifecycle fields required")
        if event["kind"] not in KINDS or event["device_id"] != device_id:
            raise ValueError("lifecycle classification/device scope mismatch")
        if not isinstance(event["workcell_id"], str) or not 1 <= len(event["workcell_id"]) <= 256:
            raise ValueError("bounded lifecycle workcell required")
        for key in ("event_id", "request_digest", "causal_digest"):
            if not isinstance(event[key], str) or not _DIGEST.fullmatch(event[key]):
                raise ValueError("canonical lifecycle digest required")
        for key in ("model_digest", "dependency_digest"):
            if event[key] is not None and (not isinstance(event[key], str) or not 1 <= len(event[key]) <= 256):
                raise ValueError("bounded optional lifecycle model identity required")
        for key in ("generation", "previous_generation"):
            if event[key] is not None and (type(event[key]) is not int or event[key] < 1):
                raise ValueError("positive lifecycle generation required")
        if event["event_id"] in seen:
            raise ValueError("duplicate lifecycle event in owner packet")
        seen.add(event["event_id"])
    return value


class OwnerLifecycle:
    """Connection-scoped offers/acks; a probe cannot acknowledge unoffered records."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._offered: dict[tuple[str, str], frozenset[str]] = {}

    def disconnect(self, connection_id: str) -> None:
        for key in tuple(self._offered):
            if key[0] == connection_id:
                del self._offered[key]

    async def probe(self, device_id: str, connection_id: str, acknowledgments: Any = ()) -> dict[str, Any]:
        if (not isinstance(acknowledgments, (list, tuple)) or len(acknowledgments) > MAX_EVENTS
            or any(not isinstance(identity, str) or not _DIGEST.fullmatch(identity) for identity in acknowledgments)):
            raise ValueError("bounded canonical lifecycle acknowledgments required")
        key = (connection_id, device_id)
        if not set(acknowledgments) <= self._offered.get(key, frozenset()):
            raise ValueError("cannot acknowledge lifecycle records not offered to this device/connection")
        store = self._registry.calibration_generations
        if store is None:
            return {"schema": SCHEMA, "events": []}
        # One off-loop store operation, no camera, actuator or permit access.
        def read() -> dict[str, Any]:
            for identity in acknowledgments:
                store.acknowledge_event(identity)
            events = [project_decision(decision, device_id) for decision in
                      store.pending_events(limit=MAX_EVENTS, device_id=device_id)]
            return validate_packet({"schema": SCHEMA, "events": events}, device_id)
        packet = await asyncio.to_thread(read)
        self._offered[key] = frozenset(acknowledgments) | frozenset(event["event_id"] for event in packet["events"]) 
        return packet


class LifecycleReceiver:
    """Host-side delivery through HardwareRegistry.publish_event, never a mutation path."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._accepted: dict[str, set[str]] = {}

    def observe(self, device_id: str, status: Any) -> tuple[str, ...]:
        metadata = getattr(status, "metadata", None)
        packet = metadata.get("calibration_lifecycle") if isinstance(metadata, Mapping) else None
        if packet is None:
            return ()
        packet = validate_packet(packet, device_id)
        accepted = self._accepted.setdefault(device_id, set())
        current = {event["event_id"] for event in packet["events"]}
        accepted.intersection_update(current)  # bounded by a single outstanding owner page
        acknowledgments = []
        for event in packet["events"]:
            identity = event["event_id"]
            if identity not in accepted:
                signal = HardwareEvent(
                    kind=f"calibration_{event['kind']}", device_id=device_id,
                    channel_id="calibration_lifecycle", quantity="calibration_generation",
                    detail=f"Calibration {event['kind']}; causal record {identity}",
                    value=dict(event), observed_at=time.time(), event_class=HardwareEventClass.RECORD,
                )
                if self._registry.publish_event(signal) is not True:
                    logger.warning("Calibration signal routing unavailable; durable event remains pending: %s", identity)
                    continue
                accepted.add(identity)
            acknowledgments.append(identity)
        return tuple(acknowledgments)
