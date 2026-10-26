# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declaration identity, and the operator's confirmation of it.

A declaration is an assertion -- "this is what the machine can do" -- and a file
cannot make it trustworthy on its own.  It needs two things: an *identity*, so a
change between review and use is noticed rather than silent, and a confirmation
written *out of band*, so the assertion is not merely self-attested.  Both live
here, because they are one fact seen from two sides: an entry in
``verified.json`` is the digest of the reviewed declaration plus the name of
whoever reviewed it.

Not to be confused with :mod:`leapflow.hardware.verification`, which judges
whether an operation achieved its physical goal.  This module judges whether the
*description* of a device is the one a human took responsibility for.

The digest is computed over the declaration's wire form -- node-private
transport configuration and the unresolved control bindings are stripped -- so a
node and its host derive the same value for the same declaration, and an
operator can compare the digest a node prints with the one the host recorded.

Confirmations are *only* digest-bound.  A record that names a verifier without
the digest of what they reviewed cannot be checked against anything, so it
confirms nothing; treating it as a confirmation would let a declaration be
rewritten after review and keep its authority.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from leapflow.hardware.context import HardwareContext, epoch_seconds

logger = logging.getLogger(__name__)

_VOUCHING_FIELDS = ("verified_by", "verified_at")
"""Provenance fields that record who vouched, not what was declared.

Excluded from the digest on purpose: a node that verifies its own copy of a
declaration must not thereby make its declaration look changed to the host, and
the host's own confirmation must not invalidate the digest that records it.
"""


def declaration_payload(context: HardwareContext) -> dict[str, Any]:
    """Return the canonical form a declaration's identity is computed over.

    Node-private content is removed: ``transport.config`` holds serial paths,
    driver credentials and the shared secret of a chained hop, and
    ``control_bindings`` is the same binding before resolution.  The transport
    *kind* stays, because a declaration that changes southbound transport has
    changed in a way a reviewer would want to be told about.
    """
    payload = context.to_dict()
    payload["transport"] = {"kind": context.transport.kind, "config": {}}
    payload.pop("control_bindings", None)
    provenance = dict(payload.get("provenance") or {})
    for field_name in _VOUCHING_FIELDS:
        provenance.pop(field_name, None)
    payload["provenance"] = provenance
    return payload


def declaration_digest(context: HardwareContext) -> str:
    """Return the sha256 identity of a declaration, comparable host-to-node."""
    canonical = json.dumps(
        declaration_payload(context), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_verified(source: Mapping[str, Any] | Path | str | None) -> dict[str, Any]:
    """Read confirmation records; any problem reading them means 'none recorded'.

    Never raises: this file is consulted on every admission, and a permissions
    mistake or a hand-edited YAML error must not be able to stop the hardware
    subsystem from loading.  An unreadable record is reported, because silently
    reporting "unverified" for a device an operator did confirm looks exactly
    like their record was lost.
    """
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    if not source:
        return {}
    target = Path(str(source)).expanduser()
    if not target.is_file():
        return {}
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Ignoring unreadable hardware verification file %s: %s", target, exc)
        return {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def bound_digest(source: Mapping[str, Any] | Path | str | None, device_id: str) -> str:
    """Return the digest bound to *device_id*, or ``""`` when nothing is bound."""
    entry = read_verified(source).get(device_id)
    if not isinstance(entry, Mapping):
        return ""
    return str(entry.get("declaration_digest") or "")


def confirmed_by(
    source: Mapping[str, Any] | Path | str | None,
    device_id: str,
    context: HardwareContext,
) -> str:
    """Return the verifier whose reviewed declaration this still is, or ``""``.

    The single predicate for "a human took responsibility for *this* declaration
    as it stands".  Both the local provider and the fleet use it, so a device
    cannot be confirmed through one path and read through another that applies a
    weaker rule.
    """
    entry = read_verified(source).get(device_id)
    if entry is None:
        return ""
    verifier = str(entry.get("verifier") or "") if isinstance(entry, Mapping) else ""
    if not verifier:
        logger.warning(
            "verified.json: device %r has no digest-bound confirmation; "
            "run 'leap hw verify %s' to record one",
            device_id,
            device_id,
        )
        return ""
    stored = str(entry.get("declaration_digest") or "")
    if not stored:
        logger.warning(
            "verified.json: entry for device %r names a verifier but no declaration "
            "digest, so it confirms nothing; re-run 'leap hw verify %s'",
            device_id,
            device_id,
        )
        return ""
    current = declaration_digest(context)
    if current != stored:
        logger.warning(
            "verified.json: device %r declaration changed since it was verified "
            "(reviewed=%s, current=%s); confirmation does not apply",
            device_id,
            stored[:12],
            current[:12],
        )
        return ""
    return verifier


def verification_stamp(
    source: Mapping[str, Any] | Path | str | None,
    device_id: str,
    context: HardwareContext,
) -> tuple[str, float]:
    """Return who confirmed *this* declaration, and when they did it.

    Both halves are answered together because they come from one record: a
    caller that read the verifier through one path and the timestamp through
    another could attach a name while dropping the moment of review, which is
    how ``hw describe`` reported ``verified_at: 0.0`` for a device that had just
    been re-attested.  Whether the confirmation applies at all is still decided
    by ``confirmed_by``, so this cannot become a weaker second predicate.
    """
    verifier = confirmed_by(source, device_id, context)
    if not verifier:
        return "", 0.0
    entry = read_verified(source).get(device_id)
    recorded = entry.get("verified_at") if isinstance(entry, Mapping) else None
    return verifier, epoch_seconds(recorded)


def store_verification(
    verified_path: Path | str,
    device_id: str,
    *,
    verifier: str,
    digest: str,
) -> dict[str, Any]:
    """Record one confirmation, preserving every other entry.

    Written to a sibling file and moved into place: this file is the only record
    of which declarations a human has reviewed, so an interrupted write must not
    be able to un-verify every other device on the host.
    """
    if not verifier:
        raise ValueError("a confirmation must name the person who made it")
    if not digest:
        raise ValueError("a confirmation must carry the digest of what was reviewed")
    entries = read_verified(verified_path)
    entry = {
        "verifier": verifier,
        "declaration_digest": digest,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    entries[device_id] = entry
    target = Path(str(verified_path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".tmp")
    pending.write_text(yaml.dump(entries, default_flow_style=False, sort_keys=True), encoding="utf-8")
    os.replace(pending, target)
    logger.info("Recorded %r as verifier of device %r (digest=%s)", verifier, device_id, digest[:12])
    return entry


__all__ = [
    "bound_digest",
    "confirmed_by",
    "declaration_digest",
    "declaration_payload",
    "read_verified",
    "store_verification",
    "verification_stamp",
]
