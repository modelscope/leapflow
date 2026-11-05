# Copyright (c) Alibaba, Inc. and its affiliates.
"""Explicit credential-level state machine for multi-key LLM credential pools.

A single provider may hold several API keys. Each key is an independent
credential whose health evolves over the life of the process:

    OK -> EXHAUSTED -> OK        (transient rate-limit / quota, auto-recovers)
    OK -> DEAD                   (revoked / billing failure, terminal)

``CredentialPool`` (in ``provider_chain``) drives these transitions; this
module owns the domain types so ``provider_chain`` stays focused on the chain
and failover mechanics.

Design notes:
- ``CredentialEntry`` is intentionally NOT frozen: its state mutates in place.
- ``AllCredentialsExhausted`` is a plain ``Exception`` subclass, NOT a frozen
  dataclass: CPython assigns ``__traceback__`` on the instance during every
  re-raise, which a frozen type rejects with ``FrozenInstanceError`` and thereby
  masks the real failure.
- Category -> disposition is a data table keyed by the *string value* of
  ``engine.error_classifier.ErrorCategory`` so this module never imports the
  engine layer (``engine`` imports ``llm`` at load time; the reverse would
  create an import cycle).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CredentialState(Enum):
    """Health state of a single credential (API key)."""

    OK = "ok"
    """Healthy and selectable."""

    EXHAUSTED = "exhausted"
    """Temporarily unusable (rate limit / quota). Auto-recovers after cooldown."""

    DEAD = "dead"
    """Permanently failed (revoked / billing). Never auto-recovers. Terminal."""


@dataclass
class CredentialEntry:
    """Mutable per-credential health record.

    Not frozen: ``state``, ``last_used``, cooldown and failure counters all
    change over the credential's lifetime.
    """

    api_key: str
    state: CredentialState = CredentialState.OK
    last_used: float = 0.0  # monotonic timestamp of last selection (for LRU)
    cooldown_until: float = 0.0  # monotonic deadline; EXHAUSTED -> OK when passed
    consecutive_failures: int = 0
    death_reason: str = ""


class AllCredentialsExhausted(Exception):
    """Raised when a ``CredentialPool`` has no usable (OK) credential left.

    Carries enough context for callers and the error classifier to distinguish a
    temporary drain (all keys cooling down) from a terminal one (all keys dead).

    Plain ``Exception`` subclass on purpose: a frozen dataclass exception raises
    ``FrozenInstanceError`` when Python assigns ``__traceback__`` on re-raise,
    replacing the real failure with unrelated noise.
    """

    def __init__(
        self,
        provider: str,
        *,
        total: int,
        dead: int,
        cooling_down: int,
        earliest_cooldown_until: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.total = total
        self.dead = dead
        self.cooling_down = cooling_down
        self.earliest_cooldown_until = earliest_cooldown_until

        label = provider or "provider"
        wait = ""
        if earliest_cooldown_until is not None:
            remaining = max(0.0, earliest_cooldown_until - time.monotonic())
            wait = f"; earliest recovery in ~{remaining:.0f}s"
        super().__init__(
            f"All {total} credentials for '{label}' are unusable "
            f"({dead} dead, {cooling_down} cooling down){wait}."
        )


class CredentialDisposition(Enum):
    """How a classified error affects the credential that produced it."""

    NONE = "none"
    """Not a credential-scoped error; leave the credential untouched."""

    EXHAUSTED = "exhausted"
    """Temporarily unusable — put the credential into cooldown."""

    DEAD = "dead"
    """Permanently failed — mark the credential dead. Terminal."""


# Error category values (see ``engine.error_classifier.ErrorCategory``) that
# permanently kill a credential. Billing/quota-permanent and permanent auth
# failures mean this key will not recover; retrying it wastes budget.
CREDENTIAL_DEAD_CATEGORIES = frozenset({"billing", "auth_permanent"})

# Category values that temporarily exhaust a credential — a cooldown lets it
# recover. ``auth_error`` is transient/recoverable by design (rotation to
# another key), so it is exhausted rather than killed.
CREDENTIAL_EXHAUSTED_CATEGORIES = frozenset({"rate_limited", "overloaded", "auth_error"})


def disposition_for_category(category_value: str) -> CredentialDisposition:
    """Map an ``ErrorCategory`` value to its credential disposition (data-driven)."""
    if category_value in CREDENTIAL_DEAD_CATEGORIES:
        return CredentialDisposition.DEAD
    if category_value in CREDENTIAL_EXHAUSTED_CATEGORIES:
        return CredentialDisposition.EXHAUSTED
    return CredentialDisposition.NONE
