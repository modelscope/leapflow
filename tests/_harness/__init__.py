# Copyright (c) Alibaba, Inc. and its affiliates.
"""Test harness for the real end-to-end layer.

Modules here are infrastructure, not tests:

``cassette``
    Fingerprinting, persistence and miss diagnostics for recorded LLM traffic.
``cassette_proxy``
    A local OpenAI-compatible endpoint that records, replays or forwards.
``leapd``
    Spawns and drives a real daemon subprocess.
``journey``
    The coarse-grained journey runner, with phase attribution and budgets.
"""

from __future__ import annotations

from tests._harness.cassette import (
    CassetteRecord,
    CassetteResponse,
    CassetteStore,
    context_overflow_response,
    error_response,
    fingerprint,
    json_response,
    rate_limited_response,
    record_for,
    server_error_response,
    streamed_response,
    truncated_stream_response,
)
from tests._harness.cassette_proxy import (
    LIVE,
    MODE_ENV,
    RECORD,
    REPLAY,
    SEED,
    CassetteProxy,
    Script,
    ScriptedTurn,
    answer,
    resolve_mode,
    scripted,
    store_for,
    tool_call,
)
from typing import TYPE_CHECKING, Any

from tests._harness.leapd import Leapd, await_for, hermetic_env, start_leapd

if TYPE_CHECKING:
    from tests._harness.journey import Journey, JourneyFactory, JourneyPhaseError


def __getattr__(name: str) -> Any:
    # Read-only edge benchmarks use the calculation harness without pytest.
    # Journey-only dependencies load when a caller actually asks for a journey.
    if name in {"Journey", "JourneyFactory", "JourneyPhaseError"}:
        from tests._harness import journey
        value = getattr(journey, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LIVE",
    "MODE_ENV",
    "RECORD",
    "REPLAY",
    "SEED",
    "CassetteProxy",
    "CassetteRecord",
    "CassetteResponse",
    "CassetteStore",
    "Journey",
    "JourneyFactory",
    "JourneyPhaseError",
    "Leapd",
    "Script",
    "ScriptedTurn",
    "answer",
    "await_for",
    "context_overflow_response",
    "error_response",
    "fingerprint",
    "hermetic_env",
    "json_response",
    "rate_limited_response",
    "record_for",
    "resolve_mode",
    "scripted",
    "server_error_response",
    "start_leapd",
    "store_for",
    "streamed_response",
    "tool_call",
    "truncated_stream_response",
]
