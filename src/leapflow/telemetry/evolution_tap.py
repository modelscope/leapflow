"""EvolutionTap: emit a framework-evolution fact, or do nothing at all.

One module-level optional sink, and one function that writes to it. When no sink
is installed -- the default, and the only state an in-process CLI ever sees --
``emit_trace`` costs a global read and a null check, so a probe at a mutation point
is free until someone asks to observe it.

Why a module-level global rather than an injected dependency: the probe sites are
places like the plugin registry's version bump and the trust ledger's level
transition. Those are pure, low-level objects with no service container to reach
into, and threading a sink through every one of them would put an observability
concern into their constructors. A process-wide opt-in sink keeps the call sites to
a single line and keeps the objects' own dependencies unchanged.

Three rules this module exists to guarantee, all of them from hard experience
recorded in the project's contract:

* **Telemetry never fails a turn.** ``emit_trace`` swallows everything, including
  a broken sink, and logs at debug. A probe is not allowed to have an opinion
  about whether the operation it observes succeeds.
* **A local defect is not an external failure.** Nothing here raises, so nothing
  here can be misread by the recovery classifier as a provider problem. Traces
  never enter ``RecoveryCoordinator``.
* **Cold path only.** The sink contract is "accept and return"; correlation,
  persistence and re-publication happen later, on the daemon's own schedule.
  A sink that blocks here would drag storage into a mutation point.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from leapflow.domain.evolution_trace import EvolutionStage, EvolutionTrace, EvolutionTraceSink

logger = logging.getLogger(__name__)

#: Process-wide optional sink. ``None`` means every probe is a no-op.
_SINK: EvolutionTraceSink | None = None


def install_sink(sink: EvolutionTraceSink | None) -> None:
    """Install (or clear, with ``None``) the process-wide trace sink.

    Called by the daemon once observation is enabled. Idempotent and last-wins:
    a second install replaces the first rather than fanning out, because two sinks
    would double-count every fact.
    """
    global _SINK
    _SINK = sink


def current_sink() -> EvolutionTraceSink | None:
    """The installed sink, for tests and for callers that must check first."""
    return _SINK


def is_enabled() -> bool:
    """Whether a sink is installed.

    Lets an expensive ``detail`` payload be skipped entirely rather than built and
    thrown away -- the only case where a probe site should branch.
    """
    return _SINK is not None


def emit_trace(
    stage: EvolutionStage,
    kind: str,
    *,
    correlation: Mapping[str, str] | None = None,
    summary: str = "",
    detail: Mapping[str, Any] | None = None,
    ts: float = 0.0,
) -> None:
    """Record one evolution fact. Never raises, never blocks, never no-ops loudly."""
    sink = _SINK
    if sink is None:
        return
    try:
        sink.record(
            EvolutionTrace(
                stage=stage,
                kind=str(kind),
                ts=float(ts or time.time()),
                correlation=dict(correlation or {}),
                summary=str(summary),
                detail=dict(detail or {}),
            )
        )
    except Exception:  # noqa: BLE001 - observability must never affect the observed
        logger.debug("evolution tap: sink rejected a trace", exc_info=True)


__all__ = ["current_sink", "emit_trace", "install_sink", "is_enabled"]
