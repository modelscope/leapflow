# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lightweight signal observation producer for event-driven watches.

Produces a Finding summarizing recent signal activity for the watched domain.
No LLM calls — pure rule-based observation with sub-millisecond execution.
"""

from __future__ import annotations

from leapflow.monitor.types import Finding, ProducerContext, Severity


class SignalObservationProducer:
    """Produces findings based on signal flow activity (no LLM).

    Registered for the ``signal`` domain, this producer is triggered by
    event-driven watches (e.g. fs.*, gateway.*) and emits an INFO-level
    finding confirming the trigger fired.
    """

    domain = "signal"

    async def observe(self, ctx: ProducerContext) -> list[Finding]:
        """Report that an event-driven trigger fired."""
        return [
            Finding(
                watch_id=ctx.spec.watch_id or "",
                domain=self.domain,
                title="Signal activity detected",
                description=f"Event-driven trigger fired for {ctx.spec.name} (run #{ctx.run_count + 1})",
                severity=Severity.INFO,
            )
        ]


__all__ = ["SignalObservationProducer"]
