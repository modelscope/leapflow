"""Evolution ledger: the causal view of how the framework changed itself.

Two halves that meet at :class:`~leapflow.domain.evolution_trace.EvolutionEpisode`:

* :class:`EvolutionLedger` *reconstructs* episodes from records the adaptive loop
  and observation layer already persist -- no probe, no new schema;
* :class:`LedgerEvolutionSink` *collects* live traces from probe sites for the
  facts no store retains: registry mutations, trust transitions, world-model
  proposals that were never admitted, and lifecycle openings.

The first works alone; the second only fills the gaps the first cannot see.
"""

from leapflow.evolution.ledger import DEFAULT_EPISODE_TTL_S, EvolutionLedger
from leapflow.evolution.sink import DEFAULT_BUFFER_SIZE, LedgerEvolutionSink
from leapflow.evolution.sweep import CoevolutionSweep, SweepOutcome

__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_EPISODE_TTL_S",
    "CoevolutionSweep",
    "EvolutionLedger",
    "LedgerEvolutionSink",
    "SweepOutcome",
]
