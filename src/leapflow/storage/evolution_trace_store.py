"""Durable store for framework-evolution traces.

⚠️ Not to be confused with :mod:`leapflow.storage.evolution_store`, whose
``DuckDBEvolutionStore.save_episode`` persists *skill learning* episodes. Two
different meanings of "evolution" live in this package, and both use the word
"episode": that one means "the agent practised a skill", this one means "the
framework changed itself". Named for ``EvolutionTrace`` rather than for evolution
in general precisely so the two cannot be mistaken for each other at a call site.

**JSON rather than DuckDB, deliberately.** The roadmap called for a DuckDB table;
this is a considered deviation:

* *Volume does not justify it.* Traces are written when the framework mutates --
  a plugin installs, a trust level moves, the world model proposes. Those are rare
  by nature, not per-turn. A table sized for time-series volume would carry
  connection-holder, schema and retry machinery for a file that gains a handful of
  rows a day.
* *It matches its neighbours.* The ledger already reads
  ``capability_plans.json``, ``capability_observations.json`` and
  ``proposal_queue.json`` from this same directory. One idiom for the causal
  history means one failure mode, not two.
* *Inspectable and additively versioned*, for the same reason the sibling
  capability stores chose JSON: an older record stays readable after the schema
  grows.

Retention is a hard cap on record count rather than an age, because what matters
is that the newest traces are always present -- an operator reading the board after
an incident needs the last mutations, not a complete history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

#: Keep the newest N traces. Generous relative to the write rate, and bounded so
#: the file cannot grow without limit on a long-lived profile.
DEFAULT_MAX_TRACES = 2000


class JsonEvolutionTraceStore:
    """Append-only, count-bounded JSON store for evolution traces.

    Every method degrades rather than raising: this store backs a transparency
    panel, and losing the panel is preferable to failing the operation a trace was
    describing. A corrupt or unreadable file reads as empty and is overwritten by
    the next append, which is the same choice the sibling capability stores make.
    """

    def __init__(self, path: Path, *, max_traces: int = DEFAULT_MAX_TRACES) -> None:
        self._path = Path(path)
        self._max = max(1, int(max_traces))

    @property
    def path(self) -> Path:
        return self._path

    def append(self, traces: Iterable[Mapping[str, Any]]) -> int:
        """Append serialised traces, trimming to the newest ``max_traces``.

        Takes a batch because the sink buffers: one file rewrite per flush rather
        than one per trace keeps the cost off whatever produced them.
        """
        incoming = [dict(trace) for trace in traces if isinstance(trace, Mapping)]
        if not incoming:
            return 0
        try:
            payload = self._load()
            records = payload["traces"]
            records.extend(incoming)
            # Order by time so a trim keeps the newest regardless of arrival order.
            records.sort(key=lambda item: float(item.get("ts") or 0.0))
            if len(records) > self._max:
                del records[: len(records) - self._max]
            self._write(payload)
            return len(incoming)
        except (OSError, TypeError, ValueError):
            logger.debug("evolution trace store: append failed", exc_info=True)
            return 0

    def list_traces(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """Return newest traces first."""
        records = self._load()["traces"]
        records.sort(key=lambda item: float(item.get("ts") or 0.0), reverse=True)
        return records if limit <= 0 else records[:limit]

    def count(self) -> int:
        return len(self._load()["traces"])

    # ── file access ───────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "traces": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                traces = data.get("traces")
                if isinstance(traces, list):
                    return {
                        "version": int(data.get("version") or 1),
                        "traces": [dict(t) for t in traces if isinstance(t, Mapping)],
                    }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.debug("evolution trace store: unreadable, treating as empty", exc_info=True)
        return {"version": 1, "traces": []}

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


__all__ = ["DEFAULT_MAX_TRACES", "JsonEvolutionTraceStore"]
