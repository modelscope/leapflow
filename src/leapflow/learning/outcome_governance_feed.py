"""Feed execution outcomes into lifecycle governance without touching the hot path.

Trust already accrues in production: ``TurnUsageTracker.record_tool_call`` forwards
to ``PluginUsageTracker.record``, which resolves tool -> plugin and calls
``record_success`` / ``record_failure``. So promotion and demotion work on live
traffic. **Quarantine does not** -- ``LifecycleGovernor`` had no feed, so a plugin
could fail indefinitely without ever being disabled.

The wiring has to respect two constraints that pull against each other:

* ``PluginUsageTracker.record`` is documented as a ``<1us`` hot path, and plugin
  governance is required to be cold-path -- a governance feature that measurably
  slows an ordinary turn is a defect in the feature.
* Governance is async and I/O-bound: it writes lifecycle status, appends an outcome
  record, and may call the lifecycle actor to disable a plugin.

So this splits in two. On the hot path :class:`QuarantineCandidateTracker` keeps one
integer per plugin and does nothing else -- no I/O, no awaiting, no allocation
beyond a dict entry. Crossing the threshold only *marks* a candidate. The actual
governance runs later, on a cold path, via :func:`drain_quarantine_candidates`.

The consequence is explicit and worth stating: quarantine is **deferred**, not
immediate. A plugin that crosses the threshold mid-session keeps serving until the
next drain. That is the deliberate trade for not putting I/O in the hot path; the
per-turn trust demotion still applies immediately, so a failing plugin is already
being down-ranked by the resolver while it waits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuarantineCandidate:
    """A plugin whose consecutive-failure streak crossed the threshold."""

    plugin_id: str
    tool_name: str
    failure_streak: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "tool_name": self.tool_name,
            "failure_streak": self.failure_streak,
        }


class QuarantineCandidateTracker:
    """Hot-path-safe consecutive-failure counter.

    ``record`` is called once per tool execution, so it must stay trivial: one dict
    lookup and an integer update. It never performs I/O and never awaits.
    """

    def __init__(self, *, quarantine_after: int = 3) -> None:
        self._threshold = max(1, int(quarantine_after))
        self._streaks: dict[str, int] = {}
        self._candidates: dict[str, QuarantineCandidate] = {}

    @property
    def threshold(self) -> int:
        return self._threshold

    def record(self, plugin_id: str, tool_name: str, ok: bool) -> bool:
        """Note one outcome. Returns True when this crossed the threshold.

        A success resets the streak, which is what makes an intermittently-failing
        plugin survive: only *consecutive* failures quarantine.
        """
        if not plugin_id:
            return False
        if ok:
            self._streaks.pop(plugin_id, None)
            self._candidates.pop(plugin_id, None)
            return False
        streak = self._streaks.get(plugin_id, 0) + 1
        self._streaks[plugin_id] = streak
        if streak < self._threshold:
            return False
        self._candidates[plugin_id] = QuarantineCandidate(plugin_id, tool_name, streak)
        return True

    def candidates(self) -> tuple[QuarantineCandidate, ...]:
        return tuple(self._candidates.values())

    def clear(self, plugin_id: str = "") -> None:
        """Drop one candidate, or all of them."""
        if plugin_id:
            self._candidates.pop(plugin_id, None)
            self._streaks.pop(plugin_id, None)
            return
        self._candidates.clear()
        self._streaks.clear()

    def pending(self) -> int:
        return len(self._candidates)


async def drain_quarantine_candidates(
    tracker: QuarantineCandidateTracker,
    governor: Any,
    *,
    proposal_ids: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run lifecycle governance for every marked candidate. Cold path only.

    Each candidate is reported to ``LifecycleGovernor.record_outcome`` as a failure,
    which applies trust, updates lifecycle status and (at or past the governor's own
    threshold) disables the plugin. Candidates are cleared as they are handled, so a
    second drain is a no-op rather than a double punishment.

    Failures are contained per candidate: a store error on one plugin must not stop
    the others from being governed.
    """
    handled: list[dict[str, Any]] = []
    ids = dict(proposal_ids or {})
    for candidate in tracker.candidates():
        try:
            result = await governor.record_outcome(
                proposal_id=ids.get(candidate.plugin_id, ""),
                plugin_id=candidate.plugin_id,
                tool_name=candidate.tool_name,
                ok=False,
                failure_class="consecutive_failures",
            )
            handled.append(
                {
                    "plugin_id": candidate.plugin_id,
                    "failure_streak": candidate.failure_streak,
                    "action": getattr(result, "action", ""),
                    "trust_level": getattr(result, "trust_level", ""),
                }
            )
        except Exception:  # noqa: BLE001 - governance must not break the drain
            logger.debug(
                "quarantine drain failed for %s", candidate.plugin_id, exc_info=True
            )
            continue
        finally:
            tracker.clear(candidate.plugin_id)
    return tuple(handled)


#: Empty means unrestricted: any requirement origin may drive acquisition, which is
#: the shipped behaviour. Populating it restricts authority to the listed origins.
DEFAULT_AUTHORISING_ORIGINS: tuple[str, ...] = ()


def origin_may_authorise(origin: str, authorising_origins: Sequence[str] | None) -> bool:
    """Whether a requirement of this origin may drive an acquisition.

    This is the executable form of "all self-evolution's first driver is the world
    model": set ``authorising_origins = ("world_model",)`` and a requirement raised
    by any other path can still be *recorded and resolved*, but can no longer
    authorise acquiring new code.

    Kept permissive by default so enabling it is a deliberate operator decision
    rather than a silent behaviour change.
    """
    if not authorising_origins:
        return True
    return str(origin) in {str(item) for item in authorising_origins}


def filter_authorised(
    requirements: Sequence[Any], authorising_origins: Sequence[str] | None
) -> tuple[Any, ...]:
    """Keep only the requirements permitted to drive acquisition."""
    return tuple(
        requirement
        for requirement in requirements
        if origin_may_authorise(getattr(requirement, "origin", ""), authorising_origins)
    )


__all__ = [
    "DEFAULT_AUTHORISING_ORIGINS",
    "QuarantineCandidate",
    "QuarantineCandidateTracker",
    "drain_quarantine_candidates",
    "filter_authorised",
    "origin_may_authorise",
]
