"""What the co-evolution sweep needs to see, collected where it is produced.

The cold-path sweep verifies effects, drains quarantine candidates and scans for
residue — but the facts it needs are produced in three different layers: the engine
resolves requirements, the self-management tools install artifacts, and the usage
sink sees every tool outcome. Having the CLI context reach down for those would
invert the dependency (engine must not know about the CLI), so producers write here
and the sweep reads here.

Deliberately mirrors ``telemetry/evolution_tap``: a process-level accessor, a
bounded buffer, and writes that never raise. Two properties matter:

* **Bounded.** Every buffer is a ``deque`` with a cap, because governance state must
  not grow with session length. Losing the oldest observation degrades a later sweep;
  an unbounded buffer degrades the process.
* **Hot-path safe.** ``record_tool_outcome`` is a dict lookup and a deque append. It
  performs no I/O and never awaits, so it is safe to call from the tool-outcome sink.

**A verification needs three things**: the requirement (which carries the declared
``expected_effect``), the plugin that was selected to serve it, and what was actually
observed. The binding is built from resolutions — a resolution says "this plugin was
chosen for this requirement" — and completed when that plugin's tool reports an
outcome. Only plugins the system *acquired* are bound, because verifying a
hand-installed tool against a world-model expectation is not meaningful.

**Known limitation, by design rather than omission:** an outcome carries an observed
effect only when the producer declares one. Without it, a *successful* call verifies
as ``unverifiable`` (we genuinely do not know whether the effect landed) while a
*failed* call still refutes. So today this channel can refute an acquisition but not
confirm one; confirming requires tools to report their effect.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Mapping, Sequence

from leapflow.domain.capability_requirement import CapabilityRequirement

logger = logging.getLogger(__name__)

#: Caps chosen so a long session cannot grow governance state without bound.
MAX_RESOLUTIONS = 64
MAX_VERIFICATIONS = 32
MAX_ACQUIRED = 64


class CoevolutionObservations:
    """Bounded, process-level collection point for co-evolution facts."""

    def __init__(
        self,
        *,
        max_resolutions: int = MAX_RESOLUTIONS,
        max_verifications: int = MAX_VERIFICATIONS,
        max_acquired: int = MAX_ACQUIRED,
    ) -> None:
        self._resolutions: deque[dict[str, Any]] = deque(maxlen=max(1, max_resolutions))
        self._verifications: deque[
            tuple[CapabilityRequirement, dict[str, Any], str]
        ] = deque(maxlen=max(1, max_verifications))
        self._acquired: deque[str] = deque(maxlen=max(1, max_acquired))
        # plugin_id -> the requirement it was selected to serve.
        self._bindings: dict[str, CapabilityRequirement] = {}
        # plugin_id -> the workspaces whose traffic exercised it. Governance state is
        # process-global because plugins are, so a streak can be driven by one
        # workspace and disable a plugin another one was using. That is consistent with
        # how trust already works -- a plugin that keeps failing is broken as *code* --
        # but it must not be invisible, so the contributing workspaces travel with the
        # decision and appear in its trace.
        self._workspaces: dict[str, set[str]] = {}

    # ── producers ──────────────────────────────────────────────────────────

    def record_acquisition(self, plugin_id: str) -> None:
        """Note that self-evolution installed this plugin."""
        plugin_id = str(plugin_id or "")
        if plugin_id and plugin_id not in self._acquired:
            self._acquired.append(plugin_id)

    def record_resolution(
        self,
        *,
        requirement: CapabilityRequirement | None = None,
        selected_plugin: str = "",
        exclusions: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Record one resolution outcome, and bind it if it chose an acquired plugin."""
        selected = str(selected_plugin or "")
        self._resolutions.append(
            {
                "selected_plugin": selected,
                "exclusions": {
                    str(pid): [str(r) for r in reasons]
                    for pid, reasons in dict(exclusions or {}).items()
                },
            }
        )
        # Only acquired plugins are bound: verifying a hand-installed tool against a
        # world-model expectation would be measuring the wrong thing.
        if requirement is not None and selected and selected in self._acquired:
            self._bindings[selected] = requirement

    def record_tool_outcome(
        self,
        plugin_id: str,
        tool_name: str,
        ok: bool,
        *,
        observed_effect: str = "",
        workspace: str = "",
    ) -> None:
        """Hot-path safe: pair an outcome with its bound requirement, if any.

        ``workspace`` is recorded for attribution only. It never changes a verdict --
        the same plugin id means the same code regardless of who called it -- but it
        makes a cross-workspace quarantine auditable instead of mysterious.
        """
        plugin_id = str(plugin_id or "")
        if plugin_id and workspace:
            self._workspaces.setdefault(plugin_id, set()).add(str(workspace))
        requirement = self._bindings.get(plugin_id)
        if requirement is None:
            return
        self._verifications.append(
            (
                requirement,
                {"ok": bool(ok), "observed_effect": str(observed_effect or ""),
                 "tool_name": str(tool_name or "")},
                plugin_id,
            )
        )

    # ── consumer (the sweep) ───────────────────────────────────────────────

    def drain_verifications(
        self,
    ) -> tuple[tuple[CapabilityRequirement, dict[str, Any], str], ...]:
        """Take the pending verifications, clearing them.

        Draining rather than reading keeps a sweep from re-verifying an outcome it
        already governed, which would double-count trust.
        """
        drained = tuple(self._verifications)
        self._verifications.clear()
        return drained

    def resolutions(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._resolutions)

    def acquired_plugin_ids(self) -> tuple[str, ...]:
        return tuple(self._acquired)

    def contributing_workspaces(self, plugin_id: str) -> tuple[str, ...]:
        """Which workspaces exercised this plugin, for governance attribution."""
        return tuple(sorted(self._workspaces.get(str(plugin_id or ""), ())))

    def stats(self) -> dict[str, int]:
        return {
            "resolutions": len(self._resolutions),
            "pending_verifications": len(self._verifications),
            "acquired": len(self._acquired),
            "bindings": len(self._bindings),
        }

    def reset(self) -> None:
        self._resolutions.clear()
        self._verifications.clear()
        self._acquired.clear()
        self._bindings.clear()
        self._workspaces.clear()


_OBSERVATIONS: CoevolutionObservations | None = None
_QUARANTINE_TRACKER: Any = None


def current_quarantine_tracker() -> Any:
    """The process-level consecutive-failure tracker, created on first use.

    Process-level for the same reason the registry is: plugins are shared by every
    session, so a failure streak belongs to the plugin rather than to whoever happened
    to call it. It also has to be *one* instance -- the tool-outcome sink increments it
    and the cold-path sweep drains it, and two instances would mean the sweep draining
    a tracker nobody ever fed.
    """
    global _QUARANTINE_TRACKER
    if _QUARANTINE_TRACKER is None:
        from leapflow.learning.outcome_governance_feed import QuarantineCandidateTracker

        _QUARANTINE_TRACKER = QuarantineCandidateTracker()
    return _QUARANTINE_TRACKER


def install_quarantine_tracker(tracker: Any) -> None:
    """Replace the process tracker. ``None`` resets it; used by tests."""
    global _QUARANTINE_TRACKER
    _QUARANTINE_TRACKER = tracker


def current_observations() -> CoevolutionObservations:
    """The process-level buffer, created on first use."""
    global _OBSERVATIONS
    if _OBSERVATIONS is None:
        _OBSERVATIONS = CoevolutionObservations()
    return _OBSERVATIONS


def install_observations(buffer: CoevolutionObservations | None) -> None:
    """Replace the process buffer. ``None`` resets it; used by tests."""
    global _OBSERVATIONS
    _OBSERVATIONS = buffer


def record_acquisition(plugin_id: str) -> None:
    """Module-level convenience for producers; never raises."""
    try:
        current_observations().record_acquisition(plugin_id)
    except Exception:  # noqa: BLE001 - a producer must never fail on bookkeeping
        logger.debug("coevolution observations: acquisition not recorded", exc_info=True)


def record_resolution(**kwargs: Any) -> None:
    """Module-level convenience for producers; never raises."""
    try:
        current_observations().record_resolution(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("coevolution observations: resolution not recorded", exc_info=True)


def record_tool_outcome(plugin_id: str, tool_name: str, ok: bool, **kwargs: Any) -> None:
    """Module-level convenience for the hot path; never raises."""
    try:
        current_observations().record_tool_outcome(plugin_id, tool_name, ok, **kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("coevolution observations: outcome not recorded", exc_info=True)


__all__ = [
    "MAX_ACQUIRED",
    "MAX_RESOLUTIONS",
    "MAX_VERIFICATIONS",
    "CoevolutionObservations",
    "current_observations",
    "current_quarantine_tracker",
    "install_observations",
    "install_quarantine_tracker",
    "record_acquisition",
    "record_resolution",
    "record_tool_outcome",
]
