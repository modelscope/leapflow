"""Monitor producer for framework self-evolution transparency.

Domain: ``framework_evolution``. Answers two questions the existing views cannot:

**What is the framework right now?** The plugin roster with each plugin's fiber
state and trust, the capability topology, and the tool-name conflicts. Read live
from ``get_registry()`` and the trust ledger every cycle, never from
documentation or a cached catalog -- reporting LeapFlow's own composition from
anything but the running registry is how a board ends up describing capabilities
the process does not have.

**Is the evolution pipeline actually flowing?** ``_reachability()`` walks the
pipeline segment by segment and reports the runtime evidence for each. This
exists because a whole tier of trust/probation/quarantine machinery was once
reachable from nothing in production, and no test or view could see it -- it was
found by auditing which modules had no references outside themselves. A segment
with no evidence is reported as ``no_evidence`` with the next step to take, and a
segment whose source cannot be read is ``unverifiable``. Neither is ever reported
as working: the presence of a module is not evidence that anything calls it.

This producer owns no causal history of its own: episodes are rebuilt on demand by
``EvolutionLedger`` from the decision and observation records the system already
keeps, so the timeline costs no probe and no new schema. When those records cannot
be read, ``episodes`` is empty and ``degraded`` says why -- which is what lets the
panel stay useful on a profile that has never evolved anything.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from leapflow.domain.evolution_trace import ABORTED, REOPENED, RESOLVED, STILL_OPEN
from leapflow.monitor.types import Evidence, Finding, ProducerContext, Severity, SuggestedAction

logger = logging.getLogger(__name__)

# Payload bounds. The reply travels as one JSON-RPC frame (the coordinator trims
# oldest-first at the transport), so a producer that does not bound its own
# payload pushes still-current findings out of the batch.
_MAX_ROSTER = 60
_MAX_CONFLICTS = 40
_MAX_TOPOLOGY_NODES = 240
_MAX_TOPOLOGY_EDGES = 400
_MAX_CAPABILITY_MAP = 200
_MAX_EPISODES = 20
_MAX_TRACES = 40

# Trust vocabulary. ``DRAFT`` alone cannot distinguish "new and unproven" from
# "permanently disqualified by an internal defect", so the semantic class is
# reported alongside the level.
_TRUST_CLASS = {
    "DRAFT": "new_unproven",
    "CANDIDATE": "accruing",
    "VERIFIED": "accruing",
    "PRODUCTION": "trusted",
}

#: Rendered as a table cell, so it must be a translatable key like every other
#: closed vocabulary here. ``frozen`` is lower-case for that reason -- the trust
#: *level* is an upper-case enum name, but this is a semantic class, and mixing the
#: two casings in one column made it read like two different kinds of value.
_TRUST_CLASS_FROZEN = "frozen"

#: Booleans reach the board as untranslatable ``true``/``false``: the client's value
#: translator passes non-strings straight through. Closed vocabularies are emitted as
#: keys instead, so every locale renders a word rather than a JSON literal.
_YES = "yes"
_NO = "no"

# ── Localisation boundary ────────────────────────────────────────────────
#
# The client translates every string table cell through a dictionary with a raw
# fallback, so what this producer emits decides what can be localised. The line is
# drawn deliberately:
#
# * **Closed vocabularies and segment labels are keys** -- statuses, trust classes,
#   fiber states, gap closures, verification tiers, drivers, actions, yes/no. Each
#   has a translation in every shipped locale, asserted by
#   ``test_every_closed_vocabulary_the_payload_emits_is_translated``.
# * **Operator instructions stay in English.** ``evidence`` and ``next_step`` embed
#   literal commands (``leap config set ...``) and symbol names that must not be
#   translated to remain runnable. A half-translated sentence wrapped around an
#   English command is harder to act on than a consistent English one, so these are
#   left whole rather than fragmented into interpolated keys.
# * **Identifiers are never translated** -- plugin ids, tool names, capability names.
#   They are names, not words.


#: Plugin provenance. The distinction evolution actually cares about: a built-in
#: plugin shipped with the framework, a self-acquired one the framework installed
#: into the profile itself. Only the second kind is evidence of evolution.
_BUILT_IN = "built_in"
_SELF_ACQUIRED = "self_acquired"


def _percent(value: Any) -> str:
    """Render a 0..1 model self-report as a percentage, or empty when absent.

    A confidence is a claim, never a permission, so it is shown for prioritisation
    only -- and shown in a unit a reader cannot misread as a raw score.
    """
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{max(0.0, min(1.0, ratio)) * 100:.0f}%"

# Reachability statuses. ``no_evidence`` is deliberately distinct from
# ``unverifiable``: the first means nothing was observed (idle, or unwired), the
# second means the source could not be read at all.
WIRED = "wired"
NO_EVIDENCE = "no_evidence"
UNVERIFIABLE = "unverifiable"
NOT_ADMITTED = "not_admitted"


class EvolutionProducer:
    """Emit one framework-evolution snapshot per cycle."""

    domain = "framework_evolution"

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        """Return a single Finding describing the framework and its pipeline.

        Never raises: a transparency panel that fails takes the whole board page
        with it, and the failure it would report is its own.
        """
        try:
            payload = self._build_payload(ctx)
        except Exception:  # noqa: BLE001 - observability must not break the cycle
            # Logged at warning, not debug: every read inside is individually
            # guarded and degrades to an ``unverifiable`` row, so reaching this
            # handler means a defect in *this* producer rather than an unreadable
            # source. Swallowing it keeps the cycle alive; hiding it at debug
            # would leave the panel silently absent with no reason recorded.
            logger.warning("evolution producer: snapshot failed", exc_info=True)
            return ()

        severity = self._severity(payload)
        return (
            Finding(
                watch_id=ctx.spec.watch_id or self.domain,
                domain=self.domain,
                title="Framework evolution",
                summary=payload["summary"]["headline"],
                severity=severity,
                ts=payload["observed_at"],
                tags=("framework_evolution", "self_evolution"),
                evidence=self._evidence(payload),
                suggested_actions=self._actions(payload),
                # Content fingerprint, not a timestamp: the executor skips a
                # finding whose dedup key already exists, so a key that changed
                # every cycle would grow the table without adding information,
                # and one built from the clock would defeat dedup entirely. An
                # unchanged framework keeps the previous finding, whose content
                # is identical and therefore still accurate.
                dedup_key=f"evolution:{self._fingerprint(payload)}",
                payload=payload,
            ),
        )

    # ── payload assembly ──────────────────────────────────────────────────

    def __init__(self) -> None:
        # Previous cycle's fiber states, for the transition diff below. Held on the
        # producer because it is registered once and lives as long as the daemon; a
        # fresh instance simply has no baseline and reports no transitions, which is
        # the correct answer for the first cycle after a restart.
        self._last_fibers: dict[str, str] = {}

    def _fiber_transitions(
        self, fibers: Mapping[str, str], *, readable: bool
    ) -> list[dict[str, Any]]:
        """Fiber state changes since the previous cycle.

        Derived by diffing snapshots rather than by probing the state machine, and
        that is the point: a fiber transition does not bump the registry version, so
        the ACT probe cannot see it, and putting a probe on every transition method
        would add an observability dependency to the lifecycle object for a fact the
        presentation layer can reconstruct for free.

        What this recovers is the retry path -- ``LOADING -> FAILED -> LOADING`` --
        which is invisible everywhere else: the registry never changed, so no version
        moved, and by the time a poll runs the fiber is usually back to ``active``.
        Sampling means a transition completed entirely between two cycles is missed;
        that is a stated limit of the diff, not a defect to work around, and the
        alternative was instrumenting the state machine.

        An unreadable registry returns nothing **and leaves the baseline intact**.
        Diffing against an empty snapshot would report every plugin as disposed, and
        replacing the baseline with it would then report every plugin as newly
        appeared on the next successful cycle -- two fabricated mass events from one
        transient read failure.
        """
        if not readable:
            return []
        previous, self._last_fibers = self._last_fibers, dict(fibers)
        if not previous:
            return []
        rows: list[dict[str, Any]] = []
        for plugin_id, state in sorted(fibers.items()):
            was = previous.get(plugin_id)
            if was is None:
                rows.append({"plugin_id": plugin_id, "from": "", "to": state, "kind": "appeared"})
            elif was != state:
                rows.append({"plugin_id": plugin_id, "from": was, "to": state, "kind": "moved"})
        for plugin_id in sorted(set(previous) - set(fibers)):
            rows.append(
                {"plugin_id": plugin_id, "from": previous[plugin_id], "to": "", "kind": "gone"}
            )
        return rows

    def _build_payload(self, ctx: ProducerContext) -> dict[str, Any]:
        snapshot = self._live_registry_snapshot()
        reachability = self._reachability(snapshot)
        rebuilt = self._episodes(ctx)
        # ``None`` means the history could not be rebuilt; ``()`` means there is
        # genuinely none. Collapsing the two would report a local defect as an
        # absence of data -- the same conflation the reachability rows exist to
        # prevent, and it would be inconsistent for this panel to commit it.
        episodes: tuple[Any, ...] = rebuilt or ()
        payload: dict[str, Any] = {
            "observed_at": float(getattr(ctx, "now", 0.0) or 0.0),
            "roster": snapshot["roster"],
            "topology": snapshot["topology"],
            # Two renderer-shaped projections of the same ownership relation. The
            # graph in ``topology`` is the general form, but the shipped
            # ``EntityGraph`` renderer is a badge cloud reading ``props.data`` as a
            # flat list of items with a ``name`` -- binding it a nodes/edges
            # mapping renders nothing at all and reports no fault. So the view
            # binds these instead, and ``topology`` stays for a real graph
            # renderer to consume later.
            "capability_map": snapshot["capability_map"],
            "capability_badges": snapshot["capability_badges"],
            "conflicts": snapshot["conflicts"],
            "reachability": reachability,
            # Distribution charts rather than another table: "how much of the roster
            # is trusted" and "how much of the pipeline is wired" are single-glance
            # questions that a reader should not have to answer by counting rows.
            # Both derive from fields already in the fingerprint, so neither can
            # freeze independently of the table it summarises.
            "trust_mix": self._trust_mix(snapshot["roster"]),
            "reachability_mix": self._reachability_mix(reachability),
            # Q1/Q5: growth versus inventory, and artifacts the framework acquired
            # and then never used. Both derive from roster fields already in the
            # fingerprint, so neither can freeze independently of the roster.
            "provenance_mix": self._provenance_mix(snapshot["roster"]),
            "reclaim_candidates": self._reclaim_candidates(snapshot["roster"]),
            "fiber_transitions": self._fiber_transitions(
                snapshot["fibers"], readable=bool(snapshot.get("registry_readable"))
            ),
            "episodes": [episode.to_dict() for episode in episodes],
            # Same reason as the topology projections: the Timeline renderer reads
            # ``props.data`` as a flat list of ``{title, summary, severity}``.
            "timeline": self._timeline(episodes),
            "mutation_matrix": self._mutation_matrix(episodes),
            "degraded": not episodes,
        }
        traces = self._recent_traces()
        payload["traces"] = traces
        payload["trace_feed"] = self._trace_feed(traces)
        payload["unadmitted"] = self._unadmitted(traces)
        if rebuilt is None:
            payload["degraded_kind"] = UNVERIFIABLE
            payload["degraded_reason"] = (
                "The causal history could not be rebuilt: the decision or observation "
                "records could not be read. This is a fault to investigate, not an "
                "absence of activity. The live snapshot and pipeline reachability below "
                "are unaffected."
            )
        elif not episodes:
            payload["degraded_kind"] = NO_EVIDENCE
            payload["degraded_reason"] = (
                "No capability decision has been recorded yet, so there is no causal "
                "history to rebuild. The live snapshot and pipeline reachability below "
                "are unaffected."
            )
        payload["summary"] = self._summary(snapshot, reachability, episodes, traces)
        return payload

    # ── live traces (facts no store retains) ─────────────────────────────

    def _recent_traces(self) -> list[dict[str, Any]]:
        """Flush the probe buffer, then read the newest traces back.

        Flushing here rather than on a separate schedule is what keeps the panel and
        the file consistent: this producer is the only consumer, and it runs on the
        monitor tick, which is the cold path the tap's contract requires. Probe sites
        therefore only ever buffer.

        Empty is the normal state -- no sink is installed outside the daemon, and a
        framework that has not mutated has nothing to report. Distinguished from a
        read failure only in the log, because unlike the episode history there is no
        "records exist but are unreadable" case to mistake it for: the store treats a
        corrupt file as empty by design.
        """
        try:
            from leapflow.telemetry.evolution_tap import current_sink

            sink = current_sink()
            if sink is not None and hasattr(sink, "flush"):
                sink.flush()
        except Exception:  # noqa: BLE001 - a failed flush costs freshness, not the cycle
            logger.debug("evolution producer: trace flush failed", exc_info=True)
        store = self._json_store(
            "evolution_traces_path", "evolution_trace_store", "JsonEvolutionTraceStore"
        )
        if store is None:
            return []
        try:
            return [
                dict(row)
                for row in store.list_traces(limit=_MAX_TRACES)
                if isinstance(row, Mapping)
            ]
        except Exception:  # noqa: BLE001
            logger.debug("evolution producer: traces unreadable", exc_info=True)
            return []

    @staticmethod
    def _trace_feed(traces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Flatten traces into the shape the Timeline renderer reads.

        Composition-phase registry traces are excluded: every daemon start replays
        the initial plugin load, and letting that through would bury the rare real
        mutation under a boot log. The registry marks the difference itself, so this
        is a filter on a declared fact rather than a guess about the kind's name.
        """
        rows: list[dict[str, Any]] = []
        for trace in traces:
            detail = dict(trace.get("detail") or {})
            kind = str(trace.get("kind") or "")
            if detail.get("phase") == "composition":
                continue
            severity = "info"
            if kind == "trust_frozen":
                severity = "alert"
            elif kind in ("registry_plugin_unregistered", "registry_tools_unregistered"):
                severity = "notable"
            elif detail.get("not_admitted_reason"):
                severity = "notable"
            rows.append(
                {
                    "title": f"{str(trace.get('stage') or '').upper()} · {kind}",
                    "summary": str(trace.get("summary") or ""),
                    "severity": severity,
                }
            )
        return rows

    @staticmethod
    def _unadmitted(traces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Teacher proposals that never entered the pipeline.

        The one thing on this board that exists in no store at all: an intent that
        was proposed and not admitted writes no observation, so without this the
        board would show an idle pipeline while the world model proposed on every
        session -- indistinguishable from a model with nothing to say.
        """
        rows: list[dict[str, Any]] = []
        for trace in traces:
            detail = dict(trace.get("detail") or {})
            reason = str(detail.get("not_admitted_reason") or "")
            if not reason:
                continue
            for intent in detail.get("intents") or []:
                if not isinstance(intent, Mapping):
                    continue
                rows.append(
                    {
                        "capability": str(intent.get("capability") or ""),
                        "hypothesis": str(intent.get("hypothesis") or ""),
                        # Formatted here, not in the view: a bare 0.8 in a column
                        # headed "Confidence" reads as a score out of some unstated
                        # maximum.
                        "confidence": _percent(intent.get("confidence")),
                        "reason": reason,
                    }
                )
        return rows

    @staticmethod
    def _provenance_mix(roster: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Self-acquired versus built-in, in that order.

        Self-acquired first because it is the number the board exists to report: a
        framework that has grown has a non-zero bar here, and one that has not shows
        a single built-in bar no matter how many plugins it ships with.
        """
        counts: dict[str, int] = {}
        for row in roster:
            key = str(row.get("provenance") or _BUILT_IN)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"label": name, "value": counts[name]}
            for name in (_SELF_ACQUIRED, _BUILT_IN)
            if counts.get(name)
        ]

    @staticmethod
    def _reclaim_candidates(roster: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Self-acquired artifacts holding a tool name without earning it.

        The recorded LF-10 case: a generated plugin blocked by the risk ceiling keeps
        its registration forever, occupying a tool name and padding the capability
        list, because nothing reclaims an artifact no requirement can select. Listing
        them is not the reclamation, but it is the first time the set has been
        nameable.
        """
        return [
            {
                "plugin_id": row.get("plugin_id"),
                "trust_level": row.get("trust_level"),
                "selectable": row.get("selectable"),
                "ever_used": row.get("ever_used"),
                "tool_count": row.get("tool_count"),
            }
            for row in roster
            if str(row.get("reclaimable")) == _YES
        ]

    @staticmethod
    def _trust_mix(roster: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Roster counted by trust class, in earned order.

        Fixed order rather than sorted by count, so the shape of the bars means the
        same thing on every visit. Empty classes are dropped -- a zero bar carries no
        information and only costs width.
        """
        counts: dict[str, int] = {}
        for row in roster:
            counts[str(row.get("trust_class") or "unverified")] = (
                counts.get(str(row.get("trust_class") or "unverified"), 0) + 1
            )
        order = (_TRUST_CLASS_FROZEN, "new_unproven", "accruing", "trusted", "unverified")
        return [{"label": name, "value": counts[name]} for name in order if counts.get(name)]

    @staticmethod
    def _reachability_mix(reachability: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Pipeline segments counted by status, worst first.

        The one number this board exists to make visible: how much of the evolution
        machinery shows runtime evidence versus how much only exists.
        """
        counts: dict[str, int] = {}
        for row in reachability:
            counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
        order = (WIRED, NO_EVIDENCE, NOT_ADMITTED, UNVERIFIABLE)
        return [{"label": name, "value": counts[name]} for name in order if counts.get(name)]

    def _episodes(self, ctx: ProducerContext) -> tuple[Any, ...] | None:
        """Rebuild recent episodes from existing records.

        Returns ``None`` when the history could not be rebuilt at all, and ``()``
        when it was rebuilt and is genuinely empty. The caller depends on that
        difference: "could not look" and "nothing to see" are different answers,
        and only one of them is a fault.

        The ledger reads the plan and observation stores, so it is resolved per
        cycle for the same reason the stores are: the profile layout is bound
        during deferred daemon initialisation.
        """
        plans = self._json_store(
            "capability_plans_path", "capability_plan_store", "JsonCapabilityPlanStore"
        )
        if plans is None:
            return None
        observations = self._json_store(
            "capability_observations_path",
            "capability_observation_store",
            "JsonCapabilityObservationStore",
        )
        trust, _usage = self._trust_and_usage()
        try:
            from leapflow.evolution import EvolutionLedger

            ledger = EvolutionLedger(
                plan_store=plans, observation_store=observations, trust_ledger=trust
            )
            return tuple(
                ledger.recent_episodes(
                    limit=_MAX_EPISODES, now=float(getattr(ctx, "now", 0.0) or 0.0)
                )
            )
        except Exception:  # noqa: BLE001 - a missing timeline degrades the panel, not the cycle
            logger.debug("evolution producer: ledger unavailable", exc_info=True)
            return None

    @staticmethod
    def _timeline(episodes: Sequence[Any]) -> list[dict[str, Any]]:
        """Project episodes into the flat shape the Timeline renderer reads.

        Severity per row rather than one for the panel: a regression sitting among
        ordinary episodes is the row a reader must not scroll past.
        """
        rows: list[dict[str, Any]] = []
        for episode in episodes:
            driver = episode.driver or "unknown"
            action = episode.mutation_action or "none"
            target = f" {episode.plugin_id}" if episode.plugin_id else ""
            severity = "info"
            if episode.gap_closure == REOPENED:
                severity = "alert"
            elif episode.gap_closure == STILL_OPEN or episode.status == ABORTED:
                severity = "notable"
            summary = episode.outcome or episode.status
            if episode.capability:
                summary = f"{episode.capability} · {summary}"
            if episode.hypothesis:
                summary = f"{summary} · {episode.hypothesis}"
            rows.append(
                {
                    "title": f"{driver} → {action}{target}",
                    "summary": summary,
                    "severity": severity,
                }
            )
        return rows

    @staticmethod
    def _mutation_matrix(episodes: Sequence[Any]) -> list[dict[str, Any]]:
        """One row per episode: what triggered it, what was decided, what came of it.

        The two proposal vocabularies stay in separate columns. They answer
        different questions -- "should a human accept this" versus "where is this
        capability in its journey" -- and merging them is the mistake the code
        upstream explicitly warns against.
        """
        return [
            {
                "driver": episode.driver or "unknown",
                "capability": episode.capability,
                "policy_action": episode.policy_action,
                "autonomy_level": episode.autonomy_level,
                "mutation_action": episode.mutation_action,
                "registry_delta": (
                    f"{episode.registry_before} → {episode.registry_after}"
                    if episode.registry_after >= 0
                    else ""
                ),
                "lifecycle_status": episode.lifecycle_status,
                "gap_closure": episode.gap_closure,
                "verification_tier": episode.verification_tier,
                "trust_now": episode.trust_now,
                # ``outcome`` is deliberately absent from the rendered columns: it is
                # composed English prose, so it can never localise, and it restates
                # what gap_closure + mutation_action + verification_tier already say
                # in vocabularies the client can translate. It stays on the episode
                # for the timeline, where narrative text is expected.
            }
            for episode in episodes
        ]

    def _live_registry_snapshot(self) -> dict[str, Any]:
        """Read the roster, topology and conflicts from the running registry.

        Every value here is a live read. When the registry cannot be reached the
        snapshot reports ``registry_readable=False`` rather than an empty roster,
        because "no plugins" and "could not look" are different answers and only
        one of them is ever true of a running daemon.
        """
        registry = None
        try:
            from leapflow.plugins import get_registry

            registry = get_registry()
        except Exception:  # noqa: BLE001 - degraded snapshot, not a failure
            logger.debug("evolution producer: registry unavailable", exc_info=True)

        if registry is None:
            return {
                "registry_readable": False,
                "registry_version": -1,
                "roster": [],
                # Empty rather than absent: an unreadable registry must not be read
                # as "every fiber disappeared", which is what a missing key would
                # look like to the transition diff.
                "fibers": {},
                "topology": {"nodes": [], "edges": []},
                "capability_map": [],
                "capability_badges": [],
                "conflicts": [],
            }

        trust, usage = self._trust_and_usage()
        fibers = self._fiber_states()
        owners: Mapping[str, str] = dict(getattr(registry, "tool_owners", {}) or {})
        handlers = set(dict(getattr(registry, "tool_handlers", {}) or {}))
        plugins: Mapping[str, Any] = dict(getattr(registry, "plugins", {}) or {})

        roster: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        capability_map: list[dict[str, Any]] = []
        seen_capabilities: set[str] = set()

        for plugin_id in sorted(plugins):
            plugin = plugins[plugin_id]
            tools = self._owned_tools(plugin_id, plugin, owners, handlers)
            roster.append(self._roster_row(plugin_id, tools, fibers, trust, usage))
            if len(nodes) < _MAX_TOPOLOGY_NODES:
                nodes.append({"id": f"plugin:{plugin_id}", "kind": "plugin", "label": plugin_id})
            for tool in tools:
                self._add_tool_topology(
                    plugin_id, tool, nodes, edges, capability_map, seen_capabilities
                )

        return {
            "registry_readable": True,
            "registry_version": int(getattr(registry, "version", -1) or -1),
            "roster": roster[:_MAX_ROSTER],
            "fibers": fibers,
            "topology": {"nodes": nodes, "edges": edges},
            "capability_map": capability_map[:_MAX_CAPABILITY_MAP],
            # ``name`` is the key the badge-cloud renderer reads.
            "capability_badges": [{"name": cap} for cap in sorted(seen_capabilities)],
            "conflicts": self._conflicts(registry),
        }

    @staticmethod
    def _owned_tools(
        plugin_id: str,
        plugin: Any,
        owners: Mapping[str, str],
        handlers: set[str],
    ) -> list[Any]:
        """Return the tools this plugin actually owns and that are dispatchable.

        Tool names are one global namespace arbitrated first-wins, so a plugin's
        declared tool list is not the same as the tools it owns. Filtering by
        ``tool_owners`` and by the live handler table is what makes the topology
        agree with what the model can actually call.
        """
        result: list[Any] = []
        for tool in getattr(plugin, "tools", []) or []:
            name = str(getattr(tool, "name", "") or "")
            if not name:
                continue
            if owners and owners.get(name) != plugin_id:
                continue
            if handlers and name not in handlers:
                continue
            result.append(tool)
        return result

    def _roster_row(
        self,
        plugin_id: str,
        tools: list[Any],
        fibers: Mapping[str, str],
        trust: Any,
        usage: Any,
    ) -> dict[str, Any]:
        level, trust_class, frozen = self._trust_of(plugin_id, trust)
        ever_used = self._ever_used(plugin_id, usage)
        selectable = not frozen
        provenance = self._provenance(plugin_id)
        return {
            "plugin_id": plugin_id,
            "fiber_state": fibers.get(plugin_id, "unknown"),
            "trust_level": level,
            "trust_class": trust_class,
            # Whether the framework acquired this itself. The whole board is about
            # evolution, and a built-in plugin is not evidence of any; separating the
            # two is what lets a reader see growth rather than inventory.
            "provenance": provenance,
            # A frozen plugin still reports DRAFT, and TrustScorer only *scores*
            # trust, so a frozen-but-registered plugin stays selectable unless
            # FrozenExclusionScorer is injected. Surfacing both columns is what
            # makes that window visible instead of implying it cannot happen.
            "selectable": _NO if frozen else _YES,
            # Deliberately a durable fact, not a live counter. Rates and call counts
            # change every cycle, and this finding dedups on a content fingerprint --
            # a live metric would either churn a new row on every tick or (if left
            # out of the fingerprint) freeze on the board while looking current.
            # "Has this ever been selected" is the structural question evolution
            # actually asks. Per-plugin error rates belong to ``plugin_health``.
            "ever_used": _YES if ever_used else _NO,
            "tool_count": len(tools),
            # The reclamation case, decided here rather than in the view so one
            # definition serves the roster column, the candidate list and the count:
            # self-acquired, registered, and either unselectable or never once chosen.
            "reclaimable": _YES
            if (provenance == _SELF_ACQUIRED and not (selectable and ever_used))
            else _NO,
        }

    @staticmethod
    def _provenance(plugin_id: str) -> str:
        """Whether this plugin was shipped or acquired by the framework itself.

        Read from the profile's version store, which records a source snapshot for
        every plugin installed at runtime. That is a durable fact rather than an
        inference from the module path, and it survives a reload -- a plugin's
        in-memory identity says nothing about where it came from.

        Unreadable store means ``built_in``: claiming a plugin was self-acquired on
        no evidence would overstate how much the framework has evolved, which is the
        one direction this board must never exaggerate.
        """
        try:
            from leapflow.config import get_settings
            from leapflow.storage.plugin_version_store import PluginVersionStore

            layout = getattr(get_settings(), "profile_layout", None)
            versions_dir = getattr(layout, "plugin_versions_dir", None)
            if versions_dir is None:
                return _BUILT_IN
            active = PluginVersionStore(versions_dir).active(plugin_id)
            return _SELF_ACQUIRED if active else _BUILT_IN
        except Exception:  # noqa: BLE001 - provenance is a column, not a fault
            return _BUILT_IN

    @staticmethod
    def _trust_of(plugin_id: str, trust: Any) -> tuple[str, str, bool]:
        if trust is None:
            return "unverified", "unverified", False
        try:
            level = trust.level(plugin_id).name
        except Exception:  # noqa: BLE001 - one unreadable plugin must not blank the roster
            return "unverified", "unverified", False
        frozen = False
        is_frozen = getattr(trust, "is_frozen", None)
        if callable(is_frozen):
            try:
                frozen = bool(is_frozen(plugin_id))
            except Exception:  # noqa: BLE001
                frozen = False
        return level, (_TRUST_CLASS_FROZEN if frozen else _TRUST_CLASS.get(level, "unverified")), frozen

    @staticmethod
    def _ever_used(plugin_id: str, usage: Any) -> bool:
        """Whether this plugin has ever been selected. Unknown counts as not used."""
        if usage is None:
            return False
        try:
            stats = usage.stats_for_plugin(plugin_id)
        except Exception:  # noqa: BLE001 - one unreadable plugin must not blank the roster
            return False
        return bool(stats is not None and int(getattr(stats, "total_calls", 0) or 0) > 0)

    @staticmethod
    def _add_tool_topology(
        plugin_id: str,
        tool: Any,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        capability_map: list[dict[str, Any]],
        seen_capabilities: set[str],
    ) -> None:
        """Add ``plugin -owns-> tool -provides-> capability`` to the graph and table."""
        name = str(getattr(tool, "name", "") or "")
        if not name:
            return
        if len(nodes) < _MAX_TOPOLOGY_NODES:
            nodes.append({"id": f"tool:{name}", "kind": "tool", "label": name})
        if len(edges) < _MAX_TOPOLOGY_EDGES:
            edges.append({"source": f"plugin:{plugin_id}", "target": f"tool:{name}", "kind": "owns"})
        for capability in getattr(tool, "provides_capabilities", ()) or ():
            cap = str(capability or "")
            if not cap:
                continue
            if cap not in seen_capabilities and len(nodes) < _MAX_TOPOLOGY_NODES:
                nodes.append({"id": f"capability:{cap}", "kind": "capability", "label": cap})
            seen_capabilities.add(cap)
            if len(edges) < _MAX_TOPOLOGY_EDGES:
                edges.append(
                    {"source": f"tool:{name}", "target": f"capability:{cap}", "kind": "provides"}
                )
            if len(capability_map) < _MAX_CAPABILITY_MAP:
                capability_map.append(
                    {"capability": cap, "tool": name, "plugin": plugin_id}
                )

    @staticmethod
    def _conflicts(registry: Any) -> list[dict[str, Any]]:
        try:
            conflicts = list(getattr(registry, "conflicts", []) or [])
        except Exception:  # noqa: BLE001
            return []
        rows: list[dict[str, Any]] = []
        for conflict in conflicts[:_MAX_CONFLICTS]:
            rows.append(
                {
                    "tool_name": str(getattr(conflict, "tool_name", "")),
                    "kept_plugin": str(getattr(conflict, "kept_plugin", "")),
                    "rejected_plugin": str(getattr(conflict, "rejected_plugin", "")),
                }
            )
        return rows

    @staticmethod
    def _fiber_states() -> dict[str, str]:
        try:
            from leapflow.plugins import get_scoped_registry

            fibers = getattr(get_scoped_registry(), "fibers", {}) or {}
            return {
                str(plugin_id): str(getattr(getattr(fiber, "state", ""), "value", "") or "unknown")
                for plugin_id, fiber in dict(fibers).items()
            }
        except Exception:  # noqa: BLE001 - roster degrades to fiber_state=unknown
            logger.debug("evolution producer: fiber states unavailable", exc_info=True)
            return {}

    @staticmethod
    def _trust_and_usage() -> tuple[Any, Any]:
        """Return the live trust ledger and usage tracker, or ``(None, None)``.

        Both come from the process-global advisor, which is absent in-process and
        in tests; the roster then reports ``unverified`` rather than inventing a
        level.
        """
        try:
            from leapflow.learning.plugin_advisor import get_default_advisor

            advisor = get_default_advisor()
        except Exception:  # noqa: BLE001
            return None, None
        if advisor is None:
            return None, None
        return getattr(advisor, "_trust_ledger", None), getattr(advisor, "_usage_tracker", None)

    # ── pipeline reachability ─────────────────────────────────────────────

    def _reachability(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Report the runtime evidence for each pipeline segment.

        Ordered as the pipeline runs. Each row carries the measurement that was
        actually taken, so a reader can tell "nothing has happened yet" from
        "this segment is not wired" from "the source could not be read" -- a
        distinction the presence of a module can never make.
        """
        settings = self._settings()
        observations = self._json_store(
            "capability_observations_path",
            "capability_observation_store",
            "JsonCapabilityObservationStore",
        )
        rows = [
            self._segment_world_model_driver(observations),
            self._segment_evidence_gate(settings),
            self._segment_authorising_origins(settings),
            self._segment_observations(observations),
            self._segment_lifecycle(),
            self._segment_plan_records(),
            self._segment_trust(snapshot),
        ]
        rows.extend(self._segments_awaiting_wiring())
        return rows

    @staticmethod
    def _settings() -> Any:
        try:
            from leapflow.config import get_settings

            return get_settings()
        except Exception:  # noqa: BLE001
            logger.debug("evolution producer: settings unavailable", exc_info=True)
            return None

    def _segment_world_model_driver(self, store: Any) -> dict[str, Any]:
        """Whether the world-model teacher's capability hypotheses reach the pipeline.

        The driver reports its own counts (proposed vs admitted) to the session-end
        pipeline observer, which only logs and keeps them in memory -- so that
        channel is not readable here. The one durable trace is an *admitted*
        intent, which lands as an observation of kind ``world_model_intent``.
        Absence of that trace cannot distinguish "the driver never ran" from "it
        ran and the evidence gate correctly refused the intent", so this reports
        ``unverifiable`` rather than guessing at either.
        """
        if store is None:
            return self._row(
                "world_model_driver",
                "World-model driver",
                UNVERIFIABLE,
                "observation store unreadable",
            )
        try:
            records = store.unresolved(min_count=1, limit=_MAX_ROSTER)
        except Exception:  # noqa: BLE001
            logger.debug("evolution producer: driver evidence read failed", exc_info=True)
            return self._row(
                "world_model_driver",
                "World-model driver",
                UNVERIFIABLE,
                "observation store unreadable",
            )
        admitted = [
            record
            for record in records
            if str((dict(record.get("result") or {})).get("error_type") or "")
            == "world_model_intent"
        ]
        if admitted:
            return self._row(
                "world_model_driver",
                "World-model driver",
                WIRED,
                f"{len(admitted)} admitted world-model intent(s)",
            )
        return self._row(
            "world_model_driver",
            "World-model driver",
            UNVERIFIABLE,
            "no admitted world-model intent; the driver's own counts are not persisted",
            next_step=(
                "Proposed-but-not-admitted is the correct default and leaves no durable "
                "trace. Admit the kind to make it observable: leap config set "
                "accepted_evidence_kinds \"unknown_tool,world_model_intent\""
            ),
        )

    def _segment_evidence_gate(self, settings: Any) -> dict[str, Any]:
        """Which evidence kinds may enter the observation pipeline."""
        if settings is None:
            return self._row("evidence_gate", "Evidence admission", UNVERIFIABLE, "settings unreadable")
        kinds = tuple(getattr(settings, "accepted_evidence_kinds", ()) or ())
        admitted = ", ".join(kinds) if kinds else "unknown_tool (default)"
        if "world_model_intent" in kinds:
            return self._row(
                "evidence_gate", "Evidence admission", WIRED, f"accepted: {admitted}"
            )
        return self._row(
            "evidence_gate",
            "Evidence admission",
            NOT_ADMITTED,
            f"accepted: {admitted}",
            next_step=(
                "World-model intents are proposed but not admitted. Run: leap config set "
                "accepted_evidence_kinds \"unknown_tool,world_model_intent\""
            ),
        )

    def _segment_authorising_origins(self, settings: Any) -> dict[str, Any]:
        """Which requirement origins may authorise acquiring new code."""
        if settings is None:
            return self._row(
                "authorising_origins", "Acquisition authority", UNVERIFIABLE, "settings unreadable"
            )
        origins = tuple(getattr(settings, "evolution_authorising_origins", ()) or ())
        if origins:
            return self._row(
                "authorising_origins",
                "Acquisition authority",
                WIRED,
                f"restricted to: {', '.join(origins)}",
            )
        return self._row(
            "authorising_origins",
            "Acquisition authority",
            NO_EVIDENCE,
            "unrestricted (any origin may authorise)",
            next_step=(
                "Optional hardening: leap config set evolution_authorising_origins \"world_model\""
            ),
        )

    def _segment_observations(self, store: Any) -> dict[str, Any]:
        """Whether capability evidence is accumulating."""
        if store is None:
            return self._row("observations", "Capability observations", UNVERIFIABLE, "store unreadable")
        try:
            open_records = store.unresolved(min_count=1, limit=_MAX_ROSTER)
        except Exception:  # noqa: BLE001
            logger.debug("evolution producer: observation read failed", exc_info=True)
            return self._row("observations", "Capability observations", UNVERIFIABLE, "store unreadable")
        if open_records:
            return self._row(
                "observations", "Capability observations", WIRED, f"{len(open_records)} open"
            )
        return self._row(
            "observations",
            "Capability observations",
            NO_EVIDENCE,
            "no open observations",
            next_step="Nothing to act on: no capability gap has been recorded yet.",
        )

    def _segment_lifecycle(self) -> dict[str, Any]:
        """Whether the trust/probation/quarantine tier has anything to govern."""
        store = self._json_store("capability_proposal_queue_path", "capability_proposal_queue", "JsonCapabilityProposalQueue")
        if store is None:
            return self._row("lifecycle", "Lifecycle records", UNVERIFIABLE, "queue unreadable")
        try:
            items = store.list_items(limit=0)
        except Exception:  # noqa: BLE001
            logger.debug("evolution producer: lifecycle read failed", exc_info=True)
            return self._row("lifecycle", "Lifecycle records", UNVERIFIABLE, "queue unreadable")
        if items:
            counts: dict[str, int] = {}
            for item in items:
                status = str(getattr(item, "status", "") or "unknown")
                counts[status] = counts.get(status, 0) + 1
            spread = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            return self._row("lifecycle", "Lifecycle records", WIRED, spread)
        return self._row(
            "lifecycle",
            "Lifecycle records",
            NO_EVIDENCE,
            "queue is empty",
            next_step=(
                "The governor has nothing to govern. A lifecycle record opens when "
                "plugin_propose runs."
            ),
        )

    def _segment_plan_records(self) -> dict[str, Any]:
        """Whether the adaptive policy is actually deciding."""
        store = self._json_store("capability_plans_path", "capability_plan_store", "JsonCapabilityPlanStore")
        if store is None:
            return self._row("policy", "Policy decisions", UNVERIFIABLE, "plan store unreadable")
        try:
            latest = store.latest() or {}
        except Exception:  # noqa: BLE001
            logger.debug("evolution producer: plan read failed", exc_info=True)
            return self._row("policy", "Policy decisions", UNVERIFIABLE, "plan store unreadable")
        decision = dict(latest.get("policy_decision") or {}) if isinstance(latest, Mapping) else {}
        action = str(decision.get("action") or "")
        if action:
            return self._row("policy", "Policy decisions", WIRED, f"latest action: {action}")
        if latest:
            return self._row(
                "policy", "Policy decisions", NO_EVIDENCE, "plan recorded without a policy decision"
            )
        return self._row(
            "policy",
            "Policy decisions",
            NO_EVIDENCE,
            "no capability plan recorded",
            next_step="No adaptive decision has run yet.",
        )

    def _segment_trust(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Whether trust promotion/demotion is live.

        Promotion and demotion do run on live traffic; quarantine is a separate
        segment because it needs its own feed and has historically had none.
        """
        if not snapshot.get("registry_readable"):
            return self._row("trust", "Trust accrual", UNVERIFIABLE, "registry unreadable")
        roster = list(snapshot.get("roster") or [])
        if not roster:
            return self._row("trust", "Trust accrual", NO_EVIDENCE, "no plugins registered")
        graded = [row for row in roster if str(row.get("trust_level")) != "unverified"]
        if not graded:
            return self._row(
                "trust",
                "Trust accrual",
                UNVERIFIABLE,
                "trust ledger not bound in this process",
                next_step="Trust is reported by the daemon; an in-process run binds no advisor.",
            )
        beyond_draft = [row for row in graded if str(row.get("trust_level")) != "DRAFT"]
        frozen = [row for row in graded if str(row.get("trust_class")) == _TRUST_CLASS_FROZEN]
        detail = f"{len(beyond_draft)}/{len(graded)} above DRAFT"
        if frozen:
            detail += f", {len(frozen)} frozen"
        status = WIRED if beyond_draft or frozen else NO_EVIDENCE
        return self._row("trust", "Trust accrual", status, detail)

    def _segments_awaiting_wiring(self) -> list[dict[str, Any]]:
        """Report the segments whose capability exists but produces no evidence.

        Each of these has a module in the tree. That is deliberately *not* treated
        as evidence: the module having no caller is exactly the failure mode this
        panel exists to expose, so the row reports the absence of observed output
        and names what would close it.
        """
        awaiting = (
            (
                "effect_verification",
                "Effect verification (L3)",
                "no EffectVerdict observed",
                "Closures currently rest on declared fitness (L2). Wire "
                "CapabilityEffectVerifier to verify by observed effect.",
            ),
            (
                "quarantine_feed",
                "Quarantine feed",
                "no quarantine candidate observed",
                "Trust demotion is live but quarantine has no feed. Wire "
                "QuarantineCandidateTracker and drain on a cold path.",
            ),
            (
                "reclamation",
                "Unselectable reclamation",
                "no reclamation candidate observed",
                "Wire UnselectableArtifactReaper to find artifacts no requirement can select.",
            ),
        )
        return [
            self._row(key, label, NO_EVIDENCE, evidence, next_step=next_step)
            for key, label, evidence, next_step in awaiting
        ]

    @staticmethod
    def _row(
        key: str,
        stage: str,
        status: str,
        evidence: str,
        *,
        next_step: str = "",
    ) -> dict[str, Any]:
        return {
            "key": key,
            "stage": stage,
            "status": status,
            "evidence": evidence,
            "next_step": next_step,
        }

    @staticmethod
    def _json_store(layout_attr: str, module: str, class_name: str) -> Any:
        """Build a profile-scoped JSON store, or None when the layout is absent.

        Resolved lazily per cycle rather than cached: the profile layout is bound
        during deferred daemon initialisation, so a store captured at construction
        would either be missing or belong to a stale profile.
        """
        try:
            import importlib

            from leapflow.config import get_settings

            layout = getattr(get_settings(), "profile_layout", None)
            path = getattr(layout, layout_attr, None) if layout is not None else None
            if path is None:
                return None
            store_module = importlib.import_module(f"leapflow.storage.{module}")
            return getattr(store_module, class_name)(path)
        except Exception:  # noqa: BLE001 - a missing store is a degraded row, not a fault
            logger.debug("evolution producer: %s unavailable", class_name, exc_info=True)
            return None

    # ── summary / severity / evidence ─────────────────────────────────────

    def _summary(
        self,
        snapshot: Mapping[str, Any],
        reachability: Sequence[Mapping[str, Any]],
        episodes: Sequence[Any],
        traces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        roster = list(snapshot.get("roster") or [])
        conflicts = list(snapshot.get("conflicts") or [])
        frozen = [row for row in roster if str(row.get("trust_class")) == _TRUST_CLASS_FROZEN]
        unselectable = [row for row in roster if row.get("selectable") == _NO]
        self_acquired = [row for row in roster if str(row.get("provenance")) == _SELF_ACQUIRED]
        reclaimable = [row for row in roster if str(row.get("reclaimable")) == _YES]
        blocked = [row for row in reachability if row["status"] in (NO_EVIDENCE, NOT_ADMITTED)]
        unverifiable = [row for row in reachability if row["status"] == UNVERIFIABLE]
        regressions = [ep for ep in episodes if ep.gap_closure == REOPENED]
        mutations = [ep for ep in episodes if ep.framework_changed]
        resolved = [ep for ep in episodes if ep.gap_closure == RESOLVED]
        runtime_traces = [
            t for t in traces if dict(t.get("detail") or {}).get("phase") != "composition"
        ]
        frozen_traces = [t for t in traces if str(t.get("kind")) == "trust_frozen"]
        unadmitted = sum(
            len(dict(t.get("detail") or {}).get("intents") or [])
            for t in traces
            if dict(t.get("detail") or {}).get("not_admitted_reason")
        )

        if not snapshot.get("registry_readable"):
            headline = "Runtime state could not be verified: the plugin registry is unreachable."
        elif regressions:
            headline = (
                f"{len(regressions)} evolution(s) regressed: a gap that was closed has recurred."
            )
        elif frozen:
            headline = (
                f"{len(roster)} plugins registered; {len(frozen)} frozen by an internal defect."
            )
        elif conflicts:
            headline = f"{len(roster)} plugins registered; {len(conflicts)} tool-name conflicts."
        else:
            headline = (
                f"{len(roster)} plugins registered; {len(episodes)} recent episode(s); "
                f"{len(reachability) - len(blocked) - len(unverifiable)}"
                f"/{len(reachability)} pipeline segments show runtime evidence."
            )

        return {
            "headline": headline,
            "registry_readable": bool(snapshot.get("registry_readable")),
            "registry_version": snapshot.get("registry_version", -1),
            "active_plugins": len(roster),
            "tool_count": sum(int(row.get("tool_count") or 0) for row in roster),
            "conflict_count": len(conflicts),
            "frozen_count": len(frozen),
            "unselectable_count": len(unselectable),
            # Q1: the headline number for "is it growing". Distinct from the plugin
            # count, which a framework has on day one without evolving at all.
            "self_acquired_count": len(self_acquired),
            # Q5: acquired and not earning its registration.
            "reclaimable_count": len(reclaimable),
            "episode_count": len(episodes),
            "mutation_count": len(mutations),
            "regression_count": len(regressions),
            "resolved_count": len(resolved),
            # Traces are the live half: facts no store retains. Counted separately
            # from episodes because they answer "what just happened" rather than
            # "why", and a reader must not read one as the other.
            "trace_count": len(runtime_traces),
            "observed": bool(traces),
            "frozen_trace_count": len(frozen_traces),
            # Proposed by the world model and admitted by nothing. Zero is both the
            # healthy state and the switched-off state; the panel says which.
            "unadmitted_intent_count": unadmitted,
            "segments_total": len(reachability),
            "segments_with_evidence": len(reachability) - len(blocked) - len(unverifiable),
            # Every closure the engine records today rests on declared fitness, so
            # the view must say so rather than let a reader infer that a retired
            # observation proves the capability works. Shown whenever there is a
            # closure to qualify.
            "l2_only_closures": bool(resolved),
            # Gates the second stat row. A healthy framework shows four numbers, not
            # eight; this turns on only when one of them is non-zero, so the row's
            # presence is itself the signal.
            "attention": bool(
                regressions
                or frozen
                or conflicts
                or unadmitted
                or frozen_traces
                or reclaimable
            ),
            "suggestions": self._suggestions(frozen, conflicts, blocked, regressions),
        }

    @staticmethod
    def _suggestions(
        frozen: Sequence[Mapping[str, Any]],
        conflicts: Sequence[Mapping[str, Any]],
        blocked: Sequence[Mapping[str, Any]],
        regressions: Sequence[Any] = (),
    ) -> list[dict[str, str]]:
        chips: list[dict[str, str]] = []
        for episode in list(regressions)[:3]:
            chips.append(
                {
                    "label": f"Regression on {episode.capability or episode.plugin_id or 'a capability'}",
                    "detail": (
                        "A retired observation recurred: the evolution looked successful "
                        "and the gap came back."
                    ),
                }
            )
        for row in frozen[:3]:
            chips.append(
                {
                    "label": f"Inspect frozen plugin {row.get('plugin_id')}",
                    "detail": "Frozen by an internal defect; it still reports DRAFT.",
                }
            )
        for row in conflicts[:3]:
            chips.append(
                {
                    "label": f"Resolve conflict on {row.get('tool_name')}",
                    "detail": f"{row.get('rejected_plugin')} lost the name to {row.get('kept_plugin')}.",
                }
            )
        for row in blocked[:4]:
            if row.get("next_step"):
                chips.append({"label": row["stage"], "detail": row["next_step"]})
        return chips

    @staticmethod
    def _severity(payload: Mapping[str, Any]) -> Severity:
        """Escalate only on facts a person must act on.

        A segment with no evidence is not an alert: an idle pipeline and an
        unadmitted evidence kind are both correct, quiet states. A regression is
        the opposite -- a gap that was closed has come back, so an evolution that
        looked successful was not -- and it is the one finding here worth waking
        someone for. Conflicts and frozen-yet-selectable plugins sit between:
        each means the live capability set is not what it appears to be.
        """
        summary = dict(payload.get("summary") or {})
        if summary.get("regression_count"):
            return Severity.ALERT
        if not summary.get("registry_readable"):
            return Severity.NOTABLE
        if summary.get("frozen_count") or summary.get("conflict_count"):
            return Severity.NOTABLE
        return Severity.INFO

    @staticmethod
    def _evidence(payload: Mapping[str, Any]) -> tuple[Evidence, ...]:
        summary = dict(payload.get("summary") or {})
        rows = [
            Evidence(kind="metric", label="registry_version", value=str(summary.get("registry_version"))),
            Evidence(kind="metric", label="active_plugins", value=str(summary.get("active_plugins"))),
            Evidence(kind="metric", label="tools", value=str(summary.get("tool_count"))),
            Evidence(kind="metric", label="episodes", value=str(summary.get("episode_count"))),
            Evidence(
                kind="metric",
                label="pipeline_evidence",
                value=f"{summary.get('segments_with_evidence')}/{summary.get('segments_total')}",
            ),
        ]
        if summary.get("regression_count"):
            rows.append(
                Evidence(
                    kind="metric", label="regressions", value=str(summary.get("regression_count"))
                )
            )
        if summary.get("conflict_count"):
            rows.append(
                Evidence(kind="metric", label="conflicts", value=str(summary.get("conflict_count")))
            )
        if summary.get("frozen_count"):
            rows.append(
                Evidence(kind="metric", label="frozen", value=str(summary.get("frozen_count")))
            )
        return tuple(rows)

    @staticmethod
    def _actions(payload: Mapping[str, Any]) -> tuple[SuggestedAction, ...]:
        return (
            SuggestedAction(
                name="plugin_list",
                label="Inspect live plugin registry",
                kind="intent",
                params={},
            ),
        )

    @staticmethod
    def _fingerprint(payload: Mapping[str, Any]) -> str:
        """Identify the framework *state*, so an unchanged framework re-notifies once.

        Built from what a reader would react to -- the registry version, each
        plugin's fiber state, trust and whether it has ever been selected,
        conflicts, and each pipeline segment's status -- and deliberately not from
        ``observed_at``, which changes every cycle and would defeat dedup
        entirely. Every field the board renders is covered here: a rendered value
        left out of the fingerprint freezes on the page while still looking
        current, which is worse than not showing it.
        """
        summary = dict(payload.get("summary") or {})
        roster = [
            f"{row.get('plugin_id')}:{row.get('fiber_state')}:{row.get('trust_level')}"
            f":{row.get('selectable')}:{row.get('ever_used')}:{row.get('provenance')}"
            for row in payload.get("roster") or []
        ]
        conflicts = [
            f"{row.get('tool_name')}>{row.get('rejected_plugin')}"
            for row in payload.get("conflicts") or []
        ]
        segments = [f"{row.get('key')}={row.get('status')}" for row in payload.get("reachability") or []]
        # Episodes participate by identity and outcome. Without them a new episode
        # would leave the key unchanged and the executor would skip the write,
        # freezing the timeline on the page while the ledger moved on.
        episodes = [
            f"{row.get('episode_id')}:{row.get('status')}:{row.get('gap_closure')}"
            f":{row.get('mutation_action')}:{row.get('trust_now')}"
            for row in payload.get("episodes") or []
        ]
        # Traces participate by identity, for the same reason: a new trace with an
        # unchanged registry (a trust transition, an unadmitted proposal) must still
        # refresh the page.
        traces = [
            f"{row.get('trace_id')}" for row in payload.get("traces") or []
        ]
        # Fiber transitions are a per-cycle delta, so they enter the key too. One
        # consequence is deliberate: after a retry that ends where it started, the
        # next quiet cycle reproduces the pre-transition key and its write is
        # skipped, so the board keeps showing the retry rather than erasing the only
        # evidence it ever happened.
        transitions = [
            f"{row.get('plugin_id')}:{row.get('from')}>{row.get('to')}"
            for row in payload.get("fiber_transitions") or []
        ]
        material = "|".join(
            [
                str(summary.get("registry_version")),
                str(int(bool(summary.get("registry_readable")))),
                ",".join(sorted(roster)),
                ",".join(sorted(conflicts)),
                ",".join(segments),
                ",".join(episodes),
                ",".join(sorted(traces)),
                ",".join(transitions),
            ]
        )
        import hashlib

        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


__all__ = ["EvolutionProducer", "NOT_ADMITTED", "NO_EVIDENCE", "UNVERIFIABLE", "WIRED"]
