"""Monitor producer for adaptive capability decision visibility."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from leapflow.monitor.types import Evidence, Finding, ProducerContext, Severity, SuggestedAction

logger = logging.getLogger(__name__)


class CapabilityAdaptationProducer:
    """Emit findings from stored capability resolution / plan records."""

    domain = "capability_adaptation"

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        store = self._resolve_store(ctx)
        if store is None:
            return ()
        latest = store.latest()
        if not latest:
            return ()
        plan = latest.get("plan") or {}
        executable = bool(plan.get("executable"))
        mutation = latest.get("mutation") or {}
        mutation_ok = mutation and mutation.get("ok") is False
        severity = Severity.NOTABLE if (not executable or mutation_ok) else Severity.INFO
        selected = self._selected_tools(latest)
        missing = plan.get("missing_dependencies") or []
        evidence = [
            Evidence(kind="record", label="record_id", value=str(latest.get("record_id") or "")),
            Evidence(kind="metric", label="selected_tools", value=", ".join(selected) or "-"),
        ]
        phase = str(latest.get("phase") or "")
        if phase:
            evidence.append(Evidence(kind="metric", label="loop_phase", value=phase))
        action = str(mutation.get("action") or "") if isinstance(mutation, dict) else ""
        if action:
            evidence.append(Evidence(kind="metric", label="mutation_action", value=action))
        before_version = latest.get("registry_version_before")
        after_version = latest.get("registry_version_after")
        if before_version is not None and after_version is not None:
            evidence.append(
                Evidence(
                    kind="metric",
                    label="registry_delta",
                    value=f"{before_version}->{after_version}",
                )
            )
        delta = latest.get("decision_delta") or {}
        if isinstance(delta, dict) and delta.get("changed"):
            evidence.append(
                Evidence(kind="metric", label="selected_delta", value=str(delta.get("changed")))
            )
        observation_ids = latest.get("observation_ids") or []
        if observation_ids:
            evidence.append(
                Evidence(kind="metric", label="observation_count", value=str(len(observation_ids)))
            )
        proposal = latest.get("proposal") or {}
        if isinstance(proposal, dict) and proposal.get("proposal_id"):
            evidence.append(
                Evidence(kind="record", label="proposal_id", value=str(proposal.get("proposal_id")))
            )
            evidence.append(
                Evidence(
                    kind="metric", label="proposal_status", value=str(proposal.get("status") or "")
                )
            )
        policy_decision = latest.get("policy_decision") or {}
        if isinstance(policy_decision, dict) and policy_decision.get("action"):
            evidence.append(
                Evidence(
                    kind="metric", label="policy_action", value=str(policy_decision.get("action"))
                )
            )
            evidence.append(
                Evidence(
                    kind="metric",
                    label="autonomy_level",
                    value=str(policy_decision.get("autonomy_level") or ""),
                )
            )
        if missing:
            evidence.append(
                Evidence(kind="metric", label="missing_dependencies", value=str(len(missing)))
            )
        return (
            Finding(
                watch_id=ctx.spec.watch_id or self.domain,
                domain=self.domain,
                title="Adaptive plugin capability decision recorded",
                summary=(
                    "Latest capability plan is executable."
                    if executable
                    else "Latest capability plan has unresolved dependencies."
                ),
                severity=severity,
                tags=("capability_adaptation", "plugin_plan"),
                evidence=tuple(evidence),
                suggested_actions=(
                    SuggestedAction(
                        name="plugin_plan",
                        label="Inspect plugin plan",
                        kind="intent",
                        params={"latest": True},
                    ),
                ),
                dedup_key=f"capability_plan:{latest.get('record_id') or plan.get('plan_id') or 'latest'}",
                # The board renders from ``payload`` -- the domain-private escape hatch --
                # while ``evidence`` is the label/value summary a person skims. Only
                # evidence was ever set, so every panel on the capability board bound to
                # ``capability_plan.*`` resolved against an empty mapping: correct
                # headings, correct columns, no rows, and nothing anywhere reported a
                # fault. Requirements, plan steps, deltas and lifecycle results are lists
                # of records that cannot be expressed as label/value pairs at all, which
                # is exactly what this field exists for.
                payload={**latest, "observation_count": len(observation_ids)},
            ),
        )

    def _resolve_store(self, ctx: ProducerContext):
        services = getattr(ctx, "services", None)
        store = getattr(services, "capability_plan_store", None) if services is not None else None
        if store is not None:
            return store
        try:
            from leapflow.config import get_settings
            from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore

            profile_layout = getattr(get_settings(), "profile_layout", None)
            if profile_layout is None:
                return None
            return JsonCapabilityPlanStore(Path(profile_layout.capability_plans_path))
        except (ImportError, RuntimeError, AttributeError, OSError) as exc:
            logger.debug("capability adaptation store unavailable: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _selected_tools(record: dict) -> tuple[str, ...]:
        tools: list[str] = []
        for resolution in record.get("resolutions") or []:
            selected = resolution.get("selected") or {}
            candidate = selected.get("candidate") or {}
            tool_name = str(candidate.get("tool_name") or "")
            if tool_name:
                tools.append(tool_name)
        return tuple(tools)


__all__ = ["CapabilityAdaptationProducer"]
