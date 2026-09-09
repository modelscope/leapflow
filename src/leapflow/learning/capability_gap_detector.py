"""Capability gap detection for plugin self-evolution.

The detector is intentionally side-effect free: it only turns structured runtime
evidence into a reviewable PluginProposal. Generation, approval, and install
remain separate steps owned by plugin governance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.evolution_intent import (
    MODEL_AUTHORED_RISK_CEILING,
    WORLD_MODEL_INTENT,
    EvolutionIntent,
)
from leapflow.domain.plugin_proposal import GapEvidence, PluginProposal, ProposedToolSpec, RiskLevel

_SAFE_IDENTIFIER = re.compile(r"[^a-z0-9_]+")

# Requirement origins a declared-evidence payload may claim. Anything else falls
# back to ``environment_probe`` so a malformed payload cannot smuggle in an
# origin the domain layer does not recognise.
_DECLARED_ORIGINS = frozenset(
    {"unknown_tool", "explicit_request", "environment_probe", "task_contract", "world_model"}
)
_DEFAULT_DECLARED_ORIGIN = "environment_probe"


def _slug(value: str, *, fallback: str) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace(".", " ")
    text = _SAFE_IDENTIFIER.sub("_", text).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or fallback


class CapabilityGapDetector:
    """Build plugin proposals from structured missing-capability evidence."""

    def proposal_from_unknown_tool(
        self,
        result: Mapping[str, Any],
        *,
        requested_capability: str = "",
    ) -> PluginProposal | None:
        """Create a proposal from ToolRegistry.unknown_result() payloads."""
        if result.get("error_type") != "unknown_tool":
            return None
        missing = str(result.get("original_tool_name") or "unknown_tool")
        summary = requested_capability.strip() or f"Provide the missing tool '{missing}'."
        tool_name = _slug(missing, fallback="generated_tool")
        plugin_id = _slug(f"{tool_name}_plugin", fallback="generated_tool_plugin")
        evidence = GapEvidence.create(
            "unknown_tool",
            f"Runtime attempted unknown tool '{missing}'.",
            confidence=0.82,
            metadata={
                "original_tool_name": missing,
                "suggestions": ",".join(str(item) for item in result.get("suggestions", [])[:5]),
                "recovery_hint": str(result.get("recovery_hint") or ""),
            },
        )
        proposed_tool = ProposedToolSpec(
            name=tool_name,
            description=summary,
            risk_level="read_only",
            mutates_state=False,
        )
        return PluginProposal.create(
            plugin_id=plugin_id,
            capability_summary=summary,
            gap_type="tool_plugin",
            risk_level="read_only",
            evidence=(evidence,),
            proposed_tools=(proposed_tool,),
        )

    def proposal_from_evolution_intent(
        self,
        intent: EvolutionIntent,
        *,
        risk_ceiling: RiskLevel = MODEL_AUTHORED_RISK_CEILING,
    ) -> PluginProposal:
        """Create a side-effect-free proposal from a world-model intent.

        This is how a world-model hypothesis reaches the surface that actually
        leads to governed acquisition: the same ``PluginProposal`` shape that
        ``self_management.plugin_propose`` produces, so it flows on through
        ``plugin_generate`` (validated code, no install) and ``plugin_install``
        (approval-gated). Creating a proposal mutates nothing.

        The proposal's risk level is the *clamped* ceiling, never the level the
        authoring model asked for; the original request is preserved in the
        evidence metadata for audit.
        """
        effective = intent.effective_risk_ceiling(risk_ceiling)
        metadata: dict[str, Any] = {
            "intent_id": intent.intent_id,
            "capability": intent.capability,
            "confidence": intent.confidence,
        }
        for key, value in (
            ("target_affordance", intent.target_affordance),
            ("expected_effect", intent.expected_effect),
            ("rationale", intent.rationale),
        ):
            if value:
                metadata[key] = value
        if effective != str(intent.max_risk_level):
            metadata["requested_max_risk_level"] = str(intent.max_risk_level)
        if intent.evidence_ids:
            metadata["evidence_ids"] = ",".join(intent.evidence_ids)

        evidence = GapEvidence.create(
            WORLD_MODEL_INTENT,
            intent.hypothesis,
            confidence=intent.confidence,
            metadata=metadata,
        )
        tool_name = _slug(intent.capability, fallback="generated_tool")
        mutates = effective in {"high", "mutating", "external"}
        proposed_tool = ProposedToolSpec(
            name=tool_name,
            description=intent.expected_effect or intent.hypothesis,
            risk_level=effective,  # type: ignore[arg-type]
            mutates_state=mutates,
        )
        return PluginProposal.create(
            plugin_id=_slug(f"{tool_name}_plugin", fallback="generated_tool_plugin"),
            capability_summary=intent.hypothesis,
            gap_type="tool_plugin",
            risk_level=effective,  # type: ignore[arg-type]
            evidence=(evidence,),
            proposed_tools=(proposed_tool,),
        )

    def proposal_from_capability_request(
        self,
        requested_capability: str,
        *,
        plugin_id: str = "",
        proposed_tool_names: Sequence[str] = (),
        risk_level: RiskLevel = "read_only",
        evidence_summary: str = "",
    ) -> PluginProposal:
        """Create a proposal from an explicit user/operator capability request.

        This method does not classify free-form intent. The caller supplies the
        request as evidence, making it suitable for self-management tools and
        UI-driven review flows.
        """
        capability = str(requested_capability or "").strip()
        if not capability:
            raise ValueError("requested_capability is required")
        pid = _slug(plugin_id or f"{capability[:48]}_plugin", fallback="generated_plugin")
        names = tuple(proposed_tool_names) or (_slug(capability[:48], fallback="generated_tool"),)
        evidence = GapEvidence.create(
            "explicit_capability_request",
            evidence_summary or capability,
            confidence=0.9,
            metadata={"requested_capability": capability},
        )
        tools = tuple(
            ProposedToolSpec(
                name=_slug(name, fallback="generated_tool"),
                description=f"Implement capability: {capability}",
                risk_level=risk_level,
                mutates_state=risk_level in {"high", "mutating", "external"},
            )
            for name in names
        )
        return PluginProposal.create(
            plugin_id=pid,
            capability_summary=capability,
            gap_type="tool_plugin",
            risk_level=risk_level,
            evidence=(evidence,),
            proposed_tools=tools,
        )

    def requirements_from_tool_results(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        min_count: int = 1,
    ) -> tuple[CapabilityRequirement, ...]:
        """Aggregate structured evidence into reviewable capability needs.

        This is the observation-only bridge from runtime evidence to adaptive
        resolution. It creates no code, performs no install, and never infers a
        capability name from user text.

        Two evidence shapes are recognised, both declaration-driven:

        * ``unknown_tool`` results, bucketed by the structured
          ``original_tool_name`` emitted by the tool registry (unchanged).
        * any other ``error_type`` that **declares** its ``capability``. Without a
          declared capability the payload is ignored, which keeps the "never
          infer capability from text" rule intact while letting environment- and
          world-model-derived evidence reach the same governed pipeline.

        Widening the accepted evidence set is the job of
        ``CapabilityEvidenceClassifier``; this method is what turns the admitted
        evidence into requirements. Both halves are required -- admitting an
        evidence kind whose payload cannot become a requirement would persist
        observations that silently never produce one.
        """
        unknown_buckets: dict[str, list[Mapping[str, Any]]] = {}
        declared_buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for result in results:
            kind = str(result.get("error_type") or "")
            if kind == "unknown_tool":
                key = str(result.get("original_tool_name") or "unknown_tool")
                unknown_buckets.setdefault(key, []).append(result)
                continue
            capability = str(result.get("capability") or "").strip()
            if not kind or not capability:
                continue
            declared_buckets.setdefault((kind, capability), []).append(result)

        requirements: list[CapabilityRequirement] = []
        for key, bucket in sorted(unknown_buckets.items()):
            if len(bucket) < min_count:
                continue
            latest = bucket[-1]
            requirements.append(
                CapabilityRequirement.create(
                    _slug(key, fallback="generated_tool"),
                    "unknown_tool",
                    evidence=f"Runtime attempted unknown tool '{key}'.",
                    metadata={
                        "original_tool_name": key,
                        "occurrences": len(bucket),
                        "suggestions": ",".join(
                            str(item) for item in latest.get("suggestions", [])[:5]
                        ),
                        "recovery_hint": str(latest.get("recovery_hint") or ""),
                    },
                    requirement_id=f"req-unknown-tool-{_slug(key, fallback='generated_tool')}",
                )
            )
        for (kind, capability), bucket in sorted(declared_buckets.items()):
            if len(bucket) < min_count:
                continue
            requirements.append(
                self._requirement_from_declared(kind, capability, bucket)
            )
        return tuple(requirements)

    def _requirement_from_declared(
        self,
        kind: str,
        capability: str,
        bucket: Sequence[Mapping[str, Any]],
    ) -> CapabilityRequirement:
        """Build a requirement from declared (non-unknown-tool) evidence.

        Every field is read from the payload's declarations; nothing is inferred.
        """
        latest = bucket[-1]
        origin = str(latest.get("origin") or "")
        if origin not in _DECLARED_ORIGINS:
            origin = _DEFAULT_DECLARED_ORIGIN
        evidence = str(latest.get("evidence") or latest.get("recovery_hint") or "")
        metadata: dict[str, Any] = {
            "evidence_kind": kind,
            "occurrences": len(bucket),
        }
        # Propagated declarations. ``target_affordance`` and ``expected_effect``
        # are what tell a later generation step *what to build against* and *how to
        # verify it*; dropping them would leave the requirement unactionable.
        for field_name in (
            "failure_code",
            "recovery_hint",
            "confidence",
            "intent_id",
            "target_affordance",
            "expected_effect",
            "requested_max_risk_level",
        ):
            value = latest.get(field_name)
            if value not in (None, ""):
                metadata[field_name] = value
        suggestions = latest.get("suggestions") or ()
        if suggestions:
            metadata["suggestions"] = ",".join(str(item) for item in list(suggestions)[:5])
        kwargs: dict[str, Any] = {
            "evidence": evidence,
            "metadata": metadata,
            "requirement_id": str(latest.get("requirement_id") or "")
            or f"req-{_slug(kind, fallback='evidence')}-{_slug(capability, fallback='capability')}",
        }
        max_risk = latest.get("max_risk_level")
        if max_risk:
            kwargs["max_risk_level"] = max_risk
        required = latest.get("required_platform_capabilities")
        if required:
            kwargs["required_platform_capabilities"] = list(required)
        return CapabilityRequirement.create(capability, origin, **kwargs)  # type: ignore[arg-type]

    def proposals_from_tool_results(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        min_count: int = 1,
    ) -> tuple[PluginProposal, ...]:
        """Aggregate unknown-tool results into proposals by original tool name."""
        buckets: dict[str, list[Mapping[str, Any]]] = {}
        for result in results:
            if result.get("error_type") != "unknown_tool":
                continue
            key = str(result.get("original_tool_name") or "unknown_tool")
            buckets.setdefault(key, []).append(result)

        proposals: list[PluginProposal] = []
        for key, bucket in sorted(buckets.items()):
            if len(bucket) < min_count:
                continue
            proposal = self.proposal_from_unknown_tool(
                bucket[-1],
                requested_capability=f"Provide a tool compatible with repeated missing call '{key}'.",
            )
            if proposal is not None:
                proposals.append(proposal)
        return tuple(proposals)
