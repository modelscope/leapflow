# Copyright (c) Alibaba, Inc. and its affiliates.
"""Verdict Synthesizer.

Takes all stage results and produces the final CompatibilityReport verdict.
Aggregation logic:
  - If any stage has verdict=INCOMPATIBLE → final=INCOMPATIBLE
  - If security recommends reject → final=INCOMPATIBLE
  - If any stage has verdict=ADAPTABLE → final=ADAPTABLE
  - If any stage has verdict=PARTIAL → final=PARTIAL
  - Otherwise → final=COMPATIBLE

Also generates AdapterSpec when final verdict is ADAPTABLE.
"""

from __future__ import annotations

from leapflow.learning.compatibility.protocol import (
    AdapterSpec,
    CompatibilityReport,
    PluginManifestInput,
    StageResult,
    Verdict,
)


def synthesize_verdict(
    manifest: PluginManifestInput,
    stages: list[StageResult],
) -> CompatibilityReport:
    """Synthesize a final CompatibilityReport from all stage results.

    Args:
        manifest: The parsed plugin manifest.
        stages: All stage results (stages 1-6 in order).

    Returns:
        A complete CompatibilityReport with final verdict and metadata.
    """
    # Extract target_protocol from category_resolver stage
    target_protocol: str | None = None
    for sr in stages:
        if sr.stage_name == "category_resolver" and sr.evidence:
            target_protocol = sr.evidence.get("target_protocol")
            break

    # Check for INCOMPATIBLE verdicts (first one wins as rejection reason)
    for sr in stages:
        if sr.verdict == Verdict.INCOMPATIBLE:
            return CompatibilityReport(
                manifest=manifest,
                stages=stages,
                final_verdict=Verdict.INCOMPATIBLE,
                target_protocol=target_protocol,
                rejection_reason=sr.details,
                adaptation_notes=[],
                adapter_spec=None,
            )

    # Check for security rejection recommendation
    for sr in stages:
        if sr.stage_name == "security_classifier":
            if sr.evidence and sr.evidence.get("recommendation") == "reject":
                return CompatibilityReport(
                    manifest=manifest,
                    stages=stages,
                    final_verdict=Verdict.INCOMPATIBLE,
                    target_protocol=target_protocol,
                    rejection_reason=sr.details,
                    adaptation_notes=[],
                    adapter_spec=None,
                )

    # Collect adaptation notes from all stages
    adaptation_notes: list[str] = []
    has_adaptable = False
    has_partial = False

    for sr in stages:
        if sr.verdict == Verdict.ADAPTABLE:
            has_adaptable = True
            if sr.details:
                adaptation_notes.append(sr.details)
        elif sr.verdict == Verdict.PARTIAL:
            has_partial = True
            if sr.details:
                adaptation_notes.append(sr.details)

    # Determine final verdict
    if has_adaptable:
        final_verdict = Verdict.ADAPTABLE
    elif has_partial:
        final_verdict = Verdict.PARTIAL
    else:
        final_verdict = Verdict.COMPATIBLE

    # Generate AdapterSpec for ADAPTABLE verdicts
    adapter_spec: AdapterSpec | None = None
    if final_verdict == Verdict.ADAPTABLE and target_protocol:
        adapter_spec = _build_adapter_spec(manifest, target_protocol, stages)

    return CompatibilityReport(
        manifest=manifest,
        stages=stages,
        final_verdict=final_verdict,
        target_protocol=target_protocol,
        adaptation_notes=adaptation_notes,
        adapter_spec=adapter_spec,
    )


def _build_adapter_spec(
    manifest: PluginManifestInput,
    target_protocol: str,
    stages: list[StageResult],
) -> AdapterSpec:
    """Build an AdapterSpec based on source language, execution model, and bridge requirements."""
    # Determine bridge type from source language
    if manifest.source_language.lower() in ("typescript", "javascript"):
        bridge_type = "json_rpc_bridge"
    elif any(
        sr.stage_name == "execution_model_analyzer"
        and sr.evidence.get("requires_bridge")
        for sr in stages
    ):
        bridge_type = "json_rpc_bridge"
    else:
        bridge_type = "protocol_wrapper"

    # Collect shim methods from dependency checker
    shim_methods: list[str] = []
    for sr in stages:
        if sr.stage_name == "dependency_checker" and sr.evidence:
            shim_methods = list(sr.evidence.get("shimmable", []))
            break

    # Estimate complexity
    adaptable_count = sum(1 for sr in stages if sr.verdict == Verdict.ADAPTABLE)
    if adaptable_count >= 3:
        complexity = "high"
    elif adaptable_count >= 2:
        complexity = "medium"
    else:
        complexity = "low"

    return AdapterSpec(
        source_interface=manifest.category,
        target_protocol=target_protocol,
        bridge_type=bridge_type,
        shim_methods=shim_methods,
        estimated_complexity=complexity,
    )
