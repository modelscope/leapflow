# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pluggability Boundary Taxonomy — the authoritative decision table.

Maps DSH plugin category strings to LeapFlow compatibility verdicts.
This is a pure-data module with no I/O or side effects beyond building
the taxonomy dict at import time.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from leapflow.learning.compatibility.protocol import Verdict


class TaxonomyEntry(NamedTuple):
    """Single entry in the pluggability taxonomy."""

    target_protocol: Optional[str]
    verdict: Verdict
    reason: str


# ═══════════════════════════════════════════════════════════════════════
# PLUGGABILITY_TAXONOMY: Frozen lookup table mapping DSH category strings
# to LeapFlow compatibility classification.
#
# Source: §7.2 of deepseek_harness_compatibility_analysis.md
# ═══════════════════════════════════════════════════════════════════════

PLUGGABILITY_TAXONOMY: dict[str, TaxonomyEntry] = {
    # ─── COMPATIBLE: Direct tool mapping ───────────────────────────────
    "tools": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Direct mapping via bridge adapter (TS→JSON-RPC)",
    ),
    "web": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "web-search": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "web-fetch": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "filesystem": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "fs": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "shell": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "terminal": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "todo": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    "plan": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.COMPATIBLE,
        reason="Pure tool functionality; maps directly",
    ),
    # ─── ADAPTABLE: Needs bridge or interface translation ─────────────
    "llm": TaxonomyEntry(
        target_protocol="LLMProviderPlugin",
        verdict=Verdict.ADAPTABLE,
        reason="DSH LLM providers use streaming callbacks + Cordis events; needs async generator wrapper",
    ),
    "code-runtime": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.ADAPTABLE,
        reason="Tool surface maps; runtime engine lifecycle needs adapter wrapping",
    ),
    "lsp": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.ADAPTABLE,
        reason="LSP tool surface maps; LSP client lifecycle needs adapter wrapping",
    ),
    "mcp": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.ADAPTABLE,
        reason="MCP tool surface maps; MCP server lifecycle needs JSON-RPC bridge",
    ),
    "signal": TaxonomyEntry(
        target_protocol="SignalSource",
        verdict=Verdict.ADAPTABLE,
        reason="Needs translation from Cordis events to LeapFlow signal protocol",
    ),
    "feedback": TaxonomyEntry(
        target_protocol="SignalSource",
        verdict=Verdict.ADAPTABLE,
        reason="Needs translation from Cordis events to LeapFlow InteractionSignal",
    ),
    # ─── PARTIAL: Subset of features usable ───────────────────────────
    "guard": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.PARTIAL,
        reason="Guard logic can be exposed as advisory tools but cannot intercept the execution pipeline",
    ),
    "scheduler": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.PARTIAL,
        reason="Scheduling tools mappable; scheduling engine is not pluggable",
    ),
    "skill": TaxonomyEntry(
        target_protocol="ToolPlugin",
        verdict=Verdict.PARTIAL,
        reason="Skill catalog/loader tools mappable; skill execution model differs",
    ),
    # ─── INCOMPATIBLE: Targets non-pluggable system layers ────────────
    "agent-loop": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow engine is a single hardened OODA execution loop with PCD; "
            "replacing it breaks session safety, recovery, and context invariants"
        ),
    ),
    "session": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow uses DuckDB as architecturally fixed storage with deep "
            "EventBus, recovery checkpoint, and audit integration"
        ),
    ),
    "compaction": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow context governance (PCD + adaptive depth + 4-layer truncation) "
            "is a hardened subsystem; replacing it breaks session safety"
        ),
    ),
    "scope": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "Cordis hierarchical scope has no LeapFlow equivalent; "
            "LeapFlow uses flat ScopedToolRegistry + session isolation"
        ),
    ),
    "context": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow PromptAssemblyPlan + Semantic Focus Plane are integral "
            "to engine; cannot be replaced without breaking PCD"
        ),
    ),
    "subagent": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow subagent delegation is engine-internal; "
            "exposing as plugin surface violates session identity contracts"
        ),
    ),
    "settings": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow has fixed config system (Settings + leap config + layered YAML); "
            "replacing it would break all config consumers"
        ),
    ),
    "sdk": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "Wire protocols are architecture-bound; "
            "LeapFlow uses daemon RPC, not Cordis JSON-RPC SDK"
        ),
    ),
    "sandbox": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "LeapFlow has its own SandboxHost subprocess isolation; "
            "sandbox backends are not a plugin surface"
        ),
    ),
    "workflow": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "Workflow execution is engine-integrated; "
            "not cross-portable between fundamentally different runtimes"
        ),
    ),
    "identity": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason="Identity management is security-critical and profile-bound; not a plugin surface",
    ),
    "credentials": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason="Security boundary; LeapFlow has its own CredentialVault and profile-scoped secrets",
    ),
    "interaction": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason="Session-coupled UX state; architecture-bound to LeapFlow TUI/daemon interaction model",
    ),
    "extensions": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason=(
            "Self-modification is engine-internal; "
            "cannot be exposed to foreign plugins without breaking Progressive Trust"
        ),
    ),
    "hooks": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason="Hook bridges are LeapFlow-specific daemon integration; foreign hooks cannot be mapped",
    ),
    "storage": TaxonomyEntry(
        target_protocol=None,
        verdict=Verdict.INCOMPATIBLE,
        reason="DuckDB persistence is architecturally fixed; no storage plugin surface exists",
    ),
}

# Immutable view — prevent accidental mutation at runtime
PLUGGABILITY_TAXONOMY = dict(PLUGGABILITY_TAXONOMY)  # type: ignore[assignment]

_FALLBACK = TaxonomyEntry(
    target_protocol=None,
    verdict=Verdict.INCOMPATIBLE,
    reason="Unknown category; not recognized in the pluggability taxonomy",
)


def resolve_category(category: str) -> TaxonomyEntry:
    """Look up a DSH category in the pluggability taxonomy.

    Returns the matching TaxonomyEntry, or a fallback INCOMPATIBLE entry
    for unrecognized categories.
    """
    return PLUGGABILITY_TAXONOMY.get(category, _FALLBACK)
