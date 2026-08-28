"""Stage 4: Dependency Checker.

Checks declared_dependencies against what LeapFlow can provide.
Uses three classification sets:
  - satisfiable: LeapFlow natively provides this service
  - shimmable: Can be faked/shimmed with acceptable loss
  - blocking: Cannot be provided; blocks installation
"""

from __future__ import annotations

from typing import List

from leapflow.learning.compatibility.protocol import (
    DependencyFeasibility,
    PluginManifestInput,
    StageResult,
    Verdict,
)

# ═══════════════════════════════════════════════════════════════════════
# Known dependency classifications. Names are normalized and matched exactly.
# ═══════════════════════════════════════════════════════════════════════

# Dependencies LeapFlow can actually provide to a foreign runtime in P0.
# npm libraries are intentionally absent: P0 never runs npm install/build, so a
# package depending on node-fetch/axios is not satisfiable merely because Python
# has an HTTP client with a similar purpose.
SATISFIABLE_DEPS: set[str] = {
    # Core runtime services LeapFlow provides
    "config",
    "event_bus",
    "registry",
    "approval_gate",
    "llm_provider",
    "memory_manager",
    "storage",
    "duckdb",
    "plugin_registry",
    "tool_registry",
    "signal_bus",
    "settings",
    "scheduler",
    "file_read_gate",
    "research_ledger",
}

SHIMMABLE_DEPS: set[str] = {
    # Can be shimmed with thin wrappers or stubs
    "cordis",
    "cordis-context",
    "dsh-sdk",
    "dsh-config",
    "dsh-logger",
    "dsh-events",
    "dsh-metrics",
    "dsh-telemetry",
    "logger",
    "metrics",
    "telemetry",
}

BLOCKING_DEPS: set[str] = {
    # Cannot be provided — architecture-bound to DSH
    "cordis-scope",
    "dsh-scope-service",
    "dsh-session-persistence",
    "dsh-hooks-sdk",
    "dsh-agent-loop",
    "dsh-compaction",
    "dsh-identity",
    "dsh-workflow-engine",
    "cordis-lifecycle",
}


def _classify_dep(dep: str, *, foreign_runtime: bool = False) -> DependencyFeasibility:
    """Classify one exact, normalized dependency name.

    Substring matching made unrelated packages inherit privileged classifications
    (for example a name containing ``config`` became satisfiable). Dependency
    names are stable protocol identifiers, so exact matching is both simpler and
    safer. Unknown dependencies from a foreign runtime are blocking in P0 because
    LeapFlow neither installs packages nor proves that they are bundled. Native
    manifests may still name runtime dependencies injected by the host.
    """
    dep_lower = dep.lower().strip()
    if dep_lower in SATISFIABLE_DEPS:
        return DependencyFeasibility.SATISFIABLE
    if dep_lower in SHIMMABLE_DEPS:
        return DependencyFeasibility.SHIMMABLE
    if dep_lower in BLOCKING_DEPS:
        return DependencyFeasibility.BLOCKING
    if foreign_runtime:
        return DependencyFeasibility.BLOCKING
    return DependencyFeasibility.SATISFIABLE


class DependencyChecker:
    """Check declared dependencies for satisfiability within LeapFlow."""

    stage_name: str = "dependency_checker"

    def assess(
        self, manifest: PluginManifestInput, prior_results: List[StageResult]
    ) -> StageResult:
        """Classify each declared dependency and produce an aggregate verdict.

        - All satisfiable → COMPATIBLE (passed=True)
        - Some shimmable, none blocking → ADAPTABLE (passed=True)
        - Any blocking → INCOMPATIBLE (passed=False)
        """
        deps = manifest.declared_dependencies
        if not deps:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=None,
                details="No dependencies declared; no conflicts",
                evidence={"dependencies": [], "classification": {}},
            )

        classification: dict[str, str] = {}
        blocking: list[str] = []
        shimmable: list[str] = []
        satisfiable: list[str] = []

        foreign_runtime = manifest.source_format == "dsh" or manifest.source_language.lower() in {
            "javascript",
            "typescript",
        }
        for dep in deps:
            feasibility = _classify_dep(dep, foreign_runtime=foreign_runtime)
            classification[dep] = feasibility.value
            if feasibility == DependencyFeasibility.BLOCKING:
                blocking.append(dep)
            elif feasibility == DependencyFeasibility.SHIMMABLE:
                shimmable.append(dep)
            else:
                satisfiable.append(dep)

        evidence = {
            "dependencies": deps,
            "classification": classification,
            "satisfiable": satisfiable,
            "shimmable": shimmable,
            "blocking": blocking,
        }

        if blocking:
            return StageResult(
                stage_name=self.stage_name,
                passed=False,
                verdict=Verdict.INCOMPATIBLE,
                details=(
                    f"Blocking or unavailable dependencies cannot be satisfied in P0: {blocking}. "
                    "DSH packages must be self-contained pre-built bundles; npm install/build "
                    "and architecture-bound DSH services are not available."
                ),
                evidence=evidence,
            )

        if shimmable:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=Verdict.ADAPTABLE,
                details=(
                    f"Dependencies {shimmable} need shim layers; "
                    f"remaining {len(satisfiable)} are natively satisfiable"
                ),
                evidence=evidence,
            )

        return StageResult(
            stage_name=self.stage_name,
            passed=True,
            verdict=None,
            details=f"All {len(satisfiable)} dependencies are satisfiable",
            evidence=evidence,
        )
