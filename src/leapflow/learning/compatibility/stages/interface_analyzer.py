"""Stage 3: Interface Analyzer.

Checks whether the plugin's declared_interfaces list includes methods/attributes
that map to what the target LeapFlow Protocol requires.

For P1 this is a pattern-matching heuristic (not AST analysis).
A dict maps protocol names to required interface patterns; the plugin's
declared_interfaces are scored against those patterns.
"""

from __future__ import annotations

from typing import List

from leapflow.learning.compatibility.protocol import (
    PluginManifestInput,
    StageResult,
    Verdict,
)

# ═══════════════════════════════════════════════════════════════════════
# Required interface patterns per target protocol.
# Each entry: protocol_name → list of acceptable interface pattern sets.
# A plugin must declare at least ONE pattern from the list to be considered
# compatible with the protocol.
# ═══════════════════════════════════════════════════════════════════════

REQUIRED_INTERFACE_PATTERNS: dict[str, list[set[str]]] = {
    "ToolPlugin": [
        # Any tool-like interface: execute, invoke, call, run, handle, etc.
        {"execute"},
        {"invoke"},
        {"call"},
        {"run"},
        {"handle"},
        {"call_tool"},
        {"tools"},
        {"describe"},
        # DSH tool patterns
        {"web_search"},
        {"web_fetch"},
        {"fs_read"},
        {"fs_write"},
        {"shell_exec"},
        {"connect"},
    ],
    "LLMProviderPlugin": [
        # LLM provider must declare model/generate/complete/chat-like interfaces
        {"generate"},
        {"complete"},
        {"chat"},
        {"stream"},
        {"model"},
        {"create_completion"},
    ],
    "SignalSource": [
        # Signal sources need observe/emit/subscribe-like interfaces
        {"observe"},
        {"emit"},
        {"subscribe"},
        {"on_event"},
        {"signal"},
        {"listen"},
    ],
}

# Broad patterns: if the declared interface contains any substring from this
# set for the protocol, treat it as a match (fuzzy fallback).
_FUZZY_PATTERNS: dict[str, list[str]] = {
    "ToolPlugin": ["tool", "exec", "invoke", "call", "run", "handle", "fetch", "search", "read", "write"],
    "LLMProviderPlugin": ["llm", "model", "generat", "complet", "chat", "stream", "infer"],
    "SignalSource": ["signal", "event", "observ", "emit", "subscrib", "listen"],
}


class InterfaceAnalyzer:
    """Analyze whether declared interfaces satisfy the target protocol requirements."""

    stage_name: str = "interface_analyzer"

    def assess(
        self, manifest: PluginManifestInput, prior_results: List[StageResult]
    ) -> StageResult:
        """Check declared_interfaces against target protocol requirements.

        Uses the target_protocol from Stage 2 (category_resolver) evidence.
        If no interfaces are declared but the category passed Stage 2,
        returns passed=True with a note (benefit of the doubt for P1).
        """
        # Extract target protocol from prior stage 2 result
        target_protocol: str | None = None
        for pr in prior_results:
            if pr.stage_name == "category_resolver" and pr.evidence:
                target_protocol = pr.evidence.get("target_protocol")
                break

        if not target_protocol:
            # No target protocol means category was likely INCOMPATIBLE;
            # this stage shouldn't have been reached, but be defensive.
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=None,
                details="No target protocol resolved; skipping interface analysis",
                evidence={"target_protocol": None, "match_type": "skipped"},
            )

        declared = manifest.declared_interfaces

        # Missing interfaces are not evidence of compatibility. JavaScript/Cordis
        # plugins often register tools dynamically, so the restricted Node worker
        # must discover the real public surface before installation. For native
        # manifests, absence is still a partial contract rather than an assumed
        # match.
        if not declared:
            requires_discovery = manifest.source_language.lower().strip() in {
                "typescript", "javascript", "rust", "go"
            }
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=Verdict.ADAPTABLE if requires_discovery else None,
                details=(
                    f"No interfaces declared for {target_protocol}; restricted runtime "
                    "discovery is required before the plugin is installable"
                    if requires_discovery
                    else f"No interfaces declared; native {target_protocol} validation is deferred to import"
                ),
                evidence={
                    "target_protocol": target_protocol,
                    "declared_interfaces": [],
                    "match_type": (
                        "runtime_discovery_required" if requires_discovery else "native_import_required"
                    ),
                    "requires_runtime_discovery": requires_discovery,
                },
            )

        # Exact match check
        exact_match = self._check_exact_match(target_protocol, declared)
        if exact_match:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=None,
                details=f"Interfaces match {target_protocol} requirements (exact: {exact_match})",
                evidence={
                    "target_protocol": target_protocol,
                    "declared_interfaces": declared,
                    "matched_patterns": exact_match,
                    "match_type": "exact",
                },
            )

        # Fuzzy match check
        fuzzy_matches = self._check_fuzzy_match(target_protocol, declared)
        if fuzzy_matches:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=Verdict.ADAPTABLE,
                details=(
                    f"Interfaces partially match {target_protocol} via fuzzy patterns "
                    f"({', '.join(fuzzy_matches)}); adapter may be needed"
                ),
                evidence={
                    "target_protocol": target_protocol,
                    "declared_interfaces": declared,
                    "fuzzy_matches": fuzzy_matches,
                    "match_type": "fuzzy",
                },
            )

        # No match at all — incompatible interfaces
        required = REQUIRED_INTERFACE_PATTERNS.get(target_protocol, [])
        required_flat = sorted({p for s in required for p in s})
        return StageResult(
            stage_name=self.stage_name,
            passed=False,
            verdict=Verdict.INCOMPATIBLE,
            details=(
                f"Declared interfaces {declared} do not match any known pattern "
                f"for {target_protocol}. Expected at least one of: {required_flat}"
            ),
            evidence={
                "target_protocol": target_protocol,
                "declared_interfaces": declared,
                "expected_patterns": required_flat,
                "match_type": "none",
            },
        )

    @staticmethod
    def _check_exact_match(protocol: str, declared: list[str]) -> list[str]:
        """Check for exact matches against known required patterns."""
        patterns = REQUIRED_INTERFACE_PATTERNS.get(protocol, [])
        matches: list[str] = []
        declared_lower = {d.lower() for d in declared}
        for pattern_set in patterns:
            if pattern_set & declared_lower:
                matches.extend(pattern_set & declared_lower)
        return sorted(set(matches))

    @staticmethod
    def _check_fuzzy_match(protocol: str, declared: list[str]) -> list[str]:
        """Check for fuzzy substring matches."""
        substrings = _FUZZY_PATTERNS.get(protocol, [])
        matches: list[str] = []
        for iface in declared:
            iface_lower = iface.lower()
            for sub in substrings:
                if sub in iface_lower:
                    matches.append(iface)
                    break
        return matches
