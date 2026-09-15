# Copyright (c) Alibaba, Inc. and its affiliates.
"""Stage 2: Category Resolver.

Looks up the manifest's category in the PLUGGABILITY_TAXONOMY and
produces a verdict based on whether the category is pluggable in LeapFlow.
"""

from __future__ import annotations

from typing import List

from leapflow.learning.compatibility.protocol import (
    PluginManifestInput,
    StageResult,
    Verdict,
)
from leapflow.learning.compatibility.taxonomy import resolve_category


class CategoryResolver:
    """Resolve DSH category to LeapFlow pluggability verdict via taxonomy lookup."""

    stage_name: str = "category_resolver"

    def assess(
        self, manifest: PluginManifestInput, prior_results: List[StageResult]
    ) -> StageResult:
        """Look up category in taxonomy and produce a verdict.

        If the category maps to INCOMPATIBLE, returns passed=False.
        If COMPATIBLE/ADAPTABLE/PARTIAL, returns passed=True with target protocol.
        """
        entry = resolve_category(manifest.category)

        if entry.verdict == Verdict.INCOMPATIBLE:
            return StageResult(
                stage_name=self.stage_name,
                passed=False,
                verdict=Verdict.INCOMPATIBLE,
                details=entry.reason,
                evidence={
                    "category": manifest.category,
                    "target_protocol": None,
                    "pluggability": "not_pluggable",
                },
            )

        return StageResult(
            stage_name=self.stage_name,
            passed=True,
            verdict=entry.verdict,
            details=entry.reason,
            evidence={
                "category": manifest.category,
                "target_protocol": entry.target_protocol,
                "pluggability": "pluggable",
            },
        )
