# Copyright (c) Alibaba, Inc. and its affiliates.
"""Assessment pipeline stages."""

from typing import List, Protocol, runtime_checkable

from leapflow.learning.compatibility.protocol import PluginManifestInput, StageResult


@runtime_checkable
class AssessmentStage(Protocol):
    """A single stage in the compatibility assessment pipeline."""

    stage_name: str

    def assess(
        self, manifest: PluginManifestInput, prior_results: List[StageResult]
    ) -> StageResult:
        """Run this assessment stage and return a StageResult."""
        ...
