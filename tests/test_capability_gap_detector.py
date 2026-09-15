# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for capability gap detection and plugin proposals."""
from __future__ import annotations

import dataclasses

import pytest

from leapflow.domain.plugin_proposal import GapEvidence, PluginProposal, ProposedToolSpec
from leapflow.learning.capability_gap_detector import CapabilityGapDetector


pytestmark = pytest.mark.unit


def test_plugin_proposal_domain_types_are_frozen() -> None:
    evidence = GapEvidence.create("explicit", "Need JSON tools", confidence=2.0)
    tool = ProposedToolSpec(name="json_validate", description="Validate JSON")
    proposal = PluginProposal.create(
        plugin_id="json_tools",
        capability_summary="Validate and pretty print JSON",
        evidence=(evidence,),
        proposed_tools=(tool,),
    )

    assert dataclasses.is_dataclass(proposal)
    assert proposal.evidence[0].confidence == 1.0
    assert proposal.to_dict()["proposed_tools"][0]["name"] == "json_validate"
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.plugin_id = "other"  # type: ignore[misc]


def test_detector_builds_proposal_from_unknown_tool_result() -> None:
    detector = CapabilityGapDetector()
    result = {
        "ok": False,
        "error_type": "unknown_tool",
        "original_tool_name": "json.pretty-print",
        "suggestions": ["json_validate"],
        "recovery_hint": "No registered JSON formatter.",
    }

    proposal = detector.proposal_from_unknown_tool(result)

    assert proposal is not None
    assert proposal.plugin_id == "json_pretty_print_plugin"
    assert proposal.proposed_tools[0].name == "json_pretty_print"
    assert proposal.evidence[0].evidence_type == "unknown_tool"
    assert proposal.to_dict()["risk_level"] == "read_only"


def test_detector_aggregates_repeated_unknown_tools() -> None:
    detector = CapabilityGapDetector()
    results = [
        {"error_type": "unknown_tool", "original_tool_name": "foo.tool"},
        {"error_type": "unknown_tool", "original_tool_name": "foo.tool"},
        {"error_type": "unknown_tool", "original_tool_name": "bar.tool"},
    ]

    proposals = detector.proposals_from_tool_results(results, min_count=2)

    assert len(proposals) == 1
    assert proposals[0].proposed_tools[0].name == "foo_tool"


def test_detector_aggregates_unknown_tools_into_requirements() -> None:
    detector = CapabilityGapDetector()
    results = [
        {
            "error_type": "unknown_tool",
            "original_tool_name": "json.pretty-print",
            "suggestions": ["json_validate"],
            "recovery_hint": "No formatter is registered.",
        },
        {"error_type": "unknown_tool", "original_tool_name": "json.pretty-print"},
    ]

    requirements = detector.requirements_from_tool_results(results, min_count=2)

    assert len(requirements) == 1
    req = requirements[0]
    assert req.requirement_id == "req-unknown-tool-json_pretty_print"
    assert req.capability == "json_pretty_print"
    assert req.origin == "unknown_tool"
    assert dict(req.metadata)["occurrences"] == "2"


def test_detector_builds_proposal_from_explicit_request() -> None:
    detector = CapabilityGapDetector()

    proposal = detector.proposal_from_capability_request(
        "Validate JSON and pretty-print it",
        plugin_id="json_tools",
        proposed_tool_names=("json_validate", "json_pretty_print"),
    )

    assert proposal.plugin_id == "json_tools"
    assert [tool.name for tool in proposal.proposed_tools] == [
        "json_validate",
        "json_pretty_print",
    ]
    assert proposal.evidence[0].evidence_type == "explicit_capability_request"


def test_detector_rejects_empty_capability_request() -> None:
    detector = CapabilityGapDetector()

    with pytest.raises(ValueError):
        detector.proposal_from_capability_request("")
