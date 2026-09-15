#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deterministic adaptive plugin scenario-matrix experiment.

This harness validates the adaptive decision layer above plugin lifecycle
mechanics:

    environment fingerprint -> capability requirement -> candidate resolution
    -> transparent rejection/selection evidence -> declarative orchestration plan

The P0 experiment is intentionally self-contained under ``temp/plugin_exp``. It
uses synthetic candidates and structured environment facts, and does not call an
LLM, network, daemon process, or real plugin installation path.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leapflow.analysis.environment_probe import EnvironmentProbe  # noqa: E402
from leapflow.domain.capability_requirement import CapabilityRequirement  # noqa: E402
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest  # noqa: E402
from leapflow.learning.plugin_stats import PluginUsageTracker  # noqa: E402
from leapflow.learning.plugin_trust import PluginTrustLedger  # noqa: E402
from leapflow.plugins.capability_plan import CapabilityPlan  # noqa: E402
from leapflow.plugins.capability_resolver import (  # noqa: E402
    CapabilityCandidate,
    CapabilityResolver,
    ResolverContext,
    candidates_from_registry,
)
from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore  # noqa: E402


@dataclass(frozen=True)
class UsageSampleSpec:
    """One deterministic usage sample injected into PluginUsageTracker."""

    tool_name: str
    ok: bool
    duration_ms: float
    count: int = 1


@dataclass(frozen=True)
class ScenarioSpec:
    """One scenario in the adaptive plugin experiment matrix."""

    name: str
    description: str
    platform_capabilities: tuple[Capability, ...]
    workspace_files: tuple[str, ...]
    requirements: tuple[CapabilityRequirement, ...]
    candidates: tuple[CapabilityCandidate, ...]
    trust_successes: tuple[tuple[str, int], ...] = ()
    usage_samples: tuple[UsageSampleSpec, ...] = ()
    expectations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentStrategyItem:
    """Roadmap entry printed into reports and README."""

    priority: str
    title: str
    goal: str
    needs_framework_change: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "title": self.title,
            "goal": self.goal,
            "needs_framework_change": self.needs_framework_change,
        }


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _report_paths() -> tuple[Path, Path, Path, Path]:
    out = EXP_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    return (
        out / f"{stamp}-adaptive-plugin-matrix.json",
        out / f"{stamp}-adaptive-plugin-matrix.md",
        out / f"{stamp}-adaptive-plugin-matrix.html",
        out / f"{stamp}-real-registry-metadata-gaps.md",
    )


def _manifest(*caps: Capability) -> PlatformManifest:
    return PlatformManifest(
        platform_id=PlatformID.DARWIN_15,
        os_version="15.0",
        capabilities=frozenset(caps),
    )


def _req(
    capability: str,
    origin: str = "task_contract",
    *,
    evidence: str = "",
    max_risk_level: str = "external",
    approval_mode: str = "review_required",
) -> CapabilityRequirement:
    return CapabilityRequirement.create(
        capability,
        origin,  # type: ignore[arg-type]
        evidence=evidence or f"Scenario requires {capability}.",
        max_risk_level=max_risk_level,  # type: ignore[arg-type]
        approval_mode=approval_mode,  # type: ignore[arg-type]
        requirement_id=f"req-{capability.replace('.', '-')}",
    )


def _candidate(
    plugin_id: str,
    tool_name: str,
    *,
    provides: tuple[str, ...],
    requires: tuple[str, ...] = (),
    requires_platform: tuple[str, ...] = (),
    risk_level: str = "read_only",
    requires_approval: bool = False,
    mutates_state: bool = False,
) -> CapabilityCandidate:
    return CapabilityCandidate(
        plugin_id=plugin_id,
        tool_name=tool_name,
        provides_capabilities=provides,
        requires_capabilities=requires,
        requires_platform_capabilities=requires_platform,
        risk_level=risk_level,
        requires_approval=requires_approval,
        mutates_state=mutates_state,
    )


def _base_candidates(*, include_reader: bool = True) -> tuple[CapabilityCandidate, ...]:
    candidates: list[CapabilityCandidate] = [
        _candidate("json_draft", "json_draft_pretty", provides=("json.pretty",)),
        _candidate("json_stable", "json_stable_pretty", provides=("json.pretty",)),
        _candidate(
            "json_shell",
            "shell_json_pretty",
            provides=("json.pretty",),
            requires_platform=("shell.exec",),
            risk_level="external",
            requires_approval=True,
            mutates_state=True,
        ),
    ]
    if include_reader:
        candidates.append(_candidate("json_reader", "json_read", provides=("json.read",)))
    candidates.append(
        _candidate(
            "json_reporter",
            "json_report",
            provides=("json.report",),
            requires=("json.read",),
        )
    )
    return tuple(candidates)


def _cycle_candidates() -> tuple[CapabilityCandidate, ...]:
    return (
        _candidate("cycle_a", "tool_a", provides=("cap.a",), requires=("cap.b",)),
        _candidate("cycle_b", "tool_b", provides=("cap.b",), requires=("cap.a",)),
    )


def _default_usage(
    *, stable_bad: bool = False, draft_strong: bool = False
) -> tuple[UsageSampleSpec, ...]:
    if stable_bad or draft_strong:
        return (
            UsageSampleSpec("json_stable_pretty", True, 20.0, count=2),
            UsageSampleSpec("json_stable_pretty", False, 80.0, count=6),
            UsageSampleSpec("json_draft_pretty", True, 5.0, count=10),
        )
    return (
        UsageSampleSpec("json_stable_pretty", True, 6.0, count=8),
        UsageSampleSpec("json_draft_pretty", True, 8.0, count=6),
        UsageSampleSpec("json_draft_pretty", False, 8.0, count=4),
    )


def _scenarios() -> tuple[ScenarioSpec, ...]:
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    json_pretty = _req(
        "json.pretty",
        "explicit_request",
        evidence="Need to pretty print JSON with the best available plugin.",
    )
    json_read = _req("json.read", evidence="Provider capability required by json.report.")
    json_report = _req(
        "json.report",
        evidence="Need a report that depends on reading JSON first.",
    )
    unknown_requirements = CapabilityGapDetector().requirements_from_tool_results(
        (
            {
                "error_type": "unknown_tool",
                "original_tool_name": "json_pretty",
                "suggestions": ["json_stable_pretty"],
                "recovery_hint": "No exact json_pretty tool is registered.",
            },
            {"error_type": "unknown_tool", "original_tool_name": "json_pretty"},
        ),
        min_count=2,
    )
    unknown_json_pretty = unknown_requirements[0]
    unknown_candidates = _base_candidates() + (
        _candidate("json_unknown_adapter", "json_pretty_unknown", provides=("json_pretty",)),
    )
    return (
        ScenarioSpec(
            name="file_ops_only",
            description="Baseline Python workspace: shell-dependent candidate is unavailable.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("pyproject.toml",),
            requirements=(json_pretty, json_read, json_report),
            candidates=_base_candidates(),
            trust_successes=("json_stable", 3),
            usage_samples=_default_usage(),
            expectations={
                "selected": {"json.pretty": "json_stable_pretty"},
                "excluded": {"json_shell": "missing platform capabilities: shell.exec"},
                "plan_before": ("json_read", "json_report"),
                "executable": True,
            },
        ),
        ScenarioSpec(
            name="shell_enabled",
            description="Same workspace with shell.exec available: shell candidate becomes eligible.",
            platform_capabilities=(Capability.FILE_OPS, Capability.SHELL_EXEC),
            workspace_files=("pyproject.toml",),
            requirements=(json_pretty, json_read, json_report),
            candidates=_base_candidates(),
            trust_successes=("json_stable", 3),
            usage_samples=_default_usage(),
            expectations={
                "selected": {"json.pretty": "json_stable_pretty"},
                "not_excluded": {"json.pretty": "json_shell"},
                "executable": True,
            },
        ),
        ScenarioSpec(
            name="trust_flip",
            description="Trust and reliability evidence flips json.pretty selection to the draft plugin.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("pyproject.toml",),
            requirements=(json_pretty,),
            candidates=_base_candidates(),
            trust_successes=("json_draft", 3),
            usage_samples=_default_usage(stable_bad=True, draft_strong=True),
            expectations={"selected": {"json.pretty": "json_draft_pretty"}, "executable": True},
        ),
        ScenarioSpec(
            name="risk_limit_read_only",
            description="Requirement forbids external risk, so the only matching external candidate is excluded.",
            platform_capabilities=(Capability.FILE_OPS, Capability.SHELL_EXEC),
            workspace_files=("pyproject.toml",),
            requirements=(
                _req(
                    "json.pretty",
                    evidence="Read-only caller refuses external side effects.",
                    max_risk_level="read_only",
                ),
            ),
            candidates=(
                _candidate(
                    "json_shell",
                    "shell_json_pretty",
                    provides=("json.pretty",),
                    requires_platform=("shell.exec",),
                    risk_level="external",
                    requires_approval=True,
                    mutates_state=True,
                ),
            ),
            expectations={
                "unmet": ("json.pretty",),
                "excluded": {"json_shell": "exceeds max"},
                "executable": True,
            },
        ),
        ScenarioSpec(
            name="missing_dependency",
            description="Selected report tool has no selected provider for json.read.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("pyproject.toml",),
            requirements=(json_report,),
            candidates=_base_candidates(include_reader=False),
            expectations={
                "selected": {"json.report": "json_report"},
                "missing_dependency": "json.read",
                "executable": False,
            },
        ),
        ScenarioSpec(
            name="dependency_cycle",
            description="Two selected tools depend on each other, so the plan reports a cycle.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("pyproject.toml",),
            requirements=(
                _req("cap.a", evidence="Cycle scenario requires cap.a."),
                _req("cap.b", evidence="Cycle scenario requires cap.b."),
            ),
            candidates=_cycle_candidates(),
            expectations={"cycle_detected": True, "executable": False},
        ),
        ScenarioSpec(
            name="unmet_requirement",
            description="No candidate declares csv.parse, so the requirement stays unmet.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("pyproject.toml",),
            requirements=(_req("csv.parse", evidence="No plugin currently provides CSV parsing."),),
            candidates=_base_candidates(),
            expectations={"unmet": ("csv.parse",), "executable": True},
        ),
        ScenarioSpec(
            name="unknown_tool_ingestion",
            description="Repeated unknown_tool evidence is converted into a requirement and resolved.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("pyproject.toml",),
            requirements=(unknown_json_pretty,),
            candidates=unknown_candidates,
            trust_successes=(("json_unknown_adapter", 3),),
            usage_samples=(UsageSampleSpec("json_pretty_unknown", True, 4.0, count=6),),
            expectations={
                "selected": {"json_pretty": "json_pretty_unknown"},
                "origin": {"json_pretty": "unknown_tool"},
                "executable": True,
            },
        ),
        ScenarioSpec(
            name="node_workspace_marker",
            description="Node workspace marker changes the environment fingerprint without changing candidates.",
            platform_capabilities=(Capability.FILE_OPS,),
            workspace_files=("package.json",),
            requirements=(json_pretty,),
            candidates=_base_candidates(),
            trust_successes=("json_stable", 3),
            usage_samples=_default_usage(),
            expectations={"selected": {"json.pretty": "json_stable_pretty"}, "executable": True},
        ),
    )


def _normalize_trust_successes(raw: tuple[Any, ...]) -> tuple[tuple[str, int], ...]:
    if not raw:
        return ()
    if len(raw) == 2 and isinstance(raw[0], str):
        return ((str(raw[0]), int(raw[1])),)
    return tuple((str(item[0]), int(item[1])) for item in raw)  # type: ignore[index]


def _prepare_workspace(name: str, files: Iterable[str]) -> Path:
    workspace = EXP_ROOT / "workspaces" / name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for rel in files:
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "package.json":
            path.write_text('{"name":"adaptive-demo"}\n', encoding="utf-8")
        elif rel == "pyproject.toml":
            path.write_text("[project]\nname='adaptive-demo'\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
    return workspace


def _build_context(spec: ScenarioSpec) -> ResolverContext:
    workspace = _prepare_workspace(spec.name, spec.workspace_files)
    marker_names = tuple(sorted({"pyproject.toml", "package.json", *spec.workspace_files}))
    env = EnvironmentProbe(workspace_markers=marker_names).probe(
        platform_manifest=_manifest(*spec.platform_capabilities),
        workspace_root=workspace,
    )

    trust = PluginTrustLedger(candidate_at=1, verified_at=2, production_at=3)
    for plugin_id, count in _normalize_trust_successes(spec.trust_successes):
        for _ in range(count):
            trust.record_success(plugin_id)

    usage = PluginUsageTracker()
    usage._get_reverse_index = lambda: {c.tool_name: c.plugin_id for c in spec.candidates}
    for sample in spec.usage_samples:
        for _ in range(sample.count):
            usage.record(sample.tool_name, sample.ok, sample.duration_ms)
    return ResolverContext(environment=env, trust_ledger=trust, usage_tracker=usage)


def _resolution_by_capability(resolutions: Iterable[Any]) -> dict[str, Any]:
    return {r.requirement.capability: r for r in resolutions}


def _selected_scores(resolutions: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(r.selected for r in resolutions if r.selected is not None)


def _candidate_score_by_plugin(resolution: Any, plugin_id: str) -> Any | None:
    for scored in resolution.candidates:
        if scored.candidate.plugin_id == plugin_id:
            return scored
    return None


def _evaluate_expectations(
    spec: ScenarioSpec,
    resolutions: tuple[Any, ...],
    plan: CapabilityPlan,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    by_capability = _resolution_by_capability(resolutions)
    expected_selected = dict(spec.expectations.get("selected") or {})
    for capability, expected_tool in expected_selected.items():
        selected = by_capability.get(capability).selected if capability in by_capability else None
        tool_name = selected.candidate.tool_name if selected is not None else ""
        if tool_name != expected_tool:
            errors.append(
                f"{capability}: expected selected {expected_tool}, got {tool_name or '(none)'}"
            )

    for capability, expected_origin in dict(spec.expectations.get("origin") or {}).items():
        resolution = by_capability.get(capability)
        origin = resolution.requirement.origin if resolution is not None else ""
        if origin != expected_origin:
            errors.append(
                f"{capability}: expected origin {expected_origin}, got {origin or '(none)'}"
            )

    for capability in spec.expectations.get("unmet") or ():
        if capability not in by_capability or not by_capability[capability].unmet:
            errors.append(f"{capability}: expected unmet requirement")

    for plugin_id, expected_fragment in dict(spec.expectations.get("excluded") or {}).items():
        matched = False
        for resolution in resolutions:
            scored = _candidate_score_by_plugin(resolution, plugin_id)
            if scored is not None and any(
                expected_fragment in reason for reason in scored.exclusion_reasons
            ):
                matched = True
                break
        if not matched:
            errors.append(f"{plugin_id}: expected exclusion containing {expected_fragment!r}")

    not_excluded = spec.expectations.get("not_excluded")
    if isinstance(not_excluded, dict):
        checks = tuple(
            (str(capability), str(plugin_id)) for capability, plugin_id in not_excluded.items()
        )
    elif not_excluded:
        checks = tuple(
            (resolution.requirement.capability, str(not_excluded)) for resolution in resolutions
        )
    else:
        checks = ()
    for capability, plugin_id in checks:
        resolution = by_capability.get(capability)
        scored = (
            _candidate_score_by_plugin(resolution, plugin_id) if resolution is not None else None
        )
        if scored is not None and scored.exclusion_reasons:
            errors.append(
                f"{plugin_id}/{capability}: expected no hard exclusion, got {scored.exclusion_reasons}"
            )

    missing_dependency = spec.expectations.get("missing_dependency")
    if missing_dependency and missing_dependency not in [
        m.capability for m in plan.missing_dependencies
    ]:
        errors.append(f"expected missing dependency {missing_dependency!r}")

    if "cycle_detected" in spec.expectations and plan.cycle_detected != bool(
        spec.expectations["cycle_detected"]
    ):
        errors.append(
            f"expected cycle_detected={spec.expectations['cycle_detected']}, got {plan.cycle_detected}"
        )

    if "executable" in spec.expectations and plan.executable != bool(
        spec.expectations["executable"]
    ):
        errors.append(
            f"expected executable={spec.expectations['executable']}, got {plan.executable}"
        )

    before = spec.expectations.get("plan_before")
    if before:
        order = [step.tool_name for step in plan.steps]
        left, right = before
        if left not in order or right not in order or order.index(left) >= order.index(right):
            errors.append(f"expected plan order {left} before {right}, got {order}")

    return (not errors, errors)


def _run_scenario(spec: ScenarioSpec) -> dict[str, Any]:
    context = _build_context(spec)
    resolver = CapabilityResolver()
    resolutions = resolver.resolve_all(spec.requirements, spec.candidates, context)
    plan = CapabilityPlan.from_scores(_selected_scores(resolutions), plan_id=f"plan-{spec.name}")
    ok, errors = _evaluate_expectations(spec, resolutions, plan)
    return {
        "name": spec.name,
        "description": spec.description,
        "ok": ok,
        "errors": errors,
        "environment": context.environment.to_dict(),
        "requirements": [r.to_dict() for r in spec.requirements],
        "resolutions": [r.to_dict() for r in resolutions],
        "plan": plan.to_dict(),
        "summary": _scenario_summary(resolutions, plan),
    }


def _scenario_summary(resolutions: tuple[Any, ...], plan: CapabilityPlan) -> dict[str, Any]:
    selected = {}
    unmet = []
    excluded: list[dict[str, Any]] = []
    for resolution in resolutions:
        capability = resolution.requirement.capability
        if resolution.selected is None:
            unmet.append(capability)
        else:
            selected[capability] = resolution.selected.candidate.tool_name
        for scored in resolution.candidates:
            if any(
                reason.startswith("candidate does not declare capability")
                for reason in scored.exclusion_reasons
            ):
                continue
            hard_reasons = [
                reason
                for reason in scored.exclusion_reasons
                if not reason.startswith("candidate does not declare capability")
            ]
            if hard_reasons:
                excluded.append(
                    {
                        "capability": capability,
                        "plugin_id": scored.candidate.plugin_id,
                        "tool_name": scored.candidate.tool_name,
                        "reasons": hard_reasons,
                    }
                )
    return {
        "selected": selected,
        "unmet": unmet,
        "excluded": excluded,
        "plan_order": [step.tool_name for step in plan.steps],
        "plan_executable": plan.executable,
        "cycle_detected": plan.cycle_detected,
        "missing_dependencies": [m.to_dict() for m in plan.missing_dependencies],
    }


def _comparisons(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {item["name"]: item for item in scenarios}
    result: list[dict[str, Any]] = []
    baseline = by_name.get("file_ops_only")
    shell = by_name.get("shell_enabled")
    node = by_name.get("node_workspace_marker")
    if baseline and shell:
        baseline_shell = _find_exclusion(baseline, "json_shell", capability="json.pretty")
        shell_shell = _find_exclusion(shell, "json_shell", capability="json.pretty")
        result.append(
            {
                "name": "environment_capability_delta",
                "ok": bool(baseline_shell) and not bool(shell_shell),
                "baseline_exclusion": baseline_shell,
                "shell_enabled_exclusion": shell_shell,
            }
        )
    if baseline and node:
        result.append(
            {
                "name": "workspace_marker_delta",
                "ok": baseline["environment"]["fingerprint_id"]
                != node["environment"]["fingerprint_id"],
                "baseline_markers": baseline["environment"]["workspace_markers"],
                "node_markers": node["environment"]["workspace_markers"],
            }
        )
    return result


def _find_exclusion(scenario: dict[str, Any], plugin_id: str, *, capability: str = "") -> list[str]:
    for item in scenario["summary"].get("excluded") or []:
        if item.get("plugin_id") != plugin_id:
            continue
        if capability and item.get("capability") != capability:
            continue
        return list(item.get("reasons") or [])
    return []


def _plugin_source(plugin: Any) -> str:
    """Classify a registry plugin source for report segmentation."""
    if getattr(plugin, "__leapflow_plugin_path__", ""):
        return "profile"
    module_name = str(getattr(plugin.__class__, "__module__", ""))
    if module_name.startswith("leapflow.plugins.tool_plugins."):
        return "builtin"
    return "external"


def _ratio(part: int, total: int) -> float:
    """Return a stable four-decimal coverage ratio."""
    return round(part / total, 4) if total else 0.0


def _real_registry_snapshot() -> dict[str, Any]:
    """Inspect live registry candidates and capability metadata coverage.

    This is a P1 experiment section, not a framework mutation. It assembles the
    in-process registry and reports data quality gaps in ToolMetadata, segmented
    by source so built-in coverage is not obscured by profile-scoped plugins.
    """
    from leapflow.plugins import get_registry

    registry = get_registry()
    registry.assemble()
    candidates = candidates_from_registry(registry)
    plugin_sources = {
        plugin_id: _plugin_source(plugin) for plugin_id, plugin in registry.plugins.items()
    }
    by_plugin: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, Any]] = []
    provides_count = 0
    requires_count = 0
    platform_count = 0
    approval_count = 0
    mutating_count = 0
    for candidate in candidates:
        source = plugin_sources.get(candidate.plugin_id, "external")
        plugin_bucket = by_plugin.setdefault(
            candidate.plugin_id,
            {
                "source": source,
                "tool_count": 0,
                "provides_count": 0,
                "platform_requirement_count": 0,
            },
        )
        source_bucket = by_source.setdefault(
            source,
            {
                "plugin_ids": set(),
                "candidate_count": 0,
                "provides_count": 0,
                "platform_requirement_count": 0,
                "metadata_gap_count": 0,
            },
        )
        source_bucket["plugin_ids"].add(candidate.plugin_id)
        plugin_bucket["tool_count"] += 1
        source_bucket["candidate_count"] += 1
        if candidate.provides_capabilities:
            provides_count += 1
            plugin_bucket["provides_count"] += 1
            source_bucket["provides_count"] += 1
        if candidate.requires_capabilities:
            requires_count += 1
        if candidate.requires_platform_capabilities:
            platform_count += 1
            plugin_bucket["platform_requirement_count"] += 1
            source_bucket["platform_requirement_count"] += 1
        if candidate.requires_approval:
            approval_count += 1
        if candidate.mutates_state:
            mutating_count += 1
        missing: list[str] = []
        if not candidate.provides_capabilities:
            missing.append("provides_capabilities")
        if candidate.mutates_state and not candidate.requires_platform_capabilities:
            missing.append("requires_platform_capabilities_for_mutating_tool")
        if missing:
            source_bucket["metadata_gap_count"] += 1
            gaps.append(
                {
                    "source": source,
                    "plugin_id": candidate.plugin_id,
                    "tool_name": candidate.tool_name,
                    "risk_level": candidate.risk_level,
                    "missing": missing,
                }
            )

    source_coverage = {}
    for source, stats in sorted(by_source.items()):
        candidate_count = int(stats["candidate_count"])
        source_coverage[source] = {
            "plugin_count": len(stats["plugin_ids"]),
            "candidate_count": candidate_count,
            "declared_provides_count": int(stats["provides_count"]),
            "declared_platform_requirements_count": int(stats["platform_requirement_count"]),
            "metadata_gap_count": int(stats["metadata_gap_count"]),
            "provides_ratio": _ratio(int(stats["provides_count"]), candidate_count),
            "platform_requirement_ratio": _ratio(
                int(stats["platform_requirement_count"]), candidate_count
            ),
        }

    return {
        "candidate_count": len(candidates),
        "plugin_count": len(by_plugin),
        "declared_provides_count": provides_count,
        "declared_requires_count": requires_count,
        "declared_platform_requirements_count": platform_count,
        "approval_required_count": approval_count,
        "mutating_count": mutating_count,
        "coverage": {
            "provides_ratio": _ratio(provides_count, len(candidates)),
            "platform_requirement_ratio": _ratio(platform_count, len(candidates)),
        },
        "source_coverage": source_coverage,
        "plugins": dict(sorted(by_plugin.items())),
        "metadata_gap_count": len(gaps),
        "metadata_gaps": gaps,
        "top_metadata_gaps": gaps[:30],
        "conflict_count": len(getattr(registry, "conflicts", [])),
        "conflicts": [
            {
                "tool_name": conflict.tool_name,
                "kept_plugin": conflict.kept_plugin,
                "rejected_plugin": conflict.rejected_plugin,
            }
            for conflict in getattr(registry, "conflicts", [])
        ],
    }


def _strategy() -> tuple[ExperimentStrategyItem, ...]:
    return (
        ExperimentStrategyItem(
            "P0",
            "Scenario matrix in temp/plugin_exp",
            "Cover deterministic environment / trust / risk / dependency variations without framework changes.",
        ),
        ExperimentStrategyItem(
            "P1",
            "Real registry candidate source",
            "Use candidates_from_registry(get_registry()) to validate live ToolMetadata declarations.",
        ),
        ExperimentStrategyItem(
            "P1",
            "Unknown-tool evidence ingestion",
            "Feed real unknown_tool payloads through CapabilityGapDetector.requirements_from_tool_results().",
        ),
        ExperimentStrategyItem(
            "P2",
            "Runtime registry mutation smoke",
            "Install/disable fixture plugins and prove adaptive decisions change with live catalog state.",
            needs_framework_change=True,
        ),
        ExperimentStrategyItem(
            "P2",
            "LeapBoard capability view smoke",
            "Render capability.yaml with stored records and verify user-facing transparency.",
            needs_framework_change=True,
        ),
        ExperimentStrategyItem(
            "P3",
            "Autonomous closed-loop governance",
            "Connect requirement observation, resolver, approval, plugin generation, install, and post-use feedback.",
            needs_framework_change=True,
        ),
    )


def _closed_loop_plugin_code() -> str:
    """Return deterministic plugin source used by the closed-loop experiment."""
    return """from __future__ import annotations

import json
from typing import Any

from leapflow.plugins.protocol import ToolMetadata


async def json_pretty_loop(text: str = "", **kwargs: Any) -> dict[str, Any]:
    payload = text or kwargs.get("payload") or "{}"
    try:
        parsed = json.loads(str(payload))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "content": json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)}


class JsonPrettyLoopPlugin:
    @property
    def plugin_id(self) -> str:
        return "json_pretty_loop_plugin"

    @property
    def category(self) -> str:
        return "formatting"

    @property
    def dependencies(self) -> list[str]:
        return []

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="json_pretty_loop",
                description="Pretty-print a JSON string for adaptive closed-loop experiments.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "JSON text to format"},
                        "payload": {"type": "string", "description": "Alternative JSON text field"},
                    },
                },
                handler=json_pretty_loop,
                x_leapflow={
                    "category": "formatting",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("json.pretty",),
                requires_platform_capabilities=("file.ops",),
            )
        ]

    def bind_runtime(self, **deps: Any) -> None:
        return None


plugin = JsonPrettyLoopPlugin()
"""


class _ClosedLoopAllowGate:
    """Approval gate used only inside the isolated experiment registry."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def evaluate(self, action: Any) -> Any:
        self.requests.append(
            {
                "platform": getattr(action, "platform", ""),
                "action": getattr(action, "action", ""),
                "payload": dict(getattr(action, "payload", {}) or {}),
            }
        )

        class Decision:
            approved = True
            denial_message = ""

        return Decision()


def _closed_loop_requirement_text(plugin_id: str) -> str:
    """Return the explicit live-generation contract for the experiment."""
    return (
        "Create a read-only LeapFlow ToolPlugin for adaptive plugin closed-loop testing. "
        f"The plugin_id must be {plugin_id!r}. Expose exactly one async tool named "
        "json_pretty_live that accepts **kwargs and reads a JSON string from the 'text' "
        "argument, returning {'ok': True, 'content': <pretty JSON>} with indent=2 and "
        "sort_keys=True. On JSON parse errors return {'ok': False, 'error': <message>}. "
        "The ToolMetadata must set x_leapflow with category='formatting', "
        "risk_level='read_only', schema_cost='low', requires_approval=False, and must "
        "declare provides_capabilities=('json.pretty',) plus "
        "requires_platform_capabilities=('file.ops',). Do not perform file, network, "
        "shell, subprocess, eval, exec, or import-time side effects."
    )


def _build_live_generation_provider() -> tuple[Any | None, dict[str, Any]]:
    """Build an OpenAI-compatible provider from the real default profile config."""
    try:
        from leapflow.config_loader import load_config_bundle
        from leapflow.layout import PathLayout
        from leapflow.llm.openai_provider import OpenAIChat

        layout = PathLayout(Path.home() / ".leapflow")
        profile_layout = layout.profile("default")
        bundle = load_config_bundle(layout, profile_layout, REPO_ROOT)
        llm = bundle.values.get("llm") or {}
        missing = [key for key in ("base_url", "api_key", "model") if not llm.get(key)]
        if missing:
            return None, {"ok": False, "stage": "config", "missing": missing}
        provider = OpenAIChat(
            api_key=str(llm["api_key"]),
            base_url=str(llm["base_url"]),
            model=str(llm["model"]),
            max_retries=int(llm.get("max_retries") or 2),
        )
        return provider, {
            "ok": True,
            "stage": "config",
            "profile": "default",
            "model": str(llm.get("model") or ""),
            "base_url_configured": bool(llm.get("base_url")),
            "api_key_configured": bool(llm.get("api_key")),
            "config_warnings_count": len(bundle.warnings),
        }
    except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
        return None, {"ok": False, "stage": "config", "error": str(exc)}


async def _generate_closed_loop_plugin_code(plugin_id: str) -> dict[str, Any]:
    """Generate and validate the closed-loop plugin with the real configured LLM."""
    from leapflow.learning.plugin_generator import PluginGenerationRequest, PluginGenerator

    provider, config_payload = _build_live_generation_provider()
    if provider is None:
        return {"ok": False, "mode": "live_generation", "config": config_payload}
    generator = PluginGenerator(llm_provider=provider)
    attempts: list[dict[str, Any]] = []
    description = _closed_loop_requirement_text(plugin_id)
    for index in range(2):
        request = PluginGenerationRequest(plugin_id=plugin_id, description=description)
        started = time.perf_counter()
        result = await generator.generate_and_validate(request)
        attempt = {
            "attempt": index + 1,
            "ok": bool(result.get("ok")),
            "stage": str(result.get("stage") or ""),
            "duration_s": round(time.perf_counter() - started, 3),
            "exposed_tools": list(result.get("exposed_tools") or []),
            "error": str(result.get("error") or ""),
        }
        attempts.append(attempt)
        if result.get("ok"):
            return {
                "ok": True,
                "mode": "live_generation",
                "config": config_payload,
                "attempts": attempts,
                "code": str(result.get("code") or ""),
                "exposed_tools": list(result.get("exposed_tools") or []),
            }
        description = (
            _closed_loop_requirement_text(plugin_id)
            + "\n\nPrevious validation failed. Fix this exact problem: "
            + str(result.get("error") or "unknown validation error")[:800]
        )
    return {
        "ok": False,
        "mode": "live_generation",
        "config": config_payload,
        "attempts": attempts,
        "error": attempts[-1].get("error", "generation failed")
        if attempts
        else "generation failed",
    }


async def _run_closed_loop_experiment(*, live_generation: bool = True) -> dict[str, Any]:
    """Run an isolated install→select→execute→disable→remove registry loop."""
    import leapflow.plugins as plugin_api
    from leapflow.learning.plugin_stats import PluginUsageTracker
    from leapflow.learning.plugin_trust import PluginTrustLedger
    from leapflow.plugins.adaptive_loop import (
        AdaptiveLoopRequest,
        AdaptivePluginLoop,
        SelfManagementLifecycleActor,
    )
    from leapflow.plugins.handler_invocation import invoke_tool_handler
    from leapflow.plugins.registry import ToolPluginRegistry
    from leapflow.plugins.scoped_registry import ScopedToolRegistry
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin
    from leapflow.storage.plugin_version_store import PluginVersionStore

    loop_id = f"closed-loop-{_timestamp()}"
    work_root = EXP_ROOT / "work" / loop_id
    install_dir = work_root / "profile" / "plugins"
    plan_store = JsonCapabilityPlanStore(
        work_root / "profile" / "plugins" / "capability_plans.json"
    )
    version_store = PluginVersionStore(work_root / "profile" / "plugins" / "versions")
    approval_gate = _ClosedLoopAllowGate()

    registry = ToolPluginRegistry()
    self_management = SelfManagementPlugin()
    self_management.bind_runtime(
        plugin_approval_gate=approval_gate,
        plugin_install_dir=str(install_dir),
        plugin_version_store=version_store,
        capability_plan_store=plan_store,
    )
    registry.register(self_management)
    registry.assemble()
    scoped = ScopedToolRegistry(registry)
    scoped.adopt_existing_plugins()

    old_registry = getattr(plugin_api, "_registry", None)
    old_scoped = getattr(plugin_api, "_scoped_registry", None)
    plugin_api._registry = registry
    plugin_api._scoped_registry = scoped
    try:
        requirement = _req(
            "json.pretty",
            "explicit_request",
            evidence="Closed-loop experiment requires a live json.pretty provider.",
        )
        environment = EnvironmentProbe(workspace_markers=("pyproject.toml",)).probe(
            platform_manifest=_manifest(Capability.FILE_OPS),
            workspace_root=work_root,
        )
        plugin_id = "json_pretty_live_plugin" if live_generation else "json_pretty_loop_plugin"
        trust = PluginTrustLedger(candidate_at=1, verified_at=2, production_at=3)
        usage = PluginUsageTracker()
        usage._get_reverse_index = lambda: {
            "json_pretty_live": plugin_id,
            "json_pretty_loop": plugin_id,
        }
        actor = SelfManagementLifecycleActor(self_management)
        loop = AdaptivePluginLoop(
            registry=registry,
            plan_store=plan_store,
            lifecycle_actor=actor,
            trust_ledger=trust,
            usage_tracker=usage,
        )
        request = AdaptiveLoopRequest(
            environment=environment,
            requirements=(requirement,),
            source="temp_plugin_exp_closed_loop",
            loop_id=loop_id,
        )

        phases: list[dict[str, Any]] = []

        before = loop.resolve_once(request, loop_id=loop_id, phase="before")
        phases.append(_closed_loop_phase("before", before))

        generation_payload: dict[str, Any]
        if live_generation:
            generation_payload = await _generate_closed_loop_plugin_code(plugin_id)
            if not generation_payload.get("ok"):
                return {
                    "ok": False,
                    "loop_id": loop_id,
                    "mode": "live_generation",
                    "live_generation": generation_payload,
                    "isolated_root": str(work_root),
                    "plan_store": str(plan_store.path),
                    "approval_requests": approval_gate.requests,
                    "selected_by_phase": {phase["phase"]: phase["selected"] for phase in phases},
                    "registry_version_final": registry.version,
                    "cleanup": {"source_exists_after_remove": False},
                    "phases": phases,
                }
            plugin_code = str(generation_payload.get("code") or "")
            version_label = "closed-loop-live-generation"
        else:
            generation_payload = {
                "ok": True,
                "mode": "fixture",
                "exposed_tools": ["json_pretty_loop"],
            }
            plugin_code = _closed_loop_plugin_code()
            version_label = "closed-loop-fixture"

        install_result = await actor.install(
            plugin_id=plugin_id,
            code=plugin_code,
            version_label=version_label,
        )
        after_install = loop.resolve_once(
            request,
            loop_id=loop_id,
            phase="after_install",
            mutation={"action": "install", "plugin_id": plugin_id},
            registry_version_before=before.registry_version,
            registry_version_after=registry.version,
        )
        phases.append(_closed_loop_phase("after_install", after_install, install_result))

        selected_tool = ""
        if after_install.resolutions[0].selected is not None:
            selected_tool = after_install.resolutions[0].selected.candidate.tool_name
        execution_result: dict[str, Any] = {"ok": False, "error": "tool not installed"}
        handler = registry.tool_handlers.get(selected_tool)
        if handler is not None:
            started = time.perf_counter()
            execution_result = await invoke_tool_handler(handler, {"text": '{"b":2,"a":1}'})
            duration_ms = (time.perf_counter() - started) * 1000
            usage.record(selected_tool, bool(execution_result.get("ok", False)), duration_ms)
            if execution_result.get("ok"):
                trust.record_success(plugin_id)
        after_execute = loop.resolve_once(
            request,
            loop_id=loop_id,
            phase="after_execute",
            mutation={"action": "execute", "plugin_id": plugin_id, "tool_name": selected_tool},
            registry_version_before=registry.version,
            registry_version_after=registry.version,
        )
        phases.append(_closed_loop_phase("after_execute", after_execute, execution_result))

        disable_result = await actor.disable(plugin_id=plugin_id)
        after_disable = loop.resolve_once(
            request,
            loop_id=loop_id,
            phase="after_disable",
            mutation={"action": "disable", "plugin_id": plugin_id},
            registry_version_before=after_execute.registry_version,
            registry_version_after=registry.version,
        )
        phases.append(_closed_loop_phase("after_disable", after_disable, disable_result))

        remove_result = await actor.remove(plugin_id=plugin_id, delete_source=True)
        after_remove = loop.resolve_once(
            request,
            loop_id=loop_id,
            phase="after_remove",
            mutation={"action": "remove", "plugin_id": plugin_id},
            registry_version_before=after_disable.registry_version,
            registry_version_after=registry.version,
        )
        phases.append(_closed_loop_phase("after_remove", after_remove, remove_result))

        source_path = install_dir / f"{plugin_id}.py"
        selected_by_phase = {phase["phase"]: phase["selected"] for phase in phases}
        ok = (
            before.resolutions[0].selected is None
            and generation_payload.get("ok") is True
            and install_result.get("ok") is True
            and after_install.resolutions[0].selected is not None
            and execution_result.get("ok") is True
            and disable_result.get("ok") is True
            and after_disable.resolutions[0].selected is None
            and remove_result.get("ok") is True
            and after_remove.resolutions[0].selected is None
            and not source_path.exists()
        )
        return {
            "ok": ok,
            "loop_id": loop_id,
            "mode": "live_generation" if live_generation else "fixture",
            "live_generation": generation_payload,
            "isolated_root": str(work_root),
            "plan_store": str(plan_store.path),
            "approval_requests": approval_gate.requests,
            "selected_by_phase": selected_by_phase,
            "registry_version_final": registry.version,
            "cleanup": {"source_exists_after_remove": source_path.exists()},
            "phases": phases,
        }
    finally:
        plugin_api._registry = old_registry
        plugin_api._scoped_registry = old_scoped


def _closed_loop_phase(
    phase: str,
    decision: Any,
    action_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = {}
    for resolution in decision.resolutions:
        if resolution.selected is None:
            continue
        selected[resolution.requirement.capability] = resolution.selected.candidate.tool_name
    return {
        "phase": phase,
        "record_id": decision.record.get("record_id", ""),
        "registry_version": decision.registry_version,
        "candidate_count": len(decision.candidates),
        "selected": selected,
        "executable": decision.plan.executable,
        "plan_order": [step.tool_name for step in decision.plan.steps],
        "action_result": dict(action_result or {}),
    }


async def _run_autonomous_long_run(*, live_generation: bool = True) -> dict[str, Any]:
    """Run a long-horizon autonomous evolution governance scenario."""
    from leapflow.analysis.environment_catalog import EnvironmentCatalog, EnvironmentMarker
    from leapflow.learning.capability_observation import (
        CapabilityObservationBuffer,
        CapabilityObservationService,
    )
    from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
    from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
    from leapflow.plugins.lifecycle_governor import LifecycleGovernor
    from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore
    from leapflow.storage.capability_proposal_queue import JsonCapabilityProposalQueue
    from leapflow.storage.plugin_outcome_store import JsonPluginOutcomeStore

    run_id = f"autonomous-long-run-{_timestamp()}"
    work_root = EXP_ROOT / "work" / run_id
    workspace = work_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "pyproject.toml").write_text(
        "[project]\nname='autonomous-demo'\n", encoding="utf-8"
    )
    catalog = EnvironmentCatalog.from_markers(
        (
            EnvironmentMarker(
                "pyproject.toml", category="language", source="experiment", tags=("python",)
            ),
            EnvironmentMarker(
                "package.json", category="language", source="experiment", tags=("node",)
            ),
        )
    )
    environment = EnvironmentProbe.from_catalog(catalog).probe(
        platform_manifest=_manifest(Capability.FILE_OPS),
        workspace_root=workspace,
        catalog=catalog,
    )

    observation_store = JsonCapabilityObservationStore(work_root / "observations.json")
    observation_service = CapabilityObservationService(observation_store)
    observation_buffer = CapabilityObservationBuffer()
    for _ in range(2):
        observation_buffer.add_result(
            {
                "error_type": "unknown_tool",
                "original_tool_name": "json_pretty_live",
                "suggestions": ["plugin_generate"],
                "recovery_hint": "No live JSON pretty plugin is registered yet.",
            }
        )
    observation_records = observation_service.flush_buffer(
        observation_buffer,
        environment=environment,
        source="temp_plugin_exp_long_run",
        session_id=run_id,
        turn_id="turn-observe",
        workspace_root=str(workspace),
    )
    requirements = observation_service.requirements(min_count=2)

    proposal_queue = JsonCapabilityProposalQueue(work_root / "proposals.json")
    proposal = proposal_queue.enqueue(
        requirements=requirements,
        environment=environment.to_dict(),
        risk={"risk_level": "read_only"},
        source="temp_plugin_exp_long_run",
        observation_ids=tuple(str(record.get("observation_id")) for record in observation_records),
        metadata={
            "plugin_id": "json_pretty_live_plugin" if live_generation else "json_pretty_loop_plugin"
        },
    )
    policy = AdaptiveEvolutionPolicy(autonomy_level="trusted_autonomous")
    decisions = []
    initial_decision = policy.decide(proposal)
    decisions.append(initial_decision.to_dict())
    proposal = proposal_queue.update(proposal.proposal_id, status="GENERATED") or proposal
    install_decision = policy.decide(proposal, sandbox_validated=True)
    decisions.append(install_decision.to_dict())

    closed_loop = await _run_closed_loop_experiment(live_generation=live_generation)

    outcome_store = JsonPluginOutcomeStore(work_root / "outcomes.json")
    governor = LifecycleGovernor(
        proposal_queue=proposal_queue,
        outcome_store=outcome_store,
        trust_ledger=PluginTrustLedger(candidate_at=1, verified_at=2, production_at=3),
        quarantine_after=2,
        verified_at=PluginTrustLevel.VERIFIED,
    )
    plugin_id = "json_pretty_live_plugin" if live_generation else "json_pretty_loop_plugin"
    selected_tool = (
        (closed_loop.get("selected_by_phase") or {}).get("after_install", {}).get("json.pretty", "")
    )
    governance_results = []
    if closed_loop.get("ok") and selected_tool:
        proposal_queue.update(
            proposal.proposal_id,
            status="INSTALLED",
            install_result={"ok": True, "plugin_id": plugin_id},
        )
        for idx in range(2):
            governance_results.append(
                (
                    await governor.record_outcome(
                        proposal_id=proposal.proposal_id,
                        plugin_id=plugin_id,
                        tool_name=selected_tool,
                        ok=True,
                        requirement_id=requirements[0].requirement_id if requirements else "",
                        plan_id=str(
                            (closed_loop.get("phases") or [{}])[
                                min(idx + 1, len(closed_loop.get("phases") or [{}]) - 1)
                            ].get("record_id")
                            or ""
                        ),
                        duration_ms=5.0,
                    )
                ).to_dict()
            )
        for _ in range(2):
            governance_results.append(
                (
                    await governor.record_outcome(
                        proposal_id=proposal.proposal_id,
                        plugin_id=plugin_id,
                        tool_name=selected_tool,
                        ok=False,
                        failure_class="synthetic_probation_failure",
                    )
                ).to_dict()
            )

    final_proposal = proposal_queue.get(proposal.proposal_id)
    return {
        "ok": bool(requirements and closed_loop.get("ok") and final_proposal is not None),
        "run_id": run_id,
        "environment": environment.to_dict(),
        "observation_records": list(observation_records),
        "requirements": [requirement.to_dict() for requirement in requirements],
        "proposal": final_proposal.to_dict() if final_proposal is not None else proposal.to_dict(),
        "policy_decisions": decisions,
        "closed_loop": closed_loop,
        "governance_results": governance_results,
        "stores": {
            "observations": str(observation_store.path),
            "proposals": str(proposal_queue.path),
            "outcomes": str(outcome_store.path),
        },
    }


def run(
    *,
    closed_loop: bool = False,
    live_generation: bool = True,
    autonomous_long_run: bool = False,
) -> dict[str, Any]:
    specs = _scenarios()
    scenario_payloads = [_run_scenario(spec) for spec in specs]
    comparisons = _comparisons(scenario_payloads)
    real_registry_snapshot = _real_registry_snapshot()
    suite_ok = all(item["ok"] for item in scenario_payloads) and all(
        item["ok"] for item in comparisons
    )
    store = JsonCapabilityPlanStore(EXP_ROOT / "work" / "capability_plans.json")
    run_id = f"adaptive-matrix-{_timestamp()}"
    for scenario in scenario_payloads:
        store.add_record(
            environment=scenario["environment"],
            requirements=scenario["requirements"],
            resolutions=scenario["resolutions"],
            plan=scenario["plan"],
            source="temp_plugin_exp_matrix",
            record_id=f"{run_id}:{scenario['name']}",
        )
    payload = {
        "ok": suite_ok,
        "run_id": run_id,
        "scenario_count": len(scenario_payloads),
        "passed": sum(1 for item in scenario_payloads if item["ok"]),
        "failed": [item["name"] for item in scenario_payloads if not item["ok"]],
        "scenarios": scenario_payloads,
        "comparisons": comparisons,
        "real_registry_snapshot": real_registry_snapshot,
        "strategy": [item.to_dict() for item in _strategy()],
    }
    if closed_loop:
        closed_loop_payload = asyncio.run(
            _run_closed_loop_experiment(live_generation=live_generation)
        )
        payload["closed_loop"] = closed_loop_payload
        payload["ok"] = bool(payload["ok"] and closed_loop_payload.get("ok"))
    if autonomous_long_run:
        autonomous_payload = asyncio.run(_run_autonomous_long_run(live_generation=live_generation))
        payload["autonomous_long_run"] = autonomous_payload
        payload["ok"] = bool(payload["ok"] and autonomous_payload.get("ok"))
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Adaptive Plugin Scenario Matrix Report",
        "",
        f"- Overall: {'PASS' if payload['ok'] else 'FAIL'}",
        f"- Run ID: `{payload['run_id']}`",
        f"- Scenarios: {payload['passed']}/{payload['scenario_count']} passed",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | OK | Selected | Plan | Key exclusions / diagnostics |",
        "|---|---:|---|---|---|",
    ]
    for scenario in payload["scenarios"]:
        summary = scenario["summary"]
        selected = ", ".join(f"{cap}->{tool}" for cap, tool in summary["selected"].items()) or "-"
        plan = " -> ".join(summary["plan_order"]) or "-"
        diagnostics = []
        if summary["unmet"]:
            diagnostics.append("unmet=" + ",".join(summary["unmet"]))
        if summary["missing_dependencies"]:
            diagnostics.append(
                "missing="
                + ",".join(item["capability"] for item in summary["missing_dependencies"])
            )
        if summary["cycle_detected"]:
            diagnostics.append("cycle_detected")
        if summary["excluded"]:
            diagnostics.append(
                "excluded="
                + "; ".join(
                    f"{item['plugin_id']}:{','.join(item['reasons'])}"
                    for item in summary["excluded"][:2]
                )
            )
        if scenario["errors"]:
            diagnostics.append("errors=" + "; ".join(scenario["errors"]))
        lines.append(
            f"| `{scenario['name']}` | {scenario['ok']} | {selected} | {plan} | {'; '.join(diagnostics) or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Cross-scenario Comparisons",
            "",
            "| Comparison | OK | Evidence |",
            "|---|---:|---|",
        ]
    )
    for item in payload["comparisons"]:
        evidence = ", ".join(f"{k}={v}" for k, v in item.items() if k not in {"name", "ok"})
        lines.append(f"| `{item['name']}` | {item['ok']} | {evidence} |")

    registry_snapshot = payload.get("real_registry_snapshot") or {}
    coverage = registry_snapshot.get("coverage") or {}
    source_coverage = registry_snapshot.get("source_coverage") or {}
    lines.extend(
        [
            "",
            "## Real Registry Candidate Snapshot",
            "",
            f"- Candidates: {registry_snapshot.get('candidate_count', 0)}",
            f"- Plugins: {registry_snapshot.get('plugin_count', 0)}",
            f"- Declared `provides_capabilities`: {registry_snapshot.get('declared_provides_count', 0)} "
            f"({coverage.get('provides_ratio', 0.0)})",
            f"- Declared `requires_platform_capabilities`: {registry_snapshot.get('declared_platform_requirements_count', 0)} "
            f"({coverage.get('platform_requirement_ratio', 0.0)})",
            f"- Metadata gaps: {registry_snapshot.get('metadata_gap_count', 0)}",
            f"- Tool-name conflicts: {registry_snapshot.get('conflict_count', 0)}",
            "",
            "### Source Coverage",
            "",
            "| Source | Plugins | Candidates | Provides coverage | Platform coverage | Gaps |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source, stats in sorted(source_coverage.items()):
        lines.append(
            f"| `{source}` | {stats.get('plugin_count', 0)} | {stats.get('candidate_count', 0)} | "
            f"{stats.get('declared_provides_count', 0)} ({stats.get('provides_ratio', 0.0)}) | "
            f"{stats.get('declared_platform_requirements_count', 0)} "
            f"({stats.get('platform_requirement_ratio', 0.0)}) | {stats.get('metadata_gap_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "### Metadata Gaps",
            "",
            "| Source | Plugin | Tool | Risk | Missing metadata |",
            "|---|---|---|---|---|",
        ]
    )
    for gap in (
        registry_snapshot.get("top_metadata_gaps") or registry_snapshot.get("metadata_gaps") or []
    )[:12]:
        lines.append(
            f"| `{gap.get('source', 'external')}` | `{gap.get('plugin_id')}` | `{gap.get('tool_name')}` | "
            f"{gap.get('risk_level')} | {', '.join(gap.get('missing') or [])} |"
        )

    closed_loop = payload.get("closed_loop") or {}
    if closed_loop:
        lines.extend(
            [
                "",
                "## Closed-loop Mutation Timeline",
                "",
                f"- Overall: {'PASS' if closed_loop.get('ok') else 'FAIL'}",
                f"- Loop ID: `{closed_loop.get('loop_id')}`",
                f"- Mode: `{closed_loop.get('mode', 'fixture')}`",
                f"- Isolated root: `{closed_loop.get('isolated_root')}`",
                "",
                "| Phase | Registry version | Candidates | Selected | Executable | Action result |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for phase in closed_loop.get("phases") or []:
            selected = (
                ", ".join(f"{cap}->{tool}" for cap, tool in (phase.get("selected") or {}).items())
                or "-"
            )
            action = phase.get("action_result") or {}
            action_text = action.get("action") or (
                "ok" if action.get("ok") else action.get("error", "-")
            )
            lines.append(
                f"| `{phase.get('phase')}` | {phase.get('registry_version')} | "
                f"{phase.get('candidate_count')} | {selected} | {phase.get('executable')} | {action_text or '-'} |"
            )

    autonomous = payload.get("autonomous_long_run") or {}
    if autonomous:
        proposal = autonomous.get("proposal") or {}
        lines.extend(
            [
                "",
                "## Autonomous Long-run Governance",
                "",
                f"- Overall: {'PASS' if autonomous.get('ok') else 'FAIL'}",
                f"- Run ID: `{autonomous.get('run_id')}`",
                f"- Proposal: `{proposal.get('proposal_id', '')}` ({proposal.get('status', '')})",
                f"- Observations: {len(autonomous.get('observation_records') or [])}",
                f"- Governance events: {len(autonomous.get('governance_results') or [])}",
                "",
                "| Step | Action | Reason | Approval |",
                "|---|---|---|---:|",
            ]
        )
        for idx, decision in enumerate(autonomous.get("policy_decisions") or [], start=1):
            lines.append(
                f"| {idx} | `{decision.get('action')}` | {decision.get('reason', '')} | "
                f"{decision.get('requires_approval')} |"
            )
        lines.extend(
            [
                "",
                "| Governance | Plugin | Trust | Failure streak |",
                "|---|---|---|---:|",
            ]
        )
        for item in autonomous.get("governance_results") or []:
            lines.append(
                f"| `{item.get('action')}` | `{item.get('plugin_id')}` | "
                f"{item.get('trust_level')} | {item.get('failure_streak')} |"
            )

    lines.extend(
        [
            "",
            "## Roadmap",
            "",
            "| Priority | Title | Goal | Framework change? |",
            "|---|---|---|---:|",
        ]
    )
    for item in payload["strategy"]:
        lines.append(
            f"| {item['priority']} | {item['title']} | {item['goal']} | {item['needs_framework_change']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_metadata_gap_markdown(payload: dict[str, Any]) -> str:
    """Render the complete real-registry metadata gap audit."""
    snapshot = payload.get("real_registry_snapshot") or {}
    gaps = snapshot.get("metadata_gaps") or []
    source_coverage = snapshot.get("source_coverage") or {}
    lines = [
        "# Real Registry Metadata Gap Report",
        "",
        f"- Candidates: {snapshot.get('candidate_count', 0)}",
        f"- Plugins: {snapshot.get('plugin_count', 0)}",
        f"- Metadata gaps: {snapshot.get('metadata_gap_count', 0)}",
        f"- Tool-name conflicts: {snapshot.get('conflict_count', 0)}",
        "",
        "## Source Coverage",
        "",
        "| Source | Plugins | Candidates | Provides coverage | Platform coverage | Gaps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, stats in sorted(source_coverage.items()):
        lines.append(
            f"| `{source}` | {stats.get('plugin_count', 0)} | {stats.get('candidate_count', 0)} | "
            f"{stats.get('declared_provides_count', 0)} ({stats.get('provides_ratio', 0.0)}) | "
            f"{stats.get('declared_platform_requirements_count', 0)} "
            f"({stats.get('platform_requirement_ratio', 0.0)}) | {stats.get('metadata_gap_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Gap Table",
            "",
            "| Source | Plugin | Tool | Risk | Missing metadata |",
            "|---|---|---|---|---|",
        ]
    )
    for gap in gaps:
        lines.append(
            f"| `{gap.get('source', 'external')}` | `{gap.get('plugin_id')}` | `{gap.get('tool_name')}` | "
            f"{gap.get('risk_level')} | {', '.join(gap.get('missing') or [])} |"
        )
    lines.extend(
        [
            "",
            "## Suggested Next Capability Metadata Pass",
            "",
            "1. Start with high-value read-only tools (`file_read`, `code_search`, `repo_map`, `git_query`).",
            "2. Add `provides_capabilities` before using real registry candidates in selection scenarios.",
            "3. Add `requires_platform_capabilities` to mutating / external tools before enabling environment-fit decisions.",
            "4. Keep this report as the before/after audit for metadata coverage improvements.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(payload: dict[str, Any]) -> str:
    """Render a self-contained HTML dashboard for the experiment."""
    scenario_rows = []
    for scenario in payload.get("scenarios") or []:
        summary = scenario.get("summary") or {}
        selected = (
            ", ".join(f"{cap} → {tool}" for cap, tool in (summary.get("selected") or {}).items())
            or "-"
        )
        plan = " → ".join(summary.get("plan_order") or []) or "-"
        diagnostics = []
        if summary.get("unmet"):
            diagnostics.append("unmet: " + ", ".join(summary["unmet"]))
        if summary.get("missing_dependencies"):
            diagnostics.append("missing deps")
        if summary.get("cycle_detected"):
            diagnostics.append("cycle")
        if summary.get("excluded"):
            diagnostics.append("hard exclusions")
        scenario_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(scenario.get('name')))}</code></td>"
            f"<td class={'ok' if scenario.get('ok') else 'bad'}>{scenario.get('ok')}</td>"
            f"<td>{html.escape(selected)}</td>"
            f"<td>{html.escape(plan)}</td>"
            f"<td>{html.escape('; '.join(diagnostics) or '-')}</td>"
            "</tr>"
        )

    snapshot = payload.get("real_registry_snapshot") or {}
    coverage = snapshot.get("coverage") or {}
    source_coverage = snapshot.get("source_coverage") or {}
    builtin_stats = source_coverage.get("builtin") or {}
    profile_stats = source_coverage.get("profile") or {}
    source_rows = []
    for source, stats in sorted(source_coverage.items()):
        source_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(source))}</code></td>"
            f"<td>{stats.get('plugin_count', 0)}</td>"
            f"<td>{stats.get('candidate_count', 0)}</td>"
            f"<td>{stats.get('declared_provides_count', 0)} ({stats.get('provides_ratio', 0.0)})</td>"
            f"<td>{stats.get('declared_platform_requirements_count', 0)} "
            f"({stats.get('platform_requirement_ratio', 0.0)})</td>"
            f"<td>{stats.get('metadata_gap_count', 0)}</td>"
            "</tr>"
        )
    gaps = snapshot.get("metadata_gaps") or []
    gap_rows = []
    for gap in gaps[:25]:
        gap_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(gap.get('source', 'external')))}</code></td>"
            f"<td><code>{html.escape(str(gap.get('plugin_id')))}</code></td>"
            f"<td><code>{html.escape(str(gap.get('tool_name')))}</code></td>"
            f"<td>{html.escape(str(gap.get('risk_level')))}</td>"
            f"<td>{html.escape(', '.join(gap.get('missing') or []))}</td>"
            "</tr>"
        )

    closed_loop = payload.get("closed_loop") or {}
    closed_loop_rows = []
    for phase in closed_loop.get("phases") or []:
        selected = (
            ", ".join(f"{cap} → {tool}" for cap, tool in (phase.get("selected") or {}).items())
            or "-"
        )
        action = phase.get("action_result") or {}
        action_text = action.get("action") or (
            "ok" if action.get("ok") else action.get("error", "-")
        )
        closed_loop_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(phase.get('phase')))}</code></td>"
            f"<td>{phase.get('registry_version')}</td>"
            f"<td>{phase.get('candidate_count')}</td>"
            f"<td>{html.escape(selected)}</td>"
            f"<td>{phase.get('executable')}</td>"
            f"<td>{html.escape(str(action_text or '-'))}</td>"
            "</tr>"
        )
    closed_loop_section = ""
    if closed_loop:
        closed_loop_section = f"""
<h2>Closed-loop Mutation Timeline</h2>
<div class=\"card-grid\">
  <div class=\"card\"><div class=\"label\">Closed loop</div><div class=\"metric {"ok" if closed_loop.get("ok") else "bad"}\">{"PASS" if closed_loop.get("ok") else "FAIL"}</div></div>
  <div class=\"card\"><div class=\"label\">Mode</div><div class=\"metric\">{html.escape(str(closed_loop.get("mode", "fixture")))}</div></div>
  <div class=\"card\"><div class=\"label\">Loop phases</div><div class=\"metric\">{len(closed_loop.get("phases") or [])}</div></div>
  <div class=\"card\"><div class=\"label\">Approval requests</div><div class=\"metric\">{len(closed_loop.get("approval_requests") or [])}</div></div>
  <div class=\"card\"><div class=\"label\">Source cleanup</div><div class=\"metric {"bad" if (closed_loop.get("cleanup") or {}).get("source_exists_after_remove") else "ok"}\">{"leftover" if (closed_loop.get("cleanup") or {}).get("source_exists_after_remove") else "clean"}</div></div>
</div>
<table><thead><tr><th>Phase</th><th>Registry version</th><th>Candidates</th><th>Selected</th><th>Executable</th><th>Action</th></tr></thead><tbody>{"".join(closed_loop_rows)}</tbody></table>
"""

    autonomous = payload.get("autonomous_long_run") or {}
    policy_rows = []
    for decision in autonomous.get("policy_decisions") or []:
        policy_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(decision.get('action')))}</code></td>"
            f"<td>{html.escape(str(decision.get('reason') or ''))}</td>"
            f"<td>{decision.get('requires_approval')}</td>"
            "</tr>"
        )
    governance_rows = []
    for item in autonomous.get("governance_results") or []:
        governance_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('action')))}</code></td>"
            f"<td><code>{html.escape(str(item.get('plugin_id')))}</code></td>"
            f"<td>{html.escape(str(item.get('trust_level')))}</td>"
            f"<td>{item.get('failure_streak')}</td>"
            "</tr>"
        )
    autonomous_section = ""
    if autonomous:
        proposal = autonomous.get("proposal") or {}
        autonomous_section = f"""
<h2>Autonomous Long-run Governance</h2>
<div class=\"card-grid\">
  <div class=\"card\"><div class=\"label\">Autonomous run</div><div class=\"metric {"ok" if autonomous.get("ok") else "bad"}\">{"PASS" if autonomous.get("ok") else "FAIL"}</div></div>
  <div class=\"card\"><div class=\"label\">Observations</div><div class=\"metric\">{len(autonomous.get("observation_records") or [])}</div></div>
  <div class=\"card\"><div class=\"label\">Proposal status</div><div class=\"metric\">{html.escape(str(proposal.get("status") or ""))}</div></div>
  <div class=\"card\"><div class=\"label\">Governance events</div><div class=\"metric\">{len(autonomous.get("governance_results") or [])}</div></div>
</div>
<h3>Policy decisions</h3>
<table><thead><tr><th>Action</th><th>Reason</th><th>Requires approval</th></tr></thead><tbody>{"".join(policy_rows)}</tbody></table>
<h3>Governance timeline</h3>
<table><thead><tr><th>Action</th><th>Plugin</th><th>Trust</th><th>Failure streak</th></tr></thead><tbody>{"".join(governance_rows)}</tbody></table>
"""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Adaptive Plugin Experiment</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; background: #f7f7f4; color: #1f2933; }}
h1, h2 {{ letter-spacing: -0.02em; }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0; }}
.card {{ background: white; border: 1px solid #d8d8d2; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
.metric {{ font-size: 28px; font-weight: 700; }}
.label {{ color: #667085; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 16px 0 32px; }}
th, td {{ border: 1px solid #d8d8d2; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #ecebe4; }}
.ok {{ color: #166534; font-weight: 700; }}
.bad {{ color: #991b1b; font-weight: 700; }}
code {{ background: #f0f0ea; padding: 1px 4px; border-radius: 4px; }}
.bar {{ height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
.bar > span {{ display: block; height: 100%; background: #2563eb; }}
</style>
</head>
<body>
<h1>Adaptive Plugin Scenario Matrix</h1>
<div class=\"card-grid\">
  <div class=\"card\"><div class=\"label\">Overall</div><div class=\"metric {"ok" if payload.get("ok") else "bad"}\">{"PASS" if payload.get("ok") else "FAIL"}</div></div>
  <div class=\"card\"><div class=\"label\">Scenarios passed</div><div class=\"metric\">{payload.get("passed")}/{payload.get("scenario_count")}</div></div>
  <div class=\"card\"><div class=\"label\">Registry candidates</div><div class=\"metric\">{snapshot.get("candidate_count", 0)}</div></div>
  <div class=\"card\"><div class=\"label\">Metadata gaps</div><div class=\"metric\">{snapshot.get("metadata_gap_count", 0)}</div></div>
  <div class=\"card\"><div class=\"label\">Built-in gaps</div><div class=\"metric\">{builtin_stats.get("metadata_gap_count", 0)}</div></div>
  <div class=\"card\"><div class=\"label\">Profile gaps</div><div class=\"metric\">{profile_stats.get("metadata_gap_count", 0)}</div></div>
</div>
<h2>Scenario Matrix</h2>
<table><thead><tr><th>Scenario</th><th>OK</th><th>Selected</th><th>Plan</th><th>Diagnostics</th></tr></thead><tbody>{"".join(scenario_rows)}</tbody></table>
<h2>Real Registry Metadata Coverage</h2>
<div class=\"card-grid\">
  <div class=\"card\"><div class=\"label\">Provides coverage</div><div class=\"metric\">{coverage.get("provides_ratio", 0.0)}</div><div class=\"bar\"><span style=\"width:{float(coverage.get("provides_ratio", 0.0)) * 100}%\"></span></div></div>
  <div class=\"card\"><div class=\"label\">Platform requirement coverage</div><div class=\"metric\">{coverage.get("platform_requirement_ratio", 0.0)}</div><div class=\"bar\"><span style=\"width:{float(coverage.get("platform_requirement_ratio", 0.0)) * 100}%\"></span></div></div>
  <div class=\"card\"><div class=\"label\">Tool-name conflicts</div><div class=\"metric\">{snapshot.get("conflict_count", 0)}</div></div>
</div>
<h3>Source Coverage</h3>
<table><thead><tr><th>Source</th><th>Plugins</th><th>Candidates</th><th>Provides</th><th>Platform reqs</th><th>Gaps</th></tr></thead><tbody>{"".join(source_rows)}</tbody></table>
<h3>Metadata Gaps</h3>
<table><thead><tr><th>Source</th><th>Plugin</th><th>Tool</th><th>Risk</th><th>Missing metadata</th></tr></thead><tbody>{"".join(gap_rows)}</tbody></table>
{closed_loop_section}{autonomous_section}</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the adaptive plugin scenario matrix.")
    parser.add_argument(
        "--json-only", action="store_true", help="Write JSON only; skip Markdown report."
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="Run isolated install/execute/disable/remove registry mutation loop.",
    )
    parser.add_argument(
        "--autonomous-long-run",
        action="store_true",
        help="Run durable observation/proposal/policy/governor long-run scenario.",
    )
    parser.add_argument(
        "--live-generation",
        action="store_true",
        dest="live_generation",
        help="Use live LLM plugin generation in the closed-loop run (default).",
    )
    parser.add_argument(
        "--no-live-generation",
        action="store_false",
        dest="live_generation",
        help="Use deterministic fixture plugin code instead of live LLM generation.",
    )
    parser.set_defaults(live_generation=True)
    args = parser.parse_args(argv)
    payload = run(
        closed_loop=args.closed_loop,
        live_generation=args.live_generation,
        autonomous_long_run=args.autonomous_long_run,
    )
    json_path, markdown_path, html_path, gaps_path = _report_paths()
    payload["reports"] = {
        "json": str(json_path),
        "markdown": str(markdown_path) if not args.json_only else "",
        "html": str(html_path) if not args.json_only else "",
        "metadata_gaps": str(gaps_path) if not args.json_only else "",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.json_only:
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
        html_path.write_text(_render_html(payload), encoding="utf-8")
        gaps_path.write_text(_render_metadata_gap_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "json": str(json_path),
                "markdown": payload["reports"]["markdown"],
                "html": payload["reports"]["html"],
                "metadata_gaps": payload["reports"]["metadata_gaps"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
