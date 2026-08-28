"""Protocol and data definitions for Plugin Compatibility Assessment Engine.

Defines the core domain types used across all assessment stages.
All types are frozen dataclasses to guarantee immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Verdict(Enum):
    """Final compatibility classification."""

    COMPATIBLE = "compatible"  # Direct install; no modification needed
    ADAPTABLE = "adaptable"  # Needs a thin adapter/shim (auto-generatable)
    PARTIAL = "partial"  # Subset of features usable; limitations documented
    INCOMPATIBLE = "incompatible"  # Targets a system layer LeapFlow doesn't expose


class PluggabilityStatus(Enum):
    """Whether a system layer is exposed as a plugin surface."""

    PLUGGABLE = "pluggable"
    ADAPTABLE = "adaptable"
    PARTIAL = "partial"
    NOT_PLUGGABLE = "not_pluggable"


class DependencyFeasibility(Enum):
    """Whether a required dependency can be satisfied."""

    SATISFIABLE = "satisfiable"  # LeapFlow provides this service
    SHIMMABLE = "shimmable"  # Can be faked/shimmed with acceptable loss
    BLOCKING = "blocking"  # Cannot be provided; blocks installation


class SecurityRisk(Enum):
    """Risk classification for plugin permissions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PluginSourceKind(Enum):
    """Foreign plugin source layouts understood by the compatibility layer."""

    MANIFEST_ONLY = "manifest_only"
    DSH_PACKAGE = "dsh_package"
    CORDIS_DYNAMIC_EXPORT = "cordis_dynamic_export"
    LEAPFLOW_NATIVE = "leapflow_native"


class ComponentKind(Enum):
    """Independently assessed pieces of a foreign plugin bundle."""

    HOST = "host"
    CLIENT = "client"


class ComponentStatus(Enum):
    """Whether one component can execute in the current LeapFlow runtime."""

    CANDIDATE = "candidate"
    RUNTIME_READY = "runtime_ready"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ComponentCompatibility:
    """Compatibility verdict for one source component."""

    name: str
    kind: ComponentKind
    status: ComponentStatus
    reason: str
    entry_point: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    """Concrete bridge plan attached to a compatibility report.

    Static assessment produces candidate components. Runtime discovery replaces
    the host component with RUNTIME_READY only after a restricted Node worker has
    loaded the source and returned valid public tool descriptors.
    """

    source_kind: PluginSourceKind
    source_root: str = ""
    entry_point: str = ""
    runtime: str = ""
    bundle_sha256: str = ""
    source_files: tuple[str, ...] = ()
    requires_discovery: bool = False
    dependencies: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    components: tuple[ComponentCompatibility, ...] = ()
    blockers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def runtime_ready(self) -> bool:
        return any(
            component.kind == ComponentKind.HOST
            and component.status == ComponentStatus.RUNTIME_READY
            for component in self.components
        )

    @property
    def installable_candidate(self) -> bool:
        return not self.blockers and any(
            component.kind == ComponentKind.HOST
            and component.status in {ComponentStatus.CANDIDATE, ComponentStatus.RUNTIME_READY}
            for component in self.components
        )


@dataclass(frozen=True)
class PluginManifestInput:
    """Unified input format for assessment — normalizes DSH package.json
    and LeapFlow PluginManifest into a common structure."""

    name: str
    version: str
    category: str
    declared_interfaces: list[str] = field(default_factory=list)
    declared_dependencies: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    execution_model: str = "async"
    permissions: list[str] = field(default_factory=list)
    source_language: str = "python"
    raw_manifest: dict[str, Any] = field(default_factory=dict)
    source_format: str = "leapflow"  # "leapflow" | "dsh"


@dataclass(frozen=True)
class AdapterSpec:
    """Specification for an auto-generated adapter when verdict is ADAPTABLE."""

    source_interface: str
    target_protocol: str
    bridge_type: str  # "json_rpc_bridge" | "protocol_wrapper" | "shim_layer"
    shim_methods: list[str] = field(default_factory=list)
    estimated_complexity: str = "low"  # "low" | "medium" | "high"


@dataclass(frozen=True)
class StageResult:
    """Result produced by a single assessment stage."""

    stage_name: str
    passed: bool
    verdict: Optional[Verdict] = None
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityReport:
    """Complete assessment output — the single artifact produced by the pipeline."""

    manifest: PluginManifestInput
    stages: list[StageResult] = field(default_factory=list)
    final_verdict: Verdict = Verdict.INCOMPATIBLE
    target_protocol: Optional[str] = None
    rejection_reason: Optional[str] = None
    adaptation_notes: list[str] = field(default_factory=list)
    adapter_spec: Optional[AdapterSpec] = None
    execution_plan: Optional[ExecutionPlan] = None

    def is_installable(self) -> bool:
        """Whether this report proves a plugin is ready for installation.

        Manifest-only reports preserve the historical enum-based answer. Source
        bundles are stricter: an untrusted JavaScript bundle becomes installable
        only after restricted runtime discovery has produced a real host tool.
        Static ADAPTABLE/PARTIAL is a candidate, never proof of executability.
        """
        if self.execution_plan is not None:
            return (
                self.final_verdict in (Verdict.ADAPTABLE, Verdict.PARTIAL)
                and not self.execution_plan.blockers
                and self.execution_plan.runtime_ready
            )
        if self.manifest.source_format == "dsh":
            # A package.json-shaped dict can be classified, but no source has
            # been bounded, hashed or executed. It is not an install artifact.
            return False
        return self.final_verdict in (
            Verdict.COMPATIBLE,
            Verdict.ADAPTABLE,
            Verdict.PARTIAL,
        )
