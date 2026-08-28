"""Plugin Compatibility Assessment Engine.

Evaluates foreign plugins (primarily from deepseek-harness ecosystem)
for LeapFlow compatibility before installation is attempted.
"""

from leapflow.learning.compatibility.pipeline import assess_plugin
from leapflow.learning.compatibility.protocol import (
    CompatibilityReport,
    ComponentCompatibility,
    ComponentKind,
    ComponentStatus,
    ExecutionPlan,
    PluginManifestInput,
    PluginSourceKind,
    Verdict,
)
from leapflow.learning.compatibility.source_inspector import (
    SourceInspection,
    SourceInspectionError,
    inspect_plugin_source,
)

__all__ = [
    "assess_plugin",
    "CompatibilityReport",
    "ComponentCompatibility",
    "ComponentKind",
    "ComponentStatus",
    "ExecutionPlan",
    "PluginManifestInput",
    "PluginSourceKind",
    "SourceInspection",
    "SourceInspectionError",
    "Verdict",
    "inspect_plugin_source",
]
