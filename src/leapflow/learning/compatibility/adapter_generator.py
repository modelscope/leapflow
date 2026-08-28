"""Deterministic wrapper generation for runtime-discovered DSH plugins.

A manifest cannot prove a foreign tool exists or is executable. Wrapper source
is therefore emitted only from a descriptor produced by restricted Node runtime
discovery. LLM-authored runtime adapters are intentionally disabled: an LLM is
not a security authority for schemas, permissions, or process capabilities.
"""
from __future__ import annotations

from typing import Any

from leapflow.learning.compatibility.protocol import AdapterSpec, PluginManifestInput


# ═══════════════════════════════════════════════════════════════════════
# Template mode (no LLM)
# ═══════════════════════════════════════════════════════════════════════


def generate_adapter_template(
    spec: AdapterSpec, manifest: PluginManifestInput
) -> str:
    """Render a validated wrapper from a runtime-discovered DSH descriptor.

    Static manifests are insufficient: the previous template invented tools from
    declared interface names, pointed Python ``SandboxHost`` at JavaScript, and
    omitted the module-level ``plugin`` required by the actual installer. It
    compiled, but could never execute. Runtime discovery is now the authority;
    callers pass its descriptor through ``x_leapflow_runtime_descriptor``.
    """
    del spec  # bridge selection is already encoded by the runtime descriptor
    raw = manifest.raw_manifest or {}
    descriptor = raw.get("x_leapflow_runtime_descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            "DSH adapter generation requires restricted runtime discovery; "
            "use plugin_install(source_path=...)"
        )
    from leapflow.plugins.dsh.descriptor import (
        DshPluginDescriptor,
        render_python_wrapper,
    )

    return render_python_wrapper(DshPluginDescriptor.from_dict(descriptor))


def generate_adapter_with_llm(
    spec: AdapterSpec, manifest: PluginManifestInput, llm_provider: Any
) -> str:
    """Return the deterministic runtime-discovered wrapper.

    P3 may add LLM suggestions around the deterministic descriptor, but the LLM
    must never rewrite the executable security boundary.
    """
    del llm_provider
    return generate_adapter_template(spec, manifest)
