"""Prepare and validate DSH plugin installations before registry mutation."""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from leapflow.learning.compatibility import assess_plugin, inspect_plugin_source
from leapflow.learning.compatibility.protocol import (
    CompatibilityReport,
    ComponentCompatibility,
    ComponentKind,
    ComponentStatus,
    Verdict,
)
from leapflow.plugins.dsh.bundle import stage_runtime_bundle
from leapflow.plugins.dsh.capabilities import DshCapabilityBroker
from leapflow.plugins.dsh.descriptor import (
    DshPluginDescriptor,
    DshToolDescriptor,
    normalize_plugin_id,
    render_python_wrapper,
)
from leapflow.plugins.dsh.node_host import DshNodeHost


class DshInstallError(RuntimeError):
    """A DSH source failed assessment or restricted runtime discovery."""


@dataclass
class PreparedDshInstallation:
    plugin_id: str
    staging_root: Path
    final_root: Path
    wrapper_path: Path
    descriptor: DshPluginDescriptor
    wrapper_source: str
    compatibility: CompatibilityReport

    def cleanup(self) -> None:
        shutil.rmtree(self.staging_root, ignore_errors=True)


async def prepare_dsh_installation(
    source_path: str | Path,
    *,
    plugin_id: str,
    plugins_dir: str | Path,
    dsh_plugins_dir: str | Path,
    broker: DshCapabilityBroker | None = None,
    settings: Any = None,
) -> PreparedDshInstallation:
    """Inspect, stage and discover a DSH plugin without making it live."""
    inspection = inspect_plugin_source(source_path)
    report = assess_plugin(source_path)
    plan = report.execution_plan
    if plan is None:
        raise DshInstallError("DSH assessment did not produce an execution plan")
    if report.final_verdict == Verdict.INCOMPATIBLE or plan.blockers:
        reason = report.rejection_reason or "; ".join(plan.blockers)
        raise DshInstallError(reason or "DSH plugin is incompatible")

    normalized_id = normalize_plugin_id(plugin_id or inspection.manifest.name)
    plugins_root = Path(plugins_dir).expanduser().resolve()
    dsh_root = Path(dsh_plugins_dir).expanduser().resolve()
    final_root = dsh_root / normalized_id
    wrapper_path = plugins_root / f"{normalized_id}.py"
    if final_root.exists() or wrapper_path.exists():
        raise DshInstallError(f"DSH plugin '{normalized_id}' is already installed")

    staging, runtime_entry = stage_runtime_bundle(inspection, dsh_root, normalized_id)
    try:
        host = _node_host(
            staging,
            source_kind=plan.source_kind.value,
            entry_point=runtime_entry,
            broker=broker,
            settings=settings,
        )
        try:
            response = await host.discover()
            if not response.ok:
                diagnostic = host.stderr_tail.strip()
                suffix = f" Worker stderr: {diagnostic}" if diagnostic else ""
                raise DshInstallError(
                    f"Restricted DSH discovery failed: {response.error}.{suffix}"
                )
            discovery = response.result
            if not isinstance(discovery, dict):
                raise DshInstallError("Restricted DSH discovery returned a non-object result")
            raw_tools = discovery.get("tools")
            if not isinstance(raw_tools, list) or not raw_tools:
                raise DshInstallError(
                    "DSH source exposed no public registerTool tools; handler channels and client UI "
                    "are not published as LeapFlow tools in P0"
                )
            tools = tuple(DshToolDescriptor.from_dict(dict(item)) for item in raw_tools)
        finally:
            await host.stop()

        components = []
        for component in plan.components:
            if component.kind == ComponentKind.HOST:
                components.append(
                    ComponentCompatibility(
                        name=component.name,
                        kind=component.kind,
                        status=ComponentStatus.RUNTIME_READY,
                        reason=f"Restricted Node discovery exposed {len(tools)} public tool(s)",
                        entry_point=runtime_entry,
                        metadata={
                            **component.metadata,
                            "tools": [tool.name for tool in tools],
                            "handler_channels": list(discovery.get("handler_channels") or ()),
                            "node_version": str(discovery.get("node_version") or ""),
                        },
                    )
                )
            else:
                components.append(component)
        ready_plan = replace(
            plan,
            source_root=str(final_root),
            entry_point=runtime_entry,
            requires_discovery=False,
            components=tuple(components),
        )
        final_verdict = (
            Verdict.PARTIAL
            if any(item.status == ComponentStatus.UNSUPPORTED for item in components)
            else Verdict.ADAPTABLE
        )
        ready_report = replace(
            report,
            final_verdict=final_verdict,
            execution_plan=ready_plan,
        )
        descriptor = DshPluginDescriptor(
            plugin_id=normalized_id,
            name=inspection.manifest.name,
            source_kind=plan.source_kind.value,
            bundle_root=str(final_root),
            entry_point=runtime_entry,
            bundle_sha256=plan.bundle_sha256,
            runtime_sha256=hashlib.sha256((staging / runtime_entry).read_bytes()).hexdigest(),
            source_files=plan.source_files,
            tools=tools,
            permissions=plan.permissions,
            limitations=plan.limitations,
            client_components=tuple(
                {
                    "name": item.name,
                    "status": item.status.value,
                    "reason": item.reason,
                    **item.metadata,
                }
                for item in components
                if item.kind == ComponentKind.CLIENT
            ),
        )
        return PreparedDshInstallation(
            plugin_id=normalized_id,
            staging_root=staging,
            final_root=final_root,
            wrapper_path=wrapper_path,
            descriptor=descriptor,
            wrapper_source=render_python_wrapper(descriptor),
            compatibility=ready_report,
        )
    except Exception:
        # prepare_dsh_installation has no owner to call Prepared.cleanup() until
        # it returns. Any discovery/schema/runtime failure must therefore remove
        # the private staging copy here.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _node_host(
    source_root: Path,
    *,
    source_kind: str,
    entry_point: str,
    broker: DshCapabilityBroker | None,
    settings: Any,
) -> DshNodeHost:
    return DshNodeHost(
        source_root,
        source_kind=source_kind,
        entry_point=entry_point,
        broker=broker,
        invoke_timeout_s=float(
            getattr(settings, "plugins_dsh_invoke_timeout_s", 30.0)
        ),
        discovery_timeout_s=float(
            getattr(settings, "plugins_dsh_discovery_timeout_s", 10.0)
        ),
        max_line_bytes=int(
            getattr(settings, "plugins_dsh_max_message_bytes", 1_000_000)
        ),
        max_stderr_bytes=int(
            getattr(settings, "plugins_dsh_max_stderr_bytes", 64_000)
        ),
        max_memory_mb=int(
            getattr(settings, "plugins_dsh_max_memory_mb", 128)
        ),
    )
