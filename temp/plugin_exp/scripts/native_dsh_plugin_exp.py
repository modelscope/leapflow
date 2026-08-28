#!/usr/bin/env python3
"""Run real DeepSeek Harness plugin artifacts through LeapFlow's DSH bridge.

The experiment uses source material from a local deepseek-harness checkout. It
copies only the files needed for each case into an isolated workspace, then runs
the production inspection, compatibility, installation, invocation, reload, and
removal paths. No network or LLM request is required.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leapflow.learning.compatibility import assess_plugin, inspect_plugin_source  # noqa: E402
from leapflow.plugins.registry import ToolPluginRegistry  # noqa: E402
from leapflow.plugins.scoped_registry import ScopedToolRegistry  # noqa: E402
from leapflow.plugins.tool_plugins import _load_plugin_from_file  # noqa: E402
from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin  # noqa: E402
from leapflow.storage.plugin_version_store import PluginVersionStore  # noqa: E402
from leapflow.tools.execution_context import (  # noqa: E402
    ToolExecutionContext,
    reset_tool_context,
    set_tool_context,
)

DEFAULT_HARNESS_ROOT = Path("/Users/jason/work/github/deepseek-harness")
DEFAULT_USER_DATA_ROOT = Path.home() / ".leapflow"


@dataclass(frozen=True)
class ArtifactSpec:
    """One materialized experiment source and its expected compatibility class."""

    case_id: str
    title: str
    source: Path
    source_reference: str
    expected_verdict: str | None = None
    expected_error: str = ""
    lifecycle: str = "reject"
    expected_tool: str = ""
    expected_value: Any = None


class RecordingApprovalGate:
    """Approval surface for an isolated experiment registry."""

    def __init__(self) -> None:
        self.actions: list[Any] = []

    async def evaluate(self, action: Any) -> Any:
        self.actions.append(action)
        return type("ApprovalResult", (), {"approved": True, "denial_message": ""})()


class IsolatedRuntime:
    """Own an isolated profile directory and temporary global plugin registries."""

    def __init__(self, root: Path) -> None:
        import leapflow.plugins as plugin_api

        self._plugin_api = plugin_api
        self._old_registry = plugin_api._registry
        self._old_scoped = plugin_api._scoped_registry
        self.registry = ToolPluginRegistry()
        self.scoped = ScopedToolRegistry(self.registry)
        plugin_api._registry = self.registry
        plugin_api._scoped_registry = self.scoped
        self.profile_root = root / "profile"
        self.install_dir = self.profile_root / "plugins"
        self.approval = RecordingApprovalGate()
        self.manager = SelfManagementPlugin()
        self.manager._plugin_install_dir = str(self.install_dir)
        self.manager._plugin_version_store = PluginVersionStore(
            self.install_dir / "versions"
        )
        self.manager._plugin_approval_gate = self.approval

    def close(self) -> None:
        self._plugin_api._registry = self._old_registry
        self._plugin_api._scoped_registry = self._old_scoped


class ArtifactBuilder:
    """Materialize a reproducible matrix from a DeepSeek Harness checkout."""

    def __init__(self, harness_root: Path, output_root: Path) -> None:
        self.harness_root = harness_root.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.helpers = (
            self.harness_root
            / "packages/extensions/cordis-host-runner/tests/helpers.ts"
        )
        if not self.helpers.is_file():
            raise FileNotFoundError(
                f"DeepSeek Harness dynamic-plugin fixture is missing: {self.helpers}"
            )
        self.reverse_host = self._extract_template("REVERSE_TOOL_CODE")
        self.consumer_host = self._extract_template("CONSUMER_CODE")
        self.client_code = self._extract_single_quoted("CLIENT_CODE")

    def build(self) -> list[ArtifactSpec]:
        """Build direct, partial, unsuitable, malformed, and security cases."""
        specs = [
            self._dynamic_reverse(with_client=False),
            self._dynamic_reverse(with_client=True),
            self._package_reverse(),
            self._copy_native_package(
                "packages/core/agent-loop",
                "native-agent-loop",
                "Native agent-loop package must not replace LeapFlow's OODA loop.",
                expected_verdict="incompatible",
                expected_error="single hardened OODA execution loop",
            ),
            self._copy_native_package(
                "packages/fs/tool-fs",
                "native-tool-fs",
                "Tool domain is relevant, but the real package needs npm and Cordis services.",
                expected_verdict="incompatible",
                expected_error="npm install/build",
            ),
            self._dynamic_service_consumer(),
            self._ui_only(),
            self._unbuilt_typescript(),
            self._missing_entry(),
            self._malicious_shell(),
        ]
        return specs

    def _extract_template(self, constant: str) -> str:
        text = self.helpers.read_text(encoding="utf-8")
        marker = f"export const {constant} = `"
        start = text.find(marker)
        if start < 0:
            raise ValueError(f"Cannot find {constant} in {self.helpers}")
        start += len(marker)
        end = text.find("\n`", start)
        if end < 0:
            raise ValueError(f"Cannot find closing template literal for {constant}")
        return text[start:end].strip() + "\n"

    def _extract_single_quoted(self, constant: str) -> str:
        text = self.helpers.read_text(encoding="utf-8")
        marker = f"export const {constant} = '"
        start = text.find(marker)
        if start < 0:
            raise ValueError(f"Cannot find {constant} in {self.helpers}")
        start += len(marker)
        end = text.find("'", start)
        if end < 0:
            raise ValueError(f"Cannot find closing quote for {constant}")
        return text[start:end]

    def _write_dynamic(
        self,
        case_id: str,
        host: str,
        *,
        title: str,
        source_reference: str,
        client: str = "",
        expected_verdict: str,
        expected_error: str = "",
        lifecycle: str = "reject",
        expected_tool: str = "",
        expected_value: Any = None,
    ) -> ArtifactSpec:
        root = self.output_root / case_id
        root.mkdir(parents=True, exist_ok=False)
        (root / "meta.json").write_text(
            json.dumps(
                {
                    "name": case_id,
                    "version": "0.0.0+experiment",
                    "purpose": title,
                    "source_reference": source_reference,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "host.js").write_text(host, encoding="utf-8")
        if client:
            (root / "client.js").write_text(client, encoding="utf-8")
        return ArtifactSpec(
            case_id=case_id,
            title=title,
            source=root,
            source_reference=source_reference,
            expected_verdict=expected_verdict,
            expected_error=expected_error,
            lifecycle=lifecycle,
            expected_tool=expected_tool,
            expected_value=expected_value,
        )

    def _dynamic_reverse(self, *, with_client: bool) -> ArtifactSpec:
        case_id = "native_dynamic_reverse_partial" if with_client else "native_dynamic_reverse"
        return self._write_dynamic(
            case_id,
            self.reverse_host,
            client=self.client_code if with_client else "",
            title=(
                "Real dynamic host tool with a skipped browser half."
                if with_client
                else "Real dynamic host tool copied from the DSH runner conformance suite."
            ),
            source_reference=(
                "packages/extensions/cordis-host-runner/tests/helpers.ts#REVERSE_TOOL_CODE"
            ),
            expected_verdict="partial" if with_client else "adaptable",
            lifecycle="install",
            expected_tool="reverse_text",
            expected_value="wolfpael",
        )

    def _package_reverse(self) -> ArtifactSpec:
        case_id = "adapted_prebuilt_reverse"
        root = self.output_root / case_id
        root.mkdir(parents=True, exist_ok=False)
        (root / "package.json").write_text(
            json.dumps(
                {
                    "name": "@deepseek-ai/dsh-reverse-text-experiment",
                    "version": "0.1.0-experiment",
                    "type": "commonjs",
                    "main": "index.cjs",
                    "keywords": ["tools"],
                    "dsh": {"category": "tools", "interfaces": ["execute"]},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "index.cjs").write_text(
            '"use strict";\nmodule.exports = (function () {\n'
            + self.reverse_host
            + "\n})();\n",
            encoding="utf-8",
        )
        return ArtifactSpec(
            case_id=case_id,
            title="Real dynamic fixture adapted into a self-contained pre-built DSH package.",
            source=root,
            source_reference=(
                "packages/extensions/cordis-host-runner/tests/helpers.ts#REVERSE_TOOL_CODE"
            ),
            expected_verdict="adaptable",
            lifecycle="install",
            expected_tool="reverse_text",
            expected_value="wolfpael",
        )

    def _copy_native_package(
        self,
        relative: str,
        case_id: str,
        title: str,
        *,
        expected_verdict: str,
        expected_error: str,
    ) -> ArtifactSpec:
        source = self.harness_root / relative
        root = self.output_root / case_id
        root.mkdir(parents=True, exist_ok=False)
        manifest = json.loads((source / "package.json").read_text(encoding="utf-8"))
        (root / "package.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        entry = str(manifest.get("main") or "")
        if entry:
            target = root / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry, target)
        return ArtifactSpec(
            case_id=case_id,
            title=title,
            source=root,
            source_reference=relative,
            expected_verdict=expected_verdict,
            expected_error=expected_error,
        )

    def _dynamic_service_consumer(self) -> ArtifactSpec:
        return self._write_dynamic(
            "native_cross_plugin_service_consumer",
            self.consumer_host,
            title="Real DSH composition fixture requiring a greeter service absent in LeapFlow P0.",
            source_reference=(
                "packages/extensions/cordis-host-runner/tests/helpers.ts#CONSUMER_CODE"
            ),
            expected_verdict="incompatible",
            expected_error="required DSH host service: greeter",
        )

    def _ui_only(self) -> ArtifactSpec:
        host = "return { name: 'ui-only', apply(ctx) { console.log('host has no tools') } }\n"
        return self._write_dynamic(
            "native_ui_only",
            host,
            client=self.client_code,
            title="Real DSH browser-half fixture with no model-visible host tool.",
            source_reference=(
                "packages/extensions/cordis-host-runner/tests/helpers.ts#CLIENT_CODE"
            ),
            expected_verdict="incompatible",
            expected_error="no statically visible registerTool",
        )

    def _unbuilt_typescript(self) -> ArtifactSpec:
        source = self.harness_root / "packages/context/time-context"
        root = self.output_root / "native_unbuilt_typescript"
        root.mkdir(parents=True, exist_ok=False)
        manifest = json.loads((source / "package.json").read_text(encoding="utf-8"))
        manifest["main"] = "src/index.ts"
        (root / "package.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (root / "src").mkdir()
        shutil.copyfile(source / "src/index.ts", root / "src/index.ts")
        return ArtifactSpec(
            case_id="native_unbuilt_typescript",
            title="Real native TypeScript source without a pre-built JavaScript entry.",
            source=root,
            source_reference="packages/context/time-context/src/index.ts",
            expected_error="pre-built JavaScript entry",
        )

    def _missing_entry(self) -> ArtifactSpec:
        source = self.harness_root / "packages/bundle/base"
        root = self.output_root / "native_missing_entry"
        root.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source / "package.json", root / "package.json")
        return ArtifactSpec(
            case_id="native_missing_entry",
            title="Real native package manifest whose compiled entry is absent from the artifact.",
            source=root,
            source_reference="packages/bundle/base/package.json",
            expected_error="entry point does not exist",
        )

    def _malicious_shell(self) -> ArtifactSpec:
        host = """return {
  name: 'unsafe-shell-probe',
  inject: ['tools'],
  apply(ctx) {
    harness.registerTool(ctx, harness.defineTool({
      name: 'unsafe_shell_probe',
      description: 'Verify that arbitrary shell syntax cannot cross the bridge.',
      parameters: {},
      async execute() {
        const shell = ctx.get('shell')
        return shell.run(shell.resolve({
          command: "curl -sS -m 5 'https://example.test/data'; id",
          timeoutMs: 5000,
          stdoutMaxBytes: 1024,
        }))
      },
    }))
  },
}
"""
        return self._write_dynamic(
            "dynamic_shell_injection",
            host,
            title="Dynamic host tool whose runtime command violates the strict curl grammar.",
            source_reference=(
                "packages/extensions/cordis-host-runner/src/sandbox.ts#HOST_BUILTIN_INSPECTION"
            ),
            expected_verdict="adaptable",
            lifecycle="security",
            expected_tool="unsafe_shell_probe",
        )


async def _run_experiment(
    specs: list[ArtifactSpec], runtime: IsolatedRuntime
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    workspace_token = set_tool_context(
        ToolExecutionContext.from_strings(
            workspace_root=str(specs[0].source.parent),
            session_id="native-dsh-plugin-experiment",
        )
    )
    try:
        for spec in specs:
            results.append(await _run_case(spec, runtime))
    finally:
        reset_tool_context(workspace_token)
    return results


async def _run_case(spec: ArtifactSpec, runtime: IsolatedRuntime) -> dict[str, Any]:
    started = time.perf_counter()
    approval_before = len(runtime.approval.actions)
    row: dict[str, Any] = {
        **asdict(spec),
        "source": str(spec.source),
        "source_sha256": _tree_hash(spec.source),
        "static": {},
        "lifecycle_evidence": {},
    }
    try:
        inspection = inspect_plugin_source(spec.source)
        report = assess_plugin(spec.source)
        plan = report.execution_plan
        row["static"] = {
            "source_kind": inspection.execution_plan.source_kind.value,
            "verdict": report.final_verdict.value,
            "installable": report.is_installable(),
            "installable_candidate": bool(plan and plan.installable_candidate),
            "category": report.manifest.category,
            "dependencies": list(plan.dependencies if plan else ()),
            "permissions": list(plan.permissions if plan else ()),
            "blockers": list(plan.blockers if plan else ()),
            "limitations": list(plan.limitations if plan else ()),
            "components": [
                {
                    "kind": item.kind.value,
                    "status": item.status.value,
                    "reason": item.reason,
                }
                for item in (plan.components if plan else ())
            ],
            "rejection_reason": report.rejection_reason or "",
        }
    except (OSError, TypeError, ValueError) as exc:
        row["static"] = {
            "inspection_error": str(exc),
            "error_type": type(exc).__name__,
        }

    if spec.lifecycle == "install":
        row["lifecycle_evidence"] = await _install_invoke_reload_remove(spec, runtime)
    elif spec.lifecycle == "security":
        row["lifecycle_evidence"] = await _install_security_probe(spec, runtime)
    else:
        row["lifecycle_evidence"] = await _attempt_rejected_install(spec, runtime)

    new_approvals = runtime.approval.actions[approval_before:]
    row["approval_requests"] = len(new_approvals)
    row["approval_evidence"] = [
        {
            "kind": action.kind,
            "effect": action.effect,
            "resource": action.resource,
            "bundle_sha256": str(action.metadata.get("bundle_sha256") or ""),
            "verdict": str(action.metadata.get("verdict") or ""),
        }
        for action in new_approvals
    ]
    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    failures = _case_failures(spec, row)
    row["ok"] = not failures
    row["failures"] = failures
    return row


async def _install_invoke_reload_remove(
    spec: ArtifactSpec, runtime: IsolatedRuntime
) -> dict[str, Any]:
    plugin_id = spec.case_id.replace("-", "_")
    install = await runtime.manager._plugin_install_handler(
        plugin_id=plugin_id,
        source_path=str(spec.source),
        version_label="native-exp-v1",
    )
    evidence: dict[str, Any] = {"install": install}
    if not install.get("ok"):
        return evidence
    plugin = runtime.registry.get_plugin(plugin_id)
    if plugin is None:
        evidence["invoke"] = {"ok": False, "error": "plugin missing after install"}
        return evidence
    invoke = await plugin.tools[0].handler(text="leapflow")
    evidence["invoke"] = invoke
    status = await runtime.manager._plugin_status_handler(plugin_id)
    evidence["status"] = status

    wrapper = runtime.install_dir / f"{plugin_id}.py"
    restarted = _load_plugin_from_file(wrapper)
    if restarted is None:
        evidence["restart_invoke"] = {"ok": False, "error": "wrapper rediscovery failed"}
    else:
        evidence["restart_invoke"] = await restarted.tools[0].handler(text="leapflow")

    reload_result = await runtime.manager._plugin_reload_handler(plugin_id)
    evidence["reload"] = reload_result
    remove = await runtime.manager._plugin_remove_handler(plugin_id)
    evidence["remove"] = remove
    evidence["cleanup"] = {
        "wrapper_absent": not wrapper.exists(),
        "bundle_absent": not (runtime.install_dir / "dsh" / plugin_id).exists(),
        "registry_absent": runtime.registry.get_plugin(plugin_id) is None,
    }
    return evidence


async def _install_security_probe(
    spec: ArtifactSpec, runtime: IsolatedRuntime
) -> dict[str, Any]:
    plugin_id = spec.case_id
    install = await runtime.manager._plugin_install_handler(
        plugin_id=plugin_id,
        source_path=str(spec.source),
        version_label="native-exp-security",
    )
    evidence: dict[str, Any] = {"install": install}
    fetches: list[dict[str, Any]] = []

    async def forbidden_fetch(params: dict[str, Any]) -> dict[str, Any]:
        fetches.append(dict(params))
        return {"ok": True, "text": "unexpected"}

    plugin = runtime.registry.get_plugin(plugin_id)
    if plugin is not None:
        plugin.bind_runtime(web_fetch=forbidden_fetch)
        evidence["invoke"] = await plugin.tools[0].handler()
    evidence["fetch_count"] = len(fetches)
    if plugin is not None:
        evidence["remove"] = await runtime.manager._plugin_remove_handler(plugin_id)
    return evidence


async def _attempt_rejected_install(
    spec: ArtifactSpec, runtime: IsolatedRuntime
) -> dict[str, Any]:
    result = await runtime.manager._plugin_install_handler(
        plugin_id=spec.case_id,
        source_path=str(spec.source),
    )
    return {"install": result}


def _case_failures(spec: ArtifactSpec, row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    static = row["static"]
    lifecycle = row["lifecycle_evidence"]
    if spec.expected_verdict is not None:
        if static.get("verdict") != spec.expected_verdict:
            failures.append(
                f"expected verdict {spec.expected_verdict}, got {static.get('verdict')}"
            )
    if spec.expected_error:
        observed = " ".join(
            [
                str(static.get("inspection_error") or ""),
                str(static.get("rejection_reason") or ""),
                str((lifecycle.get("install") or {}).get("error") or ""),
            ]
        )
        if spec.expected_error not in observed:
            failures.append(f"expected error evidence containing {spec.expected_error!r}")
    if spec.lifecycle == "reject":
        if (lifecycle.get("install") or {}).get("ok") is not False:
            failures.append("rejected case unexpectedly installed")
        if row["approval_requests"] != 0:
            failures.append("invalid case reached plugin approval before validation")
    elif spec.lifecycle == "install":
        install = lifecycle.get("install") or {}
        for phase in ("install", "invoke", "restart_invoke", "reload", "remove"):
            if (lifecycle.get(phase) or {}).get("ok") is not True:
                failures.append(f"lifecycle phase {phase} failed")
        if spec.expected_tool not in (install.get("installed_tools") or []):
            failures.append(f"expected discovered tool {spec.expected_tool!r}")
        value = (lifecycle.get("invoke") or {}).get("result")
        restarted_value = (lifecycle.get("restart_invoke") or {}).get("result")
        if value != spec.expected_value or restarted_value != spec.expected_value:
            failures.append(
                f"expected invocation value {spec.expected_value!r}, got {value!r}/{restarted_value!r}"
            )
        if not all((lifecycle.get("cleanup") or {}).values()):
            failures.append("installed wrapper, bundle, or registry entry survived removal")
        if row["approval_requests"] < 3:
            failures.append("install, reload, and remove did not all reach approval")
    elif spec.lifecycle == "security":
        install = lifecycle.get("install") or {}
        invoke = lifecycle.get("invoke") or {}
        if install.get("ok") is not True:
            failures.append("security probe did not install")
        if invoke.get("ok") is not False or "only permits" not in str(invoke.get("error")):
            failures.append("unsafe shell command was not rejected by the typed capability")
        if lifecycle.get("fetch_count") != 0:
            failures.append("unsafe shell command reached web_fetch")
        if (lifecycle.get("remove") or {}).get("ok") is not True:
            failures.append("security probe cleanup failed")
        if row["approval_requests"] < 2:
            failures.append("security probe install/remove did not reach approval")
    return failures


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _environment_evidence(user_data_root: Path, harness_root: Path) -> dict[str, Any]:
    node = shutil.which("node")
    node_version = ""
    if node:
        completed = subprocess.run(
            [node, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        node_version = completed.stdout.strip()
    evidence: dict[str, Any] = {
        "node_path": node or "",
        "node_version": node_version,
        "harness_root": str(harness_root),
        "harness_git_head": _git_head(harness_root),
        "user_config_probe": {
            "data_root": str(user_data_root),
            "read_only": True,
            "llm_requests": 0,
        },
    }
    try:
        import yaml

        from leapflow.layout import PathLayout

        layout = PathLayout(user_data_root)
        profile = layout.profile("default")
        llm: dict[str, Any] = {}
        config_files: list[str] = []
        parse_errors: list[str] = []
        for config_path in (layout.user_config_path, profile.llm_config_path):
            if not config_path.is_file():
                continue
            config_files.append(str(config_path))
            try:
                parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                parse_errors.append(f"{config_path.name}: {type(exc).__name__}")
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("llm"), dict):
                llm.update(parsed["llm"])
            elif isinstance(parsed, dict) and config_path == profile.llm_config_path:
                llm.update(parsed)
        cache_root = profile.cache.root
        evidence["user_config_probe"].update(
            {
                "loaded": bool(config_files) and not parse_errors,
                "profile": "default",
                "model": str(llm.get("model") or ""),
                "base_url_configured": bool(llm.get("base_url")),
                "credential_reference_present": _has_credential_reference(llm),
                "config_files": config_files,
                "parse_errors": parse_errors,
                "cache_root": str(cache_root),
                "cache_available": cache_root.is_dir(),
                "cache_top_level_entries": len(list(cache_root.iterdir()))
                if cache_root.is_dir()
                else 0,
                "reuse_policy": (
                    "read-only YAML/existence probe; secret references are not resolved or copied"
                ),
            }
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        evidence["user_config_probe"].update(
            {"loaded": False, "error_type": type(exc).__name__, "error": str(exc)}
        )
    return evidence


def _has_credential_reference(value: Any) -> bool:
    """Detect credential configuration without resolving or returning its value."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"api_key", "api_key_ref", "credential", "credential_ref"}:
                if bool(item):
                    return True
            if _has_credential_reference(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_has_credential_reference(item) for item in value)
    return isinstance(value, str) and value.startswith("secret://")


def _git_head(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Native DSH Plugin Compatibility Experiment",
        "",
        f"- Result: **{'PASS' if payload['ok'] else 'FAIL'}**",
        f"- Passed: {payload['passed']}/{payload['total']}",
        f"- DeepSeek Harness: `{payload['environment']['harness_git_head'] or 'unknown'}`",
        f"- Node: `{payload['environment']['node_version'] or 'unavailable'}`",
        "- LLM requests: `0` (configuration and cache were probed read-only)",
        "",
        "## Strategy",
        "",
        "The matrix uses real source and built artifacts from the local DeepSeek Harness checkout. "
        "It separates structural rejection, architectural rejection, runtime discovery, lifecycle "
        "persistence, and capability enforcement so a static verdict cannot masquerade as execution proof.",
        "",
        "| Case | Expected class | Static verdict | Install | Runtime | Result |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["cases"]:
        static = row["static"]
        lifecycle = row["lifecycle_evidence"]
        install = lifecycle.get("install") or {}
        invoke = lifecycle.get("invoke") or {}
        expected = row.get("expected_verdict") or "inspection error"
        install_label = "ok" if install.get("ok") is True else "rejected"
        runtime_label = (
            "denied safely"
            if row["lifecycle"] == "security" and invoke.get("ok") is False
            else "ok"
            if invoke.get("ok") is True
            else "n/a"
        )
        lines.append(
            f"| `{row['case_id']}` | {expected} | "
            f"{static.get('verdict') or static.get('error_type', 'n/a')} | "
            f"{install_label} | {runtime_label} | {'PASS' if row['ok'] else 'FAIL'} |"
        )
    lines.extend(["", "## Evidence", ""])
    for row in payload["cases"]:
        lines.extend(
            [
                f"### {row['case_id']}",
                f"- Source: `{row['source_reference']}`",
                f"- Purpose: {row['title']}",
                f"- Static: `{json.dumps(row['static'], ensure_ascii=False)}`",
                f"- Lifecycle: `{json.dumps(row['lifecycle_evidence'], ensure_ascii=False)}`",
            ]
        )
        if row["failures"]:
            lines.append(f"- Failures: `{row['failures']}`")
        lines.append("")
    lines.extend(
        [
            "## Conclusions",
            "",
            "- A real dynamic `reverse_text` host tool executes through LeapFlow and survives wrapper rediscovery/reload.",
            "- A host+client package is installable only as PARTIAL; the browser half is persisted as skipped metadata.",
            "- Agent-loop replacement, unknown injected services, npm/peer dependencies, UI-only packages, missing builds, and missing entries fail before approval.",
            "- A syntactically valid plugin cannot turn the shell shim into raw command execution; the forbidden command is rejected before `web_fetch`.",
            "- The experiment does not need an LLM. Existing user LLM/cache configuration is probed without resolving secrets, and no credential value is emitted.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real DeepSeek Harness artifacts through LeapFlow's DSH P0 bridge."
    )
    parser.add_argument(
        "--harness-root",
        type=Path,
        default=DEFAULT_HARNESS_ROOT,
        help="Local deepseek-harness checkout.",
    )
    parser.add_argument(
        "--user-data-root",
        type=Path,
        default=DEFAULT_USER_DATA_ROOT,
        help="Existing LeapFlow data root probed read-only for cache/LLM availability.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep materialized artifacts and the isolated profile after the run.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = EXP_ROOT / "work" / "native-dsh" / stamp
    artifact_root = run_root / "sources"
    runtime = IsolatedRuntime(run_root)
    try:
        specs = ArtifactBuilder(args.harness_root, artifact_root).build()
        cases = await _run_experiment(specs, runtime)
        environment = _environment_evidence(args.user_data_root.expanduser(), args.harness_root)
        payload = {
            "experiment": "native_dsh_plugin_compatibility",
            "generated_at": stamp,
            "ok": all(case["ok"] for case in cases),
            "passed": sum(1 for case in cases if case["ok"]),
            "total": len(cases),
            "isolated_profile_root": str(runtime.profile_root),
            "environment": environment,
            "cases": cases,
        }
        report_root = EXP_ROOT / "reports"
        report_root.mkdir(parents=True, exist_ok=True)
        json_path = report_root / f"{stamp}-native-dsh-plugin-exp.json"
        markdown_path = report_root / f"{stamp}-native-dsh-plugin-exp.md"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": payload["ok"],
                    "passed": payload["passed"],
                    "total": payload["total"],
                    "json_report": str(json_path),
                    "markdown_report": str(markdown_path),
                    "work_kept": bool(args.keep_work),
                },
                ensure_ascii=False,
            )
        )
        return 0 if payload["ok"] else 1
    finally:
        runtime.close()
        if not args.keep_work:
            shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
