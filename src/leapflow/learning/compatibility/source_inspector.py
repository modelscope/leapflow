"""Inspect real DSH/Cordis source bundles without executing foreign code.

Static inspection establishes source identity, bounds, integrity and component
shape. It deliberately does not claim runtime compatibility: only the restricted
Node discovery path may promote a host component from CANDIDATE to RUNTIME_READY.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leapflow.learning.compatibility.protocol import (
    ComponentCompatibility,
    ComponentKind,
    ComponentStatus,
    ExecutionPlan,
    PluginManifestInput,
    PluginSourceKind,
)

from leapflow.learning.compatibility.stages.manifest_parser import extract_dsh_category

_MAX_BUNDLE_FILES = 128
_MAX_BUNDLE_BYTES = 5_000_000
_JS_ENTRY_SUFFIXES = frozenset({".js", ".mjs", ".cjs"})
_TOOL_RE = re.compile(r"(?:harness\.)?registerTool\s*\(")
_HANDLER_RE = re.compile(r"harness\.handle\s*\(\s*(['\"])([^'\"]+)\1")
_SLOT_RE = re.compile(r"slots\.inject\s*\(\s*(['\"])([^'\"]+)\1")
_CTX_SERVICE_RE = re.compile(r"ctx\.get\s*\(\s*(['\"])([^'\"]+)\1")
_INJECT_RE = re.compile(r"\binject\s*[:=]\s*\[(?P<items>[^\]]*)\]", re.DOTALL)
_QUOTED_NAME_RE = re.compile(r"(['\"])(?P<name>[^'\"]+)\1")
_P0_HOST_SERVICES = frozenset({"shell", "tools"})


class SourceInspectionError(ValueError):
    """The source bundle is malformed, unsafe, or unsupported in P0."""


@dataclass(frozen=True)
class SourceInspection:
    """Static source inspection result consumed by the assessment pipeline."""

    manifest: PluginManifestInput
    execution_plan: ExecutionPlan


def inspect_plugin_source(source: str | Path) -> SourceInspection:
    """Inspect a DSH package or dynamic Cordis export directory.

    Supported P0 shapes:
    - ``package.json`` plus an existing pre-built ``.js/.mjs/.cjs`` entry.
    - ``meta.json`` plus ``host.js`` and optional ``client.js`` function bodies.

    Symlinks and nested path escapes are rejected before any source is read.
    """
    path = Path(source).expanduser()
    if path.is_symlink():
        raise SourceInspectionError(f"Plugin source must not be a symlink: {path}")
    if path.is_file():
        if path.name not in {"package.json", "meta.json"}:
            raise SourceInspectionError(
                "DSH source file must be package.json or meta.json; pass the bundle directory"
            )
        root = path.parent
    elif path.is_dir():
        root = path
    else:
        raise SourceInspectionError(f"Plugin source does not exist: {path}")

    root = root.resolve()
    files, bundle_bytes, bundle_hash = _bounded_bundle(root)
    source_files = tuple(item.relative_to(root).as_posix() for item in files)
    file_names = set(source_files)
    if "package.json" in file_names:
        return _inspect_package(root, bundle_bytes, bundle_hash, source_files)
    if "meta.json" in file_names and "host.js" in file_names:
        return _inspect_dynamic_export(root, bundle_bytes, bundle_hash, source_files)
    raise SourceInspectionError(
        "Unrecognized DSH source layout: expected package.json, or meta.json + host.js"
    )


def _bounded_bundle(root: Path) -> tuple[list[Path], int, str]:
    files: list[Path] = []
    total = 0
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise SourceInspectionError(f"Plugin bundle contains a symlink: {item}")
        resolved = item.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise SourceInspectionError(f"Plugin bundle path escapes its root: {item}") from exc
        if item.is_dir():
            continue
        if not item.is_file():
            raise SourceInspectionError(f"Plugin bundle contains a non-regular file: {item}")
        files.append(item)
        if len(files) > _MAX_BUNDLE_FILES:
            raise SourceInspectionError(
                f"Plugin bundle exceeds {_MAX_BUNDLE_FILES} files"
            )
        data = item.read_bytes()
        total += len(data)
        if total > _MAX_BUNDLE_BYTES:
            raise SourceInspectionError(
                f"Plugin bundle exceeds {_MAX_BUNDLE_BYTES} bytes"
            )
        rel_bytes = relative.as_posix().encode("utf-8")
        digest.update(len(rel_bytes).to_bytes(4, "big"))
        digest.update(rel_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return files, total, digest.hexdigest()


def hash_source_files(root: str | Path, source_files: tuple[str, ...]) -> str:
    """Hash the exact original source-file set recorded during inspection."""
    base = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    for relative_text in sorted(source_files):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise SourceInspectionError(f"Unsafe recorded source path: {relative_text}")
        item = base / relative
        if item.is_symlink() or not item.is_file():
            raise SourceInspectionError(f"Recorded source file is missing or unsafe: {item}")
        data = item.read_bytes()
        rel_bytes = relative.as_posix().encode("utf-8")
        digest.update(len(rel_bytes).to_bytes(4, "big"))
        digest.update(rel_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceInspectionError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceInspectionError(f"{path.name} must contain a JSON object")
    return value


def _dependency_names(raw: dict[str, Any]) -> tuple[str, ...]:
    """Return runtime package dependencies from all npm dependency sections."""
    names: set[str] = set()
    for field in ("dependencies", "peerDependencies", "optionalDependencies"):
        value = raw.get(field) or {}
        if not isinstance(value, dict):
            raise SourceInspectionError(f"package.json {field} must be an object")
        names.update(str(key) for key in value)
    return tuple(sorted(names))


def _safe_entry(root: Path, raw_entry: str) -> tuple[Path, str]:
    if not raw_entry:
        raise SourceInspectionError("DSH package is missing its pre-built JavaScript entry")
    entry = (root / raw_entry).resolve()
    try:
        relative = entry.relative_to(root)
    except ValueError as exc:
        raise SourceInspectionError(f"DSH entry point escapes the bundle: {raw_entry}") from exc
    if entry.is_symlink() or not entry.is_file():
        raise SourceInspectionError(f"DSH entry point does not exist: {raw_entry}")
    if entry.suffix.lower() not in _JS_ENTRY_SUFFIXES:
        if entry.suffix.lower() in {".ts", ".tsx"}:
            raise SourceInspectionError(
                "P0 requires a pre-built JavaScript entry; TypeScript build/install is not supported"
            )
        raise SourceInspectionError(
            f"Unsupported DSH entry suffix {entry.suffix!r}; expected .js/.mjs/.cjs"
        )
    return entry, relative.as_posix()


def _inspect_package(
    root: Path,
    bundle_bytes: int,
    bundle_hash: str,
    source_files: tuple[str, ...],
) -> SourceInspection:
    raw = _read_json_object(root / "package.json")
    name = str(raw.get("name") or "").strip()
    version = str(raw.get("version") or "").strip()
    if not name or not version:
        raise SourceInspectionError("DSH package.json requires non-empty name and version")
    raw_entry = str(raw.get("main") or raw.get("module") or "").strip()
    entry, relative_entry = _safe_entry(root, raw_entry)
    dependencies = _dependency_names(raw)
    blockers: list[str] = []
    if dependencies:
        blockers.append(
            "P0 does not install npm dependencies; provide a self-contained pre-built bundle"
        )
    source = entry.read_text(encoding="utf-8")
    services = _extract_services(source)
    permissions = _permissions_for_services(services)
    blockers.extend(_service_blockers(services))
    metadata = raw.get("dsh", raw.get("leapflow", {}))
    category = extract_dsh_category(raw) or "tools"
    interfaces = []
    if isinstance(metadata, dict) and isinstance(metadata.get("interfaces"), list):
        interfaces = [str(value) for value in metadata["interfaces"]]
    components = (
        ComponentCompatibility(
            name="host",
            kind=ComponentKind.HOST,
            status=ComponentStatus.CANDIDATE,
            reason="Pre-built JavaScript entry requires restricted Node runtime discovery",
            entry_point=relative_entry,
            metadata={"services": list(services), "declared_interfaces": interfaces},
        ),
    )
    manifest_raw = dict(raw)
    manifest_raw["x_leapflow_source"] = _source_metadata(
        root, PluginSourceKind.DSH_PACKAGE, relative_entry, bundle_bytes, bundle_hash
    )
    manifest = PluginManifestInput(
        name=name,
        version=version,
        category=category,
        declared_interfaces=interfaces,
        declared_dependencies=list(dependencies),
        execution_model="subprocess",
        permissions=list(permissions),
        source_language="javascript",
        raw_manifest=manifest_raw,
        source_format="dsh",
    )
    plan = ExecutionPlan(
        source_kind=PluginSourceKind.DSH_PACKAGE,
        source_root=str(root),
        entry_point=relative_entry,
        runtime="node",
        bundle_sha256=bundle_hash,
        source_files=source_files,
        requires_discovery=True,
        dependencies=dependencies,
        permissions=permissions,
        components=components,
        blockers=tuple(blockers),
    )
    return SourceInspection(manifest=manifest, execution_plan=plan)


def _inspect_dynamic_export(
    root: Path,
    bundle_bytes: int,
    bundle_hash: str,
    source_files: tuple[str, ...],
) -> SourceInspection:
    meta = _read_json_object(root / "meta.json")
    name = str(meta.get("name") or "").strip()
    if not name:
        raise SourceInspectionError("Cordis export meta.json requires a non-empty name")
    version = str(meta.get("version") or "0.0.0+export")
    host_source = (root / "host.js").read_text(encoding="utf-8")
    services = _extract_services(host_source)
    permissions = _permissions_for_services(services)
    public_tool_markers = len(_TOOL_RE.findall(host_source))
    handler_names = tuple(match[1] for match in _HANDLER_RE.findall(host_source))
    blockers = _service_blockers(services)
    if public_tool_markers == 0:
        blockers.append(
            "Dynamic export contains no statically visible registerTool call; "
            "P0 does not publish private handler channels as LeapFlow tools"
        )
    components: list[ComponentCompatibility] = [
        ComponentCompatibility(
            name="host",
            kind=ComponentKind.HOST,
            status=ComponentStatus.CANDIDATE,
            reason="Dynamic Cordis host requires restricted runtime discovery",
            entry_point="host.js",
            metadata={
                "services": list(services),
                "public_tool_markers": public_tool_markers,
                "handler_channels": list(handler_names),
            },
        )
    ]
    limitations: list[str] = []
    client_path = root / "client.js"
    if client_path.exists():
        client_source = client_path.read_text(encoding="utf-8")
        slots = tuple(match[1] for match in _SLOT_RE.findall(client_source))
        components.append(
            ComponentCompatibility(
                name="client",
                kind=ComponentKind.CLIENT,
                status=ComponentStatus.UNSUPPORTED,
                reason="Cordis React/slots client UI is not executable in LeapFlow P0",
                entry_point="client.js",
                metadata={"slots": list(slots)},
            )
        )
        limitations.append(
            "client.js UI was detected and will be skipped; only safe host tools can be installed"
        )
    raw = dict(meta)
    raw.update(
        {
            "main": "host.js",
            "keywords": ["tools"],
            "dsh": {
                "category": "tools",
                "interfaces": [],
                "permissions": list(permissions),
                "execution_model": "subprocess",
            },
            "x_leapflow_source": _source_metadata(
                root,
                PluginSourceKind.CORDIS_DYNAMIC_EXPORT,
                "host.js",
                bundle_bytes,
                bundle_hash,
            ),
        }
    )
    manifest = PluginManifestInput(
        name=name,
        version=version,
        category=extract_dsh_category(meta) or "tools",
        declared_interfaces=[],
        declared_dependencies=[],
        execution_model="subprocess",
        permissions=list(permissions),
        source_language="javascript",
        raw_manifest=raw,
        source_format="dsh",
    )
    plan = ExecutionPlan(
        source_kind=PluginSourceKind.CORDIS_DYNAMIC_EXPORT,
        source_root=str(root),
        entry_point="host.js",
        runtime="node",
        bundle_sha256=bundle_hash,
        source_files=source_files,
        requires_discovery=True,
        permissions=permissions,
        components=tuple(components),
        blockers=tuple(blockers),
        limitations=tuple(limitations),
    )
    return SourceInspection(manifest=manifest, execution_plan=plan)


def _extract_services(source: str) -> tuple[str, ...]:
    services = {match[1] for match in _CTX_SERVICE_RE.findall(source)}
    for match in _INJECT_RE.finditer(source):
        services.update(
            quoted.group("name") for quoted in _QUOTED_NAME_RE.finditer(match.group("items"))
        )
    return tuple(sorted(services))


def _service_blockers(services: tuple[str, ...]) -> list[str]:
    return [
        f"P0 does not expose required DSH host service: {service}"
        for service in services
        if service not in _P0_HOST_SERVICES
    ]


def _permissions_for_services(services: tuple[str, ...]) -> tuple[str, ...]:
    permissions: set[str] = set()
    for service in services:
        if service in {"shell", "tools"}:
            if service == "shell":
                # The P0 shell compatibility surface is a strict curl-to-HTTP shim;
                # no raw process execution is exposed to the plugin.
                permissions.add("network.outbound")
                permissions.add("compat.shell.curl_get")
        elif service in {"slots", "layout", "timer"}:
            permissions.add(f"client.{service}")
        else:
            permissions.add(f"unknown.service.{service}")
    return tuple(sorted(permissions))


def _source_metadata(
    root: Path,
    kind: PluginSourceKind,
    entry_point: str,
    bundle_bytes: int,
    bundle_hash: str,
) -> dict[str, Any]:
    return {
        "source_kind": kind.value,
        "source_root": str(root),
        "entry_point": entry_point,
        "bundle_bytes": bundle_bytes,
        "bundle_sha256": bundle_hash,
    }
