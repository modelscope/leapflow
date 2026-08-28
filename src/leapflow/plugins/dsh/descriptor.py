"""Persistent descriptors and wrapper generation for installed DSH plugins."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DESCRIPTOR_VERSION = 1
_PLUGIN_ID_RE = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class DshToolDescriptor:
    name: str
    description: str
    parameters_schema: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DshToolDescriptor":
        name = str(value.get("name") or "")
        if not name or not name.replace("_", "").isalnum() or name.lower() != name:
            raise ValueError(f"Invalid DSH tool name: {name!r}")
        description = str(value.get("description") or "").strip()
        if not description:
            raise ValueError(f"DSH tool {name!r} requires a description")
        schema = value.get("parameters_schema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"DSH tool {name!r} requires an object parameters schema")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"DSH tool {name!r} schema.properties must be an object")
        required = schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or any(item not in properties for item in required)
        ):
            raise ValueError(
                f"DSH tool {name!r} schema.required must name declared properties"
            )
        return cls(name=name, description=description, parameters_schema=dict(schema))


@dataclass(frozen=True)
class DshPluginDescriptor:
    plugin_id: str
    name: str
    source_kind: str
    bundle_root: str
    entry_point: str
    bundle_sha256: str
    runtime_sha256: str
    source_files: tuple[str, ...]
    tools: tuple[DshToolDescriptor, ...]
    permissions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    client_components: tuple[dict[str, Any], ...] = ()
    descriptor_version: int = _DESCRIPTOR_VERSION
    category: str = "bridge"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_files"] = list(self.source_files)
        value["tools"] = [asdict(tool) for tool in self.tools]
        value["permissions"] = list(self.permissions)
        value["limitations"] = list(self.limitations)
        value["client_components"] = [dict(item) for item in self.client_components]
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DshPluginDescriptor":
        if int(value.get("descriptor_version") or 0) != _DESCRIPTOR_VERSION:
            raise ValueError("Unsupported DSH plugin descriptor version")
        plugin_id = normalize_plugin_id(str(value.get("plugin_id") or ""))
        if plugin_id != value.get("plugin_id"):
            raise ValueError("DSH descriptor plugin_id is not normalized")
        root = Path(str(value.get("bundle_root") or "")).expanduser().resolve()
        source_files = tuple(str(item) for item in value.get("source_files") or ())
        if not source_files:
            raise ValueError("DSH descriptor has no recorded source files")
        expected_hash = str(value.get("bundle_sha256") or "")
        expected_runtime_hash = str(value.get("runtime_sha256") or "")
        if not expected_hash or not expected_runtime_hash:
            raise ValueError("DSH descriptor requires source and runtime hashes")
        raw_tools = value.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValueError("DSH descriptor must expose at least one public tool")
        tools = tuple(DshToolDescriptor.from_dict(dict(item)) for item in raw_tools)
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("DSH descriptor contains duplicate tool names")
        descriptor = cls(
            plugin_id=plugin_id,
            name=str(value.get("name") or plugin_id),
            source_kind=str(value.get("source_kind") or ""),
            bundle_root=str(root),
            entry_point=str(value.get("entry_point") or ""),
            bundle_sha256=expected_hash,
            runtime_sha256=expected_runtime_hash,
            source_files=source_files,
            tools=tools,
            permissions=tuple(str(item) for item in value.get("permissions") or ()),
            limitations=tuple(str(item) for item in value.get("limitations") or ()),
            client_components=tuple(dict(item) for item in value.get("client_components") or ()),
            descriptor_version=_DESCRIPTOR_VERSION,
            category=str(value.get("category") or "bridge"),
        )
        descriptor.verify_integrity()
        return descriptor

    def verify_integrity(self) -> None:
        """Refuse execution when installed source or the runtime wrapper changed."""
        from leapflow.learning.compatibility.source_inspector import hash_source_files

        root = Path(self.bundle_root).expanduser().resolve()
        if hash_source_files(root, self.source_files) != self.bundle_sha256:
            raise ValueError("DSH source bundle hash does not match its approved descriptor")
        entry = (root / self.entry_point).resolve()
        try:
            entry.relative_to(root)
        except ValueError as exc:
            raise ValueError("DSH descriptor entry point escapes bundle root") from exc
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"DSH descriptor entry point is missing or unsafe: {entry}")
        if _sha256_file(entry) != self.runtime_sha256:
            raise ValueError("DSH runtime entry hash does not match its approved descriptor")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DshPluginDescriptor":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("DSH descriptor must contain a JSON object")
        return cls.from_dict(value)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_plugin_id(value: str) -> str:
    normalized = _PLUGIN_ID_RE.sub("_", value.lower().replace("-", "_"))
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("DSH plugin id normalizes to an empty value")
    if normalized[0].isdigit():
        normalized = f"dsh_{normalized}"
    return normalized


def render_python_wrapper(descriptor: DshPluginDescriptor) -> str:
    """Render a deterministic native wrapper discoverable after daemon restart."""
    payload = repr(descriptor.to_dict())
    return (
        '"""Generated LeapFlow wrapper for an installed DSH plugin bundle."""\n'
        "from leapflow.plugins.dsh.plugin import DshBridgePlugin\n\n"
        f"plugin = DshBridgePlugin({payload})\n"
    )
