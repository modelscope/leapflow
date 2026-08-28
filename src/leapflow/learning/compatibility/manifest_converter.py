"""DSH package.json → LeapFlow compatibility descriptor conversion.

This is metadata for assessment and audit, not a ``MarketplaceClient`` install
manifest. A DSH ``main`` such as ``dist/index.js`` is not a Python entry point;
passing it to the Python-only marketplace would fabricate ``dist/index.js.py``.
Executable DSH bundles must enter through ``plugin_install(source_path=...)``.
"""

from __future__ import annotations

import re
from typing import Any


def convert_dsh_to_leapflow(dsh_manifest: dict) -> dict:
    """Convert a DSH manifest into a non-installable compatibility descriptor.

    Field mapping:
        - ``name``        → stripped of ``@org/`` and ``dsh-`` prefixes, hyphens
                            replaced with underscores → LeapFlow ``name``
        - ``version``     → passed through (defaults to ``"0.0.0"``)
        - ``main``        → ``entry_point`` (the JS entry, for bridge reference)
        - ``description`` → passed through (defaults to ``""``)
        - ``dsh.category`` / ``keywords[0]`` → informs ``requires_sandbox``
                            (always ``True`` for TS plugins)
        - ``checksum_sha256`` → ``None`` (computed at install time)
        - ``x_dsh_original`` → the full original manifest, preserved for audit

    Args:
        dsh_manifest: A DSH package.json-like manifest dict.

    Returns:
        A LeapFlow PluginManifest-compatible dict.
    """
    raw_name = dsh_manifest.get("name", "")
    leapflow_name = _normalize_name(raw_name)

    version = dsh_manifest.get("version", "0.0.0")
    entry_point = dsh_manifest.get("main", "")
    description = dsh_manifest.get("description", "")

    category = _infer_category(dsh_manifest)

    # DSH plugins are TypeScript/JavaScript and always run through a subprocess
    # bridge, so they are always treated as untrusted and require sandboxing.
    requires_sandbox = True

    dependencies = _extract_dependencies(dsh_manifest)

    return {
        "name": leapflow_name,
        "version": version,
        "entry_point": entry_point,
        "description": description,
        "plugin_type": "tool",
        "source_language": "javascript",
        "artifact_type": "dsh_source_bundle",
        "requires_sandbox": requires_sandbox,
        "dependencies": dependencies,
        "checksum_sha256": None,  # computed at install time, not conversion
        "x_dsh_category": category,
        "x_dsh_original": dict(dsh_manifest),
    }


def _normalize_name(raw_name: str) -> str:
    """Normalize a DSH package name into a LeapFlow plugin name.

    Strips a leading ``@org/`` scope, a ``dsh-`` prefix, then replaces
    hyphens with underscores. Any remaining non-identifier character (dots,
    quotes, whitespace, ...) is collapsed to an underscore so the result is
    always a valid Python identifier fragment suitable for code generation.
    Returns ``"unknown_plugin"`` for empty input.
    """
    if not isinstance(raw_name, str) or not raw_name:
        return "unknown_plugin"

    name = raw_name
    # Strip org scope like "@deepseek-ai/"
    if "/" in name:
        name = name.split("/", 1)[1]
    # Strip a leading "dsh-" prefix
    if name.startswith("dsh-"):
        name = name[len("dsh-"):]
    # LeapFlow plugin names are snake_case identifiers
    name = name.replace("-", "_")
    # Keep only valid Python identifier characters so downstream code
    # generation (class names, plugin ids) never emits invalid source.
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    name = name.strip("_")
    return name or "unknown_plugin"


def _infer_category(dsh_manifest: dict) -> str:
    """Infer a category label from the ``dsh``/``leapflow`` section or keywords."""
    metadata = dsh_manifest.get("dsh", dsh_manifest.get("leapflow", {}))
    if isinstance(metadata, dict):
        category = metadata.get("category")
        if isinstance(category, str) and category:
            return category

    keywords = dsh_manifest.get("keywords", [])
    if isinstance(keywords, list):
        for kw in keywords:
            if isinstance(kw, str) and kw:
                return kw

    return ""


def _extract_dependencies(dsh_manifest: dict) -> list[str]:
    """Extract dependency names from a DSH ``dependencies`` mapping."""
    deps_raw: Any = dsh_manifest.get("dependencies", {})
    if isinstance(deps_raw, dict):
        return list(deps_raw.keys())
    if isinstance(deps_raw, list):
        return [d for d in deps_raw if isinstance(d, str)]
    return []
