# Copyright (c) Alibaba, Inc. and its affiliates.
"""Benchmark manifest loading and strict validation.

A manifest is a YAML file (or directory of YAML files) that declares a
benchmark's identity, adapter, tier, scenarios, metrics, gates, and
reproducibility constraints.  The loader rejects malformed input with typed
``ManifestError`` instances instead of propagating ad-hoc dicts.

Dependencies: standard library + PyYAML (already in project deps).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from benchmarks.models import BenchmarkManifest, ManifestError, Scenario

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# Schema constants
# ════════════════════════════════════════════════════════════════

_REQUIRED_FIELDS = ("id", "version", "adapter", "tier")
_ALLOWED_FIELDS = frozenset({
    "id", "version", "source", "paper", "repo", "license", "setup_guidance",
    "note", "adapter", "tier", "tags", "required", "dependencies", "timeouts",
    "seeds", "metrics", "gates", "official", "scenarios",
})
_ALLOWED_SCENARIO_FIELDS = frozenset({
    "scenario_id", "name", "description", "adapter_id", "tags",
    "timeout_seconds", "required", "parameters",
})

_STRING_FIELDS = (
    "id", "version", "source", "paper", "repo", "license", "setup_guidance", "note", "adapter",
)
_INT_FIELDS = ("tier",)
_BOOL_FIELDS = ("required",)
_STRING_LIST_FIELDS = ("tags", "dependencies", "metrics")
_INT_LIST_FIELDS = ("seeds",)

_VALID_TIERS = frozenset(range(6))  # 0..5; tier 5 is explicit live-provider evaluation


# ════════════════════════════════════════════════════════════════
# Validation helpers
# ════════════════════════════════════════════════════════════════


def _error(field_name: str, message: str, path: str) -> ManifestError:
    """Build a typed validation error."""
    return ManifestError(field=field_name, message=message, path=path)


def _validate_root_fields(data: Mapping[str, Any], path: str) -> list[ManifestError]:
    """Validate root keys and scalar values."""
    errors = [_error(str(key), "unknown field", path) for key in data if key not in _ALLOWED_FIELDS]
    errors.extend(
        _error(field_name, "required field missing", path)
        for field_name in _REQUIRED_FIELDS if field_name not in data
    )
    for field_name in _STRING_FIELDS:
        value = data.get(field_name)
        if value is not None and not isinstance(value, str):
            errors.append(_error(field_name, f"expected string, got {type(value).__name__}", path))
        elif field_name in _REQUIRED_FIELDS and isinstance(value, str) and not value.strip():
            errors.append(_error(field_name, "must not be empty", path))
    for field_name in _INT_FIELDS:
        value = data.get(field_name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            errors.append(_error(field_name, f"expected integer, got {type(value).__name__}", path))
    for field_name in _BOOL_FIELDS:
        value = data.get(field_name)
        if value is not None and not isinstance(value, bool):
            errors.append(_error(field_name, f"expected boolean, got {type(value).__name__}", path))
    tier = data.get("tier")
    if isinstance(tier, int) and not isinstance(tier, bool) and tier not in _VALID_TIERS:
        errors.append(_error("tier", f"tier must be 0..5, got {tier}", path))
    return errors


def _validate_list_fields(data: Mapping[str, Any], path: str) -> list[ManifestError]:
    """Validate homogeneous list fields."""
    errors: list[ManifestError] = []
    for field_name in _STRING_LIST_FIELDS:
        value = data.get(field_name)
        if value is not None and not isinstance(value, (list, tuple)):
            errors.append(_error(field_name, f"expected list, got {type(value).__name__}", path))
        elif value is not None and not all(isinstance(item, str) for item in value):
            errors.append(_error(field_name, "all items must be strings", path))
    for field_name in _INT_LIST_FIELDS:
        value = data.get(field_name)
        if value is not None and not isinstance(value, (list, tuple)):
            errors.append(_error(field_name, f"expected list, got {type(value).__name__}", path))
        elif value is not None and not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            errors.append(_error(field_name, "all items must be integers", path))
    return errors


def _validate_number_mapping(
    data: Mapping[str, Any], field_name: str, path: str, *, positive: bool,
) -> list[ManifestError]:
    """Validate a string-to-number mapping."""
    value = data.get(field_name)
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [_error(field_name, f"expected mapping, got {type(value).__name__}", path)]
    errors: list[ManifestError] = []
    for key, item in value.items():
        item_field = f"{field_name}.{key}"
        if not isinstance(key, str):
            errors.append(_error(item_field, "mapping keys must be strings", path))
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(_error(item_field, f"expected number, got {type(item).__name__}", path))
        elif positive and item <= 0:
            errors.append(_error(item_field, "must be greater than zero", path))
    return errors


def _validate_scenario(data: Mapping[str, Any], index: int, path: str) -> list[ManifestError]:
    """Validate one inline scenario declaration."""
    prefix = f"scenarios[{index}]"
    errors = [
        _error(f"{prefix}.{key}", "unknown field", path)
        for key in data if key not in _ALLOWED_SCENARIO_FIELDS
    ]
    identity = data.get("scenario_id", data.get("name"))
    if not isinstance(identity, str) or not identity.strip():
        errors.append(_error(prefix, "scenario must have a non-empty string id or name", path))
    for field_name in ("scenario_id", "name", "description", "adapter_id"):
        value = data.get(field_name)
        if value is not None and not isinstance(value, str):
            errors.append(_error(f"{prefix}.{field_name}", "expected string", path))
    tags = data.get("tags")
    if tags is not None and (
        not isinstance(tags, (list, tuple)) or not all(isinstance(tag, str) for tag in tags)
    ):
        errors.append(_error(f"{prefix}.tags", "expected a list of strings", path))
    timeout = data.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
    ):
        errors.append(_error(f"{prefix}.timeout_seconds", "expected a positive number", path))
    required = data.get("required")
    if required is not None and not isinstance(required, bool):
        errors.append(_error(f"{prefix}.required", "expected boolean", path))
    if data.get("parameters") is not None and not isinstance(data["parameters"], Mapping):
        errors.append(_error(f"{prefix}.parameters", "expected mapping", path))
    return errors


def _validate_scenarios(data: Mapping[str, Any], path: str) -> list[ManifestError]:
    """Validate inline scenarios and reject duplicate ids."""
    scenarios = data.get("scenarios")
    if scenarios is None:
        return []
    if not isinstance(scenarios, (list, tuple)):
        return [_error("scenarios", f"expected list, got {type(scenarios).__name__}", path)]
    errors: list[ManifestError] = []
    seen: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            errors.append(_error(f"scenarios[{index}]", "each scenario must be a mapping", path))
            continue
        errors.extend(_validate_scenario(scenario, index, path))
        identity = scenario.get("scenario_id", scenario.get("name"))
        if isinstance(identity, str) and identity in seen:
            errors.append(_error(f"scenarios[{index}]", f"duplicate scenario id {identity!r}", path))
        elif isinstance(identity, str):
            seen.add(identity)
    return errors


def _validate_raw(data: Any, path: str = "") -> list[ManifestError]:
    """Validate raw YAML data and return typed errors."""
    if not isinstance(data, Mapping):
        return [_error("root", "manifest must be a mapping", path)]
    errors = _validate_root_fields(data, path)
    errors.extend(_validate_list_fields(data, path))
    errors.extend(_validate_number_mapping(data, "timeouts", path, positive=True))
    errors.extend(_validate_number_mapping(data, "gates", path, positive=False))
    official = data.get("official")
    if official is not None and not isinstance(official, Mapping):
        errors.append(_error("official", f"expected mapping, got {type(official).__name__}", path))
    errors.extend(_validate_scenarios(data, path))
    return errors


def _parse_scenario(data: Mapping[str, Any], adapter_id: str = "") -> Scenario:
    """Parse one scenario entry from a manifest."""
    sid = str(data.get("scenario_id") or data.get("name") or "")
    return Scenario(
        scenario_id=sid,
        name=str(data.get("name", sid)),
        description=str(data.get("description", "")),
        adapter_id=str(data.get("adapter_id", adapter_id)),
        tags=tuple(str(t) for t in data.get("tags", ())),
        timeout_seconds=(
            float(data["timeout_seconds"]) if data.get("timeout_seconds") is not None else None
        ),
        required=bool(data.get("required", False)),
        parameters=dict(data.get("parameters", {})),
    )


def _parse_manifest(data: Mapping[str, Any], path: str = "") -> BenchmarkManifest:
    """Convert validated raw data into a ``BenchmarkManifest``."""
    adapter_id = str(data.get("adapter", ""))
    scenarios_raw = data.get("scenarios", ())
    scenarios = tuple(
        _parse_scenario(s, adapter_id=adapter_id)
        for s in scenarios_raw
        if isinstance(s, Mapping)
    )

    return BenchmarkManifest(
        id=str(data.get("id", "")),
        version=str(data.get("version", "0.0.0")),
        source=str(data.get("source", "")),
        paper=str(data.get("paper", "")),
        repo=str(data.get("repo", "")),
        license=str(data.get("license", "")),
        setup_guidance=str(data.get("setup_guidance", "")),
        note=str(data.get("note", "")),
        adapter=adapter_id,
        tier=int(data.get("tier", 0)),
        tags=tuple(str(t) for t in data.get("tags", ())),
        required=bool(data.get("required", False)),
        dependencies=tuple(str(d) for d in data.get("dependencies", ())),
        timeouts={str(k): float(v) for k, v in (data.get("timeouts") or {}).items()},
        seeds=tuple(int(s) for s in data.get("seeds", (42,))),
        metrics=tuple(str(m) for m in data.get("metrics", ())),
        gates={str(k): float(v) for k, v in (data.get("gates") or {}).items()},
        official=dict(data.get("official") or {}),
        scenarios=scenarios,
    )


# ════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════


def load_manifest(
    source: str | Path,
) -> tuple[BenchmarkManifest | None, list[ManifestError]]:
    """Load and validate a single manifest YAML file.

    Returns ``(manifest, errors)``.  When *errors* is non-empty the manifest
    may be ``None`` (fatal) or partially populated (warnings).
    """
    path = Path(source)
    if not path.exists():
        return None, [ManifestError(field="path", message="file not found", path=str(path))]

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [ManifestError(field="path", message=str(exc), path=str(path))]

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, [ManifestError(field="yaml", message=str(exc), path=str(path))]

    if data is None:
        return None, [ManifestError(field="root", message="empty document", path=str(path))]

    if not isinstance(data, Mapping):
        return None, _validate_raw(data, path=str(path))

    errors = _validate_raw(data, path=str(path))
    if errors:
        return None, errors

    manifest = _parse_manifest(data, path=str(path))
    return manifest, errors


def load_manifests(
    source: str | Path,
) -> tuple[list[BenchmarkManifest], list[ManifestError]]:
    """Load manifests from a file or directory (sorted by id for stability).

    If *source* is a directory, all ``*.yaml`` and ``*.yml`` files in it are
    loaded (non-recursively).  Order is stable: files sorted by name, then
    manifests sorted by ``id``.
    """
    path = Path(source)
    all_manifests: list[BenchmarkManifest] = []
    all_errors: list[ManifestError] = []

    if path.is_file():
        manifest, errors = load_manifest(path)
        all_errors.extend(errors)
        if manifest is not None:
            all_manifests.append(manifest)
    elif path.is_dir():
        files = sorted(
            f for f in path.iterdir()
            if f.is_file() and f.suffix in (".yaml", ".yml")
        )
        for f in files:
            manifest, errors = load_manifest(f)
            all_errors.extend(errors)
            if manifest is not None:
                all_manifests.append(manifest)
    else:
        all_errors.append(ManifestError(
            field="path", message="not a file or directory", path=str(path),
        ))

    all_manifests.sort(key=lambda m: m.id)
    return all_manifests, all_errors


__all__ = [
    "load_manifest",
    "load_manifests",
]
