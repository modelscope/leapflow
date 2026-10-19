# Copyright (c) Alibaba, Inc. and its affiliates.
"""Validate skill arguments against a skill's declared JSON-Schema subset.

The node validates every argument set itself, before ``plan`` runs: the host's
LLM wrote these values, and a skill must be able to trust its ``args`` mapping
without re-checking types. The supported subset is exactly what skill cards
need -- scalars, enums, arrays, and nested objects such as the steps of a motion
program -- so the node needs no schema library.

Supported keywords: ``type`` (``object``, ``number``, ``integer``, ``string``,
``boolean``, ``array``), ``properties``, ``required``, ``additionalProperties``
(default false at every level: an unknown argument is a mistake, not an
extension), ``default``, ``enum``, ``minimum``, ``maximum``, ``items``,
``minItems``, ``maxItems``, ``minLength``, ``maxLength``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS,
    SkillError,
)

_SCALAR_TYPES = frozenset({"number", "integer", "string", "boolean"})


def validate_args(schema: Mapping[str, Any], args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return *args* with defaults applied, or raise ``SkillError(invalid_args)``."""
    if args is not None and not isinstance(args, Mapping):
        raise _invalid("arguments must be an object")
    return _check_object("", schema, args or {})


def _check_object(path: str, schema: Mapping[str, Any], value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(f"argument {path!r} must be an object")
    supplied = dict(value)
    prefix = f"{path}." if path else ""
    where = f" in {path!r}" if path else ""
    properties: Mapping[str, Any] = schema.get("properties") or {}
    if not schema.get("additionalProperties", False):
        unknown = sorted(str(key) for key in set(supplied) - set(properties))
        if unknown:
            raise _invalid(
                f"unknown argument(s) {', '.join(unknown)}{where}; "
                f"accepted: {', '.join(sorted(properties)) or 'none'}"
            )
    result: dict[str, Any] = {}
    for name, prop in properties.items():
        if name in supplied and supplied[name] is not None:
            result[name] = _check(f"{prefix}{name}", prop, supplied[name])
        elif "default" in prop:
            result[name] = prop["default"]
    missing = [name for name in schema.get("required") or () if name not in result]
    if missing:
        raise _invalid(f"missing required argument(s) {', '.join(missing)}{where}")
    for name in set(supplied) - set(properties):
        result[name] = supplied[name]
    return result


def _check(name: str, prop: Mapping[str, Any], value: Any) -> Any:
    kind = str(prop.get("type") or "")
    if kind == "object":
        return _check_object(name, prop, value)
    if kind == "array":
        return _check_array(name, prop, value)
    if kind not in _SCALAR_TYPES:
        raise SkillError(
            FAILURE_INVALID_ARGS, f"argument {name!r} declares unsupported type {kind!r}"
        )
    return _check_scalar(name, kind, prop, value)


def _check_scalar(name: str, kind: str, prop: Mapping[str, Any], value: Any) -> Any:
    if kind == "boolean":
        if not isinstance(value, bool):
            raise _invalid(f"argument {name!r} must be a boolean")
        checked: Any = value
    elif kind == "string":
        if not isinstance(value, str):
            raise _invalid(f"argument {name!r} must be a string")
        _check_length(name, prop, len(value), "characters")
        checked = value
    else:
        checked = _number(name, kind, value)
        _check_range(name, prop, checked)
    allowed = prop.get("enum")
    if allowed is not None and checked not in allowed:
        raise _invalid(f"argument {name!r} must be one of {list(allowed)}")
    return checked


def _check_array(name: str, prop: Mapping[str, Any], value: Any) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise _invalid(f"argument {name!r} must be an array")
    items = list(value)
    lengths = {"minLength": prop.get("minItems"), "maxLength": prop.get("maxItems")}
    _check_length(name, lengths, len(items), "items")
    item_schema: Mapping[str, Any] = prop.get("items") or {}
    if not item_schema.get("type"):
        return items
    return [_check(f"{name}[{i}]", item_schema, v) for i, v in enumerate(items)]


def _number(name: str, kind: str, value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"argument {name!r} must be a {kind}")
    if not math.isfinite(float(value)):
        raise _invalid(f"argument {name!r} must be finite")
    if kind == "integer":
        if float(value) != int(value):
            raise _invalid(f"argument {name!r} must be an integer")
        return int(value)
    return float(value)


def _check_range(name: str, prop: Mapping[str, Any], value: float) -> None:
    low, high = prop.get("minimum"), prop.get("maximum")
    if low is not None and value < float(low):
        raise _invalid(f"argument {name!r}={value} is below the minimum {low}")
    if high is not None and value > float(high):
        raise _invalid(f"argument {name!r}={value} is above the maximum {high}")


def _check_length(name: str, prop: Mapping[str, Any], length: int, unit: str) -> None:
    low, high = prop.get("minLength"), prop.get("maxLength")
    if low is not None and length < int(low):
        raise _invalid(f"argument {name!r} needs at least {low} {unit}")
    if high is not None and length > int(high):
        raise _invalid(f"argument {name!r} allows at most {high} {unit}")


def _invalid(message: str) -> SkillError:
    return SkillError(FAILURE_INVALID_ARGS, message)


__all__ = ["validate_args"]
