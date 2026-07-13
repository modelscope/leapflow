"""Action registry utilities for App Connector platform actions."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping

from leapflow.gateway.connectors.protocol import ActionResult, ActionSpec
from leapflow.security.redact import redact_sensitive_text


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating an action payload against a compact JSON schema subset."""

    ok: bool
    error: str = ""


class ActionRegistry:
    """Registry of platform actions keyed by domain.operation names."""

    def __init__(self, specs: Mapping[str, ActionSpec] | None = None) -> None:
        self._specs = dict(specs or {})

    @classmethod
    def from_module(cls, module_path: str) -> "ActionRegistry":
        """Load action specs from an action-pack module."""
        module = import_module(module_path)
        raw_specs = getattr(module, "ACTION_SPECS", None)
        if raw_specs is None:
            raw_specs = getattr(module, "FEISHU_ACTION_SPECS", None)
        if not isinstance(raw_specs, Mapping):
            raise ValueError(f"Action pack has no ACTION_SPECS mapping: {module_path}")
        return cls({str(key): value for key, value in raw_specs.items()})

    def all(self) -> Mapping[str, ActionSpec]:
        """Return all registered action specs."""
        return dict(self._specs)

    def get(self, action: str) -> ActionSpec | None:
        """Return one registered action spec."""
        return self._specs.get(action)

    def validate(self, action: str, payload: Mapping[str, Any]) -> ValidationResult:
        """Validate payload using the registered action schema."""
        spec = self.get(action)
        if spec is None:
            return ValidationResult(ok=False, error=f"Unknown platform action: {action}")
        return validate_payload(spec, payload)


def validate_payload(spec: ActionSpec, payload: Mapping[str, Any]) -> ValidationResult:
    """Validate a payload against the compact schema subset used by action packs."""
    schema = spec.schema or {}
    required = schema.get("required") or ()
    missing = [str(key) for key in required if _is_missing(payload.get(str(key)))]
    if missing:
        return ValidationResult(ok=False, error=f"Missing required fields: {', '.join(missing)}")

    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        return ValidationResult(ok=True)
    for key, rule in properties.items():
        if key not in payload or payload[key] is None:
            continue
        if not isinstance(rule, Mapping):
            continue
        expected = str(rule.get("type") or "")
        if expected and not _matches_type(payload[key], expected):
            return ValidationResult(ok=False, error=f"Field '{key}' must be {expected}")
    return ValidationResult(ok=True)


def summarize_action_result(spec: ActionSpec, result: ActionResult) -> dict[str, Any]:
    """Apply the action output policy before returning data to the LLM context."""
    if not result.ok:
        return {
            "ok": False,
            "error": redact_sensitive_text(result.error, force=True),
            "output_policy": spec.output_policy,
        }

    summary: dict[str, Any] = {
        "ok": True,
        "resource_id": result.resource_id,
        "output_policy": spec.output_policy,
    }
    if spec.output_policy == "raw":
        summary["data"] = dict(result.data)
        return summary

    compact_data = _compact_data(result.data)
    if compact_data:
        summary["data"] = compact_data
    if "next_steps" in spec.backend_config:
        summary["next_steps"] = spec.backend_config["next_steps"]
    return summary


def _compact_data(data: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("id", "message_id", "token", "url", "link", "permalink", "title", "name"):
        value = data.get(key)
        if value:
            compact[key] = str(value)
    nested = data.get("message") if isinstance(data.get("message"), Mapping) else None
    if isinstance(nested, Mapping) and nested.get("message_id"):
        compact.setdefault("message_id", str(nested["message_id"]))
    return compact


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in {"number", "integer"}:
        return isinstance(value, int if expected == "integer" else (int, float)) and not isinstance(value, bool)
    return True
