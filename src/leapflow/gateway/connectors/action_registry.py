"""Action registry utilities for App Connector platform actions.

Supports three sources in priority order:

1. Static specs (from Python action pack or YAML) — verified, production-grade
2. Dynamically discovered specs (via CLI ``--help``) — draft, high-risk
3. Neither: returns ``None``

Discovered specs are stored separately and never overwrite static specs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from leapflow.gateway.connectors.protocol import ActionResult, ActionSpec
from leapflow.security.redact import redact_sensitive_text

if TYPE_CHECKING:
    from leapflow.gateway.connectors.protocol import ActionDiscovery

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating an action payload against a compact JSON schema subset."""

    ok: bool
    error: str = ""


class ActionRegistry:
    """Registry of platform actions keyed by domain.operation names.

    An optional :class:`ActionDiscovery` source can be attached to
    provide fallback command discovery when static specs lack a match.
    Static specs always take priority over discovered ones.
    """

    def __init__(
        self,
        specs: Mapping[str, ActionSpec] | None = None,
        *,
        discovery: "ActionDiscovery | None" = None,
    ) -> None:
        self._specs = dict(specs or {})
        self._discovered: dict[str, ActionSpec] = {}
        self._discovery = discovery

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

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ActionRegistry":
        """Load action specs from a YAML action pack file.

        YAML structure::

            version: "1"
            platform: feishu
            actions:
              im.send_message:
                description: "..."
                effect: send
                risk_level: high
                output_policy: summary
                schema:
                  required: [chat_id, text]
                  properties:
                    chat_id: {type: string}
                backend:
                  kind: cli
                  argv: [im, +messages-send, --chat-id, "{chat_id}"]
                  timeout_s: 30

        Each ``backend.*`` field maps to ``ActionSpec.backend_config``.
        """
        import yaml  # pyyaml is a project dependency

        with Path(yaml_path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, Mapping):
            raise ValueError(f"YAML action pack must be a mapping: {yaml_path}")
        actions_data = data.get("actions") or {}
        if not isinstance(actions_data, Mapping):
            raise ValueError(f"YAML action pack 'actions' must be a mapping: {yaml_path}")

        specs: dict[str, ActionSpec] = {}
        for name, action_data in actions_data.items():
            if not isinstance(action_data, Mapping):
                continue
            backend = action_data.get("backend") or {}
            backend_config: dict[str, Any] = {}
            if "argv" in backend:
                backend_config["argv"] = list(str(a) for a in (backend["argv"] or []))
            if "dry_run_argv" in backend:
                backend_config["dry_run_argv"] = list(str(a) for a in (backend["dry_run_argv"] or []))
            if "approval_summary" in backend:
                backend_config["approval_summary"] = str(backend["approval_summary"])
            if "next_steps" in backend:
                backend_config["next_steps"] = [str(s) for s in (backend["next_steps"] or [])]
            if "timeout_s" in backend:
                backend_config["timeout_s"] = float(backend["timeout_s"])
            if "output_args" in backend:
                backend_config["output_args"] = list(str(a) for a in (backend["output_args"] or []))

            schema = action_data.get("schema") or {}
            specs[str(name)] = ActionSpec(
                name=str(name),
                backend_kind=str((backend.get("kind") or "cli")),
                description=str(action_data.get("description") or ""),
                effect=str(action_data.get("effect") or "execute"),
                schema=dict(schema) if isinstance(schema, Mapping) else {},
                backend_config=backend_config,
                risk_level=str(action_data.get("risk_level") or "medium"),
                output_policy=str(action_data.get("output_policy") or "summary"),
            )
        return cls(specs)

    @property
    def discovery(self) -> "ActionDiscovery | None":
        return self._discovery

    @discovery.setter
    def discovery(self, value: "ActionDiscovery | None") -> None:
        self._discovery = value

    def merge_discovered(self, specs: Sequence[ActionSpec]) -> int:
        """Merge externally discovered specs without overwriting static ones.

        Returns the number of newly added discovered specs.
        """
        added = 0
        for spec in specs:
            if spec.name not in self._specs and spec.name not in self._discovered:
                self._discovered[spec.name] = spec
                added += 1
        return added

    def all(self) -> Mapping[str, ActionSpec]:
        """Return all action specs (static + discovered)."""
        merged = dict(self._discovered)
        merged.update(self._specs)
        return merged

    def static_specs(self) -> Mapping[str, ActionSpec]:
        """Return only verified static specs."""
        return dict(self._specs)

    def discovered_specs(self) -> Mapping[str, ActionSpec]:
        """Return only dynamically discovered specs."""
        return dict(self._discovered)

    def get(self, action: str) -> ActionSpec | None:
        """Return an action spec — static first, then discovered."""
        return self._specs.get(action) or self._discovered.get(action)

    def is_discovered(self, action: str) -> bool:
        """Return True if the action came from dynamic discovery."""
        return action not in self._specs and action in self._discovered

    def validate(self, action: str, payload: Mapping[str, Any]) -> ValidationResult:
        """Validate payload using the registered action schema."""
        spec = self.get(action)
        if spec is None:
            return ValidationResult(ok=False, error=f"Unknown platform action: {action}")
        return validate_payload(spec, payload)

    async def refresh_discovery(self, *, groups: Sequence[str] = ()) -> int:
        """Run the attached discovery source and merge results.

        Returns the number of newly discovered actions, or 0 if no
        discovery source is configured.
        """
        if self._discovery is None:
            return 0
        try:
            specs = await self._discovery.discover_actions(groups=groups)
        except Exception as exc:
            logger.debug("discovery.refresh_failed error=%s", exc)
            return 0
        return self.merge_discovered(specs)


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
