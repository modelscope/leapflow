"""YAML template rendering into a validated ViewSpec (the SDUI authoring layer).

Templates are authored in YAML per scenario and compiled at runtime into a
ViewSpec by binding live data into a component tree. Binding is intentionally
minimal and safe -- whitelisted dotted paths and ``{{ path }}`` interpolation,
never arbitrary evaluation. A ``repeat`` directive expands one node per item in a
bound list (e.g. one FindingCard per finding).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

from leapflow.dashboard.viewspec import SCHEMA_VERSION, normalize_viewspec

_INDEX_RE = re.compile(r"^(.*?)\[(\d+)\]$")
_FULL_RE = re.compile(r"^\{\{\s*(.+?)\s*\}\}$")
_PART_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")

# Inline fallback so rendering works even if no template files are present.
_DEFAULT_GENERIC: dict[str, Any] = {
    "template": "generic",
    "title": "{{ title }}",
    "layout": [
        {
            "type": "Board",
            "props": {"title": "Findings"},
            "children": [
                {
                    "type": "FindingCard",
                    "repeat": "findings",
                    "as": "f",
                    "props": {
                        "title": "{{ f.title }}",
                        "summary": "{{ f.summary }}",
                        "severity": "{{ f.severity }}",
                        "bind": "f",
                    },
                }
            ],
        }
    ],
}


def resolve_path(data: Any, path: str) -> Any:
    """Resolve a dotted path with optional ``[i]`` indices; None when missing."""
    current = data
    for raw_part in str(path).split("."):
        if not raw_part:
            continue
        part, index = raw_part, None
        match = _INDEX_RE.match(raw_part)
        if match:
            part, index = match.group(1), int(match.group(2))
        if part:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        if index is not None:
            if isinstance(current, (list, tuple)) and 0 <= index < len(current):
                current = current[index]
            else:
                return None
    return current


def _coerce_str(value: Any) -> str:
    return "" if value is None else str(value)


def bind_value(value: Any, data: Mapping[str, Any]) -> Any:
    """Bind ``{{ path }}`` templates inside strings/dicts/lists against ``data``."""
    if isinstance(value, str):
        full = _FULL_RE.match(value.strip())
        if full:
            return resolve_path(data, full.group(1))
        return _PART_RE.sub(lambda m: _coerce_str(resolve_path(data, m.group(1))), value)
    if isinstance(value, Mapping):
        return {key: bind_value(val, data) for key, val in value.items()}
    if isinstance(value, list):
        return [bind_value(item, data) for item in value]
    return value


def _bind_props(props: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in props.items():
        if key == "bind" and isinstance(value, str):
            out["data"] = resolve_path(data, value)
        else:
            out[key] = bind_value(value, data)
    return out


def render_node(node: Any, data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render one template node into zero or more concrete nodes."""
    if not isinstance(node, Mapping):
        return [{"type": "Markdown", "props": {"text": str(node)}}]

    repeat = node.get("repeat")
    if isinstance(repeat, str) and repeat:
        items = resolve_path(data, repeat)
        if not isinstance(items, (list, tuple)):
            return []
        as_name = str(node.get("as") or "item")
        base = {key: value for key, value in node.items() if key not in ("repeat", "as")}
        expanded: list[dict[str, Any]] = []
        for item in items:
            scope = dict(data)
            scope[as_name] = item
            expanded.extend(render_node(base, scope))
        return expanded

    rendered: dict[str, Any] = {"type": str(node.get("type", ""))}
    rendered["props"] = _bind_props(dict(node.get("props") or {}), data)
    action = node.get("action")
    if isinstance(action, Mapping):
        rendered["action"] = bind_value(dict(action), data)
    children = node.get("children")
    if isinstance(children, list):
        kids: list[dict[str, Any]] = []
        for child in children:
            kids.extend(render_node(child, data))
        rendered["children"] = kids
    return [rendered]


def render_template(template: Any, data: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Compile a template dict + data into a normalized, render-safe ViewSpec."""
    template = template if isinstance(template, Mapping) else {}
    data = data or {}
    layout = template.get("layout", template.get("root", []))
    if not isinstance(layout, list):
        layout = [layout]
    root: list[dict[str, Any]] = []
    for node in layout:
        root.extend(render_node(node, data))
    return normalize_viewspec({
        "schema_version": template.get("schema_version", SCHEMA_VERSION),
        "title": bind_value(template.get("title", ""), data),
        "domain": template.get("domain", ""),
        "root": root,
        "meta": {
            "template": template.get("template", ""),
            "refresh": dict(template.get("refresh") or {}),
        },
    })


class TemplateLibrary:
    """Loads YAML templates from a builtin dir plus an optional override dir.

    Profile-level override templates take precedence over builtin ones.
    """

    def __init__(
        self,
        builtin_dir: Optional[Path] = None,
        override_dir: Optional[Path] = None,
    ) -> None:
        self._builtin = builtin_dir or (Path(__file__).parent / "templates")
        self._override = override_dir

    def _dirs(self) -> list[Path]:
        return [d for d in (self._override, self._builtin) if d is not None]

    def names(self) -> list[str]:
        """Return available template names (override + builtin)."""
        found: set[str] = set()
        for directory in self._dirs():
            if directory.exists():
                for path in directory.glob("*.yaml"):
                    found.add(path.stem)
        return sorted(found)

    def load(self, name: str) -> Optional[dict[str, Any]]:
        """Load a raw template dict by name, or None when not found."""
        import yaml

        for directory in self._dirs():
            path = directory / f"{name}.yaml"
            if path.exists():
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                return loaded if isinstance(loaded, dict) else None
        return None

    def render(self, name: str, data: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Render a template by name, falling back to a builtin generic view."""
        template = self.load(name) or self.load("generic") or _DEFAULT_GENERIC
        return render_template(template, data or {})


__all__ = [
    "resolve_path",
    "bind_value",
    "render_node",
    "render_template",
    "TemplateLibrary",
]
