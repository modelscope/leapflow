"""ViewSpec: the declarative, validated UI contract for the dashboard (SDUI).

A ViewSpec is a JSON-serializable tree of components drawn from a fixed,
versioned catalog. Templates and the engine may author ViewSpecs, but they can
only reference catalog component types and a whitelisted action protocol -- never
arbitrary HTML/JS. Unknown component types degrade to a Markdown node so a view
never fails to render.

Node shape::

    {"type": "Card", "props": {...}, "children": [...], "action": {...}?}

Action shape (bidirectional protocol)::

    {"kind": "nav|rpc|intent|approval", "name": "...", "params": {...}}
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 1
FALLBACK_TYPE = "Markdown"
ACTION_KINDS = frozenset({"nav", "rpc", "intent", "approval"})

# Fixed, versioned component vocabulary grouped by purpose. The frontend renderer
# implements exactly these types; adding a scenario reuses them (or registers a
# ``Custom`` escape-hatch renderer), never new core code.
COMPONENT_CATALOG: dict[str, tuple[str, ...]] = {
    "layout": ("Page", "Section", "Grid", "Row", "Col", "Tabs", "Tab", "Card", "Drawer", "Toolbar"),
    "display": (
        "Stat", "Table", "List", "Timeline", "Board", "Markdown", "EntityGraph",
        "Gauge", "ProgressBar", "Badge",
    ),
    "chart": ("LineChart", "AreaChart", "BarChart", "CandlestickChart", "Sparkline", "Heatmap", "PieChart"),
    "evidence": ("LinkCard", "Quote", "CitationList"),
    "interactive": ("Button", "FilterBar", "Select", "DateRange", "Search", "Form", "Slider", "TagInput"),
    "agent": ("FindingCard", "InsightCard", "StoryPanel", "ApprovalPrompt", "SuggestionChips", "AskBox"),
    "escape": ("Custom", "Raw"),
}

COMPONENT_TYPES: frozenset[str] = frozenset(
    name for group in COMPONENT_CATALOG.values() for name in group
) | {FALLBACK_TYPE}


class ViewSpecError(ValueError):
    """Raised only by strict validation helpers, never by normalization."""


def make_markdown(text: str, **props: Any) -> dict[str, Any]:
    """Build a Markdown node (also the safe fallback for unknown types)."""
    return {"type": "Markdown", "props": {"text": str(text), **props}}


def _normalize_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, Mapping):
        return None
    kind = str(action.get("kind", ""))
    if kind not in ACTION_KINDS:
        return None
    result: dict[str, Any] = {
        "kind": kind,
        "name": str(action.get("name", "")),
        "params": dict(action.get("params") or {}),
    }
    if action.get("confirm"):
        result["confirm"] = True
    return result


def normalize_node(node: Any) -> dict[str, Any]:
    """Return a safe, catalog-valid node, degrading unknown types to Markdown."""
    if not isinstance(node, Mapping):
        return make_markdown(str(node))
    ntype = str(node.get("type", ""))
    if ntype not in COMPONENT_TYPES:
        return make_markdown(
            f"Unsupported component: {ntype or '(missing type)'}",
            _unsupported=ntype,
        )
    out: dict[str, Any] = {"type": ntype, "props": dict(node.get("props") or {})}
    action = _normalize_action(node.get("action"))
    if action is not None:
        out["action"] = action
    children = node.get("children")
    if isinstance(children, list):
        out["children"] = [normalize_node(child) for child in children]
    return out


def normalize_viewspec(spec: Any) -> dict[str, Any]:
    """Return a fully normalized, render-safe ViewSpec (never raises).

    Accepts ``root`` or legacy ``layout`` for the node list.
    """
    data = spec if isinstance(spec, Mapping) else {}
    root = data.get("root", data.get("layout", []))
    if not isinstance(root, list):
        root = [root]
    return {
        "schema_version": int(data.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
        "title": str(data.get("title", "")),
        "domain": str(data.get("domain", "")),
        "root": [normalize_node(node) for node in root],
        "meta": dict(data.get("meta") or {}),
    }


def validate_viewspec(spec: Any) -> list[str]:
    """Return a list of strict-validation error strings (empty when valid).

    Used by templates/tests to catch authoring mistakes; runtime rendering uses
    ``normalize_viewspec`` which degrades instead of failing.
    """
    errors: list[str] = []
    if not isinstance(spec, Mapping):
        return ["ViewSpec must be a mapping"]
    version = spec.get("schema_version", SCHEMA_VERSION)
    if int(version or 0) != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {version!r} (expected {SCHEMA_VERSION})")
    root = spec.get("root", spec.get("layout"))
    if not isinstance(root, list):
        errors.append("ViewSpec.root must be a list of components")
        root = []

    def _walk(node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            errors.append(f"{path}: node must be a mapping")
            return
        ntype = str(node.get("type", ""))
        if ntype not in COMPONENT_TYPES:
            errors.append(f"{path}: unknown component type {ntype or '(missing)'!r}")
        action = node.get("action")
        if action is not None:
            if not isinstance(action, Mapping) or str(action.get("kind")) not in ACTION_KINDS:
                errors.append(f"{path}: invalid action (kind must be one of {sorted(ACTION_KINDS)})")
        children = node.get("children")
        if children is not None:
            if not isinstance(children, list):
                errors.append(f"{path}: children must be a list")
            else:
                for i, child in enumerate(children):
                    _walk(child, f"{path}.children[{i}]")

    for i, node in enumerate(root):
        _walk(node, f"root[{i}]")
    return errors


__all__ = [
    "SCHEMA_VERSION",
    "FALLBACK_TYPE",
    "ACTION_KINDS",
    "COMPONENT_CATALOG",
    "COMPONENT_TYPES",
    "ViewSpecError",
    "make_markdown",
    "normalize_node",
    "normalize_viewspec",
    "validate_viewspec",
]
