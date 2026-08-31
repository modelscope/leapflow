"""Hermetic tests for the dashboard SDUI core: ViewSpec, templates, intent."""

from __future__ import annotations

from leapflow.dashboard import (
    DashboardIntent,
    TemplateLibrary,
    normalize_viewspec,
    render_template,
    validate_viewspec,
)
from leapflow.dashboard.templates import bind_value, resolve_path


# ── ViewSpec catalog + validation + fallback ───────────────────────────────


def test_normalize_degrades_unknown_component_to_markdown() -> None:
    spec = normalize_viewspec({
        "title": "T",
        "root": [
            {"type": "Card", "children": [{"type": "Nonexistent", "props": {"x": 1}}]},
        ],
    })
    assert spec["schema_version"] == 1
    card = spec["root"][0]
    assert card["type"] == "Card"
    child = card["children"][0]
    assert child["type"] == "Markdown"
    assert "Unsupported component" in child["props"]["text"]


def test_normalize_keeps_valid_action_and_drops_invalid() -> None:
    spec = normalize_viewspec({
        "root": [
            {"type": "Button", "action": {"kind": "rpc", "name": "watch.pause", "params": {"id": "x"}}},
            {"type": "Button", "action": {"kind": "evil", "name": "hack"}},
        ],
    })
    assert spec["root"][0]["action"]["kind"] == "rpc"
    assert "action" not in spec["root"][1]


def test_validate_reports_unknown_type_and_bad_action() -> None:
    errors = validate_viewspec({
        "schema_version": 1,
        "root": [
            {"type": "Bogus"},
            {"type": "Button", "action": {"kind": "nope"}},
        ],
    })
    assert any("unknown component type" in e for e in errors)
    assert any("invalid action" in e for e in errors)


def test_validate_accepts_clean_spec() -> None:
    assert validate_viewspec({
        "schema_version": 1,
        "root": [{"type": "Card", "children": [{"type": "Markdown", "props": {"text": "hi"}}]}],
    }) == []


# ── Template binding ────────────────────────────────────────────────────────


def test_resolve_path_supports_dots_and_indices() -> None:
    data = {"a": {"b": [{"c": 7}]}}
    assert resolve_path(data, "a.b[0].c") == 7
    assert resolve_path(data, "a.missing") is None
    assert resolve_path(data, "a.b[5]") is None


def test_bind_value_full_and_interpolated() -> None:
    data = {"finding": {"title": "Spike", "score": 0.9}}
    assert bind_value("{{ finding.score }}", data) == 0.9  # full match preserves type
    assert bind_value("T: {{ finding.title }}", data) == "T: Spike"  # interpolation -> str


def test_bind_value_multiple_placeholders_interpolate() -> None:
    # Two placeholders + a literal must interpolate to a string, not be misread
    # as a single bogus dotted path (which previously resolved to None).
    data = {"observation": {"artifacts_included": 2, "artifact_count": 3}}
    assert (
        bind_value(
            "{{ observation.artifacts_included }}/{{ observation.artifact_count }}", data
        )
        == "2/3"
    )


def test_render_template_repeat_and_bind() -> None:
    template = {
        "template": "demo",
        "title": "Watch {{ name }}",
        "layout": [
            {
                "type": "Board",
                "children": [
                    {
                        "type": "FindingCard",
                        "repeat": "findings",
                        "as": "f",
                        "props": {"title": "{{ f.title }}", "bind": "f"},
                    }
                ],
            }
        ],
    }
    data = {"name": "AAPL", "findings": [{"title": "a"}, {"title": "b"}]}
    spec = render_template(template, data)
    assert spec["title"] == "Watch AAPL"
    cards = spec["root"][0]["children"]
    assert [c["props"]["title"] for c in cards] == ["a", "b"]
    assert cards[0]["props"]["data"] == {"title": "a"}  # bind -> data


def test_render_template_repeat_missing_list_yields_no_children() -> None:
    template = {"layout": [{"type": "Board", "children": [
        {"type": "FindingCard", "repeat": "nope", "props": {}}]}]}
    spec = render_template(template, {})
    assert spec["root"][0]["children"] == []


def test_template_library_generic_renders_session_analysis() -> None:
    lib = TemplateLibrary()
    assert "generic" in lib.names()
    spec = lib.render("generic", {"title": "Session Analysis", "analysis": {
        "story": "arc",
        "insights": [{"title": "i", "summary": "s", "severity": "notable"}],
        "action_items": [], "decisions": [], "open_questions": [],
        "entities": ["Alice"], "next_prompts": ["ask"],
    }, "observation": {"refresh_reason": "manual_refresh", "context_scope": "text_only"}, "artifact_context": []})
    assert spec["title"] == "Session Analysis"
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for n in nodes:
            flat.append(n)
            _walk(n.get("children") or [])

    _walk(spec["root"])
    types = {n["type"] for n in flat}
    assert "StoryPanel" in types
    assert len([n for n in flat if n["type"] == "InsightCard"]) == 1


def test_template_library_unknown_falls_back_to_generic() -> None:
    lib = TemplateLibrary()
    spec = lib.render("does-not-exist", {"title": "F", "findings": []})
    assert spec["meta"]["template"] == "generic"


def test_template_library_hides_retired_builtin_nav_even_with_override(tmp_path) -> None:
    for name in ("finance", "research", "sentiment"):
        (tmp_path / f"{name}.yaml").write_text(
            f"template: {name}\ntitle: '{name}'\nlayout:\n  - type: Page\n    props:\n      title: '{name}'\n",
            encoding="utf-8",
        )
    (tmp_path / "crypto.yaml").write_text(
        "template: crypto\ntitle: 'Crypto'\nlayout:\n  - type: Page\n    props:\n      title: 'Crypto'\n",
        encoding="utf-8",
    )

    lib = TemplateLibrary(override_dir=tmp_path)

    assert "crypto" in lib.visible_names()
    assert "finance" not in lib.visible_names()
    assert "research" not in lib.visible_names()
    assert "sentiment" not in lib.visible_names()
    assert {"finance", "research", "sentiment"}.issubset(set(lib.hidden_names()))


# ── DashboardIntent (dual entry) ────────────────────────────────────────────


def test_intent_from_args_first_token_is_template() -> None:
    assert DashboardIntent.from_args("finance").template == "finance"
    assert DashboardIntent.from_args("research extra tokens").template == "research"
    assert DashboardIntent.from_args("").template == ""


def test_intent_from_params_reads_template() -> None:
    assert DashboardIntent.from_params({"template": "research"}).template == "research"
    assert DashboardIntent.from_params({}).to_dict() == {"template": ""}


# ════════════════════════════════════════════════════════════════
# Template bindings the engine will silently ignore
# ════════════════════════════════════════════════════════════════


def test_no_template_binds_in_a_shape_the_engine_drops() -> None:
    """The two mistakes here produce a panel with headings and no rows.

    ``render_node`` expands ``repeat`` only when it is a *string*, and it reads
    ``bind`` only from inside ``props``. Neither mistake raises: the template loads,
    the panel renders, the columns appear, and every row is missing. Four tables on
    the capability board shipped that way -- ``repeat`` as a mapping and ``bind`` at
    node level -- and nothing failed anywhere.
    """
    from leapflow.dashboard.templates import TemplateLibrary

    library = TemplateLibrary()
    problems: list[str] = []

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            repeat = node.get("repeat")
            if repeat is not None and not isinstance(repeat, str):
                problems.append(f"{where}: repeat must be a path string, got {type(repeat).__name__}")
            # Only a component node is checked for a stray ``bind``. Inside ``props``
            # it is the correct spelling, and ``type`` is what tells the two apart.
            if "type" in node and "bind" in node:
                problems.append(f"{where}: bind must live inside props, not on the node")
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")

    for name in library.names():
        walk(library.load(name), name)
    assert not problems, "bindings the engine ignores:\n  " + "\n  ".join(problems)


def test_every_table_column_declares_the_key_it_reads() -> None:
    """A bare column label renders a header over an empty cell.

    The Table renderer reads ``props.columns[].key`` against each row mapping, so a
    column given as a plain string has no key to read and shows nothing under it.
    """
    from leapflow.dashboard.templates import TemplateLibrary

    library = TemplateLibrary()
    problems: list[str] = []

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            props = node.get("props")
            if node.get("type") == "Table" and isinstance(props, dict):
                columns = props.get("columns")
                if isinstance(columns, list):
                    for index, column in enumerate(columns):
                        if not isinstance(column, dict) or not column.get("key"):
                            problems.append(f"{where} column[{index}] declares no key: {column!r}")
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")

    for name in library.names():
        walk(library.load(name), name)
    assert not problems, "table columns with no key:\n  " + "\n  ".join(problems)
