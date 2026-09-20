# Copyright (c) Alibaba, Inc. and its affiliates.
"""Static regressions for the frontend defects measured on a live board.

There is no JS test runner in this repository, so these guard the shipped source
directly — the same approach ``test_dashboard_i18n_static.py`` takes. Each test names
the symptom a reader saw, because that is what must never come back.
"""
from __future__ import annotations

import re
from pathlib import Path

_STATIC = Path(__file__).parents[1] / "src" / "leapflow" / "dashboard" / "static"
_APP_JS = _STATIC / "app.js"
_STYLES = _STATIC / "styles.css"
_INDEX = _STATIC / "index.html"


def _app() -> str:
    return _APP_JS.read_text(encoding="utf-8")


# ════════════════════════════════════════════════════════════════
# Markdown: nine notices rendered as raw markup
# ════════════════════════════════════════════════════════════════


def test_markdown_node_is_parsed_not_printed_verbatim() -> None:
    """The defect: the component named Markdown did not parse Markdown.

    It escaped its text and inserted it as-is, so every explanatory aside on the
    evolution board showed a literal ``>`` and literal ``**`` — nine of eleven
    Markdown notices in the shipped templates, all of them on the one board whose
    job is explaining how the framework evolves.
    """
    src = _app()
    assert "Markdown: (n) => renderMarkdown(" in src
    assert 'Markdown: (n) => el("div", "md prose", esc(' not in src, (
        "the renderer must not go back to printing escaped text verbatim"
    )


def test_markdown_escapes_before_it_transforms() -> None:
    """Order is the whole security argument: escape, then add only known tags.

    ``esc`` must be applied to the text before any transform runs, so prose can never
    introduce an element and the only tags in the output are the ones added here.
    """
    src = _app()
    assert "mdInline(esc(" in src, "transforms must run on already-escaped text"
    assert "esc(mdInline(" not in src, (
        "escaping after transforming would neutralise the tags this renderer adds"
    )


def test_markdown_recognises_exactly_the_three_authored_constructs() -> None:
    """Blockquote, bold, inline code — and no general parser was pulled in.

    The SDUI contract is a closed component catalog; widening the trusted surface to
    a Markdown library to gain three constructs would be a bad trade.
    """
    src = _app()
    assert '<blockquote' in src or 'el("blockquote"' in src
    assert "<strong>$1</strong>" in src
    assert "<code>" in src
    for library in ("marked", "showdown", "markdown-it", "remark"):
        assert library not in src, f"no third-party markdown parser ({library})"


def test_parsed_prose_releases_pre_wrap() -> None:
    """The blocks the renderer emits must not inherit whitespace preservation.

    ``.md`` sets ``pre-wrap`` for raw text elsewhere; leaving it on would reintroduce
    the folded-scalar indentation the YAML carried as visible leading gaps.
    """
    css = _STYLES.read_text(encoding="utf-8")
    assert ".md.prose { white-space: normal; }" in css
    assert ".md.prose code" in css, "inline code needs a monospace treatment"


# ════════════════════════════════════════════════════════════════
# Live lens: five of six metrics froze while looking current
# ════════════════════════════════════════════════════════════════


def test_live_metrics_declare_their_provenance() -> None:
    """The defect: snapshot-derived figures sat on load-time values silently.

    A presentation event carries one fact, never a recount, so those metrics cannot
    move between producer cycles. They are now labelled rather than redrawn as though
    they had, which is the distinction the producer's own fingerprint docstring
    warns about.
    """
    src = _app()
    assert "function evolutionLiveStat(label, value, provenance)" in src
    assert '"stat prov-" + provenance' in src
    assert '"Live since snapshot"' in src, (
        "the count of uncounted increments is the figure that makes the strip honest"
    )
    for label, provenance in (
        ("Event count", "live"),
        ("Live since snapshot", "live"),
        ("Episodes", "snapshot"),
        ("Mutation", "snapshot"),
        ("Regressions", "snapshot"),
        ("Unadmitted intents", "snapshot"),
    ):
        assert re.search(
            rf'\["{re.escape(label)}",[^\]]*"{provenance}"\]', src
        ), f"{label!r} must be declared as {provenance}-derived"


def test_live_lens_counts_increments_against_the_snapshot_instant() -> None:
    """Only events strictly newer than the snapshot are uncounted by it."""
    src = _app()
    assert "const baselineAt = Number(snapshot.observed_at || 0);" in src
    assert "Number(event.ts || 0) > baselineAt" in src


def test_drift_is_stated_only_when_the_two_provenances_diverge() -> None:
    """A permanent caveat is noise; a count that appears when true is a fact."""
    src = _app()
    assert "if (since > 0)" in src
    assert "Snapshot metrics refresh on the next monitor cycle." in src


def test_empty_lanes_say_what_would_appear_and_why_nothing_has() -> None:
    """"No live events" alone left quiet indistinguishable from broken."""
    src = _app()
    assert '"lane-hint"' in src
    for hint in (
        "Environment events appear when a probe records a change in the surroundings.",
        "Decision events appear when a recorded observation drives a capability choice.",
        "Governance events appear when trust, quarantine or reclamation moves.",
    ):
        assert hint in src, f"missing lane guidance: {hint!r}"


def test_lanes_are_equal_height_rather_than_a_fixed_short_box() -> None:
    """Three lanes with different fill read as a ragged row when height is fixed."""
    css = _STYLES.read_text(encoding="utf-8")
    assert "align-items: stretch" in css
    assert "min-height: 148px" not in css, "the fixed lane height is what made it ragged"
    assert ".evolution-live-lane .lane-hint" in css


# ════════════════════════════════════════════════════════════════
# Sparkline: a one-point series reported "No entries."
# ════════════════════════════════════════════════════════════════


def test_a_single_sample_series_is_drawn_not_discarded() -> None:
    """The defect: ``length >= 2`` filtered out the only reading a new board has.

    The chart then printed "No entries.", telling the reader there was no data when
    there was exactly one observation.
    """
    src = _app()
    assert "g.points.length >= 1" in src
    assert "g.points.length >= 2" not in src, (
        "a one-point series must not be filtered back out"
    )
    assert "g.points.length === 1" in src, "a lone sample needs its own marker branch"
    assert 'svgEl("circle")' in src


# ════════════════════════════════════════════════════════════════
# Provenance bar: page-level freshness on every lens
# ════════════════════════════════════════════════════════════════


def test_the_provenance_bar_renders_on_every_view_fetch() -> None:
    """Freshness is page chrome, not a panel a template author must remember."""
    src = _app()
    assert "renderProvenance(payload.meta || {})" in src
    assert "function renderProvenance(meta)" in src


def test_the_bar_offers_a_refresh_so_the_loop_closes() -> None:
    """Several boards prescribe a fix; without this the reader cannot confirm it landed."""
    src = _app()
    assert '"Refresh now"' in src
    assert 'name: "watch.refresh"' in src


def test_an_unobserved_board_is_not_aged_from_the_clock() -> None:
    """Saying "0s ago" where nothing was observed invents the fact the bar exists for."""
    src = _app()
    assert 't("Not yet observed")' in src


def test_static_assets_are_cache_busted_together() -> None:
    """A stale cached app.js against a new stylesheet is its own class of bug report."""
    html = _INDEX.read_text(encoding="utf-8")
    versions = re.findall(r"/static/(?:app\.js|styles\.css)\?v=([\w-]+)", html)
    assert len(versions) == 2, "both assets must carry a cache-busting version"
    assert versions[0] == versions[1], "asset versions must be bumped together"
    assert versions[0] != "status-i18n-20260808", (
        "the version predates the markdown and provenance changes"
    )


# ════════════════════════════════════════════════════════════════
# Bar charts: long labels overlapped their tracks in narrow cells
# ════════════════════════════════════════════════════════════════


def test_bar_chart_labels_wrap_without_consuming_the_track() -> None:
    """Long translated labels must never be allowed to overlap a value or bar.

    ``max-content`` paired with ``white-space: nowrap`` made the maturity label
    ``new_unproven`` paint over its track. The base grid reserves a bounded label
    column, and a narrow chart promotes the track to a dedicated second row.
    """
    css = _STYLES.read_text(encoding="utf-8")
    assert ".chart { min-height: 120px; min-width: 0; container-type: inline-size; }" in css
    assert "grid-template-columns: fit-content(13rem) minmax(4rem, 1fr) max-content;" in css
    assert "column-gap: 0.65rem;" in css
    assert ".bar-label { min-width: 0; white-space: normal; overflow-wrap: anywhere; }" in css
    assert ".bar-track { min-width: 0; width: 100%;" in css
    assert "@container (max-width: 18rem)" in css
    assert '"label value"\n      "track track"' in css


def test_bar_chart_labels_preserve_full_text_for_accessibility() -> None:
    """Wrapping is visible; the title retains the complete translated label."""
    src = _app()
    assert 'const label = el("span", "bar-label", esc(tx(row.label)));' in src
    assert "label.title = tx(row.label);" in src
