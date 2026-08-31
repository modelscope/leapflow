"""Static regression guards for LeapBoard frontend i18n coverage.

There is no JS test runner in this repository yet, so these tests protect the
high-value failure mode directly in the source: dynamic LeapBoard strings must
use t()/tx()/fmt() instead of hardcoded English in render paths.
"""
from __future__ import annotations

import re

from pathlib import Path

_APP_JS = Path(__file__).parents[1] / "src" / "leapflow" / "dashboard" / "static" / "app.js"


def _source() -> str:
    return _APP_JS.read_text(encoding="utf-8")


def test_signal_timeline_dynamic_text_uses_i18n_helpers() -> None:
    src = _source()

    assert 'fmt("seconds ago"' in src
    assert 'fmt("minutes ago"' in src
    assert 'fmt("hours ago"' in src
    assert 't("All")' in src
    assert 'fmt("Showing {shown} of {total} recent events."' in src
    assert 'fmt("Showing {shown} of {total} {family} events."' in src
    assert 'badge.textContent = "\\u26a0 " + t("stale build")' in src
    assert 'badge.textContent = "\\u26a0 stale build"' not in src
    assert '+ "s ago"' not in src
    assert 'footer.textContent = "Showing " +' not in src


def test_tables_and_chart_labels_route_through_translation() -> None:
    src = _source()

    assert 'esc(tx(row.label))' in src
    assert 'esc(tx(v))' in src


def test_connection_status_updates_data_i18n_and_text_together() -> None:
    src = _source()

    assert "function setConnectionStatus(key)" in src
    assert "statusEl.dataset.i18n = key" in src
    assert "statusEl.textContent = t(key)" in src
    assert 'ws.onopen = () => { setConnectionStatus("live"); };' in src
    assert 'ws.onclose = () => { setConnectionStatus("reconnecting…");' in src
    assert 'statusEl.textContent = t("live")' not in src
    assert 'statusEl.textContent = t("reconnecting…")' not in src


def test_i18n_patch_covers_supported_locales_and_signal_keys() -> None:
    src = _source()

    for lang in ("en", "zh", "fr", "es", "ar", "ru"):
        assert f"    {lang}: {{" in src
    for key in (
        "Noise suppressed",
        "Stream events",
        "Active watches",
        "Watch portfolio",
        "Signal health summary",
        "Trigger coverage",
        "stale_build_title",
        "signal.family.clipboard",
        # The physical bench board. Its family key is derived from the ``hw.`` event
        # prefix rather than enumerated anywhere, so a missing translation here is
        # the only way the omission ever shows.
        "signal.family.hw",
        "Physical bench",
        "Envelope conformance",
        "Sampling health",
        "Learned command outcomes",
        "connected",
        "正在连接",
        "已连接",
        "正在重连",
    ):
        assert key in src


def test_every_locale_translates_the_hardware_family() -> None:
    """Six-language coverage is a product requirement, not a best effort.

    Checked per locale rather than once across the file, because a single occurrence
    satisfies a substring search while leaving five languages falling back to the
    English key.
    """
    src = _source()
    assert src.count('"signal.family.hw"') == 6


# ════════════════════════════════════════════════════════════════
# Every template literal must be translatable in every locale
# ════════════════════════════════════════════════════════════════


def _translation_tables() -> dict[str, set[str]]:
    """Union the keys of every translation table the page merges, per locale.

    Discovered from the source rather than named. Naming them is how this check went
    stale before: the two original tables were asserted while a third carried most of
    the strings, so a green test coexisted with five untranslated boards.
    """
    source = _source()
    tables: dict[str, set[str]] = {}
    for name in re.findall(r"const (I18N\w*) = \{", source):
        block = re.search(rf"const {name} = \{{\n(.*?)\n  \}};", source, re.S)
        if block is None:
            continue
        for locale, body in re.findall(
            r"^    (\w+):\s*\{(.*?)\}(?:,)?$", block.group(1), re.M | re.S
        ):
            tables.setdefault(locale, set()).update(
                re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', body)
            )
    return tables


def _template_literals(node: object, found: set[str]) -> None:
    """Collect every literal a renderer will display, skipping bound expressions."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("title", "subtitle", "label", "caption"):
                if isinstance(value, str) and "{{" not in value:
                    found.add(value)
            if key == "columns" and isinstance(value, list):
                for column in value:
                    if isinstance(column, dict) and isinstance(column.get("label"), str):
                        found.add(column["label"])
            _template_literals(value, found)
    elif isinstance(node, list):
        for value in node:
            _template_literals(value, found)


def test_every_board_template_literal_is_translated_in_every_locale() -> None:
    """The gap the old i18n test could not see.

    ``test_i18n_patch_covers_supported_locales_and_signal_keys`` checks signal keys, so
    it stayed green while five of seven boards rendered English in every language:
    capability at 0 of 31 strings, hardware at 14 of 44, finance at 3 of 18. A lens
    added after the translation tables were written simply was not translated, and
    nothing anywhere said so.

    Asserted per template *and* per locale, because a single total would let one
    language lag behind the rest unnoticed.
    """
    import yaml

    tables = _translation_tables()
    locales = sorted(locale for locale in tables if locale != "en")
    assert locales, "no non-English locales found, so this check would pass vacuously"

    template_dir = _APP_JS.parent.parent / "templates"
    templates = sorted(template_dir.glob("*.yaml"))
    assert templates, "no templates found, so this check would pass vacuously"

    problems: list[str] = []
    for path in templates:
        found: set[str] = set()
        _template_literals(yaml.safe_load(path.read_text(encoding="utf-8")), found)
        for locale in locales:
            missing = sorted(found - tables[locale])
            if missing:
                problems.append(
                    f"{path.stem}/{locale}: {len(missing)} untranslated, e.g. {missing[:3]}"
                )
    assert not problems, "board literals with no translation:\n  " + "\n  ".join(problems)
