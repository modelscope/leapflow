from __future__ import annotations

from os import terminal_size

from prompt_toolkit.styles import Style as PTStyle
from rich.rule import Rule

from leapflow.cli.banner import _BannerPalette, display_rich_banner
from leapflow.cli.tui_app.app import LeapApp
from leapflow.cli.tui_app.console import LeapConsole, _TerminalBackgroundCodeBlock, _TerminalBackgroundMarkdown
from leapflow.cli.tui_app.status import StatusBar
from leapflow.cli.tui_app.theme import (
    _DARK,
    _LIGHT,
    contrast_ratio,
    detect_light_mode,
    detect_theme,
    parse_hex_color,
    relative_luminance,
    resolve_theme,
)


def _last_hex(style: str) -> str:
    for part in reversed(style.split()):
        if part.startswith("#"):
            return part
    raise AssertionError(f"No hex color found in style: {style}")


def _style_for(theme):
    return PTStyle.from_dict({
        "input-area": f"bg:{theme.input_bg} {theme.input_text} bold",
        "input-area.disabled": f"bg:{theme.input_bg} {theme.input_disabled_text}",
        "prompt": theme.prompt_char,
        "prompt.working": theme.accent_dim,
        "prompt.recording": theme.recording,
        "prompt.paused": theme.prompt_paused,
        "prompt.executing": theme.executing,
        "status-bar": f"bg:{theme.toolbar_bg} {theme.statusbar_fg}",
        "status-bar.strong": f"bg:{theme.toolbar_bg} bold {theme.statusbar_accent}",
        "status-bar.dim": f"bg:{theme.toolbar_bg} {theme.statusbar_dim}",
        "status-bar.good": f"bg:{theme.toolbar_bg} {theme.statusbar_good}",
        "status-bar.warn": f"bg:{theme.toolbar_bg} {theme.warning}",
        "status-bar.bad": f"bg:{theme.toolbar_bg} {theme.error}",
        "hint": theme.text_dim,
        "auto-suggest": theme.auto_suggest,
        "placeholder": f"{theme.input_placeholder} nobold",
        "selection": f"bg:{theme.input_selection_bg} {theme.input_selection_fg}",
    })


def test_hex_parsing_and_contrast_math() -> None:
    assert parse_hex_color("#FFFFFF") == (255, 255, 255)
    assert relative_luminance("#000000") == 0.0
    assert contrast_ratio("#FFFFFF", "#000000") == 21.0


def test_dark_background_resolves_readable_input_text() -> None:
    theme = resolve_theme(_LIGHT, terminal_bg="#0B1F24")

    assert theme.input_text in {"#FFFFFF", "#F8FAFC", "#E5E7EB", "#D1D5DB"}
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0
    assert contrast_ratio(theme.auto_suggest, theme.input_bg) >= 4.5
    assert contrast_ratio(theme.input_placeholder, theme.input_bg) >= 3.0


def test_placeholder_is_visually_subordinate_on_dark_theme() -> None:
    theme = resolve_theme(_DARK, terminal_bg="#0B1F24")

    assert theme.input_placeholder == "#64748B"
    assert contrast_ratio(theme.input_placeholder, theme.input_bg) >= 3.0
    assert contrast_ratio(theme.input_placeholder, theme.input_bg) < contrast_ratio(theme.input_text, theme.input_bg)


def test_light_background_resolves_readable_input_text() -> None:
    theme = resolve_theme(_DARK, terminal_bg="#F8FAFC")

    assert theme.input_text in {"#111827", "#1A1A1A", "#000000", "#374151"}
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0
    assert contrast_ratio(theme.input_placeholder, theme.input_bg) >= 3.0
    assert contrast_ratio(theme.border, theme.input_bg) >= 4.5
    assert contrast_ratio(theme.input_border, theme.input_bg) >= 4.5
    assert contrast_ratio(theme.input_focus_border, theme.input_bg) >= 5.0
    assert contrast_ratio(theme.toolbar_fg, theme.toolbar_bg) >= 4.5
    assert contrast_ratio(_last_hex(theme.panel_title), theme.input_bg) >= 5.0


def test_light_theme_uses_terminal_background_for_tui_surfaces() -> None:
    theme = detect_theme(
        {"LEAPFLOW_TUI_THEME": "light", "LEAPFLOW_TUI_BG": "#FFFFFF"},
        is_tty=True,
    )

    assert theme.input_bg == "#FFFFFF"
    assert theme.toolbar_bg == "#FFFFFF"
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0
    assert contrast_ratio(theme.toolbar_fg, theme.toolbar_bg) >= 4.5


def test_theme_and_background_overrides_are_deterministic() -> None:
    env = {"LEAPFLOW_TUI_THEME": "light", "LEAPFLOW_TUI_BG": "#102A2E"}
    theme = detect_theme(env, is_tty=True)

    assert theme.name == "light"
    assert theme.terminal_bg == "#102A2E"
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0


def test_colorfgbg_background_detection() -> None:
    assert detect_light_mode({"COLORFGBG": "0;15"}) is True
    assert detect_light_mode({"COLORFGBG": "15;0"}) is False
    assert detect_light_mode({"ITERM_PROFILE": "Light"}) is True
    assert detect_light_mode({"ITERM_PROFILE": "Dark"}) is False


def test_terminal_program_alone_does_not_force_light_theme() -> None:
    theme = detect_theme({"TERM_PROGRAM": "Apple_Terminal"}, is_tty=True)

    assert theme.name == "dark"
    assert theme.input_bg == "#0B1F24"
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0


def test_prompt_toolkit_accepts_base_and_resolved_theme_styles() -> None:
    _style_for(_DARK)
    _style_for(_LIGHT)
    _style_for(resolve_theme(_DARK, terminal_bg="#102A2E"))
    _style_for(resolve_theme(_LIGHT, terminal_bg="#F8FAFC"))


def test_leap_app_style_builder_accepts_resolved_theme(tmp_path) -> None:
    theme = resolve_theme(_DARK, terminal_bg="#102A2E")
    app = LeapApp(
        console=None,
        theme=theme,
        status=lambda: [],
        history_path=tmp_path / "tui_history",
        on_input=None,
    )

    assert app._build_style() is not None
    # Input height is adaptive (a callable), floored at the original 4 rows and
    # capped to a fraction of the terminal; it stays content-sized.
    height = app._input_area.window.height
    resolved = height() if callable(height) else height
    assert resolved.min == 1
    assert resolved.max >= 4
    assert app._input_area.window.dont_extend_height() is True
    app._input_area.buffer.text = "x" * 200
    grown = height() if callable(height) else height
    assert grown.preferred >= resolved.preferred


def test_leap_app_layout_keeps_status_breathing_gap(tmp_path) -> None:
    theme = resolve_theme(_DARK, terminal_bg="#102A2E")
    app = LeapApp(
        console=None,
        theme=theme,
        status=lambda: [],
        history_path=tmp_path / "tui_history",
        on_input=None,
    )

    root = app._app.layout.container.content
    children = root.children

    assert len(children) == 6
    status_gap = children[2].content
    status_bar = children[3]
    input_hint = children[4].content
    input_area = children[5].content
    assert status_gap.style == "class:status-gap"
    assert status_gap.height == 1
    assert app._build_style().get_attrs_for_style_str("class:status-gap").bgcolor == ""
    assert status_bar.style == "class:status-bar"
    assert input_hint.style == "class:hint"
    assert input_hint.height == 1
    assert input_area is app._input_area.window


def test_console_system_supports_visual_spacing() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def print(self, *args, **kwargs) -> None:
            self.calls.append((args, kwargs))

    console = LeapConsole(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    capture = CaptureConsole()
    console._console = capture  # type: ignore[assignment]

    console.system("After reinstalling LeapFlow, use `leap daemon restart`.", margin_bottom=1)

    assert capture.calls == [
        (("  After reinstalling LeapFlow, use `leap daemon restart`.",), {"style": "leap.dim"}),
        ((), {}),
    ]


def test_rich_banner_accepts_resolved_theme(capsys) -> None:
    theme = resolve_theme(_LIGHT, terminal_bg="#FFFFFF")

    display_rich_banner(
        model="provider/qwen3-plus",
        cwd="/tmp/work",
        platform_online=False,
        tool_defs=[],
        skills=[],
        context_length=1_000_000,
        show_welcome=False,
        theme=theme,
    )

    output = capsys.readouterr().out
    assert "LeapFlow" in output
    assert "1M ctx" in output
    assert "#FFF8DC" not in output


def test_rich_banner_keeps_warm_brand_palette() -> None:
    theme = resolve_theme(_LIGHT, terminal_bg="#FFFFFF")
    palette = _BannerPalette(theme)

    assert theme.accent == "#334155"
    assert palette.accent == "#FFBF00"
    assert palette.accent_dim == "#B8860B"
    assert palette.border == "#CD7F32"
    assert palette.text == "#FFF8DC"


def test_status_bar_uses_warm_brand_palette() -> None:
    dark = resolve_theme(_DARK, terminal_bg="#0B1F24")
    light = resolve_theme(_LIGHT, terminal_bg="#FFFFFF")

    assert dark.statusbar_fg == "#CD7F32"
    assert dark.statusbar_accent == "#FFBF00"
    assert dark.statusbar_dim == "#B8860B"
    assert dark.statusbar_good == "#FFBF00"
    assert light.statusbar_fg == "#8B5E34"
    assert light.statusbar_dim == "#A16207"
    assert light.statusbar_accent in {"#8B5E34", "#B8860B"}
    assert light.statusbar_good in {"#8B5E34", "#B8860B"}
    assert contrast_ratio(dark.statusbar_fg, dark.toolbar_bg) >= 4.5
    assert contrast_ratio(light.statusbar_fg, light.toolbar_bg) >= 4.5


def test_markdown_code_blocks_use_terminal_background() -> None:
    from rich.console import Console

    assert _TerminalBackgroundMarkdown.elements["fence"] is _TerminalBackgroundCodeBlock
    block = _TerminalBackgroundCodeBlock("text", "monokai")
    block.text = "config -> engine"

    syntax = next(block.__rich_console__(Console(), Console().options))

    assert syntax.background_color == "default"
    assert syntax.word_wrap is False


def test_markdown_headings_use_professional_palette() -> None:
    console = LeapConsole(resolve_theme(_DARK, terminal_bg="#0B1F24")).raw

    h2 = console.get_style("markdown.h2")
    h1 = console.get_style("markdown.h1")

    assert h2.color is not None
    assert h1.color is not None
    assert str(h2.color).lower() != "magenta"
    assert h2.color.triplet is not None
    assert h2.color.triplet.hex.lower() != "ff00ff"
    assert h1.bgcolor is None
    assert h2.bgcolor is None


def test_dark_theme_uses_low_saturation_terminal_palette() -> None:
    theme = resolve_theme(_DARK, terminal_bg="#0B1F24")

    assert theme.accent == "#E6DDC4"
    assert theme.text == "#F1EAD6"
    assert theme.text_muted == "#9AA6A3"
    assert theme.border == "#D8D1BB"
    assert "#FF00FF" not in {theme.accent, theme.text, theme.text_dim, theme.border}


def test_inline_markdown_code_uses_terminal_background_style() -> None:
    console = LeapConsole(resolve_theme(_LIGHT, terminal_bg="#FFFFFF")).raw

    style = console.get_style("markdown.code")

    assert style.bgcolor is None
    assert style.color is not None
    assert style.bold is True


def test_answer_label_uses_warm_left_aligned_boundary(monkeypatch) -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.rendered = []

        def print(self, renderable) -> None:
            self.rendered.append(renderable)

    capture = CaptureConsole()
    theme = resolve_theme(_DARK, terminal_bg="#0B1F24")
    leap_console = LeapConsole(theme)
    rich_console = leap_console.raw
    monkeypatch.setattr(leap_console, "_console", capture)

    leap_console.answer_label()

    rule = capture.rendered[0]
    answer_border = rich_console.get_style("leap.answer_border")
    answer_title = rich_console.get_style("leap.answer_title")
    assert isinstance(rule, Rule)
    assert rule.align == "left"
    assert rule.style == "leap.answer_border"
    assert str(rule.title) == " LeapFlow "
    assert rule.title.style == "leap.answer_title"
    assert answer_border.color is not None
    assert answer_title.color is not None
    assert answer_border.color.triplet.hex == theme.statusbar_dim.lower()
    assert answer_title.color.triplet.hex == theme.statusbar_accent.lower()


def test_status_bar_compacts_on_narrow_terminal(monkeypatch) -> None:
    monkeypatch.setattr(
        "leapflow.cli.tui_app.status.shutil.get_terminal_size",
        lambda: terminal_size((48, 24)),
    )
    status = StatusBar(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    status.update(
        model_name="very-long-model-name-that-would-overflow",
        context_used=50_000,
        context_max=100_000,
    )

    rendered = "".join(text for _, text in status())
    assert "very-long-model" not in rendered
    assert "50%" in rendered
    assert "[" not in rendered


def test_status_bar_shows_compact_m_for_default_context_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        "leapflow.cli.tui_app.status.shutil.get_terminal_size",
        lambda: terminal_size((120, 24)),
    )
    status = StatusBar(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    status.update(
        model_name="qwen3.7-plus",
        context_used=240,
        context_max=1_000_000,
    )

    rendered = "".join(text for _, text in status())
    assert "0.2K/1M" in rendered
    assert "<0.1%" in rendered
    assert "[█░░░░░░░░░]" in rendered


def test_status_bar_shows_adaptive_context_state(monkeypatch) -> None:
    monkeypatch.setattr(
        "leapflow.cli.tui_app.status.shutil.get_terminal_size",
        lambda: terminal_size((120, 24)),
    )
    status = StatusBar(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    status.update(
        model_name="qwen3.7-plus",
        context_used=80_000,
        context_max=100_000,
        context_state="research",
    )

    rendered = "".join(text for _, text in status())
    assert "80%" in rendered
    assert "research" in rendered