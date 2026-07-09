from __future__ import annotations

from prompt_toolkit.styles import Style as PTStyle

from leapflow.cli.tui_app.app import LeapApp
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


def _style_for(theme):
    return PTStyle.from_dict({
        "input-area": f"bg:{theme.input_bg} {theme.input_text} bold",
        "input-area.disabled": f"bg:{theme.input_bg} {theme.input_disabled_text}",
        "prompt": theme.prompt_char,
        "prompt.working": theme.accent_dim,
        "prompt.recording": theme.recording,
        "prompt.paused": theme.prompt_paused,
        "prompt.executing": theme.executing,
        "status-bar": f"bg:{theme.toolbar_bg} {theme.toolbar_fg}",
        "status-bar.strong": f"bg:{theme.toolbar_bg} bold {theme.accent}",
        "status-bar.dim": f"bg:{theme.toolbar_bg} {theme.text_muted}",
        "status-bar.good": f"bg:{theme.toolbar_bg} {theme.success}",
        "status-bar.warn": f"bg:{theme.toolbar_bg} {theme.warning}",
        "status-bar.bad": f"bg:{theme.toolbar_bg} {theme.error}",
        "hint": theme.text_dim,
        "auto-suggest": theme.auto_suggest,
        "placeholder": theme.input_placeholder,
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


def test_light_background_resolves_readable_input_text() -> None:
    theme = resolve_theme(_DARK, terminal_bg="#F8FAFC")

    assert theme.input_text in {"#111827", "#1A1A1A", "#000000", "#374151"}
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0
    assert contrast_ratio(theme.input_placeholder, theme.input_bg) >= 4.5


def test_theme_and_background_overrides_are_deterministic() -> None:
    env = {"LEAPFLOW_TUI_THEME": "light", "LEAPFLOW_TUI_BG": "#102A2E"}
    theme = detect_theme(env, is_tty=True)

    assert theme.name == "light"
    assert theme.terminal_bg == "#102A2E"
    assert contrast_ratio(theme.input_text, theme.input_bg) >= 7.0


def test_colorfgbg_background_detection() -> None:
    assert detect_light_mode({"COLORFGBG": "0;15"}) is True
    assert detect_light_mode({"COLORFGBG": "15;0"}) is False


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
        data_dir=tmp_path,
        on_input=None,
    )

    assert app._build_style() is not None
    assert app._input_area.window.height.max == 4
    assert app._input_area.window.dont_extend_height() is True
