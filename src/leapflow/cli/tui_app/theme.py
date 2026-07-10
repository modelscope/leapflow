"""Adaptive theming with contrast-aware input colors.

Detects terminal background color via conservative environment heuristics and
resolves a coherent palette for the entire TUI.  Components consume theme
tokens only; no widget should hardcode ANSI sequences or raw colors.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Mapping

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_INPUT_TEXT_MIN_CONTRAST = 7.0
_SECONDARY_TEXT_MIN_CONTRAST = 4.5
_PROMPT_MIN_CONTRAST = 5.0

_DARK_TERMINAL_BG = "#0B1F24"
_LIGHT_TERMINAL_BG = "#F8FAFC"

_DARK_INPUT_CANDIDATES = (
    "#FFFFFF",
    "#F8FAFC",
    "#E5E7EB",
    "#D1D5DB",
)
_LIGHT_INPUT_CANDIDATES = (
    "#111827",
    "#1A1A1A",
    "#000000",
    "#374151",
)
_PLACEHOLDER_CANDIDATES = (
    "#94A3B8",
    "#CBD5E1",
    "#64748B",
    "#475569",
    "#808080",
)
_PROMPT_CANDIDATES = (
    "#E6DDC4",
    "#F1EAD6",
    "#CBD5E1",
    "#94A3B8",
    "#64748B",
    "#475569",
    "#334155",
    "#FFFFFF",
    "#111827",
)
_BORDER_CANDIDATES = (
    "#D8D1BB",
    "#C8C3B3",
    "#A8B2AF",
    "#94A3B8",
    "#64748B",
    "#475569",
    "#FFFFFF",
    "#111827",
)
_TEXT_CANDIDATES = (
    "#F1EAD6",
    "#F8FAFC",
    "#E5E7EB",
    "#CBD5E1",
    "#111827",
    "#1A1A1A",
    "#000000",
)
_MUTED_TEXT_CANDIDATES = (
    "#C8C3B3",
    "#A8B2AF",
    "#94A3B8",
    "#CBD5E1",
    "#64748B",
    "#475569",
    "#808080",
)
_ANSI_BG = {
    0: "#000000",
    1: "#800000",
    2: "#008000",
    3: "#808000",
    4: "#000080",
    5: "#800080",
    6: "#008080",
    7: "#C0C0C0",
    8: "#808080",
    9: "#FF0000",
    10: "#00FF00",
    11: "#FFFF00",
    12: "#0000FF",
    13: "#FF00FF",
    14: "#00FFFF",
    15: "#FFFFFF",
}


@dataclass(frozen=True)
class Theme:
    """Immutable base color palette consumed by TUI components."""

    name: str
    accent: str
    accent_dim: str
    success: str
    warning: str
    error: str
    info: str
    text: str
    text_dim: str
    text_muted: str
    border: str
    border_dim: str
    panel_title: str
    recording: str
    executing: str
    code_bg: str
    prompt_char: str
    input_text: str
    input_bg: str
    input_placeholder: str
    input_border: str
    input_focus_border: str
    input_selection_bg: str
    input_selection_fg: str
    input_disabled_text: str
    toolbar_bg: str
    toolbar_fg: str
    statusbar_fg: str
    statusbar_accent: str
    statusbar_dim: str
    statusbar_good: str
    prompt_paused: str
    auto_suggest: str


@dataclass(frozen=True)
class ResolvedTheme(Theme):
    """Theme after terminal-background and contrast normalization."""

    terminal_bg: str
    base_name: str


_DARK = Theme(
    name="dark",
    accent="#E6DDC4",
    accent_dim="#A8B2AF",
    success="#A7C7A1",
    warning="#D8C690",
    error="#F08A8A",
    info="#C8D3CF",
    text="#F1EAD6",
    text_dim="#C8C3B3",
    text_muted="#9AA6A3",
    border="#D8D1BB",
    border_dim="#8FA09B",
    panel_title="bold #E6DDC4",
    recording="bold #F08A8A",
    executing="bold #A7C7A1",
    code_bg="#0B1F24",
    prompt_char="bold #E6DDC4",
    input_text="#F8FAFC",
    input_bg=_DARK_TERMINAL_BG,
    input_placeholder="#94A3B8",
    input_border="#8FA09B",
    input_focus_border="#E6DDC4",
    input_selection_bg="#334155",
    input_selection_fg="#F8FAFC",
    input_disabled_text="#94A3B8",
    toolbar_bg="#102A2F",
    toolbar_fg="#A8B2AF",
    statusbar_fg="#CD7F32",
    statusbar_accent="#FFBF00",
    statusbar_dim="#B8860B",
    statusbar_good="#FFBF00",
    prompt_paused="bold #D8C690",
    auto_suggest="#94A3B8",
)

_LIGHT = Theme(
    name="light",
    accent="#334155",
    accent_dim="#64748B",
    success="#2F6F4E",
    warning="#8A6D1D",
    error="#B42318",
    info="#475569",
    text="#111827",
    text_dim="#475569",
    text_muted="#64748B",
    border="#64748B",
    border_dim="#94A3B8",
    panel_title="bold #334155",
    recording="bold #B42318",
    executing="bold #2F6F4E",
    code_bg="#F8FAFC",
    prompt_char="bold #334155",
    input_text="#111827",
    input_bg=_LIGHT_TERMINAL_BG,
    input_placeholder="#64748B",
    input_border="#64748B",
    input_focus_border="#334155",
    input_selection_bg="#D6E4FF",
    input_selection_fg="#111827",
    input_disabled_text="#64748B",
    toolbar_bg=_LIGHT_TERMINAL_BG,
    toolbar_fg="#475569",
    statusbar_fg="#8B5E34",
    statusbar_accent="#B8860B",
    statusbar_dim="#A16207",
    statusbar_good="#B8860B",
    prompt_paused="bold #8A6D1D",
    auto_suggest="#64748B",
)


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse a #RRGGBB color into RGB channels."""
    if not _HEX_RE.match(value):
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def relative_luminance(color: str) -> float:
    """Return WCAG relative luminance for a #RRGGBB color."""
    channels = []
    for channel in parse_hex_color(color):
        value = channel / 255
        if value <= 0.03928:
            channels.append(value / 12.92)
        else:
            channels.append(((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """Return WCAG contrast ratio between two #RRGGBB colors."""
    fg_luminance = relative_luminance(foreground)
    bg_luminance = relative_luminance(background)
    lighter = max(fg_luminance, bg_luminance)
    darker = min(fg_luminance, bg_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def is_light_color(color: str) -> bool:
    """Return True when the color is perceived as a light background."""
    return relative_luminance(color) >= 0.5


def ensure_contrast(
    preferred: str,
    background: str,
    *,
    min_ratio: float,
    candidates: tuple[str, ...],
) -> str:
    """Choose the preferred color or the closest candidate that is readable."""
    valid_candidates = (preferred, *candidates)
    best_color = preferred
    best_ratio = -1.0
    for color in valid_candidates:
        try:
            ratio = contrast_ratio(color, background)
        except ValueError:
            continue
        if ratio >= min_ratio:
            return color
        if ratio > best_ratio:
            best_color = color
            best_ratio = ratio
    return best_color


def _style_color(style: str) -> str | None:
    for part in reversed(style.split()):
        if _HEX_RE.match(part):
            return part
    return None


def _with_style_color(style: str, color: str) -> str:
    parts = style.split()
    replaced = False
    next_parts: list[str] = []
    for part in parts:
        if _HEX_RE.match(part):
            next_parts.append(color)
            replaced = True
        else:
            next_parts.append(part)
    if not replaced:
        next_parts.append(color)
    return " ".join(next_parts)


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    source = os.environ if env is None else env
    return source.get(key, "")


def _explicit_theme_name(env: Mapping[str, str] | None = None) -> str:
    explicit = _env_value(env, "LEAPFLOW_TUI_THEME").strip().lower()
    return explicit if explicit in {"light", "dark"} else ""


def _background_from_env(env: Mapping[str, str] | None = None) -> str | None:
    explicit_bg = _env_value(env, "LEAPFLOW_TUI_BG").strip()
    if explicit_bg:
        if _HEX_RE.match(explicit_bg):
            return explicit_bg.upper()
        return None

    colorfgbg = _env_value(env, "COLORFGBG")
    if ";" in colorfgbg:
        try:
            bg_index = int(colorfgbg.rsplit(";", 1)[1])
        except (ValueError, IndexError):
            return None
        return _ANSI_BG.get(bg_index)

    return None


def _looks_light_from_name(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    light_markers = ("light", "day", "paper", "latte", "cream", "solarized-light")
    dark_markers = ("dark", "night", "black", "dim", "moon", "dracula")
    if any(marker in normalized for marker in dark_markers):
        return False
    return any(marker in normalized for marker in light_markers)


def detect_light_mode(env: Mapping[str, str] | None = None) -> bool:
    """Conservatively detect whether the terminal background is light."""
    explicit = _explicit_theme_name(env)
    if explicit:
        return explicit == "light"

    background = _background_from_env(env)
    if background is not None:
        return is_light_color(background)

    for key in ("ITERM_PROFILE", "TERMINAL_PROFILE", "WT_PROFILE_ID"):
        if _looks_light_from_name(_env_value(env, key)):
            return True

    return False


def _select_base_theme(env: Mapping[str, str] | None = None) -> Theme:
    explicit = _explicit_theme_name(env)
    if explicit == "light":
        return _LIGHT
    if explicit == "dark":
        return _DARK
    return _LIGHT if detect_light_mode(env) else _DARK


def resolve_theme(
    base: Theme,
    *,
    env: Mapping[str, str] | None = None,
    terminal_bg: str | None = None,
) -> ResolvedTheme:
    """Resolve a base theme into contrast-safe TUI colors."""
    background = terminal_bg or _background_from_env(env) or base.input_bg
    if not _HEX_RE.match(background):
        background = base.input_bg
    background = background.upper()
    light_background = is_light_color(background)
    surface_bg = background if light_background else base.toolbar_bg
    input_candidates = _LIGHT_INPUT_CANDIDATES if light_background else _DARK_INPUT_CANDIDATES
    input_text = ensure_contrast(
        base.input_text,
        background,
        min_ratio=_INPUT_TEXT_MIN_CONTRAST,
        candidates=input_candidates,
    )
    input_placeholder = ensure_contrast(
        base.input_placeholder,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_PLACEHOLDER_CANDIDATES,
    )
    auto_suggest = ensure_contrast(
        base.auto_suggest,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_PLACEHOLDER_CANDIDATES,
    )
    input_disabled_text = ensure_contrast(
        base.input_disabled_text,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_PLACEHOLDER_CANDIDATES,
    )
    text = ensure_contrast(
        base.text,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_TEXT_CANDIDATES,
    )
    text_dim = ensure_contrast(
        base.text_dim,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_MUTED_TEXT_CANDIDATES,
    )
    text_muted = ensure_contrast(
        base.text_muted,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_MUTED_TEXT_CANDIDATES,
    )
    accent = ensure_contrast(
        base.accent,
        background,
        min_ratio=_PROMPT_MIN_CONTRAST,
        candidates=_PROMPT_CANDIDATES,
    )
    accent_dim = ensure_contrast(
        base.accent_dim,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_BORDER_CANDIDATES,
    )
    border = ensure_contrast(
        base.border,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_BORDER_CANDIDATES,
    )
    border_dim = ensure_contrast(
        base.border_dim,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_BORDER_CANDIDATES,
    )
    input_border = ensure_contrast(
        base.input_border,
        background,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_BORDER_CANDIDATES,
    )
    toolbar_fg = ensure_contrast(
        base.toolbar_fg,
        surface_bg,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=_MUTED_TEXT_CANDIDATES,
    )
    statusbar_fg = ensure_contrast(
        base.statusbar_fg,
        surface_bg,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=(
            base.statusbar_accent,
            base.statusbar_dim,
            "#FFBF00",
            "#CD7F32",
            "#8B5E34",
        ),
    )
    statusbar_accent = ensure_contrast(
        base.statusbar_accent,
        surface_bg,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=("#FFBF00", "#D97706", "#B8860B", "#8B5E34"),
    )
    statusbar_dim = ensure_contrast(
        base.statusbar_dim,
        surface_bg,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=(base.statusbar_fg, "#CD7F32", "#B8860B", "#8B5E34"),
    )
    statusbar_good = ensure_contrast(
        base.statusbar_good,
        surface_bg,
        min_ratio=_SECONDARY_TEXT_MIN_CONTRAST,
        candidates=(
            base.statusbar_accent,
            "#FFBF00",
            "#D97706",
            "#B8860B",
        ),
    )
    prompt_color = ensure_contrast(
        _style_color(base.prompt_char) or base.accent,
        background,
        min_ratio=_PROMPT_MIN_CONTRAST,
        candidates=_PROMPT_CANDIDATES,
    )
    prompt_paused_color = ensure_contrast(
        _style_color(base.prompt_paused) or base.warning,
        background,
        min_ratio=_PROMPT_MIN_CONTRAST,
        candidates=_PROMPT_CANDIDATES,
    )
    focus_border = ensure_contrast(
        base.input_focus_border,
        background,
        min_ratio=_PROMPT_MIN_CONTRAST,
        candidates=_BORDER_CANDIDATES,
    )
    panel_title_color = ensure_contrast(
        _style_color(base.panel_title) or accent,
        background,
        min_ratio=_PROMPT_MIN_CONTRAST,
        candidates=_PROMPT_CANDIDATES,
    )

    return ResolvedTheme(
        name=base.name,
        accent=accent,
        accent_dim=accent_dim,
        success=base.success,
        warning=base.warning,
        error=base.error,
        info=base.info,
        text=text,
        text_dim=text_dim,
        text_muted=text_muted,
        border=border,
        border_dim=border_dim,
        panel_title=_with_style_color(base.panel_title, panel_title_color),
        recording=base.recording,
        executing=base.executing,
        code_bg=base.code_bg,
        prompt_char=_with_style_color(base.prompt_char, prompt_color),
        input_text=input_text,
        input_bg=background,
        input_placeholder=input_placeholder,
        input_border=input_border,
        input_focus_border=focus_border,
        input_selection_bg=base.input_selection_bg,
        input_selection_fg=base.input_selection_fg,
        input_disabled_text=input_disabled_text,
        toolbar_bg=surface_bg,
        toolbar_fg=toolbar_fg,
        statusbar_fg=statusbar_fg,
        statusbar_accent=statusbar_accent,
        statusbar_dim=statusbar_dim,
        statusbar_good=statusbar_good,
        prompt_paused=_with_style_color(base.prompt_paused, prompt_paused_color),
        auto_suggest=auto_suggest,
        terminal_bg=background,
        base_name=base.name,
    )


def detect_theme(
    env: Mapping[str, str] | None = None,
    *,
    is_tty: bool | None = None,
) -> ResolvedTheme:
    """Return a contrast-safe theme based on terminal environment."""
    tty = sys.stdout.isatty() if is_tty is None else is_tty
    has_override = bool(_explicit_theme_name(env) or _background_from_env(env))
    if not tty and not has_override:
        return resolve_theme(_DARK, env=env, terminal_bg=_DARK.input_bg)
    return resolve_theme(_select_base_theme(env), env=env)
