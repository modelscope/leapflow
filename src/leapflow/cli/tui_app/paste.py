"""Safe paste handling for the TUI input buffer.

This module keeps high-risk pasted content out of the visible prompt_toolkit
buffer while preserving the full logical text for command submission.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LARGE_PASTE_CHAR_THRESHOLD = 4_000
LARGE_PASTE_LINE_THRESHOLD = 24
FRAGMENTED_PASTE_CHAR_THRESHOLD = 800
FRAGMENTED_PASTE_LINE_THRESHOLD = 8
PASTE_FRAGMENT_WINDOW_S = 0.08

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")


@dataclass
class PasteBlock:
    """Full pasted content stored outside the visible prompt buffer."""

    marker: str
    text: str


class PasteHeuristics:
    """Decide which text is unsafe to render directly in a terminal input row."""

    def should_compact_block(self, text: str) -> bool:
        """Return True when a single insert should be represented as a marker."""
        if len(text) >= LARGE_PASTE_CHAR_THRESHOLD:
            return True
        if text.count("\n") + 1 >= LARGE_PASTE_LINE_THRESHOLD:
            return True
        return self.has_display_unsafe_controls(text)

    def should_compact_fragment_window(self, text: str) -> bool:
        """Return True when accumulated small inserts look like one pasted block."""
        if len(text) >= FRAGMENTED_PASTE_CHAR_THRESHOLD:
            return True
        return text.count("\n") + 1 >= FRAGMENTED_PASTE_LINE_THRESHOLD

    def normalize_original(self, text: str) -> str:
        """Preserve semantic text while removing terminal-rendering control bytes."""
        normalized = _ANSI_ESCAPE_RE.sub("", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        return "".join(char for char in normalized if self._is_safe_original_char(char))

    def has_display_unsafe_controls(self, text: str) -> bool:
        """Return True for controls that should never reach the visible buffer."""
        if _ANSI_ESCAPE_RE.search(text):
            return True
        return any(not self._is_safe_display_char(char) for char in text)

    @staticmethod
    def _is_safe_original_char(char: str) -> bool:
        code = ord(char)
        if char in {"\n", "\t"}:
            return True
        if code < 0x20 or code == 0x7F:
            return False
        if 0x200B <= code <= 0x200F:
            return False
        if 0x202A <= code <= 0x202E:
            return False
        if 0x2066 <= code <= 0x2069:
            return False
        return True

    def _is_safe_display_char(self, char: str) -> bool:
        if char in {"\n", "\t"}:
            return True
        return self._is_safe_original_char(char)


class PasteStore:
    """Side-channel store that maps safe visible markers to full pasted text."""

    def __init__(self, heuristics: PasteHeuristics | None = None) -> None:
        self._heuristics = heuristics or PasteHeuristics()
        self._next_paste_id = 1
        self._blocks: dict[str, PasteBlock] = {}

    @property
    def has_blocks(self) -> bool:
        return bool(self._blocks)

    @property
    def blocks(self) -> dict[str, PasteBlock]:
        return self._blocks

    def create_marker(self, text: str) -> str:
        """Store pasted text and return an ASCII-only display marker."""
        paste_id = self._next_paste_id
        self._next_paste_id += 1
        normalized = self._heuristics.normalize_original(text)
        line_count = normalized.count("\n") + 1
        marker = (
            f"[pasted block #{paste_id}: {self._compact_tokens(normalized)}, "
            f"{line_count} lines; full text will be submitted]"
        )
        self._blocks[marker] = PasteBlock(marker=marker, text=normalized)
        return marker

    def append_to_marker(self, marker: str, text: str) -> None:
        """Append continuation chunks to an existing pasted block."""
        block = self._blocks.get(marker)
        if block is None:
            return
        block.text += self._heuristics.normalize_original(text)

    def resolve(self, text: str) -> str:
        """Replace visible paste markers with full logical pasted text."""
        if not self._blocks:
            return text
        resolved = text
        for marker, block in self._blocks.items():
            if marker in resolved:
                resolved = resolved.replace(marker, block.text)
        self.clear()
        return resolved

    def clear(self) -> None:
        """Clear all stored paste blocks."""
        self._blocks.clear()

    @staticmethod
    def _compact_tokens(text: str) -> str:
        size = len(text)
        if size < 1_000:
            return f"{size} chars"
        if size < 1_000_000:
            return f"{size / 1000:.1f}K chars"
        return f"{size / 1_000_000:.1f}M chars"
