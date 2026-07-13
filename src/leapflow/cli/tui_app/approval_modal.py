"""Prompt-toolkit native approval modal for LeapFlow TUI.

Renders a bordered panel with action summary, detail, risk reason,
and selectable choices — fully within the prompt_toolkit layout.
Keyboard: ↑/↓ navigate, Enter confirm, Esc deny, or type shortcut keys.
"""
from __future__ import annotations

import asyncio
import shutil
import textwrap
from dataclasses import dataclass

from leapflow.cli.approval_view import (
    ApprovalChoice,
    build_approval_choices,
    remaining_seconds,
    resolve_approval_choice,
    risk_reason,
    title_for_approval,
    truncate_detail,
)
from leapflow.security.approval import ApprovalDecision, ApprovalRequest
from leapflow.security.redact import redact_sensitive_text

Fragment = tuple[str, str]

_MIN_WIDTH = 58
_MAX_WIDTH = 104
_DETAIL_LINES = 3
_REASON_LINES = 2


@dataclass
class ApprovalModal:
    """Stateful approval modal rendered inside the prompt-toolkit layout."""

    request: ApprovalRequest
    choices: list[ApprovalChoice]
    selected_index: int
    show_details: bool
    future: asyncio.Future[ApprovalDecision]

    @classmethod
    def create(cls, request: ApprovalRequest) -> ApprovalModal:
        choices = build_approval_choices(request)
        selected_index = _default_index(choices, request.default_choice)
        future = asyncio.get_running_loop().create_future()
        return cls(
            request=request,
            choices=choices,
            selected_index=selected_index,
            show_details=False,
            future=future,
        )

    @property
    def done(self) -> bool:
        return self.future.done()

    def move(self, delta: int) -> None:
        if not self.choices:
            return
        self.selected_index = (self.selected_index + delta) % len(self.choices)

    def choose_selected(self) -> None:
        if not self.choices:
            self.resolve(ApprovalDecision.DENY)
            return
        self._choose(self.choices[self.selected_index])

    def choose_text(self, text: str) -> bool:
        choice = resolve_approval_choice(text.strip().lower(), self.choices)
        if choice is None:
            return False
        self._choose(choice)
        return True

    def deny(self) -> None:
        self.resolve(ApprovalDecision.DENY)

    def resolve(self, decision: ApprovalDecision) -> None:
        if not self.future.done():
            self.future.set_result(decision)

    def fragments(self) -> list[Fragment]:
        """Build all fragments for the modal — no height truncation.

        The prompt_toolkit Window handles clipping/scrolling.
        Content lines are limited by static caps (_DETAIL_LINES,
        _REASON_LINES) to keep the panel concise.
        """
        width = _modal_width()
        inner = width - 4
        title = f"⚠ {title_for_approval(self.request)}"

        lines: list[list[Fragment]] = []

        # ── Top border ──
        lines.append(_border_top(width, title))

        # ── Summary ──
        summary = str(self.request.display.get("summary") or self.request.category)
        for text in _wrap(summary, inner)[:2]:
            lines.append(_content_line(text, inner, "class:approval.summary"))

        # ── Blank separator ──
        lines.append(_content_line("", inner, ""))

        # ── Detail ──
        detail_raw = redact_sensitive_text(self.request.detail, force=True)
        if not self.show_details:
            detail_raw = truncate_detail(
                detail_raw, max_lines=_DETAIL_LINES, width=inner - 4,
            )
        detail_wrapped: list[str] = []
        for raw_line in detail_raw.splitlines() or [""]:
            detail_wrapped.extend(_wrap(raw_line, inner - 4))
        detail_limit = 50 if self.show_details else _DETAIL_LINES
        for text in detail_wrapped[:detail_limit]:
            lines.append(_content_line(f"  {text}", inner, "class:approval.detail"))

        # ── Risk reason ──
        reason = str(self.request.display.get("reason") or risk_reason(self.request))
        if reason:
            lines.append(_content_line("", inner, ""))
            lines.append(_content_line(
                "Why approval is needed:", inner, "class:approval.label",
            ))
            for text in _wrap(reason, inner - 4)[:_REASON_LINES]:
                lines.append(_content_line(f"  {text}", inner, "class:approval.dim"))

        # ── Timeout ──
        remaining = remaining_seconds(self.request)
        if remaining is not None:
            lines.append(_content_line(
                f"  Auto-deny in {int(remaining)}s",
                inner,
                "class:approval.dim",
            ))

        # ── Separator + keyboard hint ──
        lines.append(_content_line("", inner, ""))
        lines.append(_content_line(
            "  ↑↓ navigate · Enter confirm · Esc deny",
            inner,
            "class:approval.dim",
        ))

        # ── Choices ──
        for idx, choice in enumerate(self.choices):
            selected = idx == self.selected_index
            marker = "▸" if selected else " "
            label = f"  {marker} {idx + 1}. {choice.label}"
            style = "class:approval.selected" if selected else "class:approval.option"
            lines.append(_content_line(label, inner, style))

        # ── Bottom border ──
        lines.append(_border_bottom(width))

        # ── Flatten to fragment list ──
        result: list[Fragment] = []
        for line in lines:
            result.extend(line)
            result.append(("", "\n"))
        return result

    def line_count(self) -> int:
        """Return the number of content lines (for Window height sizing)."""
        count = 1 + 2 + 1  # top border, summary(+blank), blank
        count += min(_DETAIL_LINES, 3)
        reason = str(self.request.display.get("reason") or risk_reason(self.request))
        if reason:
            count += 1 + 1 + min(_REASON_LINES, 2)
        if self.request.expires_at is not None:
            count += 1
        count += 1 + 1 + len(self.choices) + 1  # blank, hint, choices, bottom
        return count

    def _choose(self, choice: ApprovalChoice) -> None:
        if choice.key == "show_details":
            self.show_details = True
            return
        self.resolve(choice.decision or ApprovalDecision.DENY)


def _default_index(choices: list[ApprovalChoice], default_choice: str) -> int:
    for index, choice in enumerate(choices):
        if choice.key == default_choice:
            return index
    return 0


def _modal_width() -> int:
    columns = shutil.get_terminal_size((100, 24)).columns
    return min(_MAX_WIDTH, max(_MIN_WIDTH, columns - 6))


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=max(20, width),
        replace_whitespace=False,
        drop_whitespace=False,
    ) or [""]


def _border_top(width: int, title: str) -> list[Fragment]:
    label = f" {title} "
    label = label[: max(0, width - 4)]
    available = max(0, width - 2 - len(label))
    left = available // 2
    right = available - left
    return [
        ("class:approval.border", "╭" + "─" * left),
        ("class:approval.title", label),
        ("class:approval.border", "─" * right + "╮"),
    ]


def _border_bottom(width: int) -> list[Fragment]:
    return [("class:approval.border", "╰" + "─" * (width - 2) + "╯")]


def _content_line(text: str, width: int, style: str = "") -> list[Fragment]:
    clipped = text[:width]
    padding = " " * max(0, width - len(clipped))
    return [
        ("class:approval.border", "│ "),
        (style, clipped + padding),
        ("class:approval.border", " │"),
    ]


def request_is_expired(request: ApprovalRequest) -> bool:
    remaining = remaining_seconds(request)
    return remaining is not None and remaining <= 0.0
