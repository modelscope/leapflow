"""Prompt-toolkit native approval modal for LeapFlow TUI.

Renders a bordered panel with action summary, detail, risk reason,
and selectable choices — fully within the prompt_toolkit layout.
Keyboard: ↑/↓ navigate, Enter confirm, Esc deny, or type shortcut keys.

Height adaptation: when ``max_lines`` is passed to :meth:`fragments` or
:meth:`line_count`, the modal trims variable content (detail → reason →
timeout → summary) while always preserving the frame and choices.
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
_ContentLine = list[Fragment]

_MIN_WIDTH = 58
_MAX_WIDTH = 104
_DETAIL_LINES = 3
_REASON_LINES = 2
_MIN_MODAL_HEIGHT = 8


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

    # ── Content section builders ──────────────────────────────────────

    def _summary_lines(self, inner: int) -> list[_ContentLine]:
        summary = str(self.request.display.get("summary") or self.request.category)
        lines: list[_ContentLine] = [
            _content_line(t, inner, "class:approval.summary")
            for t in _wrap(summary, inner)[:2]
        ]
        lines.append(_content_line("", inner, ""))
        return lines

    def _detail_lines(self, inner: int) -> list[_ContentLine]:
        detail_raw = redact_sensitive_text(self.request.detail, force=True)
        if not self.show_details:
            detail_raw = truncate_detail(
                detail_raw, max_lines=_DETAIL_LINES, width=inner - 4,
            )
        detail_wrapped: list[str] = []
        for raw_line in detail_raw.splitlines() or [""]:
            detail_wrapped.extend(_wrap(raw_line, inner - 4))
        limit = 50 if self.show_details else _DETAIL_LINES
        return [
            _content_line(f"  {t}", inner, "class:approval.detail")
            for t in detail_wrapped[:limit]
        ]

    def _reason_lines(self, inner: int) -> list[_ContentLine]:
        reason = str(self.request.display.get("reason") or risk_reason(self.request))
        if not reason:
            return []
        lines: list[_ContentLine] = [
            _content_line("", inner, ""),
            _content_line("Why approval is needed:", inner, "class:approval.label"),
        ]
        for t in _wrap(reason, inner - 4)[:_REASON_LINES]:
            lines.append(_content_line(f"  {t}", inner, "class:approval.dim"))
        return lines

    def _timeout_lines(self, inner: int) -> list[_ContentLine]:
        remaining = remaining_seconds(self.request)
        if remaining is None:
            return []
        return [_content_line(
            f"  Auto-deny in {int(remaining)}s", inner, "class:approval.dim",
        )]

    def _choices_lines(self, inner: int) -> list[_ContentLine]:
        lines: list[_ContentLine] = [
            _content_line("", inner, ""),
            _content_line(
                "  ↑↓ navigate · Enter confirm · Esc deny",
                inner, "class:approval.dim",
            ),
        ]
        for idx, choice in enumerate(self.choices):
            selected = idx == self.selected_index
            marker = "▸" if selected else " "
            label = f"  {marker} {idx + 1}. {choice.label}"
            style = "class:approval.selected" if selected else "class:approval.option"
            lines.append(_content_line(label, inner, style))
        return lines

    # ── Public rendering API ──────────────────────────────────────────

    def fragments(self, *, max_lines: int = 0) -> list[Fragment]:
        """Build fragments for the modal, adapting to height constraints.

        When *max_lines* > 0, variable content (summary, detail, reason,
        timeout) is progressively trimmed — in ascending priority order —
        to fit.  Frame borders and choices are never trimmed.
        """
        width = _modal_width()
        inner = width - 4
        title = f"⚠ {title_for_approval(self.request)}"

        fixed_top = [_border_top(width, title)]
        fixed_bottom_content = self._choices_lines(inner)
        fixed_bottom_border = [_border_bottom(width)]
        fixed_count = len(fixed_top) + len(fixed_bottom_content) + len(fixed_bottom_border)

        variable_sections = [
            self._summary_lines(inner),
            self._detail_lines(inner),
            self._reason_lines(inner),
            self._timeout_lines(inner),
        ]

        budget = (
            max(0, max_lines - fixed_count)
            if max_lines > 0
            else sum(len(s) for s in variable_sections)
        )

        variable_lines: list[_ContentLine] = []
        for section in variable_sections:
            if budget <= 0:
                break
            take = section[:budget]
            variable_lines.extend(take)
            budget -= len(take)

        all_lines = fixed_top + variable_lines + fixed_bottom_content + fixed_bottom_border

        result: list[Fragment] = []
        for line in all_lines:
            result.extend(line)
            result.append(("", "\n"))
        return result

    def line_count(self, *, max_lines: int = 0) -> int:
        """Return the number of content lines (for Window height sizing).

        When *max_lines* > 0, the count is clamped to that budget while
        guaranteeing the frame and choices are always accounted for.
        """
        inner = _modal_width() - 4
        fixed = (
            1  # top border
            + len(self._choices_lines(inner))
            + 1  # bottom border
        )
        variable = sum(
            len(s) for s in (
                self._summary_lines(inner),
                self._detail_lines(inner),
                self._reason_lines(inner),
                self._timeout_lines(inner),
            )
        )
        total = fixed + variable
        if max_lines > 0:
            return min(total, max(fixed, max_lines))
        return total

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
