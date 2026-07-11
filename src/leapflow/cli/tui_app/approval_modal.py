"""Prompt-toolkit native approval modal for LeapFlow TUI."""
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
_MIN_HEIGHT = 10
_MAX_HEIGHT = 24
_MAX_DETAIL_LINES = 3
_MAX_REASON_LINES = 2


@dataclass
class ApprovalModal:
    """Stateful approval modal rendered inside the prompt-toolkit layout."""

    request: ApprovalRequest
    choices: list[ApprovalChoice]
    selected_index: int
    show_details: bool
    future: asyncio.Future[ApprovalDecision]

    @classmethod
    def create(cls, request: ApprovalRequest) -> "ApprovalModal":
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
        """Return whether the user has already made a decision."""
        return self.future.done()

    def move(self, delta: int) -> None:
        """Move the selected option by ``delta`` rows."""
        if not self.choices:
            return
        self.selected_index = (self.selected_index + delta) % len(self.choices)

    def choose_selected(self) -> None:
        """Accept the currently selected option."""
        if not self.choices:
            self.resolve(ApprovalDecision.DENY)
            return
        self._choose(self.choices[self.selected_index])

    def choose_text(self, text: str) -> bool:
        """Resolve a typed shortcut or numeric choice."""
        choice = resolve_approval_choice(text.strip().lower(), self.choices)
        if choice is None:
            return False
        self._choose(choice)
        return True

    def deny(self) -> None:
        """Fail closed when the user cancels or the request expires."""
        self.resolve(ApprovalDecision.DENY)

    def resolve(self, decision: ApprovalDecision) -> None:
        """Resolve the modal future once."""
        if not self.future.done():
            self.future.set_result(decision)

    def fragments(self, *, max_height: int | None = None) -> list[Fragment]:
        """Render the modal as a complete bordered component."""
        width = _modal_width()
        height = _modal_height(max_height)
        inner_width = width - 4
        title = f"⚠ {title_for_approval(self.request)}"
        lines: list[list[Fragment]] = []
        lines.append(_border_top(width, title))

        summary = str(self.request.display.get("summary") or self.request.category)
        lines.extend(_limited_content_lines(
            _wrap(summary, inner_width),
            inner_width,
            "class:approval.summary",
            limit=1,
        ))

        reason = str(self.request.display.get("reason") or risk_reason(self.request))
        remaining = remaining_seconds(self.request)
        fixed_count = (
            1  # top border
            + 1  # summary
            + 1  # action detail label
            + (1 if reason else 0)  # reason label
            + (1 if remaining is not None else 0)
            + 1  # keyboard hint
            + len(self.choices)
            + 1  # bottom border
        )
        content_budget = max(2, height - fixed_count)
        detail_budget = min(_MAX_DETAIL_LINES, max(1, content_budget // 2))
        reason_budget = min(_MAX_REASON_LINES, max(0, content_budget - detail_budget))

        lines.append(_content_line("Action detail:", inner_width, "class:approval.label"))
        detail = redact_sensitive_text(self.request.detail, force=True)
        if not self.show_details:
            detail = truncate_detail(detail, max_lines=detail_budget, width=inner_width - 2)
        detail_lines: list[str] = []
        for raw_line in detail.splitlines() or [""]:
            detail_lines.extend(_wrap(raw_line, inner_width - 2))
        lines.extend(_limited_content_lines(
            [f"  {line}" for line in detail_lines],
            inner_width,
            "class:approval.detail",
            limit=detail_budget,
        ))

        if reason:
            lines.append(_content_line("Why approval is needed:", inner_width, "class:approval.label"))
            reason_lines = [f"- {line}" for line in _wrap(reason, inner_width - 2)]
            lines.extend(_limited_content_lines(
                reason_lines,
                inner_width,
                "class:approval.dim",
                limit=reason_budget,
            ))

        if remaining is not None:
            lines.append(_content_line(
                f"Defaults to Deny in {int(remaining)}s.",
                inner_width,
                "class:approval.dim",
            ))

        lines.append(_content_line("↑/↓ choose · Enter confirm · Esc deny", inner_width, "class:approval.dim"))
        for idx, choice in enumerate(self.choices, start=1):
            selected = idx - 1 == self.selected_index
            marker = "▶" if selected else " "
            text = f" {marker} {idx}. {choice.label}"
            style = "class:approval.selected" if selected else "class:approval.option"
            lines.append(_content_line(text, inner_width, style))

        lines.append(_border_bottom(width))
        if len(lines) > height:
            lines = _preserve_frame_and_choices(lines, height, len(self.choices))
        fragments: list[Fragment] = []
        for line in lines:
            fragments.extend(line)
            fragments.append(("", "\n"))
        return fragments

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


def _modal_height(max_height: int | None = None) -> int:
    if max_height is not None:
        return max(_MIN_HEIGHT, max_height)
    rows = shutil.get_terminal_size((100, 24)).lines
    return min(_MAX_HEIGHT, max(_MIN_HEIGHT, rows - 5))


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=max(20, width),
        replace_whitespace=False,
        drop_whitespace=False,
    ) or [""]


def _limited_content_lines(
    lines: list[str],
    width: int,
    style: str,
    *,
    limit: int,
) -> list[list[Fragment]]:
    if limit <= 0:
        return []
    visible = list(lines[:limit]) or [""]
    if len(lines) > limit:
        visible[-1] = _with_ellipsis(visible[-1], width)
    return [_content_line(line, width, style) for line in visible]


def _with_ellipsis(text: str, width: int) -> str:
    if width <= 1:
        return "…"
    if len(text) >= width:
        return text[: width - 1] + "…"
    return f"{text} …"


def _preserve_frame_and_choices(
    lines: list[list[Fragment]],
    height: int,
    choice_count: int,
) -> list[list[Fragment]]:
    if len(lines) <= height:
        return lines
    tail_count = choice_count + 1
    head_budget = max(1, height - tail_count)
    return [*lines[:head_budget], *lines[-tail_count:]]


def _border_top(width: int, title: str) -> list[Fragment]:
    label = f" {title} "
    available = max(0, width - 2 - len(label))
    left = available // 2
    right = available - left
    return [
        ("class:approval.border", "╭" + "─" * left),
        ("class:approval.title", label[: max(0, width - 2)]),
        ("class:approval.border", "─" * right + "╮"),
    ]


def _border_bottom(width: int) -> list[Fragment]:
    return [("class:approval.border", "╰" + "─" * (width - 2) + "╯")]


def _content_line(text: str, width: int, style: str = "") -> list[Fragment]:
    clipped = text[:width]
    padding = " " * max(0, width - len(clipped))
    return [
        ("class:approval.border", "│ "),
        (style, clipped),
        ("", padding),
        ("class:approval.border", " │"),
    ]


def request_is_expired(request: ApprovalRequest) -> bool:
    """Return True when an approval request is past its deadline."""
    remaining = remaining_seconds(request)
    return remaining is not None and remaining <= 0.0
