"""Terminal approval view helpers for LeapFlow CLI/TUI surfaces."""
from __future__ import annotations

import asyncio
import sys
import textwrap
import time
from dataclasses import dataclass

from leapflow.security.approval import ApprovalDecision, ApprovalRequest
from leapflow.security.redact import redact_sensitive_text


@dataclass(frozen=True)
class ApprovalChoice:
    """One selectable approval choice."""

    key: str
    label: str
    decision: ApprovalDecision | None = None


_CHOICE_LABELS = {
    "allow_once": "Allow once",
    "allow_session": "Allow for this session",
    "allow_always": "Add to permanent allowlist",
    "deny": "Deny",
    "deny_always": "Deny for this session",
    "show_details": "Show full details",
}

_CHOICE_DECISIONS = {
    "allow_once": ApprovalDecision.ALLOW_ONCE,
    "allow_session": ApprovalDecision.ALLOW_SESSION,
    "allow_always": ApprovalDecision.ALLOW_ALWAYS,
    "deny": ApprovalDecision.DENY,
    "deny_always": ApprovalDecision.DENY_ALWAYS,
}


async def prompt_approval(request: ApprovalRequest) -> ApprovalDecision:
    """Render an approval prompt and return a user decision."""
    if not sys.stdin.isatty():
        return ApprovalDecision.DENY

    choices = _build_choices(request)
    show_details = False
    while True:
        if _is_expired(request):
            return ApprovalDecision.DENY
        _render(request, choices, show_details=show_details)
        try:
            answer = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, lambda: input("  → ").strip().lower(),
                ),
                timeout=_remaining_seconds(request),
            )
        except TimeoutError:
            return ApprovalDecision.DENY
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY

        selected = _resolve_choice(answer, choices)
        if selected is None:
            return ApprovalDecision.DENY
        if selected.key == "show_details":
            show_details = True
            continue
        return selected.decision or ApprovalDecision.DENY


def _build_choices(request: ApprovalRequest) -> list[ApprovalChoice]:
    keys = list(request.choices or ("allow_once", "allow_session", "deny"))
    if "deny" not in keys:
        keys.append("deny")
    choices = []
    for key in keys:
        choices.append(ApprovalChoice(
            key=key,
            label=_CHOICE_LABELS.get(key, key.replace("_", " ").title()),
            decision=_CHOICE_DECISIONS.get(key),
        ))
    return choices


def _resolve_choice(answer: str, choices: list[ApprovalChoice]) -> ApprovalChoice | None:
    if not answer:
        return next((choice for choice in choices if choice.key == "deny"), None)
    aliases = {
        "y": "allow_once",
        "yes": "allow_once",
        "o": "allow_once",
        "once": "allow_once",
        "s": "allow_session",
        "session": "allow_session",
        "a": "allow_always",
        "always": "allow_always",
        "n": "deny",
        "no": "deny",
        "d": "deny",
        "deny": "deny",
        "v": "show_details",
        "view": "show_details",
        "full": "show_details",
    }
    key = aliases.get(answer, answer)
    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    return next((choice for choice in choices if choice.key == key), None)


def _render(request: ApprovalRequest, choices: list[ApprovalChoice], *, show_details: bool) -> None:
    title = str(request.display.get("title") or _title_for(request))
    summary = str(request.display.get("summary") or request.category)
    reason = str(request.display.get("reason") or _risk_reason(request))
    detail = redact_sensitive_text(request.detail, force=True)
    if not show_details:
        detail = _truncate_detail(detail)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console(stderr=True, highlight=False)
        body = Text()
        body.append(f"{summary}\n\n", style="bold")
        body.append("Action detail:\n", style="dim")
        body.append(_indent(detail) + "\n\n", style="yellow")
        if reason:
            body.append("Why approval is needed:\n", style="dim")
            for line in textwrap.wrap(reason, width=72) or [reason]:
                body.append(f"- {line}\n", style="dim")
            body.append("\n")
        remaining = _remaining_seconds(request)
        if remaining is not None:
            body.append(f"Defaults to Deny in {int(remaining)}s.\n\n", style="dim")
        for idx, choice in enumerate(choices, start=1):
            body.append(f"  {idx}. {choice.label}\n", style="bold" if choice.key == request.default_choice else "")
        console.print(Panel(
            body,
            title=f"[bold yellow]⚠ {title}[/]",
            border_style="yellow",
            padding=(0, 1),
        ))
    except ImportError:
        sys.stderr.write(f"⚠ {title}\n\n{summary}\n\n{detail}\n\n")
        if reason:
            sys.stderr.write(f"Why approval is needed: {reason}\n\n")
        remaining = _remaining_seconds(request)
        if remaining is not None:
            sys.stderr.write(f"Defaults to Deny in {int(remaining)}s.\n\n")
        for idx, choice in enumerate(choices, start=1):
            sys.stderr.write(f"  {idx}. {choice.label}\n")
        sys.stderr.flush()


def _title_for(request: ApprovalRequest) -> str:
    if request.risk is not None:
        if request.risk.level.value == "high":
            return "High Risk Action"
        if request.risk.level.value == "critical":
            return "Critical Action"
    return "Action Approval"


def _risk_reason(request: ApprovalRequest) -> str:
    if request.risk is None:
        return ""
    if request.risk.explanation:
        return request.risk.explanation
    return ", ".join(request.risk.reasons)


def _remaining_seconds(request: ApprovalRequest) -> float | None:
    if request.expires_at is None:
        return None
    return max(0.0, float(request.expires_at) - time.time())


def _is_expired(request: ApprovalRequest) -> bool:
    remaining = _remaining_seconds(request)
    return remaining is not None and remaining <= 0.0


def _truncate_detail(text: str, *, max_lines: int = 6, width: int = 88) -> str:
    wrapped: list[str] = []
    for line in text.splitlines() or [text]:
        wrapped.extend(textwrap.wrap(line, width=width, replace_whitespace=False) or [""])
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    return "\n".join(wrapped[: max_lines - 1] + ["… (choose Show full details)"])


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())
