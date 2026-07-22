"""Structured research ledger for long-horizon task state (mechanism 5, W3).

A compact, bounded record of the active task's accumulated findings, open
questions, decisions / excluded paths, and next step. It is re-injected into the
*volatile tail* of every turn (never the cached prefix) so long-task state
survives context compression no matter how much raw history is summarized or
dropped -- the highest-priority guarantee for multi-turn deep work: information
integrity first, compression ratio second.

The ledger is maintained by the agent via the cheap ``research_note`` tool. It
is per-task (reset each turn) and deliberately bounded (per-list cap + per-note
truncation + dedupe) so it stays small and high-signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Recognized note kinds. ``resolved`` closes a matching open question and files
# it as a finding, so the open-question set reflects only what remains.
LEDGER_KINDS = ("finding", "open_question", "resolved", "decision", "next_step")

_MAX_ITEMS = 24     # per-list cap; keep the most recent (SNR over completeness)
_MAX_CHARS = 200    # per-note cap: one concise sentence (bounds every-round re-injection)


@dataclass
class ResearchLedger:
    """Per-task structured state; bounded and re-injected each turn."""

    max_items: int = _MAX_ITEMS
    max_chars: int = _MAX_CHARS

    def __post_init__(self) -> None:
        self._findings: List[str] = []
        self._open_questions: List[str] = []
        self._decisions: List[str] = []
        self._next_step: str = ""
        self._on_change: Optional[Callable[[], None]] = None

    def set_change_listener(self, callback: Optional[Callable[[], None]]) -> None:
        """Install a listener fired after each successful note (e.g. persist)."""
        self._on_change = callback

    def reset(self) -> None:
        """Clear all state at the start of a new task/turn."""
        self._findings.clear()
        self._open_questions.clear()
        self._decisions.clear()
        self._next_step = ""

    def note(self, kind: str, text: str) -> bool:
        """Record a structured note. Returns False for invalid kind/empty text."""
        text = (text or "").strip()[: self.max_chars]
        if not text:
            return False
        kind = (kind or "").strip().lower()
        if kind == "finding":
            self._append(self._findings, text)
        elif kind == "open_question":
            self._append(self._open_questions, text)
        elif kind == "resolved":
            self._resolve(text)
        elif kind == "decision":
            self._append(self._decisions, text)
        elif kind == "next_step":
            self._next_step = text
        else:
            return False
        if self._on_change is not None:
            self._on_change()
        return True

    def _append(self, bucket: List[str], text: str) -> None:
        if text in bucket:
            bucket.remove(text)          # dedupe: move to most-recent
        bucket.append(text)
        if len(bucket) > self.max_items:  # keep most recent
            del bucket[: len(bucket) - self.max_items]

    def _resolve(self, text: str) -> None:
        low = text.lower()
        for i, question in enumerate(self._open_questions):
            ql = question.lower()
            if low in ql or ql in low:
                self._open_questions.pop(i)
                break
        self._append(self._findings, text)

    @property
    def open_question_count(self) -> int:
        return len(self._open_questions)

    @property
    def is_empty(self) -> bool:
        return not (
            self._findings or self._open_questions or self._decisions or self._next_step
        )

    def render(self) -> str:
        """Compact, injectable block. Empty ledger renders to an empty string."""
        if self.is_empty:
            return ""
        lines = [
            "## Research Ledger (task state; authoritative and preserved across compression)"
        ]
        if self._findings:
            lines.append("Findings:")
            lines.extend(f"- {item}" for item in self._findings)
        if self._open_questions:
            lines.append("Open questions:")
            lines.extend(f"- {item}" for item in self._open_questions)
        if self._decisions:
            lines.append("Decisions / excluded paths:")
            lines.extend(f"- {item}" for item in self._decisions)
        if self._next_step:
            lines.append(f"Next step: {self._next_step}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "findings": list(self._findings),
            "open_questions": list(self._open_questions),
            "decisions": list(self._decisions),
            "next_step": self._next_step,
            "open_question_count": self.open_question_count,
        }

    def to_state(self) -> Dict[str, Any]:
        """Persistable snapshot (excludes derived counters)."""
        return {
            "findings": list(self._findings),
            "open_questions": list(self._open_questions),
            "decisions": list(self._decisions),
            "next_step": self._next_step,
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Replace state from a persisted snapshot; does not fire the listener."""
        self.reset()
        if not state:
            return
        self._findings = [str(x) for x in state.get("findings", []) if str(x).strip()][-self.max_items:]
        self._open_questions = [str(x) for x in state.get("open_questions", []) if str(x).strip()][-self.max_items:]
        self._decisions = [str(x) for x in state.get("decisions", []) if str(x).strip()][-self.max_items:]
        self._next_step = str(state.get("next_step", ""))
