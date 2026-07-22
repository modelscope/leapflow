"""Per-turn recovery state — one-shot guards preventing infinite recovery loops.

Each recovery strategy can fire at most once per turn. Prevents:
- Double force-compress on consecutive overflows
- Repeated native→text fallback after first attempt
- OAuth refresh storms
- Infinite grammar/format recovery
- Provider failover loops

Inspired by hermes TurnRetryState, adapted for leapflow's async-first design.
Extended with provider-specific recovery strategies for multi-provider support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnRecoveryState:
    """Mutable per-turn state tracking which recovery strategies have fired.

    Each flag starts False and flips True on first use. Callers check
    ``try_X()`` which returns True only on the first call.
    """

    _compressed: bool = field(default=False, repr=False)
    _native_fallback: bool = field(default=False, repr=False)
    _credential_rotated: bool = field(default=False, repr=False)
    _image_shrunk: bool = field(default=False, repr=False)
    _format_recovery: bool = field(default=False, repr=False)
    _provider_failover: bool = field(default=False, repr=False)
    _thinking_disabled: bool = field(default=False, repr=False)
    _length_continuation: bool = field(default=False, repr=False)
    _multimodal_strip: bool = field(default=False, repr=False)

    consecutive_api_errors: int = 0
    consecutive_tool_failures: int = 0
    last_error_category: Optional[str] = field(default=None, repr=False)

    def try_compress(self) -> bool:
        """Attempt force-compress recovery. Returns True only on first call."""
        if self._compressed:
            return False
        self._compressed = True
        return True

    def try_native_fallback(self) -> bool:
        """Attempt native→text tool calling fallback. First call only."""
        if self._native_fallback:
            return False
        self._native_fallback = True
        return True

    def try_credential_rotate(self) -> bool:
        """Attempt credential rotation. First call only."""
        if self._credential_rotated:
            return False
        self._credential_rotated = True
        return True

    def try_image_shrink(self) -> bool:
        """Attempt image payload shrink. First call only."""
        if self._image_shrunk:
            return False
        self._image_shrunk = True
        return True

    def try_format_recovery(self) -> bool:
        """Attempt format/grammar recovery. First call only."""
        if self._format_recovery:
            return False
        self._format_recovery = True
        return True

    def try_provider_failover(self) -> bool:
        """Attempt provider failover (switch to fallback provider). First call only."""
        if self._provider_failover:
            return False
        self._provider_failover = True
        return True

    def try_disable_thinking(self) -> bool:
        """Attempt disabling thinking/reasoning mode. First call only.

        Some providers (e.g., Claude with extended thinking) may fail
        when thinking is enabled with certain message shapes.
        """
        if self._thinking_disabled:
            return False
        self._thinking_disabled = True
        return True

    def try_length_continuation(self) -> bool:
        """Attempt continuation for max_tokens truncation. First call only."""
        if self._length_continuation:
            return False
        self._length_continuation = True
        return True

    def try_multimodal_strip(self) -> bool:
        """Attempt stripping multimodal content (images) from messages. First call only.

        Used when provider returns multimodal-related errors.
        """
        if self._multimodal_strip:
            return False
        self._multimodal_strip = True
        return True

    def rearm_after_progress(self) -> bool:
        """Re-arm content-level one-shot recovery guards after genuine progress.

        A long task is a single turn spanning many iterations, and legitimately
        needs the same *content* recovery more than once — e.g. several max_tokens
        continuations or context force-compressions across the turn. Once the
        task has made progress (so it is not in a recovery storm), these guards
        are re-armed so recovery stays available for the rest of the turn.

        Infrastructure-level one-shots (native->text fallback, provider failover,
        credential rotation) stay strict for the whole turn — they are storm-prone
        and are already bounded separately by the RecoveryBudget. Returns True if
        any guard was actually re-armed.
        """
        rearmed = (
            self._compressed or self._length_continuation or self._format_recovery
            or self._image_shrunk or self._multimodal_strip or self._thinking_disabled
        )
        if rearmed:
            self._compressed = False
            self._length_continuation = False
            self._format_recovery = False
            self._image_shrunk = False
            self._multimodal_strip = False
            self._thinking_disabled = False
        return rearmed

    def record_api_error(self, category: Optional[str] = None) -> int:
        """Increment API error counter. Returns new count."""
        self.consecutive_api_errors += 1
        self.last_error_category = category
        return self.consecutive_api_errors

    def record_api_success(self) -> None:
        self.consecutive_api_errors = 0
        self.last_error_category = None

    def record_tool_failure(self) -> int:
        self.consecutive_tool_failures += 1
        return self.consecutive_tool_failures

    def record_tool_success(self) -> None:
        self.consecutive_tool_failures = 0
