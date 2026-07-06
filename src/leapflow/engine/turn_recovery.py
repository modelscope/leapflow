"""Per-turn recovery state — one-shot guards preventing infinite recovery loops.

Each recovery strategy can fire at most once per turn. Prevents:
- Double force-compress on consecutive overflows
- Repeated native→text fallback after first attempt
- OAuth refresh storms
- Infinite grammar/format recovery

Inspired by hermes TurnRetryState, adapted for leapflow's async-first design.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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

    consecutive_api_errors: int = 0
    consecutive_tool_failures: int = 0

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

    def record_api_error(self) -> int:
        """Increment API error counter. Returns new count."""
        self.consecutive_api_errors += 1
        return self.consecutive_api_errors

    def record_api_success(self) -> None:
        self.consecutive_api_errors = 0

    def record_tool_failure(self) -> int:
        self.consecutive_tool_failures += 1
        return self.consecutive_tool_failures

    def record_tool_success(self) -> None:
        self.consecutive_tool_failures = 0
