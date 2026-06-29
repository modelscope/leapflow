"""Shared CLI utilities — recording animation, guards, perceptual-field helpers.

Progress reporters and stage configs have been relocated to
``leapflow.utils.progress``. They are re-exported here for backward
compatibility so existing ``from leapflow.cli.helpers import …`` paths
continue to work.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

# Backward-compatibility re-exports — implementation lives in utils.progress.
from leapflow.utils.progress import (
    StopPhaseProgress,
    VerboseLearnProgress,
    finish_learn_progress,
    install_learn_progress,
    install_stop_progress,
)

if TYPE_CHECKING:
    from leapflow.cli.context import Context


_DIM = "\033[2m"
_RESET = "\033[0m"


def require_initialized(ctx: "Context") -> None:
    """Guard: abort early if async initialization failed."""
    if ctx.engine is None or ctx.session is None or ctx.registry is None:
        raise RuntimeError(
            "LEAP Agent failed to initialize. Check logs for errors."
        )


async def blink_recording(stop_event: asyncio.Event) -> None:
    """Show red recording indicator while learn is active."""
    if not sys.stdout.isatty():
        print("Recording... (type 'stop' to finish)", flush=True)
        await stop_event.wait()
        return

    print(
        "\033[31m● Recording...\033[0m  "
        "\033[2m(type 'stop' to finish)\033[0m",
        flush=True,
    )
    await stop_event.wait()
    print(f"{_DIM}  Recording stopped.{_RESET}", flush=True)


def get_perceptual_field_filter(ctx: "Context"):
    """Retrieve the PerceptualFieldFilter from the recorder's filter chain (or None)."""
    from leapflow.recording.perceptual_field import PerceptualFieldFilter
    for filt in ctx.imitation.recorder._attention_filters:
        if isinstance(filt, PerceptualFieldFilter):
            return filt
    return None


__all__ = [
    # CLI-specific
    "blink_recording",
    "get_perceptual_field_filter",
    "require_initialized",
    # Re-exports from utils.progress (backward compatibility)
    "StopPhaseProgress",
    "VerboseLearnProgress",
    "finish_learn_progress",
    "install_learn_progress",
    "install_stop_progress",
]
