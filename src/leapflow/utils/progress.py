"""Verbose progress reporters for multi-stage pipelines (CLI-friendly).

Originally extracted from ``leapflow.cli.helpers``. These reporters print
dim ANSI-styled lines to stdout, suitable for interactive terminal usage.

The classes here are deliberately kept generic enough to be reused by any
multi-stage pipeline (learning, recording finalize, future P0-P3 fixes),
even though their default stage labels are tuned for the learning and
recording stop phases respectively.

Public API:
    - ``VerboseLearnProgress``: callable progress reporter for learning
      pipeline stages (segment → abstract → intent → distill → activate).
    - ``StopPhaseProgress``: callable progress reporter for the recording
      stop / finalize phase.
    - ``install_learn_progress(ctx)``: wire a learn reporter to a CLI
      context's session.
    - ``finish_learn_progress()``: cleanup hook (currently a no-op).
    - ``install_stop_progress(ctx)``: wire a stop-phase reporter to a CLI
      context's imitation pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.cli.context import Context


_DIM = "\033[2m"
_RESET = "\033[0m"

_LEARN_STAGE_CONFIG = {
    "segment": ("Segmenting trajectory", False),
    "abstract": ("Abstracting actions", False),
    "intent": ("Inferring intent", True),
    "distill": ("Distilling skills", True),
    "activate": ("Activating skills", False),
}

_STOP_STAGE_CONFIG = {
    "drain": "Flushing event buffer",
    "save": "Saving trajectory",
    "visual_stop": "Stopping visual capture",
    "extract": "Analyzing captured frames",
    "extract.refine": "Refining keyframes",
    "extract.preprocess": "Preprocessing frame pairs",
    "extract.signals": "Collecting interaction signals",
    "extract.vlm": "Extracting actions from frames",
    "inject": "Building steps from visual actions",
}


def _dim(msg: str) -> None:
    """Print a dim grey line to stdout (consistent with rest of CLI)."""
    print(f"{_DIM}  {msg}{_RESET}", flush=True)


class VerboseLearnProgress:
    """Structured verbose progress reporter for the learning pipeline.

    Uses ``print()`` to stdout with dim ANSI coloring, consistent with
    the rest of the learn CLI output. Reusable for any pipeline whose
    stages are listed in ``_LEARN_STAGE_CONFIG``; otherwise falls back
    to the raw stage name.
    """

    def __init__(self) -> None:
        self._current_stage = ""
        self._stage_order = ["segment", "abstract", "intent", "distill", "activate"]

    def __call__(self, stage: str, current: int, total: int) -> None:
        if stage != self._current_stage:
            self._enter_stage(stage, total)
        self._report_item(stage, current, total)

    def _enter_stage(self, stage: str, total: int) -> None:
        label, is_llm = _LEARN_STAGE_CONFIG.get(stage, (stage, False))
        is_last = stage == self._stage_order[-1]
        connector = "└" if is_last else "├"
        llm_tag = " [LLM]" if is_llm else ""
        _dim(f"{connector} {label}...{llm_tag}")
        self._current_stage = stage

    def _report_item(self, stage: str, current: int, total: int) -> None:
        is_last_stage = stage == self._stage_order[-1]
        prefix = "   " if is_last_stage else "│  "
        if stage == "segment" and current == total and total > 0:
            _dim(f"{prefix} → {total} episodes")
        elif stage == "abstract" and current == total and total > 0:
            _dim(f"{prefix} → {total} episodes abstracted")
        elif stage == "intent" and current == total and total > 0:
            _dim(f"{prefix} → {total} intents inferred")
        elif stage == "distill" and current == total and total > 0:
            _dim(f"{prefix} → {total} candidates")
        elif stage == "activate" and current == total and total > 0:
            _dim(f"{prefix} → done")


def install_learn_progress(ctx: "Context") -> None:
    """Wire up a verbose progress reporter for the learning pipeline."""
    reporter = VerboseLearnProgress()
    ctx.session.set_on_learn_progress(reporter)


def finish_learn_progress() -> None:
    """No-op — verbose reporter prints inline, no cleanup needed."""
    pass


class StopPhaseProgress:
    """Progress reporter for the recording stop / finalize phase.

    Uses ``print()`` with dim styling, matching the learn CLI output.
    Callback signature: ``(stage, current, total)``. Stages outside of
    ``_STOP_STAGE_CONFIG`` are rendered with their raw name (dot →
    space, capitalized).
    """

    def __init__(self) -> None:
        self._current_stage = ""
        self._last_vlm_report = 0

    def __call__(self, stage: str, current: int = 0, total: int = 0) -> None:
        if stage != self._current_stage:
            self._enter_stage(stage, current, total)
            return
        if stage == "extract.vlm" and total > 0:
            self._report_vlm(current, total)

    def _enter_stage(self, stage: str, current: int, total: int) -> None:
        label = _STOP_STAGE_CONFIG.get(stage, stage.replace(".", " ").capitalize())
        if stage == "extract" and total > 0:
            label = f"{label} ({total} frames)"
        elif stage == "extract.vlm" and total > 0:
            llm_tag = " [LLM]"
            _dim(f"├─ {label}...{llm_tag}")
            self._current_stage = stage
            self._last_vlm_report = 0
            self._report_vlm(current, total)
            return
        _dim(f"├─ {label}...")
        self._current_stage = stage

    def _report_vlm(self, current: int, total: int) -> None:
        if total <= 0:
            return
        if current <= 1 or current >= total or current - self._last_vlm_report >= 3:
            _dim(f"│  → {current}/{total} frame pairs")
            self._last_vlm_report = current


def install_stop_progress(ctx: "Context") -> StopPhaseProgress:
    """Wire up progress reporting for the recording stop / finalize phase."""
    reporter = StopPhaseProgress()
    ctx.imitation.stop_progress_callback = reporter
    return reporter


__all__ = [
    "StopPhaseProgress",
    "VerboseLearnProgress",
    "finish_learn_progress",
    "install_learn_progress",
    "install_stop_progress",
]
