"""Pipeline observer protocol — provides observability for multi-phase learning pipelines.

Ensures that learning pipeline failures are visible rather than silently swallowed.
Core principle: "Failures must be observable — silent swallowing is the most dangerous failure mode."
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineObserver(Protocol):
    """Observer for multi-phase pipeline execution lifecycle."""

    def on_phase_start(self, phase: str) -> None:
        """Called when a pipeline phase begins."""
        ...

    def on_phase_success(self, phase: str, duration_s: float, metrics: Dict[str, Any]) -> None:
        """Called when a pipeline phase completes successfully."""
        ...

    def on_phase_failure(self, phase: str, error: Exception, duration_s: float) -> None:
        """Called when a pipeline phase fails."""
        ...

    def on_pipeline_complete(self, total_duration_s: float, phases_ok: int, phases_failed: int) -> None:
        """Called when the entire pipeline finishes."""
        ...


class StructuredPipelineLogger:
    """Default PipelineObserver implementation — structured logging with metrics.

    Upgrades silent logger.debug failures to logger.warning for visibility.
    Tracks per-phase timing and success/failure counts.
    """

    def __init__(self, pipeline_name: str = "session_end_learning") -> None:
        self._pipeline_name = pipeline_name
        self._phase_starts: Dict[str, float] = {}
        self._results: list[Dict[str, Any]] = []

    def on_phase_start(self, phase: str) -> None:
        self._phase_starts[phase] = time.perf_counter()
        logger.debug("%s.%s: started", self._pipeline_name, phase)

    def on_phase_success(self, phase: str, duration_s: float, metrics: Dict[str, Any]) -> None:
        self._results.append({"phase": phase, "ok": True, "duration_s": duration_s, **metrics})
        logger.info(
            "%s.%s: completed in %.2fs | %s",
            self._pipeline_name, phase, duration_s,
            " ".join(f"{k}={v}" for k, v in metrics.items()) or "no metrics",
        )

    def on_phase_failure(self, phase: str, error: Exception, duration_s: float) -> None:
        self._results.append({"phase": phase, "ok": False, "duration_s": duration_s, "error": str(error)})
        logger.warning(
            "%s.%s: FAILED after %.2fs — %s: %s",
            self._pipeline_name, phase, duration_s,
            type(error).__name__, error,
            exc_info=True,
        )

    def on_pipeline_complete(self, total_duration_s: float, phases_ok: int, phases_failed: int) -> None:
        logger.info(
            "%s: pipeline complete in %.2fs | phases_ok=%d phases_failed=%d",
            self._pipeline_name, total_duration_s, phases_ok, phases_failed,
        )

    @property
    def results(self) -> list[Dict[str, Any]]:
        """Access phase results for testing or aggregation."""
        return self._results
