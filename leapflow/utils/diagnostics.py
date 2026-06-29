"""Lightweight pipeline tracing and structured diagnostics.

Provides non-intrusive instrumentation for multi-stage processing pipelines.

Usage:
    tracer = PipelineTracer("causal_fusion")
    with tracer.stage("denoise"):
        events = denoiser.process(raw_events)
        tracer.metric("input_count", len(raw_events))
        tracer.metric("output_count", len(events))

    logger.info(tracer.summary_line())
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageRecord:
    """Record of a single pipeline stage execution."""

    name: str
    elapsed_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class PipelineTracer:
    """Lightweight pipeline tracer for structured logging.

    Non-intrusive: all operations are no-ops when disabled (zero overhead).
    Thread-safe for single-pipeline use (one tracer per pipeline invocation).

    Args:
        pipeline_name: Identifier for the pipeline being traced.
        enabled: If False, all operations become no-ops.
        log_level: Logging level for automatic stage logging (default DEBUG).
        logger_override: Custom logger; falls back to module logger.
    """

    def __init__(
        self,
        pipeline_name: str,
        *,
        enabled: bool = True,
        log_level: int = logging.DEBUG,
        logger_override: Optional[logging.Logger] = None,
    ) -> None:
        self._name = pipeline_name
        self._enabled = enabled
        self._log_level = log_level
        self._logger = logger_override or logger
        self._stages: List[StageRecord] = []
        self._current: Optional[StageRecord] = None
        self._start_time: float = time.perf_counter()
        self._global_metrics: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager for timing a pipeline stage.

        Usage:
            with tracer.stage("alignment"):
                result = aligner.align(data)
        """
        if not self._enabled:
            yield
            return

        record = StageRecord(name=name)
        self._current = record
        t0 = time.perf_counter()
        try:
            yield
        finally:
            record.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._stages.append(record)
            self._current = None
            self._logger.log(
                self._log_level,
                "[%s] %s: elapsed_ms=%.2f %s",
                self._name,
                name,
                record.elapsed_ms,
                self._format_metrics(record.metrics),
            )

    def metric(self, key: str, value: Any) -> None:
        """Record a metric for the current stage (or global if outside a stage)."""
        if not self._enabled:
            return
        if self._current is not None:
            self._current.metrics[key] = value
        else:
            self._global_metrics[key] = value

    def total_elapsed_ms(self) -> float:
        """Total elapsed time since tracer creation."""
        return (time.perf_counter() - self._start_time) * 1000.0

    def summary(self) -> Dict[str, Any]:
        """Full structured summary of pipeline execution."""
        return {
            "pipeline": self._name,
            "total_ms": round(self.total_elapsed_ms(), 2),
            "stages": [
                {
                    "name": s.name,
                    "elapsed_ms": round(s.elapsed_ms, 2),
                    **s.metrics,
                }
                for s in self._stages
            ],
            **self._global_metrics,
        }

    def summary_line(self) -> str:
        """One-line summary for logging.

        Format: ``[pipeline] total=Xms | stage1=Yms stage2=Zms | key=val``
        """
        if not self._enabled:
            return f"[{self._name}] tracing disabled"

        stages_str = " ".join(
            f"{s.name}={s.elapsed_ms:.1f}ms" for s in self._stages
        )
        metrics_str = " ".join(
            f"{k}={v}" for k, v in self._global_metrics.items()
        )
        parts = [f"[{self._name}] total={self.total_elapsed_ms():.1f}ms"]
        if stages_str:
            parts.append(stages_str)
        if metrics_str:
            parts.append(metrics_str)
        return " | ".join(parts)

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        if not metrics:
            return ""
        return "| " + " ".join(f"{k}={v}" for k, v in metrics.items())


__all__ = ["PipelineTracer", "StageRecord"]
