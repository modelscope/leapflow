# Copyright (c) Alibaba, Inc. and its affiliates.
"""CV Algorithm Plugin Protocol + Registry.

Allows community-contributed computer vision algorithms to replace or augment
the built-in optical flow, phash, scene cut, text diff, and UI detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class CVProcessor(Protocol):
    """Protocol for pluggable CV analysis algorithms.

    Implementations provide a specific frame comparison or analysis capability
    (e.g., optical flow, structural similarity, scene detection).
    """

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor (e.g., 'optical_flow')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this processor does."""
        ...

    def process(self, frame_a: bytes, frame_b: bytes, **kwargs: Any) -> Dict[str, Any]:
        """Analyze two frames and return structured analysis results.

        Parameters
        ----------
        frame_a : bytes
            Raw image bytes of the first frame.
        frame_b : bytes
            Raw image bytes of the second frame.
        **kwargs
            Additional processor-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Structured analysis results. Schema is processor-defined but should
            include at minimum a 'score' or 'result' key.
        """
        ...


class CVProcessorRegistry:
    """Registry for CV processing algorithms.

    Provides registration, lookup, and dispatch for pluggable CV processors.
    Thread-safe for read operations; registration should happen during startup.
    """

    def __init__(self) -> None:
        self._processors: Dict[str, CVProcessor] = {}

    def register(self, processor: CVProcessor) -> None:
        """Register a CV processor instance.

        Raises
        ------
        TypeError
            If processor does not conform to CVProcessor protocol.
        ValueError
            If a processor with the same id is already registered.
        """
        if not isinstance(processor, CVProcessor):
            raise TypeError(f"Not a CVProcessor: {type(processor)}")
        pid = processor.processor_id
        if pid in self._processors:
            raise ValueError(f"Duplicate processor_id: {pid!r}")
        self._processors[pid] = processor

    def get(self, processor_id: str) -> Optional[CVProcessor]:
        """Retrieve a processor by id. Returns None if not found."""
        return self._processors.get(processor_id)

    def list_available(self) -> List[str]:
        """List all registered processor ids."""
        return list(self._processors.keys())

    def process_with(
        self, processor_id: str, frame_a: bytes, frame_b: bytes, **kwargs: Any
    ) -> Dict[str, Any]:
        """Dispatch processing to a named processor.

        Raises
        ------
        KeyError
            If no processor with the given id is registered.
        """
        proc = self._processors.get(processor_id)
        if proc is None:
            raise KeyError(f"No CV processor registered with id: {processor_id!r}")
        return proc.process(frame_a, frame_b, **kwargs)
