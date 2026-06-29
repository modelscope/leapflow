"""UI element detection — pluggable backend protocol.

Defines the interface for detecting common UI elements (buttons, text fields,
checkboxes, etc.) in screenshots. Includes a stub implementation that returns
empty results when no model backend is configured.

Real backends (CoreML, ONNX, YOLO) can be plugged in by implementing the
UIDetectionBackend protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class UIElement:
    """A detected UI element with bounding box and classification."""

    cls: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 0.0
    label: str = ""


@dataclass
class UIDetectionResult:
    """Result of UI element detection on a single frame."""

    elements: List[UIElement] = field(default_factory=list)
    has_text_field: bool = False
    has_dialog: bool = False
    button_count: int = 0

    @classmethod
    def from_elements(cls, elements: List[UIElement]) -> "UIDetectionResult":
        return cls(
            elements=elements,
            has_text_field=any(e.cls == "text_field" for e in elements),
            has_dialog=any(e.cls == "dialog" for e in elements),
            button_count=sum(1 for e in elements if e.cls == "button"),
        )


@runtime_checkable
class UIDetectionBackend(Protocol):
    """Protocol for pluggable UI element detection backends."""

    async def detect(self, frame_data: bytes) -> UIDetectionResult:
        """Run detection on frame data and return results."""
        ...


class StubUIDetector:
    """No-op detector that returns empty results.

    Used when no model backend is available. The system gracefully
    degrades — CV features are simply absent and VLM handles all
    perception directly.
    """

    async def detect(self, frame_data: bytes) -> UIDetectionResult:
        return UIDetectionResult()


class UIElementDetector:
    """Facade for UI element detection with backend delegation.

    Accepts any UIDetectionBackend implementation. Falls back to
    StubUIDetector if none is provided.
    """

    __slots__ = ("_backend",)

    def __init__(self, backend: Optional[UIDetectionBackend] = None) -> None:
        self._backend = backend or StubUIDetector()

    async def detect(self, frame_data: bytes) -> UIDetectionResult:
        """Detect UI elements in the given frame."""
        return await self._backend.detect(frame_data)
