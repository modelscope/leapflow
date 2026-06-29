"""Feature extraction — pluggable OCR, UI detection, and embedding backends."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.perception.cv.ui_detect import UIDetectionResult, UIElementDetector
from leapflow.perception.types import FrameFeatures, Keyframe


@runtime_checkable
class OCRBackend(Protocol):
    """Protocol for OCR engines (Apple Vision, PaddleOCR, etc.)."""

    async def detect(self, frame_data: bytes) -> Dict[str, Any]:
        """Returns {"regions": [{"text": str, "bbox": tuple, "confidence": float}], "full_text": str}"""
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for image embedding models (CLIP, DINOv2, etc.)."""

    async def encode(self, frame_data: bytes) -> List[float]:
        """Returns embedding vector (e.g., 512-dim for CLIP ViT-B/32)."""
        ...


class StubOCR:
    """No-op OCR that returns empty results."""

    async def detect(self, frame_data: bytes) -> Dict[str, Any]:
        return {"regions": [], "full_text": ""}


class StubEmbedding:
    """No-op embedding that returns zeros."""

    async def encode(self, frame_data: bytes) -> List[float]:
        return []


class FeatureExtractor:
    """Composite feature extractor running OCR, UI detection, and embedding in parallel.

    All backends are pluggable protocols. When a backend is unavailable,
    the corresponding feature field is simply empty — downstream consumers
    handle the absence gracefully.
    """

    __slots__ = ("_ocr", "_ui_detector", "_embedding")

    def __init__(
        self,
        ocr: Optional[OCRBackend] = None,
        ui_detector: Optional[UIElementDetector] = None,
        embedding: Optional[EmbeddingBackend] = None,
    ) -> None:
        self._ocr = ocr or StubOCR()
        self._ui_detector = ui_detector or UIElementDetector()
        self._embedding = embedding or StubEmbedding()

    async def extract(self, keyframe: Keyframe) -> FrameFeatures:
        """Extract all features from a keyframe in parallel."""
        ocr_task = asyncio.create_task(self._ocr.detect(keyframe.image))
        ui_task = asyncio.create_task(self._ui_detector.detect(keyframe.image))
        embed_task = asyncio.create_task(self._embedding.encode(keyframe.image))

        ocr_result, ui_result, embedding = await asyncio.gather(
            ocr_task, ui_task, embed_task
        )

        text_regions = ocr_result.get("regions", [])
        full_text = ocr_result.get("full_text", "")
        detected_app = self._infer_app_from_titlebar(text_regions)

        return FrameFeatures(
            frame_ref=keyframe.ref,
            timestamp=keyframe.timestamp,
            text_regions=text_regions,
            full_text=full_text,
            ui_elements=[{"cls": e.cls, "bbox": e.bbox, "label": e.label}
                         for e in ui_result.elements],
            embedding=embedding if embedding else None,
            detected_app=detected_app,
            focus_region=self._detect_focus(ui_result),
        )

    async def extract_batch(self, keyframes: List[Keyframe], concurrency: int = 5) -> List[FrameFeatures]:
        """Extract features from multiple keyframes with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(kf: Keyframe) -> FrameFeatures:
            async with semaphore:
                return await self.extract(kf)

        return await asyncio.gather(*[_bounded(kf) for kf in keyframes])

    @staticmethod
    def _infer_app_from_titlebar(text_regions: List[Dict[str, Any]]) -> str:
        """Attempt to identify app from top-of-screen text regions."""
        if not text_regions:
            return ""
        # Heuristic: look for text in the top 10% of the screen (title bar area)
        top_regions = [r for r in text_regions if r.get("bbox") and r["bbox"][1] < 50]
        if top_regions:
            return top_regions[0].get("text", "")
        return ""

    @staticmethod
    def _detect_focus(ui_result: UIDetectionResult) -> Any:
        """Identify the focused UI element (has cursor or selected state)."""
        for e in ui_result.elements:
            if e.cls == "text_field":
                return e.bbox
        return None
