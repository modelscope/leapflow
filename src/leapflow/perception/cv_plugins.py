# Copyright (c) Alibaba, Inc. and its affiliates.
"""CV algorithm plugins — wraps existing ``perception/cv/`` algorithms as
``CVProcessor`` instances.

Demonstrates how existing CV functionality is packaged as pluggable
processors that can be discovered, replaced, or augmented by the community.
Each processor is a thin adapter over the underlying algorithm and returns
a structured, JSON-serialisable result dict.

The wrappers degrade gracefully: if an underlying dependency (Pillow for
pHash, cv2/numpy for optical flow) is missing, ``process()`` returns an
error record instead of raising, so a registry that mixes optional
processors stays usable.
"""

from __future__ import annotations

from typing import Any, Dict

from leapflow.perception.cv_processor import CVProcessor, CVProcessorRegistry


class PhashProcessor:
    """Perceptual-hash similarity processor.

    Wraps ``perception.cv.phash.phash_64`` + ``hamming_distance`` and reports
    a normalised similarity score in ``[0, 1]``.
    """

    @property
    def processor_id(self) -> str:
        return "phash"

    @property
    def description(self) -> str:
        return "Perceptual hash (pHash-64) for image similarity detection"

    def process(self, frame_a: bytes, frame_b: bytes, **kwargs: Any) -> Dict[str, Any]:
        """Compute pHash similarity between two frames.

        ``kwargs`` accepts ``threshold`` (default ``0.9``) used to set the
        boolean ``is_similar`` flag on the returned record.
        """
        try:
            from leapflow.perception.cv.phash import hamming_distance, phash_64

            hash_a = phash_64(frame_a)
            hash_b = phash_64(frame_b)
            distance = hamming_distance(hash_a, hash_b)
            # phash_64 returns 8 bytes → 64 bits of hash space.
            similarity = 1.0 - (distance / 64.0)
            threshold = float(kwargs.get("threshold", 0.9))
            return {
                "processor": "phash",
                "hash_a": hash_a.hex(),
                "hash_b": hash_b.hex(),
                "distance": distance,
                "similarity": similarity,
                "is_similar": similarity >= threshold,
            }
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            # RuntimeError covers phash's "Pillow is required" degradation path.
            return {"processor": "phash", "error": str(exc)}


class OpticalFlowProcessor:
    """Optical-flow motion classification processor.

    Wraps ``perception.cv.optical_flow.OpticalFlowAnalyzer`` and reports the
    magnitude summary plus the classified motion type.
    """

    def __init__(self) -> None:
        self._analyzer: Any = None  # lazy-initialised on first process() call

    @property
    def processor_id(self) -> str:
        return "optical_flow"

    @property
    def description(self) -> str:
        return "Farneback optical flow analysis for motion/change detection between frames"

    def process(self, frame_a: bytes, frame_b: bytes, **kwargs: Any) -> Dict[str, Any]:
        """Compute optical flow between two frames.

        ``kwargs`` accepts ``threshold`` (default ``1.0``) used to set the
        boolean ``has_motion`` flag on the returned record.
        """
        try:
            if self._analyzer is None:
                from leapflow.perception.cv.optical_flow import OpticalFlowAnalyzer

                self._analyzer = OpticalFlowAnalyzer()

            analysis = self._analyzer.analyze(frame_a, frame_b)
            threshold = float(kwargs.get("threshold", 1.0))
            return {
                "processor": "optical_flow",
                "mean_magnitude": analysis.mean_magnitude,
                "max_magnitude": analysis.max_magnitude,
                "motion_type": analysis.motion_type,
                "is_scroll": analysis.is_scroll,
                "scroll_direction": analysis.scroll_direction,
                "localized_regions": list(analysis.localized_regions),
                "has_motion": analysis.mean_magnitude > threshold,
            }
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            return {"processor": "optical_flow", "error": str(exc)}


def build_default_cv_registry() -> CVProcessorRegistry:
    """Create a ``CVProcessorRegistry`` prepopulated with built-in algorithms."""
    registry = CVProcessorRegistry()
    registry.register(PhashProcessor())
    registry.register(OpticalFlowProcessor())
    return registry


__all__ = [
    "PhashProcessor",
    "OpticalFlowProcessor",
    "build_default_cv_registry",
]


# Runtime Protocol sanity check: fail loudly at import if these ever drift out
# of conformance with the CVProcessor Protocol.
assert isinstance(PhashProcessor(), CVProcessor)
assert isinstance(OpticalFlowProcessor(), CVProcessor)
