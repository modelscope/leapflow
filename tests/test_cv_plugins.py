# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for CV algorithm plugins and the CVProcessor Protocol (Fix D4).

Covers the default registry contents, Protocol conformance, and both the
dependency-present and dependency-missing (graceful-degrade) branches of the
built-in processors. The degrade branch is forced via monkeypatch so it runs
regardless of whether Pillow / cv2 are installed in the environment.

Hermetic: no network, no LLM. Optional native deps are never required.
"""

from __future__ import annotations

import io

import pytest

from leapflow.perception.cv_plugins import (
    OpticalFlowProcessor,
    PhashProcessor,
    build_default_cv_registry,
)
from leapflow.perception.cv_processor import CVProcessor, CVProcessorRegistry


class TestDefaultRegistry:
    """build_default_cv_registry() wiring contract."""

    def test_registers_phash_and_optical_flow(self) -> None:
        registry = build_default_cv_registry()
        assert isinstance(registry, CVProcessorRegistry)
        assert sorted(registry.list_available()) == ["optical_flow", "phash"]

    def test_lookup_returns_registered_instances(self) -> None:
        registry = build_default_cv_registry()
        assert isinstance(registry.get("phash"), PhashProcessor)
        assert isinstance(registry.get("optical_flow"), OpticalFlowProcessor)
        assert registry.get("does_not_exist") is None


class TestProtocolConformance:
    """Both processors must satisfy the runtime_checkable CVProcessor Protocol."""

    def test_phash_conforms(self) -> None:
        proc = PhashProcessor()
        assert isinstance(proc, CVProcessor)
        assert proc.processor_id == "phash"
        assert isinstance(proc.description, str) and proc.description

    def test_optical_flow_conforms(self) -> None:
        proc = OpticalFlowProcessor()
        assert isinstance(proc, CVProcessor)
        assert proc.processor_id == "optical_flow"
        assert isinstance(proc.description, str) and proc.description

    def test_registry_rejects_non_processor(self) -> None:
        registry = CVProcessorRegistry()
        with pytest.raises(TypeError):
            registry.register(object())  # type: ignore[arg-type]

    def test_registry_rejects_duplicate_id(self) -> None:
        registry = CVProcessorRegistry()
        registry.register(PhashProcessor())
        with pytest.raises(ValueError):
            registry.register(PhashProcessor())


class TestPhashProcessor:
    """PhashProcessor.process across dependency-present / -missing branches."""

    def test_graceful_degrade_when_pillow_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With Pillow unavailable, process() returns an error record, not a raise."""
        monkeypatch.setattr("leapflow.perception.cv.phash._HAS_PIL", False)
        result = PhashProcessor().process(b"anything", b"anything")
        assert result["processor"] == "phash"
        assert "error" in result
        assert "Pillow" in result["error"]

    def test_present_branch_computes_similarity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With Pillow available, identical images score a perfect similarity."""
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
        # Ensure the module's capability flag reflects a present dependency even
        # if a prior test patched it False within the same session.
        monkeypatch.setattr("leapflow.perception.cv.phash._HAS_PIL", True)

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(120, 60, 200)).save(buf, format="PNG")
        img_bytes = buf.getvalue()

        result = PhashProcessor().process(img_bytes, img_bytes, threshold=0.9)
        assert "error" not in result
        assert result["processor"] == "phash"
        assert result["distance"] == 0
        assert result["similarity"] == pytest.approx(1.0)
        assert result["is_similar"] is True


class TestOpticalFlowProcessor:
    """OpticalFlowProcessor.process across dependency-present / -missing branches."""

    def test_graceful_degrade_when_cv2_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With cv2 unavailable, process() returns neutral values, not a raise."""
        monkeypatch.setattr("leapflow.perception.cv.optical_flow._HAS_CV2", False)
        result = OpticalFlowProcessor().process(b"frame_a", b"frame_b")
        assert result["processor"] == "optical_flow"
        assert "error" not in result
        assert result["mean_magnitude"] == 0.0
        assert result["has_motion"] is False
        assert result["motion_type"] == "static"

    def test_present_branch_with_valid_frames(self) -> None:
        """With cv2 available, identical frames produce numeric, low-motion output."""
        cv2 = pytest.importorskip("cv2", reason="cv2 not installed")
        np = pytest.importorskip("numpy", reason="numpy not installed")

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", frame)
        assert ok
        frame_bytes = encoded.tobytes()

        result = OpticalFlowProcessor().process(frame_bytes, frame_bytes, threshold=1.0)
        assert result["processor"] == "optical_flow"
        assert "error" not in result
        assert isinstance(result["mean_magnitude"], float)
        # Identical frames → no motion above threshold.
        assert result["has_motion"] is False


class TestRegistryDispatch:
    """process_with routes to the named processor and raises on unknown ids."""

    def test_dispatch_returns_processor_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("leapflow.perception.cv.optical_flow._HAS_CV2", False)
        registry = build_default_cv_registry()
        result = registry.process_with("optical_flow", b"a", b"b")
        assert result["processor"] == "optical_flow"

    def test_dispatch_unknown_id_raises(self) -> None:
        registry = build_default_cv_registry()
        with pytest.raises(KeyError):
            registry.process_with("nope", b"a", b"b")
