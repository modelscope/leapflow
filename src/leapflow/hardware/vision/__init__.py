# Copyright (c) Alibaba, Inc. and its affiliates.
"""Vendor-neutral vision plumbing: frame types and the per-camera FrameHub."""

from __future__ import annotations

from leapflow.hardware.vision.frames import (
    ACQUIRE_STREAMING,
    ACQUIRE_TRIGGERED,
    ALIGN_HARDWARE,
    ALIGN_NONE,
    ALIGN_SOFTWARE,
    SYNC_HARDWARE,
    SYNC_NONE,
    SYNC_TIMESTAMP,
    FrameCapabilities,
    FrameProducer,
    Frameset,
    ImagePlane,
)
from leapflow.hardware.vision.hub import (
    FAILURE_FRAME_TIMEOUT,
    FAILURE_SOURCE_CLOSED,
    FrameHub,
)

__all__ = [
    "ACQUIRE_STREAMING",
    "ACQUIRE_TRIGGERED",
    "ALIGN_HARDWARE",
    "ALIGN_NONE",
    "ALIGN_SOFTWARE",
    "FAILURE_FRAME_TIMEOUT",
    "FAILURE_SOURCE_CLOSED",
    "SYNC_HARDWARE",
    "SYNC_NONE",
    "SYNC_TIMESTAMP",
    "FrameCapabilities",
    "FrameHub",
    "FrameProducer",
    "Frameset",
    "ImagePlane",
]
