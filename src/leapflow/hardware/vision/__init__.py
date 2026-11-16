# Copyright (c) Alibaba, Inc. and its affiliates.
"""Vendor-neutral vision plumbing: frame types, the per-camera FrameHub, capture groups,
the pump-health tracker that turns hub stats into hardware signals, and the ROS 2
image-topic adapters for either owner mode (``ros`` subscribes, ``ros_bridge`` publishes)."""

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
    accepts_keyword,
)
from leapflow.hardware.vision.group import (
    DEFAULT_SYNC_ATTEMPTS,
    DEFAULT_SYNC_TOLERANCE_MS,
    CaptureGroup,
    GroupCapture,
    GroupMember,
    bounded_capture,
)
from leapflow.hardware.vision.health import (
    FRAME_HUB_CHANNEL,
    FrameHealthTracker,
    fold_pump_stats,
)
from leapflow.hardware.vision.hub import (
    FAILURE_FRAME_TIMEOUT,
    FAILURE_SOURCE_CLOSED,
    FrameHub,
)
from leapflow.hardware.vision.ros import (
    FrameTopicBridge,
    RosImageProducer,
    TopicHubs,
)

__all__ = [
    "ACQUIRE_STREAMING",
    "ACQUIRE_TRIGGERED",
    "ALIGN_HARDWARE",
    "ALIGN_NONE",
    "ALIGN_SOFTWARE",
    "DEFAULT_SYNC_ATTEMPTS",
    "DEFAULT_SYNC_TOLERANCE_MS",
    "FAILURE_FRAME_TIMEOUT",
    "FAILURE_SOURCE_CLOSED",
    "FRAME_HUB_CHANNEL",
    "SYNC_HARDWARE",
    "SYNC_NONE",
    "SYNC_TIMESTAMP",
    "CaptureGroup",
    "FrameCapabilities",
    "FrameHealthTracker",
    "FrameHub",
    "FrameProducer",
    "FrameTopicBridge",
    "Frameset",
    "GroupCapture",
    "GroupMember",
    "ImagePlane",
    "RosImageProducer",
    "TopicHubs",
    "accepts_keyword",
    "bounded_capture",
    "fold_pump_stats",
]
