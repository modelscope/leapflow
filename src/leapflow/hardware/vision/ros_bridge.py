# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sidecar: republish a LeapFlow frame channel as a ROS 2 image topic.

Owner mode ``leapflow``: the camera stays opened by the node's frame hub, and ROS
tools (RViz, Foxglove, a ROS perception node) get a ``sensor_msgs/CompressedImage``
topic without a second opener. The sidecar is an ordinary LHP client, so the node's
consent, hold and disconnect-release rules apply to it like to any viewer::

    source /opt/ros/<distro>/setup.bash
    PYTHONPATH=<leapflow>/src python3 -m leapflow.hardware.vision.ros_bridge \\
        --endpoint tcp://127.0.0.1:8767 --device sensor_realsense_435if \\
        --channel color.preview --token-file ~/.lhp-token

It runs in the interpreter that carries the distribution's rclpy (its system
Python), which is why it is a separate process rather than part of the node.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Sequence

from leapflow.hardware.transport import FrameTransport, TransportError
from leapflow.hardware.vision.ros import (
    DEFAULT_BRIDGE_FPS,
    BridgeStats,
    FrameTopicBridge,
    ros_image_publisher,
)

logger = logging.getLogger(__name__)

_TOKEN_ENV = "LEAPFLOW_LHP_TOKEN"


def default_topic(device_id: str, channel_id: str) -> str:
    """``/leapflow/<device>/<channel path>/compressed``, the image_transport layout."""
    path = "/".join(part for part in channel_id.replace(".", "/").split("/") if part)
    return f"/leapflow/{device_id}/{path}/compressed"


async def serve_bridge(
    transport: FrameTransport,
    node: Any,
    *,
    channel_id: str,
    topic: str,
    frame_id: str = "",
    max_width: int = 0,
    quality: int = 0,
    fps: float = DEFAULT_BRIDGE_FPS,
    count: int = 0,
    stop: asyncio.Event | None = None,
) -> BridgeStats:
    """Bridge one channel of an opened *transport* onto *topic* of an rclpy *node*.

    Returns when *stop* is set or *count* frames were published. Closing the
    transport (which releases the node's hold) stays with the caller.
    """
    publish = ros_image_publisher(node, topic, frame_id=frame_id)
    bridge = FrameTopicBridge(
        functools.partial(transport.read_frame, channel_id),
        publish,
        max_width=max_width,
        quality=quality,
        fps=fps,
    )
    logger.info("bridging %s -> %s at <= %g fps", channel_id, topic, fps)
    return await bridge.run(stop, limit=count)


async def _run(args: argparse.Namespace, token: str) -> int:
    # Function-local: these need the ROS distribution, which --help must not.
    try:
        import rclpy
    except ImportError:
        sys.stderr.write("error: rclpy is not importable; source /opt/ros/<distro>/setup.bash\n")
        return 2
    from leapflow.hardware.context import HardwareContext
    from leapflow.hardware.transports.lhp import LhpTransport

    transport = LhpTransport(
        {"endpoint": args.endpoint, "token": token, "remote_device_id": args.device}
    )
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(args.node_name)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - platform without loop signals
            pass
    stats = BridgeStats()
    try:
        await transport.open(HardwareContext(device_id=args.device))
        stats = await serve_bridge(
            transport,
            node,
            channel_id=args.channel,
            topic=args.topic or default_topic(args.device, args.channel),
            frame_id=args.frame_id,
            max_width=args.max_width,
            quality=args.quality,
            fps=args.fps,
            count=args.count,
            stop=stop,
        )
    except TransportError as exc:
        sys.stderr.write(f"error: {exc.failure_code or 'transport_error'}: {exc}\n")
        return 1
    finally:
        await transport.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(
        f"published={stats.published} repeated={stats.repeated} errors={stats.errors} "
        f"last_sequence={stats.last_sequence}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m leapflow.hardware.vision.ros_bridge``."""
    parser = argparse.ArgumentParser(
        prog="python -m leapflow.hardware.vision.ros_bridge",
        description="Republish a LeapFlow frame channel as a ROS 2 CompressedImage topic.",
    )
    parser.add_argument("--endpoint", required=True, help="LHP node, e.g. tcp://127.0.0.1:8767")
    parser.add_argument("--device", required=True, help="Device id on the node")
    parser.add_argument("--channel", required=True, help="Frame channel id, e.g. color.preview")
    parser.add_argument(
        "--topic", default="", help="Default: /leapflow/<device>/<channel>/compressed"
    )
    parser.add_argument("--frame-id", default="", help="Header frame_id (default: the channel id)")
    parser.add_argument("--max-width", type=int, default=640, help="Preview width (default 640)")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality (default 70)")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_BRIDGE_FPS, help="Upper bound on the republish rate"
    )
    parser.add_argument("--count", type=int, default=0, help="Stop after N frames (0: run)")
    parser.add_argument("--node-name", default="leapflow_frame_bridge", help="ROS node name")
    parser.add_argument(
        "--token-file", default="", help=f"File holding the LHP token (or env {_TOKEN_ENV})"
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    token = os.environ.get(_TOKEN_ENV, "")
    if args.token_file:
        token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    try:
        return asyncio.run(_run(args, token))
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        return 130


__all__ = ["default_topic", "main", "serve_bridge"]


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
