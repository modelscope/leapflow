# Copyright (c) Alibaba, Inc. and its affiliates.
"""ROS2 transport: bridges ROS2 topics, services and actions to HCP.

Maps the Robot Operating System 2 communication primitives to the
six-method HardwareTransport contract.  Topic subscriptions become
read channels, topic publishers and service clients become write
channels, and action clients become long-running write operations.

rclpy is an optional dependency.  When unavailable, ``open()`` returns
``TransportStatus(connected=False)`` and no other method is callable.
The transport is discovered through the standard entry-point mechanism
and is available when ``pip install leapflow[robot-ros2]`` is installed.

Channel-to-ROS2-primitive mapping is declared in the YAML config produced
by ``control_binding_resolver._resolve_ros2``::

    topics:
      joint_states: "/robot/joint_states"
      joint_commands: "/robot/joint_commands"
    services:
      set_mode: "/robot/set_mode"
    actions:
      follow_trajectory: "/robot/follow_joint_trajectory"
    node_name: "leapflow_bridge"
    namespace: ""
    qos_depth: 10
    spin_period_s: 0.01

There is deliberately no top-level ``import rclpy``.  Every use is
function-local with a ``try/except ImportError``, so the module can be
imported, inspected, and rejected cleanly when rclpy is absent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

logger = logging.getLogger(__name__)

# Staleness threshold: if the last message on a subscribed topic is older
# than this many seconds, ``read()`` reports Quality.STALE.
_STALE_THRESHOLD_S = 2.0


class ROS2Transport:
    """Six-method transport over ROS2 topics, services and actions.

    All rclpy interaction is deferred to ``open()`` and runs inside
    ``asyncio.to_thread`` to avoid blocking the event loop.  A background
    spin task pumps callbacks so that subscriber messages arrive.
    """

    kind: str = "ros2"

    def __init__(
        self,
        topics: Mapping[str, str] | None = None,
        services: Mapping[str, str] | None = None,
        actions: Mapping[str, str] | None = None,
        node_name: str = "leapflow_bridge",
        namespace: str = "",
        qos_depth: int = 10,
        spin_period_s: float = 0.01,
    ) -> None:
        self._topics: dict[str, str] = dict(topics or {})
        self._services: dict[str, str] = dict(services or {})
        self._actions: dict[str, str] = dict(actions or {})
        self._node_name = node_name
        self._namespace = namespace
        self._qos_depth = qos_depth
        self._spin_period_s = spin_period_s

        # ROS2 runtime state -- populated by open().
        self._node: Any = None
        self._subscribers: dict[str, Any] = {}   # channel_id -> Subscription
        self._publishers: dict[str, Any] = {}    # channel_id -> Publisher
        self._service_clients: dict[str, Any] = {}  # channel_id -> Client
        self._latest_msgs: dict[str, Any] = {}   # channel_id -> latest message
        self._msg_timestamps: dict[str, float] = {}  # channel_id -> monotonic
        self._spin_task: asyncio.Task[None] | None = None
        self._connected = False
        self._context: HardwareContext | None = None
        self._sequence: dict[str, int] = {}
        # Set of channel_ids that map to command topics (for halt).
        self._cmd_channels: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Initialize rclpy, create Node, set up subscriptions and publishers.

        Uses ``asyncio.to_thread`` for rclpy.init() and node creation.
        Starts a background spin task for receiving messages.

        Channel mapping from *context*:

        - READ / READWRITE channels with a matching topic config →
          subscriber (populates ``_latest_msgs`` for ``read()``).
        - WRITE / READWRITE channels with a matching topic config →
          publisher (used by ``write()`` and ``halt()``).
        - Channels matching a service config → service client.
        """
        self._context = context

        try:
            rclpy = _require_rclpy()
        except TransportError:
            self._connected = False
            return TransportStatus(
                connected=False,
                halt_supported=False,
                detail="rclpy is not installed; install with: pip install leapflow[robot-ros2]",
            )

        # Init rclpy (idempotent -- rclpy.ok() checks first).
        def _init_node() -> Any:
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node(self._node_name, namespace=self._namespace or None)
            return node

        try:
            self._node = await asyncio.to_thread(_init_node)
        except Exception as exc:
            self._connected = False
            raise TransportError(
                f"failed to create ROS2 node: {exc}",
                failure_code="ros2_node_create_failed",
            ) from exc

        # Wire up subscriptions, publishers and service clients.
        self._setup_channels(context)
        self._connected = True

        # Start background spin loop so subscriber callbacks fire.
        loop = asyncio.get_running_loop()
        self._spin_task = loop.create_task(self._spin_loop(), name="ros2-spin")

        logger.info(
            "ROS2 transport opened: node=%s, %d subscribers, %d publishers, %d service clients",
            self._node_name,
            len(self._subscribers),
            len(self._publishers),
            len(self._service_clients),
        )
        return await self.probe()

    async def close(self) -> TransportStatus:
        """Destroy Node and cancel spin task.  Idempotent, never raises."""
        # Cancel the spin task first so no callbacks fire during teardown.
        if self._spin_task is not None:
            self._spin_task.cancel()
            try:
                await self._spin_task
            except (asyncio.CancelledError, Exception):
                pass
            self._spin_task = None

        node = self._node
        if node is not None:
            try:
                await asyncio.to_thread(node.destroy_node)
            except Exception:
                logger.debug("ROS2 node destroy raised; suppressed", exc_info=True)

        self._node = None
        self._subscribers.clear()
        self._publishers.clear()
        self._service_clients.clear()
        self._latest_msgs.clear()
        self._msg_timestamps.clear()
        self._cmd_channels.clear()
        self._connected = False

        return TransportStatus(
            connected=False,
            halt_supported=False,
            detail="ros2 transport closed",
        )

    # ── Data plane ────────────────────────────────────────────────────

    async def read(self, channel_id: str) -> Reading:
        """Return the latest cached message for the subscribed topic.

        Messages are received by the background spin task and cached in
        ``_latest_msgs``.  ``read()`` returns the most recent value
        without blocking.  Quality is STALE if no message has been
        received within ``_STALE_THRESHOLD_S``.
        """
        self._require_open(channel_id)
        if channel_id not in self._subscribers:
            raise TransportError(
                f"channel {channel_id!r} has no ROS2 subscription",
                failure_code="ros2_no_subscription",
            )

        raw_msg = self._latest_msgs.get(channel_id)
        if raw_msg is None:
            # No message received yet -- report as stale with None value.
            return Reading(
                device_id=self._device_id,
                channel_id=channel_id,
                value=None,
                sequence=self._next_seq(channel_id),
                quality=Quality.STALE.value,
            )

        # Determine staleness from monotonic timestamp.
        ts = self._msg_timestamps.get(channel_id, 0.0)
        elapsed = time.monotonic() - ts
        quality = Quality.OK.value if elapsed < _STALE_THRESHOLD_S else Quality.STALE.value

        value = self._parse_msg(channel_id, raw_msg)
        ch = self._context.channel(channel_id) if self._context else None

        return Reading(
            device_id=self._device_id,
            channel_id=channel_id,
            value=value,
            quantity=ch.quantity if ch else "",
            unit=ch.unit if ch else "",
            sequence=self._next_seq(channel_id),
            quality=quality,
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Publish to a topic or call a service.

        For topics: publish message, ``side_effect=COMMITTED``
        (fire-and-forget, ROS2 pub is non-blocking).

        For services: call and wait for response,
        ``side_effect`` based on the service response.
        """
        self._require_open(channel_id)

        # Service call path.
        if channel_id in self._service_clients:
            return await self._call_service(channel_id, value)

        # Topic publish path.
        pub = self._publishers.get(channel_id)
        if pub is None:
            raise TransportError(
                f"channel {channel_id!r} has no ROS2 publisher or service client",
                failure_code="ros2_no_publisher",
            )

        try:
            msg = self._build_write_msg(channel_id, value, pub)
            await asyncio.to_thread(pub.publish, msg)
        except TransportError:
            raise
        except Exception as exc:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"ROS2 publish failed: {exc}",
                failure_code="ros2_publish_failed",
            )

        return WriteOutcome(
            ok=True,
            side_effect_state=SIDE_EFFECT_COMMITTED,
            settled=True,
        )

    async def probe(self) -> TransportStatus:
        """Check Node health and topic availability."""
        if not self._connected or self._node is None:
            return TransportStatus(
                connected=False,
                halt_supported=False,
                detail="ros2 transport not connected",
            )

        # Check that the node is still alive.
        try:
            # rclpy nodes expose a handle that is None after destruction.
            alive = self._node.handle is not None
        except Exception:
            alive = self._connected

        has_cmd = len(self._cmd_channels) > 0
        return TransportStatus(
            connected=alive,
            halt_supported=has_cmd,
            detail=f"ros2 node {self._node_name!r}",
            latency_ms=self._spin_period_s * 1000.0,
            metadata={
                "node_name": self._node_name,
                "namespace": self._namespace,
                "subscribers": len(self._subscribers),
                "publishers": len(self._publishers),
                "service_clients": len(self._service_clients),
            },
        )

    async def halt(self) -> TransportStatus:
        """Publish zero-velocity commands to all command topics.

        Lock-free (HCP requirement).  Iterates ``_cmd_channels`` and
        publishes a zeroed message to each without acquiring any lock.
        """
        if not self._connected or self._node is None:
            return TransportStatus(
                connected=False,
                halt_supported=False,
                detail="halt: not connected",
            )

        if not self._cmd_channels:
            return TransportStatus(
                connected=True,
                halt_supported=False,
                detail="halt: no command topics registered",
            )

        errors: list[str] = []
        for cid in self._cmd_channels:
            pub = self._publishers.get(cid)
            if pub is None:
                continue
            try:
                msg = self._build_zero_msg(pub)
                # Publish in thread -- lock-free, no shared state modified.
                await asyncio.to_thread(pub.publish, msg)
            except Exception as exc:
                errors.append(f"{cid}: {exc}")

        detail = "halted" if not errors else f"halt partial: {'; '.join(errors)}"
        return TransportStatus(
            connected=self._connected,
            halt_supported=True,
            detail=detail,
        )

    # ── Internal: spin loop ───────────────────────────────────────────

    async def _spin_loop(self) -> None:
        """Background loop: ``rclpy.spin_once`` in a thread.

        Runs until cancelled.  Each iteration spins the node once with a
        bounded timeout so that cancellation is responsive.
        """
        rclpy = _require_rclpy()
        timeout = self._spin_period_s
        while self._connected and self._node is not None:
            try:
                await asyncio.to_thread(rclpy.spin_once, self._node, timeout_sec=timeout)
            except asyncio.CancelledError:
                return
            except Exception:
                # spin_once may raise if the node was destroyed concurrently.
                logger.debug("ROS2 spin_once raised; will retry", exc_info=True)
                await asyncio.sleep(timeout)

    # ── Internal: channel setup ───────────────────────────────────────

    def _setup_channels(self, context: HardwareContext) -> None:
        """Create ROS2 subscriptions, publishers and service clients.

        Matches declared channels against the topic/service/action config
        maps using channel_id as the logical key.
        """
        from functools import partial

        rclpy_qos = self._make_qos()

        for ch in context.channels:
            cid = ch.channel_id

            # Topic subscription for readable channels.
            topic = self._topics.get(cid)
            if topic and ch.is_readable:
                msg_type = self._resolve_topic_type(topic)
                if msg_type is not None:
                    sub = self._node.create_subscription(
                        msg_type,
                        topic,
                        partial(self._msg_callback, cid),
                        qos_profile=rclpy_qos,
                    )
                    self._subscribers[cid] = sub
                    logger.debug("ROS2 subscriber: %s -> %s", cid, topic)

            # Topic publisher for writable channels.
            if topic and ch.is_writable:
                msg_type = self._resolve_topic_type(topic)
                if msg_type is not None:
                    pub = self._node.create_publisher(msg_type, topic, qos_profile=rclpy_qos)
                    self._publishers[cid] = pub
                    self._cmd_channels.add(cid)
                    logger.debug("ROS2 publisher: %s -> %s", cid, topic)

            # Service client.
            svc_name = self._services.get(cid)
            if svc_name and ch.is_writable:
                srv_type = self._resolve_service_type(svc_name)
                if srv_type is not None:
                    cli = self._node.create_client(srv_type, svc_name)
                    self._service_clients[cid] = cli
                    logger.debug("ROS2 service client: %s -> %s", cid, svc_name)

    def _make_qos(self) -> Any:
        """Build a QoS profile from the configured depth."""
        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            return QoSProfile(
                depth=self._qos_depth,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
        except ImportError:
            return self._qos_depth  # fallback: an int works as depth-only

    def _resolve_topic_type(self, topic_name: str) -> Any:
        """Discover the message type for a topic at runtime.

        Uses ``rclpy`` utilities to query the ROS2 graph for the topic's
        type.  Falls back to ``std_msgs/msg/String`` when the topic is
        not yet advertised (the subscriber will still receive messages
        once a publisher appears).
        """
        node = self._node
        if node is None:
            return None
        try:
            # get_topic_names_and_types returns [(name, [type_strings])].
            topic_types = node.get_topic_names_and_types()
            for name, types in topic_types:
                if name == topic_name and types:
                    return _import_msg_type(types[0])
        except Exception:
            logger.debug(
                "Could not discover type for topic %r; using String",
                topic_name, exc_info=True,
            )

        # Fallback: use std_msgs/msg/String as a generic carrier.
        return _import_msg_type("std_msgs/msg/String")

    def _resolve_service_type(self, service_name: str) -> Any:
        """Discover the service type for a service at runtime.

        Uses ``rclpy`` utilities to query the ROS2 graph.  Returns None
        when the service type cannot be determined.
        """
        node = self._node
        if node is None:
            return None
        try:
            svc_types = node.get_service_names_and_types()
            for name, types in svc_types:
                if name == service_name and types:
                    return _import_msg_type(types[0])
        except Exception:
            logger.debug(
                "Could not discover type for service %r",
                service_name, exc_info=True,
            )
        return None

    # ── Internal: message handling ────────────────────────────────────

    def _msg_callback(self, channel_id: str, msg: Any) -> None:
        """Cache incoming message for ``read()``."""
        self._latest_msgs[channel_id] = msg
        self._msg_timestamps[channel_id] = time.monotonic()

    def _parse_msg(self, channel_id: str, msg: Any) -> Any:
        """Extract a Python-friendly value from a ROS2 message.

        Tries several strategies:
        1. ``sensor_msgs/JointState`` → dict of joint positions.
        2. Messages with a ``data`` attribute → the data value.
        3. Fall back to converting the message to a dict via ``get_fields_and_field_types``.
        """
        # Strategy 1: JointState-like messages.
        if hasattr(msg, "position") and hasattr(msg, "name"):
            return self._parse_joint_state_msg(msg)

        # Strategy 2: simple data field (Float64, String, Bool, etc.).
        if hasattr(msg, "data"):
            return msg.data

        # Strategy 3: generic conversion to dict.
        return _msg_to_dict(msg)

    def _parse_joint_state_msg(self, msg: Any) -> dict[str, float]:
        """Extract joint positions/velocities from a JointState message."""
        result: dict[str, float] = {}
        names = list(msg.name) if hasattr(msg, "name") else []
        positions = list(msg.position) if hasattr(msg, "position") else []
        velocities = list(msg.velocity) if hasattr(msg, "velocity") else []

        for i, name in enumerate(names):
            if i < len(positions):
                result[f"{name}.position"] = float(positions[i])
            if i < len(velocities):
                result[f"{name}.velocity"] = float(velocities[i])

        return result

    def _build_write_msg(self, channel_id: str, value: Any, publisher: Any) -> Any:
        """Build a ROS2 message from a channel value for publishing.

        Uses the publisher's message type.  For simple types (float, int,
        str, bool) wraps the value in a ``data`` field.  For dicts,
        attempts to set message fields from dict keys.
        """
        msg_type = publisher.msg_type
        msg = msg_type()

        if isinstance(value, Mapping):
            # Set fields from dict.
            for k, v in value.items():
                if hasattr(msg, k):
                    setattr(msg, k, v)
        elif hasattr(msg, "data"):
            msg.data = value
        else:
            # Attempt to set the first field.
            fields = _msg_fields(msg)
            if fields:
                setattr(msg, fields[0], value)

        return msg

    def _build_zero_msg(self, publisher: Any) -> Any:
        """Build a zero-valued message for halt.

        Creates a default-constructed message, which for numeric types
        (Float64, Twist, JointTrajectory) produces zeros.
        """
        return publisher.msg_type()

    def _build_joint_command_msg(self, value: Any) -> Any:
        """Build a JointTrajectoryPoint or similar from a channel value.

        Used for trajectory-style command topics.  *value* may be a list
        of joint positions or a dict mapping joint names to positions.
        """
        try:
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError:
            return value  # Cannot build typed message without ROS2 msg package.

        point = JointTrajectoryPoint()
        if isinstance(value, (list, tuple)):
            point.positions = [float(v) for v in value]
        elif isinstance(value, Mapping):
            point.positions = [float(v) for v in value.values()]
        return point

    # ── Internal: service calls ───────────────────────────────────────

    async def _call_service(self, channel_id: str, value: Any) -> WriteOutcome:
        """Call a ROS2 service and return a WriteOutcome."""
        cli = self._service_clients[channel_id]

        # Wait for service availability (bounded).
        try:
            ready = await asyncio.to_thread(cli.wait_for_service, timeout_sec=5.0)
            if not ready:
                return WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    error=f"service for {channel_id!r} not available after 5s",
                    failure_code="ros2_service_unavailable",
                )
        except Exception as exc:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"service wait failed: {exc}",
                failure_code="ros2_service_wait_failed",
            )

        # Build request.
        req = cli.srv_type.Request()
        if isinstance(value, Mapping):
            for k, v in value.items():
                if hasattr(req, k):
                    setattr(req, k, v)
        elif hasattr(req, "data"):
            req.data = value

        # Call the service.
        try:
            future = cli.call_async(req)
            response = await asyncio.to_thread(
                lambda: future.result(timeout_sec=10.0) if hasattr(future, "result") else None,
            )
        except Exception as exc:
            # The request may have reached the server.
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"ROS2 service call failed: {exc}",
                failure_code="ros2_service_call_failed",
            )

        # Parse response.
        success = True
        if hasattr(response, "success"):
            success = bool(response.success)
        elif hasattr(response, "result"):
            success = bool(response.result)

        return WriteOutcome(
            ok=success,
            side_effect_state=SIDE_EFFECT_COMMITTED if success else SIDE_EFFECT_UNKNOWN,
            settled=True,
            raw=_msg_to_dict(response) if response else {},
        )

    # ── Internal: helpers ─────────────────────────────────────────────

    @property
    def _device_id(self) -> str:
        return self._context.device_id if self._context else ""

    def _require_open(self, channel_id: str) -> None:
        """Raise when the transport is not connected."""
        if not self._connected or self._node is None:
            raise TransportError(
                f"ros2 transport for {channel_id!r} is not open",
                failure_code="transport_not_open",
            )

    def _next_seq(self, channel_id: str) -> int:
        """Advance and return the sequence counter for a channel."""
        seq = self._sequence.get(channel_id, 0) + 1
        self._sequence[channel_id] = seq
        return seq


# ── Module-level helpers ──────────────────────────────────────────────


def _require_rclpy() -> Any:
    """Return the ``rclpy`` module or raise TransportError."""
    try:
        import rclpy  # type: ignore[import-untyped]
        return rclpy
    except ImportError as exc:
        raise TransportError(
            "rclpy is not installed; install with: pip install leapflow[robot-ros2]",
            failure_code="rclpy_not_installed",
        ) from exc


def _import_msg_type(type_string: str) -> Any:
    """Import a ROS2 message type from its type string.

    Type strings look like ``"sensor_msgs/msg/JointState"`` or
    ``"std_srvs/srv/SetBool"``.  This maps to
    ``sensor_msgs.msg.JointState`` / ``std_srvs.srv.SetBool``.
    """
    import importlib

    parts = type_string.replace("/", ".")
    module_path, _, class_name = parts.rpartition(".")
    if not module_path:
        return None
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name, None)
    except ImportError:
        logger.debug("Cannot import ROS2 type %r", type_string, exc_info=True)
        return None


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a ROS2 message to a plain dict, best-effort."""
    if msg is None:
        return {}
    # rclpy messages expose get_fields_and_field_types().
    if hasattr(msg, "get_fields_and_field_types"):
        fields = msg.get_fields_and_field_types()
        return {k: getattr(msg, k, None) for k in fields}
    if hasattr(msg, "__slots__"):
        return {s: getattr(msg, s, None) for s in msg.__slots__}
    return {}


def _msg_fields(msg: Any) -> list[str]:
    """Return the field names of a ROS2 message."""
    if hasattr(msg, "get_fields_and_field_types"):
        return list(msg.get_fields_and_field_types().keys())
    if hasattr(msg, "__slots__"):
        return list(msg.__slots__)
    return []


# ── Factory ───────────────────────────────────────────────────────────


def build_transport(config: Mapping[str, Any]) -> ROS2Transport:
    """Factory for the transport registry.

    Config keys match what ``control_binding_resolver._resolve_ros2``
    produces:

    - ``topics``: dict[str, str] — channel_id → topic name
    - ``services``: dict[str, str] — channel_id → service name
    - ``actions``: dict[str, str] — channel_id → action name
    - ``node_name``: str — ROS2 node name (default ``leapflow_bridge``)
    - ``namespace``: str — ROS2 namespace (default ``""``)
    - ``qos_depth``: int — QoS history depth (default 10)
    - ``spin_period_s``: float — spin_once timeout (default 0.01)
    """
    return ROS2Transport(
        topics=dict(config.get("topics", {})) if isinstance(config.get("topics"), Mapping) else {},
        services=dict(config.get("services", {})) if isinstance(config.get("services"), Mapping) else {},
        actions=dict(config.get("actions", {})) if isinstance(config.get("actions"), Mapping) else {},
        node_name=str(config.get("node_name", "leapflow_bridge")),
        namespace=str(config.get("namespace", "")),
        qos_depth=int(config.get("qos_depth", 10)),
        spin_period_s=float(config.get("spin_period_s", 0.01)),
    )


__all__ = [
    "ROS2Transport",
    "build_transport",
]
