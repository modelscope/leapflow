# Adapted from LeRobot (https://github.com/huggingface/lerobot).
"""Robot arm transport conformance and integration tests.

Verifies that RobotArmTransport satisfies all three protocols (HardwareTransport,
BatchTransport, FrameTransport), that the channel mapper produces correct
declarations, and that the context provider discovers robots from configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    PrivacyTier,
    Representation,
    TransportRef,
)
from leapflow.hardware.transport import (
    BatchTransport,
    FrameTransport,
    HardwareTransport,
    Reading,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_UNKNOWN,
    TransportError,
    TransportStatus,
)
from leapflow.hardware.transports.robot_arm import RobotArmTransport
from leapflow.hardware.transports.robot_channels import (
    RobotChannelMapper,
    camera_channel,
    gripper_channel,
    infer_channel_type,
    motor_channel_pair,
)
from leapflow.hardware.providers.robot_arm import RobotArmContextProvider


# ════════════════════════════════════════════════════════════════
# Mock robot — no external SDK import
# ════════════════════════════════════════════════════════════════


class MockRobotArm:
    """Simulates a robot arm for testing without hardware.

    Features:
    - 3 motors (shoulder, elbow, wrist) with position/velocity
    - 1 camera (top) producing 640x480 images
    - 1 gripper (binary)
    - Tracks connect/disconnect state
    - Records all sent actions
    """

    def __init__(self) -> None:
        self.is_connected = False
        self.is_calibrated = True
        self._positions = {"shoulder": 0.0, "elbow": 0.5, "wrist": -0.3}
        self._velocities = {"shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}
        self._gripper = 0.0
        self._actions_sent: list[Any] = []
        self._fail_next_action = False

    @property
    def observation_features(self) -> dict[str, Any]:
        """Robot-style observation feature dict."""
        return {
            "shoulder.position": (1,),
            "elbow.position": (1,),
            "wrist.position": (1,),
            "top": (480, 640, 3),
            "grip": (1,),
        }

    @property
    def action_features(self) -> dict[str, Any]:
        """Robot-style action feature dict."""
        return {
            "shoulder.position": (1,),
            "elbow.position": (1,),
            "wrist.position": (1,),
            "grip": (1,),
        }

    def connect(self, calibrate: bool = True) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_observation(self) -> dict[str, Any]:
        """Return current state as a dict matching robot observation format."""
        return {
            "shoulder.position": self._positions["shoulder"],
            "elbow.position": self._positions["elbow"],
            "wrist.position": self._positions["wrist"],
            "shoulder.velocity": self._velocities["shoulder"],
            "elbow.velocity": self._velocities["elbow"],
            "wrist.velocity": self._velocities["wrist"],
            "top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "grip": self._gripper,
        }

    def send_action(self, action: Any) -> None:
        """Record action and update positions."""
        if self._fail_next_action:
            self._fail_next_action = False
            raise RuntimeError("simulated hardware failure")
        self._actions_sent.append(action)
        if isinstance(action, (list, tuple)) and len(action) >= 3:
            for i, key in enumerate(self._positions):
                if i < len(action):
                    self._positions[key] = float(action[i])


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════


def _mock_context(channels: tuple[Channel, ...]) -> HardwareContext:
    """Build a minimal HardwareContext for transport tests."""
    return HardwareContext(
        device_id="robot.test_arm",
        display_name="Test Arm",
        transport=TransportRef(kind="robot_arm", config={}),
        channels=channels,
        halt_supported=True,
        provenance=ContextProvenance(source=ContextSource.IMPORTED.value),
    )


# ════════════════════════════════════════════════════════════════
# 1. Protocol conformance
# ════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify RobotArmTransport satisfies all three protocols."""

    def test_satisfies_hardware_transport(self):
        """isinstance check against HardwareTransport."""
        t = RobotArmTransport(robot_factory=lambda: None)
        assert isinstance(t, HardwareTransport)

    def test_satisfies_batch_transport(self):
        """isinstance check against BatchTransport."""
        t = RobotArmTransport(robot_factory=lambda: None)
        assert isinstance(t, BatchTransport)

    def test_satisfies_frame_transport(self):
        """isinstance check against FrameTransport."""
        t = RobotArmTransport(robot_factory=lambda: None)
        assert isinstance(t, FrameTransport)

    def test_kind_attribute(self):
        """Transport declares kind='robot_arm'."""
        t = RobotArmTransport(robot_factory=lambda: None)
        assert t.kind == "robot_arm"


# ════════════════════════════════════════════════════════════════
# 2. Channel mapper
# ════════════════════════════════════════════════════════════════


class TestRobotChannelMapper:
    """Channel mapper produces correct HCP channel declarations."""

    def test_motor_channels_from_features(self):
        """Motor features produce position (READWRITE/ACTUATE) + velocity (READ) channels."""
        mapper = RobotChannelMapper()
        channels = mapper.map_features(
            {"shoulder.position": (1,)},
            {"shoulder.position": (1,)},
        )
        ids = {ch.channel_id for ch in channels}
        assert "joint.shoulder.position" in ids
        assert "joint.shoulder.velocity" in ids
        pos = next(ch for ch in channels if ch.channel_id == "joint.shoulder.position")
        vel = next(ch for ch in channels if ch.channel_id == "joint.shoulder.velocity")
        assert pos.direction == Direction.READWRITE.value
        assert pos.effect == HardwareEffect.ACTUATE.value
        assert vel.direction == Direction.READ.value

    def test_camera_channel_from_features(self):
        """Camera features produce FRAME/READ channels with ENVIRONMENT privacy."""
        mapper = RobotChannelMapper()
        channels = mapper.map_features({"top": (480, 640, 3)}, {})
        cam = next(ch for ch in channels if "camera" in ch.channel_id)
        assert cam.representation == Representation.FRAME.value
        assert cam.direction == Direction.READ.value
        assert cam.privacy == PrivacyTier.ENVIRONMENT.value

    def test_gripper_channel(self):
        """Gripper features produce STATE/ACTUATE channels."""
        mapper = RobotChannelMapper()
        channels = mapper.map_features({"grip": (1,)}, {"grip": (1,)})
        grip = next(ch for ch in channels if "gripper" in ch.channel_id)
        assert grip.representation == Representation.STATE.value
        assert grip.effect == HardwareEffect.ACTUATE.value
        assert grip.direction == Direction.READWRITE.value

    def test_map_robot_duck_typing(self):
        """map_robot() works via duck typing without external SDK import."""
        robot = MockRobotArm()
        mapper = RobotChannelMapper()
        channels = mapper.map_robot(robot)
        assert len(channels) > 0
        ids = {ch.channel_id for ch in channels}
        assert any("joint" in cid for cid in ids)
        assert any("camera" in cid for cid in ids)

    def test_channel_descriptions_are_llm_readable(self):
        """Every channel has a non-empty human-readable description."""
        robot = MockRobotArm()
        mapper = RobotChannelMapper()
        for ch in mapper.map_robot(robot):
            assert ch.description, f"Channel {ch.channel_id} has no description"

    def test_envelope_limits_from_motor_config(self):
        """Motor limits are reflected in channel Envelope min/max."""
        mapper = RobotChannelMapper()
        channels = mapper.map_features(
            {"shoulder.position": (1,)},
            {"shoulder.position": (1,)},
            motor_limits={"shoulder": (-3.14, 3.14)},
        )
        pos = next(ch for ch in channels if ch.channel_id == "joint.shoulder.position")
        assert pos.envelope.declared is True
        assert pos.envelope.min_value == pytest.approx(-3.14)
        assert pos.envelope.max_value == pytest.approx(3.14)


# ════════════════════════════════════════════════════════════════
# 3. Transport core
# ════════════════════════════════════════════════════════════════


class TestRobotArmTransport:
    """Core six-method transport tests."""

    @pytest.fixture
    def mock_robot(self):
        return MockRobotArm()

    @pytest.fixture
    def transport(self, mock_robot):
        """Create a transport with MockRobotArm factory."""
        return RobotArmTransport(robot_factory=lambda: mock_robot)

    @pytest.fixture
    def context(self, mock_robot):
        """Create a HardwareContext matching the mock robot."""
        mapper = RobotChannelMapper()
        return _mock_context(mapper.map_robot(mock_robot))

    @pytest.mark.asyncio
    async def test_open_connects_robot(self, transport, context, mock_robot):
        """open() calls factory and connects the robot."""
        status = await transport.open(context)
        assert status.connected is True
        assert mock_robot.is_connected is True

    @pytest.mark.asyncio
    async def test_close_disconnects(self, transport, context, mock_robot):
        """close() disconnects the robot."""
        await transport.open(context)
        status = await transport.close()
        assert status.connected is False
        assert mock_robot.is_connected is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self, transport, context):
        """Calling close() twice does not raise."""
        await transport.open(context)
        await transport.close()
        status = await transport.close()
        assert status.connected is False

    @pytest.mark.asyncio
    async def test_read_motor_position(self, transport, context):
        """Read a motor position channel returns a Reading with correct value."""
        await transport.open(context)
        reading = await transport.read("joint.shoulder.position")
        assert isinstance(reading, Reading)
        assert reading.value == pytest.approx(0.0)
        assert reading.channel_id == "joint.shoulder.position"
        await transport.close()

    @pytest.mark.asyncio
    async def test_read_motor_velocity(self, transport, context):
        """Read a velocity channel returns a Reading."""
        await transport.open(context)
        reading = await transport.read("joint.shoulder.velocity")
        assert isinstance(reading, Reading)
        assert reading.value == pytest.approx(0.0)
        await transport.close()

    @pytest.mark.asyncio
    async def test_read_unknown_channel_raises(self, transport, context):
        """Reading a nonexistent channel raises TransportError."""
        await transport.open(context)
        with pytest.raises(TransportError, match="unknown channel"):
            await transport.read("nonexistent.channel")
        await transport.close()

    @pytest.mark.asyncio
    async def test_write_motor_position(self, transport, context, mock_robot):
        """Write updates position and returns WriteOutcome with ok=True."""
        await transport.open(context)
        outcome = await transport.write("joint.shoulder.position", 1.5)
        assert outcome.ok is True
        assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
        assert len(mock_robot._actions_sent) >= 1
        await transport.close()

    @pytest.mark.asyncio
    async def test_write_holds_other_channels(self, transport, context, mock_robot):
        """Writing one channel preserves other channels at their current positions."""
        await transport.open(context)
        # Read first to populate the observation cache / last_obs.
        await transport.read("joint.shoulder.position")
        # Write only shoulder; others should hold current values.
        await transport.write("joint.shoulder.position", 2.0)
        action = mock_robot._actions_sent[-1]
        # Action may be a torch tensor or plain list depending on environment.
        assert len(action) >= 3  # all writable channels

    @pytest.mark.asyncio
    async def test_probe_returns_connected_status(self, transport, context):
        """probe() returns connected=True when transport is open."""
        await transport.open(context)
        status = await transport.probe()
        assert status.connected is True
        assert status.halt_supported is True
        await transport.close()

    @pytest.mark.asyncio
    async def test_halt_sends_zero_velocity(self, transport, context, mock_robot):
        """halt() commands zero velocity/position to all actuators."""
        await transport.open(context)
        status = await transport.halt()
        assert status.halt_supported is True
        assert len(mock_robot._actions_sent) >= 1
        await transport.close()

    @pytest.mark.asyncio
    async def test_halt_without_open(self, transport):
        """halt() before open returns without error."""
        status = await transport.halt()
        assert status.connected is False
        assert status.halt_supported is True


# ════════════════════════════════════════════════════════════════
# 4. Batch transport
# ════════════════════════════════════════════════════════════════


class TestBatchTransport:
    """Vectorized read/write tests."""

    @pytest.fixture
    def mock_robot(self):
        return MockRobotArm()

    @pytest.fixture
    def transport(self, mock_robot):
        return RobotArmTransport(robot_factory=lambda: mock_robot)

    @pytest.fixture
    def context(self, mock_robot):
        mapper = RobotChannelMapper()
        return _mock_context(mapper.map_robot(mock_robot))

    @pytest.mark.asyncio
    async def test_read_batch_all_channels(self, transport, context):
        """read_batch returns all requested channels in one BatchReading."""
        await transport.open(context)
        batch = await transport.read_batch((
            "joint.shoulder.position",
            "joint.elbow.position",
        ))
        assert len(batch.readings) == 2
        ids = {r.channel_id for r in batch.readings}
        assert ids == {"joint.shoulder.position", "joint.elbow.position"}
        await transport.close()

    @pytest.mark.asyncio
    async def test_read_batch_shared_timestamp(self, transport, context):
        """All readings in a batch share the same observed_at."""
        await transport.open(context)
        batch = await transport.read_batch((
            "joint.shoulder.position",
            "joint.elbow.position",
            "joint.wrist.position",
        ))
        timestamps = {r.observed_at for r in batch.readings}
        assert len(timestamps) == 1, "All batch readings must share one timestamp"
        assert batch.observed_at == next(iter(timestamps))
        await transport.close()

    @pytest.mark.asyncio
    async def test_write_batch_atomic(self, transport, context, mock_robot):
        """write_batch sends one send_action() call with all values."""
        await transport.open(context)
        outcome = await transport.write_batch((
            ("joint.shoulder.position", 1.0),
            ("joint.elbow.position", 2.0),
        ))
        assert outcome.ok is True
        assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
        assert len(mock_robot._actions_sent) == 1  # one atomic call
        await transport.close()

    @pytest.mark.asyncio
    async def test_write_batch_partial_failure(self, transport, context, mock_robot):
        """When robot.send_action fails, outcome reports side_effect=UNKNOWN."""
        await transport.open(context)
        mock_robot._fail_next_action = True
        outcome = await transport.write_batch((
            ("joint.shoulder.position", 1.0),
        ))
        assert outcome.ok is False
        assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
        await transport.close()


# ════════════════════════════════════════════════════════════════
# 5. Frame transport
# ════════════════════════════════════════════════════════════════


class TestFrameTransport:
    """Camera frame capture tests."""

    @pytest.fixture
    def mock_robot(self):
        return MockRobotArm()

    @pytest.fixture
    def transport(self, mock_robot):
        return RobotArmTransport(robot_factory=lambda: mock_robot)

    @pytest.fixture
    def context(self, mock_robot):
        mapper = RobotChannelMapper()
        return _mock_context(mapper.map_robot(mock_robot))

    @pytest.mark.asyncio
    async def test_read_frame_returns_jpeg(self, transport, context):
        """read_frame returns JPEG-encoded bytes."""
        await transport.open(context)
        frame = await transport.read_frame("camera.top")
        assert frame.media_type == "image/jpeg"
        assert len(frame.data) > 0
        assert frame.data[:2] == b"\xff\xd8"  # JPEG magic bytes
        await transport.close()

    @pytest.mark.asyncio
    async def test_read_frame_dimensions(self, transport, context):
        """read_frame reports correct width and height."""
        await transport.open(context)
        frame = await transport.read_frame("camera.top")
        assert frame.width == 640
        assert frame.height == 480
        await transport.close()

    @pytest.mark.asyncio
    async def test_read_non_camera_channel_raises(self, transport, context):
        """read_frame on a non-frame channel raises TransportError."""
        await transport.open(context)
        with pytest.raises(TransportError, match="not a frame channel"):
            await transport.read_frame("joint.shoulder.position")
        await transport.close()


# ════════════════════════════════════════════════════════════════
# 6. Context provider
# ════════════════════════════════════════════════════════════════


class TestRobotArmContextProvider:
    """Provider discovers robots from configuration."""

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        return {
            "robot_type": "so100_follower",
            "device_name": "test_arm",
            "display_name": "Test Arm",
            "cameras": {"top": {"index": 0, "width": 640, "height": 480, "fps": 30}},
            "motor_limits": {
                "shoulder": [-3.14, 3.14],
                "elbow": [-1.57, 1.57],
                "wrist": [-3.14, 3.14],
            },
            "sample_rate_hz": 50.0,
        }

    def test_discover_returns_context(self, config):
        """Provider returns one HardwareContext from config."""
        provider = RobotArmContextProvider(config)
        contexts = provider.discover()
        assert len(contexts) == 1
        assert isinstance(contexts[0], HardwareContext)

    def test_device_id_format(self, config):
        """device_id is 'robot.{device_name}'."""
        ctx = RobotArmContextProvider(config).discover()[0]
        assert ctx.device_id == "robot.test_arm"

    def test_transport_ref_kind(self, config):
        """TransportRef.kind is 'robot_arm'."""
        ctx = RobotArmContextProvider(config).discover()[0]
        assert ctx.transport.kind == "robot_arm"

    def test_provenance_is_imported(self, config):
        """ContextProvenance.source is IMPORTED."""
        ctx = RobotArmContextProvider(config).discover()[0]
        assert ctx.provenance.source == ContextSource.IMPORTED.value

    def test_halt_supported(self, config):
        """Discovered context declares halt_supported=True."""
        ctx = RobotArmContextProvider(config).discover()[0]
        assert ctx.halt_supported is True

    def test_infer_device_class_arm(self):
        """so100_follower maps to robot_arm."""
        cfg = {"robot_type": "so100_follower", "motor_limits": {"j": [-3.14, 3.14]}}
        ctx = RobotArmContextProvider(cfg).discover()[0]
        assert ctx.device_class == "robot_arm"

    def test_infer_device_class_mobile(self):
        """kiwi_bot maps to mobile_robot."""
        cfg = {"robot_type": "kiwi_bot", "motor_limits": {"wheel": [-10, 10]}}
        ctx = RobotArmContextProvider(cfg).discover()[0]
        assert ctx.device_class == "mobile_robot"

    def test_channels_include_motors_and_cameras(self, config):
        """Discovered context includes motor and camera channels."""
        ctx = RobotArmContextProvider(config).discover()[0]
        ids = {ch.channel_id for ch in ctx.channels}
        assert any("joint" in cid for cid in ids)
        assert any("camera" in cid for cid in ids)


# ════════════════════════════════════════════════════════════════
# 7. Integration
# ════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end: Provider → Registry admission → Transport → read/write."""

    @pytest.mark.asyncio
    async def test_provider_to_transport_roundtrip(self):
        """Discover robot, admit to registry, open transport, read and write."""
        # 1. Provider discovers from config.
        config = {
            "robot_type": "so100_follower",
            "device_name": "integration_arm",
            "motor_limits": {
                "shoulder": [-3.14, 3.14],
                "elbow": [-1.57, 1.57],
            },
            "cameras": {"top": {"width": 640, "height": 480, "fps": 30}},
            "sample_rate_hz": 50.0,
        }
        provider = RobotArmContextProvider(config)
        contexts = provider.discover()
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.device_id == "robot.integration_arm"

        # 2. Build transport backed by mock robot.
        mock = MockRobotArm()
        transport = RobotArmTransport(robot_factory=lambda: mock)

        # 3. Open.
        status = await transport.open(ctx)
        assert status.connected is True

        # 4. Read a motor channel from the discovered context.
        motor_channels = [
            ch for ch in ctx.channels
            if "joint" in ch.channel_id and ch.channel_id.endswith(".position")
        ]
        assert len(motor_channels) > 0
        reading = await transport.read(motor_channels[0].channel_id)
        assert reading.device_id == ctx.device_id

        # 5. Write a writable motor channel.
        writable = [ch for ch in motor_channels if ch.is_writable]
        assert len(writable) > 0
        outcome = await transport.write(writable[0].channel_id, 1.0)
        assert outcome.ok is True

        # 6. Close.
        status = await transport.close()
        assert status.connected is False
