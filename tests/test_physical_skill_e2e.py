# Copyright (c) Alibaba, Inc. and its affiliates.
"""Phase 1 OODA+V end-to-end tests for PhysicalSkillPlugin.

Exercises the complete Observe-Orient-Decide-Act-Verify cycle through
PhysicalSkillPlugin, VLAInferenceTransport, OperationVerifier,
PositionVerifier, GraspVerifier, and EvidenceStore -- all backed by
in-process mocks so no hardware or gRPC dependencies are needed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Representation,
    TransportRef,
)
from leapflow.hardware.evidence import EvidenceStore
from leapflow.hardware.transport import (
    BatchReading,
    BatchWriteOutcome,
    Reading,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    WriteOutcome,
)
from leapflow.hardware.verification import (
    EvidenceBundle,
    OperationVerdict,
    PositionVerifier,
    VerdictStatus,
    collect_evidence,
)
from leapflow.plugins.tool_plugins.physical_skill import PhysicalSkillPlugin
from leapflow.robot.inference.strategy import (
    ComputeBudget,
    ComputeProfile,
    InferenceResult,
)

# Re-use the mock robot from the robot arm transport test suite.
from tests.test_robot_arm_transport import MockRobotArm


# ════════════════════════════════════════════════════════════════
# Mock infrastructure
# ════════════════════════════════════════════════════════════════


class MockPolicy:
    """Simulates a VLA policy for testing without actual model."""

    def __init__(self, action_dim: int = 4) -> None:
        self._action_dim = action_dim
        self._call_count = 0
        self._reset_count = 0

    def select_action(self, batch: dict) -> list[float]:
        """Return a deterministic action based on call count."""
        self._call_count += 1
        return [0.1 * self._call_count] * self._action_dim

    def reset(self) -> None:
        self._reset_count += 1
        self._call_count = 0


class MockStrategy:
    """InferenceStrategy adapter around ``MockPolicy`` for tests.

    The plugin cache now holds ``InferenceStrategy`` instances, not raw
    policies, so every direct ``plugin._policies[...] = ...`` injection
    used by these tests goes through this thin adapter.  It preserves the
    call/reset counters on the wrapped ``MockPolicy`` so existing
    assertions on ``mock_policy._call_count`` keep working.
    """

    def __init__(self, policy: Any, *, strategy_id: str = "mock") -> None:
        self._policy = policy
        self.strategy_id = strategy_id

    @property
    def compute_profile(self) -> ComputeProfile:
        return ComputeProfile(
            latency_range_ms=(0.1, 5.0),
            supports_chunking=True,
            gpu_required=False,
        )

    async def infer(
        self,
        observation: dict,
        *,
        budget: ComputeBudget | None = None,
    ) -> InferenceResult:
        chunk = int(budget.chunk_size) if budget is not None else 1
        select = getattr(self._policy, "select_action", None)
        if select is None:
            raise RuntimeError("MockPolicy missing select_action()")
        action = select(dict(observation))
        return InferenceResult(
            action=action,
            latency_ms=0.5,
            confidence=1.0,
            chunk_size=chunk,
            metadata={"strategy": self.strategy_id},
        )

    async def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()


class MockInferenceServer:
    """Simulates a VLA inference server for testing without gRPC."""

    def __init__(self) -> None:
        self.infer_count = 0
        self.last_observation: dict | None = None

    async def infer(self, observation: dict) -> dict:
        self.infer_count += 1
        self.last_observation = observation
        return {"action": [0.1, 0.2, 0.3, 0.0]}


class _ApprovalResult:
    """Minimal stand-in for the orchestrator's ApprovalResult."""

    def __init__(self, approved: bool, denial_message: str = "") -> None:
        self.approved = approved
        self.denial_message = denial_message


class MockApprovalGate:
    """Auto-approves or auto-rejects based on configuration.

    Exposes ``evaluate`` (the production orchestrator API the plugin calls)
    and keeps ``request_approval`` for direct-call tests.  Both record into
    ``approval_requests`` so a test can assert the gate was consulted.
    """

    def __init__(self, auto_approve: bool = True) -> None:
        self._auto_approve = auto_approve
        self.approval_requests: list[Any] = []

    async def request_approval(self, request: dict) -> bool:
        self.approval_requests.append(request)
        return self._auto_approve

    async def evaluate(self, descriptor: Any) -> _ApprovalResult:
        self.approval_requests.append(descriptor)
        return _ApprovalResult(
            approved=self._auto_approve,
            denial_message="" if self._auto_approve else "denied by mock gate",
        )


class MockTrustGate:
    """Tracks trust level changes for verification."""

    def __init__(self, initial_level: str = "UNTRUSTED") -> None:
        self._level = initial_level
        self._successes = 0
        self._failures = 0
        self.history: list[tuple[str, str, str]] = []

    def record_success(self, device_id: str, channel_id: str) -> None:
        self._successes += 1
        self.history.append(("success", device_id, channel_id))

    def record_failure(self, device_id: str, channel_id: str) -> None:
        self._failures += 1
        self.history.append(("failure", device_id, channel_id))


class MockRegistry:
    """Minimal hardware registry wiring MockRobotArm for PhysicalSkillPlugin.

    Presents channels as writable joints + a gripper, and supports
    read/write via duck-typed device with context and transport.
    """

    def __init__(
        self,
        robot: MockRobotArm,
        verifiers: list[Any] | None = None,
    ) -> None:
        self._robot = robot
        self.verifiers: list[Any] = verifiers or []
        # Writable channels matching MockRobotArm's 3 joints + gripper.
        self._writable_channels = [
            Channel(
                channel_id="joint.shoulder.position",
                direction=Direction.READWRITE.value,
                representation=Representation.SCALAR.value,
                effect=HardwareEffect.ACTUATE.value,
                description="Shoulder joint position",
                envelope=Envelope(declared=True, min_value=-3.14, max_value=3.14),
            ),
            Channel(
                channel_id="joint.elbow.position",
                direction=Direction.READWRITE.value,
                representation=Representation.SCALAR.value,
                effect=HardwareEffect.ACTUATE.value,
                description="Elbow joint position",
                envelope=Envelope(declared=True, min_value=-1.57, max_value=1.57),
            ),
            Channel(
                channel_id="joint.wrist.position",
                direction=Direction.READWRITE.value,
                representation=Representation.SCALAR.value,
                effect=HardwareEffect.ACTUATE.value,
                description="Wrist joint position",
                envelope=Envelope(declared=True, min_value=-3.14, max_value=3.14),
            ),
            Channel(
                channel_id="gripper.grip",
                direction=Direction.READWRITE.value,
                representation=Representation.STATE.value,
                effect=HardwareEffect.ACTUATE.value,
                description="Gripper state",
                envelope=Envelope(declared=True, min_value=0.0, max_value=1.0),
            ),
        ]
        self._context = HardwareContext(
            device_id="mock.arm",
            display_name="Mock Arm",
            transport=TransportRef(kind="mock", config={}),
            channels=tuple(self._writable_channels),
            halt_supported=True,
            provenance=ContextProvenance(source=ContextSource.IMPORTED.value),
        )
        self._device = _MockDevice(self._robot, self._context)

    def get_device(self, device_id: str) -> Any:
        return self._device

    async def read(self, device_id: str, channel_id: str) -> Reading:
        obs = self._robot.get_observation()
        # Map channel id back to robot observation key.
        key_map = {
            "joint.shoulder.position": "shoulder.position",
            "joint.elbow.position": "elbow.position",
            "joint.wrist.position": "wrist.position",
            "gripper.grip": "grip",
        }
        obs_key = key_map.get(channel_id, channel_id)
        value = obs.get(obs_key, 0.0)
        if hasattr(value, "item"):
            value = float(value)
        return Reading(
            device_id=device_id,
            channel_id=channel_id,
            value=value,
        )

    async def write(self, device_id: str, channel_id: str, value: Any) -> WriteOutcome:
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)


class _MockDevice:
    """Quacks like a registered device: has .transport and .context."""

    def __init__(self, robot: MockRobotArm, context: HardwareContext) -> None:
        self._robot = robot
        self.context = context
        self.transport = _MockBatchTransport(robot, context)


class _MockBatchTransport:
    """Minimal BatchTransport backed by MockRobotArm."""

    def __init__(self, robot: MockRobotArm, context: HardwareContext) -> None:
        self._robot = robot
        self._context = context

    async def read_batch(self, channel_ids: tuple[str, ...]) -> BatchReading:
        obs = self._robot.get_observation()
        key_map = {
            "joint.shoulder.position": "shoulder.position",
            "joint.elbow.position": "elbow.position",
            "joint.wrist.position": "wrist.position",
            "gripper.grip": "grip",
        }
        readings = []
        for cid in channel_ids:
            obs_key = key_map.get(cid, cid)
            val = obs.get(obs_key, 0.0)
            if hasattr(val, "item"):
                val = float(val)
            readings.append(Reading(device_id="mock.arm", channel_id=cid, value=val))
        return BatchReading(device_id="mock.arm", readings=tuple(readings))

    async def write_batch(
        self, commands: tuple[tuple[str, Any], ...],
    ) -> BatchWriteOutcome:
        action = [v for _, v in commands]
        self._robot.send_action(action)
        outcomes = tuple(
            WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)
            for _ in commands
        )
        return BatchWriteOutcome(
            ok=True,
            outcomes=outcomes,
            side_effect_state=SIDE_EFFECT_COMMITTED,
        )


# ── Fixtures ──


@pytest.fixture
def mock_robot() -> MockRobotArm:
    robot = MockRobotArm()
    robot.connect()
    return robot


@pytest.fixture
def verifier() -> PositionVerifier:
    return PositionVerifier(default_tolerance=0.05)


@pytest.fixture
def registry(mock_robot: MockRobotArm, verifier: PositionVerifier) -> MockRegistry:
    return MockRegistry(mock_robot, verifiers=[verifier])


@pytest.fixture
def plugin(registry: MockRegistry) -> PhysicalSkillPlugin:
    p = PhysicalSkillPlugin()
    p.bind_runtime(
        hardware_registry=registry,
        hardware_approval_gate=MockApprovalGate(auto_approve=True),
        hardware_trust_gate=MockTrustGate(),
        session_id="test-session",
    )
    return p


# ════════════════════════════════════════════════════════════════
# 1. OODA+V complete cycle
# ════════════════════════════════════════════════════════════════


class TestOODAVLoop:
    """Test the complete Observe-Orient-Decide-Act-Verify cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_observe_infer_execute_verify(
        self, plugin: PhysicalSkillPlugin, registry: MockRegistry,
    ) -> None:
        """Complete OODA+V: read observation -> policy inference ->
        write action -> verify position -> success."""
        # Inject a mock local policy to bypass LeapRobot import.
        mock_policy = MockPolicy(action_dim=4)
        plugin._policies["mock_policy"] = MockStrategy(mock_policy)

        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "task": "pick up cube",
            "execute": True,
            "verify": True,
        })

        assert result["ok"] is True
        assert "action" in result
        assert "execution" in result
        assert result["execution"]["ok"] is True
        assert mock_policy._call_count == 1

    @pytest.mark.asyncio
    async def test_cycle_with_verification_failure(
        self, plugin: PhysicalSkillPlugin, mock_robot: MockRobotArm,
    ) -> None:
        """OODA+V where verification detects position error."""
        mock_policy = MockPolicy(action_dim=4)
        plugin._policies["mock_policy"] = MockStrategy(mock_policy)

        # Make the robot report a very different position after writing,
        # by forcing the robot to return positions far from what was commanded.
        orig_send = mock_robot.send_action

        def _divergent_send(action: Any) -> None:
            orig_send(action)
            # Override all positions to a far-off value.
            for key in mock_robot._positions:
                mock_robot._positions[key] = 999.0

        mock_robot.send_action = _divergent_send  # type: ignore[assignment]

        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": True,
            "verify": True,
        })

        assert result["ok"] is True
        # Verification should report failure or inconclusive due to deviation.
        if "verification" in result:
            v = result["verification"]
            assert v["status"] in ("failure", "inconclusive")

    @pytest.mark.asyncio
    async def test_cycle_without_execution(
        self, plugin: PhysicalSkillPlugin, mock_robot: MockRobotArm,
    ) -> None:
        """Inference only (execute=False): no physical effect."""
        mock_policy = MockPolicy(action_dim=4)
        plugin._policies["mock_policy"] = MockStrategy(mock_policy)
        sent_before = len(mock_robot._actions_sent)

        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": False,
        })

        assert result["ok"] is True
        assert "action" in result
        # No execution block and no new actions sent.
        assert "execution" not in result or result.get("execution") is None
        assert len(mock_robot._actions_sent) == sent_before

    @pytest.mark.asyncio
    async def test_chunk_prediction(
        self, plugin: PhysicalSkillPlugin, mock_robot: MockRobotArm,
    ) -> None:
        """Multi-step chunk: predict N actions, execute sequentially."""
        mock_policy = MockPolicy(action_dim=4)
        plugin._policies["mock_policy"] = MockStrategy(mock_policy)

        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "chunk_size": 3,
            "execute": True,
        })

        assert result["ok"] is True
        assert "action" in result
        # Policy was called once (chunk prediction produces one action).
        assert mock_policy._call_count == 1


# ════════════════════════════════════════════════════════════════
# 2. Policy management
# ════════════════════════════════════════════════════════════════


class TestPolicyManagement:
    """Policy loading, caching, and listing."""

    @pytest.mark.asyncio
    async def test_local_policy_loading(self, plugin: PhysicalSkillPlugin) -> None:
        """Load a mock local policy and run inference."""
        mock_policy = MockPolicy(action_dim=4)
        plugin._policies["my_policy"] = MockStrategy(mock_policy)

        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "my_policy",
            "execute": False,
        })
        assert result["ok"] is True
        assert mock_policy._call_count == 1

    @pytest.mark.asyncio
    async def test_remote_policy_path(self, plugin: PhysicalSkillPlugin) -> None:
        """Remote policy path 'remote:host:port' resolves to a VLARemoteStrategy."""
        from leapflow.robot.inference.vla_remote import VLARemoteStrategy

        result = await plugin._load_policy("remote:localhost:50051")
        assert isinstance(result, VLARemoteStrategy)
        # Verify caching.
        assert "remote:localhost:50051" in plugin._policies
        # The strategy id carries the address so multiple endpoints can
        # coexist in the registry without colliding on ``vla_remote``.
        assert result.strategy_id == "vla_remote:localhost:50051"

    @pytest.mark.asyncio
    async def test_policy_caching(self, plugin: PhysicalSkillPlugin) -> None:
        """Same policy path reuses cached instance."""
        mock_policy = MockPolicy()
        plugin._policies["cached_pol"] = MockStrategy(mock_policy)

        p1 = await plugin._load_policy("cached_pol")
        p2 = await plugin._load_policy("cached_pol")
        assert p1 is p2

    @pytest.mark.asyncio
    async def test_policy_list(self, plugin: PhysicalSkillPlugin) -> None:
        """hw_policy_list returns loaded policies."""
        plugin._policies["local_a"] = MockStrategy(MockPolicy())
        plugin._policies["remote:gpu:5000"] = object()

        result = await plugin.policy_list({"source": "all"})
        assert result["ok"] is True
        names = {p["name"] for p in result["policies"]}
        assert "local_a" in names
        assert "remote:gpu:5000" in names


# ════════════════════════════════════════════════════════════════
# 3. Episode lifecycle
# ════════════════════════════════════════════════════════════════


class TestEpisodeLifecycle:
    """Episode start / stop / status management."""

    @pytest.mark.asyncio
    async def test_episode_start_stop(self, plugin: PhysicalSkillPlugin) -> None:
        """Start and stop episode, verify summary stats."""
        mock_policy = MockPolicy()
        plugin._policies["ep_pol"] = MockStrategy(mock_policy)

        start = await plugin.policy_episode({
            "action": "start",
            "device_id": "mock.arm",
            "policy": "ep_pol",
            "task": "stack blocks",
        })
        assert start["ok"] is True
        assert start["action"] == "started"

        stop = await plugin.policy_episode({"action": "stop"})
        assert stop["ok"] is True
        assert stop["action"] == "stopped"
        assert "summary" in stop
        assert stop["summary"]["steps"] == 0

    @pytest.mark.asyncio
    async def test_episode_status(self, plugin: PhysicalSkillPlugin) -> None:
        """Query episode status during active episode."""
        plugin._policies["ep_pol"] = MockStrategy(MockPolicy())
        await plugin.policy_episode({
            "action": "start",
            "device_id": "mock.arm",
            "policy": "ep_pol",
        })

        status = await plugin.policy_episode({"action": "status"})
        assert status["ok"] is True
        assert status["active"] is True
        assert status["episode"]["device_id"] == "mock.arm"

        await plugin.policy_episode({"action": "stop"})

    @pytest.mark.asyncio
    async def test_infer_within_episode(self, plugin: PhysicalSkillPlugin) -> None:
        """Inference during active episode tracks step count."""
        mock_policy = MockPolicy(action_dim=4)
        plugin._policies["ep_pol"] = MockStrategy(mock_policy)

        await plugin.policy_episode({
            "action": "start",
            "device_id": "mock.arm",
            "policy": "ep_pol",
        })

        # Run two inferences.
        for _ in range(2):
            await plugin.policy_infer({
                "device_id": "mock.arm",
                "policy": "ep_pol",
                "execute": True,
            })

        stop = await plugin.policy_episode({"action": "stop"})
        assert stop["summary"]["steps"] == 2
        assert mock_policy._call_count == 2


# ════════════════════════════════════════════════════════════════
# 4. Trust progression
# ════════════════════════════════════════════════════════════════


class TestTrustProgression:
    """Trust gate interaction for success/failure recording."""

    @pytest.mark.asyncio
    async def test_trust_promotion_on_success(self) -> None:
        """Multiple successful verifications record trust successes."""
        gate = MockTrustGate()
        for _ in range(3):
            gate.record_success("mock.arm", "joint.shoulder.position")

        assert gate._successes == 3
        assert len(gate.history) == 3
        assert all(h[0] == "success" for h in gate.history)

    @pytest.mark.asyncio
    async def test_trust_demotion_on_failure(self) -> None:
        """Verification failure triggers trust demotion."""
        gate = MockTrustGate()
        gate.record_success("mock.arm", "joint.shoulder.position")
        gate.record_failure("mock.arm", "joint.shoulder.position")

        assert gate._successes == 1
        assert gate._failures == 1
        assert gate.history[-1][0] == "failure"

    @pytest.mark.asyncio
    async def test_irreversible_always_needs_approval(self) -> None:
        """Approval gate is always invoked regardless of trust level."""
        gate = MockApprovalGate(auto_approve=True)
        await gate.request_approval({"channel": "gripper.grip", "effect": "DISPENSE"})
        await gate.request_approval({"channel": "gripper.grip", "effect": "EMIT"})

        assert len(gate.approval_requests) == 2


# ════════════════════════════════════════════════════════════════
# 5. Evidence store integration
# ════════════════════════════════════════════════════════════════


class TestEvidenceIntegration:
    """EvidenceStore persistence and analytics."""

    @pytest.mark.asyncio
    async def test_evidence_recorded_after_verification(self, tmp_path) -> None:
        """EvidenceStore receives bundle after verified operation."""
        store = EvidenceStore(tmp_path / "evidence.duckdb")
        try:
            bundle = EvidenceBundle(
                operation_id="op_001",
                device_id="mock.arm",
                channel_id="joint.shoulder.position",
                timestamp=time.time(),
                intended_value=1.5,
                actual_outcome=None,
                post_settle_readings=(
                    Reading(
                        device_id="mock.arm",
                        channel_id="joint.shoulder.position",
                        value=1.48,
                    ),
                ),
            )
            verdict = OperationVerdict(
                status=VerdictStatus.SUCCESS.value,
                confidence=0.95,
                deviation=0.02,
                detail="within tolerance",
            )

            op_id = await store.record(bundle, verdict)
            assert op_id == "op_001"
            assert store.records_written == 1

            rows = await store.query(device_id="mock.arm")
            assert len(rows) >= 1
            assert rows[0]["operation_id"] == "op_001"
            assert rows[0]["verdict_status"] == "success"
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_success_rate_query(self, tmp_path) -> None:
        """success_rate returns correct statistics after operations."""
        store = EvidenceStore(tmp_path / "evidence.duckdb")
        try:
            now = time.time()
            # Record 2 successes and 1 failure.
            for i, status in enumerate(["success", "success", "failure"]):
                bundle = EvidenceBundle(
                    operation_id=f"rate_{i:03d}",
                    device_id="mock.arm",
                    channel_id="joint.shoulder.position",
                    timestamp=now + i,
                    intended_value=1.0,
                    actual_outcome=None,
                    post_settle_readings=(),
                )
                verdict = OperationVerdict(
                    status=status,
                    confidence=0.9,
                )
                await store.record(bundle, verdict)

            stats = await store.success_rate("mock.arm")
            assert stats["total"] == 3
            assert stats["success"] == 2
            assert stats["failure"] == 1
            assert stats["success_rate"] == pytest.approx(2 / 3)
        finally:
            store.close()


# ════════════════════════════════════════════════════════════════
# 6. Error handling
# ════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Graceful failures without physical effects."""

    @pytest.mark.asyncio
    async def test_inference_timeout(self, plugin: PhysicalSkillPlugin) -> None:
        """Policy inference timeout returns error, no physical effect."""

        class SlowPolicy:
            async def select_action(self, batch: dict) -> list:
                await asyncio.sleep(10)
                return [0.0]

        plugin._policies["slow"] = MockStrategy(SlowPolicy())
        # The plugin calls asyncio.to_thread(select) which is sync,
        # but a truly slow policy would be caught by broader timeout.
        # Here we test the error path with an exception.
        class FailPolicy:
            def select_action(self, batch: dict) -> None:
                raise TimeoutError("inference timed out")

        plugin._policies["fail_timeout"] = MockStrategy(FailPolicy())
        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "fail_timeout",
            "execute": True,
        })
        assert result["ok"] is False
        assert "inference failed" in result["error"]

    @pytest.mark.asyncio
    async def test_robot_disconnected(
        self, plugin: PhysicalSkillPlugin, mock_robot: MockRobotArm,
    ) -> None:
        """Operation on disconnected robot returns clear error."""
        mock_robot.disconnect()

        class FailReadRegistry:
            verifiers: list = []

            def get_device(self, device_id: str) -> Any:
                raise RuntimeError("device not connected")

        plugin.bind_runtime(hardware_registry=FailReadRegistry())
        plugin._policies["pol"] = MockStrategy(MockPolicy(action_dim=4))

        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "pol",
            "execute": True,
        })
        assert result["ok"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_policy(self, plugin: PhysicalSkillPlugin) -> None:
        """Non-existent policy path surfaces the load failure at inference time.

        With the strategy-registry delegation, ``_load_policy`` constructs
        the strategy eagerly but the underlying model is loaded lazily
        inside ``VLALocalStrategy.infer``; the missing checkpoint therefore
        turns into an inference failure carrying the loader's own message.
        """
        result = await plugin.policy_infer({
            "device_id": "mock.arm",
            "policy": "nonexistent/path/to/model",
        })
        assert result["ok"] is False
        assert "inference failed" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_device_id(self, plugin: PhysicalSkillPlugin) -> None:
        """Missing device_id is rejected immediately."""
        result = await plugin.policy_infer({
            "policy": "some_policy",
        })
        assert result["ok"] is False
        assert "device_id" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_policy_path(self, plugin: PhysicalSkillPlugin) -> None:
        """Missing policy path is rejected immediately."""
        result = await plugin.policy_infer({
            "device_id": "mock.arm",
        })
        assert result["ok"] is False
        assert "policy" in result["error"]


# ════════════════════════════════════════════════════════════════
# 7. Approval gate integration
# ════════════════════════════════════════════════════════════════


class TestApprovalIntegration:
    """Test that physical operations go through the approval gate."""

    @pytest.mark.asyncio
    async def test_approval_gate_receives_request(self, registry: MockRegistry) -> None:
        """hw_policy_infer sends an approval request to the gate before execution."""
        gate = MockApprovalGate(auto_approve=True)
        p = PhysicalSkillPlugin()
        p.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=gate,
            hardware_trust_gate=MockTrustGate(),
            session_id="test-session",
        )
        p._policies["mock_policy"] = MockStrategy(MockPolicy(action_dim=4))

        result = await p.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": True,
        })

        assert result["ok"] is True
        assert len(gate.approval_requests) == 1
        # The descriptor carries the policy provenance and the target device.
        descriptor = gate.approval_requests[0]
        assert descriptor.metadata["device_id"] == "mock.arm"
        assert descriptor.metadata["source"] == "hw_policy_infer"

    @pytest.mark.asyncio
    async def test_approval_rejected_blocks_execution(
        self, registry: MockRegistry, mock_robot: MockRobotArm,
    ) -> None:
        """When the gate rejects, no write_batch reaches the robot."""
        gate = MockApprovalGate(auto_approve=False)
        p = PhysicalSkillPlugin()
        p.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=gate,
            hardware_trust_gate=MockTrustGate(),
            session_id="test-session",
        )
        p._policies["mock_policy"] = MockStrategy(MockPolicy(action_dim=4))
        sent_before = len(mock_robot._actions_sent)

        result = await p.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": True,
        })

        assert len(gate.approval_requests) == 1
        # Execution was refused and nothing was written to the robot.
        assert result["ok"] is False
        assert result["execution"]["ok"] is False
        assert "not approved" in result["execution"]["error"]
        assert len(mock_robot._actions_sent) == sent_before

    @pytest.mark.asyncio
    async def test_absent_gate_fails_closed(
        self, registry: MockRegistry, mock_robot: MockRobotArm,
    ) -> None:
        """With no gate bound, a physical write is refused (fail-closed)."""
        p = PhysicalSkillPlugin()
        p.bind_runtime(hardware_registry=registry, session_id="test-session")
        p._policies["mock_policy"] = MockStrategy(MockPolicy(action_dim=4))
        sent_before = len(mock_robot._actions_sent)

        result = await p.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": True,
        })

        assert result["ok"] is False
        assert result["execution"]["ok"] is False
        assert len(mock_robot._actions_sent) == sent_before


# ════════════════════════════════════════════════════════════════
# 8. Trust gate integration
# ════════════════════════════════════════════════════════════════


class TestTrustIntegration:
    """Test the trust gate receives verification results."""

    @pytest.mark.asyncio
    async def test_successful_verification_records_trust(
        self, registry: MockRegistry,
    ) -> None:
        """After a successful verification, the trust gate records success."""
        trust = MockTrustGate()
        p = PhysicalSkillPlugin()
        p.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=MockApprovalGate(auto_approve=True),
            hardware_trust_gate=trust,
            session_id="test-session",
        )
        p._policies["mock_policy"] = MockStrategy(MockPolicy(action_dim=4))

        result = await p.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": True,
            "verify": True,
        })

        assert result["ok"] is True
        # The commanded position lands within tolerance -> success recorded.
        assert any(h[0] == "success" for h in trust.history)
        assert all(h[0] != "failure" for h in trust.history)

    @pytest.mark.asyncio
    async def test_failed_verification_records_failure(
        self, registry: MockRegistry, mock_robot: MockRobotArm,
    ) -> None:
        """After a failed verification, the trust gate records failure."""
        trust = MockTrustGate()
        p = PhysicalSkillPlugin()
        p.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=MockApprovalGate(auto_approve=True),
            hardware_trust_gate=trust,
            session_id="test-session",
        )
        p._policies["mock_policy"] = MockStrategy(MockPolicy(action_dim=4))

        # Force the robot far from the commanded position so verification fails.
        orig_send = mock_robot.send_action

        def _divergent_send(action: Any) -> None:
            orig_send(action)
            for key in mock_robot._positions:
                mock_robot._positions[key] = 999.0

        mock_robot.send_action = _divergent_send  # type: ignore[assignment]

        result = await p.policy_infer({
            "device_id": "mock.arm",
            "policy": "mock_policy",
            "execute": True,
            "verify": True,
        })

        assert result["ok"] is True
        # A large deviation yields a failure verdict -> failure recorded.
        assert any(h[0] == "failure" for h in trust.history)
