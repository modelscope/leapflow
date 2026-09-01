"""Phase 3 learning-layer tests: prediction physical branch, EMA bias,
causal rules, hardware trust gate, MCP capability validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

# ════════════════════════════════════════════════════════════════
# 3.1 — PhysicalSnapshot + PredictionLoop physical branch
# ════════════════════════════════════════════════════════════════


class TestPhysicalSnapshot:
    """PhysicalSnapshot dataclass and _compare_physical branch."""

    def test_physical_snapshot_fields(self) -> None:
        from leapflow.world_model.prediction import PhysicalSnapshot
        snap = PhysicalSnapshot(
            device_id="pump-1",
            channel_id="flow_rate",
            value=50.0,
            quantity="flow_rate",
            unit="uL/s",
        )
        assert snap.device_id == "pump-1"
        assert snap.channel_id == "flow_rate"
        assert snap.value == 50.0
        assert snap.unit == "uL/s"
        assert snap.envelope is None

    def test_physical_snapshot_frozen(self) -> None:
        from leapflow.world_model.prediction import PhysicalSnapshot
        snap = PhysicalSnapshot(device_id="d", channel_id="c", value=1.0)
        with pytest.raises(AttributeError):
            snap.value = 2.0  # type: ignore[misc]

    def test_prediction_outcome_physical_delta_source(self) -> None:
        """PredictionOutcome.delta_source can be 'physical'."""
        from leapflow.world_model.prediction import Prediction, PredictionOutcome
        po = PredictionOutcome(
            prediction=Prediction("hw_configure", "effect", 0.8),
            pre_snapshot=None,
            post_snapshot=None,
            actual_effect="settled",
            delta=0.02,
            delta_source="physical",
            timestamp=time.time(),
        )
        assert po.delta_source == "physical"


class TestPredictionLoopPhysicalBranch:
    """The _compare_physical fast path (zero LLM calls)."""

    def test_hardware_learning_disabled_by_default(self) -> None:
        """Default PredictionLoop does not activate physical branch."""
        from leapflow.world_model.prediction import PredictionLoop

        loop = PredictionLoop(
            llm=AsyncMock(),
            snapshot_service=AsyncMock(),
            experience_store=AsyncMock(),
            budget=AsyncMock(),
        )
        assert loop._hardware_learning_enabled is False

    def test_hardware_learning_enabled_flag(self) -> None:
        from leapflow.world_model.prediction import PredictionLoop

        loop = PredictionLoop(
            llm=AsyncMock(),
            snapshot_service=AsyncMock(),
            experience_store=AsyncMock(),
            budget=AsyncMock(),
            hardware_learning_enabled=True,
        )
        assert loop._hardware_learning_enabled is True


# ════════════════════════════════════════════════════════════════
# IC-9 G-7 — EMA bias
# ════════════════════════════════════════════════════════════════


class TestEMABias:
    """_update_bias uses EMA (alpha=0.1) and converges under drift."""

    def test_alpha_is_0_1(self) -> None:
        from leapflow.hardware.outcome import _BIAS_ALPHA
        assert _BIAS_ALPHA == pytest.approx(0.1)

    def test_first_observation_sets_bias(self) -> None:
        from leapflow.hardware.outcome import HardwareOutcomeRecorder

        recorder = HardwareOutcomeRecorder(experience_store=object())
        key = ("dev", "ch")
        recorder._update_bias(key, 5.0)
        assert recorder._bias[key] == (5.0, 1)

    def test_ema_converges_under_constant_drift(self) -> None:
        """Under constant residual, bias should converge toward that value."""
        from leapflow.hardware.outcome import HardwareOutcomeRecorder

        recorder = HardwareOutcomeRecorder(experience_store=object())
        key = ("dev", "ch")
        for _ in range(200):
            recorder._update_bias(key, 10.0)
        bias, samples = recorder._bias[key]
        # After many updates with alpha=0.1, should be very close to 10.0
        assert abs(bias - 10.0) < 0.1
        assert samples == 200

    def test_ema_does_not_accumulate_unbounded(self) -> None:
        """Linearly increasing residual should not make bias grow without bound."""
        from leapflow.hardware.outcome import HardwareOutcomeRecorder

        recorder = HardwareOutcomeRecorder(experience_store=object())
        key = ("dev", "ch")
        for i in range(1, 100):
            recorder._update_bias(key, float(i))
        bias, _ = recorder._bias[key]
        # EMA with alpha=0.1 lags the series; it should be much less than 99
        assert bias < 99.0

    def test_calibration_for_exposes_samples(self) -> None:
        from leapflow.hardware.outcome import HardwareOutcomeRecorder

        recorder = HardwareOutcomeRecorder(experience_store=object())
        key = ("dev", "ch")
        for _ in range(5):
            recorder._update_bias(key, 2.0)
        result = recorder.calibration_for("dev", "ch")
        assert result is not None
        bias, samples = result
        assert samples == 5


# ════════════════════════════════════════════════════════════════
# IC-9 G-10 — hw_describe calibration notice
# ════════════════════════════════════════════════════════════════


class TestHwDescribeCalibrationNotice:
    """hw_describe annotates unverified devices."""

    @pytest.fixture()
    def _verified_context(self):
        from leapflow.hardware.context import (
            HC_VERSION, Channel, ContextProvenance, Direction,
            Envelope, HardwareContext, HardwareEffect, TransportRef,
        )
        return HardwareContext(
            device_id="pump-1",
            display_name="Pump",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="test"),
            channels=(
                Channel(
                    channel_id="flow",
                    quantity="flow_rate",
                    unit="uL/s",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
            provenance=ContextProvenance(verified_by="operator"),
        )

    @pytest.fixture()
    def _unverified_context(self):
        from leapflow.hardware.context import (
            HC_VERSION, Channel, ContextProvenance, Direction,
            Envelope, HardwareContext, HardwareEffect, TransportRef,
        )
        return HardwareContext(
            device_id="pump-2",
            display_name="Pump",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="test"),
            channels=(
                Channel(
                    channel_id="flow",
                    quantity="flow_rate",
                    unit="uL/s",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
            provenance=ContextProvenance(verified_by=""),
        )

    @pytest.mark.asyncio
    async def test_unverified_device_gets_calibration_notice(self, _unverified_context) -> None:
        """hw_describe returns a calibration_notice for unverified devices."""
        registry = _FakeRegistry(contexts=[_unverified_context])
        from leapflow.hardware.tools import HardwareTools
        tools = HardwareTools(registry, gate=None)
        result = await tools.hw_describe(device_id="pump-2")
        assert result["ok"] is True
        assert "calibration_notice" in result
        assert "not been independently calibrated" in result["calibration_notice"]

    @pytest.mark.asyncio
    async def test_verified_device_no_calibration_notice(self, _verified_context) -> None:
        """hw_describe omits calibration_notice for verified devices."""
        registry = _FakeRegistry(contexts=[_verified_context])
        from leapflow.hardware.tools import HardwareTools
        tools = HardwareTools(registry, gate=None)
        result = await tools.hw_describe(device_id="pump-1")
        assert result["ok"] is True
        assert "calibration_notice" not in result


class _FakeRegistry:
    """Minimal fake hardware registry for tool tests."""

    def __init__(self, contexts: list = ()) -> None:
        self._contexts = {c.device_id: c for c in contexts}
        self._described: set[str] = set()
        self.outcome_recorder = None
        self.settings = None

    def contexts(self) -> list:
        return list(self._contexts.values())

    def context(self, device_id: str):
        return self._contexts.get(device_id)

    def mark_described(self, session_id: str, device_id: str) -> None:
        self._described.add(device_id)

    def was_described(self, session_id: str, device_id: str) -> bool:
        return device_id in self._described


# ════════════════════════════════════════════════════════════════
# 3.2 — Causal rules + dynamic rule management
# ════════════════════════════════════════════════════════════════


class TestPhysicalCausalRules:
    """rules.yaml includes hardware.* namespace rules."""

    def test_hardware_rules_present_in_yaml(self) -> None:
        rules_path = Path(__file__).parent.parent / "src/leapflow/causal/rules.yaml"
        with open(rules_path, "r") as fh:
            doc = yaml.safe_load(fh)
        names = [r["name"] for r in doc["rules"]]
        assert "hw_actuate_to_reading_change" in names
        assert "threshold_exceeded_to_estop" in names
        assert "hw_configure_to_settled" in names
        assert "hw_dispense_to_volume_change" in names
        assert "hw_reading_drift_to_recalibrate" in names

    def test_hardware_rules_do_not_conflict_with_ui_rules(self) -> None:
        rules_path = Path(__file__).parent.parent / "src/leapflow/causal/rules.yaml"
        with open(rules_path, "r") as fh:
            doc = yaml.safe_load(fh)
        names = [r["name"] for r in doc["rules"]]
        # No duplicates
        assert len(names) == len(set(names))

    def test_load_rules_includes_hardware(self) -> None:
        from leapflow.causal.inference import load_rules_from_yaml
        rules_path = Path(__file__).parent.parent / "src/leapflow/causal/rules.yaml"
        rules = load_rules_from_yaml(rules_path)
        hw_rules = [r for r in rules if r.parent_channel.startswith("hardware.")]
        assert len(hw_rules) == 5


class TestDynamicRuleManagement:
    """RuleEngine.add_rule and CausalInferenceEngine.reload_rules."""

    def test_rule_engine_add_rule(self) -> None:
        from leapflow.causal.inference import CausalRule, RuleEngine
        engine = RuleEngine(rules=[])
        rule = CausalRule(name="test_rule", parent_channel="a", child_channel="b")
        engine.add_rule(rule)
        assert len(engine.rules) == 1
        assert engine.rules[0].name == "test_rule"

    def test_rule_engine_add_replaces_duplicate(self) -> None:
        from leapflow.causal.inference import CausalRule, RuleEngine
        rule1 = CausalRule(name="dup", parent_channel="a", confidence=0.5)
        rule2 = CausalRule(name="dup", parent_channel="b", confidence=0.9)
        engine = RuleEngine(rules=[rule1])
        engine.add_rule(rule2)
        assert len(engine.rules) == 1
        assert engine.rules[0].parent_channel == "b"

    def test_rule_engine_set_rules(self) -> None:
        from leapflow.causal.inference import CausalRule, RuleEngine
        engine = RuleEngine(rules=[CausalRule(name="old", parent_channel="x")])
        engine.set_rules([CausalRule(name="new", parent_channel="y")])
        assert len(engine.rules) == 1
        assert engine.rules[0].name == "new"

    def test_causal_inference_engine_add_rule(self) -> None:
        from leapflow.causal.inference import CausalInferenceEngine, CausalRule
        from leapflow.causal.channel import ChannelRegistry
        registry = ChannelRegistry()
        engine = CausalInferenceEngine(registry, rules=[])
        rule = CausalRule(name="dynamic_hw", parent_channel="hardware.test")
        engine.add_rule(rule)
        assert any(r.name == "dynamic_hw" for r in engine._rules.rules)

    def test_causal_inference_engine_reload_rules(self) -> None:
        from leapflow.causal.inference import CausalInferenceEngine, CausalRule
        from leapflow.causal.channel import ChannelRegistry
        registry = ChannelRegistry()
        engine = CausalInferenceEngine(registry, rules=[])
        # Add a dynamic rule that should survive reload
        engine.add_rule(CausalRule(name="keep_me", parent_channel="custom"))
        # Reload from default rules.yaml
        count = engine.reload_rules()
        assert count > 0
        # Dynamic rule that was not in the file should be preserved
        assert any(r.name == "keep_me" for r in engine._rules.rules)


# ════════════════════════════════════════════════════════════════
# 3.3 — HardwareTrustGate
# ════════════════════════════════════════════════════════════════


class TestHardwareTrustGate:
    """Per-(device, channel) trust lifecycle and approval exemption."""

    def test_initial_level_is_untrusted(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel
        gate = HardwareTrustGate()
        assert gate.level("d", "c") == HardwareTrustLevel.UNTRUSTED

    def test_promotion_through_levels(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel
        gate = HardwareTrustGate(candidate_at=2, verified_at=5, production_at=10)
        for _ in range(2):
            gate.record_success("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.CANDIDATE
        for _ in range(3):
            gate.record_success("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.VERIFIED
        for _ in range(5):
            gate.record_success("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.PRODUCTION

    def test_demotion_on_consecutive_failures(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel
        gate = HardwareTrustGate(candidate_at=2, verified_at=5, demote_after=2)
        for _ in range(5):
            gate.record_success("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.VERIFIED
        gate.record_failure("d", "c")
        gate.record_failure("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.CANDIDATE

    def test_hard_failure_freezes_to_untrusted(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel
        gate = HardwareTrustGate(candidate_at=1)
        gate.record_success("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.CANDIDATE
        gate.record_failure("d", "c", hard=True)
        assert gate.level("d", "c") == HardwareTrustLevel.UNTRUSTED
        # Cannot recover from hard freeze
        gate.record_success("d", "c")
        assert gate.level("d", "c") == HardwareTrustLevel.UNTRUSTED

    def test_may_skip_approval_only_for_reversible_verified(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate
        gate = HardwareTrustGate(candidate_at=1, verified_at=3)
        for _ in range(3):
            gate.record_success("d", "c")
        # Reversible channel at VERIFIED -> may skip
        assert gate.may_skip_approval("d", "c", reversible=True) is True
        # Irreversible channel at VERIFIED -> must not skip
        assert gate.may_skip_approval("d", "c", reversible=False) is False

    def test_irreversible_always_requires_approval(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate
        gate = HardwareTrustGate(candidate_at=1, verified_at=2, production_at=5)
        for _ in range(10):
            gate.record_success("d", "c")
        # Even at PRODUCTION, irreversible channels require approval
        assert gate.may_skip_approval("d", "c", reversible=False) is False

    def test_allow_permanent_mirrors_may_skip(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate
        gate = HardwareTrustGate(candidate_at=1, verified_at=3)
        for _ in range(3):
            gate.record_success("d", "c")
        assert gate.allow_permanent("d", "c", reversible=True) is True
        assert gate.allow_permanent("d", "c", reversible=False) is False

    def test_trust_record_snapshot(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel
        gate = HardwareTrustGate(candidate_at=2)
        gate.record_success("d", "c")
        gate.record_success("d", "c")
        record = gate.trust_record("d", "c")
        assert record.device_id == "d"
        assert record.channel_id == "c"
        assert record.level == HardwareTrustLevel.CANDIDATE
        assert record.consecutive_ok == 2
        assert record.consecutive_fail == 0
        assert record.frozen is False

    def test_all_records(self) -> None:
        from leapflow.hardware.trust import HardwareTrustGate
        gate = HardwareTrustGate(candidate_at=1)
        gate.record_success("d1", "c1")
        gate.record_success("d2", "c2")
        records = gate.all_records()
        assert len(records) == 2

    def test_plugin_trust_ledger_integration(self) -> None:
        """Trust events forward to PluginTrustLedger when provided."""
        from leapflow.learning.plugin_trust import PluginTrustLedger
        from leapflow.hardware.trust import HardwareTrustGate

        ledger = PluginTrustLedger()
        gate = HardwareTrustGate(plugin_trust_ledger=ledger)
        gate.record_success("d", "c")
        # Should have forwarded a success to the plugin trust ledger
        plugin_id = "hw:d:c"
        assert ledger._consecutive_ok.get(plugin_id, 0) == 1


# ════════════════════════════════════════════════════════════════
# 3.5 — MCP transport capability validation
# ════════════════════════════════════════════════════════════════


class TestMcpCapabilityValidation:
    """MCP transport open() validates declared tools against server capabilities."""

    @pytest.mark.asyncio
    async def test_open_validates_against_server_tools(self) -> None:
        """open() fails when a declared tool is missing from the server."""
        from leapflow.hardware.transports.mcp import McpTransport
        from leapflow.hardware.transport import TransportError
        from leapflow.hardware.context import (
            HC_VERSION, Channel, Direction, Envelope,
            HardwareContext, HardwareEffect, TransportRef,
        )

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value={"ok": True})
        client.list_tools = AsyncMock(return_value=[
            {"name": "bench_read"},
            {"name": "bench_status"},
        ])
        transport = McpTransport({
            "server": "test",
            "read_tool": "bench_read",
            "write_tool": "bench_write",  # NOT in server's tool list
            "probe_tool": "bench_status",
            "client": client,
        })
        context = HardwareContext(
            device_id="test-dev",
            display_name="Test",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="mcp"),
            channels=(
                Channel(
                    channel_id="ch1",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
        )
        with pytest.raises(TransportError, match="does not advertise"):
            await transport.open(context)

    @pytest.mark.asyncio
    async def test_open_succeeds_when_all_tools_present(self) -> None:
        """open() succeeds when all declared tools exist on the server."""
        from leapflow.hardware.transports.mcp import McpTransport
        from leapflow.hardware.context import (
            HC_VERSION, Channel, Direction, Envelope,
            HardwareContext, HardwareEffect, TransportRef,
        )

        client = AsyncMock()
        client.call_tool = AsyncMock(return_value={"ok": True})
        client.list_tools = AsyncMock(return_value=[
            {"name": "bench_read"},
            {"name": "bench_write"},
            {"name": "bench_status"},
        ])
        transport = McpTransport({
            "server": "test",
            "read_tool": "bench_read",
            "write_tool": "bench_write",
            "probe_tool": "bench_status",
            "client": client,
        })
        context = HardwareContext(
            device_id="test-dev",
            display_name="Test",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="mcp"),
            channels=(
                Channel(
                    channel_id="ch1",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
        )
        status = await transport.open(context)
        assert status.connected

    @pytest.mark.asyncio
    async def test_open_degrades_gracefully_when_no_list_tools(self) -> None:
        """open() does not fail when the client lacks list_tools."""
        from leapflow.hardware.transports.mcp import McpTransport
        from leapflow.hardware.context import (
            HC_VERSION, Channel, Direction, Envelope,
            HardwareContext, HardwareEffect, TransportRef,
        )

        client = AsyncMock(spec=[])  # No list_tools or tools attribute
        client.call_tool = AsyncMock(return_value={"ok": True})
        transport = McpTransport({
            "server": "test",
            "read_tool": "bench_read",
            "write_tool": "bench_write",
            "client": client,
        })
        context = HardwareContext(
            device_id="test-dev",
            display_name="Test",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="mcp"),
            channels=(
                Channel(
                    channel_id="ch1",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
        )
        # Should not raise
        status = await transport.open(context)
        assert status.connected


class TestMcpToolExecutionPolicy:
    """MCP tools without x_leapflow do not fall back to mutating_idempotent."""

    def test_mcp_tool_without_metadata_is_external(self) -> None:
        from leapflow.engine.tool_execution import execution_policy_for

        @dataclass
        class FakeSpec:
            risk_level: str = ""
            mutates_state: bool = False
            idempotency_scope: str = ""
            effect_scope: str = ""
            category: str = "mcp"

        spec = FakeSpec()
        policy = execution_policy_for("some_mcp_tool", spec)
        assert policy == "external_side_effect"

    def test_mcp_tool_with_explicit_read_only_stays_read_only(self) -> None:
        from leapflow.engine.tool_execution import execution_policy_for

        @dataclass
        class FakeSpec:
            risk_level: str = "read_only"
            mutates_state: bool = False
            idempotency_scope: str = ""
            effect_scope: str = ""
            category: str = "mcp"

        spec = FakeSpec()
        policy = execution_policy_for("some_mcp_tool", spec)
        assert policy == "read_only"

    def test_non_mcp_tool_without_metadata_stays_idempotent(self) -> None:
        from leapflow.engine.tool_execution import execution_policy_for

        @dataclass
        class FakeSpec:
            risk_level: str = ""
            mutates_state: bool = False
            idempotency_scope: str = ""
            effect_scope: str = ""
            category: str = "general"

        spec = FakeSpec()
        policy = execution_policy_for("some_tool", spec)
        assert policy == "mutating_idempotent"


# ════════════════════════════════════════════════════════════════
# Kim#3 — TrustGate wired into HardwareTools._evaluate
# ════════════════════════════════════════════════════════════════


def _make_trust_bench(
    *,
    trust_skip_enabled: bool = True,
    reversible: bool = True,
    effect: str = "actuate",
):
    """Build a minimal HardwareTools + trust gate + real approval chain."""
    from leapflow.hardware.context import (
        HC_VERSION,
        Channel,
        ContextProvenance,
        Direction,
        Envelope,
        HardwareContext,
        HardwareEffect,
        TransportRef,
    )
    from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
    from leapflow.hardware.tools import HardwareTools
    from leapflow.hardware.trust import HardwareTrustGate
    from leapflow.security.approval import (
        ApprovalDecision,
        SessionAwareGate,
    )
    from leapflow.security.orchestrator import ApprovalOrchestrator
    from leapflow.security.policy import ApprovalPolicyEngine

    effect_enum = {
        "actuate": HardwareEffect.ACTUATE.value,
        "configure": HardwareEffect.CONFIGURE.value,
        "dispense": HardwareEffect.DISPENSE.value,
    }[effect]

    context = HardwareContext(
        device_id="dev",
        hc_version=HC_VERSION,
        display_name="Test Device",
        halt_supported=True,
        transport=TransportRef(
            kind="mock", config={"values": {"ch": 10.0}},
        ),
        channels=(
            Channel(
                channel_id="ch",
                direction=Direction.READWRITE.value,
                quantity="test",
                unit="unit",
                effect=effect_enum,
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=100.0,
                    reversible=reversible,
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="operator"),
    )

    class _StaticProvider:
        kind = "static"

        def __init__(self, ctx):
            self._ctx = ctx

        def discover(self):
            return (self._ctx,)

    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            require_describe_before_write=False,
            trust_skip_enabled=trust_skip_enabled,
        ),
        providers=[_StaticProvider(context)],
    )
    registry.load()

    from tests.test_hardware_governance import ScriptedHuman

    human = ScriptedHuman(ApprovalDecision.ALLOW_ONCE)
    gate = SessionAwareGate(human)
    orchestrator = ApprovalOrchestrator(
        gate, policy=ApprovalPolicyEngine(),
    )
    trust_gate = HardwareTrustGate(candidate_at=1, verified_at=3)
    tools = HardwareTools(
        registry,
        gate=orchestrator,
        session_id="s",
        hardware_trust_gate=trust_gate,
    )
    return tools, trust_gate, human


class TestTrustGateApprovalBypass:
    """HardwareTrustGate short-circuits approval for eligible channels."""

    @pytest.mark.asyncio
    async def test_reversible_verified_with_switch_on_skips_approval(self) -> None:
        """(a) Reversible channel at VERIFIED + switch on -> trust skip, no prompt."""
        tools, trust_gate, human = _make_trust_bench(
            trust_skip_enabled=True, reversible=True, effect="actuate",
        )
        # Build trust to VERIFIED (3 successes)
        for _ in range(3):
            trust_gate.record_success("dev", "ch")
        from leapflow.hardware.trust import HardwareTrustLevel
        assert trust_gate.level("dev", "ch") == HardwareTrustLevel.VERIFIED

        result = await tools.hw_actuate(device_id="dev", channel_id="ch", value=50.0)
        assert result["ok"] is True
        # The human was never prompted
        assert len(human.prompts) == 0

    @pytest.mark.asyncio
    async def test_irreversible_actuate_at_production_still_requires_approval(self) -> None:
        """(b) Irreversible ACTUATE at PRODUCTION trust + switch on -> full approval."""
        tools, trust_gate, human = _make_trust_bench(
            trust_skip_enabled=True, reversible=False, effect="actuate",
        )
        # Push to PRODUCTION
        for _ in range(20):
            trust_gate.record_success("dev", "ch")
        from leapflow.hardware.trust import HardwareTrustLevel
        assert trust_gate.level("dev", "ch") == HardwareTrustLevel.PRODUCTION

        result = await tools.hw_actuate(device_id="dev", channel_id="ch", value=50.0)
        assert result["ok"] is True
        # The human WAS prompted
        assert len(human.prompts) == 1

    @pytest.mark.asyncio
    async def test_irreversible_dispense_at_production_still_requires_approval(self) -> None:
        """(b) Irreversible DISPENSE at PRODUCTION trust + switch on -> full approval."""
        tools, trust_gate, human = _make_trust_bench(
            trust_skip_enabled=True, reversible=False, effect="dispense",
        )
        for _ in range(20):
            trust_gate.record_success("dev", "ch")

        result = await tools.hw_dispense(device_id="dev", channel_id="ch", value=50.0)
        assert result["ok"] is True
        assert len(human.prompts) == 1  # prompted

    @pytest.mark.asyncio
    async def test_switch_default_off_always_prompts(self) -> None:
        """(c) Default config (trust_skip_enabled=False) -> always full approval."""
        tools, trust_gate, human = _make_trust_bench(
            trust_skip_enabled=False, reversible=True, effect="actuate",
        )
        for _ in range(10):
            trust_gate.record_success("dev", "ch")

        result = await tools.hw_actuate(device_id="dev", channel_id="ch", value=50.0)
        assert result["ok"] is True
        assert len(human.prompts) == 1  # prompted despite PRODUCTION trust

    @pytest.mark.asyncio
    async def test_trust_skip_audit_record(self) -> None:
        """(b) Trust skip produces an audit entry with trust_skip=True."""
        from leapflow.hardware.audit import HardwareAuditLog
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            audit = HardwareAuditLog(Path(tmp) / "audit.ndjson")
            tools, trust_gate, human = _make_trust_bench(
                trust_skip_enabled=True, reversible=True, effect="configure",
            )
            # Replace audit
            tools._audit = audit
            for _ in range(3):
                trust_gate.record_success("dev", "ch")

            result = await tools.hw_configure(device_id="dev", channel_id="ch", value=50.0)
            assert result["ok"] is True

            entries = audit.read_entries()
            trust_entries = [e for e in entries if e.action == "trust_skip"]
            assert len(trust_entries) == 1
            assert "trust_skip=True" in trust_entries[0].outcome
            assert "VERIFIED" in trust_entries[0].outcome

    @pytest.mark.asyncio
    async def test_successful_write_accrues_trust(self) -> None:
        """(d) A successful write increments the trust gate."""
        tools, trust_gate, human = _make_trust_bench(
            trust_skip_enabled=False, reversible=True, effect="configure",
        )
        from leapflow.hardware.trust import HardwareTrustLevel
        assert trust_gate.level("dev", "ch") == HardwareTrustLevel.UNTRUSTED

        # Scripted human will approve
        result = await tools.hw_configure(device_id="dev", channel_id="ch", value=50.0)
        assert result["ok"] is True
        # Trust was incremented by the write path
        rec = trust_gate.trust_record("dev", "ch")
        assert rec.consecutive_ok == 1

    @pytest.mark.asyncio
    async def test_below_verified_still_prompts(self) -> None:
        """A CANDIDATE channel still requires full approval even with switch on."""
        tools, trust_gate, human = _make_trust_bench(
            trust_skip_enabled=True, reversible=True, effect="actuate",
        )
        trust_gate.record_success("dev", "ch")  # Only CANDIDATE (at=1)
        from leapflow.hardware.trust import HardwareTrustLevel
        assert trust_gate.level("dev", "ch") == HardwareTrustLevel.CANDIDATE

        result = await tools.hw_actuate(device_id="dev", channel_id="ch", value=50.0)
        assert result["ok"] is True
        assert len(human.prompts) == 1  # prompted

    def test_config_defaults_to_false(self) -> None:
        """HardwareSettings.trust_skip_enabled defaults to False."""
        from leapflow.hardware.registry import HardwareSettings
        assert HardwareSettings().trust_skip_enabled is False


# ════════════════════════════════════════════════════════════════
# Kim#2 — MCP sync list_tools compatibility
# ════════════════════════════════════════════════════════════════


class TestMcpSyncListToolsCompat:
    """MCP _validate_server_capabilities works with sync list_tools."""

    @pytest.mark.asyncio
    async def test_sync_list_tools_success(self) -> None:
        """A sync client returning a plain list is accepted."""
        from leapflow.hardware.transports.mcp import McpTransport
        from leapflow.hardware.context import (
            HC_VERSION, Channel, Direction, Envelope,
            HardwareContext, HardwareEffect, TransportRef,
        )

        class SyncClient:
            def list_tools(self):
                return [{"name": "bench_read"}, {"name": "bench_write"}]

            async def call_tool(self, tool, args):
                return {"ok": True, "value": 42}

        transport = McpTransport({
            "server": "test",
            "read_tool": "bench_read",
            "write_tool": "bench_write",
            "client": SyncClient(),
        })
        context = HardwareContext(
            device_id="test-dev",
            display_name="Test",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="mcp"),
            channels=(
                Channel(
                    channel_id="ch1",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
        )
        status = await transport.open(context)
        assert status.connected

    @pytest.mark.asyncio
    async def test_sync_list_tools_mismatch_fails(self) -> None:
        """A sync client missing a declared tool triggers mcp_capability_mismatch."""
        from leapflow.hardware.transports.mcp import McpTransport
        from leapflow.hardware.transport import TransportError
        from leapflow.hardware.context import (
            HC_VERSION, Channel, Direction, Envelope,
            HardwareContext, HardwareEffect, TransportRef,
        )

        class SyncClient:
            def list_tools(self):
                return [{"name": "bench_read"}]  # Missing bench_write

            async def call_tool(self, tool, args):
                return {"ok": True}

        transport = McpTransport({
            "server": "test",
            "read_tool": "bench_read",
            "write_tool": "bench_write",
            "client": SyncClient(),
        })
        context = HardwareContext(
            device_id="test-dev",
            display_name="Test",
            hc_version=HC_VERSION,
            transport=TransportRef(kind="mcp"),
            channels=(
                Channel(
                    channel_id="ch1",
                    direction=Direction.READWRITE,
                    effect=HardwareEffect.CONFIGURE.value,
                    envelope=Envelope(declared=True, min_value=0, max_value=100, reversible=True),
                ),
            ),
        )
        with pytest.raises(TransportError, match="does not advertise"):
            await transport.open(context)
