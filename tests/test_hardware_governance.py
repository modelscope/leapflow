"""End-to-end governance chain for hardware commands.

Every case here drives the *production* ``ApprovalOrchestrator``, ``ApprovalPolicyEngine``,
``CompositeRiskClassifier``, and grant store. Only the human surface is a stand-in:
a scripted gate answering as a person would. That split is deliberate -- a fake that
reimplements the orchestrator's own logic would keep agreeing with the caller's
mistake, which is exactly how a dead gate stays green.

The declarations below deliberately mirror a real bench: a liquid handler whose
aspirate channel consumes an irreversible resource behind two interlocks, and a
bench node with a rate-limited actuator. They live in the test rather than in
production code, because the protocol must not know what a liquid handler is.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from leapflow.engine.tool_execution import effect_is_uncertain_on_failure, execution_policy_for
from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Interlock,
    TransportRef,
)
from leapflow.hardware.transport import SIDE_EFFECT_NONE
from leapflow.hardware.transports.mcp import set_mcp_client_provider
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.risk import build_risk_classifier
from leapflow.hardware.tools import HardwareTools, build_hardware_tools
from leapflow.security.actions import ActionDescriptor, ActionKind
from leapflow.security.approval import ApprovalDecision, ApprovalRequest, SessionAwareGate
from leapflow.security.grants import ApprovalScope, grant_key
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.security.permission_failures import is_permission_hard_stop_payload
from leapflow.security.policy import ApprovalPolicyEngine
from leapflow.security.risk import DefaultRiskClassifier, RiskLevel
from leapflow.tools.name_resolver import ToolRegistry

SESSION = "session-under-test"


# ════════════════════════════════════════════════════════════════
# Realistic declarations
# ════════════════════════════════════════════════════════════════


def liquid_handler_context(*, verified: bool = True) -> HardwareContext:
    """A liquid handler with an irreversible dispense channel behind interlocks."""
    return HardwareContext(
        device_id="fluent_p1",
        hc_version=HC_VERSION,
        display_name="Tecan Fluent",
        vendor="Tecan",
        model="Fluent",
        location="bench-2",
        halt_supported=True,
        transport=TransportRef(
            kind="mock",
            config={
                "values": {
                    "tip_state": True,
                    "deck_state": "clear",
                    "plate_temp": 22.4,
                    "aspirate": 0.0,
                },
                "halt_supported": True,
            },
        ),
        notes=(
            "Protein samples foam. A failed aspirate has already agitated the well: "
            "retrying in the same well produces more bubbles, not a clean retry. Move to "
            "a fresh well or wait for the foam to settle. This is a physical failure, "
            "not a software error."
        ),
        interlocks=(
            Interlock(
                interlock_id="tip_present",
                channel_id="tip_state",
                operator="eq",
                value=True,
                description="A tip must be mounted before aspirating.",
            ),
            Interlock(
                interlock_id="deck_clear",
                channel_id="deck_state",
                operator="eq",
                value="clear",
                description="The robotic arm must not be over the deck.",
            ),
        ),
        channels=(
            Channel(
                channel_id="tip_state",
                direction=Direction.READ.value,
                quantity="state.tip_present",
                unit="bool",
                envelope=Envelope(declared=True),
            ),
            Channel(
                channel_id="deck_state",
                direction=Direction.READ.value,
                quantity="state.deck",
                envelope=Envelope(declared=True),
            ),
            Channel(
                channel_id="plate_temp",
                direction=Direction.READ.value,
                quantity="temperature.plate",
                unit="degC",
                sample_rate_hz=10.0,
                envelope=Envelope(declared=True, min_value=4.0, max_value=99.0),
            ),
            Channel(
                channel_id="aspirate",
                direction=Direction.WRITE.value,
                quantity="volume.aspirate",
                unit="uL_per_s",
                effect=HardwareEffect.DISPENSE.value,
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=200.0,
                    max_rate=50.0,
                    quantization=0.1,
                    reversible=False,
                    requires_interlocks=("tip_present", "deck_clear"),
                    notes=(
                        "Aqueous reagents run well near 140 uL/s. Viscous protein samples "
                        "such as BSA require roughly 10 uL/s; faster rates foam the sample."
                    ),
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="lab-lead" if verified else ""),
    )


def bench_node_context() -> HardwareContext:
    """A bench node with a rate-limited, reversible actuator."""
    return HardwareContext(
        device_id="bench_node",
        hc_version=HC_VERSION,
        display_name="Bench node",
        location="desk",
        halt_supported=True,
        transport=TransportRef(
            kind="mock", config={"values": {"fan_duty": 10.0, "cpu_temp": 41.2, "aux_level": 0.0}}
        ),
        channels=(
            Channel(
                channel_id="cpu_temp",
                direction=Direction.READ.value,
                quantity="temperature.cpu",
                unit="degC",
                sample_rate_hz=1.0,
                envelope=Envelope(declared=True, min_value=0.0, max_value=95.0),
            ),
            Channel(
                channel_id="fan_duty",
                direction=Direction.READWRITE.value,
                quantity="ratio.fan_duty",
                unit="percent",
                effect=HardwareEffect.ACTUATE.value,
                verify_after_write=True,
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=80.0,
                    max_rate=20.0,
                    quantization=1.0,
                    settling_time_s=2.0,
                    reversible=True,
                    notes="Above 80 percent the bearing overheats.",
                ),
            ),
            Channel(
                channel_id="aux_level",
                direction=Direction.READWRITE.value,
                quantity="ratio.aux_level",
                unit="percent",
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(
                    declared=True, min_value=0.0, max_value=100.0, reversible=True
                ),
            ),
            Channel(
                channel_id="indicator",
                direction=Direction.WRITE.value,
                quantity="state.indicator",
                unit="bool",
                effect=HardwareEffect.CONFIGURE.value,
                envelope=Envelope(declared=True, reversible=True),
            ),
        ),
        provenance=ContextProvenance(verified_by="jason"),
    )


# ════════════════════════════════════════════════════════════════
# Harness -- real governance, scripted human
# ════════════════════════════════════════════════════════════════


class ScriptedHuman:
    """Stands in for the person at the prompt. The only fake in the chain."""

    def __init__(self, *decisions: ApprovalDecision) -> None:
        self._decisions = list(decisions)
        self.prompts: list[ApprovalRequest] = []

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.prompts.append(request)
        if not self._decisions:
            return ApprovalDecision.DENY
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


class _StaticProvider:
    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


class Bench:
    """A loaded registry plus the real approval chain wired around it."""

    def __init__(
        self,
        *contexts: HardwareContext,
        decisions: tuple[ApprovalDecision, ...] = (ApprovalDecision.ALLOW_ONCE,),
        bypass: bool = False,
        require_describe: bool = False,
        **setting_overrides: Any,
    ) -> None:
        self.registry = HardwareRegistry(
            HardwareSettings(
                enabled=True,
                require_describe_before_write=require_describe,
                **setting_overrides,
            ),
            providers=[_StaticProvider(*contexts)],
        )
        self.report = self.registry.load()
        self.human = ScriptedHuman(*decisions)
        self.gate = SessionAwareGate(self.human)
        self.orchestrator = ApprovalOrchestrator(
            self.gate,
            risk_classifier=build_risk_classifier(self.registry),
            policy=ApprovalPolicyEngine(bypass=bypass),
        )
        self.tools = HardwareTools(self.registry, gate=self.orchestrator, session_id=SESSION)

    async def transport(self, device_id: str) -> Any:
        return await self.registry.transport(device_id)

    @property
    def audit_entries(self) -> tuple[dict[str, Any], ...]:
        return self.orchestrator.audit.entries


async def _describe(bench: Bench, device_id: str) -> None:
    """Satisfy the describe-before-write precondition the way a model would."""
    await bench.tools.hw_describe(device_id=device_id)


def with_transport_config(context: HardwareContext, **overrides: Any) -> HardwareContext:
    """Return *context* with its transport config merged with *overrides*.

    Lets a test change device behaviour -- inject a failure, remove halt support,
    open an interlock -- without restating the whole declaration.
    """
    from dataclasses import replace

    merged = {**dict(context.transport.config), **overrides}
    return replace(context, transport=replace(context.transport, config=merged))


def with_values(context: HardwareContext, **values: Any) -> HardwareContext:
    """Return *context* with individual channel values overridden."""
    current = dict(dict(context.transport.config).get("values") or {})
    current.update(values)
    return with_transport_config(context, values=current)


# ════════════════════════════════════════════════════════════════
# T1 -- the foaming-retry scenario
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_t1_failed_dispense_is_reported_as_uncertain_not_retried() -> None:
    """A failed physical dispense must never look safe to repeat.

    This is the exact shape of the reported failure where an agent retried an
    aspirate in the same well, agitating the sample and producing more bubbles. The
    machine form of that bug is a partial side effect classified as replayable, so
    the assertions below pin every link: the verdict survives into the result, the
    result says what to do next, and the operator's physical explanation reaches the
    model rather than staying in a YAML file.
    """
    context = with_transport_config(
        liquid_handler_context(),
        failures=[
            {
                "channel_id": "aspirate",
                "on_call": 1,
                "repeat": True,
                "side_effect_state": "partial",
                "error": "fluid detection error",
                "failure_code": "fluid_detection",
            }
        ],
    )
    bench = Bench(context, decisions=(ApprovalDecision.ALLOW_SESSION,))
    await _describe(bench, "fluent_p1")

    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )

    assert result["ok"] is False
    # The effect verdict survives from the transport into the tool result.
    assert result["side_effect_state"] == "partial"
    assert result["effect_uncertain"] is True
    # The result tells the next turn to verify rather than repeat.
    assert "do not repeat" in result["next_step"].lower()
    assert "irreversible" in result["next_step"].lower()

    # And nothing retried on its own: exactly one attempt reached the device.
    transport = await bench.transport("fluent_p1")
    assert transport.write_attempts("aspirate") == 1
    assert transport.write_log == ()


@pytest.mark.asyncio
async def test_t1_physical_explanation_reaches_the_model() -> None:
    """Operator knowledge is only useful if the model actually receives it."""
    bench = Bench(liquid_handler_context())
    described = await bench.tools.hw_describe(device_id="fluent_p1")
    reference = described["reference"]
    assert "physical failure" in reference
    assert "not a software error" in reference
    assert "more bubbles" in reference
    # The machine-checkable verdict is rendered from the declaration, not prose.
    assert "NOT reversible" in reference


def test_t1_write_tools_are_classified_as_non_replayable() -> None:
    """The regression nail: a hardware write must not be "safe to repeat".

    ``execution_policy_for`` decides whether the recovery layer may replay a failed
    call. An unannotated tool falls through to ``mutating_idempotent`` -- "re-running
    converges" -- which is the wrong default for anything physical. This asserts the
    declared metadata actually produces the strict policy, through the same registry
    the engine builds.
    """
    registry_stub = HardwareRegistry(HardwareSettings(enabled=True))
    definitions = [
        metadata.to_openai_schema()
        for metadata in build_hardware_tools(HardwareTools(registry_stub))
    ]
    handlers = {
        definition["function"]["name"]: (lambda **_: None) for definition in definitions
    }
    resolver = ToolRegistry.from_definitions(definitions, handlers)

    for tool_name in ("hw_configure", "hw_actuate", "hw_dispense"):
        spec = resolver.specs[tool_name]
        policy = execution_policy_for(tool_name, spec)
        assert policy == "external_side_effect", (
            f"{tool_name} resolved to {policy!r}; a physical write classified as "
            "idempotent would let a failed command be replayed"
        )
        assert effect_is_uncertain_on_failure(policy) is True


def test_t1_a_declared_external_risk_now_reaches_the_execution_policy() -> None:
    """``risk_level="external"`` is honoured, so it no longer needs a second key.

    This test previously asserted the opposite -- that the declared value was
    re-inferred away -- and passed for exactly that reason. It pinned the defect
    instead of the requirement, which is why nothing ever pushed the fix: a test
    that agrees with the bug stays green forever.

    The write metadata still declares ``effect_scope`` as well, because the two
    keys answer different questions and either one alone should be sufficient to
    keep a physical command out of a replayable policy.
    """
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "hw_dispense",
                "description": "declares risk_level only",
                "parameters": {"type": "object", "properties": {}},
                "x_leapflow": {"risk_level": "external", "mutates_state": True},
            },
        }
    ]
    resolver = ToolRegistry.from_definitions(definitions, {"hw_dispense": lambda **_: None})
    spec = resolver.specs["hw_dispense"]
    assert spec.risk_level == "external"
    assert execution_policy_for("hw_dispense", spec) == "external_side_effect"
    assert effect_is_uncertain_on_failure("external_side_effect") is True


def test_t1_a_graded_risk_level_never_buys_the_read_only_policy() -> None:
    """A disclosure grade is not an effect classification.

    ``x_leapflow.risk_level`` carries two vocabularies: graded for disclosure
    (``none`` .. ``high``) and three-valued for execution. Copying ``"medium"``
    across would land a value outside ``RiskLevel`` in a typed field and match none
    of the comparisons that read it -- which is why the declaration is consulted
    rather than honoured wholesale. What must never happen is the reverse: a tool
    that declares any graded risk being resolved to ``read_only``, a policy that
    skips the execution ledger, parallelises freely and is exempt from side-effect
    gating.
    """
    for graded in ("low", "medium", "high"):
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "bench_probe",
                    "description": f"declares {graded} risk and nothing else",
                    "parameters": {"type": "object", "properties": {}},
                    "x_leapflow": {"risk_level": graded},
                },
            }
        ]
        resolver = ToolRegistry.from_definitions(definitions, {})
        spec = resolver.specs["bench_probe"]
        assert spec.risk_level == "mutating", graded
        assert execution_policy_for("bench_probe", spec) != "read_only", graded


def test_t1_an_undeclared_tool_is_not_assumed_effect_free() -> None:
    """Absence of a declaration is not a claim of safety.

    A third-party tool with an innocuous name used to resolve to ``read_only``,
    because the name matched no fragment in a hardcoded list of mutating words.
    Whether a call was gated therefore depended on the vocabulary its author
    happened to pick.
    """
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "fetch_report",
                "description": "declares no x_leapflow metadata at all",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    resolver = ToolRegistry.from_definitions(definitions, {})
    spec = resolver.specs["fetch_report"]
    assert spec.risk_level == "mutating"
    assert execution_policy_for("fetch_report", spec) != "read_only"


def test_t1_an_explicit_no_effect_claim_outranks_the_name() -> None:
    """``test_run`` contains "run" but is an inspection.

    The declaration is explicit and a substring is a guess, so the declaration
    wins. A mutating bridge still overrides both, since a declaration contradicted
    by the handler's own answer is the stale one.
    """
    def _define(name: str, **metadata: object) -> ToolRegistry:
        return ToolRegistry.from_definitions(
            [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                        "x_leapflow": metadata,
                    },
                }
            ],
            {},
        )

    claimed = _define("suite_run", risk_level="read_only")
    assert claimed.specs["suite_run"].risk_level == "read_only"

    contradicted = _define("suite_run", risk_level="read_only", mutates_state=True)
    assert contradicted.specs["suite_run"].risk_level == "mutating"


def test_t1_read_tools_stay_cheap_and_ungated() -> None:
    """Reads must not inherit the write tools' policy, or observation gets gated."""
    registry_stub = HardwareRegistry(HardwareSettings(enabled=True))
    for metadata in build_hardware_tools(HardwareTools(registry_stub)):
        if metadata.name in {"hw_list", "hw_describe", "hw_read", "hw_status"}:
            assert metadata.x_leapflow["requires_approval"] is False
            assert metadata.mutates_state is False


# ════════════════════════════════════════════════════════════════
# T2 -- hardline is above every grant and bypass
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_t2_out_of_envelope_write_is_denied_without_prompting() -> None:
    bench = Bench(bench_node_context())
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=95.0)
    assert result["ok"] is False
    assert result["failure_code"] == "approval_denied"
    assert "hardline" in result["error"].lower() or "prohibited" in result["error"].lower()
    # A command that cannot succeed must never reach a human.
    assert bench.human.prompts == []


@pytest.mark.asyncio
async def test_t2_allow_all_session_cannot_open_a_hardline() -> None:
    """Session-wide consent must not reach past the hardline boundary."""
    bench = Bench(
        bench_node_context(),
        decisions=(ApprovalDecision.ALLOW_ALL_SESSION,),
    )
    await _describe(bench, "bench_node")
    # First, an in-envelope command that arms the session bypass.
    allowed = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=40.0
    )
    assert allowed["ok"] is True

    # With the bypass armed, an out-of-envelope command must still be refused.
    denied = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=150.0
    )
    assert denied["ok"] is False
    assert denied["failure_code"] == "approval_denied"


@pytest.mark.asyncio
async def test_t2_approval_bypass_config_cannot_open_a_hardline() -> None:
    """The most permissive configuration possible still respects the hardline."""
    bench = Bench(bench_node_context(), bypass=True)
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=1000.0
    )
    assert result["ok"] is False
    assert bench.human.prompts == []


@pytest.mark.asyncio
async def test_t2_unsatisfied_interlock_is_a_hardline() -> None:
    bench = Bench(with_values(liquid_handler_context(), tip_state=False))
    await _describe(bench, "fluent_p1")
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is False
    assert bench.human.prompts == []
    transport = await bench.transport("fluent_p1")
    assert transport.write_log == ()


# ════════════════════════════════════════════════════════════════
# IC-1 -- device readiness is a fail-closed, executable hard stop
#
# Readiness ("must be homed/initialised/calibrated first") is declared by
# pointing a channel's ``requires_interlocks`` at an ``Interlock`` on a ready
# channel -- no new field. An unmet precondition must refuse the write before
# consent is ever sought (feasibility precedes consent) with an executable
# repair, and must not touch a device that declares no such precondition.
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_ic1_unready_device_is_hard_stopped_with_executable_repair() -> None:
    """An unmet readiness precondition is a deterministic, actionable hard stop.

    The command is refused before the gate is consulted, the failure names the
    exact precondition and the init to run, and the shared permission authority
    recognises it as a turn-stopping condition -- not a "retry and hope" error.
    """
    bench = Bench(with_values(liquid_handler_context(), tip_state=False))
    await _describe(bench, "fluent_p1")
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is False
    # Feasibility precedes consent: refused deterministically, no human asked.
    assert bench.human.prompts == []
    assert result["failure_code"] == "not_ready"
    assert result["failure_class"] == "device_not_ready"
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    # The executable repair names the unmet precondition, its source channel, and
    # the init routine to run before retrying.
    error = result["error"]
    assert "tip_present" in error
    assert "tip_state" in error
    assert "A tip must be mounted" in error
    assert "calibration" in error or "initialization" in error
    # Machine-readable repair mirrors the prose so a caller can act on it.
    unmet = result["repair"]["unmet"]
    assert [item["interlock_id"] for item in unmet] == ["tip_present"]
    assert unmet[0]["channel_id"] == "tip_state"
    # Recognised as a hard stop by the single authority engine and TUI consult.
    assert is_permission_hard_stop_payload(result) is True
    # Nothing reached the device.
    transport = await bench.transport("fluent_p1")
    assert transport.write_log == ()


@pytest.mark.asyncio
async def test_ic1_write_is_released_once_readiness_is_satisfied() -> None:
    """The identical command proceeds once every readiness precondition holds.

    Readiness gates feasibility, not consent: with the device ready the same
    write enters the normal approval path and, once approved, reaches the device.
    """
    bench = Bench(liquid_handler_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "fluent_p1")
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is True
    # A human was asked exactly once: the readiness hard stop did not pre-empt it.
    assert len(bench.human.prompts) == 1
    transport = await bench.transport("fluent_p1")
    assert transport.write_log != ()


@pytest.mark.asyncio
async def test_ic1_channel_without_readiness_precondition_is_unaffected() -> None:
    """A channel that declares no readiness interlock never sees the hard stop.

    Regression guard: the gate is scoped to declared preconditions, so a device
    with none goes straight to the normal approval path.
    """
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=20.0
    )
    assert result["ok"] is True
    assert result.get("failure_code") != "not_ready"
    assert len(bench.human.prompts) == 1
    transport = await bench.transport("bench_node")
    assert transport.write_log != ()


@pytest.mark.asyncio
async def test_t2_effect_class_mismatch_is_refused() -> None:
    """Defence in depth: the tool name and the declaration must agree."""
    bench = Bench(liquid_handler_context())
    await _describe(bench, "fluent_p1")
    result = await bench.tools.hw_configure(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is False
    assert result["failure_code"] == "effect_class_mismatch"
    assert "hw_dispense" in result["error"]


# ════════════════════════════════════════════════════════════════
# T3 -- undeclared and unverified contexts
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_t3_undeclared_envelope_blocks_writes_without_prompting() -> None:
    context = HardwareContext(
        device_id="loose_device",
        hc_version=HC_VERSION,
        halt_supported=True,
        transport=TransportRef(kind="mock", config={"values": {"knob": 0.0}}),
        channels=(
            Channel(
                channel_id="knob",
                direction=Direction.WRITE.value,
                effect=HardwareEffect.CONFIGURE.value,
                envelope=Envelope(declared=False),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )
    bench = Bench(context)
    await _describe(bench, "loose_device")
    result = await bench.tools.hw_configure(device_id="loose_device", channel_id="knob", value=1.0)
    assert result["ok"] is False
    assert result["failure_code"] == "channel_not_writable"
    assert bench.human.prompts == []


@pytest.mark.asyncio
async def test_t3_unverified_context_blocks_writes_by_default() -> None:
    bench = Bench(liquid_handler_context(verified=False))
    await _describe(bench, "fluent_p1")
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is False
    assert bench.human.prompts == []


@pytest.mark.asyncio
async def test_t3_unverified_context_says_so_in_its_reference() -> None:
    bench = Bench(liquid_handler_context(verified=False))
    described = await bench.tools.hw_describe(device_id="fluent_p1")
    assert "UNVERIFIED" in described["reference"]
    assert described["writable_channels"] == []


# ════════════════════════════════════════════════════════════════
# Gate failure modes -- fail closed
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_absent_gate_denies_rather_than_proceeding() -> None:
    """No gate installed means deny. A missing gate is not an open door."""
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, require_describe_before_write=False),
        providers=[_StaticProvider(bench_node_context())],
    )
    registry.load()
    tools = HardwareTools(registry, gate=None, session_id=SESSION)
    result = await tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    assert result["ok"] is False
    assert "configuration fault" in result["error"]
    transport = await registry.transport("bench_node")
    assert transport.write_log == ()


@pytest.mark.asyncio
async def test_raising_gate_denies_rather_than_propagating() -> None:
    """A broken gate must never become an open door."""

    class _ExplodingGate:
        async def evaluate(self, descriptor: ActionDescriptor) -> Any:
            raise RuntimeError("approval subsystem is down")

    registry = HardwareRegistry(
        HardwareSettings(enabled=True, require_describe_before_write=False),
        providers=[_StaticProvider(bench_node_context())],
    )
    registry.load()
    tools = HardwareTools(registry, gate=_ExplodingGate(), session_id=SESSION)
    result = await tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    assert result["ok"] is False
    assert "failed while assessing" in result["error"]
    transport = await registry.transport("bench_node")
    assert transport.write_log == ()


@pytest.mark.asyncio
async def test_user_denial_message_reaches_the_caller_verbatim() -> None:
    """A denial is terminal and must not be softened into a generic tool error."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.DENY,))
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=30.0
    )
    assert result["ok"] is False
    assert "User denied" in result["error"]
    assert "Do not retry" in result["error"]


# ════════════════════════════════════════════════════════════════
# Grant scope -- envelope band, not numeric value
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_in_band_values_reuse_one_grant() -> None:
    """One consent covers the channel's declared band, not a single value.

    Without this, an overnight run would prompt per setpoint, and a person asked to
    confirm hundreds of routine operations stops reading the prompts.
    """
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_SESSION,))
    await _describe(bench, "bench_node")
    # aux_level declares no max_rate, so this exercises grant reuse rather than pacing.
    first = await bench.tools.hw_actuate(device_id="bench_node", channel_id="aux_level", value=25.0)
    second = await bench.tools.hw_actuate(device_id="bench_node", channel_id="aux_level", value=61.0)
    third = await bench.tools.hw_actuate(device_id="bench_node", channel_id="aux_level", value=12.0)
    assert [first["ok"], second["ok"], third["ok"]] == [True, True, True]
    # The human was asked exactly once.
    assert len(bench.human.prompts) == 1


def test_widening_an_envelope_invalidates_the_narrower_grant() -> None:
    """Consent granted under narrow limits must not survive their widening."""
    narrow = Envelope(declared=True, min_value=0.0, max_value=80.0, max_rate=20.0, reversible=True)
    wide = Envelope(declared=True, min_value=0.0, max_value=200.0, max_rate=20.0, reversible=True)

    def _key(envelope: Envelope) -> str:
        descriptor = ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id="bench_node",
            channel_id="fan_duty",
            value=40.0,
            envelope_band=envelope.band_key(),
        )
        return grant_key(descriptor, ApprovalScope.SESSION)

    assert _key(narrow) != _key(wide)


def test_different_channels_never_share_a_grant() -> None:
    envelope = Envelope(declared=True, min_value=0.0, max_value=80.0)

    def _key(channel_id: str) -> str:
        descriptor = ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id="bench_node",
            channel_id=channel_id,
            value=10.0,
            envelope_band=envelope.band_key(),
        )
        return grant_key(descriptor, ApprovalScope.SESSION)

    assert _key("fan_duty") != _key("other_axis")


# ════════════════════════════════════════════════════════════════
# Emergency stop
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_rate_limit_refuses_a_too_fast_second_command() -> None:
    """``max_rate`` must actually be enforced, not merely declared.

    A declared limit that nothing enforces is worse than no limit: the reference
    document promises it, a reviewer reads it as protection, and the device is
    unprotected. ``fan_duty`` allows 20 percent per second, so 20 -> 75 back to back
    is roughly 55 in well under a second.
    """
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ALL_SESSION,))
    await _describe(bench, "bench_node")

    first = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    assert first["ok"] is True

    second = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=75.0)
    assert second["ok"] is False
    assert second["failure_code"] == "rate_limited"
    # Nothing landed, and the refusal is actionable rather than terminal.
    assert second["side_effect_state"] == "none"
    assert second["retry_after_s"] > 0
    assert "wait" in second["error"].lower()

    transport = await bench.transport("bench_node")
    assert transport.write_log == (("fan_duty", 20.0),)


@pytest.mark.asyncio
async def test_rate_limited_command_succeeds_after_waiting() -> None:
    """Pacing is "not yet", not "never" -- the same command must become valid.

    This is why a rate violation is not a hardline: a hardline is unbypassable and
    terminal, which would be the wrong verdict for something the caller can simply
    wait out.
    """
    import asyncio

    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ALL_SESSION,))
    await _describe(bench, "bench_node")
    assert (
        await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    )["ok"] is True

    refused = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=24.0
    )
    assert refused["failure_code"] == "rate_limited"

    await asyncio.sleep(refused["retry_after_s"] + 0.02)
    retried = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=24.0
    )
    assert retried["ok"] is True


@pytest.mark.asyncio
async def test_channel_without_a_rate_limit_is_not_paced() -> None:
    """Only channels that declare a slew limit are paced."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ALL_SESSION,))
    await _describe(bench, "bench_node")
    for value in (10.0, 90.0, 5.0):
        result = await bench.tools.hw_actuate(
            device_id="bench_node", channel_id="aux_level", value=value
        )
        assert result["ok"] is True, result


@pytest.mark.asyncio
async def test_first_command_is_not_rate_checked() -> None:
    """There is no interval to measure before the first command.

    It is still bounded by min/max and still requires consent; refusing it would make
    every rate-limited channel unusable from a cold start.
    """
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=80.0)
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_denied_command_does_not_move_the_rate_baseline() -> None:
    """Refusing a command must not relax the limit on the next one."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ALL_SESSION,))
    await _describe(bench, "bench_node")
    assert (
        await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=10.0)
    )["ok"] is True
    # Out of envelope: hardline-denied, and must not become the new baseline.
    assert (
        await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=200.0)
    )["ok"] is False
    # Still measured from 10, so a 60-point jump is still too fast.
    third = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=70.0
    )
    assert third["failure_code"] == "rate_limited"


@pytest.mark.asyncio
async def test_failed_write_does_not_move_the_rate_baseline() -> None:
    """A failed command's value is not a measurement of anything."""
    context = with_transport_config(
        bench_node_context(),
        failures=[{"channel_id": "aux_level", "on_call": 2, "side_effect_state": "unknown"}],
    )
    bench = Bench(context, decisions=(ApprovalDecision.ALLOW_ALL_SESSION,))
    await _describe(bench, "bench_node")
    assert (
        await bench.tools.hw_actuate(device_id="bench_node", channel_id="aux_level", value=10.0)
    )["ok"] is True
    failed = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="aux_level", value=25.0
    )
    assert failed["ok"] is False
    assert failed["effect_uncertain"] is True
    # Baseline is still the 10 that landed, not the 25 that did not.
    baseline = bench.registry.last_command("bench_node", "aux_level")
    assert baseline is not None and baseline[0] == pytest.approx(10.0)


class _RogueTransport:
    """A driver that breaks the contract by raising the wrong exception type."""

    kind = "rogue"

    def __init__(self, config: Any = None) -> None:
        self._open = False

    async def open(self, context: HardwareContext) -> Any:
        from leapflow.hardware.transport import TransportStatus

        self._open = True
        return TransportStatus(connected=True, halt_supported=True)

    async def close(self) -> Any:
        from leapflow.hardware.transport import TransportStatus

        return TransportStatus(connected=False, halt_supported=True)

    async def read(self, channel_id: str) -> Any:
        raise ValueError("driver forgot to wrap its errors")

    async def write(self, channel_id: str, value: Any) -> Any:
        raise ValueError("driver forgot to wrap its errors")

    async def probe(self) -> Any:
        from leapflow.hardware.transport import TransportStatus

        return TransportStatus(connected=self._open, halt_supported=True)

    async def halt(self) -> Any:
        from leapflow.hardware.transport import TransportStatus

        return TransportStatus(connected=self._open, halt_supported=True)


@pytest.fixture
def rogue_transport_kind():
    """Register the rogue driver, restoring the process-global table afterwards."""
    from leapflow.hardware.transports import register_transport

    globals()["_build_rogue"] = lambda config=None: _RogueTransport(config)
    undo = register_transport("rogue_test", f"{__name__}:_build_rogue")
    try:
        yield "rogue_test"
    finally:
        undo()


@pytest.mark.asyncio
async def test_driver_raising_the_wrong_exception_is_reported_as_uncertain(
    rogue_transport_kind: str,
) -> None:
    """A misbehaving driver must not turn into an unhandled crash.

    Reporting UNKNOWN is the safe reading: it blocks replay exactly as COMMITTED
    does, whereas an escaping exception carries no effect verdict at all and invites
    the caller to simply try again.
    """
    from dataclasses import replace

    context = replace(
        bench_node_context(), transport=TransportRef(kind=rogue_transport_kind, config={})
    )
    bench = Bench(context, decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    assert result["ok"] is False
    assert result["failure_code"] == "driver_contract_violation"
    assert result["side_effect_state"] == "unknown"
    assert result["effect_uncertain"] is True


@pytest.mark.asyncio
async def test_interlock_read_raising_the_wrong_exception_fails_closed(
    rogue_transport_kind: str,
) -> None:
    """An interlock that cannot be checked is an interlock that is not satisfied."""
    from dataclasses import replace

    context = replace(
        liquid_handler_context(), transport=TransportRef(kind=rogue_transport_kind, config={})
    )
    bench = Bench(context, decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "fluent_p1")
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is False
    # Hardline, so the human was never asked.
    assert bench.human.prompts == []


@pytest.mark.parametrize("bad_value", ["fast", True, float("nan"), float("inf"), None])
@pytest.mark.asyncio
async def test_non_numeric_value_on_a_numeric_channel_is_refused(bad_value: Any) -> None:
    """An unevaluable value must not pass the one check standing before the device."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=bad_value
    )
    assert result["ok"] is False
    assert bench.human.prompts == []
    transport = await bench.transport("bench_node")
    assert transport.write_log == ()


@pytest.mark.asyncio
async def test_state_channel_still_accepts_a_boolean() -> None:
    """Tightening numeric channels must not break state channels."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "bench_node")
    result = await bench.tools.hw_configure(
        device_id="bench_node", channel_id="indicator", value=True
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_concurrent_first_use_opens_one_transport() -> None:
    """Two concurrent first calls must not each open a connection.

    The second instance would be connected but unreferenced -- a leaked port that
    nothing will ever close. ``open()`` being idempotent does not help, because the
    idempotence is per instance and this race produces two.
    """
    import asyncio

    bench = Bench(bench_node_context())
    transports = await asyncio.gather(
        *(bench.registry.transport("bench_node") for _ in range(8))
    )
    assert len({id(item) for item in transports}) == 1
    assert bench.registry.opened_devices() == ("bench_node",)


@pytest.mark.asyncio
async def test_estop_never_prompts() -> None:
    """Waiting for consent to stop a moving machine is physically absurd."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.DENY,))
    result = await bench.tools.hw_estop(device_id="bench_node")
    assert result["ok"] is True
    assert result["halted"] is True
    assert bench.human.prompts == []
    transport = await bench.transport("bench_node")
    assert transport.halt_calls == 1


@pytest.mark.asyncio
async def test_estop_reports_when_the_device_cannot_halt() -> None:
    bench = Bench(with_transport_config(bench_node_context(), halt_supported=False))
    result = await bench.tools.hw_estop(device_id="bench_node")
    assert result["ok"] is False
    assert result["halted"] is False


# ════════════════════════════════════════════════════════════════
# Describe before write
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_write_without_describe_is_refused_with_a_repair_instruction() -> None:
    bench = Bench(bench_node_context(), require_describe=True)
    result = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    assert result["ok"] is False
    assert result["failure_code"] == "describe_required"
    assert "hw_describe" in result["error"]
    assert bench.human.prompts == []


@pytest.mark.asyncio
async def test_write_after_describe_proceeds() -> None:
    bench = Bench(bench_node_context(), require_describe=True)
    await bench.tools.hw_describe(device_id="bench_node")
    result = await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=20.0)
    assert result["ok"] is True


# ════════════════════════════════════════════════════════════════
# Full happy path and audit
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_end_to_end_discover_describe_read_command() -> None:
    """The whole chain a model walks, on two devices at once."""
    bench = Bench(
        bench_node_context(),
        liquid_handler_context(),
        decisions=(ApprovalDecision.ALLOW_ONCE,),
        require_describe=True,
    )
    assert set(bench.report.admitted) == {"bench_node", "fluent_p1"}

    listing = await bench.tools.hw_list()
    assert listing["count"] == 2
    # The index stays cheap: no envelope data in it.
    assert all("envelope" not in str(entry) for entry in listing["devices"])

    described = await bench.tools.hw_describe(device_id="bench_node")
    assert "fan_duty" in described["writable_channels"]
    assert described["streaming_channels"] == ["cpu_temp"]

    reading = await bench.tools.hw_read(device_id="bench_node", channel_id="cpu_temp")
    assert reading["ok"] is True
    assert reading["reading"]["value"] == pytest.approx(41.2)
    assert reading["reading"]["unit"] == "degC"

    status = await bench.tools.hw_status(device_id="bench_node")
    assert status["status"]["connected"] is True

    commanded = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=25.0
    )
    assert commanded["ok"] is True
    assert commanded["readback"]["value"] == pytest.approx(25.0)
    # A setpoint with inertia says so instead of being reported as settled.
    assert commanded["settling_time_s"] == pytest.approx(2.0)
    assert "stabilise" in commanded["next_step"]

    transport = await bench.transport("bench_node")
    assert transport.write_log == (("fan_duty", 25.0),)


@pytest.mark.asyncio
async def test_approval_prompt_names_the_machine_and_its_location() -> None:
    """In the physical world, *which* machine is safety information."""
    bench = Bench(liquid_handler_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "fluent_p1")
    await bench.tools.hw_dispense(device_id="fluent_p1", channel_id="aspirate", value=10.0)
    assert bench.human.prompts
    request = bench.human.prompts[0]
    assert "bench-2" in request.display["summary"]
    assert "fluent_p1.aspirate" in request.detail
    assert "10.0 uL_per_s" in request.detail


@pytest.mark.asyncio
async def test_every_decision_is_audited() -> None:
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "bench_node")
    await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=30.0)
    await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=900.0)
    kinds = [entry["action_kind"] for entry in bench.audit_entries]
    decisions = [entry["decision"] for entry in bench.audit_entries]
    assert kinds == [ActionKind.DEVICE_ACTUATE.value, ActionKind.DEVICE_ACTUATE.value]
    assert decisions[0].startswith("allow")
    assert decisions[1].startswith("deny")


@pytest.mark.asyncio
async def test_audit_detail_carries_no_raw_transport_payload() -> None:
    """Audit text is human-facing and persisted; it must stay a description."""
    bench = Bench(liquid_handler_context(), decisions=(ApprovalDecision.ALLOW_ONCE,))
    await _describe(bench, "fluent_p1")
    await bench.tools.hw_dispense(device_id="fluent_p1", channel_id="aspirate", value=10.0)
    for entry in bench.audit_entries:
        assert "config" not in entry["detail"]
        assert entry["resource"].startswith("fluent_p1:aspirate@")


# ════════════════════════════════════════════════════════════════
# Composite classifier neutrality
# ════════════════════════════════════════════════════════════════


def test_composite_classifier_leaves_existing_kinds_untouched() -> None:
    """Adding a domain must not change how a shell command is assessed."""
    registry = HardwareRegistry(HardwareSettings(enabled=True))
    composed = build_risk_classifier(registry)
    default = DefaultRiskClassifier()
    for descriptor in (
        ActionDescriptor.shell("rm -rf / "),
        ActionDescriptor.shell("ls -la"),
        ActionDescriptor.file_read("/etc/hosts"),
    ):
        assert composed.assess(descriptor).to_dict() == default.assess(descriptor).to_dict()


def test_build_risk_classifier_without_registry_is_the_default() -> None:
    """With hardware absent, behaviour must be byte-identical to before."""
    assert isinstance(build_risk_classifier(None), DefaultRiskClassifier)


def test_unresolvable_device_command_is_a_hardline() -> None:
    """A command we cannot describe is one we must not let a human wave through."""
    registry = HardwareRegistry(HardwareSettings(enabled=True))
    classifier = build_risk_classifier(registry)
    descriptor = ActionDescriptor.device(
        kind=ActionKind.DEVICE_ACTUATE.value,
        device_id="ghost_device",
        channel_id="ghost_channel",
        value=1.0,
    )
    assessment = classifier.assess(descriptor)
    assert assessment.hardline is True
    assert assessment.allow_permanent is False


def test_device_read_is_assessed_as_safe() -> None:
    registry = HardwareRegistry(HardwareSettings(enabled=True))
    classifier = build_risk_classifier(registry)
    descriptor = ActionDescriptor.device(
        kind=ActionKind.DEVICE_READ.value, device_id="any", channel_id="any"
    )
    assert classifier.assess(descriptor).hardline is False


def test_estop_is_never_assessed_as_gated() -> None:
    registry = HardwareRegistry(HardwareSettings(enabled=True))
    classifier = build_risk_classifier(registry)
    descriptor = ActionDescriptor.device(
        kind=ActionKind.DEVICE_ESTOP.value, device_id="any", channel_id="any"
    )
    assessment = classifier.assess(descriptor)
    assert assessment.hardline is False
    assert assessment.level.value == "safe"


# ════════════════════════════════════════════════════════════════
# Reusable consent posture for irreversible / external-output writes (issue #34)
# ════════════════════════════════════════════════════════════════


def _single_channel_context(
    *, effect: str, reversible: bool, direction: str = Direction.READWRITE.value
) -> HardwareContext:
    """A one-writable-channel device for isolating a risk-tier decision."""
    return HardwareContext(
        device_id="rig",
        hc_version=HC_VERSION,
        display_name="Rig",
        location="bench",
        halt_supported=True,
        transport=TransportRef(kind="mock", config={"values": {"chan": 0.0}}),
        channels=(
            Channel(
                channel_id="chan",
                direction=direction,
                quantity="ratio.chan",
                unit="percent",
                effect=effect,
                envelope=Envelope(
                    declared=True, min_value=0.0, max_value=100.0, reversible=reversible
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="lab-lead"),
    )


def _loaded_registry(context: HardwareContext) -> HardwareRegistry:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, require_describe_before_write=False),
        providers=[_StaticProvider(context)],
    )
    registry.load()
    return registry


def _in_envelope_descriptor(kind: str, value: float) -> ActionDescriptor:
    """Build a descriptor that clears every feasibility gate so ``_tier_for`` runs."""
    return ActionDescriptor.device(
        kind=kind,
        device_id="rig",
        channel_id="chan",
        value=value,
        metadata={"value_in_envelope": True, "interlocks_satisfied": True},
    )


def test_irreversible_actuate_forbids_reusable_consent() -> None:
    """An irreversible ACTUATE is HIGH and must never buy a reusable grant.

    Reusable session/profile consent for an effect that cannot be undone would let
    a session-wide bypass, earned from a lower-risk approval, silently authorise it.
    """
    registry = _loaded_registry(
        _single_channel_context(effect=HardwareEffect.ACTUATE.value, reversible=False)
    )
    classifier = build_risk_classifier(registry)
    assessment = classifier.assess(
        _in_envelope_descriptor(ActionKind.DEVICE_ACTUATE.value, 40.0)
    )
    assert assessment.level == RiskLevel.HIGH
    assert assessment.allow_permanent is False
    assert "irreversible" in assessment.reasons


def test_dispense_forbids_reusable_consent_even_when_declared_reversible() -> None:
    """DISPENSE outputs material into the world, so it is treated as irreversible.

    Even a declaration that marks the channel reversible cannot un-dispense a
    substance, so reusable consent is withheld regardless of ``envelope.reversible``.
    """
    registry = _loaded_registry(
        _single_channel_context(
            effect=HardwareEffect.DISPENSE.value,
            reversible=True,
            direction=Direction.WRITE.value,
        )
    )
    classifier = build_risk_classifier(registry)
    assessment = classifier.assess(
        _in_envelope_descriptor(ActionKind.DEVICE_DISPENSE.value, 40.0)
    )
    assert assessment.level == RiskLevel.HIGH
    assert assessment.allow_permanent is False


def test_reversible_actuate_keeps_reusable_consent() -> None:
    """Regression guard: a reversible setpoint write keeps band-scoped reuse.

    Tightening the irreversible case must not make routine reversible motion prompt
    on every command -- that is what disables the gate in practice.
    """
    registry = _loaded_registry(
        _single_channel_context(effect=HardwareEffect.ACTUATE.value, reversible=True)
    )
    classifier = build_risk_classifier(registry)
    assessment = classifier.assess(
        _in_envelope_descriptor(ActionKind.DEVICE_ACTUATE.value, 40.0)
    )
    assert assessment.level == RiskLevel.HIGH
    assert assessment.allow_permanent is True
    assert "irreversible" not in assessment.reasons


def test_reversible_configure_setpoint_keeps_reusable_consent() -> None:
    """A reversible CONFIGURE setpoint is MEDIUM and unaffected by the tightening."""
    registry = _loaded_registry(
        _single_channel_context(effect=HardwareEffect.CONFIGURE.value, reversible=True)
    )
    classifier = build_risk_classifier(registry)
    assessment = classifier.assess(
        _in_envelope_descriptor(ActionKind.DEVICE_CONFIGURE.value, 40.0)
    )
    assert assessment.level == RiskLevel.MEDIUM
    assert assessment.allow_permanent is True


# ════════════════════════════════════════════════════════════════
# Plugin surface
# ════════════════════════════════════════════════════════════════


def test_plugin_exposes_no_tools_until_a_registry_is_bound() -> None:
    """Default-off must be inert, so the tool index is unchanged."""
    from leapflow.hardware.plugin import HardwareContextPlugin

    plugin = HardwareContextPlugin()
    assert plugin.tools == []
    assert plugin.plugin_id == "hardware_context"


def test_plugin_exposes_exactly_eight_tools_when_bound() -> None:
    """Tool count is fixed regardless of how many devices exist."""
    from leapflow.hardware.plugin import HardwareContextPlugin

    registry = HardwareRegistry(
        HardwareSettings(enabled=True),
        providers=[_StaticProvider(bench_node_context(), liquid_handler_context())],
    )
    registry.load()
    plugin = HardwareContextPlugin()
    plugin.bind_runtime(hardware_registry=registry, hardware_approval_gate=None)
    names = sorted(tool.name for tool in plugin.tools)
    assert names == [
        "hw_actuate",
        "hw_configure",
        "hw_describe",
        "hw_dispense",
        "hw_estop",
        "hw_list",
        "hw_read",
        "hw_status",
    ]


def test_plugin_registers_teardown_as_an_async_effect() -> None:
    """``close_all`` is a coroutine; the sync variant would drop it unawaited."""
    from leapflow.hardware.plugin import HardwareContextPlugin

    recorded: dict[str, Any] = {}

    class _Scope:
        def effect(self, cleanup: Any) -> None:
            recorded["sync"] = cleanup

        def async_effect(self, cleanup: Any) -> None:
            recorded["async"] = cleanup

    registry = HardwareRegistry(HardwareSettings(enabled=True))
    plugin = HardwareContextPlugin()
    plugin.bind_runtime(hardware_registry=registry, effect_scope=_Scope())
    assert "async" in recorded
    assert "sync" not in recorded


# ═════════════════════════════════════════════════════════════
# Configuration -- default-off equivalence and the enabled path
# ═════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_disabling_envelope_grant_asks_every_time() -> None:
    """The knob must actually withdraw reusable consent, not merely exist.

    With ``envelope_grant`` off, the orchestrator must never offer a session scope, so
    no grant is stored and each command is decided on its own.
    """
    bench = Bench(
        bench_node_context(),
        decisions=(ApprovalDecision.ALLOW_SESSION,),
        envelope_grant=False,
    )
    await _describe(bench, "bench_node")
    for value in (10.0, 40.0, 70.0):
        result = await bench.tools.hw_actuate(
            device_id="bench_node", channel_id="aux_level", value=value
        )
        assert result["ok"] is True, result
    # Asked once per distinct command, because grant identity now includes the value.
    assert len(bench.human.prompts) == 3
    # And the profile-wide "always" choice is withheld as well.
    assert all("allow_always" not in request.choices for request in bench.human.prompts)


@pytest.mark.asyncio
async def test_envelope_grant_enabled_is_the_default() -> None:
    """The permissive default is deliberate: prompting per motion gets gates disabled."""
    bench = Bench(bench_node_context(), decisions=(ApprovalDecision.ALLOW_SESSION,))
    await _describe(bench, "bench_node")
    for value in (10.0, 40.0, 70.0):
        assert (
            await bench.tools.hw_actuate(
                device_id="bench_node", channel_id="aux_level", value=value
            )
        )["ok"] is True
    assert len(bench.human.prompts) == 1


def test_hardware_config_keys_are_discoverable() -> None:
    """Every durable hardware setting must be reachable through leap config.

    A knob that only exists in YAML is not a supported configuration surface, so the
    catalog membership is asserted rather than assumed.
    """
    from leapflow.config import get_settings
    from leapflow.config_service import ConfigService

    service = ConfigService(get_settings())
    keys = {key for key in service.writable_keys() if key.startswith("hardware.")}
    assert keys == {
        "hardware.enabled",
        "hardware.devices_dir",
        "hardware.max_devices",
        "hardware.unverified_policy",
        "hardware.require_describe",
        "hardware.envelope_grant",
        "hardware.trust_skip_enabled",
        "hardware.stream_enabled",
        "hardware.stream_ring_capacity",
        "hardware.persist_readings",
        "hardware.downsample_interval_s",
        "hardware.raw_retention_days",
        "hardware.history_retention_days",
        "hardware.raw_segment_mb",
        "hardware.reading_store_sensitive",
        "hardware.providers",
        "hardware.host_interval_s",
        "hardware.host_include",
        "hardware.host_exclude",
        "hardware.rediscover_interval_s",
        "hardware.media_screens",
        "hardware.media_microphones",
        "hardware.preview_max_fps",
        "hardware.preview_max_width",
        "hardware.preview_quality",
        "hardware.preview_idle_timeout_s",
    }
    for key in keys:
        view = service.describe(key)
        assert view.description, f"{key} has no description in the catalog"
        # Enabling hardware composes the approval classifier at construction, so it
        # cannot take effect on a running daemon.
        assert view.hot_reload == "restart-required"


def test_explicitly_disabled_hardware_leaves_the_classifier_untouched() -> None:
    """An explicit hard disable remains inert, even though passive discovery defaults on."""
    from leapflow.hardware.registry import build_registry

    class _Settings:
        hardware_enabled = False
        profile_layout = None

    registry = build_registry(_Settings())
    assert registry is None
    assert isinstance(build_risk_classifier(registry), DefaultRiskClassifier)


@pytest.mark.asyncio
async def test_enabled_profile_drives_the_whole_chain_from_declarations(tmp_path) -> None:
    """End to end from a declaration file to a gated physical command.

    Exercises the seam a real deployment uses -- settings, layout-derived declaration
    directory, YAML provider, admission, tool surface, production approval chain -- so
    that none of it is verified only through hand-built objects.
    """
    import yaml

    from leapflow.hardware.registry import build_registry
    from leapflow.hardware.tools import HardwareTools

    devices = tmp_path / "devices"
    devices.mkdir()
    (devices / "rig.yaml").write_text(
        yaml.safe_dump(
            {
                "hc_version": HC_VERSION,
                "device_id": "rig",
                "display_name": "Bench rig",
                "location": "bench-7",
                "halt_supported": True,
                "transport": {"kind": "mock", "config": {"values": {"level": 5.0}}},
                "notes": "Do not exceed 50 percent while the cover is open.",
                "channels": [
                    {
                        "channel_id": "level",
                        "direction": "readwrite",
                        "quantity": "ratio.level",
                        "unit": "percent",
                        "effect": "actuate",
                        "envelope": {
                            "declared": True,
                            "min_value": 0.0,
                            "max_value": 50.0,
                            "reversible": True,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "verified.json").write_text(yaml.safe_dump({"rig": "jason"}), encoding="utf-8")

    class _Settings:
        hardware_enabled = True
        # Pinned to the declaration provider: this test is about a hand-written
        # declaration driving the whole chain, and the default set also enumerates
        # this host, whose channels have nothing to do with what is asserted below.
        hardware_providers = "yaml"
        hardware_devices_dir = str(devices)
        hardware_max_devices = 16
        hardware_unverified_policy = "deny_write"
        hardware_require_describe = True
        hardware_stream_enabled = True
        hardware_stream_ring_capacity = 64
        profile_layout = None

    registry = build_registry(_Settings())
    assert registry is not None
    assert registry.report.admitted == ("rig",)

    human = ScriptedHuman(ApprovalDecision.ALLOW_ONCE)
    orchestrator = ApprovalOrchestrator(
        SessionAwareGate(human),
        risk_classifier=build_risk_classifier(registry),
        policy=ApprovalPolicyEngine(),
    )
    tools = HardwareTools(registry, gate=orchestrator, session_id=SESSION)

    # The declaration was confirmed out of band, so the channel is commandable.
    described = await tools.hw_describe(device_id="rig")
    assert described["writable_channels"] == ["level"]
    assert "VERIFIED by jason" in described["reference"]
    assert "cover is open" in described["reference"]

    ok = await tools.hw_actuate(device_id="rig", channel_id="level", value=30.0)
    assert ok["ok"] is True
    assert "bench-7" in human.prompts[0].display["summary"]

    # And the declared ceiling still holds, without asking anyone.
    too_high = await tools.hw_actuate(device_id="rig", channel_id="level", value=80.0)
    assert too_high["ok"] is False
    assert len(human.prompts) == 1

    await registry.close_all()


# ════════════════════════════════════════════════════════════════
# Reachability precedes consent
# ════════════════════════════════════════════════════════════════


def _unreachable_context(**config: Any) -> HardwareContext:
    """A bench node behind a transport that cannot be reached.

    ``kind: mcp`` with no client installed is the shortest honest way to build one:
    the transport refuses to open, exactly as a dead serial port or a stopped server
    would, and the refusal comes from production code rather than a test double.
    """
    base = bench_node_context()
    return replace(base, transport=TransportRef(kind="mcp", config=config or {
        "read_tool": "r", "write_tool": "w", "halt_tool": "h",
        "channel_arg": "c", "value_arg": "v",
    }))


@pytest.mark.asyncio
async def test_an_unreachable_device_never_reaches_the_human() -> None:
    """An action that cannot succeed must not be put in front of a person.

    Before this, the order was resolve -> validate -> approve -> *open the transport*,
    so a device that was never reachable produced a prompt, a consent, and only then
    the failure. Asking somebody to authorise a command that cannot be delivered is
    how people learn to click through prompts, and the prompt they learn to dismiss is
    the same one that guards a command which *can* be delivered.
    """
    bench = Bench(_unreachable_context())
    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=30.0
    )
    assert result["ok"] is False
    assert bench.human.prompts == [], "nobody may be asked about an undeliverable command"
    assert result["failure_code"] == "mcp_client_unavailable"
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    assert "hw_status" in result["error"], "a refusal must name the next step"


@pytest.mark.asyncio
async def test_a_reachable_device_is_still_put_to_the_human() -> None:
    """The check must gate on reachability alone, not quietly swallow the prompt."""
    bench = Bench(bench_node_context())
    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=30.0
    )
    assert result["ok"] is True
    assert len(bench.human.prompts) == 1


@pytest.mark.asyncio
async def test_a_device_that_dies_after_opening_is_caught_before_consent() -> None:
    """A cached transport is why an open alone is not enough.

    ``transport()`` caches, so a connection that was live and then died is handed back
    without ``open()`` ever running again -- the common failure for a server-backed
    device whose server restarted. Only a probe sees it.
    """

    class _Dying:
        """Answers the first probe (during open) and fails every one after it."""

        def __init__(self) -> None:
            self.probes = 0

        async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            if tool_name != "status":
                return {"ok": True, "value": 0.0}
            self.probes += 1
            if self.probes == 1:
                return {"ok": True}
            return {"ok": False, "error": "session closed"}

    dying = _Dying()
    undo = set_mcp_client_provider(lambda: dying)
    try:
        bench = Bench(
            _unreachable_context(
                read_tool="r", write_tool="w", probe_tool="status", halt_tool="h",
                channel_arg="c", value_arg="v",
            )
        )
        result = await bench.tools.hw_actuate(
            device_id="bench_node", channel_id="fan_duty", value=30.0
        )
    finally:
        undo()
    assert result["ok"] is False
    assert bench.human.prompts == []
    assert result["failure_code"] == "device_unreachable"
    assert result["side_effect_state"] == SIDE_EFFECT_NONE


@pytest.mark.asyncio
async def test_a_dead_transport_is_dropped_so_the_next_attempt_reconnects() -> None:
    """Otherwise one transient outage disables the device for the process's life.

    The refusal must not outlive the condition that caused it: a cached dead session
    answers every probe with "not connected", and nothing else would ever call
    ``open()`` again.
    """

    class _Recovering:
        """Fails the probe once after opening, then behaves."""

        def __init__(self) -> None:
            self.probes = 0
            self.opens = 0

        async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            if tool_name != "status":
                return {"ok": True, "value": 0.0}
            self.probes += 1
            # Probe 1 runs inside the first open, probe 2 is the pre-consent check
            # that must fail, probe 3 runs inside the reopen after the drop.
            if self.probes == 2:
                return {"ok": False, "error": "session closed"}
            return {"ok": True}

    client = _Recovering()
    undo = set_mcp_client_provider(lambda: client)
    try:
        bench = Bench(
            _unreachable_context(
                read_tool="r", write_tool="w", probe_tool="status", halt_tool="h",
                channel_arg="c", value_arg="v",
            )
        )
        first = await bench.tools.hw_actuate(
            device_id="bench_node", channel_id="fan_duty", value=30.0
        )
        assert first["failure_code"] == "device_unreachable"
        assert bench.registry.opened_devices() == (), "the dead transport must be forgotten"

        second = await bench.tools.hw_actuate(
            device_id="bench_node", channel_id="fan_duty", value=30.0
        )
    finally:
        undo()
    assert second["ok"] is True, "the device must be usable again once the outage clears"
    assert len(bench.human.prompts) == 1, "only the deliverable command reached the human"


@pytest.mark.asyncio
async def test_emergency_stop_is_not_blocked_by_an_unreachable_probe() -> None:
    """Halt must still be attempted, because refusing to try to stop is worse.

    The pre-consent check exists to keep undeliverable commands away from a human;
    ``hw_estop`` asks no human, so gating it on reachability would only mean declining
    to attempt a stop on a device that might still be moving.
    """
    bench = Bench(_unreachable_context())
    result = await bench.tools.hw_estop(device_id="bench_node")
    assert result["ok"] is False
    assert result["failure_code"] == "mcp_client_unavailable", (
        "the failure must come from attempting the halt, not from a pre-check refusing to"
    )


@pytest.mark.asyncio
async def test_an_unreachable_refusal_reaches_the_signal_path() -> None:
    """A tool result is read once; the board and every watch need the event.

    Without this the condition that most needs to be visible is the one nothing can
    see: a bench refusing every command is indistinguishable from a bench nobody is
    using. The refusal lands in the same ring and on the same sink as a threshold
    breach, so no consumer needs to learn a second shape.
    """
    from leapflow.hardware.stream import EventKind

    published: list[Any] = []
    bench = Bench(_unreachable_context())
    bench.registry.set_event_emitter(published.append)

    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=30.0
    )
    assert result["ok"] is False

    recorded = bench.registry.recent_events(device_id="bench_node")
    assert [event.kind for event in recorded] == [EventKind.UNREACHABLE]
    assert recorded[0].channel_id == "fan_duty"
    assert recorded[0].quantity == "ratio.fan_duty", "the channel's declared quantity"
    assert recorded[0].observed_at > 1.7e9, "wall clock, since this crosses modules"

    assert [event.kind for event in published] == [EventKind.UNREACHABLE], (
        "recording it without emitting leaves the board blind, which is the whole point"
    )
    assert published[0].event_type == "hw.unreachable", "the family every consumer groups on"


@pytest.mark.asyncio
async def test_a_refusal_still_happens_when_nothing_is_listening() -> None:
    """Reporting is telemetry; it must never be what decides a command's fate."""
    bench = Bench(_unreachable_context())  # no emitter installed

    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=30.0
    )
    assert result["failure_code"] == "mcp_client_unavailable"
    assert len(bench.registry.recent_events(device_id="bench_node")) == 1


@pytest.mark.asyncio
async def test_a_raising_event_sink_does_not_fail_the_command() -> None:
    """The refusal is the answer; a broken sink must not replace it with a crash."""

    def _explode(event: Any) -> None:
        raise RuntimeError("bus is down")

    bench = Bench(_unreachable_context())
    bench.registry.set_event_emitter(_explode)

    result = await bench.tools.hw_actuate(
        device_id="bench_node", channel_id="fan_duty", value=30.0
    )
    assert result["failure_code"] == "mcp_client_unavailable"
    assert result["side_effect_state"] == SIDE_EFFECT_NONE


@pytest.mark.asyncio
async def test_the_board_shows_an_unreachable_device_as_an_alert() -> None:
    """End to end: the refusal must survive into the panel an operator reads."""
    from leapflow.hardware.observability.digest import build_digest

    bench = Bench(_unreachable_context())
    await bench.tools.hw_actuate(device_id="bench_node", channel_id="fan_duty", value=30.0)

    digest = build_digest(bench.registry)
    kinds = [str(event["kind"]) for event in digest.events]
    assert "unreachable" in kinds, f"the board timeline must carry it, got {kinds}"
    row = next(event for event in digest.events if event["kind"] == "unreachable")
    assert row["severity"] == "alert", "a bench that cannot be commanded is not routine"
    assert row["title"].startswith("unreachable · bench_node.fan_duty")


# ════════════════════════════════════════════════════════════════
# Daemon-side gate re-binding (fix for #24)
# ════════════════════════════════════════════════════════════════


class TestDaemonHardwareGateRebinding:
    """Prove that ``install_gate`` re-binds the hardware approval gate.

    Before the fix, the hardware plugin captured the pre-daemon orchestrator
    during ``initialize_critical`` and never updated it when the daemon
    installed its own stream-routed orchestrator.  Hardware writes therefore
    had no interactive approver and were refused fail-closed every time.

    These tests drive the *plugin* layer directly: they construct a
    ``HardwareContextPlugin`` with one orchestrator, call ``bind_runtime``
    with a replacement (simulating what ``install_gate`` does via
    ``ToolPluginRegistry.bind_runtime``), and verify that the live tool
    handlers resolve to the *new* gate.
    """

    @pytest.mark.asyncio
    async def test_gate_rebind_routes_through_new_orchestrator(self) -> None:
        """After re-bind, hw_actuate goes through the replacement gate."""
        from leapflow.hardware.plugin import HardwareContextPlugin

        ctx = bench_node_context()
        registry = HardwareRegistry(
            HardwareSettings(enabled=True, require_describe_before_write=False),
            providers=[_StaticProvider(ctx)],
        )
        registry.load()

        # Phase 1: initial bind with an always-denying gate (pre-daemon path).
        denying_human = ScriptedHuman(ApprovalDecision.DENY)
        old_gate = SessionAwareGate(denying_human)
        old_orchestrator = ApprovalOrchestrator(
            old_gate,
            risk_classifier=build_risk_classifier(registry),
            policy=ApprovalPolicyEngine(),
        )

        plugin = HardwareContextPlugin()
        plugin.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=old_orchestrator,
        )

        # Access tools once to trigger lazy creation -- simulates assembly.
        tools_before = plugin.tools
        assert tools_before, "plugin must expose tools after binding a registry"

        # Capture the handler for hw_actuate from the pre-rebind tools.
        actuate_meta = next(t for t in tools_before if t.name == "hw_actuate")
        handler = actuate_meta.handler

        # Confirm old gate denies.
        result = await handler(device_id="bench_node", channel_id="fan_duty", value=20.0)
        assert result["ok"] is False
        assert result["failure_code"] == "approval_denied"

        # Phase 2: re-bind with an allowing gate (daemon install_gate path).
        allowing_human = ScriptedHuman(ApprovalDecision.ALLOW_ONCE)
        new_gate = SessionAwareGate(allowing_human)
        new_orchestrator = ApprovalOrchestrator(
            new_gate,
            risk_classifier=build_risk_classifier(registry),
            policy=ApprovalPolicyEngine(),
        )

        plugin.bind_runtime(hardware_approval_gate=new_orchestrator)

        # The SAME handler object (already registered in the tool registry)
        # must now route through the new orchestrator.
        actuate_meta_after = next(t for t in plugin.tools if t.name == "hw_actuate")
        assert actuate_meta_after.handler is handler, (
            "gate-only re-bind must not replace the handler object; "
            "per-turn snapshots depend on identity stability"
        )
        result = await handler(device_id="bench_node", channel_id="fan_duty", value=20.0)
        assert result["ok"] is True, (
            f"After re-bind the daemon orchestrator should approve, got: {result}"
        )
        assert allowing_human.prompts, "the new gate must have been consulted"

    @pytest.mark.asyncio
    async def test_gate_rebind_without_registry_is_harmless(self) -> None:
        """Re-binding only the gate when no registry exists must not crash."""
        from leapflow.hardware.plugin import HardwareContextPlugin

        plugin = HardwareContextPlugin()
        # No registry bound -- plugin.tools is empty.
        plugin.bind_runtime(hardware_approval_gate="some_orchestrator")
        assert plugin.tools == []

    @pytest.mark.asyncio
    async def test_absent_gate_after_rebind_still_denies(self) -> None:
        """Fail-closed: re-binding with None gate must deny."""
        from leapflow.hardware.plugin import HardwareContextPlugin

        ctx = bench_node_context()
        registry = HardwareRegistry(
            HardwareSettings(enabled=True, require_describe_before_write=False),
            providers=[_StaticProvider(ctx)],
        )
        registry.load()

        allowing_human = ScriptedHuman(ApprovalDecision.ALLOW_ONCE)
        gate = SessionAwareGate(allowing_human)
        orchestrator = ApprovalOrchestrator(
            gate,
            risk_classifier=build_risk_classifier(registry),
            policy=ApprovalPolicyEngine(),
        )

        plugin = HardwareContextPlugin()
        plugin.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=orchestrator,
        )
        _ = plugin.tools  # force creation

        # Re-bind with None gate (simulates a broken installation path).
        plugin.bind_runtime(hardware_approval_gate=None)

        actuate_meta = next(t for t in plugin.tools if t.name == "hw_actuate")
        result = await actuate_meta.handler(
            device_id="bench_node", channel_id="fan_duty", value=20.0
        )
        assert result["ok"] is False
        assert "configuration fault" in result["error"]

    @pytest.mark.asyncio
    async def test_teardown_not_double_registered_on_gate_rebind(self) -> None:
        """Re-binding only the gate must not re-register the teardown effect."""
        from leapflow.hardware.plugin import HardwareContextPlugin

        registrations: list[Any] = []

        class _TrackingScope:
            def async_effect(self, fn: Any) -> None:
                registrations.append(fn)

        ctx = bench_node_context()
        registry = HardwareRegistry(
            HardwareSettings(enabled=True, require_describe_before_write=False),
            providers=[_StaticProvider(ctx)],
        )
        registry.load()

        plugin = HardwareContextPlugin()
        plugin.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=None,
            effect_scope=_TrackingScope(),
        )
        assert len(registrations) == 1, "first bind registers teardown"

        # Re-bind only the gate.
        plugin.bind_runtime(hardware_approval_gate="new_gate")
        assert len(registrations) == 1, "gate-only re-bind must not double-register teardown"


# ════════════════════════════════════════════════════════════════
# install_gate integration (fix for #24, Minor 2)
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_install_gate_rebinds_hardware_approval_gate() -> None:
    """ApprovalCoordinator.install_gate wires the daemon orchestrator into
    the hardware plugin, not just shell/gateway/config/desktop.

    Constructs a minimal fake context carrying a ``_hardware_registry`` and
    invokes ``install_gate`` directly.  The assertion is structural: the
    plugin's live ``HardwareTools`` instance must reference the orchestrator
    that ``install_gate`` built, not the pre-daemon one.
    """
    from leapflow.daemon.approval_coordinator import ApprovalCoordinator
    from leapflow.hardware.plugin import HardwareContextPlugin
    from leapflow.plugins.registry import ToolPluginRegistry

    # 1. Build a real hardware registry with one device.
    hw_registry = HardwareRegistry(
        HardwareSettings(enabled=True, require_describe_before_write=False),
        providers=[_StaticProvider(bench_node_context())],
    )
    hw_registry.load()

    # 2. Build the hardware plugin and wire it into a fresh tool registry.
    hw_plugin = HardwareContextPlugin()
    tool_registry = ToolPluginRegistry()
    tool_registry.register(hw_plugin)

    # 3. Initial bind with a denying gate (simulates initialize_critical).
    denying_human = ScriptedHuman(ApprovalDecision.DENY)
    pre_orchestrator = ApprovalOrchestrator(
        SessionAwareGate(denying_human),
        risk_classifier=build_risk_classifier(hw_registry),
        policy=ApprovalPolicyEngine(),
    )
    tool_registry.bind_runtime(
        hardware_registry=hw_registry,
        hardware_approval_gate=pre_orchestrator,
    )
    tool_registry.assemble()

    # Confirm tools are live and the pre-daemon gate denies.
    assert hw_plugin.tools, "plugin must expose tools after assembly"

    # 4. Build a fake ctx that carries what install_gate needs.
    class _FakeCtx:
        pass

    fake_ctx = _FakeCtx()
    fake_ctx._approval_orchestrator = pre_orchestrator  # type: ignore[attr-defined]
    fake_ctx._hardware_registry = hw_registry  # type: ignore[attr-defined]
    fake_ctx.settings = type("S", (), {  # type: ignore[attr-defined]
        "approval_bypass": False,
        "plugin_generation_enabled": False,
        "plugin_install_dir": None,
        "profile_layout": None,
        "plugin_marketplace_root": None,
        "plugin_marketplace_url": None,
        "plugin_marketplace_trusted_pubkeys": (),
    })()
    fake_ctx.llm = None  # type: ignore[attr-defined]

    class _FakeService:
        pass

    # Monkey-patch get_registry so install_gate finds our test registry.
    import leapflow.plugins as _plugins_mod
    original_get_registry = _plugins_mod.get_registry
    _plugins_mod.get_registry = lambda: tool_registry
    try:
        coordinator = ApprovalCoordinator()
        coordinator.install_gate(fake_ctx, _FakeService())
    finally:
        _plugins_mod.get_registry = original_get_registry

    # 5. The daemon orchestrator is now on fake_ctx._approval_orchestrator.
    daemon_orchestrator = fake_ctx._approval_orchestrator
    assert daemon_orchestrator is not pre_orchestrator, (
        "install_gate must replace the orchestrator"
    )

    # The hardware plugin's live tools must reference the daemon orchestrator.
    assert hw_plugin._hw_tools is not None, "tools must have been created"
    assert hw_plugin._gate is daemon_orchestrator, (
        "plugin._gate must point to the daemon orchestrator after install_gate"
    )
    assert hw_plugin._hw_tools._gate is daemon_orchestrator, (
        "the live HardwareTools instance must reference the daemon orchestrator"
    )




def test_passive_hardware_discovery_defaults_on_without_opening_devices() -> None:
    """`/board hardware` must not require a preliminary config task.

    The distinction is security-critical: default-on means passive inventory only (host
    metrics and media enumeration); camera/microphone reads remain privacy-gated. A future
    "safe by default" edit must not conflate the two and reintroduce an empty board.
    """
    from leapflow.config import Settings
    from leapflow.hardware.registry import HardwareSettings

    assert Settings.__dataclass_fields__["hardware_enabled"].default is True
    policy = HardwareSettings()
    assert policy.enabled is True
    assert policy.preview_max_fps == 12.0
    assert policy.preview_max_width == 1280
    assert policy.preview_quality == 85
