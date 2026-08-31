"""Physical outcome learning: numeric prediction error and parameter reuse.

This is the payoff for connecting hardware to the world model, and the reason the physical
domain matters to a learning agent at all: in the UI domain "was the prediction right" is
genuinely ambiguous, so the world model has to ask a model to rate the distance. Physics
has no such ambiguity. The command was 37.0, the device settled at 36.8, the error is 0.2,
and no model call is needed to know it.

The scenario driving these cases is the one from the reported liquid-handler work: an
agent discovers that water tolerates a fast aspiration rate while a viscous protein sample
does not. That discovery was made once and then lost. Here it must survive the turn.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

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
from leapflow.hardware.outcome import (
    HardwareOutcomeRecorder,
    PhysicalOutcome,
    normalized_delta,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.tools import HardwareTools
from leapflow.security.approval import ApprovalDecision, SessionAwareGate
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.security.policy import ApprovalPolicyEngine

from tests.test_hardware_governance import ScriptedHuman, with_transport_config


# ════════════════════════════════════════════════════════════════
# In-memory experience store
# ════════════════════════════════════════════════════════════════


class FakeExperienceStore:
    """Records what the real ExperienceStore would persist.

    Kept to the store's actual interface -- ``store`` and ``retrieve_similar`` with the
    same signatures -- because the point is to verify the hardware side hands over
    well-formed experience, not to reimplement keyword search.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def store(
        self,
        action_description: str,
        app_context: str,
        predicted_effect: str,
        actual_effect: str,
        delta: float,
        pre_state_summary: str = "",
        post_state_summary: str = "",
        **_: Any,
    ) -> str:
        self.records.append(
            {
                "action_description": action_description,
                "app_context": app_context,
                "predicted_effect": predicted_effect,
                "actual_effect": actual_effect,
                "delta": delta,
                "pre_state_summary": pre_state_summary,
                "post_state_summary": post_state_summary,
            }
        )
        return f"exp-{len(self.records)}"

    def retrieve_similar(self, action_desc: str, app_context: str, *, limit: int = 5, **_: Any):
        """Naive substring match, standing in for keyword retrieval."""
        tokens = [t for t in action_desc.lower().split() if len(t) >= 3]
        hits = []
        for record in self.records:
            if record["app_context"] != app_context:
                continue
            haystack = record["action_description"].lower()
            if any(token in haystack for token in tokens):
                hits.append(_Experience(record))
        return hits[:limit]


class _Experience:
    def __init__(self, record: dict[str, Any]) -> None:
        self.action_description = record["action_description"]
        self.actual_effect = record["actual_effect"]
        self.delta = record["delta"]


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


def _channel(
    *,
    settling: float = 0.0,
    verify: bool = True,
    min_value: float | None = 0.0,
    max_value: float | None = 200.0,
) -> Channel:
    return Channel(
        channel_id="aspirate",
        direction=Direction.READWRITE.value,
        quantity="volume.aspirate",
        unit="uL_per_s",
        effect=HardwareEffect.DISPENSE.value,
        verify_after_write=verify,
        envelope=Envelope(
            declared=True,
            min_value=min_value,
            max_value=max_value,
            settling_time_s=settling,
            reversible=False,
        ),
    )


def _context(**channel_kwargs: Any) -> HardwareContext:
    return HardwareContext(
        device_id="fluent_p1",
        hc_version=HC_VERSION,
        display_name="Tecan Fluent",
        location="bench-2",
        halt_supported=True,
        transport=TransportRef(kind="mock", config={"values": {"aspirate": 0.0}}),
        channels=(_channel(**channel_kwargs),),
        provenance=ContextProvenance(verified_by="lab-lead"),
    )


class _StaticProvider:
    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


class Bench:
    """Registry plus the real approval chain plus a recording experience store."""

    def __init__(self, context: HardwareContext, **overrides: Any) -> None:
        self.registry = HardwareRegistry(
            HardwareSettings(
                enabled=True,
                require_describe_before_write=False,
                persist_readings=False,
                **overrides,
            ),
            providers=[_StaticProvider(context)],
        )
        self.registry.load()
        self.store = FakeExperienceStore()
        self.registry.bind_persistence(experience_store=self.store)
        self.human = ScriptedHuman(ApprovalDecision.ALLOW_ALL_SESSION)
        self.orchestrator = ApprovalOrchestrator(
            SessionAwareGate(self.human), policy=ApprovalPolicyEngine()
        )
        self.tools = HardwareTools(self.registry, gate=self.orchestrator, session_id="s")


# ════════════════════════════════════════════════════════════════
# Delta normalisation
# ════════════════════════════════════════════════════════════════


def test_delta_is_normalised_against_the_declared_span() -> None:
    """A raw residual is not comparable across channels; the envelope makes it so.

    ``ExperienceStore`` is shared across every domain and its consumers compare delta
    against fixed thresholds, so 0.2 degrees and 0.2 microlitres per second must not be the
    same number while meaning entirely different things.
    """
    envelope = Envelope(declared=True, min_value=0.0, max_value=200.0)
    delta, residual = normalized_delta(commanded=100.0, observed=110.0, envelope=envelope)
    assert residual == pytest.approx(10.0)
    assert delta == pytest.approx(0.05)


def test_same_residual_on_a_narrower_span_is_a_larger_error() -> None:
    """Ten units off is trivial on a 0..200 channel and severe on a 0..20 one."""
    wide = Envelope(declared=True, min_value=0.0, max_value=200.0)
    narrow = Envelope(declared=True, min_value=0.0, max_value=20.0)
    wide_delta, _ = normalized_delta(commanded=100.0, observed=110.0, envelope=wide)
    narrow_delta, _ = normalized_delta(commanded=10.0, observed=20.0, envelope=narrow)
    assert narrow_delta > wide_delta


def test_delta_is_bounded_at_one() -> None:
    """Downstream thresholds assume a 0..1 range; an unbounded delta would break them."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=10.0)
    delta, _ = normalized_delta(commanded=1.0, observed=9999.0, envelope=envelope)
    assert delta == 1.0


def test_delta_falls_back_to_relative_error_without_a_span() -> None:
    """A channel with no declared bounds still yields a comparable number."""
    envelope = Envelope(declared=True)
    delta, _ = normalized_delta(commanded=50.0, observed=55.0, envelope=envelope)
    assert delta == pytest.approx(0.1)


def test_delta_of_a_zero_command_uses_the_bare_residual() -> None:
    """Relative error is undefined at zero, so it must not divide by it."""
    envelope = Envelope(declared=True)
    delta, _ = normalized_delta(commanded=0.0, observed=0.4, envelope=envelope)
    assert delta == pytest.approx(0.4)


def test_direction_does_not_change_the_delta() -> None:
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0)
    over, _ = normalized_delta(commanded=50.0, observed=60.0, envelope=envelope)
    under, _ = normalized_delta(commanded=50.0, observed=40.0, envelope=envelope)
    assert over == under


def test_residual_keeps_its_sign() -> None:
    """The normalised delta is a magnitude; the residual says which way it went."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0)
    _, residual = normalized_delta(commanded=50.0, observed=40.0, envelope=envelope)
    assert residual == pytest.approx(-10.0)


# ════════════════════════════════════════════════════════════════
# Recorder mechanics
# ════════════════════════════════════════════════════════════════


def test_recorder_is_inert_without_a_store() -> None:
    """No store means nowhere for a delta to go, so nothing is tracked."""
    recorder = HardwareOutcomeRecorder(None)
    assert recorder.enabled is False
    recorder.record_command(device_id="d", channel=_channel(), value=10.0)
    assert recorder.pending == 0
    assert recorder.observe(device_id="d", channel_id="aspirate", value=10.0) is None
    assert recorder.recall(device_id="d", channel=_channel()) == ()


def test_command_and_observation_produce_one_experience() -> None:
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    recorder.record_command(
        device_id="fluent_p1", channel=_channel(), value=10.0, conditions="BSA protein"
    )
    outcome = recorder.observe(device_id="fluent_p1", channel_id="aspirate", value=10.4)

    assert outcome is not None
    assert outcome.commanded == 10.0
    assert outcome.observed == 10.4
    assert outcome.residual == pytest.approx(0.4)
    assert outcome.delta == pytest.approx(0.002)
    assert len(store.records) == 1
    assert recorder.pending == 0


def test_conditions_reach_the_retrieval_key() -> None:
    """Conditions lead the key because that is what a later question matches on.

    "What rate worked for a viscous protein sample" is a search for the situation, not for
    a channel name.
    """
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    recorder.record_command(
        device_id="fluent_p1",
        channel=_channel(),
        value=10.0,
        conditions="viscous BSA protein, foams easily",
    )
    recorder.observe(device_id="fluent_p1", channel_id="aspirate", value=10.0)
    assert "BSA" in store.records[0]["action_description"]
    assert "volume.aspirate" in store.records[0]["action_description"]


def test_non_numeric_commands_are_not_tracked() -> None:
    """A boolean state has no residual; inventing one would poison a shared store."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    recorder.record_command(device_id="d", channel=_channel(), value=True)
    assert recorder.pending == 0
    recorder.record_command(device_id="d", channel=_channel(), value="fast")
    assert recorder.pending == 0


def test_observation_before_settling_is_not_scored() -> None:
    """A reading taken mid-transition measures the transition, not the outcome.

    Recording it as the error would teach the store something false about the device --
    which is worse than learning nothing.
    """
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    recorder.record_command(
        device_id="d", channel=_channel(settling=2.0), value=50.0, now=100.0
    )
    assert recorder.observe(device_id="d", channel_id="aspirate", value=20.0, now=101.0) is None
    assert store.records == []
    # Still pending, waiting for a reading it can trust.
    assert recorder.pending == 1

    outcome = recorder.observe(device_id="d", channel_id="aspirate", value=49.5, now=103.0)
    assert outcome is not None
    assert outcome.observed == 49.5


def test_pending_command_expires() -> None:
    """An observation arriving much later says more about the room than the command."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store, pending_ttl_s=10.0)
    recorder.record_command(device_id="d", channel=_channel(), value=50.0, now=100.0)
    assert recorder.observe(device_id="d", channel_id="aspirate", value=50.0, now=200.0) is None
    assert store.records == []
    assert recorder.pending == 0


def test_observation_without_a_command_is_ignored() -> None:
    """A stream reading on an uncommanded channel is not an outcome of anything."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    assert recorder.observe(device_id="d", channel_id="aspirate", value=10.0) is None
    assert store.records == []


def test_dropped_pending_command_is_never_scored() -> None:
    """A failed write must not be scored by whatever the device happens to read next."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    recorder.record_command(device_id="d", channel=_channel(), value=50.0)
    recorder.drop_pending("d", "aspirate")
    assert recorder.observe(device_id="d", channel_id="aspirate", value=12.0) is None
    assert store.records == []


def test_a_failing_store_does_not_raise() -> None:
    """Learning is valuable; failing the operation that produced the observation is not."""

    class _Broken:
        def store(self, **_: Any) -> str:
            raise RuntimeError("memory backend down")

        def retrieve_similar(self, *_: Any, **__: Any):
            raise RuntimeError("memory backend down")

    recorder = HardwareOutcomeRecorder(_Broken())
    recorder.record_command(device_id="d", channel=_channel(), value=10.0)
    outcome = recorder.observe(device_id="d", channel_id="aspirate", value=10.0)
    assert outcome is not None
    assert recorder.recorded == 0
    assert recorder.recall(device_id="d", channel=_channel()) == ()


def test_recall_orders_by_how_well_the_device_tracked() -> None:
    """Within equally relevant experiences, the one that actually worked leads."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    for value, observed in ((140.0, 40.0), (10.0, 10.2), (80.0, 60.0)):
        recorder.record_command(
            device_id="fluent_p1", channel=_channel(), value=value, conditions="BSA protein"
        )
        recorder.observe(device_id="fluent_p1", channel_id="aspirate", value=observed)

    rows = recorder.recall(
        device_id="fluent_p1", channel=_channel(), conditions="BSA protein", limit=3
    )
    assert len(rows) == 3
    assert rows[0]["delta"] <= rows[1]["delta"] <= rows[2]["delta"]
    # The 10 uL/s command tracked almost exactly, so it leads.
    assert "10" in rows[0]["command"]


def test_relevance_outranks_accuracy() -> None:
    """A perfectly-tracking irrelevant experience must not displace a relevant one.

    Keyword retrieval matches on the channel and unit tokens every record for this channel
    shares, so ordering by accuracy alone would answer a question about protein with a
    result about water.
    """
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)

    # Water tracked perfectly at a fast rate.
    recorder.record_command(
        device_id="d", channel=_channel(), value=140.0, conditions="aqueous water"
    )
    recorder.observe(device_id="d", channel_id="aspirate", value=140.0)
    # Protein tracked less well, but it is what the question is about.
    recorder.record_command(
        device_id="d", channel=_channel(), value=10.0, conditions="viscous BSA protein"
    )
    recorder.observe(device_id="d", channel_id="aspirate", value=12.0)

    rows = recorder.recall(
        device_id="d", channel=_channel(), conditions="viscous BSA protein", limit=2
    )
    assert "BSA" in rows[0]["command"]
    assert rows[0]["delta"] > rows[1]["delta"], "the less accurate but relevant entry leads"


def test_without_conditions_ranking_falls_back_to_accuracy() -> None:
    """With nothing to be relevant to, the best-tracking entry is the useful one."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    for value, observed in ((140.0, 40.0), (10.0, 10.1)):
        recorder.record_command(device_id="d", channel=_channel(), value=value)
        recorder.observe(device_id="d", channel_id="aspirate", value=observed)
    rows = recorder.recall(device_id="d", channel=_channel(), limit=2)
    assert rows[0]["delta"] <= rows[1]["delta"]


def test_recall_is_scoped_to_the_device() -> None:
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(store)
    recorder.record_command(device_id="dev_a", channel=_channel(), value=10.0)
    recorder.observe(device_id="dev_a", channel_id="aspirate", value=10.0)
    assert recorder.recall(device_id="dev_b", channel=_channel()) == ()


def test_outcome_reports_whether_the_device_tracked() -> None:
    outcome = PhysicalOutcome(
        device_id="d",
        channel_id="c",
        quantity="q",
        unit="u",
        commanded=10.0,
        observed=10.01,
        delta=0.001,
        residual=0.01,
    )
    assert outcome.accurate is True
    assert PhysicalOutcome(
        device_id="d",
        channel_id="c",
        quantity="q",
        unit="u",
        commanded=10.0,
        observed=50.0,
        delta=0.4,
        residual=40.0,
    ).accurate is False


# ════════════════════════════════════════════════════════════════
# Tool integration
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_a_write_with_readback_is_learned_immediately() -> None:
    """A channel that reads back and does not settle can be scored on the spot."""
    bench = Bench(_context(settling=0.0, verify=True))
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1",
        channel_id="aspirate",
        value=10.0,
        conditions="viscous BSA protein",
    )
    assert result["ok"] is True
    assert len(bench.store.records) == 1
    record = bench.store.records[0]
    assert record["app_context"] == "fluent_p1"
    assert "BSA" in record["action_description"]
    # The mock transport stores exactly what was written, so the device tracked perfectly.
    assert record["delta"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_settling_channel_is_learned_on_the_next_read() -> None:
    """Inertia defers the comparison; it must not lose it.

    This is why the observation side is separate from the write: a setpoint with settling
    time cannot be judged by the readback taken immediately after commanding it.
    """
    bench = Bench(_context(settling=0.01, verify=True))
    await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=25.0, conditions="water"
    )
    # Nothing scored yet: the value was not stable when the readback happened.
    assert bench.store.records == []

    time.sleep(0.02)
    read = await bench.tools.hw_read(device_id="fluent_p1", channel_id="aspirate")
    assert read["ok"] is True
    assert read["command_outcome"]["commanded"] == 25.0
    assert len(bench.store.records) == 1


@pytest.mark.asyncio
async def test_a_failed_write_is_not_learned_from() -> None:
    """Whatever the device settles at after a failure is not a measurement of the command."""
    bench = Bench(
        with_transport_config(
            _context(settling=0.0, verify=True),
            failures=[
                {"channel_id": "aspirate", "on_call": 1, "side_effect_state": "partial"}
            ],
        )
    )
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0, conditions="BSA"
    )
    assert result["ok"] is False
    assert bench.store.records == []
    # And a later read must not retroactively score the failed command.
    await bench.tools.hw_read(device_id="fluent_p1", channel_id="aspirate")
    assert bench.store.records == []


@pytest.mark.asyncio
async def test_a_denied_write_is_not_learned_from() -> None:
    """A refused command never reached the device, so it has no outcome."""
    bench = Bench(_context())
    bench.human = ScriptedHuman(ApprovalDecision.DENY)
    bench.orchestrator = ApprovalOrchestrator(
        SessionAwareGate(bench.human), policy=ApprovalPolicyEngine()
    )
    bench.tools = HardwareTools(
        bench.registry, gate=bench.orchestrator, session_id="s"
    )
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is False
    assert bench.store.records == []


@pytest.mark.asyncio
async def test_describe_surfaces_prior_experience() -> None:
    """The reference document becomes accumulated experience, not just declared limits."""
    bench = Bench(_context())
    await bench.tools.hw_dispense(
        device_id="fluent_p1",
        channel_id="aspirate",
        value=10.0,
        conditions="viscous BSA protein",
    )
    described = await bench.tools.hw_describe(device_id="fluent_p1")
    assert "prior_experience" in described
    rows = described["prior_experience"]["aspirate"]
    assert rows
    assert "BSA" in rows[0]["command"]


@pytest.mark.asyncio
async def test_describe_omits_prior_experience_when_there_is_none() -> None:
    """Absent data is omitted rather than reported as empty."""
    bench = Bench(_context())
    described = await bench.tools.hw_describe(device_id="fluent_p1")
    assert "prior_experience" not in described


@pytest.mark.asyncio
async def test_tools_work_without_an_experience_store() -> None:
    """Learning is an addition, not a dependency: hardware must run without it."""
    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True, require_describe_before_write=False, persist_readings=False
        ),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    assert registry.outcome_recorder is None
    human = ScriptedHuman(ApprovalDecision.ALLOW_ONCE)
    tools = HardwareTools(
        registry,
        gate=ApprovalOrchestrator(SessionAwareGate(human), policy=ApprovalPolicyEngine()),
        session_id="s",
    )
    result = await tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is True
    described = await tools.hw_describe(device_id="fluent_p1")
    assert "prior_experience" not in described


@pytest.mark.asyncio
async def test_conditions_are_optional() -> None:
    """Omitting conditions still records the outcome, just with a weaker key."""
    bench = Bench(_context())
    result = await bench.tools.hw_dispense(
        device_id="fluent_p1", channel_id="aspirate", value=10.0
    )
    assert result["ok"] is True
    assert len(bench.store.records) == 1
    assert "conditions" not in bench.store.records[0]["action_description"]


# ════════════════════════════════════════════════════════════════
# The scenario this exists for
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_an_optimisation_performed_once_is_reusable() -> None:
    """The whole point: a discovery must survive the turn that made it.

    Mirrors the reported liquid-handler work, where an agent found that water tolerates a
    fast aspiration rate while a viscous protein sample does not -- and then lost it. Here
    a later run asking about the same situation gets the answer that worked.
    """
    bench = Bench(_context(settling=0.0, verify=True))
    transport = await bench.registry.transport("fluent_p1")

    # First run: water tracks well at a fast rate.
    await bench.tools.hw_dispense(
        device_id="fluent_p1",
        channel_id="aspirate",
        value=140.0,
        conditions="aqueous reagent, water",
    )

    # Then a viscous sample is tried fast, and the device does not keep up.
    bench.registry.outcome_recorder.record_command(
        device_id="fluent_p1",
        channel=bench.registry.context("fluent_p1").channel("aspirate"),
        value=140.0,
        conditions="viscous BSA protein, foams easily",
    )
    transport.set_value("aspirate", 60.0)
    bench.registry.outcome_recorder.observe(
        device_id="fluent_p1", channel_id="aspirate", value=60.0
    )

    # And slow works for it.
    await bench.tools.hw_dispense(
        device_id="fluent_p1",
        channel_id="aspirate",
        value=10.0,
        conditions="viscous BSA protein, foams easily",
    )

    # A later run asks about that situation and is told what worked, best first.
    rows = bench.registry.outcome_recorder.recall(
        device_id="fluent_p1",
        channel=bench.registry.context("fluent_p1").channel("aspirate"),
        conditions="viscous BSA protein",
        limit=3,
    )
    assert rows, "the protein-sample experience should be recallable"
    assert "10" in rows[0]["command"], f"expected the slow rate to lead, got {rows}"
    assert rows[0]["delta"] < 0.05

    # Relevance ranks above accuracy: both protein entries lead, ordered by how well the
    # device tracked, and the unrelated water run -- which tracked perfectly -- sits last.
    relevance = [row["relevance"] for row in rows]
    assert relevance == sorted(relevance, reverse=True)
    protein_rows = [row for row in rows if "BSA" in row["command"]]
    assert len(protein_rows) == 2
    assert protein_rows == list(rows[:2])
    assert [row["delta"] for row in protein_rows] == sorted(
        row["delta"] for row in protein_rows
    )
    # The attempt that failed to track is still on record, not discarded.
    assert max(row["delta"] for row in protein_rows) > 0.05
    assert "water" in rows[-1]["command"]


# ════════════════════════════════════════════════════════════════
# Model fidelity: the residual a learning loop can actually reduce
# ════════════════════════════════════════════════════════════════


def _observe_once(recorder: Any, *, commanded: float, factor: float, at: float) -> Any:
    """Command a channel and observe the device landing ``factor`` of the way there."""
    channel = _channel()
    recorder.record_command(device_id="rig", channel=channel, value=commanded, now=at)
    return recorder.observe(
        device_id="rig", channel_id=channel.channel_id, value=commanded * factor, now=at + 60.0
    )


def test_the_first_command_is_predicted_to_land_exactly() -> None:
    """A device is expected to do what it is told until evidence says otherwise."""
    recorder = HardwareOutcomeRecorder(experience_store=FakeExperienceStore())
    outcome = _observe_once(recorder, commanded=10.0, factor=0.92, at=0.0)

    assert outcome is not None
    assert outcome.predicted == outcome.commanded
    assert outcome.model_delta == outcome.delta, (
        "with no prior observation the two residuals are the same measurement"
    )


def test_a_consistent_bias_is_learned_so_model_error_falls_while_device_error_does_not() -> None:
    """The distinction that makes this worth having.

    Before this, the expected value was always the commanded value, so the residual
    measured the *device* and could never improve: a valve that always ran eight
    percent low reported the same error on its thousandth command as on its first, and
    nothing in the loop was capable of getting better at anything.

    The device error must stay put -- the hardware did not change -- while the model
    error falls, because that second number is the only one a learning loop can drive.
    """
    recorder = HardwareOutcomeRecorder(experience_store=FakeExperienceStore())
    device_errors: list[float] = []
    model_errors: list[float] = []
    for index in range(5):
        outcome = _observe_once(recorder, commanded=10.0, factor=0.92, at=index * 200.0)
        assert outcome is not None
        device_errors.append(outcome.delta)
        model_errors.append(outcome.model_delta)

    assert len(set(round(value, 6) for value in device_errors)) == 1, (
        f"the device did not change, so its error must not either: {device_errors}"
    )
    assert model_errors[-1] < model_errors[0], (
        f"the model never improved: {model_errors}"
    )
    assert model_errors[-1] < 1e-6, f"a fixed bias must be learned exactly: {model_errors}"


def test_an_accurate_device_can_be_unpredictable_and_says_so() -> None:
    """Accuracy and predictability are different claims, and both matter.

    A device with a known bias is compensable. A device that lands somewhere different
    every time is not, however close to the command it happens to get, and collapsing
    the two into one number would hide exactly that.

    The swing is sized against the *declared span*, not against the command, because
    that is what both thresholds normalise by: on a channel declared 0-200, being three
    units off is two percent and genuinely is within tolerance. The first version of
    this test alternated by thirty percent of a command of ten and asserted the device
    looked unpredictable -- it did not, and the implementation was right.
    """
    recorder = HardwareOutcomeRecorder(experience_store=FakeExperienceStore())
    channel = _channel()
    span = channel.envelope.max_value - channel.envelope.min_value
    commanded = 100.0
    swing = span * 0.2  # far outside the 5% of span that counts as tracking

    outcome = None
    for index, direction in enumerate((1, -1, 1, -1)):
        recorder.record_command(
            device_id="rig", channel=channel, value=commanded, now=index * 200.0
        )
        outcome = recorder.observe(
            device_id="rig",
            channel_id=channel.channel_id,
            value=commanded + direction * swing,
            now=index * 200.0 + 60.0,
        )
        assert outcome is not None
    assert outcome is not None
    assert outcome.predictable is False, (
        f"an alternating device must not look understood (model delta "
        f"{outcome.model_delta:.4f} on a span of {span:g})"
    )


def test_the_learned_correction_cannot_leave_the_declared_envelope() -> None:
    """A prediction may be wrong; it may not be absurd.

    One transient caught just after settling would otherwise push the expected value
    past the limits a human wrote down, and every later model residual would be
    measured against a value the device is not permitted to reach.
    """
    recorder = HardwareOutcomeRecorder(experience_store=FakeExperienceStore())
    channel = _channel()
    span = channel.envelope.max_value - channel.envelope.min_value

    for index in range(6):  # a wildly wrong reading, repeatedly
        recorder.record_command(device_id="rig", channel=channel, value=10.0, now=index * 200.0)
        recorder.observe(
            device_id="rig", channel_id=channel.channel_id, value=10_000.0,
            now=index * 200.0 + 60.0,
        )

    calibration = recorder.calibration_for("rig", channel.channel_id)
    assert calibration is not None
    recorder.record_command(device_id="rig", channel=channel, value=10.0, now=5000.0)
    outcome = recorder.observe(
        device_id="rig", channel_id=channel.channel_id, value=10.0, now=5060.0
    )
    assert outcome is not None
    assert abs(outcome.predicted - outcome.commanded) <= span * 0.25 + 1e-9, (
        f"the correction escaped its cap: predicted {outcome.predicted} for a span of {span}"
    )


def test_the_stored_experience_names_the_correction_it_applied() -> None:
    """A later reader must be able to tell an accurate device from an understood one."""
    store = FakeExperienceStore()
    recorder = HardwareOutcomeRecorder(experience_store=store)
    _observe_once(recorder, commanded=10.0, factor=0.92, at=0.0)
    _observe_once(recorder, commanded=10.0, factor=0.92, at=200.0)

    assert store.records[0]["predicted_effect"].startswith("reach 10")
    assert "prior bias" in store.records[1]["predicted_effect"], (
        f"the second prediction was corrected but does not say so: "
        f"{store.records[1]['predicted_effect']!r}"
    )


def test_the_calibration_is_reported_per_channel_for_a_person_to_read() -> None:
    """A correction the operator cannot inspect is one they cannot disagree with."""
    recorder = HardwareOutcomeRecorder(experience_store=FakeExperienceStore())
    channel = _channel()
    assert recorder.calibration_for("rig", channel.channel_id) is None, "untested until observed"

    _observe_once(recorder, commanded=10.0, factor=0.92, at=0.0)
    calibration = recorder.calibration_for("rig", channel.channel_id)
    assert calibration is not None
    bias, samples = calibration
    assert bias < 0, "the device undershot, so the correction must be negative"
    assert samples == 1


def test_an_outcome_built_without_a_prediction_reports_the_command() -> None:
    """A numeric default for the prediction produced nonsense about a perfect device.

    ``predicted: float = 0.0`` was indistinguishable from a genuine prediction of zero,
    so an outcome constructed without it described a device that landed exactly on 50 as
    "reach 0 (commanded 50, prior bias -50)". The sentinel makes the honest starting
    point -- the command itself -- automatic rather than the caller's responsibility.
    """
    outcome = PhysicalOutcome(
        device_id="d", channel_id="c", quantity="q", unit="u",
        commanded=50.0, observed=50.0, delta=0.0, residual=0.0,
    )
    assert outcome.predicted == 50.0
    assert outcome.to_predicted_effect() == "reach 50 u"
    assert outcome.model_delta == outcome.delta, (
        "with no prediction the two residuals are the same measurement, not a stand-in"
    )
    assert outcome.model_residual == outcome.residual
    assert outcome.predictable is True
