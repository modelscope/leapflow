# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end evolution lifecycle: environment trigger through sweep.

Single-process integration test covering the full governed pipeline
introduced across Phases 1-4:

    Phase A — Environment trigger → AdaptationVerdict → EvolutionIntent
    Phase B — Capability gap detection → proposal enqueue
    Phase C — Generate, validate, register (good + INCOMPATIBLE path)
    Phase D — Dual approval → install → fiber ACTIVE → tool invocable
    Phase E — Trust accrual (DRAFT→CANDIDATE), quarantine, unfreeze (Phase 2)
    Phase F — Proposal TTL sweep (Phase 1)
    Phase G — Event chain verification (causation/correlation)

Modelled after ``test_llm_coevolution_e2e`` for the single-process setup
but exercising every Phase 1-4 feature through their real APIs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

# ---------------------------------------------------------------------------
# Valid echo plugin code -- returned by the fake LLM
# ---------------------------------------------------------------------------
ECHO_PLUGIN_CODE = '''
"""Auto-generated echo plugin for lifecycle E2E test."""
from typing import Any
from leapflow.plugins.protocol import ToolMetadata


class EchoLifecyclePlugin:
    """Echo plugin for the full lifecycle journey."""

    @property
    def plugin_id(self) -> str:
        return "echo_lifecycle"

    @property
    def category(self) -> str:
        return "custom"

    @property
    def dependencies(self) -> list:
        return []

    def bind_runtime(self, **deps: Any) -> None:
        pass

    @property
    def tools(self) -> list:
        return [ToolMetadata(
            name="echo_lifecycle_test",
            description="Echo tool for lifecycle E2E test",
            parameters_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
            handler=self._echo_handler,
            x_leapflow={"category": "custom", "risk_level": "read_only"},
            provides_capabilities=("echo.lifecycle",),
        )]

    async def _echo_handler(self, message: str = "", **kwargs: Any) -> dict:
        return {
            "ok": True,
            "echoed": message,
            "source": "lifecycle_generated",
            "observed_effect": "echoed the input message",
        }


plugin = EchoLifecyclePlugin()
'''

PLUGIN_ID = "echo_lifecycle"
TOOL_NAME = "echo_lifecycle_test"
PROFILE_ID = "test-lifecycle"


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------
class _FakeLLM:
    """Canned LLM returning the echo plugin code."""

    async def achat(self, messages: Any) -> str:
        return f"```python\n{ECHO_PLUGIN_CODE}\n```"


class _ApprovedResult:
    approved = True
    denial_message = ""


class _ApprovingGate:
    """Always-approve gate for both content and mutation approval."""

    async def evaluate(self, descriptor: Any) -> _ApprovedResult:
        return _ApprovedResult()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def evolution_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create all scratch directories for the lifecycle test."""
    dirs = {
        "plugins": tmp_path / "plugins",
        "artifacts": tmp_path / "artifacts",
        "events_db": tmp_path / "events.duckdb",
        "stats_db": tmp_path / "plugin_stats.duckdb",
    }
    dirs["plugins"].mkdir(parents=True, exist_ok=True)
    dirs["artifacts"].mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def event_store(evolution_dirs: dict[str, Path]) -> Any:
    """A real DuckDB-backed evolution event store."""
    from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore

    return DuckDBEvolutionEventStore(evolution_dirs["events_db"])


@pytest.fixture
def proposal_store(event_store: Any) -> Any:
    """A real proposal store wired to the event store."""
    from leapflow.storage.capability_proposal_queue import (
        EvolutionCapabilityProposalStore,
    )

    return EvolutionCapabilityProposalStore(
        event_store, profile_id=PROFILE_ID, proposal_ttl_hours=72,
    )


@pytest.fixture
def artifact_store(evolution_dirs: dict[str, Path]) -> Any:
    """A real CAS artifact store."""
    from leapflow.evolution.artifact_store import ContentAddressedArtifactStore

    return ContentAddressedArtifactStore(evolution_dirs["artifacts"])


@pytest.fixture
def policy() -> Any:
    """An evolution policy at generate_only autonomy (permits generation)."""
    from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy

    return AdaptiveEvolutionPolicy(autonomy_level="generate_only")


@pytest.fixture
def orchestrator(proposal_store: Any, artifact_store: Any, policy: Any) -> Any:
    """A real ProposalOrchestrator wired to stores and approval."""
    from leapflow.plugins.proposal_orchestrator import ProposalOrchestrator

    return ProposalOrchestrator(
        queue=proposal_store,
        artifact_store=artifact_store,
        approval_gate=_ApprovingGate(),
        policy=policy,
    )


@pytest.fixture
def trust_ledger() -> Any:
    """A trust ledger with short thresholds for fast promotion."""
    from leapflow.learning.plugin_trust import PluginTrustLedger

    return PluginTrustLedger(
        candidate_at=5, verified_at=20, production_at=50, demote_after=3,
    )


@pytest.fixture
def stats_store(evolution_dirs: dict[str, Path]) -> Any:
    """A real DuckDB-backed plugin stats store."""
    from leapflow.learning.plugin_stats_store import PluginStatsStore

    return PluginStatsStore(db_path=evolution_dirs["stats_db"])


@pytest.fixture
def cleanup_plugin():
    """Tear down registry / sys.modules state after the test."""
    yield
    sys.modules.pop(PLUGIN_ID, None)
    try:
        from leapflow.plugins import get_registry, get_scoped_registry

        reg = get_registry()
        scoped = get_scoped_registry()
        if PLUGIN_ID in reg.plugins:
            reg.unregister_plugin(PLUGIN_ID)
        if PLUGIN_ID in scoped._fibers:
            fiber = scoped._fibers.pop(PLUGIN_ID)
            try:
                fiber.dispose()
            except Exception:
                pass
    except Exception:
        pass


# ===================================================================
# THE TEST
# ===================================================================
@pytest.mark.asyncio
async def test_full_evolution_lifecycle(
    evolution_dirs: dict[str, Path],
    event_store: Any,
    proposal_store: Any,
    artifact_store: Any,
    policy: Any,
    orchestrator: Any,
    trust_ledger: Any,
    stats_store: Any,
    cleanup_plugin: None,
) -> None:
    """Complete governed evolution pipeline, single-process."""

    # ── Phase A — Environment Trigger and Intent ──────────────────
    from leapflow.domain.adaptation_verdict import AdaptationVerdict

    verdict = AdaptationVerdict.create(
        action="acquire",
        capability="echo.lifecycle",
        knowledge="The runtime lacks an echo capability for lifecycle testing.",
        rationale="No existing plugin provides echo.lifecycle.",
        confidence=0.9,
        max_risk_level="read_only",
        evidence_ids=("obs-001", "obs-002"),
    )
    intent = verdict.to_intent()
    assert intent is not None, "acquire verdict must produce an intent"
    assert intent.capability == "echo.lifecycle"
    assert intent.evidence_ids == ("obs-001", "obs-002")
    assert intent.max_risk_level == "read_only"
    # Identity is derived from the verdict, not minted fresh
    assert verdict.verdict_id.removeprefix("adv-") in intent.intent_id

    # ── Phase B — Capability Gap to Proposal ──────────────────────
    from leapflow.domain.capability_requirement import CapabilityRequirement
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    detector = CapabilityGapDetector()
    plugin_proposal = detector.proposal_from_evolution_intent(intent)
    assert plugin_proposal.plugin_id  # non-empty, derived from capability slug
    assert plugin_proposal.risk_level == "read_only"

    # Enqueue to the durable store
    requirement = intent.to_requirement()
    assert isinstance(requirement, CapabilityRequirement)
    item = proposal_store.enqueue(
        requirements=[requirement],
        environment={"fingerprint_id": "test-fp", "workspace_id": "ws-lifecycle"},
        risk={"risk_level": "read_only"},
        source="world_model",
        observation_ids=list(intent.evidence_ids),
        metadata={"plugin_id": PLUGIN_ID},
    )
    assert item.status == "PENDING"
    proposal_id = item.proposal_id

    # Re-enqueue is idempotent
    item2 = proposal_store.enqueue(
        requirements=[requirement],
        environment={"fingerprint_id": "test-fp", "workspace_id": "ws-lifecycle"},
    )
    assert item2.proposal_id == proposal_id, "re-enqueue must be idempotent"

    # ── Phase C — Generate and Validate ───────────────────────────
    from leapflow.learning.plugin_generator import (
        PluginGenerationRequest,
        PluginGenerator,
    )

    generator = PluginGenerator(llm_provider=_FakeLLM())
    gen_result = await generator.generate_and_validate(
        PluginGenerationRequest(
            plugin_id=PLUGIN_ID,
            description="Echo tool for lifecycle testing",
            provides_capabilities=("echo.lifecycle",),
        ),
    )
    assert gen_result["ok"], f"Generation failed: {gen_result.get('error')}"
    assert TOOL_NAME in gen_result["exposed_tools"]
    generated_code = gen_result["code"]

    # Register through the orchestrator (PENDING → GENERATED)
    registered = orchestrator.register_generated(
        proposal_id,
        generated_code,
        validation={"ok": True, "compatibility_ok": True, "exposed_tools": [TOOL_NAME]},
    )
    assert registered.status == "GENERATED"

    # ── Phase C (INCOMPATIBLE path) ───────────────────────────────
    # Create a second proposal that will fail validation
    bad_req = CapabilityRequirement.create(
        "bad.tool", "world_model", evidence="should fail",
        requirement_id="req-bad-tool",
    )
    bad_item = proposal_store.enqueue(
        requirements=[bad_req],
        environment={"fingerprint_id": "test-fp-bad"},
        risk={"risk_level": "read_only"},
        source="world_model",
    )
    assert bad_item.status == "PENDING"
    bad_proposal_id = bad_item.proposal_id

    bad_registered = orchestrator.register_generated(
        bad_proposal_id,
        "invalid code",
        validation={"ok": False, "compatibility_ok": False, "error": "syntax"},
    )
    assert bad_registered.status == "FAILED", "INCOMPATIBLE code must not reach CAS"

    # ── Phase D — Approval and Install ────────────────────────────
    # Content approval (GENERATED → APPROVED)
    content_approval = await orchestrator.approve_content(proposal_id)
    assert content_approval.approved is True
    stored = proposal_store.get(proposal_id)
    assert stored is not None and stored.status == "APPROVED"

    # Mutation approval
    mutation_approval = await orchestrator.authorize_mutation(proposal_id)
    assert mutation_approval.approved is True

    # Actual install via self_management
    from leapflow.plugins import get_registry

    reg = get_registry()
    reg.assemble()
    self_mgmt = reg.get_plugin("self_management")
    self_mgmt._plugin_approval_gate = _ApprovingGate()
    self_mgmt.bind_runtime(plugin_install_dir=str(evolution_dirs["plugins"]))

    try:
        install_result = await self_mgmt._plugin_install_handler(
            plugin_id=PLUGIN_ID,
            code=generated_code,
        )
        assert install_result["ok"], f"Install failed: {install_result.get('error')}"
        assert TOOL_NAME in install_result["installed_tools"]

        # Plugin file written
        assert (evolution_dirs["plugins"] / f"{PLUGIN_ID}.py").exists()

        # Fiber ACTIVE + tool in registry
        assert PLUGIN_ID in reg.plugins
        assert TOOL_NAME in reg.tool_handlers

        # Invoke the tool
        handler = reg.tool_handlers[TOOL_NAME]
        invoke_result = await handler(message="lifecycle hello")
        assert invoke_result["ok"] is True
        assert invoke_result["echoed"] == "lifecycle hello"

        # Record install in orchestrator lifecycle
        orchestrator.record_installed(proposal_id, install_result)
        stored = proposal_store.get(proposal_id)
        assert stored is not None and stored.status == "INSTALLED"

    finally:
        self_mgmt._plugin_approval_gate = None
        self_mgmt._plugin_install_dir = None

    # ── Phase E — Trust Accrual and Governance ────────────────────
    from leapflow.learning.plugin_trust import PluginTrustLevel

    # Start at DRAFT
    assert trust_ledger.level(PLUGIN_ID) == PluginTrustLevel.DRAFT

    # Record 5 consecutive successes → CANDIDATE
    for _ in range(5):
        trust_ledger.record_success(PLUGIN_ID)
    assert trust_ledger.level(PLUGIN_ID) == PluginTrustLevel.CANDIDATE

    # Record trust transition in DuckDB (Phase 4 feature)
    stats_store.record_trust_transition(
        plugin_id=PLUGIN_ID,
        from_level=PluginTrustLevel.DRAFT.value,
        to_level=PluginTrustLevel.CANDIDATE.value,
        trigger="consecutive_success",
        consecutive_ok=5,
        consecutive_fail=0,
    )

    # Verify trust_history (Phase 4 feature)
    from leapflow.learning.plugin_trust import trust_history

    history = trust_history(evolution_dirs["stats_db"], plugin_id=PLUGIN_ID)
    assert len(history) >= 1
    rec = history[0]
    assert rec.plugin_id == PLUGIN_ID
    assert rec.from_level == PluginTrustLevel.DRAFT.value
    assert rec.to_level == PluginTrustLevel.CANDIDATE.value
    assert rec.trigger == "consecutive_success"

    # Record 3 consecutive failures → demotion (CANDIDATE → DRAFT)
    for _ in range(3):
        trust_ledger.record_failure(PLUGIN_ID)
    assert trust_ledger.level(PLUGIN_ID) == PluginTrustLevel.DRAFT

    # Hard failure → freeze
    trust_ledger.record_success(PLUGIN_ID)  # Reset from demotion
    trust_ledger.record_failure(PLUGIN_ID, hard=True)
    assert trust_ledger.is_frozen(PLUGIN_ID)
    assert trust_ledger.level(PLUGIN_ID) == PluginTrustLevel.DRAFT

    # Transition proposal to QUARANTINED (Phase 2: quarantine recovery)
    proposal_store.transition(proposal_id, "QUARANTINED")
    stored = proposal_store.get(proposal_id)
    assert stored is not None and stored.status == "QUARANTINED"

    # Phase 2: unfreeze + QUARANTINED → PROBATION
    unfrozen = trust_ledger.unfreeze(PLUGIN_ID)
    assert unfrozen is True
    assert not trust_ledger.is_frozen(PLUGIN_ID)
    assert trust_ledger.level(PLUGIN_ID) == PluginTrustLevel.DRAFT

    proposal_store.transition(proposal_id, "PROBATION")
    stored = proposal_store.get(proposal_id)
    assert stored is not None and stored.status == "PROBATION"

    # Record the unquarantine trust transition
    stats_store.record_trust_transition(
        plugin_id=PLUGIN_ID,
        from_level=PluginTrustLevel.DRAFT.value,  # frozen→unfrozen at DRAFT
        to_level=PluginTrustLevel.DRAFT.value,
        trigger="unfreeze",
        consecutive_ok=0,
        consecutive_fail=0,
    )

    # ── Phase F — Proposal Sweep (Phase 1: TTL expiry) ────────────
    from leapflow.evolution.sweep import CoevolutionSweep, SweepOutcome

    # Create a stale proposal with expires_at in the past
    stale_req = CapabilityRequirement.create(
        "stale.tool", "world_model", evidence="stale evidence",
        requirement_id="req-stale-tool",
    )
    # Build a store with 0 TTL so proposals expire immediately
    from leapflow.storage.capability_proposal_queue import (
        EvolutionCapabilityProposalStore,
    )

    sweep_store = EvolutionCapabilityProposalStore(
        event_store, profile_id=PROFILE_ID, proposal_ttl_hours=0,
    )
    # TTL=0 means no expiry timestamp is set, so create with explicit past expiry
    stale_item, stale_event = sweep_store.prepare_enqueue(
        requirements=[stale_req],
        environment={"fingerprint_id": "test-fp-stale"},
        risk={"risk_level": "read_only"},
        source="sweep_test",
        occurred_at=time.time() - 7200,  # 2 hours ago
    )
    if stale_event is not None:
        event_store.append(stale_event)
    stale_id = stale_item.proposal_id

    # Manually set expires_at to the past by updating it with a past timestamp
    # Since TTL=0 means expires_at=None, we use the main store (TTL=72h)
    # and create a proposal with occurred_at far in the past
    from leapflow.storage.capability_proposal_queue import CapabilityProposalItem

    # Use the main proposal_store (TTL=72h) and create an already-expired proposal
    expired_req = CapabilityRequirement.create(
        "expired.tool", "world_model", evidence="expired evidence",
        requirement_id="req-expired-tool",
    )
    # Enqueue with the main store so expires_at is set
    expired_item, expired_event = proposal_store.prepare_enqueue(
        requirements=[expired_req],
        environment={"fingerprint_id": "test-fp-expired"},
        risk={"risk_level": "read_only"},
        source="sweep_test",
        occurred_at=time.time() - 72 * 3600 - 1,  # Beyond TTL
    )
    if expired_event is not None:
        event_store.append(expired_event)
    expired_id = expired_item.proposal_id

    # Verify the expired_at is in the past
    stored_expired = proposal_store.get(expired_id)
    assert stored_expired is not None
    assert stored_expired.expires_at is not None
    assert stored_expired.expires_at < time.time(), "proposal must be past its TTL"

    sweep = CoevolutionSweep(
        orchestrator=orchestrator,
        proposal_store=proposal_store,
    )
    expired_count, superseded_count = sweep._sweep_proposal_expiry()
    assert expired_count >= 1, f"expected at least 1 expired, got {expired_count}"

    # Verify the proposal is now EXPIRED
    stored_expired = proposal_store.get(expired_id)
    assert stored_expired is not None and stored_expired.status == "EXPIRED"

    # ── Phase G — Event Chain Verification ────────────────────────
    from leapflow.domain.event_types import EvolutionEventType

    # Read all events for the main proposal
    records = event_store.read(
        profile_id=PROFILE_ID,
        correlation_id=proposal_id,
        limit=500,
    )
    event_types = [r.event.event_type for r in records]

    # Key lifecycle events must be present
    assert EvolutionEventType.PROPOSAL_CREATED in event_types, (
        f"PROPOSAL_CREATED missing; got: {event_types}"
    )
    assert EvolutionEventType.PROPOSAL_GENERATED in event_types, (
        f"PROPOSAL_GENERATED missing; got: {event_types}"
    )
    assert EvolutionEventType.PROPOSAL_APPROVED in event_types, (
        f"PROPOSAL_APPROVED missing; got: {event_types}"
    )
    assert EvolutionEventType.PLUGIN_INSTALLED in event_types, (
        f"PLUGIN_INSTALLED missing; got: {event_types}"
    )
    assert EvolutionEventType.PLUGIN_QUARANTINED in event_types, (
        f"PLUGIN_QUARANTINED missing; got: {event_types}"
    )
    assert EvolutionEventType.PLUGIN_PROBATION_STARTED in event_types, (
        f"PLUGIN_PROBATION_STARTED missing; got: {event_types}"
    )

    # All events in the chain share the same correlation_id (the proposal_id)
    for record in records:
        assert record.event.context.correlation_id == proposal_id, (
            f"event {record.event.event_type} has wrong correlation_id: "
            f"{record.event.context.correlation_id}"
        )

    # The expired proposal's events also exist
    expired_records = event_store.read(
        profile_id=PROFILE_ID,
        correlation_id=expired_id,
        limit=100,
    )
    expired_event_types = [r.event.event_type for r in expired_records]
    assert EvolutionEventType.PROPOSAL_CREATED in expired_event_types
    assert EvolutionEventType.PROPOSAL_EXPIRED in expired_event_types

    # The failed proposal's events exist
    failed_records = event_store.read(
        profile_id=PROFILE_ID,
        correlation_id=bad_proposal_id,
        limit=100,
    )
    failed_event_types = [r.event.event_type for r in failed_records]
    assert EvolutionEventType.PROPOSAL_CREATED in failed_event_types
    assert EvolutionEventType.PROPOSAL_FAILED in failed_event_types

    print(
        f"[lifecycle E2E] All 7 phases passed. "
        f"Events: {len(records)} main, {len(expired_records)} expired, "
        f"{len(failed_records)} failed."
    )
