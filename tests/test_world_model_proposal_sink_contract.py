# Copyright (c) Alibaba, Inc. and its affiliates.
"""A caller-supplied proposal sink keeps its own signature.

Measured regression: the driver began passing ``observation_ids`` and ``environment``
to ``proposal_sink`` so the causal ledger could join a proposal to the observations
that caused it. The sink is supplied by the caller and its contract was
``sink(proposal)``, so every sink that had not adopted the new keywords raised
``TypeError`` -- inside a per-intent ``except Exception`` that logged at debug level
and continued. The visible effect was that **no acquisition was ever queued**, with a
green targeted test suite and no warning in the log.

Two contracts are pinned here, because either alone would have let it through:

* optional context is offered only to sinks that declare it (or take ``**kwargs``),
  so extending the causal payload can never break an existing sink;
* a wiring fault is logged as a warning rather than absorbed, so if this ever breaks
  again it says so instead of going quiet.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from leapflow.domain.adaptation_verdict import ACQUIRE, AdaptationVerdict
from leapflow.learning.world_model_driver import WorldModelEvolutionDriver
from leapflow.world_model.trajectory_grader import TeacherVerdict


class _Teacher:
    def __init__(self, verdict: TeacherVerdict) -> None:
        self._verdict = verdict

    async def grade_and_propose(self, *args: Any, **kwargs: Any) -> TeacherVerdict:
        return self._verdict


class _Intake:
    """Minimal stand-in that derives a need for what it just observed."""

    def __init__(self) -> None:
        self.observed: list[str] = []

    def observe_result(self, result, **kwargs):
        capability = str((result or {}).get("capability") or "")
        if capability:
            self.observed.append(capability)
        return {"observation_id": f"o{len(self.observed)}"}

    def requirements(self, *, min_count: int = 1, limit: int = 50):
        from leapflow.domain.capability_requirement import CapabilityRequirement

        return tuple(
            CapabilityRequirement.create(
                capability, "world_model", max_risk_level="read_only",
                requirement_id=f"req-{capability}",
            )
            for capability in dict.fromkeys(self.observed)
        )


def _drive(sink) -> Any:
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher(
            TeacherVerdict(
                grades=(),
                verdicts=(
                    AdaptationVerdict.create(ACQUIRE, "mail.send", "the app is now v3"),
                ),
            )
        ),
        intake=_Intake(),
        proposal_sink=sink,
    )
    return asyncio.run(driver.drive([{"action": "a"}], "reply in the thread"))


def test_legacy_single_argument_sink_still_receives_proposals():
    """The original contract: ``sink(proposal)`` and nothing else."""
    seen: list[Any] = []

    def sink(proposal):
        seen.append(proposal)
        return proposal.proposal_id

    result = _drive(sink)
    assert len(seen) == 1, "a one-argument sink must still be called"
    assert result.to_dict()["queued"] == 1


def test_sink_declaring_the_extras_receives_them():
    """A sink that opts in gets the causal context, so the ledger can join it."""
    captured: dict[str, Any] = {}

    def sink(proposal, *, observation_ids=(), environment=None):
        captured["observation_ids"] = tuple(observation_ids)
        captured["environment"] = environment
        return proposal.proposal_id

    result = _drive(sink)
    assert result.to_dict()["queued"] == 1
    assert captured["observation_ids"], "observation ids must reach an opted-in sink"


def test_var_keyword_sink_receives_the_extras():
    """``**kwargs`` counts as opting in; nothing needs to be listed explicitly."""
    captured: dict[str, Any] = {}

    def sink(proposal, **kwargs):
        captured.update(kwargs)
        return proposal.proposal_id

    assert _drive(sink).to_dict()["queued"] == 1
    assert "observation_ids" in captured


def test_a_sink_that_raises_a_wiring_fault_is_reported_not_swallowed(caplog):
    """A TypeError from inside the sink must be visible, not debug-only.

    The loop still continues -- one bad intent may not stop the rest -- but silence is
    what turned the original defect into an invisible outage.
    """
    def sink(proposal, *, observation_ids=(), environment=None):
        raise TypeError("sink is misconfigured")

    with caplog.at_level(logging.WARNING, logger="leapflow.learning.world_model_driver"):
        result = _drive(sink)

    assert result.to_dict()["queued"] == 0
    assert any(
        "acquisition not queued" in record.message for record in caplog.records
    ), "a wiring fault must be logged at warning level"
