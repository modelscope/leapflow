# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tier 3 live tests — real LLM provider, credential-gated, budget-bounded.

This package holds the smallest set of end-to-end tests that only a *real*
provider can prove: single-turn answering, tool-call round-trips, streaming
integrity, graceful context handling, and transient-error recovery. Everything
else is covered offline by the mock layer and the cassette-replay journeys.

The lane is deliberately expensive-to-run and cheap-per-run:

- It never runs by default. Locally, with no credentials in the environment,
  every test skips (see :mod:`tests.live.conftest`).
- Each test carries a hard call / token / wall-clock budget, enforced through
  the ``live_budget`` fixture. A test that starts burning tokens fails fast
  instead of running the bill up.
- CI runs it only on a nightly schedule or an explicit ``ci:live`` label /
  manual dispatch, where the credentials live in repository secrets.
"""
