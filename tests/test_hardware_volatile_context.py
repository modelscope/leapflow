# Copyright (c) Alibaba, Inc. and its affiliates.
"""HardwareVolatileContextProvider: session filtering, PCD gating, rendering."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from leapflow.hardware.volatile_context import HardwareVolatileContextProvider


# ── Stub registry ──────────────────────────────────────────────────────


def _stub_registry(
    *,
    contexts: tuple[Any, ...] = (),
    pending: dict[tuple[str, str], list[Any]] | None = None,
    events: tuple[Any, ...] = (),
    degraded: set[str] | None = None,
    control: dict[str, dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Return a registry-shaped object holding only state the provider reads."""
    _degraded = degraded or set()
    _control = control or {}

    return SimpleNamespace(
        contexts=lambda: contexts,
        pending_write_expectations=pending or {},
        recent_events=lambda limit=10: events[-max(0, limit):],
        is_device_degraded=lambda did: did in _degraded,
        control_loop_status=lambda did: _control.get(did),
    )


def _context(device_id: str = "heater", writable: int = 2) -> SimpleNamespace:
    channels = [SimpleNamespace() for _ in range(writable)]
    return SimpleNamespace(device_id=device_id, writable_channels=tuple(channels))


def _verdict_event(
    device_id: str = "heater",
    channel_id: str = "power",
    status: str = "verified",
    operation_id: str = "write-abc",
    observation_channel_id: str = "temperature",
    identity: str = "",
) -> SimpleNamespace:
    """Return an event shaped like WriteVerdictEvent."""
    expectation = SimpleNamespace(
        device_id=device_id,
        command_channel_id=channel_id,
        operation_id=operation_id,
        observation_channel_id=observation_channel_id,
    )
    verdict = SimpleNamespace(
        status=status,
        expectation=expectation,
    )
    return SimpleNamespace(
        kind="write_verdict",
        identity=identity,
        verdict=verdict,
        device_id=device_id,
    )


def _pending_expectation(
    device_id: str = "heater",
    command_channel_id: str = "power",
    observation_channel_id: str = "temperature",
    operation_id: str = "write-pending",
    identity: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        device_id=device_id,
        command_channel_id=command_channel_id,
        observation_channel_id=observation_channel_id,
        operation_id=operation_id,
        identity=identity,
    )


# ── Provider identity ──────────────────────────────────────────────────


class TestProviderIdentity:
    def test_provider_id(self) -> None:
        reg = _stub_registry(contexts=(_context(),))
        provider = HardwareVolatileContextProvider(reg)
        assert provider.provider_id == "hardware.runtime"

    def test_categories(self) -> None:
        reg = _stub_registry(contexts=(_context(),))
        provider = HardwareVolatileContextProvider(reg)
        assert "hardware" in provider.categories


# ── Empty / graceful degradation ───────────────────────────────────────


class TestEmptyGraceful:
    def test_no_contexts_returns_empty(self) -> None:
        reg = _stub_registry(contexts=())
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert result == ""

    def test_registry_error_returns_empty(self) -> None:
        reg = SimpleNamespace(contexts=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert result == ""


# ── Device rendering ───────────────────────────────────────────────────


class TestDeviceRendering:
    def test_available_device(self) -> None:
        reg = _stub_registry(contexts=(_context("heater", writable=3),))
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "heater" in result
        assert "available" in result
        assert "writable_channels=3" in result

    def test_degraded_device(self) -> None:
        reg = _stub_registry(
            contexts=(_context("heater"),),
            degraded={"heater"},
        )
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "degraded" in result

    def test_control_status_active(self) -> None:
        reg = _stub_registry(
            contexts=(_context("heater"),),
            control={"heater": {"state": "running", "frequency_hz": 10}},
        )
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "control=running" in result
        # Frequency only at full disclosure
        assert "frequency_hz" not in result

    def test_control_frequency_at_full(self) -> None:
        reg = _stub_registry(
            contexts=(_context("heater"),),
            control={"heater": {"state": "running", "frequency_hz": 10}},
        )
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="full")
        assert "frequency_hz=10" in result

    def test_max_devices_respected(self) -> None:
        contexts = tuple(_context(f"dev{i}") for i in range(10))
        reg = _stub_registry(contexts=contexts)
        provider = HardwareVolatileContextProvider(reg, max_devices=3)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "7 additional device(s) omitted" in result


# ── Session-scoped verdict filtering ──────────────────────────────────


class TestVerdictSessionFiltering:
    def test_own_session_verdicts_visible(self) -> None:
        events = (
            _verdict_event(identity="s1", operation_id="op1"),
            _verdict_event(identity="s2", operation_id="op2"),
        )
        reg = _stub_registry(contexts=(_context(),), events=events)
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "op1" in result
        assert "op2" not in result

    def test_empty_identity_events_visible_to_empty_session(self) -> None:
        events = (_verdict_event(identity="", operation_id="opX"),)
        reg = _stub_registry(contexts=(_context(),), events=events)
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="", disclosure_level="expanded")
        assert "opX" in result

    def test_empty_identity_events_hidden_from_named_session(self) -> None:
        events = (_verdict_event(identity="", operation_id="opY"),)
        reg = _stub_registry(contexts=(_context(),), events=events)
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "opY" not in result


# ── Session-scoped pending filtering ──────────────────────────────────


class TestPendingSessionFiltering:
    def test_own_session_pending_visible(self) -> None:
        pending = {
            ("heater", "temperature"): [
                _pending_expectation(identity="s1", operation_id="op-p1"),
                _pending_expectation(identity="s2", operation_id="op-p2"),
            ],
        }
        reg = _stub_registry(contexts=(_context(),), pending=pending)
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="s1", disclosure_level="expanded")
        assert "op-p1" in result
        assert "op-p2" not in result

    def test_empty_identity_pending_visible_to_empty_session(self) -> None:
        pending = {
            ("heater", "temperature"): [
                _pending_expectation(identity="", operation_id="opZ"),
            ],
        }
        reg = _stub_registry(contexts=(_context(),), pending=pending)
        provider = HardwareVolatileContextProvider(reg)
        result = provider.render_volatile_context(session_id="", disclosure_level="expanded")
        assert "opZ" in result


# ── belongs_to_session ────────────────────────────────────────────────


class TestBelongsToSession:
    def test_matching(self) -> None:
        assert HardwareVolatileContextProvider._belongs_to_session("s1", "s1") is True

    def test_mismatch(self) -> None:
        assert HardwareVolatileContextProvider._belongs_to_session("s1", "s2") is False

    def test_empty_owner_empty_session(self) -> None:
        assert HardwareVolatileContextProvider._belongs_to_session("", "") is True

    def test_empty_owner_named_session(self) -> None:
        assert HardwareVolatileContextProvider._belongs_to_session("", "s1") is False

    def test_named_owner_empty_session(self) -> None:
        # When session_id is empty, any event with a named owner is visible (legacy compat)
        assert HardwareVolatileContextProvider._belongs_to_session("s1", "") is False
