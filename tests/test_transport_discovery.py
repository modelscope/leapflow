"""Entry-point discovery for out-of-tree transport kinds.

Exercises ``_discover_entry_points()`` in isolation and through the public API,
ensures idempotency, no-override semantics, and regression-freedom for the four
built-in transport kinds.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import leapflow.hardware.transports as transports_mod
from leapflow.hardware.transports import (
    _EP_GROUP,
    available_transports,
    build_transport,
)


# ── Helpers ──


def _fake_entry_point(name: str, value: str) -> SimpleNamespace:
    """Lightweight stand-in for ``importlib.metadata.EntryPoint``."""
    return SimpleNamespace(name=name, value=value)


@pytest.fixture(autouse=True)
def _reset_ep_state():
    """Reset the module-level scan flag and transport table between tests.

    Each test needs a pristine ``_TRANSPORTS`` dict and ``_ep_scanned`` flag
    so that one test's mutations do not leak into the next.
    """
    original_transports = dict(transports_mod._TRANSPORTS)
    original_flag = transports_mod._ep_scanned
    yield
    transports_mod._TRANSPORTS.clear()
    transports_mod._TRANSPORTS.update(original_transports)
    transports_mod._ep_scanned = original_flag


def _reset_scan_flag() -> None:
    """Allow ``_discover_entry_points`` to run again for a single test."""
    transports_mod._ep_scanned = False


# ── Core discovery tests ──


def test_discover_merges_new_kinds_from_entry_points() -> None:
    """An entry-point whose name is absent from the table is merged."""
    _reset_scan_flag()

    fake_eps = [_fake_entry_point("host.macos", "leapflow_host.macos:build_transport")]

    with patch("importlib.metadata.entry_points", return_value=fake_eps):
        transports_mod._discover_entry_points()

    assert "host.macos" in transports_mod._TRANSPORTS
    assert transports_mod._TRANSPORTS["host.macos"] == "leapflow_host.macos:build_transport"


def test_discover_does_not_overwrite_existing_kinds() -> None:
    """A built-in kind must never be hijacked by an installed package."""
    _reset_scan_flag()

    original_mock_target = transports_mod._TRANSPORTS["mock"]
    fake_eps = [_fake_entry_point("mock", "evil_pkg.hijack:build_transport")]

    with patch("importlib.metadata.entry_points", return_value=fake_eps):
        transports_mod._discover_entry_points()

    assert transports_mod._TRANSPORTS["mock"] == original_mock_target


def test_discover_does_not_overwrite_manually_registered_kinds() -> None:
    """A kind registered via ``register_transport()`` takes precedence."""
    _reset_scan_flag()

    transports_mod._TRANSPORTS["custom.rig"] = "my_pkg.rig:build_transport"
    fake_eps = [_fake_entry_point("custom.rig", "other_pkg.rig:build")]

    with patch("importlib.metadata.entry_points", return_value=fake_eps):
        transports_mod._discover_entry_points()

    assert transports_mod._TRANSPORTS["custom.rig"] == "my_pkg.rig:build_transport"


def test_discover_is_idempotent() -> None:
    """The scan runs exactly once; a second call is a no-op."""
    _reset_scan_flag()

    call_count = 0
    real_entry_points = importlib.metadata.entry_points

    def counting_entry_points(**kwargs: Any):
        nonlocal call_count
        call_count += 1
        return real_entry_points(**kwargs)

    with patch("importlib.metadata.entry_points", side_effect=counting_entry_points):
        transports_mod._discover_entry_points()
        transports_mod._discover_entry_points()
        transports_mod._discover_entry_points()

    assert call_count == 1


def test_discover_tolerates_empty_group() -> None:
    """No entry-points installed is the common case, not an error."""
    _reset_scan_flag()

    with patch("importlib.metadata.entry_points", return_value=[]):
        transports_mod._discover_entry_points()

    # Built-in kinds still present, nothing added.
    assert set(available_transports()) >= {"mock", "simulated", "python", "mcp"}


def test_discover_tolerates_import_error() -> None:
    """If importlib.metadata is somehow absent, discovery degrades silently."""
    _reset_scan_flag()

    with patch.dict("sys.modules", {"importlib.metadata": None}):
        # Force re-import to hit the ImportError path.
        # Since _discover_entry_points does a late import, patching the module
        # in sys.modules makes the import fail.
        transports_mod._discover_entry_points()

    # No crash, built-in kinds still intact.
    assert "mock" in transports_mod._TRANSPORTS


# ── Integration through public API ──


def test_available_transports_triggers_discovery() -> None:
    """``available_transports()`` calls discovery before enumerating."""
    _reset_scan_flag()

    fake_eps = [_fake_entry_point("bench.sim", "leapflow_bench.sim:build_transport")]

    with patch("importlib.metadata.entry_points", return_value=fake_eps):
        kinds = available_transports()

    assert "bench.sim" in kinds


def test_build_transport_triggers_discovery() -> None:
    """``build_transport()`` calls discovery so an EP-only kind resolves."""
    _reset_scan_flag()

    # The factory we point to must be importable and callable.
    factory_mock = MagicMock()
    factory_mock.return_value = MagicMock()

    fake_eps = [_fake_entry_point("test.ep", "leapflow.hardware.transports.mock:build_transport")]

    with patch("importlib.metadata.entry_points", return_value=fake_eps):
        # build_transport should discover "test.ep" then resolve it normally.
        transport = build_transport("test.ep", {"values": {"sensor": 1.0}})

    assert transport is not None


def test_build_transport_with_discovered_kind_resolves_to_factory() -> None:
    """A discovered entry-point is importable and produces a transport."""
    _reset_scan_flag()

    # Wire an EP that points to the built-in mock factory -- a known-good path.
    fake_eps = [
        _fake_entry_point("ep.mock", "leapflow.hardware.transports.mock:build_transport")
    ]

    with patch("importlib.metadata.entry_points", return_value=fake_eps):
        transport = build_transport(
            "ep.mock", {"values": {"sensor": 42.0}, "halt_supported": True}
        )

    from leapflow.hardware.transport import HardwareTransport

    assert isinstance(transport, HardwareTransport)


# ── Regression: built-in kinds are unaffected ──


@pytest.mark.parametrize("kind", ["mock", "simulated", "python", "mcp"])
def test_builtin_kinds_survive_discovery(kind: str) -> None:
    """All four original kinds remain registered after discovery runs."""
    _reset_scan_flag()

    with patch("importlib.metadata.entry_points", return_value=[]):
        transports_mod._discover_entry_points()

    assert kind in transports_mod._TRANSPORTS


def test_builtin_kinds_in_available_transports() -> None:
    """``available_transports()`` always includes the four built-in kinds."""
    kinds = available_transports()
    for builtin in ("mock", "simulated", "python", "mcp"):
        assert builtin in kinds


# ── EP_GROUP constant ──


def test_ep_group_constant_value() -> None:
    """The group string is the contract between drivers and discovery."""
    assert _EP_GROUP == "leapflow.hardware.transports"
