# Copyright (c) Alibaba, Inc. and its affiliates.
"""Full fiberization coverage tests.

Verifies that all three plugin subsystems (tools, gateway adapters, LLM
providers) bring their built-in plugins under PluginFiber lifecycle
management at boot via ``adopt_existing_plugins()``.

The adoption path is additive tracking only: it must NOT re-register
plugins (which would raise Duplicate plugin_id / overwrite entries), and
it must leave every fiber in the ACTIVE state.
"""

from __future__ import annotations

import pytest

from leapflow.domain.plugin_fiber import FiberState


# ════════════════════════════════════════════════════════════════
# Tools subsystem
# ════════════════════════════════════════════════════════════════


class TestToolPluginFiberization:
    """Every built-in tool plugin gets an ACTIVE fiber after boot."""

    def test_all_builtin_tool_plugins_have_fibers(self) -> None:
        # Rebuild the plugin singletons for a clean, deterministic boot.
        import leapflow.plugins as plugins_mod

        plugins_mod._registry = None
        plugins_mod._scoped_registry = None

        reg = plugins_mod.get_registry()
        reg.assemble()
        scoped = plugins_mod.get_scoped_registry()

        plugin_ids = set(reg.plugins.keys())
        fiber_ids = set(scoped.fibers.keys())

        assert plugin_ids, "expected at least one built-in tool plugin"
        missing = plugin_ids - fiber_ids
        assert not missing, f"tool plugins without fibers: {missing}"

        for pid, fiber in scoped.fibers.items():
            assert fiber.state == FiberState.ACTIVE, (
                f"tool plugin '{pid}' fiber not ACTIVE: {fiber.state}"
            )

    def test_adopt_does_not_double_register(self) -> None:
        import leapflow.plugins as plugins_mod

        plugins_mod._registry = None
        plugins_mod._scoped_registry = None

        reg = plugins_mod.get_registry()
        scoped = plugins_mod.get_scoped_registry()

        plugin_count_before = len(reg.plugins)
        fiber_count_before = len(scoped.fibers)

        # Calling adopt again must be idempotent — no duplicate registration,
        # no additional fibers, and it must not raise.
        scoped.adopt_existing_plugins()

        assert len(reg.plugins) == plugin_count_before
        assert len(scoped.fibers) == fiber_count_before


# ════════════════════════════════════════════════════════════════
# Gateway subsystem
# ════════════════════════════════════════════════════════════════


class TestGatewayAdapterFiberization:
    """Every built-in gateway adapter gets an ACTIVE fiber after boot."""

    def test_gateway_builtin_adapters_have_fibers(self, tmp_path) -> None:
        from leapflow.gateway.server import GatewayServer

        server = GatewayServer(tmp_path)
        registry = server.adapter_registry
        scoped = server.scoped_adapter_registry

        platform_ids = set(registry.list_available())
        fiber_ids = set(scoped.fibers.keys())

        assert platform_ids, "expected at least one built-in gateway adapter"
        missing = platform_ids - fiber_ids
        assert not missing, f"gateway adapters without fibers: {missing}"

        for pid, fiber in scoped.fibers.items():
            assert fiber.state == FiberState.ACTIVE, (
                f"gateway adapter '{pid}' fiber not ACTIVE: {fiber.state}"
            )

    def test_gateway_adopt_is_idempotent(self, tmp_path) -> None:
        from leapflow.gateway.server import GatewayServer

        server = GatewayServer(tmp_path)
        registry = server.adapter_registry
        scoped = server.scoped_adapter_registry

        plugin_count_before = len(registry.list_available())
        fiber_count_before = len(scoped.fibers)

        scoped.adopt_existing_plugins()

        assert len(registry.list_available()) == plugin_count_before
        assert len(scoped.fibers) == fiber_count_before


# ════════════════════════════════════════════════════════════════
# LLM subsystem
# ════════════════════════════════════════════════════════════════


class TestLLMProviderFiberization:
    """Every built-in LLM provider gets an ACTIVE fiber after boot."""

    @pytest.fixture(autouse=True)
    def _reset_llm_registry(self):
        from leapflow.llm.provider_registry import reset_default_registry

        reset_default_registry()
        yield
        reset_default_registry()

    def test_llm_builtin_providers_have_fibers(self) -> None:
        from leapflow.llm.provider_registry import (
            get_default_registry,
            get_scoped_default_registry,
        )

        scoped = get_scoped_default_registry()
        registry = get_default_registry()

        provider_ids = set(registry.list_available())
        fiber_ids = set(scoped.fibers.keys())

        assert provider_ids, "expected at least one built-in LLM provider"
        missing = provider_ids - fiber_ids
        assert not missing, f"LLM providers without fibers: {missing}"

        for pid, fiber in scoped.fibers.items():
            assert fiber.state == FiberState.ACTIVE, (
                f"LLM provider '{pid}' fiber not ACTIVE: {fiber.state}"
            )

    def test_llm_adopt_is_idempotent(self) -> None:
        from leapflow.llm.provider_registry import (
            get_default_registry,
            get_scoped_default_registry,
        )

        scoped = get_scoped_default_registry()
        registry = get_default_registry()

        provider_count_before = len(registry.list_available())
        fiber_count_before = len(scoped.fibers)

        # get_scoped_default_registry() adopts on every call; a second call
        # must not double-register or add fibers.
        get_scoped_default_registry()

        assert len(registry.list_available()) == provider_count_before
        assert len(scoped.fibers) == fiber_count_before
