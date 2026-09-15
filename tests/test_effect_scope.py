# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for EffectScope and PluginFiber domain primitives."""

from __future__ import annotations

import asyncio

import pytest

from leapflow.domain.effect_scope import EffectScope, ScopeState
from leapflow.domain.plugin_fiber import (
    FiberState,
    IllegalStateTransition,
    PluginFiber,
)


# ════════════════════════════════════════════════════════════════
# EffectScope tests
# ════════════════════════════════════════════════════════════════


class TestEffectScopeLIFO:
    """Effects execute in reverse registration order (LIFO)."""

    def test_effect_scope_lifo_dispose_order(self) -> None:
        scope = EffectScope("lifo-test")
        order: list[int] = []
        scope.effect(lambda: order.append(1))
        scope.effect(lambda: order.append(2))
        scope.effect(lambda: order.append(3))
        scope.dispose()
        assert order == [3, 2, 1], "Effects must run in reverse registration order"


class TestEffectScopeIdempotent:
    """Calling dispose() multiple times is safe."""

    def test_effect_scope_idempotent_dispose(self) -> None:
        scope = EffectScope("idempotent-test")
        call_count = [0]
        scope.effect(lambda: call_count.__setitem__(0, call_count[0] + 1))
        scope.dispose()
        scope.dispose()  # second call should be no-op
        assert call_count[0] == 1, "Effects must only run once even with multiple dispose() calls"
        assert scope.state == ScopeState.DISPOSED


class TestEffectScopeExceptionSafe:
    """One failing cleanup must not block others."""

    def test_effect_scope_exception_safe_cleanup(self) -> None:
        scope = EffectScope("exc-safe")
        executed: list[str] = []

        scope.effect(lambda: executed.append("first"))
        scope.effect(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        scope.effect(lambda: executed.append("third"))

        # The middle effect raises, but both first and third should still run.
        # Effects run in reverse: third → boom → first
        scope.dispose()
        assert "first" in executed, "First effect must run even if a later one raised"
        assert "third" in executed, "Third effect must run even if a later one raised"
        assert scope.state == ScopeState.DISPOSED


class TestEffectScopeChildCascade:
    """Disposing parent cascades to children."""

    def test_effect_scope_child_cascade(self) -> None:
        parent = EffectScope("parent")
        child = parent.child("child")
        child_disposed = [False]
        child.effect(lambda: child_disposed.__setitem__(0, True))
        parent.dispose()
        assert child_disposed[0], "Child effects must run when parent is disposed"
        assert child.state == ScopeState.DISPOSED
        assert parent.state == ScopeState.DISPOSED

    def test_effect_scope_nested_children_order(self) -> None:
        """Multi-level hierarchy disposes in correct order (deepest first, LIFO)."""
        order: list[str] = []
        root = EffectScope("root")
        root.effect(lambda: order.append("root"))

        child_a = root.child("child-a")
        child_a.effect(lambda: order.append("child-a"))

        child_b = root.child("child-b")
        child_b.effect(lambda: order.append("child-b"))

        grandchild = child_b.child("grandchild")
        grandchild.effect(lambda: order.append("grandchild"))

        root.dispose()
        # Children reverse: child_b (with grandchild), then child_a, then root effects
        assert order.index("grandchild") < order.index("child-b")
        assert order.index("child-b") < order.index("child-a")
        assert order.index("child-a") < order.index("root")


class TestEffectScopeAfterDispose:
    """Registering effects/children on a disposed scope raises."""

    def test_effect_scope_register_after_dispose_raises(self) -> None:
        scope = EffectScope("closed")
        scope.dispose()
        with pytest.raises(RuntimeError, match="disposed"):
            scope.effect(lambda: None)

    def test_effect_scope_child_after_dispose_raises(self) -> None:
        scope = EffectScope("closed")
        scope.dispose()
        with pytest.raises(RuntimeError, match="disposed"):
            scope.child("should-fail")


class TestEffectScopeContextManager:
    """Context manager protocol disposes on exit."""

    def test_effect_scope_context_manager(self) -> None:
        executed = [False]
        with EffectScope("ctx") as scope:
            scope.effect(lambda: executed.__setitem__(0, True))
            assert scope.is_active
        assert executed[0]
        assert scope.is_disposed


class TestEffectScopeStateTransitions:
    """State transitions: ACTIVE → DISPOSING → DISPOSED."""

    def test_effect_scope_state_transitions(self) -> None:
        states_seen: list[ScopeState] = []
        scope = EffectScope("transitions")
        states_seen.append(scope.state)

        # Record state during cleanup (should be DISPOSING)
        scope.effect(lambda: states_seen.append(scope.state))
        scope.dispose()
        states_seen.append(scope.state)

        assert states_seen == [
            ScopeState.ACTIVE,
            ScopeState.DISPOSING,
            ScopeState.DISPOSED,
        ]


class TestEffectScopeDiagnostics:
    """Diagnostics properties (effect_count, child_count)."""

    def test_effect_scope_diagnostics(self) -> None:
        scope = EffectScope("diag")
        assert scope.effect_count == 0
        assert scope.child_count == 0

        scope.effect(lambda: None)
        scope.effect(lambda: None)
        assert scope.effect_count == 2

        scope.child("a")
        scope.child("b")
        scope.child("c")
        assert scope.child_count == 3


# ════════════════════════════════════════════════════════════════
# PluginFiber tests
# ════════════════════════════════════════════════════════════════


class TestFiberLifecycle:
    """Valid lifecycle: PENDING → ACTIVE → UNLOADING → DISPOSED."""

    def test_fiber_valid_lifecycle(self) -> None:
        scope = EffectScope("fiber-test")
        fiber = PluginFiber(plugin_id="test-plugin", scope=scope)
        assert fiber.state == FiberState.PENDING

        fiber.activate()
        assert fiber.state == FiberState.ACTIVE
        assert fiber.is_active

        fiber.begin_unload()
        assert fiber.state == FiberState.UNLOADING

        fiber.dispose()
        assert fiber.state == FiberState.DISPOSED
        assert fiber.is_disposed
        assert scope.is_disposed

    def test_fiber_dispose_triggers_scope_dispose(self) -> None:
        scope = EffectScope("fiber-scope")
        cleanup_ran = [False]
        scope.effect(lambda: cleanup_ran.__setitem__(0, True))

        fiber = PluginFiber(plugin_id="test", scope=scope)
        fiber.activate()
        fiber.begin_unload()
        fiber.dispose()

        assert cleanup_ran[0], "Fiber disposal must trigger scope disposal"
        assert scope.is_disposed


class TestFiberIllegalTransitions:
    """Invalid state transitions raise IllegalStateTransition."""

    def test_fiber_pending_to_disposed_allowed(self) -> None:
        scope = EffectScope("direct-dispose")
        fiber = PluginFiber(plugin_id="test", scope=scope)
        fiber.dispose()  # PENDING → DISPOSED is now valid
        assert fiber.state == FiberState.DISPOSED
        assert scope.is_disposed

    def test_fiber_illegal_transition_active_to_disposed(self) -> None:
        scope = EffectScope("bad-transition-2")
        fiber = PluginFiber(plugin_id="test", scope=scope)
        fiber.activate()
        with pytest.raises(IllegalStateTransition):
            fiber.dispose()  # ACTIVE → DISPOSED is invalid (must go through UNLOADING)

    def test_fiber_illegal_reactivate(self) -> None:
        scope = EffectScope("reactivate")
        fiber = PluginFiber(plugin_id="test", scope=scope)
        fiber.activate()
        fiber.begin_unload()
        fiber.dispose()
        with pytest.raises(IllegalStateTransition):
            fiber.activate()  # DISPOSED → ACTIVE is invalid


class TestFiberActivation:
    """Basic activation from PENDING."""

    def test_fiber_activate_from_pending(self) -> None:
        scope = EffectScope("activate")
        fiber = PluginFiber(plugin_id="test", scope=scope)
        assert not fiber.is_active
        fiber.activate()
        assert fiber.is_active
        assert not fiber.is_disposed


class TestFiberProperties:
    """Property correctness across lifecycle."""

    def test_fiber_is_active_is_disposed_properties(self) -> None:
        scope = EffectScope("props")
        fiber = PluginFiber(plugin_id="test", scope=scope)

        assert not fiber.is_active
        assert not fiber.is_disposed

        fiber.activate()
        assert fiber.is_active
        assert not fiber.is_disposed

        fiber.begin_unload()
        assert not fiber.is_active
        assert not fiber.is_disposed

        fiber.dispose()
        assert not fiber.is_active
        assert fiber.is_disposed


class TestFiberScopeEffects:
    """Effects registered on fiber's scope run on dispose."""

    def test_fiber_dispose_runs_registered_effects(self) -> None:
        scope = EffectScope("fiber-effects")
        fiber = PluginFiber(plugin_id="test", scope=scope)

        effects_log: list[str] = []
        scope.effect(lambda: effects_log.append("effect-1"))
        scope.effect(lambda: effects_log.append("effect-2"))

        fiber.activate()
        fiber.begin_unload()
        fiber.dispose()

        assert effects_log == ["effect-2", "effect-1"], "Effects must run LIFO on fiber dispose"


# ════════════════════════════════════════════════════════════════
# Extended FiberState (LOADING/FAILED) tests
# ════════════════════════════════════════════════════════════════


class TestFiberLoadingState:
    """Tests for the new LOADING/FAILED fiber states."""

    def test_pending_to_loading_to_active(self) -> None:
        """Standard async-init path: PENDING → LOADING → ACTIVE."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        assert fiber.state == FiberState.PENDING
        fiber.begin_loading()
        assert fiber.state == FiberState.LOADING
        assert fiber.is_loading
        fiber.activate()
        assert fiber.state == FiberState.ACTIVE
        assert fiber.is_active

    def test_loading_to_failed(self) -> None:
        """Init failure: LOADING → FAILED with error stored."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.begin_loading()
        err = RuntimeError("init failed")
        fiber.fail(err)
        assert fiber.state == FiberState.FAILED
        assert fiber.is_failed
        assert fiber.error is err

    def test_failed_to_loading_retry(self) -> None:
        """Retry from FAILED: FAILED → LOADING clears error."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.begin_loading()
        fiber.fail(RuntimeError("oops"))
        fiber.retry()
        assert fiber.state == FiberState.LOADING
        assert fiber.error is None

    def test_loading_to_disposed(self) -> None:
        """Abort during loading: LOADING → DISPOSED via scope dispose."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.begin_loading()
        fiber.dispose()
        assert fiber.state == FiberState.DISPOSED

    def test_failed_to_disposed(self) -> None:
        """Give up after failure: FAILED → DISPOSED."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.begin_loading()
        fiber.fail(RuntimeError("fatal"))
        fiber.dispose()
        assert fiber.state == FiberState.DISPOSED
        assert fiber.error is None  # cleared on dispose

    def test_illegal_transitions_from_loading(self) -> None:
        """LOADING cannot go to UNLOADING or PENDING."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.begin_loading()
        with pytest.raises(IllegalStateTransition):
            fiber.begin_unload()

    def test_illegal_transition_from_failed(self) -> None:
        """FAILED cannot go to ACTIVE directly (must retry through LOADING)."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.begin_loading()
        fiber.fail(RuntimeError("x"))
        with pytest.raises(IllegalStateTransition):
            fiber.activate()


class TestFiberDisposalPaths:
    """Tests for disposal from various states."""

    def test_pending_to_disposed(self) -> None:
        """Fiber that never started can be disposed directly."""
        scope = EffectScope("test")
        fiber = PluginFiber("test_plugin", scope)
        fiber.dispose()
        assert fiber.state == FiberState.DISPOSED
        assert fiber.is_disposed


# ════════════════════════════════════════════════════════════════
# Scope-bound EventBus subscription tests
# ════════════════════════════════════════════════════════════════


class TestScopeBoundSubscription:
    """Tests for EventBus scope-bound auto-cleanup."""

    def test_subscribe_with_scope_auto_unsubscribes_on_dispose(self) -> None:
        """Subscription bound to a scope is removed when scope disposes."""
        from leapflow.platform.event_bus import EventBus
        bus = EventBus.__new__(EventBus)
        bus._subscribers = {}

        scope = EffectScope("sub_scope")
        called: list = []
        cb = lambda event: called.append(event)

        bus.subscribe(cb, scope=scope)
        assert id(cb) in bus._subscribers

        scope.dispose()
        assert id(cb) not in bus._subscribers

    def test_subscribe_without_scope_survives_unrelated_dispose(self) -> None:
        """Subscription without scope is not affected by scope disposal."""
        from leapflow.platform.event_bus import EventBus
        bus = EventBus.__new__(EventBus)
        bus._subscribers = {}

        scope = EffectScope("unrelated")
        cb = lambda event: None

        bus.subscribe(cb)  # no scope
        scope.dispose()
        assert id(cb) in bus._subscribers

    def test_multiple_scope_bound_subs_cleaned_together(self) -> None:
        """Multiple subscriptions on one scope all cleaned on dispose."""
        from leapflow.platform.event_bus import EventBus
        bus = EventBus.__new__(EventBus)
        bus._subscribers = {}

        scope = EffectScope("shared")
        cb1 = lambda e: None
        cb2 = lambda e: None
        cb3 = lambda e: None  # unbound

        bus.subscribe(cb1, scope=scope)
        bus.subscribe(cb2, scope=scope)
        bus.subscribe(cb3)  # no scope

        assert len(bus._subscribers) == 3
        scope.dispose()
        assert len(bus._subscribers) == 1
        assert id(cb3) in bus._subscribers


# ════════════════════════════════════════════════════════════════
# Async EffectScope tests
# ════════════════════════════════════════════════════════════════


class TestAsyncEffectRegistration:
    """async_effect() registration behavior."""

    def test_async_effect_registers(self) -> None:
        scope = EffectScope("async-reg")
        scope.async_effect(self._noop_async)
        assert scope.async_effect_count == 1

    def test_async_effect_on_disposed_scope_raises(self) -> None:
        scope = EffectScope("closed")
        scope.dispose()
        with pytest.raises(RuntimeError, match="disposed"):
            scope.async_effect(self._noop_async)

    def test_async_effect_count_separate_from_sync(self) -> None:
        scope = EffectScope("mixed")
        scope.effect(lambda: None)
        scope.async_effect(self._noop_async)
        scope.async_effect(self._noop_async)
        assert scope.effect_count == 1
        assert scope.async_effect_count == 2

    @staticmethod
    async def _noop_async() -> None:
        pass


class TestAsyncDispose:
    """async_dispose() awaits async effects + calls sync effects."""

    @pytest.mark.asyncio
    async def test_async_dispose_runs_async_effects_lifo(self) -> None:
        scope = EffectScope("async-lifo")
        order: list[int] = []

        async def append(n: int) -> None:
            order.append(n)

        scope.async_effect(lambda: append(1))
        scope.async_effect(lambda: append(2))
        scope.async_effect(lambda: append(3))

        await scope.async_dispose()
        assert order == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_async_dispose_runs_sync_effects_after_async(self) -> None:
        scope = EffectScope("mixed-order")
        order: list[str] = []

        async def async_cleanup() -> None:
            order.append("async")

        scope.effect(lambda: order.append("sync"))
        scope.async_effect(async_cleanup)

        await scope.async_dispose()
        # Async effects run before sync effects
        assert order == ["async", "sync"]

    @pytest.mark.asyncio
    async def test_async_dispose_is_idempotent(self) -> None:
        scope = EffectScope("idem")
        count = [0]

        async def inc() -> None:
            count[0] += 1

        scope.async_effect(inc)
        await scope.async_dispose()
        await scope.async_dispose()  # second call is no-op
        assert count[0] == 1
        assert scope.state == ScopeState.DISPOSED

    @pytest.mark.asyncio
    async def test_async_dispose_exception_safe(self) -> None:
        scope = EffectScope("exc-safe")
        executed: list[str] = []

        async def good_first() -> None:
            executed.append("first")

        async def bad() -> None:
            raise RuntimeError("boom")

        async def good_last() -> None:
            executed.append("last")

        scope.async_effect(good_first)
        scope.async_effect(bad)
        scope.async_effect(good_last)

        await scope.async_dispose()
        # LIFO: good_last, bad (fails), good_first
        assert "first" in executed
        assert "last" in executed
        assert scope.state == ScopeState.DISPOSED

    @pytest.mark.asyncio
    async def test_async_dispose_cascades_to_children(self) -> None:
        parent = EffectScope("parent")
        child = parent.child("child")
        order: list[str] = []

        async def parent_cleanup() -> None:
            order.append("parent")

        async def child_cleanup() -> None:
            order.append("child")

        parent.async_effect(parent_cleanup)
        child.async_effect(child_cleanup)

        await parent.async_dispose()
        # Child disposes before parent's own effects
        assert order.index("child") < order.index("parent")
        assert child.state == ScopeState.DISPOSED
        assert parent.state == ScopeState.DISPOSED

    @pytest.mark.asyncio
    async def test_async_dispose_nested_hierarchy(self) -> None:
        """Multi-level async hierarchy disposes deepest first."""
        root = EffectScope("root")
        child = root.child("child")
        grandchild = child.child("grandchild")
        order: list[str] = []

        async def mark(name: str) -> None:
            order.append(name)

        root.async_effect(lambda: mark("root"))
        child.async_effect(lambda: mark("child"))
        grandchild.async_effect(lambda: mark("grandchild"))

        await root.async_dispose()
        assert order.index("grandchild") < order.index("child")
        assert order.index("child") < order.index("root")


class TestSyncDisposeWithAsyncEffects:
    """Sync dispose() gracefully handles async effects."""

    def test_sync_dispose_runs_async_via_asyncio_run(self) -> None:
        """When no event loop is running, asyncio.run() is used."""
        scope = EffectScope("sync-async")
        executed = [False]

        async def async_cleanup() -> None:
            executed[0] = True

        scope.async_effect(async_cleanup)
        scope.dispose()
        assert executed[0]
        assert scope.state == ScopeState.DISPOSED

    def test_sync_dispose_with_failing_async_still_completes(self) -> None:
        """Failing async effects don't block sync dispose."""
        scope = EffectScope("fail-async")
        sync_ran = [False]

        async def bad_async() -> None:
            raise RuntimeError("async boom")

        scope.async_effect(bad_async)
        scope.effect(lambda: sync_ran.__setitem__(0, True))
        scope.dispose()
        assert sync_ran[0]
        assert scope.state == ScopeState.DISPOSED

    @pytest.mark.asyncio
    async def test_sync_dispose_inside_running_loop_schedules_tasks(self) -> None:
        """When called from within a running loop, async effects become tasks."""
        scope = EffectScope("in-loop")
        executed = [False]

        async def async_cleanup() -> None:
            executed[0] = True

        scope.async_effect(async_cleanup)
        # We're inside an async test, so a loop IS running
        scope.dispose()
        # Give scheduled task time to run
        await asyncio.sleep(0.05)
        assert executed[0]
        assert scope.state == ScopeState.DISPOSED
