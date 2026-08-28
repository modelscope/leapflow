"""Executable guards for the architecture contracts in AGENTS.md.

These contracts are the ones a code review is worst at catching, because a
violation looks locally reasonable: one vendor import in a core module, one
mutable domain type, one event-subscription method on a one-shot backend. Each
test below therefore asserts the *boundary* rather than any single call site,
so the guard keeps holding as the implementation moves.

Covered contracts:
- Platform-Neutral Gateway Core (core must not import platform packages)
- Platform vs App Business Boundary (no vendor endpoints/error shapes in core)
- Plugin core vs tool implementations (plugin core must not import a tool module)
- Transport-Lifecycle Separation (one-shot actions vs long-lived observations)
- Immutable Domain Types (frozen dataclasses for domain objects)
- Protocol over ABC (extension points are runtime_checkable Protocols)
- Standalone importability (no import-time side effects)
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib
import re
import typing

import pytest

GATEWAY_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "leapflow" / "gateway"
PLUGINS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "leapflow" / "plugins"
HARDWARE_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "leapflow" / "hardware"

# Sub-packages that own platform/vendor specifics. Gateway core may define the
# contracts these implement, but must never depend on them.
_PLATFORM_PACKAGES = ("adapters", "normalizers", "action_packs", "backends", "manifests")


def _core_modules() -> list[pathlib.Path]:
    """Return gateway core modules (top-level files, excluding sub-packages)."""
    return sorted(p for p in GATEWAY_DIR.glob("*.py") if p.name != "__init__.py")


def _imported_modules(path: pathlib.Path) -> list[tuple[str, int]]:
    """Return (module, lineno) for every import in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.lineno))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
    return found


def _plugin_core_modules() -> list[pathlib.Path]:
    """Return plugin core modules (contracts, registry, lifecycle).

    ``tool_plugins/`` is excluded on purpose: those modules exist to wrap tool
    implementations, so they are the one place allowed to import them.
    """
    return sorted(p for p in PLUGINS_DIR.glob("*.py"))


# ── Platform-Neutral Gateway Core ────────────────────────────────────────


def test_gateway_core_does_not_import_platform_packages() -> None:
    """Core owns protocols, lifecycle, routing, approval, audit — not vendors.

    A core module importing an adapter/normalizer/action pack inverts the
    dependency and makes every new platform a core change.
    """
    violations: list[str] = []
    for path in _core_modules():
        for module, lineno in _imported_modules(path):
            for package in _PLATFORM_PACKAGES:
                if f"gateway.{package}" in module:
                    violations.append(f"{path.name}:{lineno} imports {module}")

    assert violations == [], (
        "gateway core must not depend on platform packages; move the "
        "platform-specific part behind a protocol or into the adapter:\n  "
        + "\n  ".join(violations)
    )


def test_gateway_core_does_not_import_vendor_sdks() -> None:
    """Vendor SDKs belong to adapters/backends, never to core modules."""
    vendor_sdk_roots = ("lark_oapi", "telebot", "telegram", "slack_sdk", "dingtalk")
    violations: list[str] = []
    for path in _core_modules():
        for module, lineno in _imported_modules(path):
            root = module.split(".")[0]
            if root in vendor_sdk_roots:
                violations.append(f"{path.name}:{lineno} imports {module}")

    assert violations == [], "vendor SDK imported by gateway core:\n  " + "\n  ".join(violations)


def test_gateway_core_has_no_vendor_endpoints_or_error_shapes() -> None:
    """Vendor wire formats must live in the app's pack/adapter, not in core.

    Vendor validators now sit in ``gateway/validators/<platform>.py``, mirroring
    ``adapters/`` and ``normalizers/``; core keeps only the neutral registry.
    """
    vendor_endpoint = re.compile(
        r"https?://[^\s\"']*(feishu|larksuite|dingtalk|telegram|slack)", re.IGNORECASE
    )
    violations: list[str] = []
    for path in _core_modules():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if vendor_endpoint.search(line):
                violations.append(f"{path.name}:{lineno}")

    assert violations == [], "vendor endpoint hardcoded in gateway core: " + ", ".join(violations)


# ── Hardware Context Protocol boundaries ─────────────────────────────────


def _hardware_modules() -> list[pathlib.Path]:
    """Return every module in the hardware package, including its seams."""
    return sorted(HARDWARE_DIR.rglob("*.py"))


def test_hardware_domain_model_is_free_of_upstream_standard_names() -> None:
    """The Hardware Context Protocol's one architectural red line.

    ``context.py`` describes what an agent must know to operate a device safely --
    facts fixed by physics and by governance, not by whichever southbound standard
    eventually carries the command. The moment a guessed upstream concept leaks into
    the domain model, the model expires when that standard is published, and the
    two-file integration promise is gone with it.

    Upstream names belong in ``providers/`` and ``transports/``, which is where a
    mapping is allowed to be wrong.
    """
    upstream_names = re.compile(r"\bmhs\b|model_hardware_standard", re.IGNORECASE)
    domain_modules = (HARDWARE_DIR / "context.py", HARDWARE_DIR / "transport.py")
    violations: list[str] = []
    for path in domain_modules:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if upstream_names.search(line):
                violations.append(f"{path.name}:{lineno}: {line.strip()}")

    assert violations == [], (
        "an upstream standard's name leaked into the hardware domain model; keep it "
        "in providers/ or transports/:\n  " + "\n  ".join(violations)
    )


def test_hardware_does_not_import_the_engine() -> None:
    """Dependency runs engine -> hardware, never back.

    ``WriteOutcome.side_effect_state`` mirrors ``SideEffectState`` as plain strings
    for exactly this reason: importing the enum would create a cycle and make the
    domain model unimportable on its own.
    """
    violations: list[str] = []
    for path in _hardware_modules():
        for module, lineno in _imported_modules(path):
            if module.startswith("leapflow.engine"):
                violations.append(f"{path.relative_to(HARDWARE_DIR)}:{lineno} imports {module}")

    assert violations == [], (
        "leapflow.hardware must not depend on leapflow.engine:\n  " + "\n  ".join(violations)
    )


def test_hardware_domain_model_does_not_import_its_own_seams() -> None:
    """The domain model must not know which providers or transports exist.

    If ``context.py`` reached for the transport table, adding a transport would
    become a change to the stable half of the protocol -- the exact coupling the
    split exists to prevent.
    """
    violations: list[str] = []
    for module, lineno in _imported_modules(HARDWARE_DIR / "context.py"):
        if "hardware.providers" in module or "hardware.transports" in module:
            violations.append(f"context.py:{lineno} imports {module}")

    assert violations == [], (
        "the hardware domain model must not import its seams:\n  " + "\n  ".join(violations)
    )


def test_hardware_transports_are_not_named_after_one_device() -> None:
    """Transports are generic mechanisms; device specifics live in declarations.

    A transport named after a particular instrument or board is a sign that device
    knowledge has moved into code, where it can no longer be reviewed or overridden
    per bench.
    """
    allowed = {"__init__.py", "mock.py", "python_callable.py"}
    present = {p.name for p in (HARDWARE_DIR / "transports").glob("*.py")}
    unexpected = present - allowed
    assert not unexpected, (
        f"unexpected transport modules {sorted(unexpected)}; a transport must be a generic "
        "mechanism, and a new one also needs a case in tests/test_hardware_transport_contract.py"
    )


# ── Plugin core vs tool implementations ──────────────────────────────────


def test_plugin_core_does_not_import_tool_implementations() -> None:
    """Plugin core owns contracts, discovery, and lifecycle — never behaviour.

    ``leapflow.plugins`` publishes whatever a plugin declares; the moment core
    reaches into ``leapflow.tools`` the dependency inverts and every new tool
    becomes a core change. Tool wrappers live in ``tool_plugins/``, which is
    exactly where that import belongs.
    """
    violations: list[str] = []
    for path in _plugin_core_modules():
        for module, lineno in _imported_modules(path):
            if module.startswith("leapflow.tools"):
                violations.append(f"{path.name}:{lineno} imports {module}")

    assert violations == [], (
        "plugin core must not depend on tool implementations; declare the tool "
        "in a tool_plugins/ module or inject it through bind_runtime():\n  "
        + "\n  ".join(violations)
    )


# ── Plugin subsystem lives under leapflow.plugins, not leapflow.tools ─────

# The plugin subsystem (contracts, registry, scoped lifecycle, marketplace,
# sandbox) was relocated to ``leapflow.plugins``. The legacy ``leapflow.tools``
# locations must stay gone: a re-created shim would silently split the single
# source of truth for the registry and let two divergent registries coexist.
_RELOCATED_TOOL_MODULES = (
    "leapflow.tools.plugins",
    "leapflow.tools.protocol",
    "leapflow.tools.plugin_registry",
    "leapflow.tools.scoped_registry",
    "leapflow.tools.marketplace",
    "leapflow.tools.sandbox",
)

_TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "leapflow" / "tools"
_RELOCATED_TOOL_PATHS = (
    _TOOLS_DIR / "plugins",
    _TOOLS_DIR / "protocol.py",
    _TOOLS_DIR / "plugin_registry.py",
    _TOOLS_DIR / "scoped_registry.py",
    _TOOLS_DIR / "marketplace",
    _TOOLS_DIR / "sandbox",
)


def test_legacy_tool_plugin_paths_are_physically_removed() -> None:
    """The pre-relocation plugin source locations must not exist on disk."""
    present = [str(p) for p in _RELOCATED_TOOL_PATHS if p.exists()]
    assert present == [], (
        "legacy plugin-subsystem source paths were re-created; the subsystem "
        "lives under src/leapflow/plugins/ and these must stay absent:\n  "
        + "\n  ".join(present)
    )


@pytest.mark.parametrize("module_name", _RELOCATED_TOOL_MODULES)
def test_legacy_tool_plugin_modules_are_not_importable(module_name: str) -> None:
    """Importing a relocated module must fail rather than resolve to a shim."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


# ── Transport-Lifecycle Separation ───────────────────────────────────────


def test_one_shot_action_backend_exposes_no_event_subscription() -> None:
    """``ExecutionBackend`` runs bounded actions; it must not stream events.

    Merging the two lets a streaming subscriber, webhook, or polling loop be
    implemented inside one-shot action execution, which is the exact coupling
    the contract forbids.
    """
    from leapflow.gateway.connectors.protocol import ExecutionBackend

    members = {name for name in dir(ExecutionBackend) if not name.startswith("_")}
    assert "execute" in members, "ExecutionBackend must run actions"
    assert members.isdisjoint({"events", "start", "stop"}), (
        "ExecutionBackend must not own long-lived observation methods; those "
        f"belong to BackendEventSource. Found: {sorted(members)}"
    )


def test_long_lived_event_source_exposes_no_action_execution() -> None:
    """``BackendEventSource`` observes; it must not execute actions."""
    from leapflow.gateway.connectors.protocol import BackendEventSource

    members = {name for name in dir(BackendEventSource) if not name.startswith("_")}
    assert {"events", "start", "stop"} <= members, "event source must own its lifecycle"
    assert members.isdisjoint({"execute", "preview"}), (
        f"BackendEventSource must not execute actions. Found: {sorted(members)}"
    )


# ── Immutable Domain Types ───────────────────────────────────────────────


_DOMAIN_TYPES = [
    ("leapflow.domain.capability_requirement", "CapabilityRequirement"),
    ("leapflow.domain.environment_fingerprint", "EnvironmentFingerprint"),
    ("leapflow.gateway.protocol", "InboundMessage"),
    ("leapflow.gateway.protocol", "OutboundContent"),
    ("leapflow.gateway.protocol", "SendTarget"),
    ("leapflow.gateway.protocol", "SendResult"),
    ("leapflow.gateway.protocol", "MessageSource"),
    ("leapflow.gateway.connectors.protocol", "ActionSpec"),
    ("leapflow.gateway.connectors.protocol", "ActionResult"),
    ("leapflow.gateway.connectors.protocol", "ActionFailure"),
    ("leapflow.engine.failure_envelope", "FailureEnvelope"),
    ("leapflow.engine.failure_envelope", "FailureContext"),
    ("leapflow.engine.failure_envelope", "RecoveryHint"),
    ("leapflow.engine.recovery_decision", "RecoveryDecision"),
    ("leapflow.engine.recovery_decision", "BackoffConfig"),
    ("leapflow.engine.recovery_decision", "RetrySemantics"),
    ("leapflow.monitor.types", "Finding"),
    ("leapflow.monitor.types", "WatchSpec"),
]


@pytest.mark.parametrize(("module_name", "type_name"), _DOMAIN_TYPES)
def test_domain_types_are_frozen(module_name: str, type_name: str) -> None:
    """Domain objects crossing module boundaries must be immutable.

    These types are passed between engine, gateway, and storage; a mutable one
    lets a downstream consumer edit shared state instead of deriving a new value.
    """
    cls = getattr(importlib.import_module(module_name), type_name)

    assert dataclasses.is_dataclass(cls), f"{type_name} must be a dataclass"
    assert cls.__dataclass_params__.frozen, (
        f"{type_name} is a shared domain type and must be frozen=True"
    )


def test_frozen_domain_type_rejects_mutation_at_runtime() -> None:
    """The frozen flag must actually block writes (not just be declared)."""
    from leapflow.engine.failure_envelope import FailureEnvelope, FailureSource, Recoverability

    envelope = FailureEnvelope.create(
        source=FailureSource.TOOL,
        category="tool_timeout",
        failure_class="transient",
        failure_code="timeout",
        message="timed out",
        recoverability=Recoverability.AUTO_RETRY,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope.message = "rewritten"  # type: ignore[misc]


# ── Protocol over ABC ────────────────────────────────────────────────────


_EXTENSION_POINTS = [
    ("leapflow.gateway.connectors.protocol", "ExecutionBackend"),
    ("leapflow.gateway.connectors.protocol", "BackendEventSource"),
    ("leapflow.engine.recovery_coordinator", "RecoveryStrategy"),
    ("leapflow.monitor.types", "MonitorProducer"),
    ("leapflow.dashboard.service", "DashboardDataProvider"),
]


@pytest.mark.parametrize(("module_name", "type_name"), _EXTENSION_POINTS)
def test_extension_points_are_runtime_checkable_protocols(
    module_name: str, type_name: str,
) -> None:
    """Extension points must be Protocols so implementations stay decoupled.

    ``runtime_checkable`` is part of the contract: registration and test code
    verify conformance with ``isinstance`` rather than by subclassing. Missing
    names fail rather than skip — a renamed extension point must be noticed.
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, type_name, None)

    assert cls is not None, f"{module_name} must export the {type_name} extension point"
    assert issubclass(cls, typing.Protocol), f"{type_name} must be a typing.Protocol"  # type: ignore[arg-type]
    assert getattr(cls, "_is_runtime_protocol", False), (
        f"{type_name} must be decorated with @runtime_checkable"
    )


# ── Standalone importability ─────────────────────────────────────────────


_STANDALONE_MODULES = [
    "leapflow.domain.capability_requirement",
    "leapflow.domain.environment_fingerprint",
    "leapflow.logging_setup",
    "leapflow.layout",
    "leapflow.config_service",
    "leapflow.gateway.trigger_policy",
    "leapflow.gateway.session_router",
    "leapflow.gateway.validators",
    "leapflow.engine.recovery_coordinator",
    "leapflow.engine.recovery_strategies",
    "leapflow.engine.failure_envelope",
    "leapflow.monitor.types",
    "leapflow.monitor.session_producer",
    "leapflow.dashboard.service",
    "leapflow.daemon.session_registry",
    "leapflow.daemon.notifications",
    "leapflow.plugins",
    "leapflow.plugins.capability_plan",
    "leapflow.plugins.capability_resolver",
    "leapflow.plugins.protocol",
    "leapflow.plugins.registry",
    "leapflow.plugins.scoped_registry",
    "leapflow.plugins.tool_plugins",
    "leapflow.plugins.marketplace",
    "leapflow.plugins.sandbox",
]


@pytest.mark.parametrize("module_name", _STANDALONE_MODULES)
def test_module_imports_standalone(module_name: str) -> None:
    """Every module must import without side effects or optional deps.

    Guards the graceful-degradation contract at import level: a module that
    needs aiohttp/duckdb/an LLM at import time breaks unrelated entry points.
    """
    assert importlib.import_module(module_name) is not None


def test_engine_self_attributes_all_exist() -> None:
    """Every ``self._x`` the engine reads must actually be defined somewhere.

    A mistyped attribute name is invisible until the line runs, and the agent
    loop wraps most of its work in broad ``except Exception`` handlers, so such a
    typo surfaces as a misclassified recovery failure rather than a crash. One
    of them (``_context_window_controller``, never assigned anywhere) made every
    turn halt while the suite stayed green.

    Names assigned anywhere in the module count as defined, including on frames
    and per-session clones; this catches the typo case, not lifecycle ordering.
    """
    import re
    from pathlib import Path

    import leapflow.engine.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    read = set(re.findall(r"self\.(_[a-z][a-z0-9_]*)", source))
    assigned = set(re.findall(r"self\.(_[a-z][a-z0-9_]*)\s*(?::[^=\n]+)?=", source))
    # Attributes may also be set from outside (session_factory clones engines).
    for module in ("leapflow.engine.session_factory", "leapflow.engine.agent_loop"):
        mod = importlib.import_module(module)
        assigned |= set(
            re.findall(r"engine\.(_[a-z][a-z0-9_]*)\s*=", Path(mod.__file__).read_text(encoding="utf-8"))
        )
    on_class = {name for name in read if hasattr(engine_module.AgentEngine, name)}

    undefined = sorted(read - assigned - on_class)
    assert not undefined, f"engine reads attributes that are never assigned: {undefined}"
