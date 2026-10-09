# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native YAML hardware onboarding benchmark."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import unseen_success_rate
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import ScriptedApprovalGate, evidence_root, make_registry
from leapflow.hardware.capability_router import CapabilityRouterPlugin
from leapflow.hardware.context import TransportRef
from leapflow.hardware.control_binding_resolver import ControlBindingResolver

_ADAPTER_ID = "native_yaml_onboarding"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("yaml_context_load", "Load example YAML HardwareContext declarations",
             adapter_id=_ADAPTER_ID, tags=("native", "onboarding", "tier1")),
    Scenario("control_binding_resolution", "Resolve YAML-derived control binding",
             adapter_id=_ADAPTER_ID, tags=("native", "onboarding", "tier1")),
    Scenario("capability_tool_generation", "Generate tools from declared affordances",
             adapter_id=_ADAPTER_ID, tags=("native", "onboarding", "tier1")),
)


class YAMLOnboardingAdapter:
    """Verify YAML discovery, binding resolution, and capability tool generation."""

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _VERSION

    async def availability(self) -> AvailabilityResult:
        if importlib.util.find_spec("yaml") is None:
            return AvailabilityResult(
                _ADAPTER_ID, False, "missing_dependency", missing_dependencies=("PyYAML",),
            )
        if not self._examples_dir().is_dir():
            return AvailabilityResult(_ADAPTER_ID, False, "hardware examples directory missing")
        return AvailabilityResult(_ADAPTER_ID, True, "ready")

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        rows = _SCENARIOS
        if tags:
            wanted = set(tags)
            rows = tuple(row for row in rows if wanted.intersection(row.tags))
        return rows[:limit] if limit > 0 else rows

    async def run_trial(
        self, scenario: Scenario, *, seed: int = 42, timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        del timeout_seconds, parameters
        started = time.time()
        available = await self.availability()
        if not available.available:
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.UNAVAILABLE, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, started_at=started, ended_at=time.time(),
                error=available.reason,
            )
        registry = None
        try:
            contexts = self._load_contexts()
            registry = make_registry(*self._runnable_contexts(contexts))
            payload, passed = self._exercise(scenario.scenario_id, contexts, registry)
            return self._result(scenario, seed, started, payload, passed)
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR, started_at=started,
                ended_at=ended, duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, error=str(exc),
                error_type=type(exc).__name__,
            )
        finally:
            if registry is not None:
                await registry.close_all()

    @staticmethod
    def _examples_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "leapflow" / "hardware" / "examples"

    def _load_contexts(self) -> tuple[Any, ...]:
        from leapflow.hardware.providers.yaml_provider import YamlContextProvider

        provider = YamlContextProvider({"devices_dir": str(self._examples_dir())})
        return provider.discover()

    @staticmethod
    def _runnable_contexts(contexts: tuple[Any, ...]) -> tuple[Any, ...]:
        """Keep YAML semantics while replacing unavailable physical transports."""
        return tuple(
            replace(
                context,
                device_id=context.device_id.replace(".", "_"),
                transport=TransportRef("simulated", {"values": {}}),
            )
            for context in contexts
        )

    def _exercise(self, scenario_id: str, contexts: tuple[Any, ...],
                  registry: Any) -> tuple[dict[str, Any], bool]:
        loaded_ids = sorted(context.device_id for context in contexts)
        if scenario_id == "yaml_context_load":
            passed = len(contexts) == 3 and "robot.so100_arm" in loaded_ids
            return {"loaded_device_ids": loaded_ids}, passed
        arm = next(context for context in contexts if context.device_id == "robot.so100_arm")
        if scenario_id == "control_binding_resolution":
            return self._resolve_binding(arm, loaded_ids)
        plugin = CapabilityRouterPlugin()
        plugin.bind_runtime(
            hardware_registry=registry,
            hardware_approval_gate=ScriptedApprovalGate(("allow_once",)),
        )
        tool_names = sorted(tool.name for tool in plugin.tools)
        expected = {"hw_grasp", "hw_place", "hw_push", "hw_pour", "hw_orchestrate"}
        return {"loaded_device_ids": loaded_ids, "tool_names": tool_names}, expected.issubset(tool_names)

    @staticmethod
    def _resolve_binding(arm: Any, loaded_ids: list[str]) -> tuple[dict[str, Any], bool]:
        robot_config = dict(arm.transport.config.get("robot_config") or {})
        raw = {"local_bus": {
            "type": "serial", "port": robot_config.get("serial_port", ""),
            "robot_type": arm.transport.config.get("robot_type", ""),
        }}
        resolver = ControlBindingResolver()
        bindings = resolver.parse(raw)
        resolved = resolver.resolve(bindings, available_transports=frozenset({"robot_arm"}))
        kind = resolved.transport_ref.kind if resolved is not None else ""
        payload = {"loaded_device_ids": loaded_ids, "binding_count": len(bindings),
                   "resolved_transport": kind, "available": bool(resolved and resolved.available)}
        return payload, bool(resolved and resolved.available and kind == "robot_arm")

    def _result(self, scenario: Scenario, seed: int, started: float,
                payload: dict[str, Any], passed: bool) -> TrialResult:
        metrics = (
            MetricValue("onboarding_success_rate", unseen_success_rate(1 if passed else 0, 1),
                        "ratio", threshold=1.0),
            MetricValue("loaded_context_count", float(len(payload.get("loaded_device_ids", ()))),
                        "count"),
        )
        ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="yaml_onboarding")
        ended = time.time()
        return TrialResult(
            "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
            metrics=metrics, evidence=(ref,), started_at=started, ended_at=ended,
            duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
            adapter_version=_VERSION, seed=seed,
            error="" if passed else "YAML onboarding invariant failed",
        )


__all__ = ["YAMLOnboardingAdapter"]
