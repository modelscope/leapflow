# Copyright (c) Alibaba, Inc. and its affiliates.
"""Composition-root helper; service runtime and hardware remain independent ports."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

from leapflow.domain.effect_scope import EffectScope
from leapflow.llm.services.contracts import TaskContractRegistry
from leapflow.hardware.model_qualification import ManagedModelQualification
from leapflow.hardware.transport import HardwareTransport
from leapflow.hardware.transports.model_service import build_transport
from leapflow.llm.services.protocol import ServiceSpec, StructuredModelClient
from leapflow.security.grants import ApprovalAuditLog


def compose_task_contracts(scope: EffectScope) -> TaskContractRegistry:
    """Compose domain contracts once; plugin withdrawal uses the existing scope."""
    from leapflow.hardware.model_contract import MODEL_TASKS
    from leapflow.hardware.service_contracts import HardwareTaskContract
    from leapflow.world_model.specialist_contract import INPUT_SCHEMAS, AdvisoryTaskContract

    registry = TaskContractRegistry()
    for task in sorted(INPUT_SCHEMAS):
        scope.effect(registry.register(AdvisoryTaskContract(task)))
    for task in sorted(MODEL_TASKS):
        scope.effect(registry.register(HardwareTaskContract(task)))
    return registry


def local_service_catalog(settings: Any, specs: tuple[ServiceSpec, ...], *, owner: Any = None) -> dict[str, Any]:
    """Use composed admission without provider construction or endpoint probes."""
    if owner is not None:
        return owner.describe(specs=specs)
    from leapflow.llm.provider_registry import get_default_registry, get_scoped_default_registry
    from leapflow.llm.services.runtime import ModelServiceRuntime

    get_scoped_default_registry()
    with EffectScope("local-model-admission") as scope:
        client = ModelServiceRuntime(get_default_registry(), specs, contracts=compose_task_contracts(scope))
        return client.describe()


def model_transport_factories(client: StructuredModelClient, settings: Any,
                              evidence_store: Callable[[], Any]) -> Mapping[str, Callable[..., HardwareTransport]]:
    """Bind one owner's factory without changing process-global transport registrations."""
    qualification = ManagedModelQualification(evidence_store, ApprovalAuditLog(settings.profile_layout.approval.audit_path))
    return {"model_service": partial(build_transport, model_services=client, qualification=qualification)}
