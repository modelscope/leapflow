# Copyright (c) Alibaba, Inc. and its affiliates.
"""Composition-root helper; service runtime and hardware remain independent ports."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

from leapflow.hardware.model_qualification import ManagedModelQualification
from leapflow.hardware.transport import HardwareTransport
from leapflow.hardware.transports.model_service import build_transport
from leapflow.llm.services.protocol import StructuredModelClient
from leapflow.security.grants import ApprovalAuditLog


def model_transport_factories(client: StructuredModelClient, settings: Any,
                              evidence_store: Callable[[], Any]) -> Mapping[str, Callable[..., HardwareTransport]]:
    """Bind one owner's factory without changing process-global transport registrations."""
    qualification = ManagedModelQualification(evidence_store, ApprovalAuditLog(settings.profile_layout.approval.audit_path))
    return {"model_service": partial(build_transport, model_services=client, qualification=qualification)}
