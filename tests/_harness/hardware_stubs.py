# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared stubs for the hardware test layer.

``ScriptedHuman`` and ``with_transport_config`` are needed by several hardware
test modules.  Keeping them here avoids cross-imports between test files —
which would couple test modules that should stay independent — while still
sharing a single, authoritative implementation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from leapflow.hardware.context import HardwareContext
from leapflow.security.approval import ApprovalDecision, ApprovalRequest


class ScriptedHuman:
    """Stands in for the person at the prompt. The only fake in the chain."""

    def __init__(self, *decisions: ApprovalDecision) -> None:
        self._decisions = list(decisions)
        self.prompts: list[ApprovalRequest] = []

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.prompts.append(request)
        if not self._decisions:
            return ApprovalDecision.DENY
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def with_transport_config(context: HardwareContext, **overrides: Any) -> HardwareContext:
    """Return *context* with its transport config merged with *overrides*.

    Lets a test change device behaviour -- inject a failure, remove halt support,
    open an interlock -- without restating the whole declaration.
    """
    merged = {**dict(context.transport.config), **overrides}
    return replace(context, transport=replace(context.transport, config=merged))
