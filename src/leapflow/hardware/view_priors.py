# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rank-only observation learning contract; no trust or motion authority."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ViewPriorSettings:
    enabled: bool = False
    ttl_s: float = 86400.0
    max_age_s: float = 300.0
    duration_scale_s: float = 30.0
    lookup_timeout_s: float = 1.0
    max_events: int = 2048
    max_pending: int = 32

    @classmethod
    def from_json(cls, raw: str) -> ViewPriorSettings:
        values = json.loads(raw or '{}')
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError('hardware.view_priors must contain only declared learning settings')
        return cls(**values)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError('view priors enabled must be boolean')
        for name in ('ttl_s', 'max_age_s', 'duration_scale_s', 'lookup_timeout_s'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'view priors {name} must be finite and positive')
        for name in ('max_events', 'max_pending'):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 65536:
                raise ValueError(f'view priors {name} must be an integer in [1, 65536]')


@runtime_checkable
class ViewPriorPort(Protocol):
    async def hints(self, workcell: str) -> Mapping[str, Any]: ...

    def expect_scope(self, workcell: str, scope: str) -> None: ...

    def record_run(self, node_id: str, run: Mapping[str, Any]) -> None: ...

    def record_decision(self, node_id: str, workcell: str, scope: str, phase: str,
                        outcome: Mapping[str, Any], decision_id: str) -> None: ...
