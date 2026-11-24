# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable service configuration values, safe during Settings bootstrap."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ServiceLimits:
    deadline_s: float = 5.0
    attempts: int = 1
    max_inflight: int = 1
    max_queue: int = 0
    max_request_bytes: int = 8388608
    max_response_bytes: int = 2097152
    max_output_tokens: int = 1024


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    provider: str
    base_url: str
    model: str
    task: str
    codec: str = "json.v1"
    output: str = "json_text"
    auth_ref: str = ""
    processing_location: str = "unknown"
    allow_remote: bool = False
    identity_mode: str = "unverified"
    limits: ServiceLimits = field(default_factory=ServiceLimits)
    declaration_json: str = "{}"
    enabled: bool = True

    def declaration(self) -> dict[str, Any]:
        """Return a detached, secret-reference-only declaration."""
        return json.loads(self.declaration_json)
