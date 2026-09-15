# Copyright (c) Alibaba, Inc. and its affiliates.
"""JSON-RPC protocol between sandbox host and worker subprocess."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class SandboxRequest:
    """A request from host to sandboxed worker."""

    request_id: str
    method: str  # "invoke_tool" | "list_tools" | "ping" | "shutdown"
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a single-line JSON string."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> SandboxRequest:
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


@dataclass(frozen=True)
class SandboxResponse:
    """A response from sandboxed worker to host."""

    request_id: str
    ok: bool
    result: Any = None
    error: str = ""
    error_type: str = ""

    def to_json(self) -> str:
        """Serialize to a single-line JSON string."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> SandboxResponse:
        """Deserialize from JSON string."""
        return cls(**json.loads(data))
