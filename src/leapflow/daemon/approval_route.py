# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared ContextVar for per-turn approval routing.

Extracted to its own module to avoid circular dependency between
service.py and approval_coordinator.py.
"""
from __future__ import annotations

import asyncio
import contextvars
from typing import Any

# Per-turn approval routing: (queue, active_request_id) or None.
approval_route: contextvars.ContextVar[
    "tuple[asyncio.Queue[Any], str] | None"
] = contextvars.ContextVar("leapd_approval_route", default=None)
