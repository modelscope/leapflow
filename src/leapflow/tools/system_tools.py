# Copyright (c) Alibaba, Inc. and its affiliates.
"""System utilities — time, environment info.

All handlers follow the unified tool convention: receive params dict, return result dict.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict

from leapflow.tools.execution_context import current_tool_context


async def time_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get current date and time in multiple formats."""
    from datetime import datetime

    now = datetime.now()
    return {
        "ok": True,
        "iso": now.isoformat(),
        "unix": time.time(),
        "human": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


async def env_info(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get system environment information."""
    ctx = current_tool_context()
    cwd = str(ctx.workspace_root) if ctx is not None else os.getcwd()
    payload = {
        "ok": True,
        "os": platform.system(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cwd": cwd,
        "user": os.environ.get("USER", "unknown"),
    }
    if ctx is not None:
        payload["workspace_root"] = str(ctx.workspace_root)
        payload["session_id"] = ctx.session_id
    return payload
