"""System utilities — time, environment info.

All handlers follow the ToolBridge convention: receive params dict, return result dict.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict


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
    return {
        "ok": True,
        "os": platform.system(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
    }
