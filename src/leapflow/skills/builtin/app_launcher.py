"""Built-in app launcher / basic automation skill."""

from __future__ import annotations

import logging

from leapflow.platform.protocol import HostRpc, Methods

logger = logging.getLogger(__name__)

_APP_ALIASES: dict[str, str] = {
    "terminal": "Terminal",
    "终端": "Terminal",
    "finder": "Finder",
    "访达": "Finder",
}


def _parse_app_and_command(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    app_name: str | None = None
    for alias, name in _APP_ALIASES.items():
        if alias in lowered or alias in text:
            app_name = name
            break

    cmd = None
    if "git status" in lowered:
        cmd = "git status"
    return app_name, cmd


async def run(rpc: HostRpc, *, user_goal: str) -> str:
    """Launch an app and optionally perform a stub AX action sequence (MVP)."""
    app, cmd = _parse_app_and_command(user_goal)
    target = app or "Terminal"
    await rpc.call(Methods.APP_LAUNCH, {"app_name": target})
    await rpc.call(Methods.APP_ACTIVATE, {"app_name": target})

    if cmd:
        await rpc.call(
            Methods.AX_PERFORM,
            {"commands": [{"type": "shell", "cmd": cmd}]},
        )
        return f"Launched app and requested command: {cmd}"
    return "Launched application."
