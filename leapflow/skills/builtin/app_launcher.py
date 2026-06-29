"""Built-in app launcher / basic automation skill."""

from __future__ import annotations

import logging

from leapflow.platform.protocol import HostRpc, Methods

logger = logging.getLogger(__name__)


def _parse_bundle_and_command(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    bundle_id = None
    if "terminal" in lowered or "终端" in text:
        bundle_id = "com.apple.Terminal"
    if "finder" in lowered or "访达" in text:
        bundle_id = "com.apple.finder"

    cmd = None
    if "git status" in lowered:
        cmd = "git status"
    return bundle_id, cmd


async def run(rpc: HostRpc, *, user_goal: str) -> str:
    """Launch an app and optionally perform a stub AX action sequence (MVP)."""
    bundle, cmd = _parse_bundle_and_command(user_goal)
    if bundle:
        await rpc.call(Methods.APP_LAUNCH, {"bundle_id": bundle})
        await rpc.call(Methods.APP_ACTIVATE, {"bundle_id": bundle})
    else:
        await rpc.call(Methods.APP_LAUNCH, {"bundle_id": "com.apple.Terminal"})

    if cmd:
        await rpc.call(
            Methods.AX_PERFORM,
            {"bundle_id": bundle or "com.apple.Terminal", "commands": [{"type": "shell", "cmd": cmd}]},
        )
        return f"Launched app and requested command: {cmd}"
    return "Launched application (mock details may apply)."
