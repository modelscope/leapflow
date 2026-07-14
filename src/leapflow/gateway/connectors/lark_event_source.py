"""Feishu/Lark event source backed by ``lark-cli event consume``."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, NamedTuple, Sequence

from leapflow.gateway.connectors.composite_event_source import CompositeEventSource
from leapflow.gateway.connectors.event_sources import CliEventSourceConfig, CliNdjsonEventSource
from leapflow.gateway.connectors.protocol import BackendEvent, EventSourceStatus

logger = logging.getLogger(__name__)

_DEFAULT_EVENT_KEYS: tuple[str, ...] = (
    "im.message.receive_v1",
    "card.action.trigger",
)


class BotIdentity(NamedTuple):
    """Resolved bot identity from ``lark-cli auth status --verify``."""

    open_id: str = ""
    app_name: str = ""


class LarkCliEventSource:
    """Feishu-specific event source wrapping ``lark-cli event consume``.

    Builds a ``CliEventSourceConfig`` tuned for the lark-cli contract:

    - stderr ready marker: ``[event] ready event_key=<key>``
    - stdin EOF triggers graceful exit (unbounded non-TTY mode)
    - ``auth status --verify --json`` to discover bot ``openId``

    Supports multiple event keys via ``CompositeEventSource``.  Each
    event key spawns its own ``lark-cli event consume`` subprocess.
    """

    platform_id = "feishu"
    backend_kind = "cli"

    def __init__(
        self,
        *,
        event_keys: Sequence[str] = _DEFAULT_EVENT_KEYS,
        binary: str = "lark-cli",
        profile: str = "",
        identity: str = "bot",
    ) -> None:
        self._event_keys = tuple(event_keys)
        self._binary = binary
        self._profile = profile
        self._identity = identity

        children = [
            self._make_child_source(key) for key in self._event_keys
        ]
        if len(children) == 1:
            self._source: CliNdjsonEventSource | CompositeEventSource = children[0]
        else:
            self._source = CompositeEventSource(children, platform_id="feishu")

    def _make_child_source(self, event_key: str) -> CliNdjsonEventSource:
        args: list[str] = ["event", "consume", event_key]
        if self._profile:
            args.extend(["--profile", self._profile])
        if self._identity:
            args.extend(["--as", self._identity])
        config = CliEventSourceConfig(
            binary=self._binary,
            args=tuple(args),
            platform_id="feishu",
            ready_pattern=r"\[event\] ready event_key=",
            error_pattern=r"\[error\]",
            ready_timeout_s=30.0,
        )
        return CliNdjsonEventSource(config)

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        return await self._source.start(checkpoint=checkpoint)

    async def stop(self) -> EventSourceStatus:
        return await self._source.stop()

    async def events(self) -> AsyncIterator[BackendEvent]:
        async for event in self._source.events():
            yield event

    async def status(self) -> EventSourceStatus:
        return await self._source.status()

    async def fetch_bot_identity(self) -> BotIdentity:
        """Fetch the bot's identity via ``lark-cli auth status --verify``.

        Returns ``BotIdentity(open_id, app_name)`` for self-message
        filtering and mention detection.  Degrades gracefully — empty
        fields disable the corresponding filter without blocking.
        """
        argv: list[str] = [self._binary]
        if self._profile:
            argv.extend(["--profile", self._profile])
        argv.extend(["auth", "status", "--json", "--verify"])
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            data = json.loads(stdout.decode("utf-8", errors="replace"))
            identities = data.get("identities") or {}
            bot_info = identities.get("bot") or {}
            open_id = str(bot_info.get("openId") or "")
            app_name = str(bot_info.get("appName") or "")
            if open_id:
                logger.info(
                    "Feishu bot identity resolved: %s… (%s)",
                    open_id[:10], app_name or "unnamed",
                )
            return BotIdentity(open_id=open_id, app_name=app_name)
        except FileNotFoundError:
            logger.warning("lark-cli binary not found for bot identity fetch")
        except asyncio.TimeoutError:
            logger.warning("lark-cli auth status timed out")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse bot identity from lark-cli: %s", exc)
        except OSError as exc:
            logger.warning("OS error fetching bot identity: %s", exc)
        return BotIdentity()
