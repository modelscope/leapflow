"""Host-side typed capabilities exposed to restricted DSH workers.

Foreign code never receives raw shell, filesystem, process or network access.
The only P0 compatibility shim translates one exact legacy curl GET shape into
LeapFlow's governed ``web_fetch`` path.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

WebFetch = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Stable protocol boundary, not natural-language classification. The entire
# command must match; the quoted URL cannot contain another quote, whitespace,
# shell metacharacters, substitutions, redirects or arbitrary pipes.
_CURL_GET = re.compile(
    r"^curl -sS -m (?P<timeout>\d{1,3}) '(?P<url>https?://[^'\s$;&|<>()`\\]+)'"
    r"(?P<iconv> \| iconv -f GB18030 -t UTF-8)?$"
)
_MAX_TIMEOUT_S = 120
_MAX_RESPONSE_BYTES = 4_194_304


class DshCapabilityError(ValueError):
    """A plugin requested a capability outside the P0 policy."""


@dataclass(frozen=True)
class CurlGetSpec:
    url: str
    timeout_s: int
    max_bytes: int
    decode_gb18030: bool


def parse_curl_get(command: str, *, stdout_max_bytes: int = 1_048_576) -> CurlGetSpec:
    """Parse the only shell compatibility shape P0 permits.

    Any deviation is rejected; nothing is executed by a shell. In particular,
    semicolons, redirection, command substitution and a second pipeline cannot
    be represented by this grammar.
    """
    match = _CURL_GET.fullmatch(str(command or ""))
    if match is None:
        raise DshCapabilityError(
            "DSH shell compatibility only permits: curl -sS -m <1..120> "
            "'<http(s) URL>' [| iconv -f GB18030 -t UTF-8]"
        )
    timeout = int(match.group("timeout"))
    if not 1 <= timeout <= _MAX_TIMEOUT_S:
        raise DshCapabilityError("curl timeout must be between 1 and 120 seconds")
    max_bytes = min(max(1, int(stdout_max_bytes)), _MAX_RESPONSE_BYTES)
    return CurlGetSpec(
        url=match.group("url"),
        timeout_s=timeout,
        max_bytes=max_bytes,
        decode_gb18030=bool(match.group("iconv")),
    )


class DshCapabilityBroker:
    """Dispatch the small, deny-by-default capability surface for one worker."""

    def __init__(self, *, web_fetch: WebFetch | None = None) -> None:
        self._web_fetch = web_fetch

    async def dispatch(self, capability: str, arguments: dict[str, Any]) -> Any:
        if capability != "compat.shell.run":
            raise DshCapabilityError(f"Unsupported DSH capability: {capability}")
        return await self._run_legacy_curl(arguments)

    async def _run_legacy_curl(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command") or "")
        try:
            stdout_limit = int(arguments.get("stdoutMaxBytes") or 1_048_576)
        except (TypeError, ValueError) as exc:
            raise DshCapabilityError("stdoutMaxBytes must be an integer") from exc
        spec = parse_curl_get(command, stdout_max_bytes=stdout_limit)
        fetch = self._web_fetch
        if fetch is None:
            from leapflow.tools.web_fetch import web_fetch

            fetch = web_fetch
        result = await fetch(
            {
                "url": spec.url,
                "timeout": spec.timeout_s,
                "max_bytes": spec.max_bytes,
                "extract": "raw_text",
                "encoding": "gb18030" if spec.decode_gb18030 else "",
            }
        )
        if not isinstance(result, dict) or result.get("ok") is not True:
            error = str(
                result.get("error") if isinstance(result, dict) else "web fetch failed"
            )
            return {
                "exitCode": 22,
                "stdout": {"text": ""},
                "stderr": {"text": error[:1000]},
            }
        raw_text = result.get("text")
        if raw_text is None and "data" in result:
            raw_text = json.dumps(result["data"], ensure_ascii=False, separators=(",", ":"))
        text = str(raw_text or result.get("body_excerpt") or "")
        return {
            "exitCode": 0,
            "stdout": {"text": _truncate_utf8(text, spec.max_bytes)},
            "stderr": {"text": ""},
        }


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Return a valid UTF-8 prefix whose encoded size respects the byte cap."""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
