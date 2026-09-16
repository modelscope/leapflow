# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cassette store: request fingerprinting, persistence, and miss diagnostics.

A cassette is one recorded OpenAI-compatible HTTP exchange. Recording real
provider traffic — instead of hand-writing response bodies — is what keeps the
LLM boundary honest: the real ``openai`` SDK, the real ``httpx`` stack and the
real SSE framing all stay in the path, so a provider-parsing defect surfaces in
a test rather than in production.

Design notes:

- **Fingerprints normalize away volatile content.** A prompt embeds timestamps,
  session ids and temp paths that change every run; without scrubbing them no
  cassette would ever match twice.
- **One cassette holds a *sequence* of responses.** Retry and failover paths
  send the *same* request repeatedly and must see *different* answers (429 then
  200). Responses are consumed in order and the last one repeats.
- **Failure injection is just an authored cassette.** A 429, a 500, a context
  overflow and a truncated SSE stream are all ordinary recordings, so replay
  needs no special-case branch and the stored file still looks like real wire
  traffic.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CASSETTE_SUFFIX = ".cassette.json"

# Volatile substrings that must not enter a fingerprint. Ordered: broader
# patterns last, so an ISO timestamp is not first eaten by the digit rule.
_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    # Undashed hex identifiers. Tool results carry a fresh ``execution_id`` on
    # every call, so without this a single tool use makes the whole prompt
    # unfingerprintable and no tool-using journey could ever replay. Bounded at 24
    # chars so ordinary hex-looking content (short hashes, colours) is untouched.
    (re.compile(r"\b[0-9a-f]{24,}\b"), "<HEX>"),
    (re.compile(r"\b(?:sess|ws|req|traj|ep|skill|watch|call)-[0-9a-zA-Z]{6,}\b"), "<ID>"),
    (re.compile(r"\bcall_[0-9a-zA-Z]{6,}\b"), "<ID>"),
    (re.compile(r"/(?:private/)?(?:var|tmp)/[^\s\"',)\]]*"), "<TMP>"),
    # Windows drive-letter paths: journeys scratch under %TEMP% and prompts
    # plus tool results embed those paths, so normalize like POSIX /tmp.
    (re.compile(r"[A-Za-z]:\\[^\s\"',)\]]*"), "<TMP>"),
    (re.compile(r"127\.0\.0\.1:\d+"), "127.0.0.1:<PORT>"),
    (re.compile(r"\b1[0-9]{9}(?:\.[0-9]+)?\b"), "<TS>"),
)


def scrub(text: str) -> str:
    """Replace run-varying substrings so equivalent prompts fingerprint equally."""
    for pattern, replacement in _SCRUBBERS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_content(content: Any) -> Any:
    """Normalize a message ``content`` field, which may be text or multimodal."""
    if isinstance(content, str):
        return scrub(content)
    if isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if isinstance(part, Mapping):
                kind = part.get("type", "")
                if kind == "text":
                    parts.append({"type": "text", "text": scrub(str(part.get("text", "")))})
                else:
                    # Image/audio payloads are large and byte-unstable; their
                    # presence and kind is what shapes the request.
                    parts.append({"type": kind})
            else:
                parts.append(scrub(str(part)))
        return parts
    if content is None:
        return None
    return scrub(str(content))


def normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a chat-completions body to its fingerprint-relevant shape.

    Tools are reduced to their *names*: including full JSON schemas would
    invalidate every cassette on a one-word description tweak, which is the
    fastest way to make a replay suite unusable.
    """
    messages: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        entry: dict[str, Any] = {
            "role": str(message.get("role", "")),
            "content": _normalize_content(message.get("content")),
        }
        calls = message.get("tool_calls") or []
        if calls:
            entry["tool_calls"] = [
                str((call.get("function") or {}).get("name", ""))
                for call in calls
                if isinstance(call, Mapping)
            ]
        if message.get("tool_call_id"):
            entry["tool_result"] = True
        messages.append(entry)

    tool_names = sorted(
        str((tool.get("function") or {}).get("name", ""))
        for tool in (payload.get("tools") or [])
        if isinstance(tool, Mapping)
    )

    normalized: dict[str, Any] = {
        "model": str(payload.get("model", "")),
        "stream": bool(payload.get("stream", False)),
        "messages": messages,
    }
    if tool_names:
        normalized["tools"] = tool_names
    temperature = payload.get("temperature")
    if isinstance(temperature, (int, float)):
        normalized["temperature"] = round(float(temperature), 2)
    return normalized


def fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the stable cassette key for a chat-completions request body."""
    canonical = json.dumps(normalize_request(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Records ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CassetteResponse:
    """One HTTP response, either a whole body or a sequence of SSE frames."""

    status: int = 200
    body: bytes = b""
    frames: tuple[bytes, ...] = ()
    content_type: str = "application/json"

    @property
    def is_stream(self) -> bool:
        """True when this response replays as server-sent events."""
        return bool(self.frames)

    def to_json(self) -> dict[str, Any]:
        """Serialize, preferring readable text over base64."""
        payload: dict[str, Any] = {"status": self.status, "content_type": self.content_type}
        if self.frames:
            payload["frames"] = [_encode(frame) for frame in self.frames]
        else:
            payload["body"] = _encode(self.body)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CassetteResponse":
        """Rebuild from :meth:`to_json` output."""
        frames = tuple(_decode(frame) for frame in payload.get("frames") or ())
        body = _decode(payload["body"]) if "body" in payload else b""
        return cls(
            status=int(payload.get("status", 200)),
            body=body,
            frames=frames,
            content_type=str(payload.get("content_type", "application/json")),
        )


@dataclass(frozen=True)
class CassetteRecord:
    """A fingerprinted request paired with the responses it produced, in order."""

    fingerprint: str
    request: dict[str, Any]
    responses: tuple[CassetteResponse, ...]
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialize the whole record."""
        return {
            "fingerprint": self.fingerprint,
            "note": self.note,
            "request": self.request,
            "responses": [response.to_json() for response in self.responses],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CassetteRecord":
        """Rebuild from :meth:`to_json` output."""
        return cls(
            fingerprint=str(payload["fingerprint"]),
            request=dict(payload.get("request") or {}),
            responses=tuple(
                CassetteResponse.from_json(item) for item in payload.get("responses") or ()
            ),
            note=str(payload.get("note", "")),
        )

    def appended(self, response: CassetteResponse) -> "CassetteRecord":
        """Return a copy with one more response at the end of the sequence."""
        return CassetteRecord(
            fingerprint=self.fingerprint,
            request=self.request,
            responses=self.responses + (response,),
            note=self.note,
        )


def _encode(raw: bytes) -> str | dict[str, str]:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"b64": base64.b64encode(raw).decode("ascii")}


def _decode(value: Any) -> bytes:
    if isinstance(value, Mapping):
        return base64.b64decode(value["b64"])
    return str(value).encode("utf-8")


def total_tokens_of(response: "CassetteResponse") -> int:
    """Return the provider-reported total token count for one response.

    Read from the response itself rather than from LeapFlow's own bookkeeping, so
    the number is the provider's and covers every client (primary, aux, VLM)
    without depending on which of them recorded what. Streamed answers report
    usage on a late frame, so frames are scanned newest-first.
    """
    if response.frames:
        for frame in reversed(response.frames):
            for payload in _json_objects_in(frame):
                usage = payload.get("usage")
                if isinstance(usage, Mapping):
                    total = usage.get("total_tokens")
                    if isinstance(total, int):
                        return total
        return 0
    if not response.body:
        return 0
    try:
        payload = json.loads(response.body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return 0
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    if isinstance(usage, Mapping) and isinstance(usage.get("total_tokens"), int):
        return int(usage["total_tokens"])
    return 0


def _json_objects_in(frame: bytes) -> Iterable[dict[str, Any]]:
    """Yield the JSON objects carried by the ``data:`` lines of an SSE frame."""
    for line in frame.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if not body or body == "[DONE]":
            continue
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


# ── Store ────────────────────────────────────────────────────────────────


class CassetteMiss(LookupError):
    """Raised in replay mode when no cassette matches the incoming request."""


@dataclass
class CassetteStore:
    """Directory-backed cassette collection, indexed by fingerprint."""

    root: Path
    _records: dict[str, CassetteRecord] = field(default_factory=dict)
    _paths: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Re-read every cassette file under ``root``."""
        self._records.clear()
        self._paths.clear()
        if not self.root.is_dir():
            return
        for path in sorted(self.root.rglob(f"*{CASSETTE_SUFFIX}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"unreadable cassette {path}: {exc}") from exc
            record = CassetteRecord.from_json(payload)
            self._records[record.fingerprint] = record
            self._paths[record.fingerprint] = path

    def __len__(self) -> int:
        return len(self._records)

    def keys(self) -> Iterable[str]:
        """Return every known fingerprint."""
        return tuple(self._records)

    def get(self, key: str) -> CassetteRecord | None:
        """Return the record for ``key``, or None when absent."""
        return self._records.get(key)

    def put(self, record: CassetteRecord) -> Path:
        """Persist ``record``, overwriting any earlier version."""
        path = self._paths.get(record.fingerprint) or (self.root / self._filename(record))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record.to_json(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._records[record.fingerprint] = record
        self._paths[record.fingerprint] = path
        return path

    @staticmethod
    def _filename(record: CassetteRecord) -> str:
        model = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(record.request.get("model", "model")))
        return f"{model}-{record.fingerprint[:16]}{CASSETTE_SUFFIX}"

    def explain_miss(self, payload: Mapping[str, Any]) -> str:
        """Describe a miss by diffing against the closest stored request.

        A bare "no cassette for <hash>" is unusable: every prompt edit produces
        one and gives no hint which message drifted, so the only response is to
        re-record everything. The nearest-neighbour diff names the change.
        """
        wanted = normalize_request(payload)
        rendered = json.dumps(wanted, indent=2, sort_keys=True, ensure_ascii=False)
        header = (
            f"no cassette for fingerprint {fingerprint(payload)} "
            f"(model={wanted.get('model')}, stream={wanted.get('stream')}, "
            f"{len(wanted.get('messages') or [])} messages, {len(self._records)} cassettes loaded)"
        )
        if not self._records:
            return f"{header}\nStore {self.root} is empty — run `make seed-cassettes`."

        best_key, best_ratio = "", -1.0
        for key, record in self._records.items():
            candidate = json.dumps(record.request, indent=2, sort_keys=True, ensure_ascii=False)
            ratio = difflib.SequenceMatcher(None, rendered, candidate).quick_ratio()
            if ratio > best_ratio:
                best_key, best_ratio = key, ratio

        nearest = json.dumps(
            self._records[best_key].request, indent=2, sort_keys=True, ensure_ascii=False
        )
        diff = difflib.unified_diff(
            nearest.splitlines(),
            rendered.splitlines(),
            fromfile=f"nearest cassette {self._paths[best_key].name}",
            tofile="incoming request",
            lineterm="",
            n=2,
        )
        return (
            f"{header}\nNearest stored request (similarity {best_ratio:.0%}):\n"
            + "\n".join(list(diff)[:80])
        )


# ── Authoring helpers (failure injection) ────────────────────────────────


def sse_frames(*deltas: str, finish_reason: str = "stop", model: str = "cassette") -> tuple[bytes, ...]:
    """Build SSE frames for a streamed text answer, terminated with ``[DONE]``."""
    frames: list[bytes] = []
    for delta in deltas:
        chunk = {
            "id": "chatcmpl-cassette",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        frames.append(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
    tail = {
        "id": "chatcmpl-cassette",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 64, "completion_tokens": 16, "total_tokens": 80},
    }
    frames.append(f"data: {json.dumps(tail)}\n\n".encode("utf-8"))
    frames.append(b"data: [DONE]\n\n")
    return tuple(frames)


def streamed_response(*deltas: str, model: str = "cassette") -> CassetteResponse:
    """A normal streamed 200 response carrying ``deltas``."""
    return CassetteResponse(
        status=200,
        frames=sse_frames(*deltas, model=model),
        content_type="text/event-stream",
    )


def json_response(
    *,
    content: str = "",
    tool_calls: Sequence[Mapping[str, Any]] = (),
    model: str = "cassette",
    finish_reason: str = "",
) -> CassetteResponse:
    """A non-streamed 200 chat completion, optionally carrying native tool calls.

    The engine's native-tool round calls ``achat(stream=False)``, so tool-calling
    turns are whole-body JSON rather than SSE. Journeys therefore need both wire
    forms, and which one applies is decided by the request, not by the author.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        arguments = call.get("arguments", {})
        calls.append(
            {
                "id": str(call.get("id") or f"call_{index + 1}"),
                "type": "function",
                "function": {
                    "name": str(call.get("name", "")),
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    if calls:
        message["tool_calls"] = calls
    body = {
        "id": "chatcmpl-cassette",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
            }
        ],
        "usage": {"prompt_tokens": 64, "completion_tokens": 16, "total_tokens": 80},
    }
    return CassetteResponse(status=200, body=json.dumps(body).encode("utf-8"))


def truncated_stream_response(*deltas: str, model: str = "cassette") -> CassetteResponse:
    """A streamed response that stops mid-flight, with no terminating frame.

    Models a provider dropping the connection: the client must recover rather
    than hand a half-parsed answer to the user.
    """
    frames = sse_frames(*deltas, model=model)
    return CassetteResponse(
        status=200,
        frames=frames[: max(1, len(frames) - 2)],
        content_type="text/event-stream",
    )


def error_response(status: int, *, code: str, message: str, kind: str = "invalid_request_error") -> CassetteResponse:
    """An OpenAI-shaped error body at ``status``.

    The shape matters: the recovery classifier reads provider error payloads, so
    an injected failure must look exactly like the real thing.
    """
    body = json.dumps({"error": {"message": message, "type": kind, "code": code}})
    return CassetteResponse(status=status, body=body.encode("utf-8"))


def rate_limited_response() -> CassetteResponse:
    """A 429 the provider layer is expected to retry."""
    return error_response(
        429,
        code="rate_limit_exceeded",
        message="Rate limit reached for requests",
        kind="rate_limit_error",
    )


def server_error_response() -> CassetteResponse:
    """A 500 the provider layer is expected to retry."""
    return error_response(
        500, code="internal_error", message="The server had an error", kind="server_error"
    )


def context_overflow_response(*, limit: int = 8192, requested: int = 9001) -> CassetteResponse:
    """A 400 context-length error, the trigger for context compression."""
    return error_response(
        400,
        code="context_length_exceeded",
        message=(
            f"This model's maximum context length is {limit} tokens. "
            f"However, your messages resulted in {requested} tokens."
        ),
    )


def record_for(
    payload: Mapping[str, Any],
    *responses: CassetteResponse,
    note: str = "",
) -> CassetteRecord:
    """Author a cassette for ``payload`` with an explicit response sequence."""
    if not responses:
        raise ValueError("a cassette needs at least one response")
    return CassetteRecord(
        fingerprint=fingerprint(payload),
        request=normalize_request(payload),
        responses=tuple(responses),
        note=note,
    )
