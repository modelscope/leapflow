# Copyright (c) Alibaba, Inc. and its affiliates.
"""Local OpenAI-compatible proxy that records, replays, or forwards LLM traffic.

Why a proxy instead of patching the provider: ``OpenAIChat`` builds its
``AsyncOpenAI`` client internally, and leapd runs as a *separate process*
(``sys.executable -m leapflow``), so in-process patching cannot reach it. A
proxy is addressed the way production addresses any provider — through
``LEAPFLOW_LLM_BASE_URL`` — which means the real ``openai`` SDK, the real
``httpx`` stack, the real SSE framing and the real retry classification all stay
in the path.

Modes (``LEAPFLOW_TEST_LLM_MODE``):

``replay``
    Serve from the cassette store. A miss is a hard failure carrying a
    nearest-neighbour diff, never a silent fallthrough.
``seed``
    Serve from the journey's declared script and persist each exchange as a
    cassette. This bootstraps a committed, offline-runnable store before any
    real credential exists; ``record`` later replaces those bodies with real
    ones.
``record``
    Forward to the real upstream and persist what comes back — into a *separate*
    ``recordings/`` store. It is evidence of what providers really send, not a
    replay input: a multi-turn agent conversation cannot be replayed from a
    recording, because turn *n*'s prompt embeds the exact round-by-round history
    of every turn before it.
``live``
    Forward to the real upstream and persist nothing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import httpx

from tests._harness.cassette import (
    CassetteRecord,
    CassetteResponse,
    CassetteStore,
    fingerprint,
    json_response,
    normalize_request,
    streamed_response,
    total_tokens_of,
)

logger = logging.getLogger(__name__)

MODE_ENV = "LEAPFLOW_TEST_LLM_MODE"
REPLAY = "replay"
SEED = "seed"
RECORD = "record"
LIVE = "live"
_MODES = (REPLAY, SEED, RECORD, LIVE)

_FORWARD_MODES = (RECORD, LIVE)


def resolve_mode(default: str = REPLAY) -> str:
    """Return the configured proxy mode, validating it early."""
    mode = (os.getenv(MODE_ENV, "") or default).strip().lower()
    if mode not in _MODES:
        raise ValueError(f"{MODE_ENV}={mode!r} is not one of {_MODES}")
    return mode


@dataclass(frozen=True)
class ScriptedTurn:
    """One model turn expressed as *semantics*, rendered to fit the request.

    The engine picks the wire form: a native-tool round sends ``stream=false``
    and expects a whole JSON body, while a plain answer round streams SSE.
    Declaring "answer this" or "call that tool" and rendering on demand keeps
    journeys readable and stops them from encoding a transport detail they do
    not control.
    """

    text: str = ""
    tool_calls: tuple[Mapping[str, Any], ...] = ()

    def render(self, *, stream: bool, model: str) -> CassetteResponse:
        """Return the response body appropriate for this request shape."""
        if self.tool_calls or not stream:
            return json_response(
                content=self.text, tool_calls=self.tool_calls, model=model
            )
        return streamed_response(*_split_for_streaming(self.text), model=model)


def answer(text: str) -> ScriptedTurn:
    """Script a final textual answer."""
    return ScriptedTurn(text=text)


def tool_call(name: str, **arguments: Any) -> ScriptedTurn:
    """Script a single native tool call."""
    return ScriptedTurn(tool_calls=({"name": name, "arguments": arguments},))


def _split_for_streaming(text: str, *, parts: int = 3) -> tuple[str, ...]:
    """Split an answer into a few deltas so multi-frame SSE parsing is exercised."""
    if not text:
        return ("",)
    size = max(1, -(-len(text) // parts))
    return tuple(text[index : index + size] for index in range(0, len(text), size))


@dataclass
class Script:
    """Ordered turns served to requests that have no cassette yet.

    A script is a *seed*, not an assertion target: it exists so a journey can be
    written and committed before a live credential is available. Its bodies are
    superseded by real recordings in ``record`` mode, and the shape check in
    ``tools/sync_fixtures.py`` is what keeps a seeded body from drifting away
    from what providers actually send.

    The **last turn repeats** for every call past the end of the script, so it
    must be a benign final answer. Ending on a tool call or a structured payload
    makes the agent loop keep re-reading it and burn its whole iteration budget.
    """

    turns: list[ScriptedTurn | CassetteResponse] = field(default_factory=list)
    _index: int = 0

    @classmethod
    def of(cls, *entries: str | ScriptedTurn | CassetteResponse) -> "Script":
        """Build a script from answer texts, scripted turns, or raw responses."""
        built: list[ScriptedTurn | CassetteResponse] = []
        for entry in entries:
            if isinstance(entry, (ScriptedTurn, CassetteResponse)):
                built.append(entry)
            else:
                built.append(ScriptedTurn(text=str(entry)))
        return cls(turns=built)

    def next_response(self, *, stream: bool, model: str) -> CassetteResponse | None:
        """Return the next scripted response; the last turn repeats."""
        if not self.turns:
            return None
        entry = self.turns[min(self._index, len(self.turns) - 1)]
        self._index += 1
        if isinstance(entry, CassetteResponse):
            return entry
        return entry.render(stream=stream, model=model)


@dataclass
class ProxyStats:
    """Observable traffic for journey assertions."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    upstream_calls: int = 0
    total_tokens: int = 0
    budget_exceeded: bool = False
    token_budget_exceeded: bool = False

    @property
    def call_count(self) -> int:
        """Number of chat-completions requests the proxy handled."""
        return len(self.requests)

    def prompts_containing(self, needle: str) -> list[dict[str, Any]]:
        """Return requests whose normalized messages mention ``needle``."""
        found = []
        for request in self.requests:
            blob = json.dumps(request.get("messages") or [], ensure_ascii=False)
            if needle in blob:
                found.append(request)
        return found


class CassetteProxy:
    """An OpenAI-compatible HTTP endpoint backed by a cassette store."""

    def __init__(
        self,
        store: CassetteStore,
        *,
        mode: str = REPLAY,
        script: Script | None = None,
        upstream_base_url: str = "",
        upstream_api_key: str = "",
        upstream_model: str = "",
        host: str = "127.0.0.1",
        max_calls: int = 0,
        max_tokens: int = 0,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self._store = store
        self._mode = mode
        self._script = script or Script()
        self._upstream_base_url = upstream_base_url.rstrip("/")
        self._upstream_api_key = upstream_api_key
        self._upstream_model = upstream_model.strip()
        self._host = host
        self._max_calls = max(0, int(max_calls))
        self._max_tokens = max(0, int(max_tokens))
        self._lock = threading.Lock()
        self._cursor: dict[str, int] = {}
        self._captured: set[str] = set()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.stats = ProxyStats()

        if mode in _FORWARD_MODES and not self._upstream_base_url:
            raise ValueError(f"mode {mode!r} needs an upstream base URL (LEAPFLOW_LLM_BASE_URL)")

    # ── Lifecycle ────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Configured proxy mode."""
        return self._mode

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL to hand to LeapFlow via config/env."""
        if self._server is None:
            raise RuntimeError("proxy is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "CassetteProxy":
        """Bind an ephemeral port and serve in a background thread."""
        proxy = self

        class _Handler(_CassetteHandler):
            proxy_ref = proxy

        self._server = ThreadingHTTPServer((self._host, 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="cassette-proxy", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut the server down and join its thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "CassetteProxy":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ── Assertions ───────────────────────────────────────────────────

    def assert_no_misses(self) -> None:
        """Fail with full diagnostics when replay could not answer a request."""
        if self.stats.misses:
            joined = "\n\n".join(self.stats.misses)
            raise AssertionError(
                f"{len(self.stats.misses)} cassette miss(es) in {self._mode} mode:\n\n{joined}"
            )

    # ── Request handling (called on server threads) ───────────────────

    def handle_chat(self, payload: Mapping[str, Any]) -> CassetteResponse:
        """Resolve one chat-completions request to a response.

        Mode decides precedence, and it matters for retry paths: a retry resends
        a *byte-identical* request, so a recording mode must keep capturing
        instead of answering from what it just stored — otherwise a "429 then
        200" sequence could never be recorded at all.
        """
        key = fingerprint(payload)
        stream = bool(payload.get("stream"))
        model = str(payload.get("model") or "cassette")
        with self._lock:
            self.stats.requests.append(normalize_request(payload))
            over_calls = 0 < self._max_calls < len(self.stats.requests)
            if over_calls:
                self.stats.budget_exceeded = True
            over_tokens = 0 < self._max_tokens <= self.stats.total_tokens
            if over_tokens:
                self.stats.token_budget_exceeded = True
        if over_calls:
            return self._refuse(
                "journey_call_budget_exceeded",
                f"journey exceeded its provider-call budget of {self._max_calls}. "
                "Either the turn stopped converging, or the prompt grew enough to "
                "need more rounds — both are regressions worth looking at, not a "
                "reason to raise the ceiling.",
            )
        if over_tokens:
            return self._refuse(
                "journey_token_budget_exceeded",
                f"journey spent {self.stats.total_tokens} tokens, past its ceiling "
                f"of {self._max_tokens}. Call count alone cannot catch this: prompt "
                "assembly growing (a longer system prompt, more tool schemas) "
                "raises the bill without adding a single round.",
            )

        response = self._resolve(key, payload, stream=stream, model=model)
        with self._lock:
            self.stats.total_tokens += total_tokens_of(response)
        return response

    def _resolve(
        self, key: str, payload: Mapping[str, Any], *, stream: bool, model: str
    ) -> CassetteResponse:
        """Answer one request from the mode's authoritative source."""
        if self._mode in _FORWARD_MODES:
            response = self._forward(payload)
            if self._mode == RECORD:
                self._capture(key, payload, response)
            return response

        if self._mode == SEED:
            response = self._script.next_response(stream=stream, model=model)
            if response is not None:
                mismatch = _shape_mismatch(response, stream=stream)
                if mismatch:
                    return self._author_error(mismatch)
                self._capture(key, payload, response)
                return response

        with self._lock:
            record = self._store.get(key)
            if record is not None:
                index = self._cursor.get(key, 0)
                self._cursor[key] = index + 1
                return record.responses[min(index, len(record.responses) - 1)]

        return self._miss(payload)

    def _refuse(self, code: str, message: str) -> CassetteResponse:
        """Refuse further provider calls with a non-retryable status.

        Enforced here rather than only asserted afterwards, because the failure
        being guarded against is a loop that does not converge: letting it run to
        the engine's iteration cap costs minutes offline and real money live. A
        400 stops it at the ceiling — a 429 or 5xx would be retried and the loop
        would continue.
        """
        logger.error("cassette-proxy: %s", message)
        body = json.dumps(
            {"error": {"message": message, "type": "invalid_request_error", "code": code}}
        )
        return CassetteResponse(status=400, body=body.encode("utf-8"))

    def _author_error(self, explanation: str) -> CassetteResponse:
        """Report an authoring mistake as a failure of the test, not of the product.

        Serving an SSE body to a ``stream=false`` request makes the SDK parse a
        string as a completion object; the resulting ``AttributeError`` is then
        correctly classified as a LeapFlow defect, and the journey appears to have
        found a product bug it did not find. Catching the mis-shape here keeps the
        blame where it belongs.
        """
        with self._lock:
            self.stats.misses.append(explanation)
        logger.error("cassette authoring error: %s", explanation)
        body = json.dumps(
            {
                "error": {
                    "message": explanation,
                    "type": "invalid_request_error",
                    "code": "cassette_shape_mismatch",
                }
            }
        )
        return CassetteResponse(status=400, body=body.encode("utf-8"))

    def _miss(self, payload: Mapping[str, Any]) -> CassetteResponse:
        """Record a replay miss and answer with a non-retryable 400.

        400 is deliberate: the provider retries 429/5xx, so answering a miss
        with a 500 would burn the whole retry budget before the test could
        report the real problem.
        """
        explanation = self._store.explain_miss(payload)
        with self._lock:
            self.stats.misses.append(explanation)
        logger.error("cassette miss: %s", explanation)
        body = json.dumps(
            {
                "error": {
                    "message": f"cassette miss ({self._mode} mode): {explanation}",
                    "type": "invalid_request_error",
                    "code": "cassette_miss",
                }
            }
        )
        return CassetteResponse(status=400, body=body.encode("utf-8"))

    def _capture(
        self, key: str, payload: Mapping[str, Any], response: CassetteResponse
    ) -> None:
        """Store ``response`` for ``key``, replacing a stale run then appending.

        The first capture of a key in this proxy's lifetime *replaces* whatever a
        previous run left behind; later captures of the same key append. Without
        the replace, re-running the recorder would keep growing every sequence
        and turn a one-off retry into a permanent one.
        """
        with self._lock:
            existing = self._store.get(key) if key in self._captured else None
            if existing is not None:
                record = existing.appended(response)
            else:
                record = CassetteRecord(
                    fingerprint=key,
                    request=normalize_request(payload),
                    responses=(response,),
                    note=f"captured in {self._mode} mode",
                )
            self._captured.add(key)
            self._store.put(record)
            self._cursor[key] = len(record.responses)

    def _forward(self, payload: Mapping[str, Any]) -> CassetteResponse:
        """Call the real upstream and capture its response verbatim.

        The outgoing ``model`` is rewritten to the provider's actual model while
        the cassette keeps the journey's stable placeholder. Without that split,
        recording under a real model name and replaying under the placeholder
        produce different fingerprints, so every re-recorded cassette would miss
        on the next replay run — which would break the record/replay cycle the
        whole design rests on.
        """
        url = f"{self._upstream_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._upstream_api_key}",
            "Content-Type": "application/json",
        }
        outgoing = dict(payload)
        if self._upstream_model:
            outgoing["model"] = self._upstream_model
        with self._lock:
            self.stats.upstream_calls += 1
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        if outgoing.get("stream"):
            frames: list[bytes] = []
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=outgoing, headers=headers) as upstream:
                    status = upstream.status_code
                    content_type = upstream.headers.get("content-type", "text/event-stream")
                    if status >= 400:
                        return CassetteResponse(
                            status=status,
                            body=upstream.read(),
                            content_type=content_type,
                        )
                    for raw in upstream.iter_raw():
                        if raw:
                            frames.append(raw)
            return CassetteResponse(
                status=status, frames=tuple(frames), content_type=content_type
            )
        with httpx.Client(timeout=timeout) as client:
            upstream = client.post(url, json=outgoing, headers=headers)
        return CassetteResponse(
            status=upstream.status_code,
            body=upstream.content,
            content_type=upstream.headers.get("content-type", "application/json"),
        )


class _CassetteHandler(BaseHTTPRequestHandler):
    """Minimal HTTP surface: chat completions plus a models probe."""

    proxy_ref: CassetteProxy
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib hook
        """Silence stdlib access logging; failures are reported by assertions."""
        logger.debug("cassette-proxy %s", fmt % args)

    def do_GET(self) -> None:
        """Answer the model-listing probe some clients issue on startup."""
        if self.path.rstrip("/").endswith("/models"):
            self._send(CassetteResponse(status=200, body=b'{"object":"list","data":[]}'))
            return
        self._send(CassetteResponse(status=404, body=b'{"error":{"message":"not found"}}'))

    def do_POST(self) -> None:
        """Serve a chat-completions request from cassette, script, or upstream."""
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(CassetteResponse(status=404, body=b'{"error":{"message":"not found"}}'))
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(
                CassetteResponse(
                    status=400,
                    body=json.dumps({"error": {"message": f"bad request body: {exc}"}}).encode(),
                )
            )
            return
        response = self.proxy_ref.handle_chat(payload)
        self._send(response)

    def _send(self, response: CassetteResponse) -> None:
        """Write a whole body, or stream SSE frames with flushes between them."""
        if response.is_stream:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for frame in response.frames:
                    self.wfile.write(f"{len(frame):X}\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("cassette-proxy: client closed mid-stream")
            return
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        try:
            self.wfile.write(response.body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("cassette-proxy: client closed before body")


def store_for(journey_id: str, *, mode: str = REPLAY, root: Path | None = None) -> CassetteStore:
    """Return the cassette store a journey should use in ``mode``.

    Recording writes to a **separate** directory from the replay store, and that
    separation is load-bearing rather than tidiness:

    - A recording run must never be able to break the offline lanes. Writing real
      traffic into the replay store does exactly that, because a multi-turn agent
      conversation cannot be replayed from a recording: turn *n*'s prompt embeds
      the round-by-round history of turns 1..n-1, so one divergence (a tool call
      the model made this time but not last time) cascades and every later turn
      misses.
    - The two artefacts answer different questions. ``cassettes/`` holds the
      deterministic inputs the replay lanes assert against; ``recordings/`` holds
      evidence of what providers actually send, which ``tools/sync_fixtures.py``
      distils into the shapes the mock layer checks against.

    Journeys keep separate directories so a nearest-neighbour miss diff stays
    relevant instead of matching an unrelated journey's prompt.
    """
    base = root or Path(__file__).resolve().parents[1] / "_fixtures"
    bucket = "recordings" if mode == RECORD else "cassettes"
    return CassetteStore(base / bucket / journey_id)


def upstream_from_env() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for forwarding modes, from process env."""
    base_url = os.getenv("LEAPFLOW_TEST_UPSTREAM_BASE_URL", "").strip()
    api_key = os.getenv("LEAPFLOW_TEST_UPSTREAM_API_KEY", "").strip()
    model = os.getenv("LEAPFLOW_TEST_UPSTREAM_MODEL", "").strip()
    # The live lane injects the ordinary LLM variables; fall back to them so a
    # single set of CI secrets drives both LeapFlow and the recorder.
    if not base_url:
        base_url = os.getenv("LEAPFLOW_LLM_BASE_URL", "").strip()
    if not api_key:
        api_key = os.getenv("LEAPFLOW_LLM_API_KEY", "").strip()
    if not model:
        model = os.getenv("LEAPFLOW_LLM_MODEL", "").strip()
    return base_url, api_key, model


def _shape_mismatch(response: CassetteResponse, *, stream: bool) -> str:
    """Return an explanation when a response cannot answer this request shape.

    Only successful bodies are checked: an error status is shape-neutral, and
    providers really do answer a streaming request with a plain JSON error.
    """
    if response.status >= 400:
        return ""
    if response.is_stream and not stream:
        return (
            "scripted an SSE response for a stream=false request. The engine's "
            "native-tool round is non-streaming, so use answer()/tool_call() — they "
            "render to fit the request — rather than a raw streamed_response()."
        )
    if stream and not response.is_stream:
        return (
            "scripted a whole-body response for a stream=true request. Use "
            "answer()/tool_call() so the wire form follows the request."
        )
    return ""


def scripted(*entries: str | ScriptedTurn | CassetteResponse) -> Script:
    """Shorthand for :meth:`Script.of`."""
    return Script.of(*entries)
