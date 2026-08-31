"""Tests for the end-to-end harness itself.

The harness is the foundation the whole real layer stands on, so it gets the
same scrutiny as production code. In particular the proxy is exercised through
the *real* ``openai`` client here, because the point of the design is that the
SDK, httpx and SSE parsing stay in the path — a harness verified only through
direct method calls would prove nothing about that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.cassette import (
    CassetteRecord,
    CassetteResponse,
    CassetteStore,
    context_overflow_response,
    fingerprint,
    normalize_request,
    rate_limited_response,
    record_for,
    scrub,
    streamed_response,
    truncated_stream_response,
)
from tests._harness.cassette_proxy import (
    REPLAY,
    SEED,
    CassetteProxy,
    Script,
    resolve_mode,
)
from tests._harness.leapd import hermetic_env

pytestmark = pytest.mark.component


def _chat_body(prompt: str, *, stream: bool = True, model: str = "cassette-model") -> dict:
    return {
        "model": model,
        "stream": stream,
        "messages": [
            {"role": "system", "content": "You are LeapFlow."},
            {"role": "user", "content": prompt},
        ],
    }


# ── Fingerprinting ───────────────────────────────────────────────────────


def test_scrub_removes_volatile_substrings() -> None:
    """Timestamps, ids and temp paths must not enter a fingerprint."""
    raw = (
        "at 2026-08-06T11:12:13Z session sess-9f3a2b1c in /var/folders/xy/T/pytest-1/ws "
        "req 550e8400-e29b-41d4-a716-446655440000 epoch 1785000000 host 127.0.0.1:54321"
    )
    cleaned = scrub(raw)
    assert "2026-08-06" not in cleaned
    assert "sess-9f3a2b1c" not in cleaned
    assert "550e8400" not in cleaned
    assert "1785000000" not in cleaned
    assert "54321" not in cleaned
    assert "<TS>" in cleaned and "<ID>" in cleaned and "<TMP>" in cleaned


def test_fingerprint_is_stable_across_runs_but_sensitive_to_intent() -> None:
    """Equivalent prompts match; a different question does not."""
    first = _chat_body("list files in /var/folders/ab/T/pytest-1/ws at 2026-08-06T00:00:00Z")
    second = _chat_body("list files in /var/folders/zz/T/pytest-9/ws at 2026-08-07T10:30:00Z")
    third = _chat_body("delete every file")

    assert fingerprint(first) == fingerprint(second)
    assert fingerprint(first) != fingerprint(third)


def test_tool_result_identifiers_do_not_break_fingerprint_stability() -> None:
    """A prompt carrying a tool result must fingerprint the same on every run.

    Tool results embed a fresh ``execution_id`` per call — an *undashed* 32-hex
    id that the dashed-UUID rule does not match. Without scrubbing it, a single
    tool use makes the follow-up prompt unique forever, so no tool-using journey
    could ever replay and the offline lanes would be permanently red.
    """
    def _with_execution_id(execution_id: str) -> dict:
        result = {
            "ok": True,
            "path": "/tmp/x/invoice.txt",
            "execution_id": execution_id,
            "tool_call_id": "call_721d22c0479f40668a3aeddb",
            "execution_policy": "read_only",
        }
        return {
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "read the invoice"},
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(result)},
            ],
        }

    first = _with_execution_id("9376486cbab04bfbaea595c5dd7a8d59a")
    second = _with_execution_id("955e2afa50bf4530943346dc01f27122")

    assert fingerprint(first) == fingerprint(second), (
        "a per-call execution id leaked into the fingerprint"
    )


def test_short_hex_content_is_not_scrubbed() -> None:
    """Scrubbing must not swallow ordinary content that merely looks like hex."""
    cleaned = scrub("the colour is deadbeef and the short sha is abc123")
    assert "deadbeef" in cleaned
    assert "abc123" in cleaned


def test_fingerprint_ignores_tool_schema_churn_but_tracks_tool_set() -> None:
    """Descriptions change constantly; the available tool set is what matters."""
    base = _chat_body("hi")
    with_v1 = {
        **base,
        "tools": [
            {"type": "function", "function": {"name": "read_file", "description": "Read a file"}}
        ],
    }
    with_v2 = {
        **base,
        "tools": [
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read a file from disk (v2)"},
            }
        ],
    }
    with_extra = {
        **base,
        "tools": [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "web_fetch"}},
        ],
    }

    assert fingerprint(with_v1) == fingerprint(with_v2)
    assert fingerprint(with_v1) != fingerprint(with_extra)


def test_stream_flag_and_model_separate_fingerprints() -> None:
    """Streaming and non-streaming responses are different recordings."""
    assert fingerprint(_chat_body("hi")) != fingerprint(_chat_body("hi", stream=False))
    assert fingerprint(_chat_body("hi")) != fingerprint(_chat_body("hi", model="other"))


def test_multimodal_content_reduces_to_part_kinds() -> None:
    """Image bytes are unstable; their presence still shapes the request."""
    payload = {
        "model": "m",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this at 2026-08-06T00:00:00Z"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
    }
    normalized = normalize_request(payload)
    parts = normalized["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this at <TS>"}
    assert parts[1] == {"type": "image_url"}


# ── Store ────────────────────────────────────────────────────────────────


def test_cassette_round_trips_through_disk(tmp_path: Path) -> None:
    """A stored cassette reloads byte-identically, streams included."""
    store = CassetteStore(tmp_path)
    body = _chat_body("hello")
    record = record_for(body, streamed_response("he", "llo"), note="greeting")
    store.put(record)

    reloaded = CassetteStore(tmp_path)
    restored = reloaded.get(fingerprint(body))
    assert restored is not None
    assert restored.note == "greeting"
    assert restored.responses[0].frames == record.responses[0].frames
    assert restored.responses[0].content_type == "text/event-stream"


def test_response_sequence_is_consumed_in_order_then_repeats(tmp_path: Path) -> None:
    """Retry paths resend an identical request and must see different answers."""
    store = CassetteStore(tmp_path)
    body = _chat_body("hello")
    store.put(record_for(body, rate_limited_response(), streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY)

    first = proxy.handle_chat(body)
    second = proxy.handle_chat(body)
    third = proxy.handle_chat(body)

    assert first.status == 429
    assert second.status == 200 and second.is_stream
    assert third.status == 200, "the last response repeats once the sequence is exhausted"


def test_explain_miss_diffs_against_the_nearest_request(tmp_path: Path) -> None:
    """A miss must name what drifted, not just report a hash."""
    store = CassetteStore(tmp_path)
    store.put(record_for(_chat_body("summarize the report"), streamed_response("ok")))

    explanation = store.explain_miss(_chat_body("summarize the invoice"))

    assert "no cassette for fingerprint" in explanation
    assert "Nearest stored request" in explanation
    assert "invoice" in explanation


def test_explain_miss_on_empty_store_points_at_recording(tmp_path: Path) -> None:
    """An empty store is a setup problem and must say so."""
    explanation = CassetteStore(tmp_path).explain_miss(_chat_body("hi"))
    assert "make seed-cassettes" in explanation


def test_binary_bodies_survive_via_base64(tmp_path: Path) -> None:
    """Non-UTF8 payloads must round-trip losslessly."""
    store = CassetteStore(tmp_path)
    raw = b"\xff\xfe\x00binary"
    store.put(
        CassetteRecord(
            fingerprint="deadbeef",
            request={"model": "m"},
            responses=(CassetteResponse(status=200, body=raw),),
        )
    )
    restored = CassetteStore(tmp_path).get("deadbeef")
    assert restored is not None
    assert restored.responses[0].body == raw


# ── Proxy over real HTTP, driven by the real OpenAI SDK ───────────────────


@pytest.mark.asyncio
async def test_replay_streams_through_the_real_openai_client(tmp_path: Path) -> None:
    """The recorded SSE frames must parse through the production provider.

    This is the whole point of the proxy: ``OpenAIChat`` — not a stub — reads the
    stream, so frame-level and usage-parsing defects are caught here.
    """
    from leapflow.llm.openai_provider import OpenAIChat

    store = CassetteStore(tmp_path)
    messages = [
        {"role": "system", "content": "You are LeapFlow."},
        {"role": "user", "content": "say hello"},
    ]
    body = {"model": "cassette-model", "stream": True, "messages": messages}
    store.put(record_for(body, streamed_response("Hel", "lo!")))

    with CassetteProxy(store, mode=REPLAY) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=1,
        )
        response = await provider.achat(messages, stream=True)

    assert response.content == "Hello!"
    assert response.usage.get("total_tokens") == 80
    proxy.assert_no_misses()
    assert proxy.stats.call_count == 1


@pytest.mark.asyncio
async def test_injected_rate_limit_is_retried_by_the_provider(tmp_path: Path) -> None:
    """A 429 recording must drive the real retry path, not a simulated one."""
    from leapflow.llm.openai_provider import OpenAIChat

    store = CassetteStore(tmp_path)
    messages = [{"role": "user", "content": "retry me"}]
    body = {"model": "cassette-model", "stream": True, "messages": messages}
    store.put(record_for(body, rate_limited_response(), streamed_response("recovered")))

    with CassetteProxy(store, mode=REPLAY) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=3,
        )
        response = await provider.achat(messages, stream=True)

    assert response.content == "recovered"
    assert proxy.stats.call_count == 2, "the provider must have retried the 429"


@pytest.mark.asyncio
async def test_context_overflow_surfaces_as_a_provider_error(tmp_path: Path) -> None:
    """A 400 context-length body must reach the caller as an error, not empty text."""
    import openai

    from leapflow.llm.openai_provider import OpenAIChat

    store = CassetteStore(tmp_path)
    messages = [{"role": "user", "content": "way too long"}]
    body = {"model": "cassette-model", "stream": True, "messages": messages}
    store.put(record_for(body, context_overflow_response()))

    with CassetteProxy(store, mode=REPLAY) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=1,
        )
        with pytest.raises(openai.APIStatusError) as caught:
            await provider.achat(messages, stream=True)

    assert "maximum context length" in str(caught.value)
    assert proxy.stats.call_count == 1, "a 400 must not be retried"


@pytest.mark.asyncio
async def test_truncated_stream_does_not_silently_yield_a_partial_answer(
    tmp_path: Path,
) -> None:
    """A dropped stream must be observable, not rendered as a complete reply."""
    from leapflow.llm.openai_provider import OpenAIChat

    store = CassetteStore(tmp_path)
    messages = [{"role": "user", "content": "long answer"}]
    body = {"model": "cassette-model", "stream": True, "messages": messages}
    store.put(record_for(body, truncated_stream_response("par", "tial")))

    with CassetteProxy(store, mode=REPLAY) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=1,
        )
        response = await provider.achat(messages, stream=True)

    # The content that did arrive is preserved, but no finish_reason was sent —
    # which is how a caller can tell the turn was cut short.
    assert response.content == "partial"
    assert not response.finish_reason


@pytest.mark.asyncio
async def test_replay_miss_fails_loudly_with_a_diff(tmp_path: Path) -> None:
    """An unmatched request must fail the test, never fall through to silence."""
    from leapflow.llm.openai_provider import OpenAIChat

    store = CassetteStore(tmp_path)
    store.put(record_for(_chat_body("known question"), streamed_response("ok")))

    with CassetteProxy(store, mode=REPLAY) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=1,
        )
        with pytest.raises(Exception):
            await provider.achat([{"role": "user", "content": "unknown question"}], stream=True)

        with pytest.raises(AssertionError) as caught:
            proxy.assert_no_misses()

    assert "cassette miss" in str(caught.value)
    assert "Nearest stored request" in str(caught.value)


@pytest.mark.asyncio
async def test_seed_mode_persists_scripted_exchanges_as_cassettes(tmp_path: Path) -> None:
    """Seeding produces a committed store that later replays offline."""
    from leapflow.llm.openai_provider import OpenAIChat

    store = CassetteStore(tmp_path)
    messages = [{"role": "user", "content": "seed me"}]

    with CassetteProxy(store, mode=SEED, script=Script.of("seeded answer")) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=1,
        )
        seeded = await provider.achat(messages, stream=True)
        proxy.assert_no_misses()

    assert seeded.content == "seeded answer"

    replayed_store = CassetteStore(tmp_path)
    assert len(replayed_store) == 1
    with CassetteProxy(replayed_store, mode=REPLAY) as proxy:
        provider = OpenAIChat(
            api_key="cassette-key",
            base_url=proxy.base_url,
            model="cassette-model",
            max_retries=1,
        )
        replayed = await provider.achat(messages, stream=True)
        proxy.assert_no_misses()

    assert replayed.content == "seeded answer"


def test_proxy_exposes_prompt_traffic_for_assertions(tmp_path: Path) -> None:
    """Journeys assert on what reached the model, e.g. that a tool result fed back."""
    store = CassetteStore(tmp_path)
    body = _chat_body("check the invoice total")
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY)

    proxy.handle_chat(body)

    assert proxy.stats.call_count == 1
    assert proxy.stats.prompts_containing("invoice total")
    assert not proxy.stats.prompts_containing("purchase order")


def test_forwarding_mode_requires_an_upstream(tmp_path: Path) -> None:
    """Record/live without an upstream is a configuration error, caught at build."""
    with pytest.raises(ValueError, match="upstream base URL"):
        CassetteProxy(CassetteStore(tmp_path), mode="record")


def test_unknown_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the mode env var must fail immediately, not default silently."""
    monkeypatch.setenv("LEAPFLOW_TEST_LLM_MODE", "reply")
    with pytest.raises(ValueError, match="not one of"):
        resolve_mode()


# ── Convergence: the provider-call ceiling ──────────────────────────────


def test_call_budget_cuts_off_a_non_converging_loop(tmp_path: Path) -> None:
    """A turn that keeps asking the model must be stopped at the ceiling.

    This is the guard against the failure the whole budget exists for: a loop
    that never converges would otherwise run to the engine's iteration cap,
    costing minutes offline and real money live. The ceiling has to be enforced
    at the boundary, not merely asserted after the fact.
    """
    store = CassetteStore(tmp_path)
    body = _chat_body("loop forever")
    store.put(record_for(body, streamed_response("again")))
    proxy = CassetteProxy(store, mode=REPLAY, max_calls=3)

    statuses = [proxy.handle_chat(body).status for _ in range(6)]

    assert statuses[:3] == [200, 200, 200], f"the budget bit too early: {statuses}"
    assert statuses[3:] == [400, 400, 400], f"calls past the ceiling were served: {statuses}"
    assert proxy.stats.budget_exceeded is True


def test_budget_refusal_is_not_retryable(tmp_path: Path) -> None:
    """The refusal must halt the loop, not feed the provider's retry logic.

    A 429 or 5xx would be retried, so the runaway turn would keep going and burn
    the retry budget on top of the call budget.
    """
    store = CassetteStore(tmp_path)
    body = _chat_body("one call only")
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY, max_calls=1)

    proxy.handle_chat(body)
    refusal = proxy.handle_chat(body)

    assert refusal.status == 400, "a retryable status would let the loop continue"
    assert b"journey_call_budget_exceeded" in refusal.body


def test_zero_budget_means_unlimited(tmp_path: Path) -> None:
    """An unset ceiling must not accidentally block every call."""
    store = CassetteStore(tmp_path)
    body = _chat_body("unbounded")
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY, max_calls=0)

    statuses = {proxy.handle_chat(body).status for _ in range(20)}

    assert statuses == {200}
    assert proxy.stats.budget_exceeded is False


def test_token_budget_cuts_off_prompt_growth(tmp_path: Path) -> None:
    """Cost must be capped by tokens too, not only by call count.

    This is the gap the call ceiling cannot close: a longer system prompt or a
    bigger tool catalogue raises the bill without adding a single round, so a
    call-count-only gate would let the live lane get quietly more expensive.
    """
    store = CassetteStore(tmp_path)
    body = _chat_body("expensive")
    # streamed_response reports 80 total tokens per response.
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY, max_calls=0, max_tokens=200)

    statuses = [proxy.handle_chat(body).status for _ in range(5)]

    assert statuses[:3] == [200, 200, 200], f"budget bit too early: {statuses}"
    assert statuses[3:] == [400, 400], f"calls past the token ceiling served: {statuses}"
    assert proxy.stats.token_budget_exceeded is True
    assert proxy.stats.budget_exceeded is False, "the call ceiling must not be blamed"


def test_token_accounting_reads_streamed_and_whole_body_usage(tmp_path: Path) -> None:
    """Usage must be counted for both wire forms the engine provokes.

    The native-tool round is non-streaming and a plain answer round streams, so
    missing either form would under-count and silently disable the ceiling.
    """
    from tests._harness.cassette import json_response, total_tokens_of

    assert total_tokens_of(streamed_response("a", "b")) == 80
    assert total_tokens_of(json_response(content="hi")) == 80
    assert total_tokens_of(rate_limited_response()) == 0, "an error body reports no usage"

    store = CassetteStore(tmp_path)
    streaming = _chat_body("stream me", stream=True)
    whole = _chat_body("whole body", stream=False)
    store.put(record_for(streaming, streamed_response("ok")))
    store.put(record_for(whole, json_response(content="ok")))
    proxy = CassetteProxy(store, mode=REPLAY)

    proxy.handle_chat(streaming)
    proxy.handle_chat(whole)

    assert proxy.stats.total_tokens == 160


def test_token_budget_refusal_is_not_retryable(tmp_path: Path) -> None:
    """The refusal must halt the loop rather than feed provider retries."""
    store = CassetteStore(tmp_path)
    body = _chat_body("one only")
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY, max_tokens=1)

    first = proxy.handle_chat(body)
    refusal = proxy.handle_chat(body)

    assert first.status == 200, "the first call must be served; the ceiling is cumulative"
    assert refusal.status == 400
    assert b"journey_token_budget_exceeded" in refusal.body


def test_journey_finish_distinguishes_token_from_call_exhaustion(tmp_path: Path) -> None:
    """The failure must name prompt growth, not blame a loop that did not happen."""
    from tests._harness.journey import Journey

    store = CassetteStore(tmp_path)
    body = _chat_body("grow")
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY, max_tokens=1)
    proxy.handle_chat(body)
    proxy.handle_chat(body)

    journey = Journey(
        journey_id="token-probe",
        proxy=proxy,
        daemon=_LogOnlyDaemon(),
        max_llm_calls=99,
        max_llm_tokens=1,
    )
    with pytest.raises(AssertionError, match="prompt growth, not a loop"):
        journey.finish()


def test_journey_finish_reports_an_exhausted_budget(tmp_path: Path) -> None:
    """``finish()`` must name the budget as the cause, not just fail somewhere."""
    from tests._harness.journey import Journey

    store = CassetteStore(tmp_path)
    body = _chat_body("loop")
    store.put(record_for(body, streamed_response("ok")))
    proxy = CassetteProxy(store, mode=REPLAY, max_calls=1)
    proxy.handle_chat(body)
    proxy.handle_chat(body)

    journey = Journey(
        journey_id="budget-probe",
        proxy=proxy,
        daemon=_LogOnlyDaemon(),
        max_llm_calls=1,
    )
    with pytest.raises(AssertionError, match="provider-call budget"):
        journey.finish()


class _LogOnlyDaemon:
    """Minimal stand-in for the daemon handle a Journey holds.

    Only ``tail_log`` is reachable from the paths under test here; starting a real
    subprocess to assert a budget message would be the kind of cost the budget
    itself exists to avoid.
    """

    def tail_log(self, limit: int = 60) -> str:
        """Return an empty log; no daemon was started for this probe."""
        return "(no daemon)"


# ── Mode awareness ───────────────────────────────────────────────


def test_forwarding_modes_ignore_the_cassette_store(tmp_path: Path) -> None:
    """Live and record must reach the provider even when a recording exists.

    Serving a stored answer in a forwarding mode would make the live lane assert
    against recordings — exactly the drift it is there to detect. It is also why
    a journey asserting on injected failures cannot run live: the injection lives
    in the store, which these modes bypass.
    """
    from tests._harness.cassette_proxy import _FORWARD_MODES

    assert set(_FORWARD_MODES) == {"record", "live"}

    store = CassetteStore(tmp_path)
    body = _chat_body("stored")
    store.put(record_for(body, streamed_response("from the store")))

    proxy = CassetteProxy(
        store,
        mode="live",
        upstream_base_url="http://127.0.0.1:1/v1",
        upstream_api_key="k",
    )
    # Port 1 refuses connections, so a forwarded call raises rather than quietly
    # returning the stored response.
    with pytest.raises(Exception):
        proxy.handle_chat(body)


# ── Daemon environment hermeticity ───────────────────────────────────────


def test_hermetic_env_drops_inherited_leapflow_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A developer's real credentials and data dir must never reach the daemon.

    Without this, a journey would read and write the user's actual profile and
    could spend real tokens while claiming to run offline.
    """
    monkeypatch.setenv("LEAPFLOW_LLM_API_KEY", "sk-real-user-key")
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(Path.home() / ".leapflow"))
    monkeypatch.setenv("LEAPFLOW_GATEWAY_MANIFEST", "/somewhere/real.yaml")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    env = hermetic_env(
        data_dir=tmp_path / "data",
        profile="default",
        llm_base_url="http://127.0.0.1:1/v1",
    )

    assert env["LEAPFLOW_LLM_API_KEY"] == "cassette-key"
    assert env["LEAPFLOW_DATA_DIR"] == str(tmp_path / "data")
    assert "LEAPFLOW_GATEWAY_MANIFEST" not in env, "inherited LEAPFLOW_* must be stripped"
    assert env["PATH"], "non-LeapFlow environment must be preserved"


def test_hermetic_env_points_every_provider_at_the_proxy(tmp_path: Path) -> None:
    """Aux and VLM clients must not be able to reach the network independently."""
    env = hermetic_env(
        data_dir=tmp_path,
        profile="default",
        llm_base_url="http://127.0.0.1:9/v1",
    )
    for key in ("LEAPFLOW_LLM_BASE_URL", "LEAPFLOW_LLM_AUX_BASE_URL", "LEAPFLOW_VLM_BASE_URL"):
        assert env[key] == "http://127.0.0.1:9/v1"


def test_hermetic_env_forces_mock_host(tmp_path: Path) -> None:
    """OS-host mocking is the one legitimate mock: CI has no macOS perception."""
    env = hermetic_env(data_dir=tmp_path, profile="default", llm_base_url="http://x/v1")
    assert env["LEAPFLOW_MOCK_HOST"] == "1"


# ════════════════════════════════════════════════════════════════
# The derived fixture is a contract, not an inventory
# ════════════════════════════════════════════════════════════════


def test_the_shape_fixture_holds_no_volatile_inventory() -> None:
    """``--check`` may only fail on provider drift, never on corpus growth.

    The file used to record how many exchanges it was distilled from, and ``--check``
    compared the whole rendered text. Adding a journey therefore turned CI red with a
    diff whose shapes were byte-identical -- ``37`` versus ``200`` -- so the gate
    stopped meaning "a provider changed" and started meaning "somebody added a test".
    A gate that fires on unrelated growth as loudly as on real drift gets ignored,
    and then the drift it exists for goes unnoticed too.
    """
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests" / "_fixtures" / "llm_responses" / "response_shapes.json"
    )
    stored = json.loads(fixture.read_text(encoding="utf-8"))
    volatile = [key for key in stored if "seen" in key or "count" in key or "total" in key]
    assert not volatile, (
        f"{volatile} vary with the size of the corpus, so --check would fail on "
        "changes that are not provider drift"
    )
    assert "completion_shapes" in stored, "the contract itself must still be there"


def test_the_shape_fixture_is_in_sync_with_the_stored_exchanges() -> None:
    """The gate's own promise: run it, and it must be green on a clean tree.

    Asserted here rather than left to CI, because a check nobody can run locally is a
    check that gets fixed by deleting it.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/sync_fixtures.py", "--check"],
        cwd=repo, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"sync_fixtures --check failed:\n{result.stdout}\n{result.stderr}"
    )
