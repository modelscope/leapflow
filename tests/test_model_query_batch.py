# Copyright (c) Alibaba, Inc. and its affiliates.
"""Serial advisory batches through the ordinary provider, consent and plugin seams."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from leapflow.domain.events import RecordScope
from leapflow.engine.context.result_budget import fit_result_to_budget
from leapflow.llm.services.config import parse_services
from leapflow.llm.services.telemetry import capture_inference
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context
from test_model_services import declaration, recorded_response
from test_model_tools import Human, compose


def inputs(count=3):
    return [{"id": f"D{index + 1}", "input": {
        "goal": "choose an eligible alternative", "candidates": [{"id": "a", "description": "eligible"}]
    }} for index in range(count)]


def bind(tmp_path, human=None, session="own"):
    return set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id=session, task_id="task",
        orchestrator=ApprovalOrchestrator(human or Human())))


@pytest.mark.asyncio
async def test_serial_batch_reuses_one_client_with_independent_accounting(tmp_path):
    owner, registry, _, plugin, requests = compose({"selected_id": "a"},
        limits={"max_inflight": 1, "max_queue": 0})
    clients = []
    original = owner._client_factory

    def construct(spec):
        client = original(spec)
        clients.append(client)
        return client

    owner._client_factory = construct
    human = Human()
    token = bind(tmp_path, human)
    try:
        with capture_inference(scope=RecordScope("", "", "")) as capture:
            result = await registry.tool_handlers["model_query_batch"]("specialist", inputs())
        assert result["ok"] is True and list(result["results"]) == ["D1", "D2", "D3"]
        assert len(clients) == 1 and clients[0].is_closed
        assert len(requests) == len(human.requests) == 3
        records = [item["consultation"] for item in result["results"].values()]
        assert len({item["call_id"] for item in records}) == 3
        assert len({item["inference_call_ids"][0] for item in records}) == 3
        assert all(item["scope"]["session_id"] == "own" and item["status"] == "eligible" for item in records)
        assert all(item["usage"]["specialist"]["calls"] == 1 for item in records)
        assert not owner._handles and not owner._generations and not plugin._behavior._calls
        assert not capture.attempts  # Foreign scope does not inherit the caller's records.
        compact = json.loads(fit_result_to_budget(result, 2500))
        assert set(compact["results"]) == {"D1", "D2", "D3"}
        for item in compact["results"].values():
            assert item["result"]["selected_id"] == "a"
            assert item["provenance"]["physical_authority"] is False
            assert item["provenance"]["scores_calibrated"] is False
        assert any(tool.name == "model_query_batch" for tool in plugin.tools)
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("requests", [[], inputs(9), inputs(1) * 2,
    [{"id": "x", "input": {}, "unexpected": True}], [{"id": "", "input": {}}]])
async def test_invalid_batch_never_acquires_or_authorizes(tmp_path, requests):
    owner, registry, _, _, sent = compose({"selected_id": "a"})
    accessed = []
    owner._client_factory = lambda spec: accessed.append(spec)
    human = Human()
    token = bind(tmp_path, human)
    try:
        result = await registry.tool_handlers["model_query_batch"]("specialist", requests)
        assert result["failure_code"] == "model_invalid_request"
        assert not accessed and not sent and not human.requests
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_aggregate_byte_budget_precedes_consent(tmp_path):
    owner, registry, _, _, sent = compose({"selected_id": "a"}, limits={"max_request_bytes": 2048})
    human = Human()
    token = bind(tmp_path, human)
    data = inputs()
    data[-1]["input"]["goal"] = "x" * 2048
    try:
        result = await registry.tool_handlers["model_query_batch"]("specialist", data)
        assert result["failure_code"] == "model_invalid_request"
        assert not sent and not human.requests and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_snapshot_and_generation_stay_pinned_during_consent(tmp_path):
    owner, registry, _, _, sent = compose({"selected_id": "a"})
    data = inputs()

    class ChangingHuman(Human):
        async def request_approval(self, request):
            data[-1]["input"]["candidates"] = [{"id": "b", "description": "changed"}]
            owner.publish(parse_services({"specialist": declaration(model="new-model", task="prediction")}))
            return await super().request_approval(request)

    token = bind(tmp_path, ChangingHuman())
    try:
        result = await registry.tool_handlers["model_query_batch"]("specialist", data)
        assert result["ok"] is True
        assert all(item["task"] == "decision" for item in result["results"].values())
        assert all(item["provenance"]["eligible_candidate_ids"] == ["a"] for item in result["results"].values())
        assert all(request["model"] == "specialist" for request in sent)
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["denial", "invalid_output", "revocation", "final_cleanup"])
async def test_batch_refusal_never_delivers_partial_advice(tmp_path, monkeypatch, mode):
    owner, registry, _, _, sent = compose({"selected_id": "a"})

    class RejectingHuman(Human):
        async def request_approval(self, request):
            if mode == "denial" and self.requests:
                self.allow = False
            if mode == "revocation" and self.requests:
                owner.publish(parse_services({"specialist": declaration(enabled=False)}))
            return await super().request_approval(request)

    if mode == "invalid_output":
        async def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=recorded_response({"selected_id": "a" if len(sent) == 1 else "foreign"}))
        owner._client_factory = lambda spec: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    if mode == "final_cleanup":
        original = owner._release

        async def release(handle):
            await original(handle)
            owner.publish(parse_services({"specialist": declaration(enabled=False)}))
            owner.publish(parse_services({"specialist": declaration()}))

        monkeypatch.setattr(owner, "_release", release)
    token = bind(tmp_path, RejectingHuman())
    try:
        result = await registry.tool_handlers["model_query_batch"]("specialist", inputs())
        assert result["ok"] is False and "results" not in result
        assert len(sent) == {"denial": 1, "invalid_output": 2, "revocation": 1, "final_cleanup": 3}[mode]
        if mode in {"denial", "invalid_output"}:
            assert result["completed"] == 1 and result["failed_id"] == "D2"
            assert len(result["consultations"]) == 1
        assert not owner._handles and not owner._generations
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("dispose", [False, True])
async def test_cancellation_or_disposal_closes_batch_and_never_runs_later_items(tmp_path, dispose):
    owner, registry, scoped, plugin, sent = compose({"selected_id": "a"})
    entered = asyncio.Event()

    async def respond(request):
        sent.append(request)
        entered.set()
        await asyncio.Event().wait()

    owner._client_factory = lambda spec: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    token = bind(tmp_path)
    try:
        task = asyncio.create_task(registry.tool_handlers["model_query_batch"]("specialist", inputs()))
        await asyncio.wait_for(entered.wait(), 2)
        if dispose:
            fiber = scoped.dispose_plugin(plugin.plugin_id)
            await fiber.scope.async_dispose()
            assert "model_query_batch" not in registry.tool_handlers
        else:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(sent) == 1 and not owner._handles and not owner._generations
        assert not plugin._behavior._calls
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["foreign_owner", "expired"])
async def test_batch_checks_each_artifact_before_consent_and_stops(tmp_path, monkeypatch, invalid):
    from leapflow.cache.manager import CacheManager
    from leapflow.config import load_config
    from test_model_tools import session_artifact

    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    first = session_artifact(cache, tmp_path, "own", "first.jpg", b"\xff\xd8\xfffixture")
    other = session_artifact(cache, tmp_path, "foreign" if invalid == "foreign_owner" else "own",
                             "other.jpg", b"\xff\xd8\xfffixture", expires_at=0 if invalid == "expired" else None)
    owner, registry, _, _, sent = compose({"selected_id": "a"}, processing_location="local")
    registry.bind_runtime(settings=settings)
    human = Human()
    token = bind(tmp_path, human)
    requests = inputs()
    requests[0]["artifacts"] = {"image": first.entry_id}
    requests[1]["artifacts"] = {"image": other.entry_id}
    try:
        result = await registry.tool_handlers["model_query_batch"]("specialist", requests)
        assert result["ok"] is False and "results" not in result
        assert result["completed"] == 1 and result["failed_id"] == "D2"
        assert len(sent) == len(human.requests) == 1
        assert not owner._handles and not owner._generations
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_two_concurrent_sessions_do_not_share_consultation_identity(tmp_path):
    owner, registry, _, _, _ = compose({"selected_id": "a"}, limits={"max_inflight": 2, "max_queue": 0})

    async def run(session):
        token = bind(tmp_path / session, session=session)
        try:
            return await registry.tool_handlers["model_query_batch"]("specialist", inputs())
        finally:
            reset_tool_context(token)

    try:
        first, second = await asyncio.gather(run("left"), run("right"))
        for result, session in ((first, "left"), (second, "right")):
            assert result["ok"] is True
            assert all(item["consultation"]["scope"]["session_id"] == session for item in result["results"].values())
        left = {item["consultation"]["call_id"] for item in first["results"].values()}
        right = {item["consultation"]["call_id"] for item in second["results"].values()}
        assert not left & right
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_competing_session_is_refused_without_queue_or_extra_requests(tmp_path):
    owner, registry, _, _, sent = compose({"selected_id": "a"}, limits={"max_inflight": 1, "max_queue": 0})
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        sent.append(json.loads(request.content))
        entered.set()
        await release.wait()
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    owner._client_factory = lambda spec: httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def run(session):
        token = bind(tmp_path / session, session=session)
        try:
            return await registry.tool_handlers["model_query_batch"]("specialist", inputs())
        finally:
            reset_tool_context(token)

    first = asyncio.create_task(run("first"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = await run("second")
        assert second["ok"] is False and second["failure_code"] == "model_busy"
        assert "results" not in second and len(sent) == 1
        release.set()
        assert (await first)["ok"] is True and len(sent) == 3
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
async def test_batch_through_primary_loop_retains_all_owned_durable_receipts(tmp_path, monkeypatch):
    from leapflow.llm.base import LLMChatResponse, ToolCallInfo
    from leapflow.tools import shell_tools
    from test_model_tools import unified_model_engine

    class Primary:
        model = "offline-primary"

        def __init__(self):
            self.calls = 0

        async def achat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMChatResponse(content="", tool_calls=[ToolCallInfo(id="batch", name="model_query_batch",
                    arguments={"service_id": "specialist", "requests": inputs()})],
                    usage={"prompt_tokens": 20, "completion_tokens": 5})
            tool = json.loads(next(m["content"] for m in reversed(messages) if m.get("role") == "tool"))
            assert set(tool["results"]) == {"D1", "D2", "D3"}
            return LLMChatResponse(content="Three independent advisory selections consumed; no motion.",
                                   usage={"prompt_tokens": 20, "completion_tokens": 5})

        async def achat_stream(self, messages, **kwargs):
            if False:
                yield ""

    owner, registry, _, _, sent = compose({"selected_id": "a"})
    primary = Primary()
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, primary)
    monkeypatch.setattr(shell_tools, "_approval_gate", ApprovalOrchestrator(Human()))
    try:
        await engine.run("/data_analysis Choose among the supplied safe alternatives.")
        rows = store.get_messages("own")
        receipts = [receipt for row in rows for receipt in (row.metadata or {}).get("collaboration_signal_receipts", [])]
        records = [json.loads(r["payload_json"])["value"] for r in receipts
                   if r["admitted"] and r["lineage_eligible"]]
        attempts = [r for r in records if "attempt_id" in r]
        consultations = [r for r in records if "inference_call_ids" in r]
        assert primary.calls == 2 and len(sent) == 3
        assert len([r for r in attempts if r["source"] == "primary"]) == 2
        assert len([r for r in attempts if r["source"] == "specialist"]) == 3
        assert len(consultations) == 3 and len({r["call_id"] for r in consultations}) == 3
        assert all(r["scope"]["session_id"] == "own" for r in records)
        assert not owner._handles
    finally:
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
async def test_batch_requires_session_and_disappears_with_withdrawn_services(tmp_path):
    owner, registry, _, plugin, sent = compose({"selected_id": "a"})
    try:
        result = await registry.tool_handlers["model_query_batch"]("specialist", inputs())
        assert result["failure_code"] == "model_unavailable" and not sent
        plugin.bind_runtime(model_services=None)
        assert not plugin.tools
    finally:
        await owner.aclose()
