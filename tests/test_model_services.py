# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic model-service contracts through real registries and provider constructors."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml

from leapflow.config import load_config
from leapflow.config_loader import load_config_bundle
from leapflow.config_service import ConfigService
from leapflow.llm.base import LLMChatResponse, ToolCallInfo
from leapflow.llm.provider_registry import LLMProviderRegistry
from leapflow.llm.services.codec import JsonCodec, image_block
from leapflow.llm.services.config import parse_services
from leapflow.llm.services.http_client import BoundedTransport
from leapflow.llm.services.protocol import ActiveCallBudget, ServiceError
from leapflow.llm.services.runtime import ModelServiceRuntime
from leapflow.llm.services.telemetry import capture_inference
from leapflow.domain.events import RecordScope
from leapflow.engine.turn_usage import TurnUsageTracker

SCHEMA = {"type": "object", "required": ["selected_id"],
          "properties": {"selected_id": {"type": "string"}}, "additionalProperties": False}


def declaration(**changes):
    return {"schema_version": 1, "provider": "openai", "base_url": "http://localhost:8080/v1",
            "model": "specialist", "task": "decision", "auth": {"mode": "none"}, **changes}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "invalid", "missing_usage", "cancelled"])
async def test_attempt_usage_survives_validation_failure_and_cancellation(mode):
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    entered = asyncio.Event()

    async def respond(request):
        entered.set()
        if mode == "cancelled":
            await asyncio.Event().wait()
        body = recorded_response({"selected_id": "a"} if mode != "invalid" else {"extra": 1})
        if mode == "missing_usage":
            body.pop("usage", None)
        return httpx.Response(200, json=body)

    owner = ModelServiceRuntime(registry, parse_services({"a": declaration()}),
        client_factory=lambda spec: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    tracker = TurnUsageTracker()
    scope = RecordScope("profile", "workspace", "session")
    handle = await owner.acquire("a")
    try:
        with capture_inference(scope=scope, task_id="task", tool_call_id="tool", sink=tracker) as capture:
            running = asyncio.create_task(handle.invoke({}, SCHEMA))
            await asyncio.wait_for(entered.wait(), 2)
            if mode == "cancelled":
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            elif mode == "invalid":
                with pytest.raises(ServiceError) as error:
                    await running
                assert error.value.failure.code == "model_invalid_output"
            else:
                result = await running
                assert result.attempts == tuple(capture.attempts)
                assert result.call_id == result.attempts[0].call_id
                assert set(json.loads(result.timing_json)) == {"queue_ms", "encoding_ms", "validation_ms"}
        assert len(capture.attempts) == 1
        attempt = capture.attempts[0]
        assert attempt.scope == scope and attempt.task_id == "task" and attempt.tool_call_id == "tool"
        assert attempt.service_id == "a" and attempt.generation == handle.generation
        assert attempt.contract_revision == handle.contract.revision
        assert len(attempt.declaration_digest) == 64
        assert attempt.status == ("cancelled" if mode == "cancelled" else "returned")
        assert {"queue_ms", "encoding_ms", "network_inference_ms"} <= capture.phase_timings.keys()
        assert capture.phase_timings["network_inference_ms"] == attempt.elapsed_ms
        assert ("validation_ms" in capture.phase_timings) is (mode != "cancelled")
        combined = tracker.combined_usage()
        assert combined["specialist"]["attempts"] == combined["combined"]["attempts"] == 1
        assert combined["primary"]["attempts"] == tracker.summary().api_calls == 0
        assert combined["specialist"]["unknown_usage_attempts"] == (1 if mode in {"missing_usage", "cancelled"} else 0)
        tracker.record_inference(attempt)
        assert tracker.combined_usage() == combined
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_concurrent_capture_and_broken_sink_do_not_share_or_fail_results():
    registry = LLMProviderRegistry()
    registry.discover_builtin()

    async def respond(request):
        await asyncio.sleep(.005)
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    owner = ModelServiceRuntime(registry, parse_services({"a": declaration(), "b": declaration()}),
        client_factory=lambda spec: httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    class BrokenSink:
        def record_inference(self, attempt):
            raise AttributeError("context telemetry defect")

    async def invoke(service, sink):
        handle = await owner.acquire(service)
        scope = RecordScope("profile", f"workspace-{service}", f"session-{service}")
        with capture_inference(scope=scope, sink=sink) as capture:
            result = await handle.invoke({}, SCHEMA)
        await handle.aclose()
        return result, capture

    tracker = TurnUsageTracker()
    try:
        (first, left), (second, right) = await asyncio.gather(invoke("a", tracker), invoke("b", BrokenSink()))
        assert first.payload() == second.payload() == {"selected_id": "a"}
        assert len(left.attempts) == len(right.attempts) == 1
        assert left.attempts[0].scope.session_id == "session-a"
        assert right.attempts[0].scope.session_id == "session-b"
        assert first.call_id != second.call_id
        assert tracker.inference_records() == tuple(left.attempts)
    finally:
        await owner.aclose()


def recorded_response(payload):
    cassette = Path(__file__).parent / "_fixtures/cassettes/r1_conversation/cassette-model-a9f2f46c8b5fb77c.cassette.json"
    body = json.loads(json.loads(cassette.read_text())["responses"][0]["body"])
    body["choices"][0]["message"]["content"] = json.dumps(payload)
    return body


@pytest.mark.parametrize("change", [
    {"base_url": "http://user:password@localhost/v1"}, {"base_url": "http://example.org/v1"},
    {"auth": {"mode": "secret", "ref": "inline-token"}}, {"auth": {"mode": "secret", "ref": "secret://profile/"}},
    {"limits": {"deadline_s": float("nan")}}, {"limits": {"max_queue": -1}},
    {"limits": {"attempts": True}}, {"request_options": {"temperature": True}},
    {"request_options": {"top_p": 2}}, {"output": "json_schema", "provider": "anthropic"},
    {"identity": {"mode": "server_reported"}}, {"api_key": "plaintext"},
    *({"enabled": value} for value in ("true", "false", 0, 1, None, [], {})),
])
def test_invalid_declaration(change):
    with pytest.raises(ValueError):
        parse_services({"test": declaration(**change)})


@pytest.mark.parametrize("enabled", [None, True, False])
def test_enabled_roundtrip_preserves_omission_and_strict_boolean(enabled):
    value = declaration(**({} if enabled is None else {"enabled": enabled}))
    spec = parse_services({"a": value})[0]
    assert spec.enabled is (True if enabled is None else enabled)
    assert spec.declaration() == {**value, "media": {"max_images": 16, "max_videos": 0}}
    assert parse_services({"a": spec.declaration()}) == (spec,)


@pytest.mark.parametrize("media", [{"max_images": -1}, {"max_videos": True},
    {"max_videos": 5}, {"audio": 1}, {"max_images": 1.5}])
def test_invalid_media_declaration(media):
    with pytest.raises(ValueError):
        parse_services({"test": declaration(media=media)})


def test_media_codec_and_capacity():
    import base64
    from dataclasses import replace
    from leapflow.domain.model_service import ServiceMedia
    from leapflow.llm.services.codec import media_block
    from leapflow.tools.model_tools import _media_type, check_media_capacity

    raw = b"\x00\x00\x00\x18ftyp" + bytes(40)
    assert _media_type(raw) == "video/mp4"
    assert _media_type(b"\x1aE\xdf\xa3" + bytes(10)) == "video/webm"
    item = {"media_type": "video/mp4", "data": base64.b64encode(raw).decode()}
    assert media_block(item, "openai")["video_url"]["url"].startswith("data:video/mp4;base64,")
    with pytest.raises(ServiceError) as error:
        media_block(item, "anthropic")
    assert error.value.failure.code == "media_unsupported"
    default = parse_services({"test": declaration()})[0]
    with pytest.raises(ServiceError):
        check_media_capacity(default, [item])
    video_spec = replace(default, media=ServiceMedia(4, 1))
    assert check_media_capacity(video_spec, [item]) == {"image": 0, "video": 1}
    with pytest.raises(ServiceError):
        check_media_capacity(video_spec, [item, item])


def test_v2_state_and_score_identity_bounds():
    from leapflow.world_model.specialist_contract import AdvisoryTaskContract
    contract = AdvisoryTaskContract("decision")
    data = {"goal": "choose", "candidates": [{"id": "a", "description": "eligible"}],
            "state": {"jaw": {"width_mm": 18}}}
    contract.validate_input(data)
    contract.validate_result(data, {"selected_id": "a", "candidate_scores": [{"id": "a", "score": 0.9}]})
    assert contract.revision == "advisory.v2"
    for state in ({"large": "x" * 8192}, {"a": {"b": {"c": {"d": 1}}}}, {"nan": float("nan")}):
        with pytest.raises(ValueError):
            contract.validate_input({**data, "state": state})
    for scores in ([{"id": "unknown", "score": 0.1}], [{"id": "a", "score": 0.1}] * 2):
        with pytest.raises(ValueError):
            contract.validate_result(data, {"selected_id": "a", "candidate_scores": scores})


def test_detached_immutable_catalog():
    original = declaration()
    spec = parse_services({"test": original})[0]
    original["model"] = "changed"
    spec.declaration()["auth"]["mode"] = "changed"
    assert spec.model == "specialist"
    assert spec.declaration()["auth"] == {"mode": "none"}
    with pytest.raises(ValueError):
        parse_services({1: declaration(), "test": declaration()})


@pytest.mark.parametrize("content,finish", [
    ('{"selected_id":"a","selected_id":"b"}', "stop"), ('```json\n{}\n```', "stop"),
    ('{"selected_id":"a"}', "length"), ('{"selected_id":NaN}', "stop"),
    ('{"selected_id":"a","extra":1}', "stop"), ('[]', "stop"),
])
def test_codec_strict(content, finish):
    with pytest.raises(ValueError):
        JsonCodec().decode(parse_services({"test": declaration()})[0],
                           LLMChatResponse(content, finish_reason=finish), SCHEMA)


@pytest.mark.parametrize("depth", [33, 1200])
@pytest.mark.parametrize("output", ["json_text", "tool_call"])
def test_codec_rejects_deep_results_before_canonicalization(depth, output):
    spec = parse_services({"test": declaration(output=output)})[0]
    content = '{"nested":' + '[' * depth + '0' + ']' * depth + '}'
    response = (LLMChatResponse("", tool_calls=[ToolCallInfo(
        "x", "submit_result", {"nested": []}, content)]) if output == "tool_call" else LLMChatResponse(content))
    with pytest.raises(ValueError, match="nesting limit"):
        JsonCodec().decode(spec, response, {"type": "object"})


def test_raw_tool_arguments_and_images():
    spec = parse_services({"test": declaration(output="tool_call")})[0]
    call = ToolCallInfo("x", "submit_result", {"selected_id": "b"},
                        '{"selected_id":"a","selected_id":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        JsonCodec().decode(spec, LLMChatResponse("", tool_calls=[call]), SCHEMA)
    image = {"media_type": "image/png", "data": "eA=="}
    assert image_block(image, "anthropic")["source"]["type"] == "base64"
    assert image_block(image, "openai")["image_url"]["url"].startswith("data:image/png")
    with pytest.raises(ServiceError):
        JsonCodec().encode(spec, {"_media": [image]}, SCHEMA)


@pytest.mark.asyncio
async def test_real_provider_isolation_reload_and_close():
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    calls, clients = [], []

    async def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    def client_factory(spec):
        client = httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond)))
        clients.append(client)
        return client

    owner = ModelServiceRuntime(registry, parse_services({"a": declaration(), "b": declaration(model="second")}),
                                client_factory=client_factory)
    primary = registry.get_provider("openai", {"api_key": "primary-secret", "base_url": "http://localhost:9999",
                                              "model": "primary"})
    first, second = await owner.acquire("a"), await owner.acquire("b")
    assert first._generation.provider is not second._generation.provider
    assert first._generation.provider is not primary
    assert first._generation.provider._max_retries == 1
    assert first._generation.provider._profile.name == "generic"
    assert (await first.invoke({"goal": "choose"}, SCHEMA)).payload() == {"selected_id": "a"}
    owner.publish(parse_services({"a": declaration(model="new-model")}))
    new = await owner.acquire("a")
    assert new.generation != first.generation
    assert first.spec.model == "specialist"
    await first.invoke({"goal": "old snapshot"}, SCHEMA)
    assert calls[-1]["model"] == "specialist"
    assert registry.get_provider("openai") is primary
    await first.aclose()
    assert clients[0].is_closed
    await owner.aclose()
    assert all(client.is_closed for client in clients)
    assert not owner._handles and not owner._generations
    await primary.aclose()


@pytest.mark.asyncio
async def test_disabled_acquisition_has_zero_secret_client_or_provider_access():
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    accessed = []
    owner = ModelServiceRuntime(registry, parse_services({"a": declaration(enabled=False,
        auth={"mode": "secret", "ref": "secret://profile/models/a"})}),
        resolve_secret=lambda ref: accessed.append(ref), client_factory=lambda spec: accessed.append(spec))
    try:
        status = owner.describe("a")["services"][0]
        assert status["configured"] and status["provider_available"] and not status["enabled"]
        assert not status["usable"] and status["unavailable_reason"] == "disabled"
        with pytest.raises(ServiceError) as refused:
            await owner.acquire("a")
        assert refused.value.failure.code == "model_service_disabled"
        assert not refused.value.failure.recoverable
        assert not accessed and not owner._handles and not registry._instances
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
async def test_revocation_discards_queued_and_inflight_results_and_never_resurrects(remove):
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def respond(request):
        requests.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    spec = declaration(limits={"max_inflight": 1, "max_queue": 1})
    owner = ModelServiceRuntime(registry, parse_services({"a": spec}),
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    first, second = await owner.acquire("a"), await owner.acquire("a")
    try:
        running = asyncio.create_task(first.invoke({}, SCHEMA))
        await asyncio.wait_for(entered.wait(), 2)
        queued = asyncio.create_task(second.invoke({}, SCHEMA))
        await asyncio.sleep(.01)
        assert first._generation.admitted == 2
        owner.publish(() if remove else parse_services({"a": {**spec, "enabled": False}}))
        owner.publish(parse_services({"a": spec}))
        release.set()
        for call in (running, queued):
            with pytest.raises(ServiceError) as refused:
                await call
            assert refused.value.failure.code == ("model_capability_removed" if remove else "model_service_disabled")
            assert not refused.value.failure.recoverable
        assert len(requests) == 1 and first._generation.admitted == 0
        fresh = await owner.acquire("a")
        assert fresh.generation != first.generation
        assert (await fresh.invoke({}, SCHEMA)).payload() == {"selected_id": "a"}
        with pytest.raises(ServiceError):
            first.check_admission()
    finally:
        release.set()
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["encode", "decode", "validate"])
async def test_revocation_during_threaded_codec_or_validation(boundary):
    import threading

    registry = LLMProviderRegistry()
    registry.discover_builtin()
    entered, release = threading.Event(), threading.Event()
    requests = []

    def wait():
        entered.set()
        assert release.wait(5)

    class WaitingCodec(JsonCodec):
        def encode(self, *args):
            if boundary == "encode":
                wait()
            return super().encode(*args)

        def decode(self, *args):
            if boundary == "decode":
                wait()
            return super().decode(*args)

    async def respond(request):
        requests.append(request)
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    spec = declaration(codec="waiting.v1")
    owner = ModelServiceRuntime(registry, parse_services({"a": spec}),
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    owner.register_codec("waiting.v1", WaitingCodec())
    handle = await owner.acquire("a")
    try:
        pending = asyncio.create_task(handle.invoke({}, SCHEMA, validate=lambda value: wait() if boundary == "validate" else None))
        assert await asyncio.to_thread(entered.wait, 3)
        owner.publish(parse_services({"a": {**spec, "enabled": False}}))
        owner.publish(parse_services({"a": spec}))
        release.set()
        with pytest.raises(ServiceError) as refused:
            await pending
        assert refused.value.failure.code == "model_service_disabled"
        assert len(requests) == (0 if boundary == "encode" else 1)
    finally:
        release.set()
        await owner.aclose()


@pytest.mark.asyncio
async def test_result_eligibility_survives_cleanup_without_resurrecting_admission():
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    owner = ModelServiceRuntime(registry, parse_services({"a": declaration()}))
    handle = await owner.acquire("a")
    await handle.aclose()
    assert not owner._generations
    handle.check_result_eligibility()
    with pytest.raises(ServiceError):
        handle.check_admission()
    owner.publish(parse_services({"a": declaration(enabled=False)}))
    owner.publish(parse_services({"a": declaration()}))
    with pytest.raises(ServiceError) as refused:
        handle.check_result_eligibility()
    assert refused.value.failure.code == "model_service_disabled"
    await owner.aclose()


@pytest.mark.asyncio
async def test_total_deadline_busy_cancel_and_retry_owner():
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    entered = asyncio.Event()
    attempts = 0

    async def respond(request):
        nonlocal attempts
        attempts += 1
        entered.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    owner = ModelServiceRuntime(registry, parse_services({"a": declaration(limits={"deadline_s": .05})}),
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    handle = await owner.acquire("a")
    pending = asyncio.create_task(handle.invoke({}, SCHEMA))
    await entered.wait()
    with pytest.raises(ServiceError) as busy:
        await handle.invoke({}, SCHEMA)
    assert busy.value.failure.code == "model_busy"
    with pytest.raises(ServiceError) as timeout:
        await pending
    assert timeout.value.failure.code == "model_timeout"
    assert attempts == 1 and handle._generation.admitted == 0
    pending = asyncio.create_task(handle.invoke({}, SCHEMA))
    await asyncio.sleep(.01)
    await handle.aclose()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,headers,body,code", [
    (302, {"location": "http://other/"}, b"", "model_redirect"),
    (200, {"content-encoding": "gzip"}, b"x", "model_response_encoding"),
    (200, {}, b"x" * 101, "model_response_too_large"),
])
async def test_bounded_http(status, headers, body, code):
    spec = parse_services({"a": declaration(limits={"max_response_bytes": 100})})[0]
    async with httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(
        lambda request: httpx.Response(status, headers=headers, content=body)))) as client:
        with pytest.raises(ServiceError) as result:
            await client.get(spec.base_url)
        assert result.value.failure.code == code
        with pytest.raises(ServiceError) as result:
            await client.get("http://other/v1")
        assert result.value.failure.category == "security"


def test_profile_only_structured_config_and_atomic_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    settings = load_config()
    service = ConfigService(settings)
    service.set_model_service("a", declaration(auth={"mode": "secret", "ref": "secret://profile/models/a"}))
    service.set_model_service("b", declaration(model="second"))
    service.set("llm.model", "keep-primary")
    before = settings.profile_layout.llm_config_path.read_bytes()
    with pytest.raises(ValueError):
        service.set_model_service("a", declaration(api_key="plaintext"))
    assert settings.profile_layout.llm_config_path.read_bytes() == before
    with pytest.raises(ValueError):
        service.set("llm.services", {}, scope="workspace")
    workspace_file = settings.layout.workspace_config_path(settings.workspace_root)
    workspace_file.parent.mkdir(parents=True)
    workspace_file.write_text(yaml.safe_dump({"llm": {"services": {"evil": declaration()}}}))
    bundle = load_config_bundle(settings.layout, settings.profile_layout, settings.workspace_root)
    assert set(bundle.values["llm"]["services"]) == {"a", "b"}
    assert not any(key.startswith("LEAPFLOW_LLM_SERVICES") for key in bundle.env)
    updated = load_config()
    assert updated.llm_services[0].auth_ref == "secret://profile/models/a"
    assert updated.llm_model == "keep-primary"
    assert ConfigService(updated).describe("llm.services").value_type == "dict"
    assert "secret://profile/models/a" in ConfigService(updated).get("llm.services").value
    service.unset_model_service("a")
    assert set(service.model_services()) == {"b"}


@pytest.mark.asyncio
async def test_cold_settings_composition_discovers_once_and_preserves_removal(tmp_path, monkeypatch):
    from leapflow.llm import provider_registry

    # Isolate the process singleton, retaining real discovery, fibers and providers.
    registry = LLMProviderRegistry()
    monkeypatch.setattr(provider_registry, "_default_registry", registry)
    monkeypatch.setattr(provider_registry, "_scoped_default_registry", None)
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    settings = load_config()
    service = ConfigService(settings)
    service.set_model_service("a", declaration())
    settings = load_config()
    owners = [ModelServiceRuntime.from_settings(settings)]
    try:
        assert owners[0].describe("a")["services"][0]["available"]
        scoped = provider_registry.get_scoped_default_registry()
        assert scoped.get_fiber("openai") is not None
        version = registry.version
        owners.append(ModelServiceRuntime.from_settings(settings))
        handle = await owners[0].acquire("a")
        assert handle.spec.model == "specialist"
        await handle.aclose()
        assert registry.version == version and not registry._instances
        for fiber in scoped.fibers.values():
            fiber.begin_unload()
            fiber.dispose()
        assert not registry.list_available()
        owners.append(ModelServiceRuntime.from_settings(settings))
        checked = service.check_model_services("a")
        assert not checked["services"][0]["provider_available"]
        assert not checked["services"][0]["endpoint_probed"]
        assert not owners[-1].describe("a")["services"][0]["available"]
        with pytest.raises(ServiceError):
            await owners[-1].acquire("a")
        assert not registry.list_available(), "A local check must not resurrect disposed providers"
    finally:
        await asyncio.gather(*(owner.aclose() for owner in owners))


@pytest.mark.asyncio
async def test_work_budget_is_aggregate_and_cancels_overrun():
    budget = ActiveCallBudget(.08)
    await budget.run(asyncio.sleep(.05))
    with pytest.raises(asyncio.TimeoutError):
        await budget.run(asyncio.sleep(.05))
    assert budget.remaining <= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["provider", "codec"])
async def test_capability_replacement_discards_inflight_response(replacement):
    registry = LLMProviderRegistry()
    registry.discover_builtin()
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    codec_name = "custom.v1" if replacement == "codec" else "json.v1"
    owner = ModelServiceRuntime(registry, parse_services({"a": declaration(codec=codec_name)}),
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    undo = owner.register_codec(codec_name, JsonCodec()) if replacement == "codec" else None
    handle = await owner.acquire("a")
    try:
        pending = asyncio.create_task(handle.invoke({}, SCHEMA))
        await entered.wait()
        if replacement == "provider":
            plugin = registry.get_plugin("openai")
            registry.unregister("openai")
            registry.register(plugin)
        else:
            undo()
            owner.register_codec(codec_name, JsonCodec())
        release.set()
        with pytest.raises(ServiceError) as refused:
            await pending
        assert refused.value.failure.code == "model_capability_removed"
        assert not handle._tasks and handle._generation.admitted == 0
    finally:
        release.set()
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_context_defers_catalog_assembly_until_optional_dependencies_bind(tmp_path, monkeypatch, configured):
    from dataclasses import replace

    import leapflow.plugins as plugins
    from leapflow.cli.context import Context
    from leapflow.hardware.context import Channel, HardwareContext, TransportRef

    monkeypatch.setattr(plugins, "_registry", None)
    monkeypatch.setattr(plugins, "_scoped_registry", None)
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("LEAPFLOW_HARDWARE_ENABLED", "1")
    monkeypatch.setenv("LEAPFLOW_HARDWARE_PROVIDERS", "yaml")
    monkeypatch.setenv("LEAPFLOW_HARDWARE_DEVICES_DIR", str(tmp_path / "devices"))
    settings = load_config()
    if configured:
        settings = replace(settings, llm_services=parse_services({"specialist": declaration()}))
    context = Context(settings, mock_host=True)
    registry = plugins.get_registry()
    try:
        # Reading tool_owners here would assemble the catalog and reproduce the bug.
        assert not registry._assembled
        hardware = context._hardware_registry
        hardware.register_external_contexts("fixture", (HardwareContext(
            device_id="fixture_bench", display_name="Fixture bench",
            transport=TransportRef(kind="simulated"),
            channels=(Channel(channel_id="value", direction="read", quantity="state"),)),))
        hardware.load()
        context._bind_hardware_plugin()
        context._bind_model_services()
        assert {"hw_list", "hw_describe", "hw_read"} <= registry.tool_handlers.keys()
        assert ("model_query" in registry.tool_handlers) is configured
    finally:
        await context.cleanup()


@pytest.mark.asyncio
async def test_strict_request_uses_nullable_required_wire_fields_and_restores_domain():
    from copy import deepcopy
    from functools import partial
    from leapflow.world_model.specialist_contract import OUTPUT_SCHEMAS, validate_result

    schema = deepcopy(OUTPUT_SCHEMAS["decision"])
    original = deepcopy(schema)
    request = {"goal": "Choose", "candidates": [{"id": "a", "description": "eligible"}]}
    requests = []

    async def respond(http_request):
        body = json.loads(http_request.content)
        requests.append(body)
        wire = body["response_format"]["json_schema"]
        assert wire["strict"] is True
        compiled = wire["schema"]
        assert set(compiled["required"]) == set(compiled["properties"])
        assert compiled["additionalProperties"] is False
        assert compiled["properties"]["ranking"]["anyOf"][1] == {"type": "null"}
        assert compiled["properties"]["model_score"]["anyOf"][1] == {"type": "null"}
        assert "anyOf" not in compiled["properties"]["selected_id"]
        return httpx.Response(200, json=recorded_response({
            "selected_id": "a", "rationale": "eligible", "ranking": None, "model_score": None,
            "candidate_scores": None}))

    providers = LLMProviderRegistry()
    providers.discover_builtin()
    owner = ModelServiceRuntime(providers, parse_services({"a": declaration(output="json_schema")}),
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    try:
        handle = await owner.acquire("a")
        result = await handle.invoke(request, schema, validate=partial(validate_result, "decision", request))
        assert result.payload() == {"selected_id": "a", "rationale": "eligible"}
        assert schema == original and len(requests) == 1
    finally:
        await owner.aclose()


@pytest.mark.parametrize("value", [
    {"selected_id": None, "rationale": "ok", "ranking": None, "model_score": None},
    {"selected_id": "a", "rationale": "ok", "model_score": None},
    {"selected_id": "a", "rationale": "ok", "ranking": [None], "model_score": None},
    {"selected_id": "a", "rationale": "ok", "ranking": None, "model_score": "bad"},
])
def test_strict_null_normalization_never_weakens_domain_validation(value):
    from leapflow.world_model.specialist_contract import OUTPUT_SCHEMAS
    spec = parse_services({"a": declaration(output="json_schema")})[0]
    with pytest.raises(ValueError):
        JsonCodec().decode(spec, LLMChatResponse(json.dumps(value)), OUTPUT_SCHEMAS["decision"])


def test_strict_nested_optional_roundtrip_and_non_strict_null_refusal():
    from leapflow.world_model.specialist_contract import array, obj, text
    schema = obj({"items": array(obj({"name": text(), "note": text()}, ("name",)))}, ("items",))
    spec = parse_services({"a": declaration(output="json_schema")})[0]
    result = {"items": [{"name": "a", "note": None}, {"name": "b", "note": "kept"}]}
    assert JsonCodec().decode(spec, LLMChatResponse(json.dumps(result)), schema) == {
        "items": [{"name": "a"}, {"name": "b", "note": "kept"}]}
    text_spec = parse_services({"a": declaration()})[0]
    with pytest.raises(ValueError):
        JsonCodec().decode(text_spec, LLMChatResponse(json.dumps(result)), schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["prediction", "grounding", "grasp_proposal", "action_chunk"])
async def test_unrepresentable_declared_strict_schema_refused_before_client_or_http(task):
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    accessed = []
    owner = ModelServiceRuntime(providers, parse_services({"a": declaration(task=task, output="json_schema")}),
                                client_factory=lambda spec: accessed.append(spec))
    try:
        status = owner.describe("a")["services"][0]
        assert not status["available"] and status["unavailable_reason"] == "strict_schema_unrepresentable"
        with pytest.raises(ServiceError):
            await owner.acquire("a")
        assert not accessed
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_unrepresentable_custom_strict_schema_refused_before_http():
    from leapflow.world_model.specialist_contract import OUTPUT_SCHEMAS
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    requests = []
    owner = ModelServiceRuntime(providers, parse_services({"a": declaration(output="json_schema")}),
        client_factory=lambda spec: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: requests.append(request))))
    try:
        handle = await owner.acquire("a")
        with pytest.raises(ServiceError) as error:
            await handle.invoke({}, OUTPUT_SCHEMAS["prediction"])
        assert error.value.failure.code == "model_invalid_request" and not requests
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_implementation", [False, True])
async def test_contract_withdrawal_and_reregistration_never_restore_old_admission(same_implementation):
    from leapflow.domain.effect_scope import EffectScope
    from leapflow.llm.services.contracts import TaskContractRegistry
    from leapflow.world_model.specialist_contract import AdvisoryTaskContract
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    contracts = TaskContractRegistry()
    scope = EffectScope("test-contract-plugin")
    contract = AdvisoryTaskContract("decision")
    scope.effect(contracts.register(contract))
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=recorded_response({"selected_id": "a"}))

    owner = ModelServiceRuntime(providers, parse_services({"a": declaration()}), contracts=contracts,
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    try:
        old = await owner.acquire("a")
        pending = asyncio.create_task(old.invoke({}, SCHEMA))
        await asyncio.wait_for(entered.wait(), 2)
        scope.dispose()
        assert not owner.describe("a")["services"][0]["available"]
        owner.register_contract(contract if same_implementation else AdvisoryTaskContract("decision"))
        release.set()
        with pytest.raises(ServiceError) as error:
            await pending
        assert error.value.failure.code == "model_capability_removed"
        fresh = await owner.acquire("a")
        assert fresh.contract.number != old.contract.number and fresh.generation != old.generation
        await old.aclose()
        with pytest.raises(ServiceError):
            old.check_result_eligibility()
        assert (await fresh.invoke({}, SCHEMA)).payload() == {"selected_id": "a"}
    finally:
        release.set()
        await owner.aclose()


@pytest.mark.asyncio
async def test_unknown_structurally_valid_contract_is_unavailable_not_implicitly_advisory():
    from leapflow.llm.services.contracts import TaskContractRegistry
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    specs = parse_services({"a": declaration(task="plugin.reason.v1")})
    owner = ModelServiceRuntime(providers, specs, contracts=TaskContractRegistry())
    try:
        status = owner.describe("a")["services"][0]
        assert status["unavailable_reason"] == "contract_unavailable" and status["classification"] is None
        with pytest.raises(ServiceError):
            await owner.acquire("a")
    finally:
        await owner.aclose()


@pytest.mark.parametrize("task", ["../decision", "decision/action", "x" * 129, "", "1decision"])
def test_task_identifiers_are_structural_not_paths(task):
    with pytest.raises(ValueError):
        parse_services({"a": declaration(task=task)})


def test_anthropic_seed_is_rejected_before_provider_admission():
    with pytest.raises(ValueError, match="Anthropic"):
        parse_services({"a": declaration(provider="anthropic", request_options={"seed": 3})})
    assert parse_services({"a": declaration(request_options={"seed": 3})})[0].provider == "openai"
