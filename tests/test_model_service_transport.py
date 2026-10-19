"""Model-service transport against a real loopback HTTP server, plus the v1 contract types."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, AsyncIterator

import pytest
from aiohttp import web

from leapflow.hardware.context import Channel, ContextProvenance, HardwareContext, TransportRef
from leapflow.hardware.model_contract import (
    IDENTITY_KEYS,
    TASK_ACTION_CHUNK,
    TASK_GROUNDING,
    ActionChunkResult,
    Detection,
    GroundingResult,
    MaskRle,
    ModelInfo,
    ModelService,
    act,
    check_request,
    ground,
    parse_response,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    HardwareTransport,
    TransportError,
)
from leapflow.hardware.transports import build_transport
from leapflow.hardware.transports.model_service import ModelServiceTransport

_INFO = {
    "task": TASK_GROUNDING,
    "name": "owlv2-sam2",
    "version": "2026.10",
    "checksum": "sha256:abc",
    "inputs": {"image": "jpeg"},
}
_ECHO = {"name": "owlv2-sam2", "version": "2026.10"}
_DETECTION = {
    "label": "red cube",
    "score": 0.91,
    "box": [10, 20, 30, 40],
    "mask_rle": {"size": [2, 2], "counts": [1, 2, 1]},
}


class _ModelServer:
    """Scripted v1 server: each ``/v1/infer`` pops the next (status, body, delay)."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        self.health_status = 200
        self.replies: deque[tuple[int, Any, float]] = deque()
        self.requests: list[dict[str, Any]] = []

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "ok", "model_info": self.info}, status=self.health_status
        )

    async def infer(self, request: web.Request) -> web.Response:
        self.requests.append(await request.json())
        status, body, delay = (
            self.replies.popleft() if self.replies else (500, {"error": "unscripted"}, 0)
        )
        if delay:
            await asyncio.sleep(delay)
        if status == 200 and isinstance(body, dict) and "model" not in body:
            body = {**body, "model": self.identity()}
        return web.json_response(body, status=status)

    def identity(self) -> dict[str, Any]:
        return {key: self.info.get(key, "") for key in IDENTITY_KEYS}


@pytest.fixture
async def server() -> AsyncIterator[tuple[_ModelServer, str, web.AppRunner]]:
    model = _ModelServer(dict(_INFO))
    app = web.Application()
    app.router.add_get("/v1/health", model.health)
    app.router.add_post("/v1/infer", model.infer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    yield model, f"http://127.0.0.1:{port}", runner
    await runner.cleanup()


def _context(config: dict[str, Any]) -> HardwareContext:
    return HardwareContext(
        device_id="perception_locator",
        display_name="Open-vocabulary locator",
        transport=TransportRef(kind="model_service", config=config),
        halt_supported=False,
        channels=(
            Channel(channel_id="model.info", direction="read", quantity="model_metadata"),
            Channel(channel_id="model.health", direction="read", quantity="model_metadata"),
            Channel(channel_id="model.latency", direction="read", quantity="model_metadata"),
        ),
        provenance=ContextProvenance(verified_by="test"),
    )


async def _open(url: str, **config: Any) -> ModelServiceTransport:
    settings = {"url": url, "task": TASK_GROUNDING, "retry_backoff_s": 0, **config}
    transport = build_transport("model_service", settings)
    assert isinstance(transport, ModelServiceTransport)
    await transport.open(_context(settings))
    return transport


async def _open_error(url: str, **config: Any) -> TransportError:
    with pytest.raises(TransportError) as error:
        await _open(url, **config)
    return error.value


# ── Open: identity is verified, fail-closed ──────────────────────────────────


async def test_open_verifies_identity_and_exposes_it_as_a_channel(server) -> None:
    _, url, _ = server
    transport = await _open(url, expect={"name": "owlv2-sam2", "version": "2026.10"})

    status = await transport.probe()
    info = (await transport.read("model.info")).value

    assert isinstance(transport, HardwareTransport)
    assert isinstance(transport, ModelService)
    assert status.connected and not status.halt_supported
    assert info["name"] == "owlv2-sam2" and info["inputs"] == {"image": "jpeg"}
    assert transport.model_info.checksum == "sha256:abc"
    assert (await transport.open(_context({}))).connected  # idempotent


async def test_a_server_serving_another_task_is_refused(server) -> None:
    _, url, _ = server

    error = await _open_error(url, task=TASK_ACTION_CHUNK)

    assert error.failure_code == "model_task_mismatch"


async def test_a_swapped_checkpoint_is_refused(server) -> None:
    _, url, _ = server

    error = await _open_error(url, expect={"version": "2026.09"})

    assert error.failure_code == "model_binding_mismatch"
    assert "2026.09" in str(error)


async def test_health_without_model_info_is_refused(server) -> None:
    model, url, _ = server
    model.info = {"task": TASK_GROUNDING}

    error = await _open_error(url)

    assert error.failure_code == "model_bad_response"


@pytest.mark.parametrize(
    ("config", "fragment"),
    [
        ({"url": "http://10.0.0.5:8000"}, "not loopback"),
        ({"url": "ftp://127.0.0.1:8000"}, "http(s)"),
        ({"url": "http://127.0.0.1:1", "task": "classify"}, "task"),
        ({"url": "http://127.0.0.1:1", "expect": {"sha": "x"}}, "expect"),
        ({"url": "http://127.0.0.1:1", "timeout_s": 0}, "timeout_s"),
    ],
)
async def test_misconfigured_declarations_are_refused_before_any_request(
    config: dict[str, Any], fragment: str
) -> None:
    error = await _open_error(**config)

    assert error.failure_code == "model_service_misconfigured"
    assert fragment in str(error)


async def test_remote_url_needs_an_explicit_opt_in() -> None:
    error = await _open_error(
        "http://10.255.255.1:9", allow_remote=True, timeout_s=0.05, max_retries=0
    )

    assert error.failure_code in {"model_unreachable", "model_timeout"}


# ── Inference ────────────────────────────────────────────────────────────────


async def test_ground_returns_typed_detections_and_records_latency(server) -> None:
    model, url, _ = server
    model.replies.append((200, {"detections": [_DETECTION], "infer_ms": 41.5}, 0))
    transport = await _open(url)

    result = await ground(transport, b"\xff\xd8jpeg", ["red cube"])

    assert isinstance(result, GroundingResult)
    assert result.detections[0].label == "red cube"
    assert result.detections[0].mask.area == 2
    assert model.requests[0]["task"] == TASK_GROUNDING
    assert model.requests[0]["queries"] == ["red cube"]
    latency = (await transport.read("model.latency")).value
    assert latency["count"] == 1 and latency["infer_p50_ms"] == 41.5


async def test_busy_is_retried_within_the_budget(server) -> None:
    model, url, _ = server
    model.replies.extend([(409, {"error": "busy"}, 0), (200, {"detections": [], "infer_ms": 5}, 0)])
    transport = await _open(url, max_retries=1)

    result = await ground(transport, b"jpeg", ["cup"])

    assert result.detections == ()
    assert len(model.requests) == 2


async def test_zero_retries_means_zero(server) -> None:
    model, url, _ = server
    model.replies.extend([(409, {"error": "busy"}, 0), (200, {"detections": [], "infer_ms": 5}, 0)])
    transport = await _open(url, max_retries=0)

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == "model_busy"
    assert len(model.requests) == 1


@pytest.mark.parametrize(("status", "code"), [(422, "model_schema_mismatch"), (500, "model_error")])
async def test_non_transient_failures_are_never_retried(server, status: int, code: str) -> None:
    model, url, _ = server
    model.replies.extend([(status, {"detail": "bad image"}, 0)] * 4)
    transport = await _open(url, max_retries=3)

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == code
    assert "bad image" in str(error.value)
    assert len(model.requests) == 1


async def test_a_slow_inference_times_out(server) -> None:
    model, url, _ = server
    model.replies.append((200, {"detections": [], "infer_ms": 1}, 0.5))
    transport = await _open(url, timeout_s=0.1, max_retries=0)

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == "model_timeout"


async def test_an_invalid_response_is_a_typed_failure(server) -> None:
    model, url, _ = server
    model.replies.append((200, {"detections": [{**_DETECTION, "score": 1.5}], "infer_ms": 1}, 0))
    transport = await _open(url)

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == "model_bad_response"


async def test_an_invalid_request_never_reaches_the_server(server) -> None:
    model, url, _ = server
    transport = await _open(url)

    with pytest.raises(TransportError) as error:
        await transport.infer({"image": "abc", "queries": []})

    assert error.value.failure_code == "model_bad_request"
    assert model.requests == []


async def test_action_chunks_come_back_as_equal_width_rows(server) -> None:
    model, url, _ = server
    model.info = {**_INFO, "task": TASK_ACTION_CHUNK}
    model.replies.append((200, {"actions": [[0.1] * 7, [0.2] * 7], "infer_ms": 80}, 0))
    transport = await _open(url, task=TASK_ACTION_CHUNK)

    result = await act(transport, {"front": b"jpeg"}, [0.0] * 7, "pick the cube")

    assert isinstance(result, ActionChunkResult)
    assert result.horizon == 2 and len(result.actions[0]) == 7
    assert model.requests[0]["task"] == "pick the cube"


async def test_a_lost_server_reads_as_unhealthy_and_fails_inference(server) -> None:
    _, url, runner = server
    transport = await _open(url, max_retries=0)
    await runner.cleanup()

    health = await transport.read("model.health")
    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert health.value["ok"] is False and health.quality == "suspect"
    assert error.value.failure_code == "model_unreachable"
    assert not (await transport.probe()).connected


# ── Identity: every response is checked against the verified model ────────────


async def test_a_response_from_another_model_is_refused(server) -> None:
    model, url, _ = server
    transport = await _open(url)
    swapped = {**model.identity(), "version": "2026.11"}
    model.replies.append((200, {"detections": [], "infer_ms": 1, "model": swapped}, 0))

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == "model_identity_changed"
    assert "2026.11" in str(error.value)


async def test_a_restarted_server_instance_is_refused(server) -> None:
    model, url, _ = server
    model.info = {**_INFO, "instance": "pid-1"}
    transport = await _open(url)
    restarted = {**model.identity(), "instance": "pid-2"}
    model.replies.append((200, {"detections": [], "infer_ms": 1, "model": restarted}, 0))

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == "model_identity_changed"


async def test_a_response_without_an_identity_echo_is_refused(server) -> None:
    model, url, _ = server
    transport = await _open(url)
    model.replies.append((200, {"detections": [], "infer_ms": 1, "model": None}, 0))

    with pytest.raises(TransportError) as error:
        await ground(transport, b"jpeg", ["cup"])

    assert error.value.failure_code == "model_bad_response"


async def test_a_live_swap_reads_unhealthy_and_refresh_adopts_it(server) -> None:
    model, url, _ = server
    transport = await _open(url)
    model.info = {**_INFO, "version": "2026.11"}

    health = (await transport.read("model.health")).value
    refreshed = await transport.refresh()
    model.replies.append((200, {"detections": [], "infer_ms": 1}, 0))
    result = await ground(transport, b"jpeg", ["cup"])

    assert health["ok"] is False and "model_identity_changed" in health["detail"]
    assert refreshed.version == "2026.11" and transport.model_info.version == "2026.11"
    assert result.model.version == "2026.11"


async def test_refresh_still_enforces_the_declared_pin(server) -> None:
    model, url, _ = server
    transport = await _open(url, expect={"version": "2026.10"})
    model.info = {**_INFO, "version": "2026.11"}

    with pytest.raises(TransportError) as error:
        await transport.refresh()

    assert error.value.failure_code == "model_binding_mismatch"


# ── Governance surface ───────────────────────────────────────────────────────


async def test_writes_are_refused_with_a_provable_none_and_halt_is_declared(server) -> None:
    _, url, _ = server
    transport = await _open(url)

    outcome = await transport.write("model.info", {"name": "other"})
    halt = await transport.halt()

    assert not outcome.ok and outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == "not_writable"
    assert halt.halt_supported is False


async def test_close_is_idempotent_and_ends_reads(server) -> None:
    _, url, _ = server
    transport = await _open(url)

    assert (await transport.close()).connected is False
    assert (await transport.close()).connected is False
    with pytest.raises(TransportError) as error:
        await transport.read("model.info")
    assert error.value.failure_code == "transport_not_open"


# ── Contract types ───────────────────────────────────────────────────────────


def test_mask_rle_is_column_major_starting_with_background() -> None:
    # 2x3 image, column-major runs: bg 1, fg 3, bg 2 -> pixels (u,v) = (0,1), (1,0), (1,1).
    mask = MaskRle.from_mapping({"size": [2, 3], "counts": [1, 3, 2]})

    assert mask.area == 3
    assert list(mask.foreground()) == [(0, 1), (1, 0), (1, 1)]
    assert list(mask.foreground(stride=2)) == [(0, 1), (1, 1)]
    assert MaskRle.from_mapping(mask.to_dict()) == mask


@pytest.mark.parametrize(
    "rle",
    [
        {"size": [2, 2], "counts": [1, 2]},
        {"size": [2, 2], "counts": "compressed"},
        {"size": [0, 2], "counts": []},
        {"size": [2, 2], "counts": [5, -1]},
    ],
)
def test_malformed_masks_are_refused(rle: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MaskRle.from_mapping(rle)


@pytest.mark.parametrize(
    "payload",
    [
        {"detections": [{**_DETECTION, "box": [30, 20, 10, 40]}], "infer_ms": 1, "model": _ECHO},
        {"detections": [{**_DETECTION, "label": ""}], "infer_ms": 1, "model": _ECHO},
        {"detections": [{"label": "cup", "score": 0.5}], "infer_ms": 1, "model": _ECHO},
        {"detections": [{"label": "cup", "score": 0.5, "point": [1]}], "infer_ms": 1, "model": _ECHO},
        {"detections": [{"label": "cup", "score": 0.5, "point": [-1, 2]}], "infer_ms": 1, "model": _ECHO},
        {"detections": [_DETECTION], "model": _ECHO},
        {"detections": "none", "infer_ms": 1, "model": _ECHO},
        {"detections": [_DETECTION], "infer_ms": 1},
    ],
)
def test_malformed_groundings_are_refused(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_response(TASK_GROUNDING, payload)


@pytest.mark.parametrize(
    ("fields", "geometry"),
    [
        ({"mask_rle": _DETECTION["mask_rle"], "box": [0, 0, 2, 2], "point": [1, 1]}, "mask"),
        ({"box": [0, 0, 2, 2], "point": [1, 1]}, "box"),
        ({"point": [1.5, 0.5]}, "point"),
    ],
)
def test_a_detection_carries_a_mask_a_box_or_a_point(fields: dict, geometry: str) -> None:
    result = parse_response(
        TASK_GROUNDING,
        {"detections": [{"label": "cup", "score": 0.7, **fields}], "infer_ms": 1, "model": _ECHO},
    )

    assert result.detections[0].geometry == geometry
    with pytest.raises(ValueError):
        Detection("cup", 0.7)


@pytest.mark.parametrize("actions", [[], [[0.1, 0.2], [0.3]], [[]], [[float("nan")]], "abc"])
def test_malformed_action_chunks_are_refused(actions: Any) -> None:
    with pytest.raises(ValueError):
        parse_response(TASK_ACTION_CHUNK, {"actions": actions, "infer_ms": 1, "model": _ECHO})


def test_requests_are_checked_before_sending() -> None:
    ok = check_request(TASK_GROUNDING, {"image": "b64", "queries": [" cup "]})

    assert ok["queries"] == ["cup"]
    with pytest.raises(ValueError):
        check_request(TASK_GROUNDING, {"image": "b64", "queries": ["x"] * 17})
    with pytest.raises(ValueError):
        check_request(TASK_ACTION_CHUNK, {"images": {"front": "b64"}, "state": [0.0], "task": ""})


def test_model_info_binding_names_the_first_mismatch() -> None:
    info = ModelInfo.from_mapping(_INFO)

    assert info.binding_problem({}) == ""
    assert info.binding_problem({"name": "owlv2-sam2", "checksum": "sha256:abc"}) == ""
    assert info.binding_problem({"checksum": "sha256:def"}).startswith("checksum")
    assert info.to_dict()["inputs"] == {"image": "jpeg"}
