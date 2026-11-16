# Copyright (c) Alibaba, Inc. and its affiliates.
"""Whether the active model sees images is decided by evidence, not by its name.

A static table that lags a vendor's release used to replace every tool image
with "the model cannot view images" for a natively multimodal model. The table
is now a hint; the model is asked once with a picture whose answer cannot be
guessed, and the verdict is remembered. Images are also deduplicated, costed by
their pixel size, described once, and an endpoint refusing the modality is
recoverable by stripping images.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.engine.recovery.error_classifier import ErrorCategory, ErrorClassifier
from leapflow.engine.tools import tool_attachments
from leapflow.engine.tools.tool_attachments import (
    ToolAttachment,
    described_message,
    describe_image,
    present_attachments,
)
from leapflow.llm.image_tokens import DEFAULT_IMAGE_TOKENS, image_part_tokens, image_size, image_tokens
from leapflow.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from leapflow.llm.vision_capability import (
    VisionCapability,
    VisionVerdict,
    VisionVerdictStore,
    judge_reply,
    probe_image,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class _Seer:
    """A model that reads the probe's two colours off the image it is sent."""

    base_url = "https://api.example.test/v1"

    def __init__(self, *, blind: bool = False, error: Exception | None = None) -> None:
        self.calls = 0
        self.blind = blind
        self.error = error

    async def achat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        from PIL import Image

        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.blind:
            return SimpleNamespace(content="I cannot see images.")
        url = messages[0]["content"][1]["image_url"]["url"]
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        names = {rgb: name for name, rgb in _palette().items()}
        left, right = image.getpixel((2, 16)), image.getpixel((61, 16))
        return SimpleNamespace(content=f"left={names[left]} right={names[right]}")


class _Status(Exception):
    def __init__(self, status: int, text: str = "") -> None:
        super().__init__(text or f"status {status}")
        self.status_code = status


def _palette() -> dict[str, tuple[int, int, int]]:
    from leapflow.llm import vision_capability

    return vision_capability._PALETTE


def _resolver(tmp_path: Path, clock: _Clock, **kwargs: Any) -> VisionCapability:
    store = VisionVerdictStore(tmp_path / "vision_verdicts.json")
    return VisionCapability(store=store, clock=clock, rng=random.Random(7), **kwargs)


# ── The probe ────────────────────────────────────────────────────────────────


def test_the_probe_image_is_a_real_two_colour_png() -> None:
    from PIL import Image

    data = base64.b64decode(probe_image((230, 20, 20), (20, 40, 230)))
    image = Image.open(io.BytesIO(data)).convert("RGB")
    assert image.size == (64, 32)
    assert image.getpixel((0, 0)) == (230, 20, 20) and image.getpixel((63, 31)) == (20, 40, 230)


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("left=red right=blue", True),
        ("The left half is red and the right half is blue.", True),
        ("left=blue right=red", False),
        ("I am unable to view images.", False),
        ("", None),
        ("   ", None),
    ],
)
def test_a_reply_is_judged_by_what_it_names(reply: str, verdict: bool | None) -> None:
    assert judge_reply(reply, "red", "blue") is verdict


async def test_a_model_that_sees_is_believed_and_remembered(tmp_path: Path) -> None:
    clock, llm = _Clock(), _Seer()
    registry = ModelCapabilityRegistry()
    registry.register("deepseek-chat", ModelCapabilities(supports_vision=False))
    resolver = _resolver(tmp_path, clock)
    assert await resolver.sees_images(llm, "deepseek-chat", hint=False, registry=registry)
    assert registry.resolve("deepseek-chat").supports_vision is True  # every reader agrees
    stored = json.loads((tmp_path / "vision_verdicts.json").read_text())
    (entry,) = stored.values()
    assert entry["vision"] is True and "answered" in entry["evidence"]
    again = _resolver(tmp_path, clock)  # a restarted daemon reads the file
    assert await again.sees_images(llm, "deepseek-chat", hint=False)
    assert llm.calls == 1


async def test_a_blind_model_is_told_apart_from_a_hint(tmp_path: Path) -> None:
    clock, llm = _Clock(), _Seer(blind=True)
    resolver = _resolver(tmp_path, clock)
    assert await resolver.sees_images(llm, "text-only", hint=True) is False
    clock.now += 8 * 86400.0  # a negative verdict expires sooner and is re-checked
    assert await resolver.sees_images(llm, "text-only", hint=True) is False
    assert llm.calls == 2


async def test_an_endpoint_refusing_the_image_is_a_verdict(tmp_path: Path) -> None:
    llm = _Seer(error=_Status(400, "unknown variant `image_url`"))
    assert await _resolver(tmp_path, _Clock()).sees_images(llm, "m", hint=True) is False


@pytest.mark.parametrize("error", [_Status(429), _Status(503), asyncio.TimeoutError()])
async def test_an_inconclusive_probe_keeps_the_hint_and_backs_off(
    tmp_path: Path, error: Exception
) -> None:
    clock, llm = _Clock(), _Seer(error=error)
    resolver = _resolver(tmp_path, clock, retry_after_s=60.0)
    assert await resolver.sees_images(llm, "m", hint=True) is True
    assert await resolver.sees_images(llm, "m", hint=False) is False
    assert llm.calls == 1  # within the backoff nothing is asked again
    clock.now += 61.0
    await resolver.sees_images(llm, "m", hint=True)
    assert llm.calls == 2
    assert not (tmp_path / "vision_verdicts.json").exists()


async def test_the_operator_setting_wins_without_a_probe(tmp_path: Path) -> None:
    llm = _Seer(blind=True)
    resolver = _resolver(tmp_path, _Clock(), mode="true")
    assert await resolver.sees_images(llm, "m", hint=False) is True
    resolver.configure(mode="false")
    assert await resolver.sees_images(_Seer(), "m", hint=True) is False
    assert llm.calls == 0


async def test_concurrent_first_images_share_one_probe(tmp_path: Path) -> None:
    llm = _Seer()
    resolver = _resolver(tmp_path, _Clock())
    answers = await asyncio.gather(*(resolver.sees_images(llm, "m", hint=False) for _ in range(5)))
    assert answers == [True] * 5 and llm.calls == 1


async def test_verdicts_are_per_endpoint(tmp_path: Path) -> None:
    clock = _Clock()
    resolver = _resolver(tmp_path, clock)
    other = _Seer(blind=True)
    other.base_url = "https://elsewhere.test/v1"
    assert await resolver.sees_images(_Seer(), "m", hint=False) is True
    assert await resolver.sees_images(other, "m", hint=True) is False


def test_a_damaged_verdict_file_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "vision_verdicts.json"
    path.write_text("{not json")
    store = VisionVerdictStore(path)
    assert store.get("k") is None
    store.put("k", VisionVerdict(True, 1.0, "seen"))
    assert VisionVerdictStore(path).get("k") == VisionVerdict(True, 1.0, "seen")


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-vl2", "qwen3.6-flash", "qwen3-vl-plus"])
def test_multimodal_generations_are_hinted_as_seeing(model: str) -> None:
    assert ModelCapabilityRegistry().resolve(model).supports_vision is True


# ── Image cost ───────────────────────────────────────────────────────────────


def _data_url(size: tuple[int, int], fmt: str = "JPEG") -> str:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(output, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_an_image_is_costed_by_its_pixels(fmt: str) -> None:
    small, large = _data_url((384, 288), fmt), _data_url((1280, 964), fmt)
    assert image_size(small) == (384, 288) and image_size(large) == (1280, 964)
    thumb = image_part_tokens({"type": "image_url", "image_url": {"url": small}})
    mosaic = image_part_tokens({"type": "image_url", "image_url": {"url": large}})
    assert thumb < DEFAULT_IMAGE_TOKENS / 4 < mosaic
    assert image_tokens(384, 288) == thumb


def test_an_unreadable_image_keeps_the_flat_estimate() -> None:
    url_part = {"type": "image_url", "image_url": {"url": "https://example.test/a.jpg"}}
    assert image_part_tokens(url_part) == DEFAULT_IMAGE_TOKENS
    assert image_tokens(0, 10) == DEFAULT_IMAGE_TOKENS


# ── Showing tool images ──────────────────────────────────────────────────────


def _item(call: str, colour: int) -> ToolAttachment:
    data = _data_url((32, 24)).split(",", 1)[1] if colour == 0 else _data_url((32 + colour, 24)).split(",", 1)[1]
    return ToolAttachment(call, "hw_skill_run", "image/jpeg", data, f"view {call}")


def _live(messages: list[dict[str, Any]]) -> list[str]:
    return tool_attachments._live_image_urls(messages)


async def test_an_image_already_live_is_not_sent_again() -> None:
    messages: list[dict[str, Any]] = []
    await present_attachments(messages, [_item("c1", 0)], keep=3, sees_images=True)
    await present_attachments(messages, [_item("c2", 0), _item("c3", 1)], keep=3, sees_images=True)
    assert len(_live(messages)) == 2
    text = messages[-1]["content"][0]["text"]
    assert "c2" in text and "identical to an image already shown" in text


async def test_a_repeat_whose_copy_would_retire_is_shown_again() -> None:
    messages: list[dict[str, Any]] = []
    await present_attachments(messages, [_item("c1", 0)], keep=2, sees_images=True)
    batch = [_item("c2", 1), _item("c3", 2), _item("c4", 0)]
    await present_attachments(messages, batch, keep=2, sees_images=True)
    live = _live(messages)
    assert len(live) == 2 and batch[2].data_url in live
    assert "identical" not in messages[-1]["content"][0]["text"]


async def test_a_blind_model_gets_a_hint_on_how_to_show_images() -> None:
    message = described_message([_item("c1", 0)], [""])
    assert "llm.supports_vision true" in message["content"]
    assert "cannot view images" not in message["content"]


async def test_an_image_is_described_once() -> None:
    tool_attachments._DESCRIPTIONS.clear()
    calls: list[int] = []

    async def achat(messages: list[Any], **kwargs: Any) -> Any:
        calls.append(kwargs["max_tokens"])
        return SimpleNamespace(content="a cup on a table")

    vlm = SimpleNamespace(achat=achat)
    item = _item("c1", 0)
    assert await describe_image(vlm, item) == "a cup on a table"
    assert await describe_image(vlm, item) == "a cup on a table"
    assert len(calls) == 1 and calls[0] >= 400


# ── An endpoint that refuses images ──────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "unknown variant `image_url`, expected `text`",
        "image input is not supported by this model",
        "Image too large",
    ],
)
def test_an_image_refusal_strips_images_and_retries(text: str) -> None:
    assert ErrorClassifier().classify(_Status(400, text)) == ErrorCategory.IMAGE_TOO_LARGE
