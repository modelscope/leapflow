# Copyright (c) Alibaba, Inc. and its affiliates.
"""Whether the active model can see images, decided by evidence rather than by name.

A static model table always lags releases: entries written when a vendor's line
was text-only keep saying so after the vendor ships native vision, and every
image a tool returns is then replaced by "the model cannot view images" while
the model could have read it. So the static answer is only a *hint*. The first
time an image is about to be shown, :class:`VisionCapability` asks the model
itself with a tiny generated picture whose answer cannot be guessed (two halves
in randomly drawn colours) and remembers the verdict per endpoint and model in
the profile cache, so the probe costs one short call per model per TTL.

Resolution order, first match wins:

1. The operator's explicit ``llm.supports_vision`` (``true``/``false``).
2. A fresh remembered verdict for this endpoint and model.
3. A probe (single-flight per key); a conclusive answer is persisted.
4. The static hint, retried no sooner than the backoff when the probe was
   inconclusive (timeout, rate limit, network, empty answer).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import struct
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

VERDICTS_FILE = "vision_verdicts.json"

# Colours any vision model names the same way; drawn two at a time so a model
# that answers without looking is right about one probe in fifteen.
_PALETTE: Dict[str, tuple[int, int, int]] = {
    "red": (230, 20, 20),
    "green": (20, 180, 40),
    "blue": (20, 40, 230),
    "yellow": (245, 225, 20),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}
# A 4xx on an image request means the endpoint refused the modality; auth,
# quota and rate limits say nothing about it.
_REFUSING_STATUS = frozenset({400, 404, 415, 422})
_INCONCLUSIVE_STATUS = frozenset({401, 402, 403, 408, 409, 429})


@dataclass(frozen=True)
class VisionVerdict:
    """One remembered answer to "does this model see images"."""

    vision: bool
    checked_at: float
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"vision": self.vision, "checked_at": self.checked_at, "evidence": self.evidence}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Optional["VisionVerdict"]:
        try:
            return cls(bool(data["vision"]), float(data["checked_at"]), str(data.get("evidence") or ""))
        except (KeyError, TypeError, ValueError):
            return None


class VisionVerdictStore:
    """Verdicts persisted as one small JSON file; a damaged file reads as empty."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._cache: Dict[str, VisionVerdict] | None = None

    def get(self, key: str) -> Optional[VisionVerdict]:
        return self._load().get(key)

    def put(self, key: str, verdict: VisionVerdict) -> None:
        verdicts = self._load()
        verdicts[key] = verdict
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._path.with_suffix(".tmp")
            payload = {name: item.to_dict() for name, item in verdicts.items()}
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, self._path)
        except OSError:
            logger.warning("could not persist vision verdicts to %s", self._path, exc_info=True)

    def _load(self) -> Dict[str, VisionVerdict]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        if self._path is None or not self._path.exists():
            return self._cache
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("ignoring unreadable vision verdicts at %s", self._path, exc_info=True)
            return self._cache
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                verdict = VisionVerdict.from_mapping(value) if isinstance(value, Mapping) else None
                if verdict is not None:
                    self._cache[str(key)] = verdict
        return self._cache


def probe_image(left: tuple[int, int, int], right: tuple[int, int, int]) -> str:
    """A 64x32 PNG, *left* colour on the left half and *right* on the right, as base64."""
    width, height = 64, 32
    row = b"\x00" + bytes(left) * (width // 2) + bytes(right) * (width - width // 2)

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def judge_reply(text: str, left: str, right: str) -> Optional[bool]:
    """``True`` when the reply names both colours on the right sides, ``False``
    when it answered something else, ``None`` when it answered nothing."""
    answer = " ".join(str(text or "").lower().split())
    if not answer:
        return None
    named_left = re.search(r"left\W{0,3}(?:half\W{0,3})?(?:is\W{0,3})?([a-z]+)", answer)
    named_right = re.search(r"right\W{0,3}(?:half\W{0,3})?(?:is\W{0,3})?([a-z]+)", answer)
    if (
        named_left and named_right
        and named_left.group(1) in _PALETTE and named_right.group(1) in _PALETTE
    ):
        return named_left.group(1) == left and named_right.group(1) == right
    return left in answer and right in answer


class VisionCapability:
    """Evidence-based answer to "can this model see images", cached per endpoint and model."""

    def __init__(
        self,
        *,
        store: VisionVerdictStore,
        mode: str = "auto",
        positive_ttl_s: float = 30 * 86400.0,
        negative_ttl_s: float = 7 * 86400.0,
        retry_after_s: float = 600.0,
        probe_timeout_s: float = 20.0,
        probe_max_tokens: int = 400,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        self._store = store
        self._mode = str(mode or "auto").strip().lower()
        self._positive_ttl_s = float(positive_ttl_s)
        self._negative_ttl_s = float(negative_ttl_s)
        self._retry_after_s = float(retry_after_s)
        self._probe_timeout_s = float(probe_timeout_s)
        self._probe_max_tokens = int(probe_max_tokens)
        self._clock = clock
        self._rng = rng or random.Random()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._retry_at: Dict[str, float] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def configure(self, *, mode: str) -> None:
        """Apply a reloaded ``llm.supports_vision`` without forgetting verdicts."""
        self._mode = str(mode or "auto").strip().lower()
        self._retry_at.clear()

    @staticmethod
    def key(llm: Any, model: str) -> str:
        base_url = getattr(llm, "base_url", None) or getattr(llm, "_base_url", "") or ""
        return f"{str(base_url).rstrip('/')}|{model}"

    def remembered(self, llm: Any, model: str) -> Optional[VisionVerdict]:
        """The fresh stored verdict for *model* behind *llm*, if any."""
        verdict = self._store.get(self.key(llm, model))
        if verdict is None:
            return None
        ttl = self._positive_ttl_s if verdict.vision else self._negative_ttl_s
        return verdict if self._clock() - verdict.checked_at <= ttl else None

    async def sees_images(
        self, llm: Any, model: str, *, hint: bool, registry: Any = None
    ) -> bool:
        """Whether *model* (served by *llm*) can see images; *hint* is the static guess.

        A conclusive verdict is also written into *registry* so every other
        reader of the model's capabilities agrees with what was observed.
        """
        if self._mode in {"true", "false"}:
            return self._mode == "true"
        if not model or llm is None or not callable(getattr(llm, "achat", None)):
            return hint
        key = self.key(llm, model)
        verdict = self.remembered(llm, model)
        if verdict is None and self._clock() >= self._retry_at.get(key, 0.0):
            lock = self._locks.setdefault(key, asyncio.Lock())
            async with lock:
                verdict = self.remembered(llm, model) or await self._probe(llm, model, key)
        if verdict is None:
            return hint
        _record(registry, model, verdict.vision)
        return verdict.vision

    async def _probe(self, llm: Any, model: str, key: str) -> Optional[VisionVerdict]:
        if self._clock() < self._retry_at.get(key, 0.0):
            return None
        left, right = self._rng.sample(sorted(_PALETTE), 2)
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The image has a left half and a right half, each one solid colour. "
                        "Reply with exactly one line: left=<colour> right=<colour>"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{probe_image(_PALETTE[left], _PALETTE[right])}"
                    },
                },
            ],
        }
        started = time.monotonic()
        try:
            reply = await asyncio.wait_for(
                llm.achat(
                    [message], stream=False, enable_thinking=False,
                    max_tokens=self._probe_max_tokens,
                ),
                timeout=self._probe_timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure is classified below
            return self._settle(key, model, _refusal(exc), f"{type(exc).__name__}: {exc}"[:240])
        answer = str(getattr(reply, "content", "") or "")
        elapsed = time.monotonic() - started
        evidence = f"asked left={left} right={right}, answered {answer.strip()[:120]!r} in {elapsed:.1f}s"
        return self._settle(key, model, judge_reply(answer, left, right), evidence)

    def _settle(
        self, key: str, model: str, vision: Optional[bool], evidence: str
    ) -> Optional[VisionVerdict]:
        if vision is None:
            self._retry_at[key] = self._clock() + self._retry_after_s
            logger.info("vision probe for %s was inconclusive (%s)", model, evidence)
            return None
        verdict = VisionVerdict(vision, self._clock(), evidence)
        self._store.put(key, verdict)
        logger.info("vision probe: %s %s images (%s)", model, "sees" if vision else "does not see", evidence)
        return verdict


def _refusal(exc: BaseException) -> Optional[bool]:
    """``False`` when the endpoint rejected the image request itself, else ``None``."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    try:
        code = int(status) if status is not None else 0
    except (TypeError, ValueError):
        code = 0
    if code in _REFUSING_STATUS:
        return False
    if code in _INCONCLUSIVE_STATUS or code >= 500 or code == 0:
        return None
    return False if 400 <= code < 500 else None


def _record(registry: Any, model: str, vision: bool) -> None:
    if registry is None:
        return
    try:
        caps = registry.resolve(model)
        if bool(caps.supports_vision) != vision:
            registry.register(model, replace(caps, supports_vision=vision))
    except (AttributeError, TypeError, ValueError):
        logger.debug("could not record the vision verdict for %s", model, exc_info=True)


def verdicts_path(settings: Any) -> Path | None:
    """Where the profile keeps its vision verdicts, or ``None`` without a layout."""
    layout = getattr(settings, "profile_layout", None)
    cache = getattr(layout, "cache", None)
    if cache is None:
        return None
    try:
        return cache.category_dir(scope="profile", category="llm") / VERDICTS_FILE
    except (TypeError, ValueError):
        return None


__all__ = [
    "VERDICTS_FILE",
    "VisionCapability",
    "VisionVerdict",
    "VisionVerdictStore",
    "judge_reply",
    "probe_image",
    "verdicts_path",
]
