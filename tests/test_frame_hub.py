# Copyright (c) Alibaba, Inc. and its affiliates.
"""FrameHub: one pump per camera, framesets shared by every consumer."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any

import pytest

from leapflow.hardware.transport import TransportError
from leapflow.hardware.vision import (
    ALIGN_SOFTWARE,
    FAILURE_FRAME_TIMEOUT,
    FAILURE_SOURCE_CLOSED,
    FrameCapabilities,
    FrameHub,
    FrameProducer,
    ImagePlane,
)


class _SteppedProducer:
    """Delivers exactly the framesets a test pushes, and counts every grab."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self.grabs = 0

    def capabilities(self) -> FrameCapabilities:
        return FrameCapabilities(planes=("color", "depth"), alignment=ALIGN_SOFTWARE)

    def grab(self, timeout_s: float) -> Any:
        try:
            item = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None  # an idle poll is not a failure; only pushed errors are
        if isinstance(item, Exception):
            raise item
        self.grabs += 1
        return item

    def push(self, item: Any) -> None:
        self._queue.put(item)


def _wait_for(predicate: Any, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


@pytest.fixture
def stepped() -> Any:
    producer = _SteppedProducer()
    hub = FrameHub(producer, name="cam", poll_timeout_s=0.05, error_backoff_s=0.0).start()
    try:
        yield producer, hub
    finally:
        hub.stop()


def test_a_producer_is_structurally_a_frame_producer() -> None:
    assert isinstance(_SteppedProducer(), FrameProducer)
    assert _SteppedProducer().capabilities().to_dict()["alignment"] == "software"


def test_image_planes_reject_impossible_geometry() -> None:
    ImagePlane("color", "rgb8", 2, 1, 6, b"\x00" * 6)
    with pytest.raises(ValueError):
        ImagePlane("color", "rgb8", 0, 1, 6, b"")
    with pytest.raises(ValueError):
        ImagePlane("color", "rgb8", 4, 1, 3, b"\x00" * 3)


def test_consumers_share_one_frameset_instead_of_stealing(stepped: Any) -> None:
    producer, hub = stepped
    producer.push("native-1")

    async def scenario() -> list[Any]:
        return list(await asyncio.gather(*(hub.latest(timeout_s=2.0) for _ in range(4))))

    framesets = asyncio.run(scenario())
    assert {fs.sequence for fs in framesets} == {1}
    assert {fs.native for fs in framesets} == {"native-1"}
    assert producer.grabs == 1


def test_freshness_conditions_wait_for_a_newer_frameset(stepped: Any) -> None:
    producer, hub = stepped
    producer.push("old")
    first = asyncio.run(hub.latest(timeout_s=2.0))
    called_at = time.monotonic()

    async def scenario() -> tuple[Any, Any]:
        after = asyncio.create_task(hub.latest(after_monotonic=called_at, timeout_s=2.0))
        newer = asyncio.create_task(hub.latest(min_sequence=first.sequence + 1, timeout_s=2.0))
        await asyncio.sleep(0.05)
        assert not after.done() and not newer.done()
        producer.push("new")
        return await after, await newer

    after, newer = asyncio.run(scenario())
    assert after.native == newer.native == "new"
    assert after.sequence == first.sequence + 1
    assert after.monotonic_at > called_at


def test_a_timeout_is_typed_and_names_the_producer_failure(stepped: Any) -> None:
    producer, hub = stepped
    producer.push(RuntimeError("Frame didn't arrive within 50"))
    _wait_for(lambda: hub.stats()["failures"] >= 1)

    with pytest.raises(TransportError) as error:
        asyncio.run(hub.latest(timeout_s=0.05))

    assert error.value.failure_code == FAILURE_FRAME_TIMEOUT
    assert "Frame didn't arrive" in str(error.value)
    producer.push("recovered")
    assert asyncio.run(hub.latest(timeout_s=2.0)).native == "recovered"
    assert hub.stats()["last_error"] == ""


def test_a_view_is_computed_once_per_frameset_off_the_loop_and_unlocked(stepped: Any) -> None:
    producer, hub = stepped
    producer.push("native")
    calls: list[int] = []
    gate = threading.Event()

    def encode(frameset: Any) -> str:
        calls.append(threading.get_ident())
        hub.stats()  # would deadlock if a hub lock were held around views
        assert gate.wait(2.0)
        return f"jpeg-{frameset.sequence}"

    async def scenario() -> list[str]:
        frameset = await hub.latest(timeout_s=2.0)
        readers = [asyncio.create_task(hub.view(frameset, "jpeg", encode)) for _ in range(3)]
        await asyncio.sleep(0.05)
        gate.set()
        return list(await asyncio.gather(*readers))

    loop_thread = threading.get_ident()
    assert asyncio.run(scenario()) == ["jpeg-1"] * 3
    assert len(calls) == 1 and calls[0] != loop_thread


def test_a_failed_view_is_retried_rather_than_cached(stepped: Any) -> None:
    producer, hub = stepped
    producer.push("native")
    attempts: list[int] = []

    def flaky(frameset: Any) -> str:
        attempts.append(frameset.sequence)
        if len(attempts) == 1:
            raise ValueError("decode failed")
        return "ok"

    async def scenario() -> str:
        frameset = await hub.latest(timeout_s=2.0)
        with pytest.raises(ValueError):
            await hub.view(frameset, "jpeg", flaky)
        return await hub.view(frameset, "jpeg", flaky)

    assert asyncio.run(scenario()) == "ok"
    assert attempts == [1, 1]


def test_views_of_framesets_that_left_the_ring_are_dropped(stepped: Any) -> None:
    producer, hub = stepped
    producer.push("one")

    async def first_view() -> Any:
        frameset = await hub.latest(timeout_s=2.0)
        await hub.view(frameset, "jpeg", lambda fs: "x")
        return frameset

    first = asyncio.run(first_view())
    assert hub.stats()["views"] == 1
    for name in ("two", "three"):
        producer.push(name)
    _wait_for(lambda: hub.stats()["sequence"] == first.sequence + 2)
    assert hub.stats()["views"] == 0


def test_cancelling_a_reader_does_not_cancel_a_shared_view(stepped: Any) -> None:
    producer, hub = stepped
    producer.push("native")
    gate = threading.Event()

    def slow(frameset: Any) -> str:
        assert gate.wait(2.0)
        return "done"

    async def scenario() -> str:
        frameset = await hub.latest(timeout_s=2.0)
        impatient = asyncio.create_task(hub.view(frameset, "jpeg", slow))
        patient = asyncio.create_task(hub.view(frameset, "jpeg", slow))
        await asyncio.sleep(0.05)
        impatient.cancel()
        gate.set()
        return await patient

    assert asyncio.run(scenario()) == "done"


def test_stop_fails_waiters_and_waits_for_in_flight_views() -> None:
    producer = _SteppedProducer()
    hub = FrameHub(producer, name="cam", poll_timeout_s=0.05).start()
    producer.push("native")
    gate = threading.Event()
    finished = threading.Event()

    def slow(frameset: Any) -> str:
        assert gate.wait(2.0)
        finished.set()
        return "done"

    async def scenario() -> None:
        frameset = await hub.latest(timeout_s=2.0)
        view = asyncio.create_task(hub.view(frameset, "jpeg", slow))
        waiter = asyncio.create_task(hub.latest(min_sequence=99, timeout_s=5.0))
        await asyncio.sleep(0.05)
        stopper = asyncio.create_task(asyncio.to_thread(hub.stop))
        await asyncio.sleep(0.1)
        assert not stopper.done(), "stop must wait for the in-flight view"
        with pytest.raises(TransportError) as error:
            await waiter
        assert error.value.failure_code == FAILURE_SOURCE_CLOSED
        gate.set()
        await stopper
        assert finished.is_set()
        assert await view == "done"
        with pytest.raises(TransportError) as closed:
            await hub.view(frameset, "other", slow)
        assert closed.value.failure_code == FAILURE_SOURCE_CLOSED

    asyncio.run(scenario())
    assert not hub.running
    with pytest.raises(TransportError):
        hub.start()


def test_stats_report_rate_age_and_demand(stepped: Any) -> None:
    producer, hub = stepped
    for index in range(3):
        producer.push(index)
        time.sleep(0.02)
    _wait_for(lambda: hub.stats()["sequence"] == 3)
    stats = hub.stats()
    assert stats["running"] is True
    assert stats["fps"] > 0
    assert stats["frame_age_ms"] is not None and stats["frame_age_ms"] >= 0
    assert stats["failures"] == 0
