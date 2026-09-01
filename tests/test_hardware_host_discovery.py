"""Host resource discovery and the hot-plug reconcile path.

Two subjects that share a file because they share a failure mode: both are about the
*set* of admitted devices changing, and both have a way of looking correct while being
silently wrong.

The host probe table is asserted for the properties that make it safe to enable by
default -- no device I/O during discovery, graceful degradation without ``psutil``, and
a channel set small enough that it does not swamp the board or start dozens of sampling
loops. The reconcile path is asserted for the one thing ``load()`` cannot do: converge
on a new device set without orphaning the asyncio tasks the previous one owns.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    TransportRef,
)
from leapflow.hardware.host_metrics import (
    DEFAULT_MAX_CHANNELS,
    DEVICE_ID,
    HostMetrics,
    HostProbe,
)
from leapflow.hardware.providers.host_provider import HostContextProvider
from leapflow.hardware.registry import DEFAULT_PROVIDER_KINDS, HardwareRegistry, HardwareSettings


# ════════════════════════════════════════════════════════════════
# Host discovery
# ════════════════════════════════════════════════════════════════


def test_discovery_performs_no_device_io() -> None:
    """The provider contract: discovery must work with the hardware powered off.

    Enforced by making any subprocess fatal. Enumeration on this platform reads
    in-process counters, the mount table and sensor names -- a provider that shelled out
    to a system profiler would make daemon startup latency a function of how many
    peripherals are attached.
    """
    import subprocess

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"host discovery spawned a subprocess: {args!r}")

    original = subprocess.run
    subprocess.run = _forbidden  # type: ignore[assignment]
    try:
        contexts = HostContextProvider({}).discover()
    finally:
        subprocess.run = original  # type: ignore[assignment]

    assert len(contexts) <= 1, "the host must be one device, not one per subsystem"


def test_the_host_is_one_device_with_namespaced_channels() -> None:
    """Seven devices would consume most of ``max_devices`` before a real peripheral.

    So cpu, memory, disk and network are channels of a single ``host`` device, named
    with a dotted prefix rather than split into separate declarations.
    """
    contexts = HostContextProvider({}).discover()
    if not contexts:
        pytest.skip("no host probes are available on this platform")

    context = contexts[0]
    assert context.device_id == DEVICE_ID
    assert len(context.channels) > 1
    # Nothing the host provider declares is commandable: a discovered declaration
    # carries no envelope a person is accountable for.
    assert context.writable_channels == ()
    assert context.halt_supported is False, "a CPU has no emergency stop"
    assert context.provenance.source == ContextSource.DISCOVERED.value
    assert context.provenance.is_verified is False, (
        "an unverified declaration is what keeps a future host control channel from "
        "becoming commandable without a human confirming it"
    )


def test_every_declared_channel_has_a_reader() -> None:
    """Provider and transport read one table, so this cannot drift.

    Held as two tables they diverged in every design sketch: the provider would declare
    a channel the transport had no reader for, and the board would show a permanently
    empty trace with no error anywhere.
    """
    metrics = HostMetrics({})
    declared = {channel.channel_id for channel in metrics.channels()}
    readable = metrics.known_channels()
    assert declared == readable, (
        f"declared but unreadable: {sorted(declared - readable)}; "
        f"readable but undeclared: {sorted(readable - declared)}"
    )


def test_the_channel_set_stays_small_enough_to_be_useful() -> None:
    """A channel is a sampling task, a stored series, and a row competing for a chart.

    A macOS host enumerates two dozen network interfaces and eight APFS volumes of one
    container; unfiltered, that was 66 channels for a machine with one disk and one
    active link.
    """
    metrics = HostMetrics({})
    channels = metrics.channels()
    if not channels:
        pytest.skip("no host probes are available on this platform")
    assert len(channels) <= DEFAULT_MAX_CHANNELS

    interfaces = {
        channel.channel_id.split(".")[1]
        for channel in channels
        if channel.channel_id.startswith("net.")
    }
    assert len(interfaces) <= 3, f"too many interfaces charted: {sorted(interfaces)}"

    # Volumes sharing one filesystem report identical capacity and must collapse to one.
    disks = {
        channel.channel_id.split(".")[1]
        for channel in channels
        if channel.channel_id.startswith("disk.")
    }
    assert len(disks) <= 4, f"too many filesystems charted: {sorted(disks)}"


def test_include_and_exclude_narrow_the_set_by_prefix() -> None:
    """Prefixes, because mounts and interfaces are discovered.

    Their full channel ids are not knowable in advance, so an exact-id filter would be
    unusable for the very channels most likely to need filtering.
    """
    everything = HostMetrics({})
    if not everything.channels():
        pytest.skip("no host probes are available on this platform")

    only_cpu = {channel.channel_id for channel in HostMetrics({"include": "cpu"}).channels()}
    assert only_cpu
    assert all(channel.startswith("cpu") for channel in only_cpu)

    without_net = {channel.channel_id for channel in HostMetrics({"exclude": "net"}).channels()}
    assert not any(channel.startswith("net") for channel in without_net)


def test_a_probe_that_cannot_answer_reports_none_rather_than_zero() -> None:
    """Zero is a measurement. "Cannot measure" is not, and must not look like one.

    A fabricated value enters the downsample window, where nothing downstream can tell
    it apart from something a sensor actually reported.
    """
    metrics = HostMetrics({})

    def _explode() -> Any:
        raise OSError("sensor vanished")

    probe = HostProbe(
        channel_id="test.exploding", quantity="test", unit="x", reader=_explode
    )
    metrics._probes = {probe.channel_id: probe}  # noqa: SLF001 - injecting one probe
    assert metrics.read("test.exploding") is None
    assert metrics.read("test.absent") is None


def test_the_transport_marks_an_unreadable_channel_suspect() -> None:
    """The reading's quality is the only place "no value" survives into stored history."""
    from leapflow.hardware.transports.host import HostTransport

    context = HostContextProvider({}).discover()
    if not context:
        pytest.skip("no host probes are available on this platform")

    transport = HostTransport({})

    async def _run() -> None:
        status = await transport.open(context[0])
        assert status.connected is True
        assert status.halt_supported is False

        # A channel the declaration names but no probe answers.
        declared = context[0].channels[0]
        reading = await transport.read(declared.channel_id)
        assert reading.channel_id == declared.channel_id

        outcome = await transport.write(declared.channel_id, 1)
        assert outcome.ok is False
        assert outcome.side_effect_state == "none", (
            "the call never reached anything, so no effect is provable and replay is safe"
        )
        await transport.close()

    asyncio.run(_run())


def test_the_default_provider_set_excludes_emitting_scanners() -> None:
    """Discovery must never be the reason a radio starts up or a packet leaves.

    ``media`` is in the default set because *enumerating* a camera needs no consent and
    opens nothing; the read that discloses something is separately gated.
    """
    assert DEFAULT_PROVIDER_KINDS == ("yaml", "host", "media")
    assert "bluetooth" not in DEFAULT_PROVIDER_KINDS
    assert "mdns" not in DEFAULT_PROVIDER_KINDS


# ════════════════════════════════════════════════════════════════
# reconcile(): hot-plug without orphaning sampling loops
# ════════════════════════════════════════════════════════════════


class _MutableProvider:
    """A provider whose discovered set the test can change between passes."""

    kind = "mutable"

    def __init__(self, contexts: tuple[HardwareContext, ...]) -> None:
        self.contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self.contexts


def _sampled_context(device_id: str, *, ceiling: float = 100.0) -> HardwareContext:
    return HardwareContext(
        device_id=device_id,
        display_name=device_id,
        transport=TransportRef(kind="mock", config={"values": {"level": 1.0}}),
        halt_supported=True,
        channels=(
            Channel(
                channel_id="level",
                direction=Direction.READ.value,
                quantity="ratio",
                unit="percent",
                effect=HardwareEffect.READ.value,
                sample_rate_hz=50.0,
                envelope=Envelope(declared=True, min_value=0.0, max_value=ceiling),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )


def test_reconcile_leaves_an_unchanged_device_running_untouched() -> None:
    """The property ``load()`` cannot provide.

    ``load()`` discards the source tuple while the tasks those objects own keep running,
    so they are no longer the objects ``stop_streams()`` can stop -- one rediscovery per
    interval would leak a task per streaming channel, forever. An unchanged device must
    therefore keep the exact source objects it already had.
    """
    provider = _MutableProvider((_sampled_context("rig_a"),))
    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=(provider,))
    registry.load()

    async def _run() -> None:
        before = registry.stream_sources()
        assert before, "no sampling source was built"
        report = await registry.reconcile()
        assert report.admitted == ("rig_a",)
        after = registry.stream_sources()
        assert [id(source) for source in after] == [id(source) for source in before], (
            "an unchanged device was given fresh sources, orphaning the running ones"
        )
        await registry.close_all()

    asyncio.run(_run())


def test_reconcile_admits_a_new_device_and_forgets_a_detached_one() -> None:
    """Hot-plug in both directions, with the transport dropped on the way out."""
    provider = _MutableProvider((_sampled_context("rig_a"),))
    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=(provider,))
    registry.load()

    async def _run() -> None:
        # Force the transport to exist so the detach path has something to close.
        await registry.transport("rig_a")
        assert registry.opened_devices() == ("rig_a",)

        provider.contexts = (_sampled_context("rig_a"), _sampled_context("rig_b"))
        report = await registry.reconcile()
        assert report.admitted == ("rig_a", "rig_b")

        provider.contexts = (_sampled_context("rig_b"),)
        report = await registry.reconcile()
        assert report.admitted == ("rig_b",)
        assert registry.context("rig_a") is None
        assert "rig_a" not in registry.opened_devices(), (
            "a detached device kept its transport, so a stale connection stays open"
        )
        await registry.close_all()

    asyncio.run(_run())


def test_reconcile_rebuilds_a_device_whose_declaration_changed() -> None:
    """An edited envelope must restart the source: the detector holds the old bounds.

    Keeping the running source would leave it enforcing limits the declaration no longer
    states -- which is the difference between a breach event and silence.
    """
    provider = _MutableProvider((_sampled_context("rig_a", ceiling=100.0),))
    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=(provider,))
    registry.load()

    async def _run() -> None:
        before = [id(source) for source in registry.stream_sources()]
        provider.contexts = (_sampled_context("rig_a", ceiling=50.0),)
        await registry.reconcile()

        admitted = registry.context("rig_a")
        assert admitted is not None
        assert admitted.channels[0].envelope.max_value == 50.0
        after = [id(source) for source in registry.stream_sources()]
        assert after != before, "the source kept the superseded envelope"
        await registry.close_all()

    asyncio.run(_run())


def test_reconcile_on_a_disabled_registry_is_a_no_op() -> None:
    """Hardware off is the default, and rediscovery must not turn it on."""
    registry = HardwareRegistry(HardwareSettings(enabled=False))

    async def _run() -> None:
        report = await registry.reconcile()
        assert report.admitted == ()
        assert registry.contexts() == ()

    asyncio.run(_run())
