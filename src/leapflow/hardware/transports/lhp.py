# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP transport: the host side of the hardware fabric.

Adapts an LHP node (``tcp://host:port``, ``unix:///path``, or an in-process
``LhpNodeServer`` object) onto the six-method ``HardwareTransport`` contract,
plus the ``BatchTransport`` side protocol. The registry, tools, safety policy,
and streaming above it cannot tell an LHP device from a locally attached one --
that is the point: remoteness is a transport configuration fact, not a second
architecture.

Two contracts carry special weight here:

- **Identity localization.** The node's ``device_id`` is local to the node
  (``host``); the host admits it under a namespaced id (``jetson__host``). Every
  reading that crosses the wire is relabelled on arrival and its ``monotonic_at``
  is re-stamped from the host clock -- a remote monotonic counter is meaningless
  locally, while ``observed_at`` is wall-clock and travels as observed.

- **The write error table (frozen spec §6).** An error frame is proof the
  command could not be attempted; ``internal_error``, ``timeout`` and
  ``connection_closed`` are the exceptions -- after those a write may have
  reached the device, so it degrades to ``WriteOutcome(unknown)`` and blocks
  replay instead of raising. An error is not proof that nothing happened.

``probe`` and ``halt`` are deliberately not serialised by callers (see
``HardwareTransport``): this transport keeps them responsive by issuing each
request on the shared client as its own message, so the node's per-connection
concurrency applies end to end.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.lhp.client import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DirectLhpClient,
    SocketLhpClient,
)
from leapflow.hardware.lhp.protocol import (
    WRITE_EFFECT_UNKNOWN_CODES,
    LhpRemoteError,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_UNKNOWN,
    BatchReading,
    BatchWriteOutcome,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

logger = logging.getLogger(__name__)


class LhpTransport:
    """One device served by a remote (or embedded) LHP node."""

    kind = "lhp"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self._server = config.get("server")  # direct binding (in-process)
        self._endpoint = str(config.get("endpoint") or "")
        self._token = str(config.get("token") or "")
        self._expected_node_id = str(config.get("node_id") or "")
        self._remote_device_id = str(config.get("remote_device_id") or "")
        self._timeout_s = float(config.get("timeout_s") or DEFAULT_REQUEST_TIMEOUT_S)
        self._client: DirectLhpClient | SocketLhpClient | None = None
        self._context: HardwareContext | None = None
        self._connected = False

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Connect to the node and complete the hello exchange. Idempotent."""
        if self._connected:
            return await self.probe()
        if self._server is None and not self._endpoint:
            raise TransportError(
                "LHP transport needs either an 'endpoint' (tcp://host:port, unix:///path) "
                "or a direct 'server' object",
                failure_code="lhp_binding_missing",
            )
        self._context = context
        if not self._remote_device_id:
            # The pre-namespacing id: the host context id is what the node
            # advertised, unless the fleet manager said otherwise.
            self._remote_device_id = context.device_id
        try:
            client = self._build_client()
            hello = await client.connect()
        except LhpRemoteError as exc:
            raise TransportError(
                f"LHP open failed: {exc}", failure_code=f"lhp_{exc.code}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise TransportError(f"LHP open failed: {exc}", failure_code="lhp_connect_failed") from exc
        node_id = str((hello or {}).get("node_id") or "")
        if self._expected_node_id and node_id and node_id != self._expected_node_id:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask the mismatch
                logger.debug("LHP client close after node mismatch failed", exc_info=True)
            raise TransportError(
                f"LHP node identity mismatch: expected {self._expected_node_id!r}, got {node_id!r}",
                failure_code="lhp_node_mismatch",
            )
        self._client = client
        self._connected = True
        return TransportStatus(
            connected=True,
            halt_supported=bool(context.halt_supported),
            detail=f"lhp node {node_id or '?'}",
        )

    async def close(self) -> TransportStatus:
        """Release the connection. Must never raise."""
        client, self._client = self._client, None
        self._connected = False
        if client is not None:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                logger.warning("LHP client close failed: %s", exc, exc_info=True)
        return TransportStatus(connected=False, halt_supported=False, detail="closed")

    # ── Data plane ──

    async def read(self, channel_id: str) -> Reading:
        client = self._require_client()
        try:
            result = await client.request(
                "read", {"device_id": self._remote_device_id, "channel_id": channel_id}
            )
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        if not isinstance(result, Mapping):
            raise TransportError(
                "LHP read returned a malformed result", failure_code="lhp_malformed_result"
            )
        return self._localize_reading(result, channel_id)

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        client = self._require_client()
        try:
            result = await client.request(
                "write",
                {"device_id": self._remote_device_id, "channel_id": channel_id, "value": value},
            )
        except LhpRemoteError as exc:
            if exc.code in WRITE_EFFECT_UNKNOWN_CODES:
                # The command may have reached the device; a raise would invite a retry.
                return WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_UNKNOWN,
                    error=str(exc),
                    failure_code=exc.code,
                )
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        if not isinstance(result, Mapping):
            raise TransportError(
                "LHP write returned a malformed result", failure_code="lhp_malformed_result"
            )
        outcome = self._outcome_from_payload(result)
        if outcome.readback is not None and outcome.readback.channel_id != channel_id:
            # A readback for another channel is not evidence for this write.
            # The write itself may nonetheless have landed, so this degrades to
            # ``unknown`` rather than raising -- an error is not proof either way.
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=(
                    f"readback came back for channel {outcome.readback.channel_id!r}, "
                    f"not {channel_id!r}"
                ),
                failure_code="lhp_result_mismatch",
            )
        return outcome

    async def probe(self) -> TransportStatus:
        """Health check. Never raises; a dead link is reported as disconnected."""
        client = self._client
        if not self._connected or client is None:
            return TransportStatus(connected=False, detail="lhp transport is not open")
        try:
            result = await client.request("probe", {"device_id": self._remote_device_id})
        except LhpRemoteError as exc:
            return TransportStatus(
                connected=False, halt_supported=False, detail=f"{exc.code}: {exc}"
            )
        if not isinstance(result, Mapping):
            return TransportStatus(connected=False, detail="malformed probe result")
        return TransportStatus(
            connected=bool(result.get("connected", False)),
            halt_supported=bool(result.get("halt_supported", False)),
            detail=str(result.get("detail") or ""),
            latency_ms=float(result.get("latency_ms") or 0.0),
        )

    async def halt(self) -> TransportStatus:
        """Emergency stop. Never raises and is never serialised by callers."""
        client = self._client
        if not self._connected or client is None:
            return TransportStatus(connected=False, halt_supported=False, detail="lhp transport is not open")
        try:
            result = await client.request("halt", {"device_id": self._remote_device_id})
        except LhpRemoteError as exc:
            logger.error(
                "LHP halt for %r failed: %s", self._remote_device_id, exc, exc_info=True
            )
            return TransportStatus(
                connected=False, halt_supported=False, detail=f"{exc.code}: {exc}"
            )
        if not isinstance(result, Mapping):
            return TransportStatus(connected=False, halt_supported=False, detail="malformed halt result")
        return TransportStatus(
            connected=bool(result.get("connected", False)),
            halt_supported=bool(result.get("halt_supported", False)),
            detail=str(result.get("detail") or ""),
        )

    # ── Batch side-protocol ──

    async def read_batch(self, channel_ids: tuple[str, ...]) -> BatchReading:
        client = self._require_client()
        try:
            result = await client.request(
                "read_batch",
                {"device_id": self._remote_device_id, "channel_ids": list(channel_ids)},
            )
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        if not isinstance(result, Mapping):
            raise TransportError(
                "LHP read_batch returned a malformed result", failure_code="lhp_malformed_result"
            )
        raw_readings = result.get("readings")
        readings: list[Reading] = []
        if isinstance(raw_readings, (list, tuple)):
            for item in raw_readings:
                if isinstance(item, Mapping):
                    channel_id = str(item.get("channel_id") or "")
                    readings.append(self._localize_reading(item, channel_id))
        observed = _as_positive_float(result.get("observed_at")) or _latest_observed_at(readings)
        return BatchReading(
            device_id=self._local_device_id(),
            readings=tuple(readings),
            observed_at=observed if observed > 0 else time.time(),
        )

    async def write_batch(self, commands: tuple[tuple[str, Any], ...]) -> BatchWriteOutcome:
        client = self._require_client()
        try:
            result = await client.request(
                "write_batch",
                {
                    "device_id": self._remote_device_id,
                    "commands": [[channel_id, value] for channel_id, value in commands],
                },
            )
        except LhpRemoteError as exc:
            if exc.code in WRITE_EFFECT_UNKNOWN_CODES:
                return BatchWriteOutcome(ok=False, outcomes=(), side_effect_state=SIDE_EFFECT_UNKNOWN)
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        if not isinstance(result, Mapping):
            raise TransportError(
                "LHP write_batch returned a malformed result", failure_code="lhp_malformed_result"
            )
        raw_outcomes = result.get("outcomes")
        outcomes: list[WriteOutcome] = []
        if isinstance(raw_outcomes, (list, tuple)):
            for index, item in enumerate(raw_outcomes):
                if not isinstance(item, Mapping):
                    continue
                outcome = self._outcome_from_payload(item)
                expected_channel = commands[index][0] if index < len(commands) else ""
                if (
                    outcome.readback is not None
                    and expected_channel
                    and outcome.readback.channel_id != expected_channel
                ):
                    # Same evidence rule as the single-write path: a readback
                    # for a different channel is evidence for nothing, and
                    # whether this command landed is now unknown -- so it
                    # degrades to ``unknown`` rather than mislabelling.
                    outcome = WriteOutcome(
                        ok=False,
                        side_effect_state=SIDE_EFFECT_UNKNOWN,
                        error=(
                            "readback came back for channel "
                            f"{outcome.readback.channel_id!r}, not {expected_channel!r}"
                        ),
                        failure_code="lhp_result_mismatch",
                    )
                outcomes.append(outcome)
        return BatchWriteOutcome(
            ok=bool(result.get("ok", False)),
            outcomes=tuple(outcomes),
            side_effect_state=str(result.get("side_effect_state") or SIDE_EFFECT_UNKNOWN),
        )

    # ── Internals ──

    def _build_client(self) -> DirectLhpClient | SocketLhpClient:
        if self._server is not None:
            return DirectLhpClient(self._server)
        return SocketLhpClient(
            self._endpoint,
            token=self._token,
            timeout_s=self._timeout_s,
        )

    def _require_client(self) -> DirectLhpClient | SocketLhpClient:
        client = self._client
        if not self._connected or client is None:
            raise TransportError(
                f"LHP transport for {self._remote_device_id or '?'} is not open",
                failure_code="transport_not_open",
            )
        return client

    def _local_device_id(self) -> str:
        context = self._context
        return context.device_id if context is not None else self._remote_device_id

    def _localize_reading(self, payload: Mapping[str, Any], channel_id: str) -> Reading:
        """Rebuild a wire reading under the host-side identity and host clock.

        The channel identity is verified rather than relabelled: a reading that
        comes back for a different channel than was asked for is a protocol
        fault, and quietly fixing its label would present the wrong value as the
        right one.
        """
        wire_channel = str(payload.get("channel_id") or "")
        if wire_channel and wire_channel != channel_id:
            raise TransportError(
                f"LHP node returned a reading for channel {wire_channel!r}, not {channel_id!r}",
                failure_code="lhp_result_mismatch",
            )
        observed = _as_positive_float(payload.get("observed_at"))
        return Reading(
            device_id=self._local_device_id(),
            channel_id=channel_id,
            value=payload.get("value"),
            quantity=str(payload.get("quantity") or ""),
            unit=str(payload.get("unit") or ""),
            observed_at=observed if observed > 0 else time.time(),
            sequence=int(payload.get("sequence") or 0),
            quality=str(payload.get("quality") or Quality.OK.value),
        )

    def _outcome_from_payload(self, payload: Mapping[str, Any]) -> WriteOutcome:
        raw_readback = payload.get("readback")
        readback: Reading | None = None
        if isinstance(raw_readback, Mapping):
            readback = self._localize_reading(
                raw_readback, str(raw_readback.get("channel_id") or "")
            )
        return WriteOutcome(
            ok=bool(payload.get("ok", False)),
            side_effect_state=str(payload.get("side_effect_state") or SIDE_EFFECT_UNKNOWN),
            readback=readback,
            settled=bool(payload.get("settled", False)),
            error=str(payload.get("error") or ""),
            failure_code=str(payload.get("failure_code") or ""),
            preview=bool(payload.get("preview", False)),
        )


def _as_positive_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    numeric = float(value)
    return numeric if numeric > 0 else 0.0


def _latest_observed_at(readings: list[Reading]) -> float:
    """Fallback batch stamp: the newest per-reading wall clock, or 0.0."""
    return max((reading.observed_at for reading in readings), default=0.0)


def build_transport(config: Mapping[str, Any] | None = None) -> LhpTransport:
    """Factory registered in the transport table."""
    return LhpTransport(config)


__all__ = ["LhpTransport", "build_transport"]
