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

import base64
import binascii
import logging
import math
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.lhp.authority import WRITE_MODE_PERMIT_REQUIRED
from leapflow.hardware.lhp.client import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DirectLhpClient,
    SocketLhpClient,
)
from leapflow.hardware.lhp.permit import PERMIT_PARAM, request_digest
from leapflow.hardware.lhp.protocol import (
    CAP_FRAME_FRESHNESS_V1,
    CAP_PERMIT_V1,
    ERR_UNSUPPORTED_CAPABILITY,
    FAILURE_PERMIT_REQUIRED,
    MAX_FRAME_AGE_MS,
    MAX_FRAME_DATA_BYTES,
    MAX_FRAME_RATE_HZ,
    MAX_FRAME_SEQUENCE,
    MAX_FRAME_WIDTH,
    SAFE_FRAME_MEDIA_TYPES,
    WRITE_EFFECT_UNKNOWN_CODES,
    LhpRemoteError,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    BatchReading,
    BatchWriteOutcome,
    FrameReading,
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
        self._permit_issuer = config.get("permit_issuer")  # PermitIssuer (not serialized)
        self._client: DirectLhpClient | SocketLhpClient | None = None
        self._context: HardwareContext | None = None
        self._connected = False
        self._capabilities: frozenset[str] = frozenset()
        self._frame_channels: set[str] = set()
        self._remote_write_mode: str = ""  # from hello
        self._remote_node_id: str = ""  # from hello

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
            raise TransportError(f"LHP open failed: {exc}", failure_code=f"lhp_{exc.code}") from exc
        except (OSError, ValueError) as exc:
            raise TransportError(
                f"LHP open failed: {exc}", failure_code="lhp_connect_failed"
            ) from exc
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
        raw_capabilities = hello.get("capabilities", ()) if isinstance(hello, Mapping) else ()
        self._capabilities = frozenset(
            str(item) for item in raw_capabilities if isinstance(item, str) and item
        )
        self._remote_write_mode = (
            str(hello.get("write_mode") or "") if isinstance(hello, Mapping) else ""
        )
        self._remote_node_id = node_id or self._expected_node_id
        self._client = client
        self._connected = True
        return TransportStatus(
            connected=True,
            halt_supported=bool(context.halt_supported),
            detail=f"lhp node {node_id or '?'}",
            metadata={
                "protocol": hello.get("protocol"),
                "capabilities": sorted(self._capabilities),
            },
        )

    async def close(self) -> TransportStatus:
        """Release remote frame claims, then the connection. Must never raise."""
        client = self._client
        if client is not None and self._connected and "release_frame" in self._capabilities:
            for channel_id in tuple(self._frame_channels):
                try:
                    await client.request(
                        "release_frame",
                        {"device_id": self._remote_device_id, "channel_id": channel_id},
                    )
                except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                    logger.warning(
                        "LHP remote frame release failed for %r/%r: %s",
                        self._remote_device_id,
                        channel_id,
                        exc,
                        exc_info=True,
                    )
        self._frame_channels.clear()
        self._capabilities = frozenset()
        self._client = None
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

    async def read_frame(
        self,
        channel_id: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
        min_sequence: int = 0,
        after_monotonic: float | None = None,
    ) -> FrameReading:
        """Fetch one bounded frame while preserving host-side identity and timebase.

        Freshness bounds cross the wire as the node's own sequence and an *age*:
        ``after_monotonic`` is this host's clock, meaningless on the node, so it is
        sent as "no older than now minus the bound" and the node re-anchors it to its
        own clock -- stricter by the link latency, never looser.
        """
        client = self._require_client()
        required = {"read_frame", "release_frame"}
        if not required.issubset(self._capabilities):
            raise TransportError(
                "LHP node does not advertise bounded frame read and release capabilities",
                failure_code=ERR_UNSUPPORTED_CAPABILITY,
            )
        if max_width < 0 or max_width > MAX_FRAME_WIDTH:
            raise TransportError(
                "frame width is outside protocol bounds", failure_code="bad_request"
            )
        if quality < 0 or quality > 100:
            raise TransportError(
                "frame quality is outside protocol bounds", failure_code="bad_request"
            )
        if not math.isfinite(float(fps)) or fps < 0 or fps > MAX_FRAME_RATE_HZ:
            raise TransportError(
                "frame rate is outside protocol bounds", failure_code="bad_request"
            )
        params: dict[str, Any] = {
            "device_id": self._remote_device_id,
            "channel_id": channel_id,
            "max_width": int(max_width),
            "quality": int(quality),
            "fps": float(fps),
        }
        params.update(self._freshness_params(min_sequence, after_monotonic))
        try:
            result = await client.request("read_frame", params)
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        frame = self._frame_from_payload(result, channel_id)
        self._frame_channels.add(channel_id)
        return frame

    def _freshness_params(
        self, min_sequence: int, after_monotonic: float | None
    ) -> dict[str, Any]:
        """Translate host freshness bounds to wire parameters, or refuse them."""
        if not min_sequence and after_monotonic is None:
            return {}
        if CAP_FRAME_FRESHNESS_V1 not in self._capabilities:
            raise TransportError(
                "LHP node cannot honour frame freshness bounds; upgrade the node",
                failure_code=ERR_UNSUPPORTED_CAPABILITY,
            )
        if isinstance(min_sequence, bool) or not 0 <= int(min_sequence) <= MAX_FRAME_SEQUENCE:
            raise TransportError(
                "frame sequence bound is outside protocol bounds", failure_code="bad_request"
            )
        params: dict[str, Any] = {}
        if min_sequence:
            params["min_sequence"] = int(min_sequence)
        if after_monotonic is not None:
            age_ms = (time.monotonic() - float(after_monotonic)) * 1000.0
            if not math.isfinite(age_ms):
                raise TransportError(
                    "frame freshness bound is not a finite instant", failure_code="bad_request"
                )
            # A future bound means "after now"; an ancient one is clamped, which is stricter.
            params["max_age_ms"] = min(max(0.0, age_ms), MAX_FRAME_AGE_MS)
        return params

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        client = self._require_client()
        params: dict[str, Any] = {
            "device_id": self._remote_device_id,
            "channel_id": channel_id,
            "value": value,
        }
        refusal = self._attach_permit("write", params)
        if refusal:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=refusal,
                failure_code=FAILURE_PERMIT_REQUIRED,
            )
        try:
            result = await client.request("write", params)
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
        hub = result.get("frame_hub")
        return TransportStatus(
            connected=bool(result.get("connected", False)),
            halt_supported=bool(result.get("halt_supported", False)),
            detail=str(result.get("detail") or ""),
            latency_ms=float(result.get("latency_ms") or 0.0),
            metadata={"frame_hub": dict(hub)} if isinstance(hub, Mapping) else {},
        )

    async def halt(self) -> TransportStatus:
        """Emergency stop. Never raises and is never serialised by callers.

        Every branch that could not put the stop on the wire answers
        ``halt_delivered=False`` while still reporting the device's *declared*
        halt capability: a remote arm whose link is gone may still be travelling,
        and the fleet has to be able to tell that apart from a camera that
        released its own capture.
        """
        client = self._client
        if not self._connected or client is None:
            return TransportStatus(
                connected=False,
                halt_supported=self._declared_halt_support(),
                halt_delivered=False,
                detail="lhp transport is not open; the stop was not sent",
            )
        try:
            result = await client.request("halt", {"device_id": self._remote_device_id})
        except LhpRemoteError as exc:
            logger.error("LHP halt for %r failed: %s", self._remote_device_id, exc, exc_info=True)
            return TransportStatus(
                connected=False,
                halt_supported=self._declared_halt_support(),
                halt_delivered=False,
                detail=f"{exc.code}: {exc}",
            )
        if not isinstance(result, Mapping):
            return TransportStatus(
                connected=False,
                halt_supported=self._declared_halt_support(),
                halt_delivered=False,
                detail="malformed halt result",
            )
        return TransportStatus(
            connected=bool(result.get("connected", False)),
            halt_supported=bool(result.get("halt_supported", False)),
            halt_delivered=bool(result.get("halt_delivered", True)),
            detail=str(result.get("detail") or ""),
        )

    def _declared_halt_support(self) -> bool:
        """The device's declared stop capability, independent of the link state."""
        return bool(getattr(self._context, "halt_supported", False))

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
        params: dict[str, Any] = {
            "device_id": self._remote_device_id,
            "commands": [[channel_id, value] for channel_id, value in commands],
        }
        refusal = self._attach_permit("write_batch", params)
        if refusal:
            outcome = WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=refusal,
                failure_code=FAILURE_PERMIT_REQUIRED,
            )
            return BatchWriteOutcome(
                ok=False,
                outcomes=tuple(outcome for _ in commands),
                side_effect_state=SIDE_EFFECT_NONE,
            )
        try:
            result = await client.request("write_batch", params)
        except LhpRemoteError as exc:
            if exc.code in WRITE_EFFECT_UNKNOWN_CODES:
                return BatchWriteOutcome(
                    ok=False, outcomes=(), side_effect_state=SIDE_EFFECT_UNKNOWN
                )
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

    # ── Remote control (control.v1) ──

    async def open_control(
        self,
        *,
        device_id: str,
        mode: str,
        frequency_hz: float,
        channels: list[str],
        targets: Any = None,
        waypoints: Any = None,
        lease_s: float = 1.0,
        comm_loss_policy: str = "halt",
    ) -> dict[str, Any]:
        """Open a control session on the remote node and start auto-renewal."""
        from leapflow.hardware.lhp.protocol import CAP_CONTROL_V1

        client = self._require_client()
        if CAP_CONTROL_V1 not in self._capabilities:
            raise TransportError(
                "remote node does not advertise control.v1",
                failure_code="remote_control_unsupported",
            )
        params: dict[str, Any] = {
            "device_id": self._remote_device_id,
            "mode": mode,
            "frequency_hz": frequency_hz,
            "channels": channels,
            "lease_s": lease_s,
            "comm_loss_policy": comm_loss_policy,
        }
        if targets is not None:
            params["targets"] = targets
        if waypoints is not None:
            params["waypoints"] = waypoints
        refusal = self._attach_permit("control.open", params)
        if refusal:
            raise TransportError(refusal, failure_code=FAILURE_PERMIT_REQUIRED)
        try:
            result = await client.request("control.open", params)
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        if not isinstance(result, Mapping) or not result.get("ok"):
            if not isinstance(result, Mapping):
                raise TransportError(
                    "LHP control.open returned a malformed result",
                    failure_code="lhp_malformed_result",
                )
            raise TransportError(
                str(result.get("error") or "control.open refused"),
                failure_code=str(result.get("failure_code") or "control_open_failed"),
            )

        control_id = str(result.get("control_id") or "")
        actual_lease = float(result.get("lease_s") or lease_s)
        # Start auto-renewal task
        self._start_renewal_loop(control_id, actual_lease)
        return dict(result)

    async def renew_control(self, control_id: str) -> dict[str, Any]:
        """Renew the lease on a remote control session."""
        client = self._require_client()
        try:
            result = await client.request("control.renew", {"control_id": control_id})
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        return dict(result) if isinstance(result, Mapping) else {}

    async def update_control(
        self, control_id: str, *, targets: Any = None, waypoints: Any = None
    ) -> dict[str, Any]:
        """Update targets/waypoints on a running session."""
        client = self._require_client()
        params: dict[str, Any] = {"control_id": control_id}
        if targets is not None:
            params["targets"] = targets
        if waypoints is not None:
            params["waypoints"] = waypoints
        refusal = self._attach_permit("control.update", params)
        if refusal:
            raise TransportError(refusal, failure_code=FAILURE_PERMIT_REQUIRED)
        try:
            result = await client.request("control.update", params)
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        return dict(result) if isinstance(result, Mapping) else {}

    async def close_control(self, control_id: str) -> dict[str, Any]:
        """Close a remote control session and stop auto-renewal."""
        self._stop_renewal_loop()
        client = self._require_client()
        try:
            result = await client.request("control.close", {"control_id": control_id})
        except LhpRemoteError as exc:
            raise TransportError(str(exc) or exc.code, failure_code=exc.code) from exc
        return dict(result) if isinstance(result, Mapping) else {}

    def _start_renewal_loop(self, control_id: str, lease_s: float) -> None:
        """Start a background task that renews the lease at lease_s/3 intervals."""
        import asyncio as _asyncio

        self._stop_renewal_loop()  # cancel any existing
        interval = max(0.1, lease_s / 3.0)

        async def _renew_loop() -> None:
            while True:
                await _asyncio.sleep(interval)
                try:
                    await self.renew_control(control_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "LHP control renewal failed for %s: %s",
                        control_id[:8],
                        exc,
                    )
                    return  # stop renewing; node will fire comm_loss_policy

        try:
            loop = _asyncio.get_running_loop()
            self._renewal_task = loop.create_task(
                _renew_loop(), name=f"lhp-control-renew-{control_id[:8]}"
            )
        except RuntimeError:
            logger.debug("no running event loop for control renewal")

    def _stop_renewal_loop(self) -> None:
        """Cancel the auto-renewal background task."""
        task = getattr(self, "_renewal_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._renewal_task = None

    # ── Permit integration ──

    def _attach_permit(self, method: str, params: dict[str, Any]) -> str:
        """Sign *params* for *method* when the node requires a permit.

        The permit binds ``request_digest(method, params)`` -- exactly what the
        node will recompute from the request it receives -- so it must be
        attached last, after every other parameter is final. Returns a host-side
        refusal message (nothing is sent), or ``""`` when the request may go.
        """
        if self._remote_write_mode != WRITE_MODE_PERMIT_REQUIRED:
            return ""
        if CAP_PERMIT_V1 not in self._capabilities:
            return ""  # the node refuses on its own; its message names the missing keys
        from leapflow.hardware.execution_authorization import current_authorization

        auth = current_authorization()
        issuer = self._permit_issuer
        if auth is None or issuer is None:
            return (
                "remote node requires a signed permit but no execution authorization "
                "is active (host-side refusal; nothing was sent)"
            )
        permit = issuer.issue(
            audience=self._remote_node_id,
            session_id=auth.session_id,
            workspace=auth.workspace,
            approval_id=auth.approval_id,
            digest=request_digest(method, params),
        )
        params[PERMIT_PARAM] = permit.to_dict()
        return ""

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

    def _frame_from_payload(self, payload: Any, channel_id: str) -> FrameReading:
        if not isinstance(payload, Mapping):
            raise TransportError(
                "LHP read_frame returned a malformed result", failure_code="lhp_malformed_result"
            )
        wire_device = str(payload.get("device_id") or "")
        wire_channel = str(payload.get("channel_id") or "")
        if wire_device != self._remote_device_id or wire_channel != channel_id:
            raise TransportError(
                "LHP node returned a frame for a different device or channel",
                failure_code="lhp_result_mismatch",
            )
        encoded = payload.get("data_b64")
        if not isinstance(encoded, str) or not encoded:
            raise TransportError("LHP frame has no payload", failure_code="lhp_malformed_result")
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise TransportError(
                "LHP frame payload is not valid base64", failure_code="lhp_malformed_result"
            ) from exc
        if not data or len(data) > MAX_FRAME_DATA_BYTES:
            raise TransportError(
                "LHP frame payload is empty or exceeds the protocol limit",
                failure_code="lhp_malformed_result",
            )
        size_bytes = payload.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes != len(data)
        ):
            raise TransportError(
                "LHP frame size metadata does not match its payload",
                failure_code="lhp_result_mismatch",
            )
        media_type = str(payload.get("media_type") or "")
        if media_type not in SAFE_FRAME_MEDIA_TYPES:
            raise TransportError(
                f"LHP frame media type {media_type!r} is not supported",
                failure_code=ERR_UNSUPPORTED_CAPABILITY,
            )
        width = _bounded_nonnegative_int(payload.get("width"), MAX_FRAME_WIDTH, "width")
        height = _bounded_nonnegative_int(payload.get("height"), MAX_FRAME_WIDTH, "height")
        sequence = _bounded_nonnegative_int(payload.get("sequence"), 2**63 - 1, "sequence")
        observed = _as_positive_float(payload.get("observed_at"))
        return FrameReading(
            device_id=self._local_device_id(),
            channel_id=channel_id,
            data=data,
            media_type=media_type,
            width=width,
            height=height,
            observed_at=observed if observed > 0 else time.time(),
            monotonic_at=time.monotonic(),
            sequence=sequence,
            quality=str(payload.get("quality") or Quality.OK.value),
            metadata=_frame_metadata(payload.get("metadata")),
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


def _frame_metadata(value: Any) -> Mapping[str, Any]:
    """Accept only JSON-object metadata returned by a remote frame producer."""
    if not isinstance(value, Mapping):
        return {}
    # Frame metadata is diagnostic/interpretive only. A shallow copy prevents a
    # remote response object from retaining mutable references in host state.
    return dict(value)


def _bounded_nonnegative_int(value: Any, upper: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > upper:
        raise TransportError(
            f"LHP frame {label} is outside protocol bounds",
            failure_code="lhp_malformed_result",
        )
    return value


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
