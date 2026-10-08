# Copyright (c) Alibaba, Inc. and its affiliates.
"""VLA inference transport: remote policy server as an HCP virtual device.

A VLA (Vision-Language-Action) model running on a GPU server is presented
to LeapFlow as a hardware device with three channels: ``model_info``
(read-only metadata), ``inference`` (write to submit observation, returns
action), and ``latency`` (read-only timing).

The transport communicates with a ``PolicyServer`` via gRPC, or
with any inference endpoint that accepts the same protobuf contract.
The transport speaks HTTP/JSON when ``server_address`` starts with ``http``
and gRPC otherwise; this is a construction-time configuration choice, not an
automatic gRPC-to-HTTP fallback.

This is deliberately a CONFIGURE effect, not ACTUATE: inference produces
a computation result, not a physical motion.  The physical effect happens
when the caller writes the returned action to an actuator transport.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

logger = logging.getLogger(__name__)

# Channels exposed by this virtual device.
_CH_MODEL_INFO = "model_info"
_CH_INFERENCE = "inference"
_CH_LATENCY = "latency"
_READABLE_CHANNELS = frozenset({_CH_MODEL_INFO, _CH_LATENCY})
_WRITABLE_CHANNELS = frozenset({_CH_INFERENCE})

# Retry back-off schedule in seconds.
_BACKOFF_SCHEDULE = (1.0, 2.0)


class VLAInferenceTransport:
    """HardwareTransport backed by a remote VLA inference server.

    Construction:
    - server_address: "host:port" for gRPC, or "http://host:port" for HTTP
    - model_name: optional model identifier for multi-model servers
    - timeout_s: per-request timeout (default 10.0)
    - max_retries: retry count on transient failures (default 2)
    """

    kind: str = "vla_inference"

    def __init__(
        self,
        server_address: str,
        *,
        model_name: str = "",
        timeout_s: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self._address = server_address
        self._model_name = model_name
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._connected = False
        self._last_latency_ms: float = 0.0
        self._model_metadata: dict[str, Any] = {}
        self._channel: Any = None  # gRPC channel
        self._stub: Any = None  # gRPC stub
        self._http_session: Any = None  # aiohttp.ClientSession
        self._use_http = server_address.startswith("http")
        self._context: HardwareContext | None = None
        self._sequence: dict[str, int] = {}
        self._pending_rpcs: list[Any] = []  # in-flight gRPC calls for halt()

    # ------------------------------------------------------------------
    # HardwareTransport (6 methods)
    # ------------------------------------------------------------------

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Establish gRPC channel (or HTTP session) to the inference server.

        For gRPC: create insecure channel, create stub, call health check.
        For HTTP: create aiohttp session, call /health endpoint.
        If neither gRPC nor HTTP client libraries are available, returns
        a disconnected status rather than raising.
        """
        self._context = context

        if self._use_http:
            ok = await self._open_http()
        else:
            ok = await self._open_grpc()

        if not ok:
            return TransportStatus(
                connected=False,
                halt_supported=False,
                detail="failed to connect to inference server",
                metadata={"address": self._address},
            )

        # Attempt initial health check and metadata fetch.
        try:
            healthy = await self._health_check()
        except Exception as exc:
            logger.warning(
                "VLA inference server health check failed during open: %s",
                exc, exc_info=True,
            )
            healthy = False

        self._connected = healthy
        if not healthy:
            await self._teardown()
            return TransportStatus(
                connected=False,
                halt_supported=True,
                detail="inference server health check failed",
                metadata={"address": self._address},
            )

        logger.info(
            "VLA inference transport opened: %s (model=%s, http=%s)",
            self._address, self._model_name or "<default>", self._use_http,
        )
        return await self.probe()

    async def close(self) -> TransportStatus:
        """Close gRPC channel / HTTP session. Idempotent, never raises."""
        await self._teardown()
        self._connected = False
        return TransportStatus(
            connected=False,
            halt_supported=True,
            detail="vla_inference transport closed",
        )

    async def read(self, channel_id: str) -> Reading:
        """Read model_info or latency channel.

        model_info: returns cached model metadata dict.
        latency: returns last inference latency in milliseconds.
        inference: raises TransportError (write-only semantics).
        """
        self._require_open(channel_id)

        if channel_id == _CH_INFERENCE:
            raise TransportError(
                f"channel {_CH_INFERENCE!r} is write-only; submit observations "
                "via write() instead",
                failure_code="channel_write_only",
            )

        if channel_id not in _READABLE_CHANNELS:
            raise TransportError(
                f"unknown channel {channel_id!r}",
                failure_code="unknown_channel",
            )

        device_id = self._context.device_id if self._context else ""

        if channel_id == _CH_MODEL_INFO:
            return Reading(
                device_id=device_id,
                channel_id=channel_id,
                value=dict(self._model_metadata),
                quantity="metadata",
                unit="",
                sequence=self._next_seq(channel_id),
                quality=Quality.OK.value,
            )

        # _CH_LATENCY
        return Reading(
            device_id=device_id,
            channel_id=channel_id,
            value=self._last_latency_ms,
            quantity="latency",
            unit="ms",
            sequence=self._next_seq(channel_id),
            quality=Quality.OK.value,
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Submit an inference request to the server.

        Only the 'inference' channel is writable.
        *value*: dict with 'observation' key containing sensor data.
        Returns: WriteOutcome with readback containing the action vector.

        side_effect_state is always NONE (computation has no physical effect).
        """
        self._require_open(channel_id)

        if channel_id != _CH_INFERENCE:
            raise TransportError(
                f"channel {channel_id!r} is read-only",
                failure_code="channel_read_only",
            )

        if not isinstance(value, Mapping):
            raise TransportError(
                "inference write expects a dict with an 'observation' key",
                failure_code="invalid_inference_input",
            )

        observation = value.get("observation")
        if observation is None:
            raise TransportError(
                "inference write requires an 'observation' key in the value dict",
                failure_code="missing_observation",
            )

        # Execute inference with timing.
        t0 = time.monotonic()
        try:
            if self._use_http:
                result = await self._infer_http(dict(value))
            else:
                result = await self._infer_grpc(dict(value))
        except TransportError:
            raise
        except Exception as exc:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"inference failed: {exc}",
                failure_code="vla_inference_failed",
            )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._last_latency_ms = elapsed_ms

        device_id = self._context.device_id if self._context else ""
        readback = Reading(
            device_id=device_id,
            channel_id=channel_id,
            value=result,
            quantity="action",
            unit="",
            sequence=self._next_seq(channel_id),
            quality=Quality.OK.value,
        )
        return WriteOutcome(
            ok=True,
            side_effect_state=SIDE_EFFECT_NONE,
            readback=readback,
            settled=True,
            raw={"latency_ms": elapsed_ms},
        )

    async def probe(self) -> TransportStatus:
        """Health check: ping the server, report GPU availability if known."""
        if not self._connected:
            return TransportStatus(
                connected=False,
                halt_supported=True,
                detail="vla_inference transport not connected",
                metadata={"address": self._address},
            )
        try:
            healthy = await self._health_check()
        except Exception as exc:
            logger.debug("VLA probe failed: %s", exc, exc_info=True)
            healthy = False

        meta: dict[str, Any] = {
            "address": self._address,
            "model_name": self._model_name,
            "use_http": self._use_http,
            "last_latency_ms": self._last_latency_ms,
        }
        if self._model_metadata:
            meta["model_metadata"] = self._model_metadata
        return TransportStatus(
            connected=healthy,
            halt_supported=True,
            detail="vla_inference transport" if healthy else "health check failed",
            latency_ms=self._last_latency_ms,
            metadata=meta,
        )

    async def halt(self) -> TransportStatus:
        """Cancel any in-flight inference requests.

        For gRPC: cancel pending RPCs.
        Returns halt_supported=True (cancellation is always possible).
        """
        cancelled = 0
        for rpc in self._pending_rpcs:
            try:
                rpc.cancel()
                cancelled += 1
            except Exception:
                pass
        self._pending_rpcs.clear()
        return TransportStatus(
            connected=self._connected,
            halt_supported=True,
            detail=f"halted ({cancelled} in-flight RPCs cancelled)"
            if cancelled
            else "halted (no in-flight RPCs)",
        )

    # ------------------------------------------------------------------
    # Internal: connection lifecycle
    # ------------------------------------------------------------------

    async def _open_grpc(self) -> bool:
        """Establish a gRPC channel. Returns False when grpc is unavailable."""
        try:
            import grpc  # type: ignore[import-untyped]
            import grpc.aio  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "grpcio not installed; cannot connect to %s via gRPC. "
                "Install with: pip install grpcio",
                self._address,
            )
            return False

        try:
            self._channel = grpc.aio.insecure_channel(self._address)
            # A lightweight connectivity check -- wait for the channel to
            # leave IDLE.  This does not guarantee the server is serving,
            # but catches obviously wrong addresses early.
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "gRPC channel to %s did not become ready within %.1fs",
                self._address, self._timeout,
            )
            await self._teardown()
            return False
        except Exception as exc:
            logger.warning(
                "gRPC channel creation failed for %s: %s",
                self._address, exc, exc_info=True,
            )
            await self._teardown()
            return False
        return True

    async def _open_http(self) -> bool:
        """Create an aiohttp session. Returns False when aiohttp is unavailable."""
        try:
            import aiohttp  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "aiohttp not installed; cannot connect to %s via HTTP. "
                "Install with: pip install aiohttp",
                self._address,
            )
            return False
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        except Exception as exc:
            logger.warning(
                "aiohttp session creation failed: %s", exc, exc_info=True,
            )
            return False
        return True

    async def _teardown(self) -> None:
        """Release gRPC channel and HTTP session. Idempotent, never raises."""
        if self._channel is not None:
            try:
                await self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._stub = None
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None
        self._pending_rpcs.clear()

    # ------------------------------------------------------------------
    # Internal: inference execution
    # ------------------------------------------------------------------

    async def _infer_grpc(self, observation: dict) -> dict:
        """Execute inference via gRPC with timeout and retry.

        Uses the generic ``UnaryUnaryMultiCallable`` interface so that the
        transport works with any protobuf service that accepts a JSON-like
        dict and returns one.  When a proper proto stub is available (e.g.
        a ``PolicyServer``), it is used directly.
        """
        if self._channel is None:
            raise TransportError(
                "gRPC channel is not open", failure_code="transport_not_open",
            )

        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                backoff = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
                await asyncio.sleep(backoff)

            try:
                result = await asyncio.wait_for(
                    self._grpc_call(observation),
                    timeout=self._timeout,
                )
                return result
            except asyncio.TimeoutError:
                last_exc = TimeoutError(
                    f"gRPC inference timed out after {self._timeout}s "
                    f"(attempt {attempt + 1}/{1 + self._max_retries})"
                )
                logger.debug("gRPC inference timeout (attempt %d)", attempt + 1)
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "gRPC inference error (attempt %d): %s",
                    attempt + 1, exc, exc_info=True,
                )

        raise TransportError(
            f"gRPC inference failed after {1 + self._max_retries} attempts: {last_exc}",
            failure_code="vla_grpc_inference_failed",
        )

    async def _grpc_call(self, observation: dict) -> dict:
        """Single gRPC call.  Wraps the stub or falls back to a generic unary RPC.

        Subclasses or future versions can override this when the proto
        contract is known.
        """
        try:
            import grpc  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TransportError(
                "grpcio unavailable", failure_code="grpc_unavailable",
            ) from exc

        # Build a generic request payload.  This is compatible with servers
        # that accept a JSON-serialisable dict (e.g. via grpc-json-proxy or
        # the async_inference module's generic endpoint).
        import json

        # Build the request payload.  The full write value is forwarded so
        # server-side fields (task, chunk_size, ...) are preserved rather than
        # silently dropped; only ``observation`` was retained before.
        payload = dict(observation) if isinstance(observation, dict) else {"observation": observation}
        if self._model_name and "model_name" not in payload:
            payload["model_name"] = self._model_name
        request_bytes = json.dumps(payload).encode()

        # Use the channel's generic unary-unary callable.
        method = self._channel.unary_unary(
            "/policy.PolicyService/Infer",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        call = method(request_bytes)
        self._pending_rpcs.append(call)
        try:
            response_bytes = await call
        finally:
            try:
                self._pending_rpcs.remove(call)
            except ValueError:
                pass

        # Parse the response.
        try:
            result = json.loads(response_bytes)
        except (json.JSONDecodeError, TypeError):
            result = {"raw": response_bytes.decode(errors="replace")}

        # Update model metadata if the server provides it.
        if isinstance(result, dict) and "model_info" in result:
            self._model_metadata.update(result["model_info"])

        return result

    async def _infer_http(self, observation: dict) -> dict:
        """Execute inference via HTTP/JSON with timeout and retry."""
        if self._http_session is None:
            raise TransportError(
                "HTTP session is not open", failure_code="transport_not_open",
            )

        url = f"{self._address.rstrip('/')}/infer"
        # Forward the full write value so server-side fields (task, chunk_size,
        # ...) are preserved rather than silently dropped.
        payload = dict(observation) if isinstance(observation, dict) else {"observation": observation}
        if self._model_name and "model_name" not in payload:
            payload["model_name"] = self._model_name

        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                backoff = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
                await asyncio.sleep(backoff)

            try:
                result = await asyncio.wait_for(
                    self._http_post(url, payload),
                    timeout=self._timeout,
                )
                return result
            except asyncio.TimeoutError:
                last_exc = TimeoutError(
                    f"HTTP inference timed out after {self._timeout}s "
                    f"(attempt {attempt + 1}/{1 + self._max_retries})"
                )
                logger.debug("HTTP inference timeout (attempt %d)", attempt + 1)
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "HTTP inference error (attempt %d): %s",
                    attempt + 1, exc, exc_info=True,
                )

        raise TransportError(
            f"HTTP inference failed after {1 + self._max_retries} attempts: {last_exc}",
            failure_code="vla_http_inference_failed",
        )

    async def _http_post(self, url: str, payload: dict) -> dict:
        """Single HTTP POST.  Returns the parsed JSON response."""
        async with self._http_session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise TransportError(
                    f"HTTP inference returned status {resp.status}: {text[:200]}",
                    failure_code="vla_http_error",
                )
            result = await resp.json()

        # Update model metadata if the server provides it.
        if isinstance(result, dict) and "model_info" in result:
            self._model_metadata.update(result["model_info"])

        return result

    # ------------------------------------------------------------------
    # Internal: health check
    # ------------------------------------------------------------------

    async def _health_check(self) -> bool:
        """Ping the server health endpoint.

        gRPC: attempt a unary call to a health-check method.
        HTTP: GET /health and expect a 200.
        """
        if self._use_http:
            return await self._health_check_http()
        return await self._health_check_grpc()

    async def _health_check_grpc(self) -> bool:
        """gRPC health check via the standard health service or a lightweight ping."""
        if self._channel is None:
            return False
        try:
            import json

            method = self._channel.unary_unary(
                "/policy.PolicyService/Health",
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            response_bytes = await asyncio.wait_for(
                method(b"{}"), timeout=min(self._timeout, 5.0),
            )
            # Parse metadata if available.
            try:
                data = json.loads(response_bytes)
                if isinstance(data, dict):
                    self._model_metadata.update(data.get("model_info", {}))
            except (json.JSONDecodeError, TypeError):
                pass
            return True
        except Exception as exc:
            logger.debug("gRPC health check failed: %s", exc)
            return False

    async def _health_check_http(self) -> bool:
        """HTTP health check: GET /health."""
        if self._http_session is None:
            return False
        url = f"{self._address.rstrip('/')}/health"
        try:
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                        if isinstance(data, dict):
                            self._model_metadata.update(data.get("model_info", {}))
                    except Exception:
                        pass
                    return True
                return False
        except Exception as exc:
            logger.debug("HTTP health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _require_open(self, channel_id: str) -> None:
        """Raise TransportError when the transport is not connected."""
        if not self._connected:
            raise TransportError(
                f"transport for {channel_id!r} is not open",
                failure_code="transport_not_open",
            )

    def _next_seq(self, channel_id: str) -> int:
        """Advance and return the sequence counter for a channel."""
        seq = self._sequence.get(channel_id, 0) + 1
        self._sequence[channel_id] = seq
        return seq


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def build_transport(config: Mapping[str, Any]) -> VLAInferenceTransport:
    """Factory for the transport registry.

    Config keys:
    - server_address (str, required): "host:port" or "http://host:port"
    - model_name (str, optional): model identifier
    - timeout_s (float, optional): per-request timeout
    - max_retries (int, optional): retry count
    """
    server_address = str(config.get("server_address", ""))
    if not server_address:
        raise TransportError(
            "vla_inference transport requires 'server_address' in config",
            failure_code="vla_config_missing_address",
        )
    return VLAInferenceTransport(
        server_address=server_address,
        model_name=str(config.get("model_name", "")),
        timeout_s=float(config.get("timeout_s", 10.0) or 10.0),
        max_retries=int(config.get("max_retries", 2) or 2),
    )


__all__ = [
    "VLAInferenceTransport",
    "build_transport",
]
