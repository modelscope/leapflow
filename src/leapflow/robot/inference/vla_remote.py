# Copyright (c) Alibaba, Inc. and its affiliates.
"""Remote inference strategy delegating to a ``PolicyServer``.

Adapts :class:`~leapflow.hardware.transports.vla_inference.VLAInferenceTransport`
to the :class:`~leapflow.robot.inference.strategy.InferenceStrategy` contract.
The model runs on a GPU server, so this process needs no GPU and no torch --
the transport is imported lazily and opened on first use, mirroring the
construct-cheap / connect-late pattern of the local strategy.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from leapflow.robot.inference.strategy import (
    ComputeBudget,
    ComputeProfile,
    InferenceResult,
)

logger = logging.getLogger(__name__)

__all__ = ["VLARemoteStrategy"]


class VLARemoteStrategy:
    """Inference strategy delegating to a remote PolicyServer.

    Wraps VLAInferenceTransport for the InferenceStrategy interface.
    """

    strategy_id = "vla_remote"

    def __init__(
        self,
        server_address: str,
        *,
        model_name: str = "",
        timeout_s: float = 10.0,
    ) -> None:
        self._address = server_address
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._transport: Any = None  # VLAInferenceTransport, created lazily
        self._connected = False

    @property
    def compute_profile(self) -> ComputeProfile:
        return ComputeProfile(
            latency_range_ms=(20.0, 500.0),
            supports_chunking=True,
            gpu_required=False,  # GPU is on the server side
            max_chunk_size=100,
        )

    async def infer(
        self,
        observation: Mapping[str, Any],
        *,
        budget: ComputeBudget | None = None,
    ) -> InferenceResult:
        """Delegate to VLAInferenceTransport.write("inference", ...)."""
        await self._ensure_open()

        chunk_size = max(1, int(budget.chunk_size) if budget is not None else 1)
        payload: dict[str, Any] = {"observation": dict(observation)}
        if self._model_name:
            payload["model_name"] = self._model_name
        if chunk_size > 1:
            payload["chunk_size"] = chunk_size

        outcome = await self._transport.write("inference", payload)
        if not getattr(outcome, "ok", False):
            raise RuntimeError(
                f"remote inference failed: {getattr(outcome, 'error', '')}"
            )

        action = outcome.readback.value if outcome.readback is not None else None
        raw = getattr(outcome, "raw", None) or {}
        latency_ms = float(raw.get("latency_ms", 0.0))

        return InferenceResult(
            action=action,
            latency_ms=latency_ms,
            confidence=1.0,
            chunk_size=chunk_size,
            metadata={"strategy": self.strategy_id, "address": self._address},
        )

    async def reset(self) -> None:
        """Reset is a no-op for a stateless remote server contract."""
        return None

    async def close(self) -> None:
        """Close the underlying transport, if one has been opened.

        Called from ``PhysicalSkillPlugin`` scope teardown so a session that
        went through a remote inference path does not leave the gRPC channel
        holding sockets after the plugin's effect scope tears down.  The
        method is idempotent and swallows transport errors -- teardown must
        not surface secondary failures.
        """
        transport = self._transport
        self._transport = None
        self._connected = False
        if transport is None:
            return
        close = getattr(transport, "close", None)
        if close is None:
            return
        try:
            await close()
        except Exception:  # noqa: BLE001 - teardown must not propagate
            logger.debug(
                "VLARemoteStrategy: transport close raised for %s",
                self._address, exc_info=True,
            )

    async def _ensure_open(self) -> None:
        """Create and open the remote transport lazily on first use."""
        if self._connected and self._transport is not None:
            return
        if self._transport is None:
            try:
                from leapflow.hardware.transports.vla_inference import (
                    VLAInferenceTransport,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "VLAInferenceTransport is not available"
                ) from exc
            self._transport = VLAInferenceTransport(
                server_address=self._address,
                model_name=self._model_name,
                timeout_s=self._timeout_s,
            )

        from leapflow.hardware.context import (
            ContextProvenance,
            ContextSource,
            HardwareContext,
            TransportRef,
        )

        ctx = HardwareContext(
            device_id=f"vla_inference.{self.strategy_id}",
            display_name="VLA inference server",
            transport=TransportRef(
                kind="vla_inference",
                config={"server_address": self._address},
            ),
            channels=(),
            halt_supported=True,
            provenance=ContextProvenance(source=ContextSource.IMPORTED.value),
        )
        status = await self._transport.open(ctx)
        self._connected = bool(getattr(status, "connected", False))
        if not self._connected:
            raise RuntimeError(
                "Remote VLA inference transport failed to connect: "
                f"{getattr(status, 'detail', '')}"
            )
