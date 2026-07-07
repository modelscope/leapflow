"""Unified platform abstraction layer — RPC, event normalization, and host adapters."""

from leapflow.platform.protocol import (
    EventHandler,
    EventTypes,
    HostRpc,
    Methods,
    RpcError,
    decode_packet,
    encode_packet,
    make_event,
    make_request,
    make_response_err,
    make_response_ok,
)
from leapflow.platform.cua_client import CuaDriverClient
from leapflow.platform.event_bus import EventBus
from leapflow.platform.facade import VirtualSystemInterface
from leapflow.platform.normalizer import EventNormalizer
from leapflow.platform.observers import ObservationDaemon, Observer, ObserverConfig

__all__ = [
    "CuaDriverClient",
    "EventBus",
    "EventHandler",
    "EventNormalizer",
    "EventTypes",
    "HostRpc",
    "Methods",
    "ObservationDaemon",
    "Observer",
    "ObserverConfig",
    "RpcError",
    "VirtualSystemInterface",
    "decode_packet",
    "encode_packet",
    "make_event",
    "make_request",
    "make_response_err",
    "make_response_ok",
]
