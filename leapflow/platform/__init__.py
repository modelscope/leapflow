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
from leapflow.platform.client import BridgeClient
from leapflow.platform.event_bus import EventBus
from leapflow.platform.facade import VirtualSystemInterface
from leapflow.platform.normalizer import EventNormalizer

__all__ = [
    "BridgeClient",
    "EventBus",
    "EventHandler",
    "EventNormalizer",
    "EventTypes",
    "HostRpc",
    "Methods",
    "RpcError",
    "VirtualSystemInterface",
    "decode_packet",
    "encode_packet",
    "make_event",
    "make_request",
    "make_response_err",
    "make_response_ok",
]
