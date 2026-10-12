# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP -- the version-negotiated LeapFlow Hardware Protocol.

The hardware fabric's data plane. One protocol, two roles: ``LhpNodeServer`` on
the node side (wraps a ``HardwareRegistry``), ``DirectLhpClient`` /
``SocketLhpClient`` on the host side, and ``LhpTransport`` (in
``transports/lhp.py``) adapting the host registry's transport contract onto
either binding.

The frozen v1 specification is ``temp/devices/plan/02-lhp-wire-protocol-v1.md``;
``protocol.py`` is the executable source for negotiated versions, capabilities,
error codes, frame shape, and the write-error classification.
"""

from __future__ import annotations

from leapflow.hardware.lhp.client import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    DirectLhpClient,
    SocketLhpClient,
)
from leapflow.hardware.lhp.protocol import (
    ERR_BAD_REQUEST,
    ERR_CONNECTION_CLOSED,
    ERR_FRAME_TOO_LARGE,
    ERR_INTERNAL_ERROR,
    ERR_PROTOCOL_MISMATCH,
    ERR_TIMEOUT,
    ERR_TRANSPORT_ERROR,
    ERR_UNAUTHORIZED,
    ERR_UNKNOWN_CHANNEL,
    ERR_UNKNOWN_DEVICE,
    ERR_UNSUPPORTED_CAPABILITY,
    FAILURE_BATCH_ABORTED,
    FAILURE_NOT_WRITABLE,
    FAILURE_SAFETY_DENIED,
    LHP_CAPABILITIES,
    LHP_DEFAULT_PORT,
    LHP_METHODS,
    LHP_PROTOCOL_VERSION,
    LHP_SUPPORTED_VERSIONS,
    LHP_V1_CAPABILITIES,
    LHP_V2_CAPABILITIES,
    MAX_FRAME_BYTES,
    MAX_FRAME_DATA_BYTES,
    MAX_FRAME_RATE_HZ,
    MAX_FRAME_WIDTH,
    SAFE_FRAME_MEDIA_TYPES,
    WRITE_EFFECT_UNKNOWN_CODES,
    LhpProtocolError,
    LhpRemoteError,
    capabilities_for_version,
    decode_frame,
    encode_frame,
    error_response,
    is_valid_device_id,
    make_request,
    ok_response,
    response_error,
)
from leapflow.hardware.lhp.server import LhpNodeServer

__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_S",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DirectLhpClient",
    "ERR_BAD_REQUEST",
    "ERR_CONNECTION_CLOSED",
    "ERR_FRAME_TOO_LARGE",
    "ERR_INTERNAL_ERROR",
    "ERR_PROTOCOL_MISMATCH",
    "ERR_TIMEOUT",
    "ERR_TRANSPORT_ERROR",
    "ERR_UNAUTHORIZED",
    "ERR_UNKNOWN_CHANNEL",
    "ERR_UNKNOWN_DEVICE",
    "ERR_UNSUPPORTED_CAPABILITY",
    "FAILURE_BATCH_ABORTED",
    "FAILURE_NOT_WRITABLE",
    "FAILURE_SAFETY_DENIED",
    "LHP_CAPABILITIES",
    "LHP_DEFAULT_PORT",
    "LHP_METHODS",
    "LHP_PROTOCOL_VERSION",
    "LHP_SUPPORTED_VERSIONS",
    "LHP_V1_CAPABILITIES",
    "LHP_V2_CAPABILITIES",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_DATA_BYTES",
    "MAX_FRAME_RATE_HZ",
    "MAX_FRAME_WIDTH",
    "SAFE_FRAME_MEDIA_TYPES",
    "WRITE_EFFECT_UNKNOWN_CODES",
    "LhpNodeServer",
    "LhpProtocolError",
    "LhpRemoteError",
    "SocketLhpClient",
    "capabilities_for_version",
    "decode_frame",
    "encode_frame",
    "error_response",
    "is_valid_device_id",
    "make_request",
    "ok_response",
    "response_error",
]
