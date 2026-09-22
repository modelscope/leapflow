# Copyright (c) Alibaba, Inc. and its affiliates.
"""Gateway module — platform adapter management and message routing.

Public API:

- Protocol types: ``MessageSource``, ``InboundMessage``, ``OutboundContent``, etc.
- ``PlatformAdapter`` Protocol + ``PlatformAdapterMixin`` for graceful degradation
- ``GatewayServer`` for adapter lifecycle and message routing
- ``ManifestLoader`` for declarative platform discovery
- ``CredentialVault`` for vault-backed ``secret://`` credential refs
- ``SessionKey`` / ``build_session_key`` for structured session routing
- ``GatewayRouter`` for per-session LLM processing of inbound messages
"""
from leapflow.gateway.adapter_registry import (
    BuiltinAdapterPlugin,
    GatewayAdapterPlugin,
    GatewayAdapterRegistry,
)
from leapflow.gateway.config_store import GatewayConfig, GatewayConfigStore
from leapflow.gateway.credential_vault import CredentialVault
from leapflow.gateway.events import (
    GatewayMessageReceived,
    GatewaySessionCreated,
    GatewaySessionEnded,
)
from leapflow.gateway.manifest import ManifestLoader, PlatformManifest
from leapflow.gateway.mixin import PlatformAdapterMixin
from leapflow.gateway.protocol import (
    InboundMessage,
    MediaAttachment,
    MessageSource,
    OutboundContent,
    PlatformAdapter,
    PlatformCapabilities,
    PlatformStatus,
    SendResult,
    SendTarget,
)
from leapflow.gateway.router import GatewayRouter
from leapflow.gateway.server import GatewayServer
from leapflow.gateway.session_router import SessionKey, build_session_key
from leapflow.gateway.validators import register_validator, validate_credentials

__all__ = [
    # Protocol types
    "MessageSource",
    "MediaAttachment",
    "InboundMessage",
    "SendTarget",
    "OutboundContent",
    "SendResult",
    "PlatformStatus",
    # Adapter contract
    "PlatformAdapter",
    "PlatformAdapterMixin",
    "PlatformCapabilities",
    # Adapter plugin registry
    "GatewayAdapterPlugin",
    "GatewayAdapterRegistry",
    "BuiltinAdapterPlugin",
    # Events
    "GatewayMessageReceived",
    "GatewaySessionCreated",
    "GatewaySessionEnded",
    # Core components
    "GatewayRouter",
    "GatewayServer",
    "ManifestLoader",
    "PlatformManifest",
    "CredentialVault",
    "GatewayConfig",
    "GatewayConfigStore",
    # Session routing
    "SessionKey",
    "build_session_key",
    # Validators
    "register_validator",
    "validate_credentials",
]
