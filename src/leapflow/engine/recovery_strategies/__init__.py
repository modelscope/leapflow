"""Built-in recovery strategies for the agent loop recovery coordinator.

Each strategy implements the RecoveryStrategy Protocol and encapsulates
a single-responsibility recovery pattern (context compression, failover,
credential rotation, etc.).
"""
from __future__ import annotations

from leapflow.engine.recovery_strategies.context_compress import ContextCompressStrategy
from leapflow.engine.recovery_strategies.credential_rotate import CredentialRotateStrategy
from leapflow.engine.recovery_strategies.jittered_retry import JitteredRetryStrategy
from leapflow.engine.recovery_strategies.multimodal_strip import MultimodalStripStrategy
from leapflow.engine.recovery_strategies.native_to_text import NativeToTextFallbackStrategy
from leapflow.engine.recovery_strategies.provider_failover import ProviderFailoverStrategy
from leapflow.engine.recovery_strategies.thinking_disable import ThinkingDisableStrategy
from leapflow.engine.recovery_strategies.tool_schema_expand import ToolSchemaExpandStrategy

__all__ = [
    "ContextCompressStrategy",
    "CredentialRotateStrategy",
    "JitteredRetryStrategy",
    "MultimodalStripStrategy",
    "NativeToTextFallbackStrategy",
    "ProviderFailoverStrategy",
    "ThinkingDisableStrategy",
    "ToolSchemaExpandStrategy",
    "default_strategies",
]


def default_strategies() -> list:
    """Return all built-in strategies in priority order (lowest priority number first)."""
    return [
        ContextCompressStrategy(),
        MultimodalStripStrategy(),
        ProviderFailoverStrategy(),
        CredentialRotateStrategy(),
        ThinkingDisableStrategy(),
        NativeToTextFallbackStrategy(),
        ToolSchemaExpandStrategy(),
        JitteredRetryStrategy(),
    ]
