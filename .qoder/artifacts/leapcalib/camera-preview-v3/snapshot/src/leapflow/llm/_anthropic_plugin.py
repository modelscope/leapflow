# Copyright (c) Alibaba, Inc. and its affiliates.
"""Anthropic LLM provider plugin.

Registers the native Anthropic Messages API provider into the LLM provider
registry.  This module guards the ``anthropic`` SDK import so the plugin
file itself can be safely imported even when the SDK is absent — the
``ImportError`` is raised at import time and caught by
``discover_builtin()`` in ``provider_registry.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from leapflow.llm.base import LLMProvider

# Eagerly verify SDK availability so discover_builtin() gets a clean
# ImportError when the SDK is missing.
from leapflow.llm.anthropic_provider import AnthropicChat, is_anthropic_available  # noqa: F401

if not is_anthropic_available():
    raise ImportError("anthropic SDK is not installed")


class AnthropicPlugin:
    """Plugin for native Anthropic Messages API provider.

    Uses explicit cache_control breakpoints rather than automatic prefix
    caching.  The engine's ``AnthropicCacheStrategy`` generates breakpoint
    markers; this provider passes them through to the Anthropic SDK.

    Config keys:
        api_key: str — Anthropic API key (required)
        model: str — Model identifier, e.g. 'claude-sonnet-4-20250514' (required)
        base_url: str — Optional API endpoint override
                        (e.g. 'https://api.deepseek.com/anthropic')
        max_retries: int — Retry count (default: 3)
        timeout_s: float — Request timeout seconds (default: 180.0)
        max_tokens: int — Max response tokens (default: 8192)
    """

    @property
    def provider_id(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic (Native Messages API)"

    @property
    def supported_models(self) -> List[str]:
        return [
            "claude-*",
            "claude-sonnet-*",
            "claude-haiku-*",
            "claude-opus-*",
        ]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supports_thinking": True,
            "credential_rotation": False,
            "cache_type": "explicit_breakpoint",
            "cache_usage_fields": [
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            ],
        }

    def create_provider(self, config: Dict[str, Any]) -> LLMProvider:
        """Create an AnthropicChat instance from config dict.

        Args:
            config: Must include 'api_key', 'model'.
                    Optional: 'base_url', 'max_retries', 'timeout_s', 'max_tokens'.

        Returns:
            Configured AnthropicChat instance.

        Raises:
            ValueError: If required keys are missing.
            ImportError: If the anthropic SDK is not installed.
        """
        api_key = config.get("api_key")
        model = config.get("model")

        if not api_key:
            raise ValueError("Anthropic provider requires 'api_key' in config")
        if not model:
            raise ValueError("Anthropic provider requires 'model' in config")

        return AnthropicChat(
            api_key=api_key,
            model=model,
            base_url=config.get("base_url"),
            max_retries=int(config.get("max_retries", 3)),
            timeout_s=float(config.get("timeout_s", 180.0)),
            max_tokens=int(config.get("max_tokens", 8192)),
        )


# Module-level singleton for auto-discovery and reload support.
plugin = AnthropicPlugin()
