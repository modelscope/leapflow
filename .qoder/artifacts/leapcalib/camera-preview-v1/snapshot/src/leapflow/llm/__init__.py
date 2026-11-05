# Copyright (c) Alibaba, Inc. and its affiliates.
"""LLM providers and message utilities."""

from leapflow.llm.base import LLMChatResponse, LLMProvider
from leapflow.llm.message_builder import (
    build_assistant_message,
    build_system_message,
    build_user_message_multimodal,
    build_user_message_text,
)
from leapflow.llm.openai_provider import OpenAIChat, OpenAIChatResponse
from leapflow.llm.provider_chain import (
    AuxiliaryClient,
    CredentialPool,
    FailoverChain,
    ProviderConfig,
    ProviderMetadata,
    parse_credential_pools,
    parse_provider_configs,
)
from leapflow.llm.model_capabilities import (
    ModelCapabilities,
    ModelCapabilityRegistry,
)
from leapflow.llm.provider_registry import (
    LLMProviderPlugin,
    LLMProviderRegistry,
    get_default_registry,
    get_scoped_default_registry,
    reset_default_registry,
    ENTRY_POINT_GROUP,
)

__all__ = [
    "LLMProvider",
    "LLMChatResponse",
    "OpenAIChat",
    "OpenAIChatResponse",
    "FailoverChain",
    "ProviderConfig",
    "ProviderMetadata",
    "CredentialPool",
    "AuxiliaryClient",
    "ModelCapabilities",
    "ModelCapabilityRegistry",
    "LLMProviderPlugin",
    "LLMProviderRegistry",
    "get_default_registry",
    "get_scoped_default_registry",
    "reset_default_registry",
    "ENTRY_POINT_GROUP",
    "parse_provider_configs",
    "parse_credential_pools",
    "build_assistant_message",
    "build_system_message",
    "build_user_message_text",
    "build_user_message_multimodal",
]
