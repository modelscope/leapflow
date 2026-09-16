# Copyright (c) Alibaba, Inc. and its affiliates.
"""Model capability registry — dynamic context length, feature flags per model.

Provides a single source of truth for model capabilities that the engine,
compressor, and provider chain can query without hardcoding per-model logic.

Design (inspired by hermes model_metadata.py, simplified):
- Static defaults for known model families (pattern-matched)
- Config overrides via ProviderConfig
- Runtime updates from successful API responses (learned context limits)
- Protocol-first for testability
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.config import DEFAULT_LLM_CONTEXT_LENGTH

logger = logging.getLogger(__name__)

_IMAGE_TOKEN_ESTIMATE = 1600


@dataclass(frozen=True)
class ModelCapabilities:
    """Capability descriptor for a specific model."""

    context_length: int = DEFAULT_LLM_CONTEXT_LENGTH
    max_output_tokens: int = 16_384
    supports_tools: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False
    supports_streaming_tools: bool = False
    tokens_per_image: int = _IMAGE_TOKEN_ESTIMATE
    # Whether ``context_length`` is trustworthy enough to cap the user's
    # configured budget. A version-specific entry is authoritative; a
    # family-wide fallback such as "qwen" is not, because it carries whatever
    # the family's largest model happened to support when the entry was written
    # and would silently shrink the window for every later generation. Model
    # names always outrun a static table, so an unrecognised or merely
    # family-matched model defers to the configured budget instead of being
    # clamped by stale data.
    authoritative: bool = True


@runtime_checkable
class CapabilitySource(Protocol):
    """Protocol for capability resolution backends."""

    def resolve(self, model: str, base_url: str = "") -> Optional[ModelCapabilities]: ...


# Version-specific entries are authoritative: the number belongs to that exact
# model. Family-wide entries (matching every generation of a vendor's line) are
# marked non-authoritative — they inform feature flags but must not cap a
# configured budget, since a newer generation usually has a larger window than
# whatever was current when the entry was added. New model names are expected to
# miss this table entirely and fall through to the configured budget; keeping the
# list current is a convenience, never a correctness requirement.
_KNOWN_MODELS: List[tuple[str, ModelCapabilities]] = [
    # OpenAI
    (r"gpt-4o", ModelCapabilities(context_length=128_000, max_output_tokens=16_384,
                                   supports_tools=True, supports_vision=True)),
    (r"gpt-4-turbo", ModelCapabilities(context_length=128_000, max_output_tokens=4_096,
                                        supports_tools=True, supports_vision=True)),
    (r"gpt-4\.1", ModelCapabilities(context_length=1_047_576, max_output_tokens=32_768,
                                     supports_tools=True, supports_vision=True)),
    (r"gpt-3\.5", ModelCapabilities(context_length=16_385, max_output_tokens=4_096,
                                     supports_tools=True)),
    (r"o[134]-", ModelCapabilities(context_length=200_000, max_output_tokens=100_000,
                                    supports_tools=True, supports_thinking=True)),
    # Claude (via proxy)
    (r"claude-3.5-sonnet", ModelCapabilities(context_length=200_000, max_output_tokens=8_192,
                                              supports_tools=True, supports_vision=True,
                                              supports_thinking=False)),
    (r"claude-3-opus", ModelCapabilities(context_length=200_000, max_output_tokens=4_096,
                                          supports_tools=True, supports_vision=True)),
    (r"claude-4", ModelCapabilities(context_length=200_000, max_output_tokens=16_384,
                                     supports_tools=True, supports_vision=True,
                                     supports_thinking=True)),
    # Family-wide fallbacks: feature flags only, context length not authoritative.
    (r"gpt-", ModelCapabilities(max_output_tokens=32_768, supports_tools=True,
                                supports_vision=True, authoritative=False)),
    (r"claude-", ModelCapabilities(max_output_tokens=16_384, supports_tools=True,
                                    supports_vision=True, supports_thinking=True,
                                    authoritative=False)),
    (r"deepseek", ModelCapabilities(max_output_tokens=8_192, supports_tools=True,
                                     supports_thinking=True, authoritative=False)),
    (r"qwen", ModelCapabilities(max_output_tokens=8_192, supports_tools=True,
                                 supports_thinking=True, authoritative=False)),
]


class ModelCapabilityRegistry:
    """Registry resolving model capabilities from config, patterns, and runtime feedback.

    Resolution order:
    1. Explicit overrides (set via register or from ProviderConfig)
    2. Runtime-learned capabilities (from API response usage)
    3. Known model patterns (regex match)
    4. Default capabilities
    """

    def __init__(self, *, default: Optional[ModelCapabilities] = None) -> None:
        self._overrides: Dict[str, ModelCapabilities] = {}
        self._learned: Dict[str, Dict[str, Any]] = {}
        # The fallback carries no model-specific knowledge, so its context length
        # must never cap a configured budget.
        self._default = default or ModelCapabilities(authoritative=False)

    def register(self, model: str, caps: ModelCapabilities) -> None:
        """Register an explicit capability override for a model."""
        self._overrides[model] = caps
        logger.debug("model_caps: registered override for %s", model)

    def resolve(self, model: str, base_url: str = "") -> ModelCapabilities:
        """Resolve capabilities for a model name."""
        if model in self._overrides:
            return self._overrides[model]

        learned = self._learned.get(model)
        pattern_caps = self._match_known(model)

        if pattern_caps and learned:
            return ModelCapabilities(
                context_length=learned.get("context_length", pattern_caps.context_length),
                max_output_tokens=learned.get("max_output_tokens", pattern_caps.max_output_tokens),
                supports_tools=pattern_caps.supports_tools,
                supports_vision=pattern_caps.supports_vision,
                supports_thinking=pattern_caps.supports_thinking,
                supports_streaming_tools=pattern_caps.supports_streaming_tools,
                tokens_per_image=pattern_caps.tokens_per_image,
                # A learned length comes from a real response, so it is
                # authoritative even when the pattern was only a family guess.
                authoritative=(
                    True if "context_length" in learned else pattern_caps.authoritative
                ),
            )

        if pattern_caps:
            return pattern_caps

        if learned:
            return ModelCapabilities(
                context_length=learned.get("context_length", self._default.context_length),
                max_output_tokens=learned.get("max_output_tokens", self._default.max_output_tokens),
                authoritative="context_length" in learned,
            )

        return self._default

    def update_from_usage(self, model: str, usage: Dict[str, int]) -> None:
        """Learn context limits from API response usage fields.

        When prompt_tokens is very high, we know the model supports at least
        that many tokens — useful for probing real limits without a /models API.
        """
        if not usage:
            return
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        if prompt <= 0:
            return

        entry = self._learned.setdefault(model, {})
        current_ctx = entry.get("context_length", 0)
        observed_total = prompt + completion
        if observed_total > current_ctx * 0.8:
            entry["context_length"] = max(current_ctx, int(observed_total * 1.2))

    def context_length(self, model: str) -> int:
        """Shorthand for resolve().context_length."""
        return self.resolve(model).context_length

    def supports_tools(self, model: str) -> bool:
        return self.resolve(model).supports_tools

    def supports_vision(self, model: str) -> bool:
        return self.resolve(model).supports_vision

    def image_token_estimate(self, model: str = "") -> int:
        if model:
            return self.resolve(model).tokens_per_image
        return _IMAGE_TOKEN_ESTIMATE

    @staticmethod
    def _match_known(model: str) -> Optional[ModelCapabilities]:
        model_lower = model.lower()
        for pattern, caps in _KNOWN_MODELS:
            if re.search(pattern, model_lower):
                return caps
        return None
