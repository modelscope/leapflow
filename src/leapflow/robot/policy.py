# Copyright (c) Alibaba, Inc. and its affiliates.
"""Policy inference abstractions for LeapRobot.

Provides the ``PreTrainedPolicy`` base class (inference-only subset) and
the ``make_policy`` factory function.  All training-related functionality
(forward pass with loss, optimizer params, PEFT/LoRA, FSDP, gradient
checkpointing, Hub push) has been removed.

torch is an optional dependency — the module is importable without it,
but instantiating a policy requires torch at runtime.
"""

from __future__ import annotations

import abc
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, ClassVar, TypeVar

# ---------------------------------------------------------------------------
# torch — optional dependency
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import Tensor, nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="PreTrainedPolicy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTION_KEY = "action"


def _require_torch() -> None:
    """Raise early if torch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for policy inference.  "
            "Install it with: pip install leapflow[robot]"
        )


# ---------------------------------------------------------------------------
# PreTrainedPolicy — inference subset
# ---------------------------------------------------------------------------

class PreTrainedPolicy(nn.Module if _TORCH_AVAILABLE else object, abc.ABC):  # type: ignore[misc]
    """Base class for inference-only policy models.

    Concrete policies must set ``name`` (str) as a class variable and
    implement the four abstract methods:

    - :meth:`reset`
    - :meth:`select_action`
    - :meth:`predict_action_chunk`
    - :meth:`forward_inference`

    Loading from a local directory of safetensors weights is supported
    via :meth:`from_pretrained`.
    """

    name: ClassVar[str | None] = None

    # Attribute names that ``drop_queued_actions`` clears.
    _action_queue_attrs: ClassVar[tuple[str, ...]] = ("_queues", "_action_queue")

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        _require_torch()
        super().__init__()
        self.config: dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # Persistence — inference-only (load, not save)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_path: str | Path,
        *,
        config: dict[str, Any] | None = None,
        device: str = "cpu",
        strict: bool = False,
        **kwargs: Any,
    ) -> T:
        """Load a pre-trained policy from a local directory.

        The directory must contain a ``model.safetensors`` file (or
        compatible weight format).  The policy is placed in eval mode.

        Args:
            pretrained_path: Path to the directory with saved weights.
            config: Optional config dict; when ``None`` a default is
                used.
            device: Target device (``"cpu"``, ``"cuda"``).
            strict: Whether to require an exact key match when loading.
            **kwargs: Extra keyword arguments forwarded to the
                constructor.

        Returns:
            An instance of this policy, weights loaded and in eval mode.
        """
        _require_torch()

        model_dir = Path(pretrained_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Pretrained path does not exist or is not a directory: {model_dir}"
            )

        # Locate weight file
        safetensors_file = model_dir / "model.safetensors"
        if not safetensors_file.exists():
            raise FileNotFoundError(
                f"Weight file not found: {safetensors_file}"
            )

        instance = cls(config=config, **kwargs)

        # Load weights via safetensors
        try:
            from safetensors.torch import load_model as load_model_safetensor
            missing, unexpected = load_model_safetensor(
                instance, str(safetensors_file), strict=strict, device=device,
            )
            if missing:
                logger.warning("Missing keys when loading weights: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys when loading weights: %s", unexpected)
        except ImportError:
            # Fallback to vanilla torch.load
            state_dict = torch.load(
                safetensors_file, map_location=device, weights_only=True,
            )
            instance.load_state_dict(state_dict, strict=strict)

        instance.to(device)
        instance.eval()
        return instance

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset episode state (caches, queues, hidden states).

        Called at the beginning of every episode.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        """Return one action to execute in the environment.

        For chunking policies this pops from an internal queue,
        only calling :meth:`predict_action_chunk` when the queue is
        empty.

        Args:
            batch: Preprocessed observation dictionary.
            **kwargs: Policy-specific inference options (e.g. noise).

        Returns:
            A single action tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        """Predict a chunk of future actions for the given observation.

        Child classes using action chunking should use this inside
        :meth:`select_action` to fill the queue.

        Args:
            batch: Preprocessed observation dictionary.
            **kwargs: Policy-specific inference options.

        Returns:
            Tensor of shape ``(chunk_size, action_dim)``.
        """
        raise NotImplementedError

    def forward_inference(self, batch: dict[str, Any]) -> Any:
        """Run a single-step forward pass for inference.

        Default implementation delegates to :meth:`select_action`.
        Override to provide a more efficient single-step path.
        """
        return self.select_action(batch)

    # ------------------------------------------------------------------
    # Action queue management
    # ------------------------------------------------------------------

    def drop_queued_actions(self) -> None:
        """Discard pre-computed actions so the next call recomputes.

        Useful when a mid-episode conditioning change (e.g. new language
        instruction) must take effect immediately rather than after the
        queue drains.
        """
        for attr in self._action_queue_attrs:
            queue = getattr(self, attr, None)
            if isinstance(queue, dict):
                action_deque = queue.get(_ACTION_KEY)
                if action_deque is not None:
                    action_deque.clear()
            elif queue is not None:
                queue.clear()

    def count_queued_actions(self) -> int:
        """Number of pre-computed actions waiting to be served."""
        total = 0
        for attr in self._action_queue_attrs:
            queue = getattr(self, attr, None)
            if isinstance(queue, dict):
                action_deque = queue.get(_ACTION_KEY)
                total += len(action_deque) if action_deque is not None else 0
            elif queue is not None:
                total += len(queue)
        return total


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Registry of known policy classes keyed by name.
_POLICY_REGISTRY: dict[str, type[PreTrainedPolicy]] = {}


def register_policy(cls: type[PreTrainedPolicy]) -> type[PreTrainedPolicy]:
    """Class decorator that registers a policy in the factory.

    Usage::

        @register_policy
        class MyPolicy(PreTrainedPolicy):
            name = "my_policy"
            ...
    """
    if not cls.name:
        raise TypeError(f"{cls.__name__} must define a 'name' class variable")
    _POLICY_REGISTRY[cls.name] = cls
    return cls


def make_policy(
    policy_type: str,
    pretrained_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> PreTrainedPolicy:
    """Instantiate a policy by registered name.

    When *pretrained_path* is supplied the weights are loaded from disk;
    otherwise a freshly initialised instance is returned.

    Args:
        policy_type: Registered policy name (e.g. ``"act"``,
            ``"diffusion"``).
        pretrained_path: Optional directory containing saved weights.
        config: Policy configuration dict.
        device: Target device.
        **kwargs: Extra arguments forwarded to the constructor.

    Returns:
        A ready-to-use policy instance.

    Raises:
        ValueError: If *policy_type* is not registered.
    """
    _require_torch()

    policy_cls = _POLICY_REGISTRY.get(policy_type)
    if policy_cls is None:
        available = sorted(_POLICY_REGISTRY.keys())
        raise ValueError(
            f"Unknown policy type '{policy_type}'.  "
            f"Available: {available}"
        )

    if pretrained_path is not None:
        return policy_cls.from_pretrained(
            pretrained_path, config=config, device=device, **kwargs,
        )

    policy = policy_cls(config=config, **kwargs)
    policy.to(device)
    return policy


__all__ = [
    "PreTrainedPolicy",
    "register_policy",
    "make_policy",
]
