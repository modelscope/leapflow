"""Default implementations for optional ``PlatformAdapter`` capabilities.

Adapters mix this in to get graceful degradation for methods they do not
natively support.  The ``Protocol`` defines the contract; this mixin
provides sensible defaults.  Adapters override only what they can do
natively.

Separated from the Protocol to avoid the Hermes 5.6K-line base-class
problem — each adapter composes behaviour freely.
"""
from __future__ import annotations

from typing import Sequence

from leapflow.gateway.protocol import OutboundContent, SendResult, SendTarget


class PlatformAdapterMixin:
    """Optional mixin — graceful degradation for unsupported methods.

    Usage::

        class FeishuAdapter(PlatformAdapterMixin):
            supports_async_delivery = True
            max_message_length = 8000
            ...

            # Override natively-supported methods; inherit degradation for the rest.
    """

    supports_async_delivery: bool = True
    splits_long_messages: bool = False
    max_message_length: int = 4000

    # ── Message editing ──────────────────────────────────────

    async def edit_message(
        self,
        target: SendTarget,
        message_id: str,
        content: OutboundContent,
        *,
        finalize: bool = False,
    ) -> SendResult:
        return SendResult(ok=False, error="edit not supported")

    # ── Rich media (degrade to text) ─────────────────────────

    async def send_image(
        self,
        target: SendTarget,
        image_url: str,
        caption: str = "",
        **kw: object,
    ) -> SendResult:
        text = f"[Image] {image_url}"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(target, OutboundContent(text=text))  # type: ignore[attr-defined]

    async def send_document(
        self,
        target: SendTarget,
        file_name: str,
        caption: str = "",
        **kw: object,
    ) -> SendResult:
        """Degrade to a friendly notice — never leaks host file paths."""
        text = f"[Document: {file_name}]"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(target, OutboundContent(text=text))  # type: ignore[attr-defined]

    # ── Interactive UX (degrade to numbered list) ─────────────

    async def send_clarify(
        self,
        target: SendTarget,
        question: str,
        choices: Sequence[str],
        **kw: object,
    ) -> SendResult:
        lines = [question, ""]
        for i, choice in enumerate(choices, 1):
            lines.append(f"{i}. {choice}")
        return await self.send(target, OutboundContent(text="\n".join(lines)))  # type: ignore[attr-defined]

    # ── Typing indicators (no-op) ────────────────────────────

    async def send_typing(self, target: SendTarget, **kw: object) -> None:
        pass

    async def stop_typing(self, target: SendTarget) -> None:
        pass
