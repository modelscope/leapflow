"""Helpers for multimodal OpenAI-compatible chat messages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

MessageDict = Dict[str, Any]


def build_user_message_text(text: str) -> MessageDict:
    """Build a simple user message with string content."""
    return {"role": "user", "content": text}


def build_assistant_message(text: str) -> MessageDict:
    """Build a simple assistant message."""
    return {"role": "assistant", "content": text}


def build_user_message_multimodal(
    text: str,
    *,
    images_base64: Optional[Sequence[str]] = None,
    image_urls: Optional[Sequence[str]] = None,
    image_mime: str = "image/png",
) -> MessageDict:
    """Build a multimodal user message (OpenAI-compatible content parts).

    Args:
        text: Primary user text.
        images_base64: Raw base64 payloads; prefixed as data URLs.
        image_urls: HTTP(S) image URLs.
        image_mime: MIME type used for data URL encoding of base64 images.

    Returns:
        A chat message dict suitable for Chat Completions APIs.
    """
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for b64 in images_base64 or ():
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{b64}"},
            }
        )
    for url in image_urls or ():
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": parts}


def build_system_message(text: str) -> MessageDict:
    """Build a system message."""
    return {"role": "system", "content": text}
