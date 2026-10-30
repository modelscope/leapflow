# Copyright (c) Alibaba, Inc. and its affiliates.
"""Images returned by tools, shown to the model next to the tool's text result.

A tool result travels as a ``role="tool"`` message whose content is text, so
a tool that has something to *show* (a camera frame, a rendered chart) puts it
under :data:`ATTACHMENTS_KEY` instead::

    {"ok": True, ..., "_attachments": [{"mime": "image/jpeg", "data": "<b64>",
                                         "label": "frame f1, 640x480"}]}

The dispatcher pops the attachments before the result reaches the ledger, the
trace, or the transcript, and after the batch appends one user message that
carries the images. That message is volatile: it is never persisted, sits in
the dynamic tail rather than the cacheable prefix, and older images are
retired to a text placeholder as newer ones arrive or before compression, so
a long session holds at most a fixed number of images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

logger = logging.getLogger(__name__)

ATTACHMENTS_KEY = "_attachments"
ATTACHMENT_FLAG = "_tool_attachment"
IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp"})

# A sanity bound on one encoded image; tools are expected to downsample first.
_MAX_DATA_CHARS = 8_000_000
_MAX_LABEL_CHARS = 200
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


@dataclass(frozen=True)
class ToolAttachment:
    """One image a tool call returned for the model to look at."""

    tool_call_id: str
    tool_name: str
    mime: str
    data: str
    label: str = ""

    @property
    def reference(self) -> str:
        label = f": {self.label}" if self.label else ""
        return f"{self.tool_name} (call {self.tool_call_id}){label}"

    def image_part(self) -> Dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{self.data}"}}


def take_attachments(
    result: Any, *, tool_call_id: str, tool_name: str
) -> tuple[ToolAttachment, ...]:
    """Pop and validate the attachments of a tool result, annotating what was kept.

    The result keeps ``attached_images`` (how many images follow) so the
    model can tie the images to the call, and ``attachments_refused`` when an
    entry was not a supported, base64-encoded image.
    """
    if not isinstance(result, dict) or ATTACHMENTS_KEY not in result:
        return ()
    raw = result.pop(ATTACHMENTS_KEY)
    items = raw if isinstance(raw, list) else [raw]
    kept: List[ToolAttachment] = []
    refused = 0
    for item in items:
        attachment = _attachment(item, tool_call_id=tool_call_id, tool_name=tool_name)
        if attachment is None:
            refused += 1
        else:
            kept.append(attachment)
    if kept:
        result["attached_images"] = len(kept)
    if refused:
        result["attachments_refused"] = refused
    return tuple(kept)


def _attachment(item: Any, *, tool_call_id: str, tool_name: str) -> ToolAttachment | None:
    if not isinstance(item, Mapping):
        return None
    mime = str(item.get("mime") or "").strip().lower()
    data = item.get("data")
    if mime not in IMAGE_MIMES or not isinstance(data, str):
        return None
    if not data or len(data) > _MAX_DATA_CHARS:
        return None
    label = " ".join(str(item.get("label") or "").split())[:_MAX_LABEL_CHARS]
    return ToolAttachment(tool_call_id, tool_name, mime, data, label)


def attachment_message(
    attachments: Sequence[ToolAttachment], *, omitted: int = 0
) -> Dict[str, Any]:
    """The volatile user message showing a batch's images, in call order."""
    lines = ["Images returned by the tool calls above (tool results carry text only):"]
    lines.extend(f"- image {i}: {item.reference}" for i, item in enumerate(attachments, 1))
    if omitted:
        lines.append(f"({omitted} older image(s) from this batch were not kept.)")
    parts: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    parts.extend(item.image_part() for item in attachments)
    return _volatile({"role": "user", "content": parts})


def described_message(
    attachments: Sequence[ToolAttachment], descriptions: Sequence[str]
) -> Dict[str, Any]:
    """The text-only stand-in used when the active model cannot see images."""
    lines = [
        "The tool calls above returned images, but the active model cannot view images.",
    ]
    for i, (item, text) in enumerate(zip(attachments, descriptions), 1):
        body = " ".join(str(text or "").split()) or "no description is available"
        lines.append(f"- image {i}: {item.reference}. Description: {body}")
    return _volatile({"role": "user", "content": "\n".join(lines)})


def _volatile(message: Dict[str, Any]) -> Dict[str, Any]:
    message["_volatile_context"] = True
    message[ATTACHMENT_FLAG] = True
    return message


def is_attachment_message(message: Mapping[str, Any]) -> bool:
    return bool(message.get(ATTACHMENT_FLAG))


def retire_tool_images(messages: List[Dict[str, Any]], *, keep: int) -> int:
    """Replace all but the newest ``keep`` attached images with a placeholder.

    Only attachment messages are touched; images the user sent are not. The
    messages are replaced in place (never mutated) so a snapshot taken earlier
    keeps its content. Returns how many images were retired.
    """
    budget = max(0, int(keep))
    retired = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        content = message.get("content")
        if not is_attachment_message(message) or not isinstance(content, list):
            continue
        images = sum(1 for part in content if _is_image(part))
        if images <= budget:
            budget -= images
            continue
        retire = images - budget
        budget = 0
        parts: List[Any] = []
        for part in content:  # images later in a message are newer: retire the first
            if retire and _is_image(part):
                retire -= 1
                retired += 1
                continue
            parts.append(part)
        parts.append({"type": "text", "text": _RETIRED_NOTE})
        messages[index] = {**message, "content": parts}
    return retired


_RETIRED_NOTE = (
    "[Older images were removed from the conversation to bound its size; "
    "call the tool again to see the current view.]"
)


def _is_image(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES


async def present_attachments(
    messages: List[Dict[str, Any]],
    attachments: Sequence[ToolAttachment],
    *,
    keep: int,
    sees_images: bool,
    vlm: Any = None,
) -> None:
    """Append one message showing a batch's images, or describing them in text.

    A model that can see images gets the newest ``keep`` of them, and older
    images already in the conversation are retired so at most ``keep`` stay
    live. A model that cannot (or ``keep`` of 0) gets a VLM description of
    each, the same fallback screenshots use; without a VLM it is told that an
    image exists but cannot be shown.
    """
    if not attachments:
        return
    budget = max(0, int(keep))
    if budget and sees_images:
        shown = list(attachments)[-budget:]
        retire_tool_images(messages, keep=budget - len(shown))
        messages.append(attachment_message(shown, omitted=len(attachments) - len(shown)))
        return
    descriptions = [await describe_image(vlm, item) for item in attachments]
    messages.append(described_message(attachments, descriptions))


def model_sees_images(registry: Any, model: str) -> bool:
    """Whether the capability registry says ``model`` accepts image input."""
    if registry is None or not model:
        return False
    try:
        return bool(registry.resolve(model).supports_vision)
    except Exception:  # noqa: BLE001 - an unknown model is treated as text-only
        logger.debug("model capability lookup failed for %s", model, exc_info=True)
        return False


async def describe_image(vlm: Any, item: ToolAttachment) -> str:
    """A short VLM description of one attached image, or ``""`` when unavailable."""
    if vlm is None:
        return ""
    from leapflow.llm.message_builder import build_user_message_multimodal

    source = f"{item.tool_name} ({item.label})" if item.label else item.tool_name
    prompt = (
        f"This image was returned by the tool {source}. Describe what it shows in "
        "3-4 concise sentences: name the visible objects and where each one is in the "
        "image, with approximate pixel coordinates when you can estimate them."
    )
    message = build_user_message_multimodal(prompt, images_base64=[item.data], image_mime=item.mime)
    try:
        reply = await vlm.achat([message], stream=False, enable_thinking=False, max_tokens=200)
    except Exception:  # noqa: BLE001 - a failed description degrades to "unavailable"
        logger.debug("tool image description failed", exc_info=True)
        return ""
    return str(getattr(reply, "content", "") or "")


__all__ = [
    "ATTACHMENTS_KEY",
    "ATTACHMENT_FLAG",
    "IMAGE_MIMES",
    "ToolAttachment",
    "attachment_message",
    "describe_image",
    "described_message",
    "is_attachment_message",
    "model_sees_images",
    "present_attachments",
    "retire_tool_images",
    "take_attachments",
]
