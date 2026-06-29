"""Video analysis prompt strategies and VLM message builders.

Follows Open/Closed Principle: extend via new implementations,
not by modifying existing code.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol

logger = logging.getLogger(__name__)


# ─── Prompt Strategy Protocol ───


class AnalysisPromptStrategy(Protocol):
    """Generate LLM message sequences for each analysis level."""

    def build_l1_messages(
        self,
        video_path: str,
        duration: float,
        top_apps: str,
        timeline_text: str,
        goal: str,
    ) -> List[Dict[str, Any]]:
        """Build L1 macro analysis messages."""
        ...

    def build_l2_messages(
        self,
        video_path: str,
        timestamp: float,
        reason: str,
        context: str,
    ) -> List[Dict[str, Any]]:
        """Build L2 moment refinement messages."""
        ...

    def build_l3_messages(
        self,
        image_data: bytes,
        timestamp: float,
        reason: str,
        nearby_events: str,
    ) -> List[Dict[str, Any]]:
        """Build L3 frame-level analysis messages."""
        ...


# ─── VLM Message Builder Protocol ───


class VLMMessageBuilder(Protocol):
    """Wrap video/image content into the message format required by a specific VLM backend."""

    def build_video_message(
        self, video_path: str, prompt: str, *, system: str = "",
    ) -> List[Dict[str, Any]]:
        """Build a message containing video content."""
        ...

    def build_image_message(
        self, image_data: bytes, prompt: str, *, system: str = "",
    ) -> List[Dict[str, Any]]:
        """Build a message containing image content."""
        ...


# ─── Default Implementations ───


class DefaultAnalysisPrompts:
    """Default analysis prompt strategy with configurable templates.

    All templates can be overridden via constructor arguments;
    built-in defaults are used when not provided.
    """

    _DEFAULT_L1_SYSTEM = (
        "You are a desktop-operation analyst.  Given a screen recording and an "
        "event timeline, extract every meaningful user action step."
    )

    _DEFAULT_L1_TEMPLATE = """## Context
- User goal: {goal}
- Duration: {duration:.1f}s
- Primary apps: {top_apps}

## Event timeline (relative to video start)
{timeline}

## Output (JSON)
Return a JSON object:
{{
  "steps": [
    {{
      "start_time": <float>,
      "end_time": <float>,
      "action": "<verb_noun>",
      "description": "<one sentence>",
      "app": "<app name>",
      "goal": "<step goal>",
      "confidence": <0-1>,
      "needs_detail": <bool>
    }}
  ],
  "overall_goal": "<string>",
  "detail_requests": [{{"timestamp": <float>, "reason": "<string>"}}],
  "frame_requests": [{{"timestamp": <float>, "reason": "<string>"}}]
}}
"""

    _DEFAULT_L2_SYSTEM = "You are a desktop-operation analyst. Always respond with valid JSON."

    _DEFAULT_L2_TEMPLATE = (
        "At relative timestamp {timestamp:.1f}s in this video, "
        "describe in detail what the user is doing.\n"
        "Context: {reason}\n"
        "Additional context: {context}\n\n"
        "Output JSON:\n"
        '{{\n'
        '  "action": "<verb_noun>",\n'
        '  "description": "<detailed description of what the user is doing>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "app": "<application name if visible>"\n'
        '}}'
    )

    _DEFAULT_L3_SYSTEM = (
        "You are a UI content analyst. Given a screenshot frame from a desktop "
        "recording, extract precise content visible on screen."
    )

    _DEFAULT_L3_TEMPLATE = (
        "Analyze this frame captured at {timestamp:.1f}s.\n"
        "Context: {reason}\n"
        "Event context: {nearby_events}\n"
        "\n"
        "Extract:\n"
        "- Exact text content visible (OCR)\n"
        "- UI element labels and states\n"
        "- Dialog/popup content if present\n"
        "\n"
        "Output JSON:\n"
        '{{"content_type": "...", "text": "...", "ui_elements": [...], "confidence": 0-1}}'
    )

    def __init__(
        self,
        *,
        message_builder: VLMMessageBuilder,
        l1_system: str = "",
        l1_template: str = "",
        l2_system: str = "",
        l2_template: str = "",
        l3_system: str = "",
        l3_template: str = "",
    ) -> None:
        self._builder = message_builder
        self._l1_system = l1_system or self._DEFAULT_L1_SYSTEM
        self._l1_template = l1_template or self._DEFAULT_L1_TEMPLATE
        self._l2_system = l2_system or self._DEFAULT_L2_SYSTEM
        self._l2_template = l2_template or self._DEFAULT_L2_TEMPLATE
        self._l3_system = l3_system or self._DEFAULT_L3_SYSTEM
        self._l3_template = l3_template or self._DEFAULT_L3_TEMPLATE

    def build_l1_messages(
        self,
        video_path: str,
        duration: float,
        top_apps: str,
        timeline_text: str,
        goal: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._l1_template.format(
            goal=goal or "unknown",
            duration=duration,
            top_apps=top_apps,
            timeline=timeline_text,
        )
        return self._builder.build_video_message(
            video_path, prompt, system=self._l1_system,
        )

    def build_l2_messages(
        self,
        video_path: str,
        timestamp: float,
        reason: str,
        context: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._l2_template.format(
            timestamp=timestamp, reason=reason, context=context or "",
        )
        return self._builder.build_video_message(
            video_path, prompt, system=self._l2_system,
        )

    def build_l3_messages(
        self,
        image_data: bytes,
        timestamp: float,
        reason: str,
        nearby_events: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._l3_template.format(
            timestamp=timestamp,
            reason=reason,
            nearby_events=nearby_events or "none",
        )
        return self._builder.build_image_message(
            image_data, prompt, system=self._l3_system,
        )


class DashScopeMessageBuilder:
    """DashScope VLM message format builder."""

    def __init__(self, *, url_scheme: str = "base64") -> None:
        self._url_scheme = url_scheme

    def build_video_message(
        self, video_path: str, prompt: str, *, system: str = "",
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        # Select transport mode based on url_scheme
        if self._url_scheme == "base64":
            # Base64 inline mode: read local file and encode as data URI
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(
                    f"Video file not found: {video_path}"
                )
            video_data = video_file.read_bytes()
            video_b64 = base64.b64encode(video_data).decode("ascii")
            video_url = f"data:video/mp4;base64,{video_b64}"
        else:
            # HTTPS URL mode: concatenate scheme + path directly
            video_url = f"{self._url_scheme}{video_path}"

        messages.append({
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        })
        return messages

    def build_image_message(
        self, image_data: bytes, prompt: str, *, system: str = "",
    ) -> List[Dict[str, Any]]:
        b64 = base64.b64encode(image_data).decode("ascii")
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        })
        return messages
