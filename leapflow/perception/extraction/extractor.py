"""Stage C: Context-Enriched VLM Extractor — action inference from frame pairs."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from leapflow.perception.types import FramePair, InferenceLevel, PairContext, TiledBatch, VisualAction

if TYPE_CHECKING:
    from leapflow.llm.base import LLMProvider

logger = logging.getLogger(__name__)


_MAX_SIGNAL_PROMPT_LINES = 8


class ContextEnrichedVLMExtractor:
    """VLM action extraction with rich context from CV preprocessing.

    Instead of asking VLM "what happened?", we ask "given these detected
    changes, what was the user's intent?" — shifting VLM from perception
    (expensive, error-prone) to reasoning (its strength).
    """

    __slots__ = ("_vlm",)

    def __init__(self, vlm: "LLMProvider") -> None:
        self._vlm = vlm

    async def extract_pair(
        self,
        pair: FramePair,
        context: Optional[PairContext] = None,
        level: InferenceLevel = InferenceLevel.STANDARD,
    ) -> List[VisualAction]:
        """Extract actions from a single frame pair using VLM.

        Args:
            pair: The before/after frame pair.
            context: Pre-computed CV context (injected into prompt).
            level: Inference level controlling model params and prompt style.

        Returns:
            List of VisualAction extracted from the pair.
        """
        from leapflow.llm.message_builder import (
            build_system_message,
            build_user_message_multimodal,
        )
        import base64

        from leapflow.perception.extraction.router import TieredInferenceRouter
        params = TieredInferenceRouter().get_model_params(level)
        target_res = params.get("resolution", 768)

        ctx = context or pair.context or PairContext()
        prompt = self._build_prompt(ctx, level)
        system = self._system_prompt(level)

        img_a = self._resize_for_vlm(pair.frame_a.image, target_res)
        img_b = self._resize_for_vlm(pair.frame_b.image, target_res)
        images_b64 = [
            base64.b64encode(img_a).decode(),
            base64.b64encode(img_b).decode(),
        ]

        messages = [
            build_system_message(system),
            build_user_message_multimodal(
                prompt,
                images_base64=images_b64,
                image_mime="image/jpeg",
            ),
        ]

        try:
            t0 = time.monotonic()
            resp = await self._vlm.achat(
                messages,
                stream=False,
                enable_thinking=False,
                max_tokens=params.get("max_tokens", 300),
            )
            dt = time.monotonic() - t0
            tokens = resp.usage.get("total_tokens", "?") if resp.usage else "?"
            logger.info(
                "VLM call done in %.1fs (level=%s, res=%d, tokens=%s)",
                dt, level.value, target_res, tokens,
            )
            return self._parse_response(resp.content, pair)
        except Exception as e:
            logger.warning("VLM extraction failed: %s", e)
            return []

    @staticmethod
    def _resize_for_vlm(image_data: bytes, max_dim: int) -> bytes:
        """Resize image to target max dimension for VLM inference."""
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_data))
            if max(img.size) <= max_dim:
                return image_data
            scale = max_dim / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            resized = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=75)
            return buf.getvalue()
        except Exception:
            return image_data

    async def extract_batch(
        self,
        pairs: List[FramePair],
        contexts: Optional[List[PairContext]] = None,
        levels: Optional[List[InferenceLevel]] = None,
    ) -> List[List[VisualAction]]:
        """Extract actions from multiple pairs (sequential for now)."""
        import asyncio

        results = []
        for i, pair in enumerate(pairs):
            ctx = contexts[i] if contexts else None
            level = levels[i] if levels else InferenceLevel.STANDARD
            actions = await self.extract_pair(pair, ctx, level)
            results.append(actions)
        return results

    async def extract_tiled_batch(
        self,
        batch: "TiledBatch",
        pairs: List[FramePair],
        level: InferenceLevel = InferenceLevel.STANDARD,
    ) -> List[List[VisualAction]]:
        """Extract actions from a tiled grid image containing multiple pairs.

        Args:
            batch: TiledBatch with grid image and multi-pair prompt.
            pairs: The FramePairs corresponding to each cell in the grid.
            level: Inference level for model params.

        Returns:
            List of action lists, one per pair in the batch.
        """
        from leapflow.llm.message_builder import (
            build_system_message,
            build_user_message_multimodal,
        )
        import base64

        n = batch.pair_count
        system = self._tiled_system_prompt(n)

        img_b64 = base64.b64encode(batch.image).decode()
        messages = [
            build_system_message(system),
            build_user_message_multimodal(
                batch.prompt,
                images_base64=[img_b64],
                image_mime="image/jpeg",
            ),
        ]

        from leapflow.perception.extraction.router import TieredInferenceRouter
        params = TieredInferenceRouter().get_model_params(level)

        try:
            t0 = time.monotonic()
            resp = await self._vlm.achat(
                messages,
                stream=False,
                enable_thinking=False,
                max_tokens=params.get("max_tokens", 300) * n,
            )
            dt = time.monotonic() - t0
            tokens = resp.usage.get("total_tokens", "?") if resp.usage else "?"
            logger.info(
                "VLM tiled call done in %.1fs (pairs=%d, level=%s, tokens=%s)",
                dt, n, level.value, tokens,
            )
            return self._parse_tiled_response(resp.content, pairs)
        except Exception as e:
            logger.warning("VLM tiled extraction failed: %s", e)
            return [[] for _ in range(n)]

    @staticmethod
    def _tiled_system_prompt(pair_count: int) -> str:
        return (
            "You are a visual action recognition system. "
            f"You will see {pair_count} frame pair(s) arranged in a grid. "
            "Each cell shows a before (left) and after (right) screenshot. "
            "Identify the user action for each pair. "
            f"Output valid JSON only: an array of length {pair_count}, "
            "each element being a list of action objects."
        )

    @staticmethod
    def _parse_tiled_response(
        content: str, pairs: List[FramePair]
    ) -> List[List[VisualAction]]:
        """Parse multi-pair VLM response into per-pair VisualAction lists."""
        content = content.strip()
        if "```" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                content = content[start:end]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(content[start:end])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tiled VLM response")
                    return [[] for _ in pairs]
            else:
                logger.warning("No JSON array found in tiled VLM response")
                return [[] for _ in pairs]

        if not isinstance(data, list):
            logger.warning("Tiled VLM response is not an array")
            return [[] for _ in pairs]

        if len(data) != len(pairs):
            logger.warning(
                "Tiled response length mismatch: got %d, expected %d",
                len(data), len(pairs),
            )

        results: List[List[VisualAction]] = []
        for idx, pair in enumerate(pairs):
            if idx >= len(data):
                results.append([])
                continue
            pair_data = data[idx]
            if not isinstance(pair_data, list):
                pair_data = [pair_data] if isinstance(pair_data, dict) else []
            actions = []
            for item in pair_data:
                if isinstance(item, dict):
                    actions.append(VisualAction(
                        action=item.get("action", "unknown"),
                        target=item.get("target", ""),
                        detail=item.get("detail", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        evidence="vlm_tiled_extraction",
                        frame_ref_a=pair.frame_a.ref,
                        frame_ref_b=pair.frame_b.ref,
                    ))
            results.append(actions)
        return results

    def _build_prompt(self, context: PairContext, level: InferenceLevel) -> str:
        """Build context-enriched prompt that guides VLM reasoning."""
        parts = []

        if context.time_delta > 0:
            parts.append(f"Time between frames: {context.time_delta:.1f}s")

        if context.app_changed:
            parts.append(f"App switch: {context.app_a} → {context.app_b}")
        elif context.app_b:
            parts.append(f"Current app: {context.app_b}")

        if context.new_text:
            texts = "; ".join(t.text for t in context.new_text[:3])
            parts.append(f"New text appeared: {texts}")

        if context.removed_text:
            texts = "; ".join(t.text for t in context.removed_text[:3])
            parts.append(f"Text disappeared: {texts}")

        if context.diff_regions:
            regions = ", ".join(r.get("location_desc", "unknown") for r in context.diff_regions[:3])
            parts.append(f"Changed regions: {regions}")

        if context.new_ui_elements:
            elements = ", ".join(e.get("cls", "unknown") for e in context.new_ui_elements[:3])
            parts.append(f"New UI elements: {elements}")

        # Signal context
        has_signals = bool(context.signals)
        if has_signals:
            parts.append("")
            parts.append("Interaction signals detected between frames:")
            t_base = context.signals[0].timestamp
            for sig in context.signals[:_MAX_SIGNAL_PROMPT_LINES]:
                parts.append(self._format_signal_line(sig, t_base))

        # Instruction
        signal_hint = " Consider the interaction signals above." if has_signals else ""
        if level == InferenceLevel.DEEP:
            parts.append(
                "\nAnalyze these two screenshots step by step. "
                "What user action(s) occurred between them? "
                f"Consider the visual changes and context above.{signal_hint} "
                "Output JSON: [{\"action\": \"...\", \"target\": \"...\", "
                "\"detail\": \"...\", \"confidence\": 0.0-1.0}]"
            )
        else:
            parts.append(
                "\nWhat user action occurred between these two frames?"
                f"{signal_hint} "
                "Output JSON: [{\"action\": \"...\", \"target\": \"...\", "
                "\"confidence\": 0.0-1.0}]"
            )

        return "\n".join(parts)

    @staticmethod
    def _format_signal_line(sig: "InteractionSignal", t_base: float) -> str:
        """Format a single interaction signal for VLM prompt injection."""
        offset = sig.timestamp - t_base
        app_short = sig.app.rsplit(".", 1)[-1] if sig.app else ""

        if sig.signal_type == "click" and sig.position:
            loc = f"({sig.position[0]}, {sig.position[1]})"
            return f"  - Click at {loc} [+{offset:.1f}s]{f' app={app_short}' if app_short else ''}"
        elif sig.signal_type == "app_switch":
            return f"  - App switch: {sig.detail} [+{offset:.1f}s]"
        elif sig.signal_type == "clipboard":
            return f"  - Clipboard: {sig.detail} [+{offset:.1f}s]"
        elif sig.signal_type == "keyboard":
            return f"  - Keyboard: {sig.detail} [+{offset:.1f}s]"
        elif sig.signal_type == "drag" and sig.position and sig.end_position:
            start = f"({sig.position[0]},{sig.position[1]})"
            end = f"({sig.end_position[0]},{sig.end_position[1]})"
            return f"  - Drag: {start} -> {end} [+{offset:.1f}s]"
        elif sig.signal_type == "scroll":
            return f"  - Scroll: {sig.detail} [+{offset:.1f}s]"
        else:
            return f"  - {sig.signal_type}: {sig.detail} [+{offset:.1f}s]"

    @staticmethod
    def _system_prompt(level: InferenceLevel) -> str:
        base = (
            "You are a visual action recognition system. "
            "Given before/after screenshots, identify user actions. "
            "Output valid JSON only."
        )
        if level == InferenceLevel.DEEP:
            return base + " Think step by step before outputting the JSON."
        return base

    @staticmethod
    def _parse_response(content: str, pair: FramePair) -> List[VisualAction]:
        """Parse VLM response JSON into VisualAction objects."""
        content = content.strip()
        # Extract JSON from markdown code blocks if present
        if "```" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                content = content[start:end]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Attempt to find JSON array in the response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(content[start:end])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            data = [data]

        actions = []
        for item in data:
            if isinstance(item, dict):
                actions.append(VisualAction(
                    action=item.get("action", "unknown"),
                    target=item.get("target", ""),
                    detail=item.get("detail", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    evidence="vlm_extraction",
                    frame_ref_a=pair.frame_a.ref,
                    frame_ref_b=pair.frame_b.ref,
                ))
        return actions
