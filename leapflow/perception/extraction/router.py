"""Tiered inference router — SKIP/LIGHT/STANDARD/DEEP level assignment."""

from __future__ import annotations

from typing import Optional

from leapflow.perception.types import FramePair, InferenceLevel, PairContext, VisualAction


class TieredInferenceRouter:
    """Route frame pairs to the appropriate VLM inference level.

    Levels:
    - SKIP: CV already determined the action (no VLM needed)
    - LIGHT: Simple transitions (app switch, scroll) → fast/cheap model
    - STANDARD: Complex interactions (text input, dialog) → full model
    - DEEP: Ambiguous cases → full model + chain-of-thought
    """

    __slots__ = ()

    def route(self, pair: FramePair, context: Optional[PairContext] = None) -> InferenceLevel:
        """Determine the inference level for this pair."""
        ctx = context or pair.context or PairContext()

        # SKIP: CV already fully resolved the action
        if ctx.app_changed and ctx.time_delta < 1.0 and not ctx.new_text:
            pair.pre_extracted_action = VisualAction(
                action="switch_app",
                target=ctx.app_b,
                confidence=0.95,
                evidence="title_bar_changed",
                frame_ref_a=pair.frame_a.ref,
                frame_ref_b=pair.frame_b.ref,
            )
            return InferenceLevel.SKIP

        transition = pair.transition_type

        # LIGHT: Simple, high-confidence transitions
        if transition in ("scroll", "wait"):
            return InferenceLevel.LIGHT

        if transition == "app_switch" and ctx.time_delta > 1.0:
            return InferenceLevel.LIGHT

        # STANDARD: Most interactive operations
        if transition in ("text_input", "click_target", "dialog_popup"):
            return InferenceLevel.STANDARD

        if transition == "navigation":
            return InferenceLevel.STANDARD

        # Default: STANDARD for unknown transitions.
        # DEEP only when strong complexity signals are present.
        if ctx.time_delta > 5.0 and len(ctx.diff_regions) >= 2:
            return InferenceLevel.DEEP

        return InferenceLevel.STANDARD

    def get_model_params(self, level: InferenceLevel) -> dict:
        """Return model configuration for the given inference level."""
        configs = {
            InferenceLevel.LIGHT: {
                "max_tokens": 150,
                "resolution": 512,
                "prompt_style": "compact",
            },
            InferenceLevel.STANDARD: {
                "max_tokens": 300,
                "resolution": 768,
                "prompt_style": "standard",
            },
            InferenceLevel.DEEP: {
                "max_tokens": 500,
                "resolution": 1024,
                "prompt_style": "detailed_cot",
            },
        }
        return configs.get(level, configs[InferenceLevel.STANDARD])
