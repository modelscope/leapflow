# Copyright (c) Alibaba, Inc. and its affiliates.
"""Stage 5: Execution Model Analyzer.

Checks execution_model and source_language compatibility with LeapFlow's
runtime capabilities.

LeapFlow supports:
  - async (native — asyncio-based engine loop)
  - sync (wrapped in executor via asyncio.to_thread)
  - subprocess (via SandboxHost isolation)

Source language:
  - python → native in-process
  - typescript/javascript → requires JSON-RPC bridge (subprocess mode)
"""

from __future__ import annotations

from typing import List

from leapflow.learning.compatibility.protocol import (
    PluginManifestInput,
    StageResult,
    Verdict,
)

# ═══════════════════════════════════════════════════════════════════════
# Execution model compatibility mapping.
# Maps DSH/foreign execution models to LeapFlow support status.
# ═══════════════════════════════════════════════════════════════════════

_EXECUTION_MODEL_MAP: dict[str, tuple[str, Verdict | None]] = {
    # model → (leapflow_equivalent, verdict_if_adaptation_needed)
    "async": ("async", None),  # Native
    "sync": ("sync", None),  # Wrapped via to_thread
    "subprocess": ("subprocess", None),  # Via SandboxHost
    "worker": ("subprocess", Verdict.ADAPTABLE),  # Map to subprocess
    "streaming": ("async", Verdict.ADAPTABLE),  # Map to async generator
    "event-driven": ("async", Verdict.ADAPTABLE),  # Map to async event loop
    "callback": ("async", Verdict.ADAPTABLE),  # Map to async with Future
}

# Source language support classification
_LANGUAGE_SUPPORT: dict[str, tuple[str, Verdict | None]] = {
    # language → (execution_mode, verdict_if_bridge_needed)
    "python": ("in_process", None),  # Native
    "typescript": ("subprocess", Verdict.ADAPTABLE),  # JSON-RPC bridge
    "javascript": ("subprocess", Verdict.ADAPTABLE),  # JSON-RPC bridge
    "rust": ("subprocess", Verdict.ADAPTABLE),  # FFI or subprocess
    "go": ("subprocess", Verdict.ADAPTABLE),  # Subprocess
}


class ExecutionModelAnalyzer:
    """Analyze execution model and source language compatibility."""

    stage_name: str = "execution_model_analyzer"

    def assess(
        self, manifest: PluginManifestInput, prior_results: List[StageResult]
    ) -> StageResult:
        """Check execution model and source language against LeapFlow capabilities.

        Produces a combined verdict from both dimensions:
        - execution model compatibility
        - source language bridge requirements
        """
        exec_model = manifest.execution_model.lower().strip()
        source_lang = manifest.source_language.lower().strip()

        # Check execution model
        model_info = _EXECUTION_MODEL_MAP.get(exec_model)
        if model_info is None:
            # Unknown execution model — partial support
            model_equiv = "subprocess"
            model_verdict = Verdict.PARTIAL
            model_note = f"Unknown execution model '{exec_model}'; will use subprocess isolation"
        else:
            model_equiv, model_verdict = model_info
            if model_verdict:
                model_note = f"Execution model '{exec_model}' maps to LeapFlow '{model_equiv}' (needs adapter)"
            else:
                model_note = f"Execution model '{exec_model}' is natively supported as '{model_equiv}'"

        # Check source language
        lang_info = _LANGUAGE_SUPPORT.get(source_lang)
        if lang_info is None:
            # Unknown language — will need subprocess bridge
            lang_mode = "subprocess"
            lang_verdict = Verdict.PARTIAL
            lang_note = f"Unknown source language '{source_lang}'; requires subprocess bridge"
        else:
            lang_mode, lang_verdict = lang_info
            if lang_verdict:
                lang_note = f"Source language '{source_lang}' requires {lang_mode} bridge"
            else:
                lang_note = f"Source language '{source_lang}' supports native {lang_mode} execution"

        # Synthesize combined verdict
        verdicts = [v for v in (model_verdict, lang_verdict) if v is not None]
        if Verdict.PARTIAL in verdicts:
            combined_verdict = Verdict.PARTIAL
        elif Verdict.ADAPTABLE in verdicts:
            combined_verdict = Verdict.ADAPTABLE
        else:
            combined_verdict = None  # Fully compatible

        evidence = {
            "execution_model": exec_model,
            "source_language": source_lang,
            "leapflow_equivalent": model_equiv,
            "language_mode": lang_mode,
            "requires_bridge": lang_verdict is not None,
            "requires_model_adapter": model_verdict is not None,
        }

        details_parts = [model_note, lang_note]
        details = "; ".join(details_parts)

        if combined_verdict == Verdict.PARTIAL:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=Verdict.PARTIAL,
                details=details,
                evidence=evidence,
            )
        elif combined_verdict == Verdict.ADAPTABLE:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=Verdict.ADAPTABLE,
                details=details,
                evidence=evidence,
            )
        else:
            return StageResult(
                stage_name=self.stage_name,
                passed=True,
                verdict=None,
                details=details,
                evidence=evidence,
            )
