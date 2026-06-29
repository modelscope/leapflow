"""Inference tiler — batch composed frame pairs into grid for VLM."""

from __future__ import annotations

import io
import logging
from typing import List, Optional, TYPE_CHECKING

from leapflow.perception.types import ComposedImage, TiledBatch

if TYPE_CHECKING:
    from leapflow.perception.types import InteractionSignal, PairContext

logger = logging.getLogger(__name__)

_MAX_SIGNALS_PER_PAIR_TILED = 4

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class InferenceTiler:
    """Tile multiple composed frame pairs into a grid for batched VLM inference.

    Groups composed pairs into grid images, allowing a single VLM call
    to extract actions from multiple transitions simultaneously.
    Includes position labels for unambiguous VLM reference.
    """

    __slots__ = ("_max_pairs", "_tile_size", "_gap", "_label_height")

    def __init__(
        self,
        max_pairs_per_tile: int = 4,
        tile_size: int = 384,
        gap: int = 4,
    ) -> None:
        self._max_pairs = max_pairs_per_tile
        self._tile_size = tile_size
        self._gap = gap
        self._label_height = 20

    def create_tiles(
        self,
        composed_pairs: List[ComposedImage],
        contexts: Optional[List["PairContext"]] = None,
    ) -> List[TiledBatch]:
        """Group composed frame pairs into tiled batches.

        Args:
            composed_pairs: Pre-composed frame pair images.
            contexts: Optional per-pair context (signals, app, time_delta).

        Returns a list of TiledBatch objects, each containing a grid image
        and the corresponding prompt for VLM multi-pair extraction.
        """
        if not _HAS_PIL or not composed_pairs:
            return []

        batches = []
        for i in range(0, len(composed_pairs), self._max_pairs):
            batch_pairs = composed_pairs[i:i + self._max_pairs]
            batch_contexts = contexts[i:i + self._max_pairs] if contexts else None
            grid_image = self._render_grid(batch_pairs)
            prompt = self._build_prompt(batch_pairs, batch_contexts)
            batches.append(TiledBatch(
                image=grid_image,
                pairs=batch_pairs,
                pair_count=len(batch_pairs),
                prompt=prompt,
            ))

        return batches

    def _render_grid(self, pairs: List[ComposedImage]) -> bytes:
        """Render pairs into a grid image with position labels."""
        n = len(pairs)
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols

        cell_w = self._tile_size
        cell_h = self._tile_size + self._label_height
        grid_w = cols * cell_w + (cols - 1) * self._gap
        grid_h = rows * cell_h + (rows - 1) * self._gap

        canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        labels = []
        for idx, pair in enumerate(pairs):
            row = idx // cols
            col = idx % cols
            x = col * (cell_w + self._gap)
            y = row * (cell_h + self._gap)

            # Draw label
            label = self._label_for(row, col)
            labels.append(label)
            draw.text((x + 4, y + 2), label, fill="white")

            # Draw pair image
            pair_img = Image.open(io.BytesIO(pair.image))
            pair_img.thumbnail((cell_w, self._tile_size), Image.LANCZOS)
            canvas.paste(pair_img, (x, y + self._label_height))

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def _build_prompt(
        self,
        pairs: List[ComposedImage],
        contexts: Optional[List["PairContext"]] = None,
    ) -> str:
        """Build a multi-pair extraction prompt for VLM with per-pair context."""
        n = len(pairs)
        positions = []
        cols = 2 if n > 1 else 1
        for idx in range(n):
            row = idx // cols
            col = idx % cols
            positions.append(self._label_for(row, col))

        sections = []
        for i in range(n):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            section = self._format_pair_section(positions[i], i + 1, ctx)
            sections.append(section)

        pair_block = "\n".join(sections)

        return (
            f"This image contains {n} frame pair(s) in a grid. "
            f"Each cell shows a before/after screenshot pair.\n\n"
            f"{pair_block}\n\n"
            f"For each pair, identify the user action performed between "
            f"the before (left) and after (right) frames. "
            f"Consider the context and interaction signals provided above.\n"
            f"Output a JSON array of length {n}, each element being a list "
            f"of actions: [[{{\"action\": ..., \"target\": ..., "
            f"\"confidence\": 0.0-1.0}}], ...]"
        )

    @staticmethod
    def _format_pair_section(
        position: str, pair_num: int, ctx: Optional["PairContext"]
    ) -> str:
        """Format context section for one pair in the tiled prompt."""
        header = f"Position {position} (pair {pair_num}):"
        if not ctx:
            return header

        details = []
        if ctx.time_delta > 0:
            details.append(f"Time: {ctx.time_delta:.1f}s")
        if ctx.app_b:
            details.append(f"App: {ctx.app_b}")
        if ctx.app_changed:
            details.append(f"Switch: {ctx.app_a} → {ctx.app_b}")

        lines = [header]
        if details:
            lines.append(f"  {' | '.join(details)}")

        if ctx.signals:
            from leapflow.perception.extraction.extractor import ContextEnrichedVLMExtractor
            t_base = ctx.signals[0].timestamp
            sig_lines = []
            for sig in ctx.signals[:_MAX_SIGNALS_PER_PAIR_TILED]:
                sig_lines.append(
                    ContextEnrichedVLMExtractor._format_signal_line(sig, t_base)
                )
            lines.append("  Signals: " + "; ".join(s.strip("- ") for s in sig_lines))

        return "\n".join(lines)

    @staticmethod
    def _label_for(row: int, col: int) -> str:
        """Generate position label (A1, A2, B1, B2, ...)."""
        return f"{chr(65 + row)}{col + 1}"
