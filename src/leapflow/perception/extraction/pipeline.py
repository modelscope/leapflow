"""Offline Extraction Pipeline — Stage A → B → C orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from leapflow.perception.extraction.feature_extractor import FeatureExtractor
from leapflow.perception.extraction.preprocessor import SemanticPreprocessor
from leapflow.perception.extraction.refiner import KeyframeRefiner
from leapflow.perception.extraction.router import TieredInferenceRouter
from leapflow.perception.storage.semantic_cache import SemanticCache
from leapflow.perception.types import (
    FramePair,
    InferenceLevel,
    Keyframe,
    PairContext,
    VisualAction,
)

if TYPE_CHECKING:
    from leapflow.llm.base import LLMProvider
    from leapflow.perception.config import PerceptionConfig

logger = logging.getLogger(__name__)


class OfflineExtractionPipeline:
    """Three-stage offline processing pipeline for visual action extraction.

    Stage A: Keyframe Refinement (local, no VLM)
        - Deduplication, transition classification, pair building, budget allocation

    Stage B: Semantic Pre-processing (local CV, no VLM)
        - OCR, UI detection, diff heatmap → PairContext for each pair

    Stage C: VLM Action Extraction (API calls)
        - Context-enriched prompts, tiered model routing, batched inference
    """

    __slots__ = (
        "_config", "_feature_extractor", "_refiner", "_preprocessor",
        "_router", "_semantic_cache", "_vlm",
    )

    def __init__(
        self,
        vlm: Optional["LLMProvider"] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        refiner: Optional[KeyframeRefiner] = None,
        preprocessor: Optional[SemanticPreprocessor] = None,
        router: Optional[TieredInferenceRouter] = None,
        semantic_cache: Optional[SemanticCache] = None,
        config: Optional["PerceptionConfig"] = None,
    ) -> None:
        self._vlm = vlm
        self._config = config
        self._feature_extractor = feature_extractor or FeatureExtractor()
        self._refiner = refiner or KeyframeRefiner()
        self._preprocessor = preprocessor or SemanticPreprocessor()
        self._router = router or TieredInferenceRouter()
        self._semantic_cache = semantic_cache or SemanticCache()

    async def extract(
        self,
        keyframes: List[Keyframe],
        *,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[VisualAction]:
        """Run the full extraction pipeline on a set of keyframes.

        Returns all extracted VisualActions in temporal order.
        """
        if not keyframes:
            return []

        def _report(stage: str, current: int = 0, total: int = 0) -> None:
            if progress:
                progress(stage, current, total)

        # Stage A: Feature extraction + Keyframe refinement
        logger.info("Stage A: Extracting features from %d keyframes", len(keyframes))
        _report("extract.refine", 0, len(keyframes))
        features = await self._feature_extractor.extract_batch(keyframes)
        for kf, feat in zip(keyframes, features):
            kf.features = feat

        refined = await self._refiner.refine(keyframes)
        logger.info(
            "Stage A complete: %d unique frames, %d pairs",
            len(refined.frames), len(refined.pairs),
        )

        if not refined.pairs:
            return []

        # Stage B: Semantic preprocessing
        logger.info("Stage B: Preprocessing %d pairs", len(refined.pairs))
        _report("extract.preprocess", 0, len(refined.pairs))
        contexts = await self._preprocessor.process_batch(refined.pairs)
        for pair, ctx in zip(refined.pairs, contexts):
            pair.context = ctx

        # Stage C: VLM extraction with routing
        logger.info("Stage C: VLM extraction")
        all_actions = await self._extract_with_routing(
            refined.pairs, contexts, progress=progress,
        )

        logger.info("Extraction complete: %d actions extracted", len(all_actions))
        return all_actions

    async def _extract_with_routing(
        self,
        pairs: List[FramePair],
        contexts: List[PairContext],
        *,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[VisualAction]:
        """Route pairs to appropriate inference levels and extract."""
        from leapflow.perception.cv.phash import hamming_distance, phash_64
        from leapflow.perception.extraction.extractor import ContextEnrichedVLMExtractor

        _PREFILTER_THRESHOLD = 0.10
        _VLM_CONCURRENCY = 4

        all_actions: List[VisualAction] = []

        def _report(stage: str, current: int = 0, total: int = 0) -> None:
            if progress:
                progress(stage, current, total)

        # Assign levels
        levels: List[InferenceLevel] = []
        for pair, ctx in zip(pairs, contexts):
            level = self._router.route(pair, ctx)
            levels.append(level)

        # Process SKIPped pairs (pre-extracted)
        for pair, level in zip(pairs, levels):
            if level == InferenceLevel.SKIP and pair.pre_extracted_action:
                all_actions.append(pair.pre_extracted_action)

        # Check semantic cache for remaining pairs
        pairs_to_extract: List[tuple[int, FramePair, PairContext, InferenceLevel]] = []
        for i, (pair, ctx, level) in enumerate(zip(pairs, contexts, levels)):
            if level == InferenceLevel.SKIP:
                continue

            cache_key = self._build_cache_key(pair)
            cached = self._semantic_cache.get(cache_key)
            if cached:
                all_actions.extend(cached)
                continue

            pairs_to_extract.append((i, pair, ctx, level))

        # Pre-filter: skip pairs with negligible visual difference
        candidate_pairs: List[tuple[int, FramePair, PairContext, InferenceLevel]] = []
        for item in pairs_to_extract:
            i, pair, ctx, level = item
            dist = hamming_distance(
                phash_64(pair.frame_a.image), phash_64(pair.frame_b.image)
            ) / 64.0
            if dist < _PREFILTER_THRESHOLD:
                logger.debug("Pair %d pre-filtered (phash dist=%.3f)", i, dist)
                continue
            candidate_pairs.append(item)

        skipped = len(pairs_to_extract) - len(candidate_pairs)
        if skipped > 0:
            logger.info(
                "Pre-filter: %d/%d pairs passed (skipped %d visually similar)",
                len(candidate_pairs), len(pairs_to_extract), skipped,
            )

        # VLM extraction: tiled or per-pair
        tiling_enabled = (
            self._config is not None
            and self._config.tiling_enabled
            and len(candidate_pairs) >= 2
        )

        if candidate_pairs and self._vlm:
            if tiling_enabled:
                tiled_actions = await self._extract_tiled(
                    candidate_pairs, _VLM_CONCURRENCY, progress,
                )
                all_actions.extend(tiled_actions)
            else:
                extractor = ContextEnrichedVLMExtractor(self._vlm)
                total = len(candidate_pairs)
                _report("extract.vlm", 0, total)

                semaphore = asyncio.Semaphore(_VLM_CONCURRENCY)
                completed_count = 0
                t_batch_start = time.monotonic()

                async def _do_one(
                    pair: FramePair, ctx: PairContext, level: InferenceLevel
                ) -> List[VisualAction]:
                    nonlocal completed_count
                    async with semaphore:
                        actions = await extractor.extract_pair(pair, ctx, level)
                        completed_count += 1
                        _report("extract.vlm", completed_count, total)
                        return actions

                tasks = [_do_one(p, c, l) for _, p, c, l in candidate_pairs]
                results = await asyncio.gather(*tasks)

                for (_, pair, _, _), actions in zip(candidate_pairs, results):
                    if actions:
                        all_actions.extend(actions)
                        cache_key = self._build_cache_key(pair)
                        self._semantic_cache.put(cache_key, actions)

                dt_batch = time.monotonic() - t_batch_start
                logger.info("VLM batch complete: %d pairs in %.1fs", total, dt_batch)

        # Sort by frame timestamp
        all_actions.sort(key=lambda a: a.frame_ref_a)
        return all_actions

    async def _extract_tiled(
        self,
        candidate_pairs: List[Tuple[int, FramePair, PairContext, InferenceLevel]],
        concurrency: int,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[VisualAction]:
        """Extract actions using tiled multi-pair VLM calls.

        Only LIGHT-level pairs are tiled (simple transitions, low resolution
        requirement). STANDARD/DEEP pairs always use high-resolution per-pair
        extraction to preserve quality for complex action recognition.
        """
        from leapflow.perception.encoding.delta import DeltaFrameComposer
        from leapflow.perception.encoding.tiler import InferenceTiler
        from leapflow.perception.extraction.extractor import ContextEnrichedVLMExtractor

        config = self._config
        tile_size = config.tiling_tile_size if config else 384
        max_frames = config.tiling_max_frames if config else 4
        gap = config.tiling_gap if config else 4

        composer = DeltaFrameComposer(side_by_side_res=tile_size)
        tiler = InferenceTiler(max_pairs_per_tile=max_frames, tile_size=tile_size, gap=gap)
        extractor = ContextEnrichedVLMExtractor(self._vlm)

        def _report(stage: str, current: int = 0, total: int = 0) -> None:
            if progress:
                progress(stage, current, total)

        # Group by inference level
        level_groups: Dict[InferenceLevel, List[Tuple[int, FramePair, PairContext, InferenceLevel]]] = defaultdict(list)
        for item in candidate_pairs:
            level_groups[item[3]].append(item)

        # Only LIGHT-level pairs are tiled. STANDARD/DEEP need high resolution
        # for accurate text/UI/target recognition — per-pair path preserves quality.
        _TILEABLE_LEVELS = frozenset({InferenceLevel.LIGHT})

        BatchItem = Tuple[
            "TiledBatch", List[FramePair], List[PairContext], InferenceLevel
        ]
        all_batches: List[BatchItem] = []
        singles: List[Tuple[FramePair, PairContext, InferenceLevel]] = []

        for level, group in level_groups.items():
            # Non-tileable levels always use per-pair extraction
            if level not in _TILEABLE_LEVELS or len(group) == 1:
                for _, pair, ctx, lvl in group:
                    singles.append((pair, ctx, lvl))
                continue

            composed_images = []
            contexts = []
            pairs_in_group = []
            for _, pair, ctx, _ in group:
                composed = composer.compose(
                    pair.frame_a.image, pair.frame_b.image, pair.change_signal,
                )
                composed_images.append(composed)
                contexts.append(ctx)
                pairs_in_group.append(pair)

            batches = tiler.create_tiles(composed_images, contexts)

            # Map batches back to their pairs/contexts
            offset = 0
            for batch in batches:
                n = batch.pair_count
                batch_pairs = pairs_in_group[offset:offset + n]
                batch_contexts = contexts[offset:offset + n]
                all_batches.append((batch, batch_pairs, batch_contexts, level))
                offset += n

            # If tiler returned empty (PIL missing), fall back to singles
            if not batches:
                for _, pair, ctx, lvl in group:
                    singles.append((pair, ctx, lvl))

        total_work = len(all_batches) + len(singles)
        _report("extract.vlm", 0, total_work)

        all_actions: List[VisualAction] = []
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0
        t_start = time.monotonic()

        async def _do_tiled(
            batch: "TiledBatch",
            batch_pairs: List[FramePair],
            level: InferenceLevel,
        ) -> List[VisualAction]:
            nonlocal completed_count
            async with semaphore:
                results = await extractor.extract_tiled_batch(batch, batch_pairs, level)
                actions = []
                for pair, pair_actions in zip(batch_pairs, results):
                    if pair_actions:
                        actions.extend(pair_actions)
                        cache_key = self._build_cache_key(pair)
                        self._semantic_cache.put(cache_key, pair_actions)
                completed_count += 1
                _report("extract.vlm", completed_count, total_work)
                return actions

        async def _do_single(
            pair: FramePair, ctx: PairContext, level: InferenceLevel
        ) -> List[VisualAction]:
            nonlocal completed_count
            async with semaphore:
                actions = await extractor.extract_pair(pair, ctx, level)
                if actions:
                    cache_key = self._build_cache_key(pair)
                    self._semantic_cache.put(cache_key, actions)
                completed_count += 1
                _report("extract.vlm", completed_count, total_work)
                return actions

        tasks = []
        for batch, batch_pairs, _, level in all_batches:
            tasks.append(_do_tiled(batch, batch_pairs, level))
        for pair, ctx, level in singles:
            tasks.append(_do_single(pair, ctx, level))

        results = await asyncio.gather(*tasks)
        for result in results:
            all_actions.extend(result)

        dt = time.monotonic() - t_start
        logger.info(
            "VLM tiled extraction: %d batches + %d singles in %.1fs",
            len(all_batches), len(singles), dt,
        )
        return all_actions

    def _build_cache_key(self, pair: FramePair) -> str:
        """Build semantic cache key from pair features."""
        a_emb = pair.frame_a.features.embedding if pair.frame_a.features else None
        b_emb = pair.frame_b.features.embedding if pair.frame_b.features else None
        app = ""
        if pair.frame_a.features:
            app = pair.frame_a.features.detected_app
        return self._semantic_cache.build_key(a_emb, b_emb, app)
