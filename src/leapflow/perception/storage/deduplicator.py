"""Frame deduplication — online hash check and offline clustering."""

from __future__ import annotations

from typing import List, Optional, Sequence

from leapflow.perception.cv.phash import hamming_distance
from leapflow.perception.types import Keyframe


class FrameDeduplicator:
    """Remove near-duplicate frames before expensive VLM processing.

    Two-pass approach:
    1. Online: fast hash comparison during capture (prevents storage waste)
    2. Offline: embedding-based clustering before VLM (removes visual redundancy)
    """

    __slots__ = ("_hash_threshold", "_embedding_threshold")

    def __init__(
        self,
        hash_threshold: float = 0.05,
        embedding_threshold: float = 0.92,
    ) -> None:
        self._hash_threshold = hash_threshold
        self._embedding_threshold = embedding_threshold

    def is_duplicate_online(
        self, new_hash: bytes, recent_hashes: Sequence[bytes], check_last: int = 3
    ) -> bool:
        """Quick online check: is this frame too similar to recent captures?

        Uses perceptual hash Hamming distance. Designed for <0.1ms.
        """
        if not recent_hashes:
            return False
        candidates = recent_hashes[-check_last:]
        hash_bits = len(new_hash) * 8
        for h in candidates:
            if len(h) != len(new_hash):
                continue
            dist = hamming_distance(new_hash, h) / max(1, hash_bits)
            if dist < self._hash_threshold:
                return True
        return False

    def deduplicate_offline(self, frames: List[Keyframe]) -> List[Keyframe]:
        """Offline deduplication using embedding similarity.

        Falls back to hash-based dedup if embeddings are not available.
        Keeps the frame with highest info_score from each cluster.
        """
        if len(frames) <= 2:
            return frames

        # Check if embeddings are available
        has_embeddings = all(
            f.features and f.features.embedding for f in frames
        )

        if has_embeddings:
            return self._dedup_by_embedding(frames)
        return self._dedup_by_hash(frames)

    def _dedup_by_embedding(self, frames: List[Keyframe]) -> List[Keyframe]:
        """Cluster by cosine similarity of CLIP embeddings."""
        clusters: List[List[Keyframe]] = []

        for frame in frames:
            placed = False
            emb = frame.features.embedding  # type: ignore[union-attr]
            for cluster in clusters:
                rep = cluster[0].features.embedding  # type: ignore[union-attr]
                sim = self._cosine_similarity(emb, rep)
                if sim > self._embedding_threshold:
                    cluster.append(frame)
                    placed = True
                    break
            if not placed:
                clusters.append([frame])

        # Keep best from each cluster
        result = []
        for cluster in clusters:
            best = max(cluster, key=lambda f: f.info_score)
            result.append(best)
        return sorted(result, key=lambda f: f.timestamp)

    def _dedup_by_hash(self, frames: List[Keyframe]) -> List[Keyframe]:
        """Fallback: dedup using image content hash from phash."""
        from leapflow.perception.cv.phash import phash_64

        result: List[Keyframe] = []
        prev_hash: Optional[bytes] = None

        for frame in frames:
            current_hash = phash_64(frame.image)
            if prev_hash is not None:
                dist = hamming_distance(current_hash, prev_hash) / 64.0
                if dist < self._hash_threshold:
                    # Duplicate — keep the one with higher score
                    if result and frame.info_score > result[-1].info_score:
                        result[-1] = frame
                    continue
            result.append(frame)
            prev_hash = current_hash

        return result

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
