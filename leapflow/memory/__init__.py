"""LeapFlow Memory Subsystem — Provider-based architecture."""
import math

from leapflow.memory.protocol import (
    MemoryEntry, MemoryKind, MemoryProvider, MemoryQuery,
    MemoryToolSchema, SignalDomain,
)
from leapflow.memory.manager import MemoryManager
from leapflow.memory.providers import (
    WorkingMemoryProvider, EpisodicMemoryProvider,
    SemanticMemoryProvider, EvolutionMemoryProvider,
)
from leapflow.memory.providers.episodic import MemoryFragment
from leapflow.memory.providers.semantic import MemoryHit
from leapflow.memory.providers.evolution import SkillEpisode


def decay_weight(
    semantic_score: float,
    age_seconds: float,
    frequency: float,
    *,
    decay_lambda: float = 1e-5,
) -> float:
    """Compute decayed weight: W = S * exp(-λ * t) * log(1 + F)."""
    if semantic_score <= 0 or frequency <= 0:
        return 0.0
    normalized_freq = 1.0 + math.log1p(float(frequency) - 1.0)
    return float(semantic_score) * math.exp(-float(decay_lambda) * float(age_seconds)) * normalized_freq


__all__ = [
    "MemoryEntry", "MemoryKind", "MemoryProvider", "MemoryQuery",
    "MemoryToolSchema", "SignalDomain", "MemoryManager",
    "WorkingMemoryProvider", "EpisodicMemoryProvider",
    "SemanticMemoryProvider", "EvolutionMemoryProvider",
    "MemoryFragment", "MemoryHit", "SkillEpisode", "decay_weight",
]
