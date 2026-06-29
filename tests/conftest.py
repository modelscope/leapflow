"""Shared fixtures, factories, and stubs for LEAP Agent scenario tests."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from leapflow.config import Settings
from leapflow.domain.events import SystemEvent
from leapflow.domain.trajectory import (
    ActionType,
    Episode,
    RawAction,
    SemanticAction,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)
from leapflow.llm.base import LLMChatResponse, LLMProvider
from leapflow.memory import (
    EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider,
)
from leapflow.skills.registry import Skill, SkillMetadata, SkillRegistry
from leapflow.storage.trajectory_store import TrajectoryStore


# ═══════════════════════════════════════════════════════════════════
# Stub LLM — scripted responses for deterministic integration tests
# ═══════════════════════════════════════════════════════════════════


class StubLLM(LLMProvider):
    """Deterministic LLM stub that cycles through scripted responses."""

    FINAL_ANSWER = (
        '{"thought":"done","action":{"type":"answer","name":"final",'
        '"payload":{"text":"ok"}}}'
    )

    def __init__(self, replies: Optional[List[str]] = None) -> None:
        self._replies = list(replies or [])
        self._call_count = 0

    async def achat(
        self,
        messages: List[dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> LLMChatResponse:
        if self._call_count < len(self._replies):
            text = self._replies[self._call_count]
        else:
            text = self.FINAL_ANSWER
        self._call_count += 1
        return LLMChatResponse(content=text)

    async def achat_stream(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if False:
            yield ""  # pragma: no cover

    @property
    def call_count(self) -> int:
        return self._call_count


# ═══════════════════════════════════════════════════════════════════
# Reusable pytest fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Ephemeral DuckDB path for storage tests."""
    return tmp_path / "test.duckdb"


@pytest.fixture
def trajectory_store(tmp_db: Path) -> TrajectoryStore:
    s = TrajectoryStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def long_term_memory(tmp_db: Path) -> SemanticMemoryProvider:
    lt = SemanticMemoryProvider(db_path=tmp_db)
    lt._ensure_connection()
    yield lt
    lt.close()


@pytest.fixture
def working_memory() -> WorkingMemoryProvider:
    return WorkingMemoryProvider(max_tokens=2048)


@pytest.fixture
def immediate_memory() -> EpisodicMemoryProvider:
    return EpisodicMemoryProvider(ttl=300.0)


@pytest.fixture
def skill_registry() -> SkillRegistry:
    return SkillRegistry()


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def skill_library(tmp_path: Path):
    from leapflow.storage.skill_library import SkillLibraryStore
    s = SkillLibraryStore(tmp_path / "skills.duckdb")
    yield s
    s.close()


@pytest.fixture
def imitation_pipeline(trajectory_store: TrajectoryStore):
    from leapflow.analysis.pipeline import ImitationPipeline
    return ImitationPipeline(store=trajectory_store)


# ═══════════════════════════════════════════════════════════════════
# Factory helpers
# ═══════════════════════════════════════════════════════════════════


def make_event(
    event_type: str,
    payload: Optional[dict] = None,
    ts: Optional[float] = None,
    source: str = "",
) -> SystemEvent:
    """Create a SystemEvent with sensible defaults."""
    return SystemEvent(
        event_type=event_type,
        source=source,
        payload=payload or {},
        timestamp=ts or time.time(),
    )


def make_action(
    name: str,
    *,
    params: Optional[dict] = None,
    description: str = "",
    raw_range: tuple[int, int] = (0, 1),
) -> SemanticAction:
    """Create a SemanticAction for synthesis/denoise tests."""
    return SemanticAction(
        action_name=name,
        description=description or name,
        parameters=params or {},
        raw_action_range=raw_range,
    )


def make_skill(
    name: str = "test_skill",
    *,
    description: str = "A test skill",
    version: int = 1,
    confidence: float = 0.7,
    triggers: Optional[List[str]] = None,
    run_fn: Optional[Any] = None,
) -> Skill:
    """Create a Skill with a default async run function."""
    async def _default_run(**kw: Any) -> str:
        return "ok"

    return Skill(
        name=name,
        description=description,
        run=run_fn or _default_run,
        triggers=triggers or [],
        metadata=SkillMetadata(
            version=version,
            confidence=confidence,
            source="test",
        ),
    )


def make_episode(
    traj_id: str = "traj_1",
    ep_id: str = "ep_1",
    *,
    app_seq: Optional[List[str]] = None,
    actions: Optional[List[SemanticAction]] = None,
) -> Episode:
    """Create an Episode for learning/feedback tests."""
    return Episode(
        trajectory_id=traj_id,
        episode_id=ep_id,
        start_idx=0,
        end_idx=3,
        app_sequence=app_seq or ["com.apple.finder"],
        semantic_actions=actions or [
            make_action("list", description="List dir", raw_range=(0, 1)),
            make_action("classify", description="Classify files", raw_range=(1, 2)),
            make_action("move", description="Move files", raw_range=(2, 3)),
        ],
    )


def make_candidate(
    title: str = "Organize downloads",
    *,
    steps: Optional[List[str]] = None,
    triggers: Optional[List[str]] = None,
    confidence: float = 0.7,
    traj_id: str = "traj_1",
    ep_id: str = "ep_1",
) -> Any:
    """Create a DistillationCandidate."""
    from leapflow.learning.distiller import DistillationCandidate
    return DistillationCandidate(
        title=title,
        trigger_phrases=triggers or ["organize files", "sort downloads"],
        steps=steps or ["List directory", "Classify by type", "Move files"],
        parameters=[{"name": "path", "description": "target dir"}],
        pre_conditions=["com.apple.finder available"],
        source_trajectory_id=traj_id,
        source_episode_id=ep_id,
        confidence=confidence,
    )


def make_settings(tmp_dir: str) -> Settings:
    """Create a Settings instance suitable for integration tests."""
    return Settings(
        llm_api_key="sk-test",
        llm_base_url="https://example.invalid/v1",
        llm_model="test-model",
        llm_max_retries=1,
        bridge_socket=Path(tmp_dir) / "sock",
        mock_host=True,
        duckdb_path=Path(tmp_dir) / "mem.duckdb",
        log_level="WARNING",
        prediction_enabled=False,
    )
