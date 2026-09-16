# Copyright (c) Alibaba, Inc. and its affiliates.
"""Scenario-based tests for the skill lifecycle.

Covers registration, invocation, SkillDocument roundtrip, doc store CRUD,
doc generation, title normalization, and Darwin undo stack behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import make_candidate, make_skill

from leapflow.domain.platform import PlatformManifest
from leapflow.learning.document import (
    ExampleDoc,
    ParameterDoc,
    SkillDocParser,
    SkillDocRenderer,
    SkillDocument,
    title_to_kebab,
)
from leapflow.learning.doc_generator import CompositeSkillDocGenerator, DocGenContext
from leapflow.platform.adapters.darwin import DarwinExecutionAdapter
from leapflow.skills.registry import SkillMetadata, SkillRegistry
from leapflow.storage.skill_docs import SkillDocStore


# ── Mock RPC for undo stack tests ──


class MockRpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.files: dict[str, str] = {}

    async def call(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, params))
        if method == "file.copy":
            src = (params or {}).get("source", "")
            dst = (params or {}).get("destination", "")
            self.files[dst] = self.files.get(src, f"content:{src}")
            return {"ok": True}
        if method == "file.move":
            return {"ok": True}
        if method == "file.delete":
            path = (params or {}).get("path", "")
            self.files.pop(path, None)
            return {"ok": True}
        if method == "file.list":
            return {"items": []}
        return {"ok": True}


@pytest.fixture
def undo_adapter() -> DarwinExecutionAdapter:
    rpc = MockRpc()
    manifest = PlatformManifest.default_darwin()
    return DarwinExecutionAdapter(rpc, manifest, undo_capacity=5)


# ═══════════════════════════════════════════════════════════════════
# Skill registry
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_register_and_invoke_skill() -> None:
    """Register a skill, invoke it, and verify the output."""
    reg = SkillRegistry()

    async def _run(**kw: Any) -> str:
        return f"done:{kw.get('path', 'none')}"

    skill = make_skill(
        "organize_files",
        run_fn=_run,
        description="Organize files by type",
    )
    reg.register(skill)

    result = await reg.invoke("organize_files", path="/tmp/downloads")
    assert result.ok is True
    assert result.output == "done:/tmp/downloads"


def test_list_skills_after_registration() -> None:
    """Register multiple skills and verify names() returns all of them."""
    reg = SkillRegistry()
    reg.register(make_skill("alpha", description="First skill"))
    reg.register(make_skill("beta", description="Second skill"))
    reg.register(make_skill("gamma", description="Third skill"))

    names = reg.names()
    assert names == ["alpha", "beta", "gamma"]
    assert reg.count == 3


def test_skill_metadata_source_tracking() -> None:
    """Metadata source and version are preserved after registration."""
    reg = SkillRegistry()
    skill = make_skill(
        "tracked_skill",
        version=3,
        description="Skill with provenance",
    )
    skill.metadata = SkillMetadata(
        source="distilled",
        version=3,
        confidence=0.85,
        source_trajectory_id="traj_42",
        source_episode_id="ep_7",
    )
    reg.register(skill)

    loaded = reg.get("tracked_skill")
    assert loaded is not None
    assert loaded.metadata.source == "distilled"
    assert loaded.metadata.version == 3
    assert loaded.metadata.source_trajectory_id == "traj_42"
    assert loaded.metadata.source_episode_id == "ep_7"


# ═══════════════════════════════════════════════════════════════════
# SkillDocument
# ═══════════════════════════════════════════════════════════════════


def test_skill_doc_roundtrip() -> None:
    """Render → parse preserves core SkillDocument fields."""
    original = SkillDocument(
        name="organize-files",
        description="Organize files by type",
        goal="Move files into type-based folders",
        instructions=["List directory", "Classify by extension", "Move files"],
        parameters=[ParameterDoc(name="path", type="str", description="Target dir")],
        examples=[
            ExampleDoc(
                trigger="organize ~/Downloads",
                actions=["list", "classify", "move"],
                result="Files organized",
            )
        ],
        allowed_tools="Bash(find:*) Bash(mv:*)",
        preconditions=["Target directory exists"],
        postconditions=["Files sorted by type"],
        metadata={"version": 1, "source": "learned"},
    )

    rendered = SkillDocRenderer().render(original)
    parsed = SkillDocParser().parse(rendered)

    assert parsed.name == original.name
    assert parsed.description == original.description
    assert parsed.goal == original.goal
    assert parsed.instructions == original.instructions
    assert parsed.allowed_tools == original.allowed_tools
    assert parsed.preconditions == original.preconditions
    assert parsed.postconditions == original.postconditions
    assert len(parsed.parameters) == 1
    assert parsed.parameters[0].name == "path"
    assert len(parsed.examples) == 1
    assert parsed.examples[0].trigger == "organize ~/Downloads"
    assert parsed.examples[0].result == "Files organized"


def test_title_to_kebab() -> None:
    """Various title formats normalize to kebab-case skill names."""
    assert title_to_kebab("Organize Files By Extension") == "organize-files-by-extension"
    assert title_to_kebab("Organize Downloads by Type") == "organize-downloads-by-type"
    assert title_to_kebab("batch_rename files") == "batch-rename-files"
    assert title_to_kebab("My Skill! (v2)") == "my-skill-v2"
    assert title_to_kebab("a - b -- c") == "a-b-c"
    assert title_to_kebab("") == "unnamed-skill"


# ═══════════════════════════════════════════════════════════════════
# SkillDocStore
# ═══════════════════════════════════════════════════════════════════


def test_skill_doc_store_crud(tmp_path: Path) -> None:
    """Save, load, exists, list_names, and delete round-trip."""
    store = SkillDocStore(tmp_path / "skills")
    doc = SkillDocument(
        name="test-skill",
        description="A test",
        goal="Test",
        instructions=["S1", "S2"],
    )

    assert not store.exists("test-skill")
    folder = store.save(doc)
    assert folder.exists()
    assert (folder / "SKILL.md").exists()
    assert store.exists("test-skill")

    loaded = store.load("test-skill")
    assert loaded is not None
    assert loaded.name == "test-skill"
    assert loaded.goal == "Test"
    assert loaded.instructions == ["S1", "S2"]

    names = store.list_names()
    assert "test-skill" in names

    assert store.delete("test-skill") is True
    assert not store.exists("test-skill")
    assert store.load("test-skill") is None


# ═══════════════════════════════════════════════════════════════════
# Doc generation
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_composite_doc_generator_template() -> None:
    """Template generator produces a doc from a known file-organize candidate."""
    gen = CompositeSkillDocGenerator(llm_generator=None)
    candidate = make_candidate(
        title="Organize files by extension",
        steps=["classify files", "sort them", "move by extension"],
        triggers=["organize"],
    )

    doc = await gen.generate(candidate, DocGenContext())
    assert doc is not None
    assert "organize" in doc.name
    assert len(doc.instructions) > 0
    assert doc.metadata.get("pattern") == "file_organize"


@pytest.mark.asyncio
async def test_skill_doc_name_dedup() -> None:
    """Existing skill names trigger a -2 suffix on generated doc names."""
    gen = CompositeSkillDocGenerator(llm_generator=None)
    candidate = make_candidate(
        title="Organize files by extension",
        steps=["classify files", "sort them", "move by extension"],
        triggers=["organize"],
    )
    context = DocGenContext(existing_skill_names=["organize-files-by-extension"])

    doc = await gen.generate(candidate, context)
    assert doc is not None
    assert doc.name == "organize-files-by-extension-2"


# ═══════════════════════════════════════════════════════════════════
# Undo stack
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_undo_stack_basic(undo_adapter: DarwinExecutionAdapter) -> None:
    """A copy operation is tracked and undo_last clears the stack."""
    await undo_adapter.perform_file_op("copy", {"source": "/a", "destination": "/b"})
    assert undo_adapter.undo_depth == 1

    result = await undo_adapter.undo_last()
    assert result.get("ok") is not False
    assert undo_adapter.undo_depth == 0


@pytest.mark.asyncio
async def test_undo_stack_capacity_limit(undo_adapter: DarwinExecutionAdapter) -> None:
    """Stack evicts oldest entries when capacity is exceeded."""
    for i in range(7):
        await undo_adapter.perform_file_op(
            "move", {"source": f"/s{i}", "destination": f"/d{i}"}
        )
    assert undo_adapter.undo_depth == 5


@pytest.mark.asyncio
async def test_undo_multiple(undo_adapter: DarwinExecutionAdapter) -> None:
    """Undoing two of three operations leaves one on the stack."""
    await undo_adapter.perform_file_op("move", {"source": "/a", "destination": "/b"})
    await undo_adapter.perform_file_op("move", {"source": "/c", "destination": "/d"})
    await undo_adapter.perform_file_op("copy", {"source": "/e", "destination": "/f"})
    assert undo_adapter.undo_depth == 3

    results = await undo_adapter.undo(2)
    assert len(results) == 2
    assert undo_adapter.undo_depth == 1


# ── Stored-skill fallback registration dedup ──


class _FakeSkillLibrary:
    """Persistence-boundary fake: returns prebuilt StoredSkills."""

    def __init__(self, skills: list) -> None:
        self._skills = skills

    def load_all_active(self) -> list:
        return list(self._skills)


def test_fallback_registration_dedups_doc_backed_skill() -> None:
    """A DuckDB skill whose SKILL.md counterpart is already registered must
    not be registered a second time under a differently derived name."""
    from leapflow.cli.context import _register_stored_skill_fallbacks
    from leapflow.learning.document import title_to_kebab
    from leapflow.skills.registry import Skill
    from leapflow.storage.skill_library import StoredSkill

    title = "移除下载过的安装包"
    kebab_name = title_to_kebab(title)
    assert kebab_name == "skill-移除下载过的安装包"

    registry = SkillRegistry()
    registry.register(Skill(
        name=kebab_name,
        description=title,
        run=lambda **kwargs: "",
    ))

    store = _FakeSkillLibrary([StoredSkill(
        skill_id="s1", title=title,
        trigger_phrases=[title], steps=["step"],
    )])

    registered = _register_stored_skill_fallbacks(store, registry, llm=None)

    assert registered == 0
    assert registry.names().count(kebab_name) == 1
    assert title not in registry.names()


def test_fallback_registration_adds_undocced_skill_under_kebab_name() -> None:
    """A DuckDB skill with no doc counterpart is registered once, under the
    shared kebab naming."""
    from leapflow.cli.context import _register_stored_skill_fallbacks
    from leapflow.learning.document import title_to_kebab
    from leapflow.storage.skill_library import StoredSkill

    title = "整理桌面文件"
    registry = SkillRegistry()
    store = _FakeSkillLibrary([StoredSkill(
        skill_id="s2", title=title,
        trigger_phrases=[title], steps=["step"],
    )])

    registered = _register_stored_skill_fallbacks(store, registry, llm=None)

    assert registered == 1
    assert registry.names() == [title_to_kebab(title)]
