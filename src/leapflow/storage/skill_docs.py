"""Filesystem-based Skill Document store.

Manages skill-name/SKILL.md folder structure on disk and provides
LLM-backed SkillFn registration for runtime execution.

When an ExecutionPort is available, skills use ToolUseSkillExecutor
(ReAct loop with real tool calls). Otherwise falls back to text-only
LLM reasoning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from leapflow.learning.document import SkillDocParser, SkillDocRenderer, SkillDocument
from leapflow.skills.registry import Skill, SkillFn, SkillMetadata, SkillParameter
from leapflow.storage.bundle_writer import BundleContext, BundleFiles, BundleWriter

logger = logging.getLogger(__name__)

_SKILL_FILENAME = "SKILL.md"


class SkillDocStore:
    """CRUD for SKILL.md folders on the filesystem."""

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._renderer = SkillDocRenderer()
        self._parser = SkillDocParser()
        self._bundle_writer = BundleWriter()

    @property
    def skills_dir(self) -> Path:
        return self._dir

    def save(self, doc: SkillDocument, *, bundle: Optional[BundleFiles] = None) -> Path:
        """Write skill-name/SKILL.md (and optional bundle files) to the filesystem."""
        folder = self._dir / doc.name
        folder.mkdir(parents=True, exist_ok=True)
        content = self._renderer.render(doc)
        skill_file = folder / _SKILL_FILENAME
        skill_file.write_text(content, encoding="utf-8")
        if bundle is not None:
            self._bundle_writer.write_bundle(folder, bundle)
        logger.info("doc_store.saved name=%s path=%s", doc.name, skill_file)
        return folder

    def load_bundle_context(self, name: str) -> BundleContext:
        """Scan a skill folder for auxiliary bundle files."""
        folder = self._dir / name
        return BundleWriter.load_bundle_context(folder)

    def load(self, name: str) -> Optional[SkillDocument]:
        """Load a skill document by name."""
        skill_file = self._dir / name / _SKILL_FILENAME
        if not skill_file.exists():
            return None
        content = skill_file.read_text(encoding="utf-8")
        return self._parser.parse(content)

    def load_all(self) -> List[SkillDocument]:
        """Load all valid skill documents from the store."""
        docs: List[SkillDocument] = []
        if not self._dir.exists():
            return docs
        for folder in sorted(self._dir.iterdir()):
            if not folder.is_dir():
                continue
            skill_file = folder / _SKILL_FILENAME
            if skill_file.exists():
                try:
                    content = skill_file.read_text(encoding="utf-8")
                    doc = self._parser.parse(content)
                    if doc.name:
                        docs.append(doc)
                except Exception as e:
                    logger.debug("doc_store.load_failed folder=%s error=%s", folder.name, e)
        return docs

    def delete(self, name: str) -> bool:
        """Remove a skill folder. Returns True if it existed."""
        folder = self._dir / name
        if not folder.exists():
            return False
        import shutil
        shutil.rmtree(folder)
        logger.info("doc_store.deleted name=%s", name)
        return True

    def list_names(self) -> List[str]:
        """List all skill names (folder names containing SKILL.md)."""
        names: List[str] = []
        if not self._dir.exists():
            return names
        for folder in sorted(self._dir.iterdir()):
            if folder.is_dir() and (folder / _SKILL_FILENAME).exists():
                names.append(folder.name)
        return names

    def exists(self, name: str) -> bool:
        return (self._dir / name / _SKILL_FILENAME).exists()

    def load_as_skill(
        self, name: str, llm: Any, *, execution: Any = None, perception: Any = None
    ) -> Optional[Skill]:
        """Load a SKILL.md and create an executable Skill for SkillRegistry.

        When execution port is provided, the skill uses a ReAct tool-use
        executor that performs real operations. Otherwise falls back to
        text-only LLM reasoning.
        """
        doc = self.load(name)
        if doc is None:
            return None
        return self._build_skill_from_doc(doc, llm, execution, perception)

    def load_all_as_skills(
        self, llm: Any, *, execution: Any = None, perception: Any = None
    ) -> List[Skill]:
        """Load all skill documents and create executable Skills."""
        skills: List[Skill] = []
        for doc in self.load_all():
            skill = self._build_skill_from_doc(doc, llm, execution, perception)
            if skill:
                skills.append(skill)
        return skills

    def _build_skill_from_doc(
        self, doc: SkillDocument, llm: Any, execution: Any = None, perception: Any = None,
    ) -> Optional[Skill]:
        """Create a SkillFn from a SkillDocument.

        Strategy: tool-use executor when execution port available,
        text-only LLM fallback otherwise.
        """
        full_content = self._renderer.render(doc)
        bundle_ctx = self.load_bundle_context(doc.name)

        if execution is not None:
            skill_fn = _make_tool_use_skill_fn(
                full_content, doc, llm, execution, bundle_ctx, perception=perception,
            )
        else:
            skill_fn = _make_llm_skill_fn(full_content, doc, llm)

        parameters = [
            SkillParameter(
                name=p.name,
                type=p.type,
                required=p.required,
                default=p.default,
                description=p.description,
            )
            for p in doc.parameters
        ]

        triggers = _extract_triggers(doc)

        metadata = SkillMetadata(
            source="skill.md",
            confidence=doc.metadata.get("confidence", 0.8),
            version=doc.metadata.get("version", 1),
        )

        return Skill(
            name=doc.name,
            description=doc.description,
            run=skill_fn,
            parameters=parameters,
            preconditions=doc.preconditions,
            postconditions=doc.postconditions,
            instructions=list(doc.instructions),
            triggers=triggers,
            metadata=metadata,
        )


def _make_tool_use_skill_fn(
    skill_content: str, doc: SkillDocument, llm: Any, execution: Any,
    bundle_context: Optional[BundleContext] = None,
    perception: Any = None,
) -> SkillFn:
    """Create a SkillFn backed by ReAct tool-use execution."""
    from leapflow.skills.bridge_factory import build_tool_bridge
    from leapflow.skills.tool_executor import ToolUseSkillExecutor

    bridge = build_tool_bridge(execution, perception)
    executor = ToolUseSkillExecutor(
        llm=llm,
        bridge=bridge,
        skill_content=skill_content,
        instructions=list(doc.instructions),
        bundle_context=bundle_context,
    )
    return executor.run


def _make_llm_skill_fn(skill_content: str, doc: SkillDocument, llm: Any) -> SkillFn:
    """Fallback: text-only LLM reasoning (no actual tool execution)."""

    async def _run(*, user_goal: str = "", **kwargs: Any) -> str:
        from leapflow.llm.message_builder import build_system_message, build_user_message_text

        params_desc = ""
        if kwargs:
            params_desc = "\n".join(f"- {k}: {v}" for k, v in kwargs.items())
            params_desc = f"\nProvided parameters:\n{params_desc}"

        goal = user_goal or doc.goal
        user_msg = f"Execute this skill. Goal: {goal}{params_desc}"

        resp = await llm.achat(
            [
                build_system_message(
                    f"You are executing a skill. Follow the instructions precisely.\n\n{skill_content}"
                ),
                build_user_message_text(user_msg),
            ],
            stream=False,
            enable_thinking=False,
        )
        return resp.content or ""

    return _run


def _extract_triggers(doc: SkillDocument) -> List[str]:
    """Extract trigger phrases from examples and description.

    Priority: explicit example triggers > quoted phrases in description.
    """
    triggers: List[str] = []

    for ex in doc.examples:
        if ex.trigger and ex.trigger not in triggers:
            triggers.append(ex.trigger)

    import re
    for match in re.findall(r'"([^"]+)"', doc.description):
        if match not in triggers:
            triggers.append(match)

    return triggers[:10]
