"""Skill activation — compile generated code into executable skills.

Bridges the gap between StoredSkill (declarative data) and Skill (executable).
Compiles code from codegen into async callables, binds ExecutionPort/PerceptionPort,
and registers them into SkillRegistry for runtime use.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from leapflow.skills.registry import Skill, SkillFn, SkillMetadata, SkillParameter

if TYPE_CHECKING:
    from leapflow.domain.trajectory import Episode
    from leapflow.learning.codegen import (
        SkillCodeGenerator,
    )
    from leapflow.learning.distiller import DistillationCandidate
    from leapflow.storage.skill_library import SkillLibraryStore
    from leapflow.skills.registry import SkillRegistry
    from leapflow.domain.events import ExecutionPort, PerceptionPort

logger = logging.getLogger(__name__)


class SkillActivator:
    """Compiles generated skill code and registers executable skills.

    Single responsibility: code compilation + SkillRegistry lifecycle.
    Depends on ExecutionPort/PerceptionPort to bind as runtime context
    for generated skill functions.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        library: SkillLibraryStore,
        execution: ExecutionPort,
        perception: PerceptionPort,
        *,
        codegen: Optional[SkillCodeGenerator] = None,
    ) -> None:
        self._registry = registry
        self._library = library
        self._execution = execution
        self._perception = perception
        self._codegen = codegen

    def activate_from_code(
        self,
        name: str,
        code: str,
        description: str,
        *,
        parameters: Optional[List[SkillParameter]] = None,
        triggers: Optional[List[str]] = None,
        metadata: Optional[SkillMetadata] = None,
    ) -> Optional[Skill]:
        """Compile code string → bind ports → register in registry.

        Returns the registered Skill, or None if compilation fails.
        """
        func_name = self._extract_func_name(code)
        if not func_name:
            logger.warning("activator.no_func_name code=%.60s...", code)
            return None

        try:
            skill_fn = self._compile_skill_fn(code, func_name)
        except Exception as e:
            logger.warning("activator.compile_failed name=%s error=%s", name, e)
            return None

        meta = metadata or SkillMetadata(source="distilled")
        skill = Skill(
            name=name,
            description=description,
            run=skill_fn,
            parameters=list(parameters or []),
            triggers=list(triggers or []),
            metadata=meta,
        )
        self._registry.register(skill)
        logger.info(
            "activator.registered name=%s source=%s v%d",
            name, meta.source, meta.version,
        )
        return skill

    def load_and_activate_all(self) -> int:
        """Load all active parameterized skills from library and activate
        those with stored code. Called at startup.

        Returns count of successfully activated skills.
        """
        records = self._library.load_all_active_parameterized()
        activated = 0

        for rec in records:
            code = rec.get("code")
            if not code:
                continue

            name = rec["name"]
            params = [
                SkillParameter(
                    name=p.get("name", ""),
                    type=p.get("type", "str"),
                    required=p.get("required", False),
                    default=p.get("default"),
                    description=p.get("description", ""),
                )
                for p in rec.get("parameters", [])
            ]
            meta = SkillMetadata(
                source=rec.get("source", "distilled"),
                source_trajectory_id=rec.get("source_trajectory_id"),
                source_episode_id=rec.get("source_episode_id"),
                confidence=rec.get("confidence", 0.5),
                version=rec.get("version", 1),
                created_at=rec.get("created_at", 0.0),
            )
            skill = self.activate_from_code(
                name, code, rec.get("description", ""),
                parameters=params,
                triggers=rec.get("triggers", []),
                metadata=meta,
            )
            if skill:
                activated += 1

        return activated

    async def activate_candidate(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
    ) -> Optional[Skill]:
        """Full pipeline: candidate → codegen → validate → persist → register.

        Called by ActiveLearningObserver after saving a new StoredSkill.
        Requires codegen to be configured.
        """
        if self._codegen is None:
            logger.debug("activator.no_codegen; skipping activation")
            return None

        from leapflow.learning.codegen import build_default_context

        ctx = build_default_context(
            existing_skills=self._registry.names(),
            episode=episode,
        )

        generated = await self._codegen.generate(candidate, ctx)
        if generated is None or not generated.is_valid:
            logger.debug("activator.codegen_failed candidate=%s", candidate.title)
            return None

        name = self._name_from_title(candidate.title)
        meta = SkillMetadata(
            source="distilled",
            source_trajectory_id=candidate.source_trajectory_id,
            source_episode_id=candidate.source_episode_id,
            confidence=generated.confidence,
            version=1,
            created_at=time.time(),
        )

        skill = self.activate_from_code(
            name, generated.code, generated.description,
            parameters=generated.parameters,
            triggers=generated.triggers,
            metadata=meta,
        )
        if skill is None:
            return None

        self._library.save_parameterized_skill(skill, code=generated.code)
        logger.info("activator.persisted name=%s", name)
        return skill

    def reactivate(self, name: str, new_code: str) -> Optional[Skill]:
        """Re-compile and re-register when a skill is updated.

        Used after feedback loop auto-improvement.
        """
        rec = self._library.load_parameterized_skill(name)
        if rec is None:
            logger.debug("activator.reactivate_not_found name=%s", name)
            return None

        code = new_code or rec.get("code", "")
        if not code:
            return None

        params = [
            SkillParameter(
                name=p.get("name", ""),
                type=p.get("type", "str"),
                required=p.get("required", False),
                default=p.get("default"),
                description=p.get("description", ""),
            )
            for p in rec.get("parameters", [])
        ]
        meta = SkillMetadata(
            source=rec.get("source", "distilled"),
            confidence=rec.get("confidence", 0.5),
            version=rec.get("version", 1) + 1,
        )
        return self.activate_from_code(
            name, code, rec.get("description", ""),
            parameters=params,
            triggers=rec.get("triggers", []),
            metadata=meta,
        )

    def deactivate(self, name: str) -> bool:
        """Unregister from registry and deactivate in library."""
        removed = self._registry.unregister(name)
        if removed:
            self._library.deactivate_parameterized(name)
        return removed

    def _compile_skill_fn(self, code: str, func_name: str) -> SkillFn:
        """Compile code string into an async callable with bound ports.

        Runs in a sandboxed namespace that restricts access to filesystem,
        module loading, and other dangerous operations. Skills can only
        interact with the system through their bound ports.
        """
        from leapflow.skills.sandbox import SandboxedNamespace

        raw_fn = SandboxedNamespace.compile_skill(code, func_name)
        execution = self._execution
        perception = self._perception

        async def skill_fn(*, user_goal: str = "", **kwargs: Any) -> str:
            result = await raw_fn(execution, perception, **kwargs)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return str(result)

        return skill_fn

    @staticmethod
    def _extract_func_name(code: str) -> Optional[str]:
        """Extract async function name from code string."""
        import re
        match = re.search(r"async\s+def\s+(\w+)", code)
        return match.group(1) if match else None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Ensure skill name is a valid identifier."""
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"skill_{sanitized}"
        return sanitized or "unnamed_skill"

    @staticmethod
    def _name_from_title(title: str) -> str:
        """Derive registry name from candidate title (user's semantic goal)."""
        import re
        name = re.sub(r"[^\w\s]", "", title.lower())
        name = re.sub(r"\s+", "_", name.strip())
        return name or "unnamed_skill"
