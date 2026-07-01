"""Hub client facade — routes operations to the appropriate backend.

Provides a unified interface for all Hub operations, delegating to
backend-specific implementations based on hub_type configuration.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from leapflow.hub.protocol import (
    HubBackend,
    PushResult,
    SkillBundle,
    SkillManifest,
    SkillSummary,
    UserInfo,
    Visibility,
)

logger = logging.getLogger(__name__)


# ─── Sync Plan ───────────────────────────────────────────────────────────────


class SyncPlan:
    """Plan describing what needs to happen during a sync operation."""

    def __init__(
        self,
        to_push: Optional[List[SkillManifest]] = None,
        to_pull: Optional[List[SkillSummary]] = None,
        conflicts: Optional[List[str]] = None,
    ):
        self.to_push: List[SkillManifest] = to_push or []
        self.to_pull: List[SkillSummary] = to_pull or []
        self.conflicts: List[str] = conflicts or []

    @property
    def is_empty(self) -> bool:
        """Return True if no sync actions are needed."""
        return not self.to_push and not self.to_pull and not self.conflicts

    def __repr__(self) -> str:
        return (
            f"SyncPlan(push={len(self.to_push)}, "
            f"pull={len(self.to_pull)}, "
            f"conflicts={len(self.conflicts)})"
        )


# ─── HubClient Facade ────────────────────────────────────────────────────────


class HubClient:
    """Central hub access point with backend registry and routing.

    Backend selection is driven by hub_type parameter (from config or explicit).
    Backends are registered lazily via factory callables to avoid import errors
    for uninstalled SDKs.
    """

    _registry: Dict[str, Callable[[], HubBackend]] = {}

    @classmethod
    def register(cls, hub_type: str, factory: Callable[[], HubBackend]) -> None:
        """Register a backend factory.

        Args:
            hub_type: Backend identifier (e.g. 'modelscope', 'local').
            factory: Callable that creates a backend instance when invoked.
        """
        cls._registry[hub_type] = factory
        logger.debug("Registered hub backend: %s", hub_type)

    @classmethod
    def available_backends(cls) -> List[str]:
        """Return list of registered backend type names."""
        return list(cls._registry.keys())

    def __init__(
        self,
        hub_type: str = "modelscope",
        default_owner: str = "",
        default_visibility: str = "private",
        repo_prefix: str = "leapflow-",
    ):
        """Initialize HubClient with config-driven defaults.

        Args:
            hub_type: Which backend to use for operations.
            default_owner: Default owner/org for repo_id construction.
            default_visibility: Default visibility for new repos.
            repo_prefix: Prefix prepended to skill names in repo_id construction.
        """
        self._hub_type = hub_type
        self._default_owner = default_owner
        self._default_visibility = Visibility(default_visibility)
        self._repo_prefix = repo_prefix
        self._backend: Optional[HubBackend] = None

    @property
    def hub_type(self) -> str:
        """Return the active hub type."""
        return self._hub_type

    @property
    def backend(self) -> HubBackend:
        """Lazily instantiate the active backend.

        Raises:
            ValueError: If the requested hub_type is not registered.
            ImportError: If the backend SDK is not available.
        """
        if self._backend is None:
            factory = self._registry.get(self._hub_type)
            if factory is None:
                available = ", ".join(self._registry.keys()) or "(none)"
                raise ValueError(
                    f"Unknown hub backend: '{self._hub_type}'. "
                    f"Available: {available}"
                )
            self._backend = factory()
            logger.info("Initialized hub backend: %s", self._hub_type)
        return self._backend

    def _build_repo_id(self, skill_name: str) -> str:
        """Construct repo_id from owner, prefix and skill name.

        Format: {owner}/{repo_prefix}{skill_name}
        """
        prefix = f"{self._default_owner}/" if self._default_owner else ""
        return f"{prefix}{self._repo_prefix}{skill_name}"

    # ─── Public API ──────────────────────────────────────────────────────────

    async def push(
        self,
        bundle: SkillBundle,
        skill_name: str | None = None,
        repo_id: str | None = None,
        visibility: Visibility | None = None,
    ) -> PushResult:
        """Push a skill bundle to the remote hub.

        Args:
            bundle: Complete skill package to upload.
            skill_name: Skill name (used to construct repo_id if repo_id not given).
            repo_id: Explicit repo_id (overrides auto-construction).
            visibility: Repository visibility (defaults to client default).

        Returns:
            PushResult with published version and URL.
        """
        if repo_id is None:
            name = skill_name or bundle.manifest.name
            repo_id = self._build_repo_id(name)

        vis = visibility or self._default_visibility

        return await self.backend.push_skill(bundle, repo_id, vis)

    async def pull(
        self,
        repo_id: str,
        version: str | None = None,
    ) -> SkillBundle:
        """Pull a skill bundle from the remote hub.

        Args:
            repo_id: Repository identifier to pull from.
            version: Specific version (None = latest).

        Returns:
            Complete SkillBundle.
        """
        return await self.backend.pull_skill(repo_id, version)

    async def search(
        self,
        query: str,
        owner: str | None = None,
    ) -> List[SkillSummary]:
        """Search for skills on the remote hub.

        Args:
            query: Free-text search query.
            owner: Filter by owner/organization.

        Returns:
            List of matching skill summaries.
        """
        return await self.backend.list_remote_skills(owner=owner, query=query)

    async def sync_skills(
        self,
        local_skills: List[SkillManifest],
        remote_owner: str | None = None,
    ) -> SyncPlan:
        """Compute a sync plan between local and remote skills.

        Compares local skill manifests with remote listings to determine
        what needs to be pushed, pulled, or resolved as conflicts.

        Args:
            local_skills: List of locally available skill manifests.
            remote_owner: Owner to filter remote skills.

        Returns:
            SyncPlan describing required actions.
        """
        owner = remote_owner or self._default_owner
        remote_skills = await self.backend.list_remote_skills(owner=owner)

        # Build lookup maps
        local_by_name = {s.name: s for s in local_skills}
        remote_by_name = {s.name: s for s in remote_skills}

        to_push: List[SkillManifest] = []
        to_pull: List[SkillSummary] = []
        conflicts: List[str] = []

        # Skills only local → push
        for name, local in local_by_name.items():
            if name not in remote_by_name:
                to_push.append(local)
            else:
                # Both exist — compare versions
                remote = remote_by_name[name]
                if local.version != remote.version:
                    conflicts.append(name)

        # Skills only remote → pull
        for name, remote in remote_by_name.items():
            if name not in local_by_name:
                to_pull.append(remote)

        return SyncPlan(to_push=to_push, to_pull=to_pull, conflicts=conflicts)

    async def login(self) -> UserInfo:
        """Authenticate with the hub backend.

        Returns:
            Authenticated user info.
        """
        return await self.backend.authenticate()

    async def whoami(self) -> UserInfo:
        """Get current authenticated user info.

        Returns:
            Current user info.
        """
        return await self.backend.authenticate()


# ─── Default Backend Registration ────────────────────────────────────────────


def _register_defaults() -> None:
    """Register built-in backend factories."""

    # Local backend — always available
    def _local_factory() -> HubBackend:
        from leapflow.hub.backends.local import LocalBackend

        return LocalBackend()  # type: ignore[return-value]

    HubClient.register("local", _local_factory)

    # ModelScope backend — available if SDK installed
    def _modelscope_factory() -> HubBackend:
        from leapflow.hub.backends.modelscope import ModelScopeBackend

        return ModelScopeBackend()  # type: ignore[return-value]

    HubClient.register("modelscope", _modelscope_factory)

    # HuggingFace backend — placeholder
    def _huggingface_factory() -> HubBackend:
        from leapflow.hub.backends.huggingface import HuggingFaceBackend

        return HuggingFaceBackend()  # type: ignore[return-value]

    HubClient.register("huggingface", _huggingface_factory)


_register_defaults()
