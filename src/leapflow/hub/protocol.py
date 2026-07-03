"""Hub protocol definitions — backend-agnostic types for cloud skill collaboration.

Defines the HubBackend Protocol and all shared data structures used across
different Hub backend implementations (ModelScope, HuggingFace, etc.).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


# ─── Enumerations ────────────────────────────────────────────────────────────


@enum.unique
class Visibility(enum.Enum):
    """Repository visibility level on a Hub platform."""

    PRIVATE = "private"
    INTERNAL = "internal"
    PUBLIC = "public"


@enum.unique
class SkillSourceTag(enum.Enum):
    """Origin tag indicating how a skill was acquired."""

    BUILTIN = "builtin"
    LEARNED = "learned"
    HUB = "hub"
    THIRD_PARTY = "third_party"


# ─── Data Classes ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UserInfo:
    """Authenticated user identity from a Hub backend."""

    username: str
    email: str = ""
    avatar_url: str = ""


@dataclass(frozen=True)
class PushResult:
    """Result returned after successfully pushing a skill bundle."""

    repo_id: str
    version: str
    url: str
    hub_type: str


@dataclass(frozen=True)
class SkillSummary:
    """Lightweight skill listing entry returned by search/list operations."""

    repo_id: str
    name: str
    description: str = ""
    version: str = ""
    downloads: int = 0
    hub_type: str = ""


@dataclass(frozen=True)
class VersionInfo:
    """Version metadata for a single skill revision."""

    version: str
    created_at: str = ""
    commit_sha: str = ""


@dataclass(frozen=True)
class SkillManifest:
    """Declarative manifest describing a skill's identity and requirements."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    parameters: List[dict] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    source_tag: str = "learned"
    tier: int = 1
    leapflow_min_version: str = "0.1.0"
    created_at: str = ""
    author: str = ""
    hub_type: str = ""
    repo_id: str = ""
    # ── Team collaboration fields ──
    content_hash: str = ""           # SHA256[:16] of (source_code + parameters + triggers)
    updated_by: str = ""             # Last modifier username
    updated_at: str = ""             # Last modification ISO timestamp
    tags: List[str] = field(default_factory=list)  # Classification tags


@dataclass(frozen=True)
class SkillBundle:
    """Complete skill package for push/pull operations.

    Contains all artifacts needed to fully reproduce a skill on another device
    or share it through a Hub platform.
    """

    manifest: SkillManifest
    source_code: str = ""
    trajectory_skeleton: str = ""
    copilot_prior: str = ""
    readme: str = ""


# ─── Hub Backend Protocol ────────────────────────────────────────────────────


@runtime_checkable
class HubBackend(Protocol):
    """Protocol for Hub platform backends.

    Each backend implementation (ModelScope, HuggingFace, etc.) must satisfy
    this interface to enable skill push/pull and cloud collaboration.

    All methods are async to support non-blocking network I/O.
    """

    @property
    def hub_type(self) -> str:
        """Return the backend identifier (e.g. 'modelscope', 'huggingface')."""
        ...

    async def authenticate(self) -> UserInfo:
        """Authenticate with the Hub and return current user info.

        Raises:
            AuthenticationError: If credentials are invalid or missing.
        """
        ...

    async def push_skill(
        self,
        bundle: SkillBundle,
        repo_id: str,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> PushResult:
        """Push a skill bundle to the remote Hub.

        Args:
            bundle: Complete skill package to upload.
            repo_id: Target repository identifier (e.g. 'owner/skill-name').
            visibility: Repository visibility level.

        Returns:
            PushResult with the published version and URL.
        """
        ...

    async def pull_skill(
        self,
        repo_id: str,
        version: Optional[str] = None,
    ) -> SkillBundle:
        """Pull a skill bundle from the remote Hub.

        Args:
            repo_id: Repository identifier to pull from.
            version: Specific version to pull (None = latest).

        Returns:
            Complete SkillBundle ready for local installation.
        """
        ...

    async def list_remote_skills(
        self,
        owner: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[SkillSummary]:
        """List skills available on the remote Hub.

        Args:
            owner: Filter by owner/organization (None = all accessible).
            query: Free-text search query.

        Returns:
            List of matching skill summaries.
        """
        ...

    async def get_skill_versions(self, repo_id: str) -> List[VersionInfo]:
        """Get version history for a specific skill repository.

        Args:
            repo_id: Repository identifier.

        Returns:
            List of versions, most recent first.
        """
        ...

    async def delete_skill(self, repo_id: str) -> None:
        """Delete a skill repository from the remote Hub.

        Args:
            repo_id: Repository identifier to delete.

        Raises:
            PermissionError: If the user lacks delete permissions.
        """
        ...


class VersionConflictError(Exception):
    """Raised when push detects a version conflict with remote."""
    pass
