"""HuggingFace Hub backend — placeholder for Phase 2.

Will be activated when huggingface-hub SDK integration is ready.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from leapflow.hub.protocol import (
    PushResult,
    SkillBundle,
    SkillSummary,
    UserInfo,
    VersionInfo,
    Visibility,
)

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED_MSG = (
    "HuggingFace backend is not yet implemented. "
    "This integration will be available in a future release."
)


class HuggingFaceBackend:
    """HubBackend implementation for HuggingFace Hub.

    Not yet implemented. Raises NotImplementedError for all operations.
    Will be activated when huggingface-hub SDK integration is ready.
    """

    hub_type = "huggingface"

    async def authenticate(self) -> UserInfo:
        """Not implemented."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def push_skill(
        self,
        bundle: SkillBundle,
        repo_id: str,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> PushResult:
        """Not implemented."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def pull_skill(
        self,
        repo_id: str,
        version: Optional[str] = None,
    ) -> SkillBundle:
        """Not implemented."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def list_remote_skills(
        self,
        owner: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[SkillSummary]:
        """Not implemented."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def get_skill_versions(self, repo_id: str) -> List[VersionInfo]:
        """Not implemented."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def delete_skill(self, repo_id: str) -> None:
        """Not implemented."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
