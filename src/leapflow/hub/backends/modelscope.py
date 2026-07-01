"""ModelScope Hub backend implementation.

Provides push/pull/search operations against ModelScope Hub (modelscope.cn).
Uses asyncio.to_thread to wrap synchronous SDK calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from leapflow.hub.protocol import (
    PushResult,
    SkillBundle,
    SkillSummary,
    UserInfo,
    VersionInfo,
    Visibility,
)
from leapflow.hub.serializer import SkillSerializer

logger = logging.getLogger(__name__)

_SDK_INSTALL_HINT = (
    "ModelScope Hub SDK not found. Install it with:\n"
    "  pip install modelscope-hub\n"
    "or:\n"
    "  uv pip install modelscope-hub"
)


class ModelScopeBackend:
    """HubBackend implementation for ModelScope (modelscope.cn / modelscope.ai).

    Wraps the modelscope_hub SDK with async interface and friendly error handling.
    """

    hub_type = "modelscope"

    def __init__(self):
        """Initialize ModelScope backend.

        Raises:
            ImportError: If modelscope_hub is not installed.
        """
        self._api: Any = None
        self._serializer = SkillSerializer()
        self._ensure_sdk()

    def _ensure_sdk(self) -> None:
        """Verify SDK availability and create API instance."""
        try:
            from modelscope_hub import HubApi  # type: ignore[import-untyped]

            self._api = HubApi()
        except ImportError:
            raise ImportError(_SDK_INSTALL_HINT) from None

    async def authenticate(self) -> UserInfo:
        """Authenticate with ModelScope and return user info."""
        try:
            info = await asyncio.to_thread(self._api.whoami)
            return UserInfo(
                username=info.get("Name", info.get("username", "")),
                email=info.get("Email", info.get("email", "")),
                avatar_url=info.get("Avatar", info.get("avatar_url", "")),
            )
        except Exception as e:
            raise RuntimeError(
                f"ModelScope authentication failed: {e}. "
                "Ensure you have logged in via `modelscope login` or set "
                "MODELSCOPE_API_TOKEN environment variable."
            ) from e

    async def push_skill(
        self,
        bundle: SkillBundle,
        repo_id: str,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> PushResult:
        """Push a skill bundle to ModelScope Hub.

        Creates the repository if it doesn't exist, then uploads all files.
        """
        # Ensure repository exists
        await self._ensure_repo(repo_id, visibility)

        # Write bundle to temporary directory
        files = self._serializer.bundle_to_files(bundle)

        with tempfile.TemporaryDirectory(prefix="leapflow_push_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for filename, content in files.items():
                file_path = tmp_path / filename
                file_path.write_text(content, encoding="utf-8")

            # Upload the folder
            version = bundle.manifest.version or "0.1.0"
            commit_message = f"Push skill {bundle.manifest.name} v{version}"

            await asyncio.to_thread(
                self._api.upload_folder,
                repo_id=repo_id,
                folder_path=str(tmp_path),
                repo_type="skill",
                commit_message=commit_message,
            )

        # Construct result URL
        base_url = os.environ.get("MODELSCOPE_HUB_URL", "https://modelscope.cn")
        url = f"{base_url}/skills/{repo_id}"

        logger.info("Pushed skill '%s' v%s to %s", bundle.manifest.name, version, url)

        return PushResult(
            repo_id=repo_id,
            version=version,
            url=url,
            hub_type=self.hub_type,
        )

    async def pull_skill(
        self,
        repo_id: str,
        version: Optional[str] = None,
    ) -> SkillBundle:
        """Pull a skill bundle from ModelScope Hub."""
        try:
            kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "repo_type": "skill",
            }
            if version:
                kwargs["revision"] = version

            local_dir = await asyncio.to_thread(
                self._api.download_repo,
                **kwargs,
            )

            # Read all files from downloaded directory
            local_path = Path(local_dir)
            files: dict[str, str] = {}
            for file_path in local_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    rel = file_path.relative_to(local_path)
                    files[str(rel)] = file_path.read_text(encoding="utf-8")

            return self._serializer.files_to_bundle(files)

        except Exception as e:
            raise RuntimeError(
                f"Failed to pull skill '{repo_id}' from ModelScope: {e}"
            ) from e

    async def list_remote_skills(
        self,
        owner: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[SkillSummary]:
        """List skills available on ModelScope Hub."""
        try:
            kwargs: dict[str, Any] = {"repo_type": "skill"}
            if owner:
                kwargs["owner"] = owner
            if query:
                kwargs["query"] = query

            repos = await asyncio.to_thread(self._api.list_repos, **kwargs)

            skills: List[SkillSummary] = []
            for repo in repos:
                skills.append(
                    SkillSummary(
                        repo_id=repo.get("id", repo.get("repo_id", "")),
                        name=repo.get("name", ""),
                        description=repo.get("description", ""),
                        version=repo.get("latest_version", ""),
                        downloads=repo.get("downloads", 0),
                        hub_type=self.hub_type,
                    )
                )
            return skills

        except Exception as e:
            logger.warning("Failed to list skills from ModelScope: %s", e)
            return []

    async def get_skill_versions(self, repo_id: str) -> List[VersionInfo]:
        """Get version history for a skill on ModelScope."""
        try:
            revisions = await asyncio.to_thread(
                self._api.list_repo_revisions,
                repo_id=repo_id,
                repo_type="skill",
            )

            versions: List[VersionInfo] = []
            for rev in revisions:
                versions.append(
                    VersionInfo(
                        version=rev.get("version", rev.get("revision", "")),
                        created_at=rev.get("created_at", ""),
                        commit_sha=rev.get("commit_sha", rev.get("sha", "")),
                    )
                )
            return versions

        except Exception as e:
            logger.warning("Failed to get versions for '%s': %s", repo_id, e)
            return []

    async def delete_skill(self, repo_id: str) -> None:
        """Delete a skill repository from ModelScope Hub."""
        try:
            await asyncio.to_thread(
                self._api.delete_repo,
                repo_id=repo_id,
                repo_type="skill",
            )
            logger.info("Deleted remote skill repository: %s", repo_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete skill '{repo_id}' from ModelScope: {e}"
            ) from e

    # ─── Private Helpers ─────────────────────────────────────────────────────

    async def _ensure_repo(self, repo_id: str, visibility: Visibility) -> None:
        """Create repository if it doesn't exist."""
        try:
            await asyncio.to_thread(
                self._api.create_repo,
                repo_id=repo_id,
                repo_type="skill",
                visibility=visibility.value,
                exist_ok=True,
            )
        except Exception as e:
            # Some SDK versions may not support exist_ok; try to handle gracefully
            error_str = str(e).lower()
            if "already exists" in error_str or "exist" in error_str:
                logger.debug("Repository %s already exists", repo_id)
            else:
                raise RuntimeError(
                    f"Failed to create repository '{repo_id}': {e}"
                ) from e
