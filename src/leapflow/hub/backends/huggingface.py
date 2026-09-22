# Copyright (c) Alibaba, Inc. and its affiliates.
"""HuggingFace Hub backend — push/pull/search skills via huggingface_hub SDK.

Implements HubBackend Protocol using the ``huggingface_hub`` library (HfApi).
Stores skills as HuggingFace datasets (repo_type="dataset") since skill
bundles are not ML models.

Authentication: ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN`` env var, or a prior
``huggingface-cli login`` session.

All synchronous HfApi calls are wrapped with ``asyncio.to_thread`` to avoid
blocking the event loop (same pattern as the ModelScope backend).
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
    "HuggingFace Hub SDK not found. Install it with:\n"
    "  pip install huggingface-hub\n"
    "or:\n"
    "  uv pip install huggingface-hub"
)

_TOKEN_MISSING_MSG = (
    "HuggingFace token not found. Set HF_TOKEN or HUGGINGFACE_TOKEN "
    "environment variable, or run 'huggingface-cli login'."
)

# Map LeapFlow visibility levels to HuggingFace visibility.
_VISIBILITY_MAP = {
    Visibility.PRIVATE: "private",
    Visibility.INTERNAL: "private",   # HF has no "internal"; fall back to private
    Visibility.PUBLIC: "public",
}

# Default timeout for HfApi network calls (seconds).
_DEFAULT_TIMEOUT = 60


def _resolve_token() -> str:
    """Resolve HuggingFace token from environment variables.

    Checks ``HF_TOKEN`` first (the canonical variable used by ``huggingface_hub``
    itself), then falls back to ``HUGGINGFACE_TOKEN``.
    """
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    )


class HuggingFaceBackend:
    """HubBackend implementation for HuggingFace Hub (huggingface.co).

    Wraps the ``huggingface_hub.HfApi`` SDK with an async interface and
    friendly error handling.  Skills are stored as HuggingFace **datasets**
    (``repo_type="dataset"``) rather than models, since skill bundles are
    source artefacts, not trained weights.
    """

    hub_type = "huggingface"

    def __init__(self, token: str = "") -> None:
        """Initialize HuggingFace backend.

        Args:
            token: Explicit HF token.  If empty, resolved from env or
                   cached CLI login session.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        self._token: str = token or _resolve_token()
        self._api: Any = None
        self._serializer = SkillSerializer()
        self._ensure_sdk()

    # ── SDK bootstrap ────────────────────────────────────────────────────

    def _ensure_sdk(self) -> None:
        """Verify SDK availability and create API instance."""
        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]

            # Pass token only when explicitly available; HfApi also reads
            # HF_TOKEN / cached login automatically.
            kwargs: dict[str, Any] = {}
            if self._token:
                kwargs["token"] = self._token
            self._api = HfApi(**kwargs)
        except ImportError:
            raise ImportError(_SDK_INSTALL_HINT) from None

    # ── HubBackend Protocol ──────────────────────────────────────────────

    async def authenticate(self) -> UserInfo:
        """Authenticate with HuggingFace and return current user info."""
        try:
            info = await asyncio.to_thread(self._api.whoami)
            return UserInfo(
                username=info.get("name", info.get("fullname", "")),
                email=info.get("email", ""),
                avatar_url=info.get("avatarUrl", ""),
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "401" in error_msg:
                raise RuntimeError(
                    f"HuggingFace authentication failed: {e}. {_TOKEN_MISSING_MSG}"
                ) from e
            raise RuntimeError(
                f"HuggingFace authentication failed: {e}. "
                "Ensure you have logged in via 'huggingface-cli login' or "
                "set the HF_TOKEN environment variable."
            ) from e

    async def push_skill(
        self,
        bundle: SkillBundle,
        repo_id: str,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> PushResult:
        """Push a skill bundle to HuggingFace Hub.

        Creates the dataset repository if it doesn't exist, then uploads
        all bundle files in a single commit.
        """
        # Ensure repository exists
        await self._ensure_repo(repo_id, visibility)

        # Serialize bundle to files
        files = self._serializer.bundle_to_files(bundle)
        version = bundle.manifest.version or "0.1.0"
        commit_message = f"Push skill {bundle.manifest.name} v{version}"

        with tempfile.TemporaryDirectory(prefix="leapflow_hf_push_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for filename, content in files.items():
                file_path = tmp_path / filename
                file_path.write_text(content, encoding="utf-8")

            # Upload the entire folder as a single commit
            await asyncio.to_thread(
                self._api.upload_folder,
                repo_id=repo_id,
                folder_path=str(tmp_path),
                repo_type="dataset",
                commit_message=commit_message,
            )

        # Construct result URL
        url = f"https://huggingface.co/datasets/{repo_id}"

        logger.info(
            "Pushed skill '%s' v%s to %s", bundle.manifest.name, version, url
        )

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
        """Pull a skill bundle from HuggingFace Hub."""
        try:
            kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "repo_type": "dataset",
            }
            if version:
                kwargs["revision"] = version

            local_dir: str = await asyncio.to_thread(
                self._api.snapshot_download,
                **kwargs,
            )

            # Read all files in a thread to avoid blocking the event loop
            def _read_files(local_path: Path) -> dict[str, str]:
                files: dict[str, str] = {}
                for file_path in local_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith("."):
                        rel = file_path.relative_to(local_path)
                        files[str(rel)] = file_path.read_text(encoding="utf-8")
                return files

            local_path = Path(local_dir)
            files = await asyncio.to_thread(_read_files, local_path)

            return self._serializer.files_to_bundle(files)

        except Exception as e:
            raise RuntimeError(
                f"Failed to pull skill '{repo_id}' from HuggingFace: {e}"
            ) from e

    async def list_remote_skills(
        self,
        owner: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[SkillSummary]:
        """List skills available on HuggingFace Hub.

        Searches datasets with an optional author filter and text query.
        """
        try:
            kwargs: dict[str, Any] = {}
            if owner:
                kwargs["author"] = owner
            if query:
                kwargs["search"] = query

            datasets = await asyncio.to_thread(
                self._api.list_datasets, **kwargs
            )

            skills: List[SkillSummary] = []
            for ds in datasets:
                repo_id = getattr(ds, "id", "")
                name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                skills.append(
                    SkillSummary(
                        repo_id=repo_id,
                        name=name,
                        description=getattr(ds, "description", "") or "",
                        version=getattr(ds, "sha", ""),
                        downloads=getattr(ds, "downloads", 0) or 0,
                        hub_type=self.hub_type,
                    )
                )
            return skills

        except Exception as e:
            logger.warning("Failed to list skills from HuggingFace: %s", e)
            return []

    async def get_skill_versions(self, repo_id: str) -> List[VersionInfo]:
        """Get version history for a skill on HuggingFace.

        Uses commit history on the dataset repository as version records.
        """
        try:
            commits = await asyncio.to_thread(
                self._api.list_repo_commits,
                repo_id=repo_id,
                repo_type="dataset",
            )

            versions: List[VersionInfo] = []
            for commit in commits:
                versions.append(
                    VersionInfo(
                        version=getattr(commit, "title", "") or "",
                        created_at=str(getattr(commit, "created_at", "") or ""),
                        commit_sha=getattr(commit, "commit_id", "") or "",
                    )
                )
            return versions

        except Exception as e:
            logger.warning(
                "Failed to get versions for '%s' on HuggingFace: %s",
                repo_id,
                e,
            )
            return []

    async def delete_skill(self, repo_id: str) -> None:
        """Delete a skill dataset repository from HuggingFace Hub."""
        try:
            await asyncio.to_thread(
                self._api.delete_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info("Deleted HuggingFace dataset repository: %s", repo_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete skill '{repo_id}' from HuggingFace: {e}"
            ) from e

    # ─── Private Helpers ─────────────────────────────────────────────────

    async def _ensure_repo(self, repo_id: str, visibility: Visibility) -> None:
        """Create dataset repository if it doesn't exist."""
        hf_visibility = _VISIBILITY_MAP.get(visibility, "private")
        try:
            await asyncio.to_thread(
                self._api.create_repo,
                repo_id=repo_id,
                repo_type="dataset",
                private=(hf_visibility == "private"),
                exist_ok=True,
            )
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "exist" in error_str:
                logger.debug("HuggingFace dataset %s already exists", repo_id)
            else:
                raise RuntimeError(
                    f"Failed to create HuggingFace dataset '{repo_id}': {e}"
                ) from e
