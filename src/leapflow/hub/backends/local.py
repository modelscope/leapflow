"""Local filesystem backend — for testing and offline use.

Stores skill bundles as directories on the local filesystem, enabling
hub operations without network connectivity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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


def _semver_key(version: str) -> tuple:
    """Parse version string to comparable tuple for correct semver sorting.

    Handles formats: '0.1.0', 'v1.2.3', '1.0'.
    Returns (0, 0, 0) on parse failure for safe fallback.
    """
    cleaned = version.strip().lstrip("v")
    try:
        return tuple(int(x) for x in cleaned.split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


class LocalBackend:
    """HubBackend using local filesystem as storage.

    Each skill is stored as a subdirectory containing manifest + artifacts.
    Suitable for testing and offline scenarios.
    """

    hub_type = "local"

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize local backend.

        Args:
            base_dir: Root directory for local hub storage.
                      Defaults to ~/.leapflow/hub/local/
        """
        if base_dir is None:
            base_dir = Path.home() / ".leapflow" / "hub" / "local"
        self._base_dir = Path(base_dir)
        self._serializer = SkillSerializer()
        logger.debug("LocalBackend initialized at %s", self._base_dir)

    @property
    def base_dir(self) -> Path:
        """Return the base directory for local storage."""
        return self._base_dir

    def _safe_repo_path(self, repo_id: str) -> Path:
        """Validate repo_id and resolve to safe path under base_dir.

        Prevents path traversal attacks via '..' segments or absolute paths.
        """
        from pathlib import PurePosixPath

        parts = PurePosixPath(repo_id).parts
        if not parts:
            raise ValueError(f"Empty repo_id")
        if any(p == ".." for p in parts):
            raise ValueError(f"Path traversal detected in repo_id: '{repo_id}'")
        if PurePosixPath(repo_id).is_absolute():
            raise ValueError(f"Absolute path not allowed in repo_id: '{repo_id}'")

        resolved = (self._base_dir / repo_id).resolve()
        if not str(resolved).startswith(str(self._base_dir.resolve())):
            raise ValueError(f"repo_id escapes base directory: '{repo_id}'")
        return resolved

    async def authenticate(self) -> UserInfo:
        """Return local user info (no actual authentication needed)."""
        username = os.environ.get("USER", os.environ.get("USERNAME", "local"))
        return UserInfo(username=username, email=f"{username}@local")

    async def push_skill(
        self,
        bundle: SkillBundle,
        repo_id: str,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> PushResult:
        """Write skill bundle to local filesystem."""
        version = bundle.manifest.version or "0.1.0"
        repo_path = self._safe_repo_path(repo_id) / version
        repo_path.mkdir(parents=True, exist_ok=True)

        # Write files
        files = self._serializer.bundle_to_files(bundle)
        for filename, content in files.items():
            file_path = repo_path / filename
            file_path.write_text(content, encoding="utf-8")

        logger.info("Pushed skill '%s' v%s to %s", bundle.manifest.name, version, repo_path)

        return PushResult(
            repo_id=repo_id,
            version=version,
            url=str(repo_path),
            hub_type=self.hub_type,
        )

    async def pull_skill(
        self,
        repo_id: str,
        version: Optional[str] = None,
    ) -> SkillBundle:
        """Read skill bundle from local filesystem."""
        repo_base = self._safe_repo_path(repo_id)

        if not repo_base.exists():
            raise FileNotFoundError(f"Skill repository not found: {repo_id}")

        # Resolve version
        if version is None:
            # Get latest version (semantic version sort, not lexicographic)
            version_dirs = [d.name for d in repo_base.iterdir() if d.is_dir()]
            if not version_dirs:
                raise FileNotFoundError(f"No versions found for: {repo_id}")
            versions = sorted(version_dirs, key=_semver_key, reverse=True)
            version = versions[0]

        version_path = repo_base / version
        if not version_path.exists():
            raise FileNotFoundError(f"Version {version} not found for: {repo_id}")

        # Read all files
        files: dict[str, str] = {}
        for file_path in version_path.iterdir():
            if file_path.is_file():
                files[file_path.name] = file_path.read_text(encoding="utf-8")

        return self._serializer.files_to_bundle(files)

    async def list_remote_skills(
        self,
        owner: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[SkillSummary]:
        """List locally stored skills."""
        skills: List[SkillSummary] = []

        if not self._base_dir.exists():
            return skills

        for repo_dir in self._base_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            repo_id = repo_dir.name

            # Filter by owner if specified
            if owner and not repo_id.startswith(f"{owner}/"):
                continue

            # Get latest version info (semantic version sort)
            versions = sorted(
                [d.name for d in repo_dir.iterdir() if d.is_dir()],
                key=_semver_key, reverse=True,
            )
            latest_version = versions[0] if versions else ""

            # Try to read manifest for description
            description = ""
            if latest_version:
                for manifest_name in ("manifest.yaml", "manifest.yml", "manifest.json"):
                    manifest_path = repo_dir / latest_version / manifest_name
                    if manifest_path.exists():
                        try:
                            text = manifest_path.read_text(encoding="utf-8")
                            if manifest_name.endswith(".json"):
                                data = json.loads(text)
                            else:
                                # Try yaml import, fallback to basic key: value parsing
                                try:
                                    import yaml
                                    data = yaml.safe_load(text) or {}
                                except ImportError:
                                    # Basic YAML-like parsing for simple key: value files
                                    data = {}
                                    for line in text.splitlines():
                                        if ":" in line and not line.startswith("#"):
                                            k, _, v = line.partition(":")
                                            data[k.strip()] = v.strip().strip('"').strip("'")
                            description = data.get("description", "")
                        except Exception:
                            pass
                        break

            # Filter by query
            if query and query.lower() not in repo_id.lower() and query.lower() not in description.lower():
                continue

            skills.append(
                SkillSummary(
                    repo_id=repo_id,
                    name=repo_id.split("/")[-1] if "/" in repo_id else repo_id,
                    description=description,
                    version=latest_version,
                    hub_type=self.hub_type,
                )
            )

        return skills

    async def get_skill_versions(self, repo_id: str) -> List[VersionInfo]:
        """List versions of a locally stored skill."""
        repo_base = self._base_dir / repo_id

        if not repo_base.exists():
            return []

        versions: List[VersionInfo] = []
        for version_dir in sorted(repo_base.iterdir(), reverse=True):
            if version_dir.is_dir():
                # Use modification time as created_at
                mtime = datetime.fromtimestamp(
                    version_dir.stat().st_mtime, tz=timezone.utc
                )
                versions.append(
                    VersionInfo(
                        version=version_dir.name,
                        created_at=mtime.isoformat(),
                    )
                )

        return versions

    async def delete_skill(self, repo_id: str) -> None:
        """Delete a skill from local storage."""
        import shutil

        repo_base = self._safe_repo_path(repo_id)

        if not repo_base.exists():
            raise FileNotFoundError(f"Skill repository not found: {repo_id}")

        shutil.rmtree(repo_base)
        logger.info("Deleted local skill repository: %s", repo_id)
