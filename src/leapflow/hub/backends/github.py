"""GitHub Hub backend — REST API-based skill push/pull/search.

Implements HubBackend Protocol using GitHub REST API (Contents + Repos).
No external SDK dependency — uses httpx (already available via OpenAI).

Authentication: GITHUB_TOKEN env var or `gh auth token` subprocess fallback.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

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

_TOKEN_MISSING_MSG = (
    "GitHub token not found. Set GITHUB_TOKEN environment variable "
    "or install the 'gh' CLI and run 'gh auth login'."
)


def _resolve_token(explicit: str = "") -> str:
    """Resolve GitHub token from explicit value, env var, or gh CLI."""
    if explicit:
        return explicit
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    # Fallback: try `gh auth token`
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


class GitHubBackend:
    """HubBackend implementation for GitHub using REST API.

    Push: Creates repo (if needed) + uploads files via Contents API.
    Pull: Downloads repo contents + reconstructs SkillBundle.
    Search: Lists user repos with configurable prefix filter.
    """

    hub_type = "github"

    def __init__(
        self,
        token: str = "",
        api_base: str = "https://api.github.com",
    ) -> None:
        self._token = _resolve_token(token)
        self._base = api_base.rstrip("/")
        self._serializer = SkillSerializer()
        self._client: Any = None  # Lazy httpx.AsyncClient

    @property
    def _http(self) -> Any:
        """Lazy-initialized async HTTP client."""
        if self._client is None:
            import httpx
            headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
            if self._token:
                headers["Authorization"] = f"token {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self._base,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    # ── HubBackend Protocol ───────────────────────────────────────────────

    async def authenticate(self) -> UserInfo:
        """Authenticate and return GitHub user info."""
        if not self._token:
            raise RuntimeError(_TOKEN_MISSING_MSG)
        resp = await self._http.get("/user")
        resp.raise_for_status()
        data = resp.json()
        return UserInfo(
            username=data.get("login", ""),
            email=data.get("email", ""),
            avatar_url=data.get("avatar_url", ""),
        )

    async def push_skill(
        self,
        bundle: SkillBundle,
        repo_id: str,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> PushResult:
        """Push skill bundle to a GitHub repository.

        Creates the repo if it doesn't exist, then uploads all files
        via the Contents API (one commit per file).
        """
        if not self._token:
            raise RuntimeError(_TOKEN_MISSING_MSG)

        # Ensure repository exists
        await self._ensure_repo(repo_id, visibility)

        # Serialize bundle to files
        files = self._serializer.bundle_to_files(bundle)
        version = bundle.manifest.version or "0.1.0"

        # Upload each file via Contents API
        for filename, content in files.items():
            await self._put_file(
                repo_id, filename, content,
                message=f"Push {bundle.manifest.name} v{version}: {filename}",
            )

        url = f"https://github.com/{repo_id}"
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
        """Pull skill bundle from a GitHub repository."""
        ref = version or "main"
        # List repo contents at root
        resp = await self._http.get(
            f"/repos/{repo_id}/contents", params={"ref": ref}
        )
        if resp.status_code == 404:
            raise FileNotFoundError(f"Repository not found: {repo_id}")
        resp.raise_for_status()

        items = resp.json()
        if not isinstance(items, list):
            raise ValueError(f"Unexpected response format for {repo_id}")

        # Download each file
        files: Dict[str, str] = {}
        for item in items:
            if item.get("type") != "file":
                continue
            name = item.get("name", "")
            download_url = item.get("download_url", "")
            if not download_url:
                continue
            file_resp = await self._http.get(download_url)
            if file_resp.status_code == 200:
                files[name] = file_resp.text
            else:
                logger.warning(
                    "Failed to download file '%s' from %s (status: %d)",
                    name, download_url, file_resp.status_code,
                )

        return self._serializer.files_to_bundle(files)

    async def list_remote_skills(
        self,
        owner: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[SkillSummary]:
        """List skill repositories on GitHub.

        Filters by owner's repos with prefix matching. If query is provided,
        uses GitHub search API for broader discovery.
        """
        if query:
            return await self._search_repos(query, owner)
        if owner:
            return await self._list_user_repos(owner)
        # No owner and no query — list authenticated user's repos
        if self._token:
            resp = await self._http.get(
                "/user/repos", params={"per_page": 100, "sort": "updated"}
            )
            if resp.status_code == 200:
                return self._repos_to_summaries(resp.json())
        return []

    async def get_skill_versions(self, repo_id: str) -> List[VersionInfo]:
        """Get version history via git tags."""
        resp = await self._http.get(f"/repos/{repo_id}/tags", params={"per_page": 50})
        if resp.status_code != 200:
            return []
        tags = resp.json()
        return [
            VersionInfo(
                version=tag.get("name", ""),
                commit_sha=tag.get("commit", {}).get("sha", ""),
            )
            for tag in tags
        ]

    async def delete_skill(self, repo_id: str) -> None:
        """Delete a GitHub repository."""
        if not self._token:
            raise RuntimeError(_TOKEN_MISSING_MSG)
        resp = await self._http.delete(f"/repos/{repo_id}")
        if resp.status_code == 404:
            raise FileNotFoundError(f"Repository not found: {repo_id}")
        resp.raise_for_status()
        logger.info("Deleted GitHub repository: %s", repo_id)

    # ── Private Helpers ───────────────────────────────────────────────────

    async def _ensure_repo(self, repo_id: str, visibility: Visibility) -> None:
        """Create repository if it doesn't exist.

        Handles both personal repos (POST /user/repos) and org repos
        (POST /orgs/{org}/repos) based on owner identity.
        """
        parts = repo_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repo_id format: '{repo_id}' (expected 'owner/name')")
        owner, name = parts

        # Check if repo exists
        resp = await self._http.get(f"/repos/{repo_id}")
        if resp.status_code == 200:
            return  # Already exists

        # Determine if owner is the authenticated user or an org
        private = visibility != Visibility.PUBLIC
        user_resp = await self._http.get("/user")
        is_personal = (
            user_resp.status_code == 200
            and user_resp.json().get("login", "") == owner
        )

        if is_personal:
            create_resp = await self._http.post("/user/repos", json={
                "name": name,
                "private": private,
                "auto_init": True,
                "description": f"LeapFlow skill: {name}",
            })
        else:
            # Organization repo
            create_resp = await self._http.post(f"/orgs/{owner}/repos", json={
                "name": name,
                "private": private,
                "auto_init": True,
                "description": f"LeapFlow skill: {name}",
            })

        if create_resp.status_code not in (201, 422):
            create_resp.raise_for_status()

    async def _put_file(
        self, repo_id: str, path: str, content: str, message: str
    ) -> None:
        """Create or update a file via Contents API."""
        url = f"/repos/{repo_id}/contents/{path}"
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # Check if file exists (need sha for update)
        existing = await self._http.get(url)
        payload: Dict[str, Any] = {
            "message": message,
            "content": encoded,
        }
        if existing.status_code == 200:
            sha = existing.json().get("sha", "")
            if sha:
                payload["sha"] = sha

        resp = await self._http.put(url, json=payload)
        resp.raise_for_status()

    async def _search_repos(
        self, query: str, owner: Optional[str] = None
    ) -> List[SkillSummary]:
        """Search GitHub repositories matching query."""
        q = f"{query} in:name,description"
        if owner:
            q += f" user:{owner}"
        resp = await self._http.get(
            "/search/repositories",
            params={"q": q, "per_page": 20, "sort": "updated"},
        )
        if resp.status_code != 200:
            return []
        items = resp.json().get("items", [])
        return self._repos_to_summaries(items)

    async def _list_user_repos(self, owner: str) -> List[SkillSummary]:
        """List repos for a specific user/org."""
        resp = await self._http.get(
            f"/users/{owner}/repos",
            params={"per_page": 100, "sort": "updated"},
        )
        if resp.status_code != 200:
            return []
        return self._repos_to_summaries(resp.json())

    @staticmethod
    def _repos_to_summaries(repos: List[Dict[str, Any]]) -> List[SkillSummary]:
        """Convert GitHub repo objects to SkillSummary list."""
        summaries: List[SkillSummary] = []
        for repo in repos:
            if not isinstance(repo, dict):
                continue
            full_name = repo.get("full_name", "")
            summaries.append(SkillSummary(
                repo_id=full_name,
                name=repo.get("name", ""),
                description=repo.get("description", "") or "",
                version="",  # GitHub doesn't have a native version field
                downloads=repo.get("stargazers_count", 0),
                hub_type="github",
            ))
        return summaries
