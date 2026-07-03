"""ModelScope Studio compute backend implementation.

Each task is deployed as a private Docker-based Studio that runs a LeapFlow
worker loop. Secrets are injected as environment variables into the container.

Uses asyncio.to_thread to wrap synchronous ModelScope Hub SDK calls.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SDK_INSTALL_HINT = (
    "ModelScope Hub SDK not found. Install it with:\n"
    "  pip install modelscope-hub\n"
    "or:\n"
    "  uv pip install modelscope-hub"
)

# Status mapping from ModelScope Studio states to our canonical states.
_STATUS_MAP: Dict[str, str] = {
    "building": "building",
    "running": "running",
    "stopped": "stopped",
    "failed": "failed",
    "pending": "building",
    "deploying": "building",
    "error": "failed",
}


class ModelScopeStudioBackend:
    """ComputeBackend implementation using ModelScope Hub Studio (Docker mode).

    Each task is deployed as a private Docker-based Studio that runs
    a LeapFlow worker loop. Secrets are injected as environment variables.

    Conforms to the ComputeBackend protocol defined in
    ``leapflow.scheduler.compute.protocol``.
    """

    backend_type = "modelscope_studio"

    def __init__(self) -> None:
        """Initialize the backend with lazy SDK import.

        Raises:
            ImportError: If the modelscope_hub SDK is not installed.
        """
        self._api: Any = None
        self._ensure_sdk()

    def _ensure_sdk(self) -> None:
        """Verify SDK availability and create API instance."""
        try:
            from modelscope_hub import HubApi  # type: ignore[import-untyped]

            self._api = HubApi()
        except ImportError:
            raise ImportError(_SDK_INSTALL_HINT) from None

    # ------------------------------------------------------------------
    # ComputeBackend interface
    # ------------------------------------------------------------------

    async def create_worker(
        self,
        worker_id: str,
        package_path: Path,
        *,
        visibility: str = "private",
    ) -> str:
        """Create a Studio repo with Docker SDK type and upload the package.

        Args:
            worker_id: Repo identifier for the Studio.
            package_path: Path to the worker package directory.
            visibility: 'private' or 'public'.

        Returns:
            The worker_id as confirmation.
        """
        # Create the Studio repository
        await asyncio.to_thread(
            self._api.create_repo,
            repo_id=worker_id,
            repo_type="studio",
            sdk_type="docker",
            visibility=visibility,
            exist_ok=True,
        )
        logger.info("Created Studio repo: %s (visibility=%s)", worker_id, visibility)

        # Upload the deployable package
        await asyncio.to_thread(
            self._api.upload_folder,
            repo_id=worker_id,
            repo_type="studio",
            folder_path=str(package_path),
            commit_message=f"Deploy worker package for {worker_id}",
        )
        logger.info("Uploaded worker package to Studio: %s", worker_id)

        return worker_id

    async def inject_secrets(
        self,
        worker_id: str,
        secrets: Dict[str, str],
    ) -> None:
        """Add secrets to the Studio (available as env vars in the container).

        Args:
            worker_id: Target Studio repo identifier.
            secrets: Key-value pairs to inject.
        """
        for key, value in secrets.items():
            await asyncio.to_thread(
                self._api.add_secret,
                repo_id=worker_id,
                repo_type="studio",
                key=key,
                value=value,
            )
        logger.info(
            "Injected %d secret(s) into Studio: %s", len(secrets), worker_id
        )

    async def deploy(self, worker_id: str) -> None:
        """Trigger Studio deployment.

        Args:
            worker_id: Target Studio repo identifier.
        """
        await asyncio.to_thread(
            self._api.deploy_repo,
            repo_id=worker_id,
            repo_type="studio",
        )
        logger.info("Deployed Studio: %s", worker_id)

    async def stop(self, worker_id: str) -> None:
        """Stop a running Studio.

        Args:
            worker_id: Target Studio repo identifier.
        """
        await asyncio.to_thread(
            self._api.stop_repo,
            repo_id=worker_id,
            repo_type="studio",
        )
        logger.info("Stopped Studio: %s", worker_id)

    async def get_status(self, worker_id: str) -> str:
        """Get Studio status.

        Args:
            worker_id: Target Studio repo identifier.

        Returns:
            Canonical status: 'building' | 'running' | 'stopped' | 'failed' | 'unknown'.
        """
        try:
            info = await asyncio.to_thread(
                self._api.get_repo,
                repo_id=worker_id,
                repo_type="studio",
            )
            raw_status = str(info.get("status", "unknown")).lower()
            return _STATUS_MAP.get(raw_status, "unknown")
        except Exception as e:
            logger.warning("Failed to get status for '%s': %s", worker_id, e)
            return "unknown"

    async def get_logs(self, worker_id: str, *, tail: int = 50) -> List[str]:
        """Get Studio run logs.

        Args:
            worker_id: Target Studio repo identifier.
            tail: Number of recent log lines to retrieve.

        Returns:
            List of log line strings.
        """
        try:
            logs = await asyncio.to_thread(
                self._api.get_repo_logs,
                repo_id=worker_id,
                repo_type="studio",
                log_type="run",
                page_size=tail,
            )
            # SDK may return a list of dicts or list of strings
            if isinstance(logs, list):
                return [
                    str(entry.get("message", entry)) if isinstance(entry, dict) else str(entry)
                    for entry in logs[-tail:]
                ]
            return []
        except Exception as e:
            logger.warning("Failed to get logs for '%s': %s", worker_id, e)
            return []

    async def destroy(self, worker_id: str) -> None:
        """Stop and permanently delete the Studio.

        Args:
            worker_id: Target Studio repo identifier.
        """
        # Best-effort stop before deletion
        try:
            await self.stop(worker_id)
        except Exception:
            pass  # Already stopped or non-existent — proceed to delete

        try:
            await asyncio.to_thread(
                self._api.delete_repo,
                repo_id=worker_id,
                repo_type="studio",
            )
            logger.info("Destroyed Studio: %s", worker_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to destroy Studio '{worker_id}': {e}"
            ) from e
