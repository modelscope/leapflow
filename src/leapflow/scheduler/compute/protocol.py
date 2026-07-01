"""Compute backend protocol for cloud task execution.

Defines the abstract interface that all compute backends must implement.
Backends manage the full lifecycle of remote workers: create → deploy → monitor → destroy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """Abstract compute backend — ModelScope Studio, HF Spaces, local Docker.

    Implementations wrap platform-specific SDKs to provide a uniform
    interface for deploying and managing cloud workers.
    """

    @property
    def backend_type(self) -> str:
        """Identifier string for this backend (e.g. 'modelscope_studio')."""
        ...

    async def create_worker(
        self,
        worker_id: str,
        package_path: Path,
        *,
        visibility: str = "private",
    ) -> str:
        """Create a remote worker instance.

        Args:
            worker_id: Unique identifier for the worker.
            package_path: Path to the deployable package directory.
            visibility: 'private' or 'public'.

        Returns:
            Instance identifier or URL.
        """
        ...

    async def inject_secrets(
        self,
        worker_id: str,
        secrets: Dict[str, str],
    ) -> None:
        """Inject environment variables/secrets into the worker.

        Args:
            worker_id: Target worker identifier.
            secrets: Key-value pairs to inject as environment variables.
        """
        ...

    async def deploy(self, worker_id: str) -> None:
        """Trigger deployment/start of the worker.

        Args:
            worker_id: Target worker identifier.
        """
        ...

    async def stop(self, worker_id: str) -> None:
        """Stop a running worker without destroying it.

        Args:
            worker_id: Target worker identifier.
        """
        ...

    async def get_status(self, worker_id: str) -> str:
        """Get worker status.

        Args:
            worker_id: Target worker identifier.

        Returns:
            One of: 'building' | 'running' | 'stopped' | 'failed' | 'unknown'.
        """
        ...

    async def get_logs(self, worker_id: str, *, tail: int = 50) -> List[str]:
        """Retrieve recent log lines from the worker.

        Args:
            worker_id: Target worker identifier.
            tail: Number of recent log lines to retrieve.

        Returns:
            List of log line strings.
        """
        ...

    async def destroy(self, worker_id: str) -> None:
        """Stop and permanently delete the worker and its resources.

        Args:
            worker_id: Target worker identifier.
        """
        ...
