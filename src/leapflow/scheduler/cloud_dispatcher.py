"""Cloud dispatcher — orchestrates cloud task deployment lifecycle.

Workflow: package → create worker → inject secrets → deploy → monitor.
Provides a high-level interface over ComputeBackend and WorkerPackager.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from leapflow.scheduler.compute.protocol import ComputeBackend
from leapflow.scheduler.types import ArmedTask
from leapflow.scheduler.worker_packager import WorkerPackager

logger = logging.getLogger(__name__)


class CloudDispatcher:
    """Orchestrates cloud task deployment lifecycle.

    Coordinates the WorkerPackager and a ComputeBackend to deploy tasks
    as self-contained cloud workers with full secret injection.

    Usage:
        backend = ModelScopeStudioBackend()
        packager = WorkerPackager()
        dispatcher = CloudDispatcher(backend, packager)
        worker_id = await dispatcher.deploy(task)
    """

    def __init__(
        self,
        compute_backend: ComputeBackend,
        packager: Optional[WorkerPackager] = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            compute_backend: Backend responsible for creating/managing workers.
            packager: Worker packager (defaults to a new WorkerPackager instance).
        """
        self._backend = compute_backend
        self._packager = packager or WorkerPackager()

    @property
    def backend_type(self) -> str:
        """Return the underlying compute backend type."""
        return self._backend.backend_type

    async def deploy(
        self,
        task: ArmedTask,
        skill_source: str = "",
        context: Optional[dict] = None,
    ) -> str:
        """Deploy a task as a cloud worker.

        Performs the full lifecycle: package → create → inject secrets → deploy.

        Args:
            task: The armed task to deploy.
            skill_source: Optional skill source code to bundle.
            context: Optional context snapshot for the worker.

        Returns:
            The worker_id of the deployed instance.

        Raises:
            RuntimeError: If any step in the deployment pipeline fails.
        """
        worker_id = f"leapflow-task-{task.task_id[:8]}"

        # 1. Package the worker
        logger.info("Packaging worker for task %s...", task.task_id[:8])
        package_path = self._packager.package(task, skill_source, context)

        try:
            # 2. Create remote worker
            logger.info("Creating remote worker: %s", worker_id)
            await self._backend.create_worker(
                worker_id, package_path, visibility="private"
            )

            # 3. Inject secrets (full task config as env var)
            logger.info("Injecting secrets into worker: %s", worker_id)
            task_config = self._build_task_config(task)
            secrets: Dict[str, str] = {
                "LEAPFLOW_TASK_CONFIG": json.dumps(task_config, ensure_ascii=False),
            }
            await self._backend.inject_secrets(worker_id, secrets)

            # 4. Deploy
            logger.info("Deploying worker: %s", worker_id)
            await self._backend.deploy(worker_id)

            logger.info(
                "Successfully deployed task %s as worker %s",
                task.task_id[:8],
                worker_id,
            )
            return worker_id
        finally:
            # Cleanup temp package directory
            try:
                import shutil
                shutil.rmtree(package_path, ignore_errors=True)
            except Exception:
                logger.debug("Failed to cleanup package at %s", package_path)

    async def status(self, worker_id: str) -> str:
        """Get the current status of a deployed worker.

        Args:
            worker_id: The worker identifier.

        Returns:
            Status string: 'building' | 'running' | 'stopped' | 'failed' | 'unknown'.
        """
        return await self._backend.get_status(worker_id)

    async def logs(self, worker_id: str, tail: int = 50) -> List[str]:
        """Retrieve recent log lines from a deployed worker.

        Args:
            worker_id: The worker identifier.
            tail: Number of recent log lines to retrieve.

        Returns:
            List of log line strings.
        """
        return await self._backend.get_logs(worker_id, tail=tail)

    async def stop(self, worker_id: str) -> None:
        """Stop a running worker without destroying it.

        Args:
            worker_id: The worker identifier.
        """
        logger.info("Stopping worker: %s", worker_id)
        await self._backend.stop(worker_id)

    async def destroy(self, worker_id: str) -> None:
        """Stop and permanently delete a worker.

        Args:
            worker_id: The worker identifier.
        """
        logger.info("Destroying worker: %s", worker_id)
        await self._backend.destroy(worker_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task_config(task: ArmedTask) -> dict:
        """Build the LEAPFLOW_TASK_CONFIG payload from an ArmedTask.

        Ensures all fields are JSON-serializable plain dicts/primitives.
        """
        trigger_config = task.trigger_config
        if isinstance(trigger_config, str):
            try:
                trigger_config = json.loads(trigger_config)
            except (json.JSONDecodeError, TypeError):
                trigger_config = {"raw": trigger_config}

        parameters = task.parameters
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except (json.JSONDecodeError, TypeError):
                parameters = {"raw": parameters}

        # Derive check_interval from trigger configuration
        check_interval = 60  # default
        if task.trigger_type == "interval":
            interval_s = trigger_config.get("interval_seconds", 60)
            check_interval = max(30, min(int(interval_s), 3600))  # 30s ~ 1h range
        elif task.trigger_type == "cron":
            check_interval = 300  # 5min check for cron (platform handles exact timing)
        elif task.trigger_type in ("condition", "event"):
            check_interval = 60  # condition/event need frequent polling

        return {
            "task_id": task.task_id,
            "skill_name": task.skill_name,
            "trigger_config": trigger_config,
            "parameters": parameters,
            "max_runs": task.max_runs,
            "check_interval": check_interval,
        }
