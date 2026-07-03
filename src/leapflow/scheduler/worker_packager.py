"""Worker packager — generates self-contained Docker packages for cloud deployment.

Produces a temporary directory containing all files needed to run a LeapFlow
worker in a Docker container on any supported compute backend.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from leapflow.scheduler.types import ArmedTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "worker.py"]
"""

_REQUIREMENTS_TEMPLATE = """\
# Minimal runtime dependencies for LeapFlow cloud worker
requests>=2.28.0
"""

_WORKER_PY_TEMPLATE = '''\
"""LeapFlow Cloud Worker — long-horizon task executor."""
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("leapflow-worker")


def main():
    """Main worker loop: parse config, evaluate triggers, execute skill."""
    config = json.loads(os.environ.get("LEAPFLOW_TASK_CONFIG", "{}"))
    if not config:
        logger.error("No LEAPFLOW_TASK_CONFIG found in environment")
        sys.exit(1)

    task_id = config.get("task_id", "unknown")
    skill_name = config.get("skill_name", "")
    trigger_config = config.get("trigger_config", {})
    parameters = config.get("parameters", {})
    max_runs = config.get("max_runs", -1)
    check_interval = config.get("check_interval", 60)

    logger.info("Worker started: task=%s skill=%s", task_id[:8], skill_name)

    run_count = 0
    while True:
        # Heartbeat
        logger.info("Heartbeat: task=%s runs=%d", task_id[:8], run_count)

        # Check trigger condition
        should_run = _check_trigger(trigger_config)

        if should_run:
            logger.info("Trigger fired: executing skill \'%s\'", skill_name)
            try:
                result = _execute_skill(skill_name, parameters)
                run_count += 1
                logger.info(
                    "Execution complete: ok=%s runs=%d",
                    result.get("ok"),
                    run_count,
                )
            except Exception as e:
                logger.error("Execution failed: %s", e)

            if max_runs > 0 and run_count >= max_runs:
                logger.info("Max runs reached (%d). Shutting down.", max_runs)
                break

        time.sleep(check_interval)


def _check_trigger(config):
    """Simplified trigger evaluation for cloud worker.

    In cloud mode, the worker itself evaluates timing. For interval/cron
    triggers the scheduler has already decided to deploy, so we fire on
    each check cycle.
    """
    # Future: implement cron expression evaluation here
    _ = config.get("trigger_type", "interval")
    return True


def _execute_skill(skill_name, parameters):
    """Execute the target skill.

    In the full implementation this would import and invoke the skill
    source code bundled into the package. For now, logs execution.
    """
    logger.info("Skill \'%s\' executed with params: %s", skill_name, parameters)
    return {"ok": True, "output": f"Executed {skill_name}"}


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# WorkerPackager
# ---------------------------------------------------------------------------


class WorkerPackager:
    """Generate self-contained Docker worker packages for cloud deployment.

    Produces a temporary directory containing:
    - Dockerfile (Python base + minimal deps)
    - worker.py (standard task execution loop)
    - requirements.txt (leapflow minimal runtime)
    - task_meta.json (non-sensitive task metadata only)

    Sensitive data (full task config) is injected as secrets at deploy time,
    not baked into the package.
    """

    def package(
        self,
        task: ArmedTask,
        skill_source: str = "",
        context_snapshot: Optional[dict] = None,
    ) -> Path:
        """Build a deployable worker package.

        Args:
            task: The armed task to deploy.
            skill_source: Optional skill source code to include.
            context_snapshot: Optional context data for the worker.

        Returns:
            Path to the temporary directory containing the package files.
        """
        tmp_dir = tempfile.mkdtemp(prefix=f"leapflow_worker_{task.task_id[:8]}_")
        pkg_path = Path(tmp_dir)

        # 1. Dockerfile
        (pkg_path / "Dockerfile").write_text(_DOCKERFILE_TEMPLATE, encoding="utf-8")

        # 2. worker.py
        (pkg_path / "worker.py").write_text(_WORKER_PY_TEMPLATE, encoding="utf-8")

        # 3. requirements.txt
        (pkg_path / "requirements.txt").write_text(
            _REQUIREMENTS_TEMPLATE, encoding="utf-8"
        )

        # 4. Task metadata (non-sensitive — no secrets here)
        meta = {
            "task_id": task.task_id,
            "skill_name": task.skill_name,
            "trigger_type": task.trigger_type,
            "execution_tier": task.execution_tier,
            "created_at": task.created_at,
        }
        if context_snapshot:
            meta["context_snapshot"] = context_snapshot

        (pkg_path / "task_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 5. Optional skill source
        if skill_source:
            (pkg_path / "skill_source.py").write_text(
                skill_source, encoding="utf-8"
            )

        logger.info(
            "Packaged worker for task %s at %s", task.task_id[:8], pkg_path
        )
        return pkg_path
