# Copyright (c) Alibaba, Inc. and its affiliates.
"""One private compute worker. It has no camera, actuator, permit, or database APIs."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import threading
import time
from pathlib import Path
from typing import Any

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.runtime.protocol import (
    WorkerBudgets,
    WorkerRequest,
    encode,
    receive,
    validate_payload,
)


def private_directory(path: Path) -> None:
    """Refuse symlink ancestors and nonprivate/foreign socket directories."""
    if not path.is_absolute():
        raise ValueError("worker socket needs an explicit absolute private path")
    for ancestor in (*reversed(path.parents), path):
        if ancestor.is_symlink():
            raise ValueError("worker socket directory has a symlink ancestor")
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("worker socket directory must be owner-private mode 0700")


def compute(request: WorkerRequest, budgets: WorkerBudgets) -> dict[str, Any]:
    validate_payload(request, budgets)
    if request.operation == "diagnostics":
        return {"pid": os.getpid(), "compute_only": True, "budgets": vars(budgets)}
    if request.operation in ("track_metric", "track_scene"):
        from leapcalib.contracts.tracking import MetricRoi, SceneConfig, TrackingConfig
        from leapcalib.estimation.tracking import track_metric, track_scene

        payload = request.payload
        first = MetricRoi.from_mapping(
            payload["first"], max_width=budgets.roi_width, max_height=budgets.roi_height
        )
        second = MetricRoi.from_mapping(
            payload["second"], max_width=budgets.roi_width, max_height=budgets.roi_height
        )
        config = TrackingConfig(**payload["config"])
        if request.operation == "track_scene":
            return {
                "scene": track_scene(
                    first,
                    second,
                    task_tracks=payload["task_tracks"],
                    background_tracks=payload["background_tracks"],
                    config=config,
                    scene_config=SceneConfig(**payload["scene_config"]),
                )
            }
        return {"tracking": track_metric(first, second, tracks=payload["tracks"], config=config)}
    from leapcalib.validation.bundle import EvidenceBundle
    from leapcalib.contracts.estimates import EstimateBundle
    from leapcalib.contracts.raw_task import RawTaskState, RawTaskTransition
    from leapcalib.contracts.observations import Stamp
    from leapcalib.estimation.raw_response import (
        RawResponseMethod,
        propose_raw,
        update_raw_response,
    )

    payload = request.payload
    if request.operation in ("solve_motion", "solve_raw_response"):
        evidence = EvidenceBundle.from_mapping(payload["evidence"])
        if evidence.topology.model.digest != request.model_digest:
            raise ValueError("worker envelope model differs from evidence")
        if request.operation == "solve_motion":
            from leapcalib.estimation.handeye import MotionMethod

            method = MotionMethod()
        else:
            method = RawResponseMethod()
        return {"estimate": method.solve(evidence, payload["options"]).to_dict()}
    fit = EstimateBundle.from_mapping(payload["fit"])
    if fit.topology.model.digest != request.model_digest:
        raise ValueError("worker envelope model differs from immutable candidate")
    if request.operation == "propose_raw":
        return {
            "proposal": propose_raw(
                fit,
                RawTaskState.from_mapping(payload["current"]),
                payload["target"],
                Stamp.from_mapping(payload["now"]),
                run_id=request.run_id,
                model_digest=request.model_digest,
            )
        }
    return {
        "estimate": update_raw_response(
            fit,
            RawTaskTransition.from_mapping(payload["sample"]),
            max_update_norm=payload.get("max_update_norm", 0.05),
        ).to_dict()
    }


async def serve(
    socket_path: str | Path,
    *,
    budgets: WorkerBudgets = WorkerBudgets(),
    stop: asyncio.Event | None = None,
) -> None:
    path = Path(socket_path)
    private_directory(path.parent)
    if path.exists() or path.is_symlink():
        raise ValueError("worker refuses existing socket endpoints")
    stop = stop if stop is not None else asyncio.Event()
    pending = 0
    lock = asyncio.Lock()
    clients: set[asyncio.Task[Any]] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal pending
        task = asyncio.current_task()
        if task is not None:
            clients.add(task)
        admitted = False
        try:
            if pending >= 1 + budgets.queued_jobs:
                return
            pending += 1
            admitted = True
            request = WorkerRequest.from_mapping(
                await asyncio.wait_for(receive(reader, budgets.request_bytes), budgets.update_s)
            )
            validate_payload(request, budgets)
            async with lock:
                remaining = request.deadline_monotonic_s - time.monotonic()
                maximum = (
                    budgets.solve_s if request.operation.startswith("solve_") else budgets.update_s
                )
                if remaining <= 0 or remaining > maximum + 0.1:
                    response = request.response(
                        failure={
                            "code": "deadline",
                            "message": "expired or excessive worker deadline",
                            "retryable": False,
                        }
                    )
                else:
                    # A timed-out numerical thread cannot be reused safely. Terminate the
                    # entire compute-only process; the owner reaps it and discards output.
                    watchdog = threading.Timer(remaining, lambda: os._exit(70))
                    watchdog.daemon = True
                    watchdog.start()
                    try:
                        result = await asyncio.to_thread(compute, request, budgets)
                        response = request.response(result=result)
                    except CalibrationFailure as failure:
                        response = request.response(failure=failure.to_dict())
                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                        ArithmeticError,
                        AttributeError,
                    ) as error:
                        response = request.response(
                            failure={
                                "code": "schema",
                                "message": str(error)[:512],
                                "retryable": False,
                            }
                        )
                    finally:
                        watchdog.cancel()
                writer.write(encode(response, budgets.response_bytes))
                await asyncio.wait_for(writer.drain(), budgets.update_s)
        except (OSError, ValueError, asyncio.IncompleteReadError, TimeoutError):
            # Malformed input cannot supply reliable IDs for a correlated failure.
            pass
        finally:
            if admitted:
                pending -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            if task is not None:
                clients.discard(task)

    server = await asyncio.start_unix_server(handle, str(path), limit=budgets.request_bytes + 4)
    os.chmod(path, 0o600)
    inode = path.lstat().st_ino
    try:
        async with server:
            await stop.wait()
    finally:
        for task in tuple(clients):
            task.cancel()
        await asyncio.gather(*tuple(clients), return_exceptions=True)
        if path.exists() and path.lstat().st_ino == inode:
            path.unlink()


def main(socket_path: str, budgets: WorkerBudgets = WorkerBudgets()) -> int:
    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for name in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(name, stop.set)
        await serve(socket_path, budgets=budgets, stop=stop)

    # This entry point owns a dedicated compute process. Make the endpoint
    # private at bind time, before the parent can observe it; chmod after the
    # asynchronous bind alone leaves a startup race. Create the 0700 directory
    # before the stricter socket umask, and restore the mask on exit.
    private_directory(Path(socket_path).parent)
    previous_mask = os.umask(0o177)
    try:
        asyncio.run(run())
    finally:
        os.umask(previous_mask)
    return 0
