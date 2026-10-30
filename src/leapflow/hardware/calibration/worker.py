# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lazy owner-bound compute subprocess. No transport, camera or DB authority is delegated."""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.runtime.protocol import (
    WorkerBudgets, WorkerRequest, encode, freeze_json, receive, validate_payload,
)
from leapcalib.runtime.server import private_directory


@runtime_checkable
class ComputeWorker(Protocol):
    """Owner-injected bounded computation; requests never grant hardware authority."""

    async def submit(
        self, operation: str, payload: Mapping[str, Any], *, workcell_id: str,
        run_id: str, model_digest: str,
    ) -> Mapping[str, Any]: ...


class CalibrationWorker:
    def __init__(self, socket_path: Path, *, enabled: bool = False, budgets: WorkerBudgets = WorkerBudgets()) -> None:
        if type(enabled) is not bool:
            raise ValueError("worker enablement must be explicit")
        self.socket_path = socket_path
        self.enabled = enabled
        self.budgets = budgets
        self._process: asyncio.subprocess.Process | None = None
        self._socket_inode: int | None = None
        self._lock = asyncio.Lock()
        self._active_task: asyncio.Task[Any] | None = None
        self._pending = 0
        self._closed = False
        self._started = False
        self._recovery_launches = 1
        self._close_task: asyncio.Task[None] | None = None
        self.last_metrics: dict[str, Any] = {}

    @classmethod
    def from_settings(cls, settings: Any) -> CalibrationWorker:
        raw = getattr(settings, "hardware_calibration_worker_budgets", "{}")
        if not isinstance(raw, str) or len(raw) > 4096:
            raise ValueError("calibration worker budgets must be bounded JSON")
        return cls(settings.profile_layout.hardware.calibration_worker_socket,
                   enabled=getattr(settings, "hardware_calibration_worker_enabled", False),
                   budgets=WorkerBudgets(**json.loads(raw)))

    async def _rss(self, pid: int) -> int:
        def linux_rss() -> int:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
            raise OSError("worker RSS unavailable")
        if sys.platform.startswith("linux"):
            return await asyncio.to_thread(linux_rss)
        proc = await asyncio.create_subprocess_exec("ps", "-o", "rss=", "-p", str(pid), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), 0.5)
            return int(output.strip()) * 1024
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

    async def _launch(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        if self._started:
            if self._recovery_launches == 0:
                raise CalibrationFailure("worker_unavailable", "bounded worker recovery exhausted")
            self._recovery_launches -= 1
        maximum_path = 103 if sys.platform == "darwin" else 107
        if len(os.fsencode(self.socket_path)) > maximum_path:
            raise CalibrationFailure("worker_unavailable", "profile worker socket path exceeds platform limit; configure a shorter profile root")
        private_directory(self.socket_path.parent)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise CalibrationFailure("worker_unavailable", "existing/foreign worker endpoint refused")
        env = {**os.environ, **{name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
        # Only the owner selects the interpreter/module/socket; requests never select code.
        deadline = time.monotonic() + self.budgets.startup_s
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(sys.executable, "-m", "leapcalib.cli", "serve", "--socket", str(self.socket_path), "--budgets-json", json.dumps(vars(self.budgets)), env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))
        try:
            self._process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            # Spawn may already have created a child; retain it before propagating
            # cancellation so the owner can terminate and reap that exact process.
            self._process = await _settle_task(spawn)
            self._started = True
            raise
        self._started = True
        while time.monotonic() < deadline:
            if self._process.returncode is not None:
                raise CalibrationFailure("worker_unavailable", "worker exited during startup")
            if self.socket_path.exists():
                info = self.socket_path.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                    raise CalibrationFailure("worker_unavailable", "worker endpoint ownership/mode refused")
                self._socket_inode = info.st_ino
                if await self._rss(self._process.pid) > self.budgets.rss_bytes:
                    raise CalibrationFailure("budget_exceeded", "worker startup RSS exceeded")
                return
            await asyncio.sleep(0.025)
        raise CalibrationFailure("worker_unavailable", "worker startup deadline exceeded")

    async def _reap(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), 1.0)
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            await process.wait()
        if self._socket_inode is not None:
            try:
                info = self.socket_path.lstat()
                if info.st_ino == self._socket_inode and info.st_uid == os.getuid() and stat.S_ISSOCK(info.st_mode):
                    self.socket_path.unlink()
            except FileNotFoundError:
                pass
            self._socket_inode = None

    async def _settle_reap(self) -> None:
        await _settle_task(asyncio.create_task(self._reap()))

    async def submit(self, operation: str, payload: Mapping[str, Any], *, workcell_id: str, run_id: str, model_digest: str, request_id: str | None = None) -> dict[str, Any]:
        if not self.enabled or self._closed:
            raise CalibrationFailure("worker_unavailable", "calibration worker disabled or closed")
        if self._pending >= 1 + self.budgets.queued_jobs:
            raise CalibrationFailure("budget_exceeded", "calibration worker queue full")
        self._pending += 1
        freeze_started = time.monotonic()
        writer: asyncio.StreamWriter | None = None
        acquired = False
        worker_used = False
        try:
            # Copy bounded plain containers before suspension. Large JSON encoding,
            # hashing and response decoding run off the hardware event loop.
            frozen = freeze_json(payload, self.budgets.request_bytes)
            freeze_elapsed = time.monotonic() - freeze_started
            await self._lock.acquire()
            acquired = True
            self._active_task = asyncio.current_task()
            if self._closed:
                raise CalibrationFailure("worker_unavailable", "owner shutting down")
            maximum = self.budgets.solve_s if isinstance(operation, str) and operation.startswith("solve_") else self.budgets.update_s
            preparation_started = time.monotonic()
            request = await _off_loop(_prepare_request, operation, frozen, self.budgets,
                request_id, workcell_id, run_id, model_digest, preparation_started + maximum)
            preparation_elapsed = freeze_elapsed + time.monotonic() - preparation_started
            if preparation_elapsed >= maximum:
                raise TimeoutError("worker preparation elapsed budget exceeded")
            worker_used = True
            await self._launch()
            start = time.monotonic() - preparation_elapsed
            wire_request, encoded = await _off_loop(_encode_request, request, start + maximum, self.budgets)
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.socket_path), limit=self.budgets.response_bytes + 4),
                _remaining(wire_request.deadline_monotonic_s))
            writer.write(encoded)
            await asyncio.wait_for(writer.drain(), _remaining(wire_request.deadline_monotonic_s))
            response = await self._response(reader, wire_request, start)
            if response["kind"] == "failure":
                failure = response["failure"]
                raise CalibrationFailure(failure["code"], failure["message"], retryable=False)
            return response["result"]
        except asyncio.CancelledError:
            if worker_used:
                await self._settle_reap()
            raise
        except (OSError, TimeoutError, ValueError, TypeError, asyncio.IncompleteReadError, CalibrationFailure) as error:
            if worker_used:
                await self._settle_reap()
            if isinstance(error, CalibrationFailure):
                raise
            if isinstance(error, TimeoutError):
                raise CalibrationFailure("deadline", "compute deadline expired; late output discarded and motion must not be replayed") from error
            if not worker_used and isinstance(error, (ValueError, TypeError)):
                raise CalibrationFailure("schema", "bounded compute input refused before worker use") from error
            raise CalibrationFailure("worker_unavailable", "compute worker failed; motion must not be replayed") from error
        finally:
            self._pending -= 1
            if acquired:
                self._active_task = None
                self._lock.release()
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

    async def _response(self, reader: asyncio.StreamReader, request: WorkerRequest, start: float) -> dict[str, Any]:
        receive_task = asyncio.create_task(receive(reader, self.budgets.response_bytes))
        peak = 0
        try:
            while not receive_task.done():
                remaining = _remaining(request.deadline_monotonic_s)
                assert self._process is not None
                rss = await asyncio.wait_for(self._rss(self._process.pid), remaining)
                peak = max(peak, rss)
                if rss > self.budgets.rss_bytes:
                    raise CalibrationFailure("budget_exceeded", "worker RSS budget exceeded")
                await asyncio.wait({receive_task}, timeout=min(0.025, _remaining(request.deadline_monotonic_s)))
            _remaining(request.deadline_monotonic_s)
            response = await _off_loop(request.validate_response, receive_task.result())
            _remaining(request.deadline_monotonic_s)
            return response
        finally:
            if not receive_task.done():
                receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
            self.last_metrics = {"elapsed_s": time.monotonic() - start, "peak_rss_bytes": peak, "job_id": request.job_id}

    async def close(self) -> None:
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _settle_task(self._close_task)
            raise

    async def _close(self) -> None:
        active = self._active_task
        if active is not None and active is not asyncio.current_task():
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)
        async with self._lock:
            await self._settle_reap()


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("late worker output or expired preparation discarded")
    return remaining


async def _settle_task(task: asyncio.Task[Any]) -> Any:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _off_loop(function: Any, *args: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Bounded JSON work cannot be killed in a thread. Settle it before releasing
        # the queue slot, preventing repeated cancellation from piling up work.
        try:
            await _settle_task(task)
        except (ValueError, TypeError):
            pass
        raise


def _prepare_request(operation: str, payload: dict[str, Any], budgets: WorkerBudgets,
                     request_id: str | None, workcell_id: str, run_id: str,
                     model_digest: str, deadline: float) -> WorkerRequest:
    request = WorkerRequest(request_id if request_id is not None else str(uuid.uuid4()),
        str(uuid.uuid4()), workcell_id, run_id, model_digest, operation, deadline, payload, digest_of(payload))
    validate_payload(request, budgets)
    # Refuse the complete byte envelope before launching a process.
    encode(request.to_dict(), budgets.request_bytes)
    return request


def _encode_request(request: WorkerRequest, deadline: float, budgets: WorkerBudgets) -> tuple[WorkerRequest, bytes]:
    from dataclasses import replace
    request = replace(request, deadline_monotonic_s=deadline)
    return request, encode(request.to_dict(), budgets.request_bytes)
