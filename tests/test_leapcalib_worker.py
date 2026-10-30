# Copyright (c) Alibaba, Inc. and its affiliates.
"""Actual private subprocess/socket regressions, never connected to hardware."""
from __future__ import annotations

import asyncio
import uuid
import stat
import struct
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.runtime.protocol import WorkerBudgets, WorkerRequest, encode, freeze_json, receive
from leapcalib.runtime.server import private_directory
from leapflow.hardware.calibration.worker import CalibrationWorker
from test_leapcalib_raw_response import raw_study

pytestmark = pytest.mark.component


def worker_at(tmp_path: Path, **kwargs: object) -> CalibrationWorker:
    return CalibrationWorker(tmp_path.resolve().parent / ("lc-" + uuid.uuid4().hex[:8]) / "s", enabled=True, **kwargs)


def test_imports_are_compute_and_vendor_free() -> None:
    code = "import sys, importlib, pkgutil, leapcalib; [importlib.import_module(m.name) for m in pkgutil.walk_packages(leapcalib.__path__, leapcalib.__name__ + '.')]; assert not any(x.startswith('leapflow') for x in sys.modules); import leapflow.hardware.calibration.worker; assert not any(x in sys.modules for x in ('numpy','cv2','pyrealsense2','rclpy')); assert not any(x.startswith('leapflow.engine') for x in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


@pytest.mark.parametrize("change", [{"request_bytes": 17 * 1024 * 1024}, {"queued_jobs": 3}, {"update_s": 3}, {"solve_s": float("inf")}, {"rss_bytes": True}])
def test_budget_ceiling_cannot_silently_expand(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        WorkerBudgets(**change)


def test_private_directory_refuses_mode_and_symlink(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    bad = base / "bad"
    bad.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="owner-private"):
        private_directory(bad)
    link = base / "link"
    link.symlink_to(bad, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        private_directory(link)


@pytest.mark.parametrize("data", [b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}', b'[' * 40 + b']' * 40, b'{"x":[' + b'0,' * 65536 + b'0]}'])
async def test_protocol_refuses_nonfinite_duplicates_and_nesting(data: bytes) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!I", len(data)) + data)
    reader.feed_eof()
    with pytest.raises(ValueError):
        await receive(reader, 2 * 1024 * 1024)


async def test_oversize_length_refused_without_waiting_for_body() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!I", 17 * 1024 * 1024))
    with pytest.raises(ValueError, match="before allocation"):
        await asyncio.wait_for(receive(reader, 16 * 1024 * 1024), 0.1)


def test_response_ids_and_input_digest_are_exact() -> None:
    request = WorkerRequest("r", "j", "cell", "run", "model", "diagnostics", time.monotonic() + 1, {}, digest_of({}))
    response = request.response(result={"ok": True})
    assert request.validate_response(response) == response
    with pytest.raises(ValueError, match="foreign"):
        request.validate_response({**response, "run_id": "other-client"})
    with pytest.raises(ValueError, match="input digest"):
        replace(request, input_digest="forged")


async def test_real_lazy_process_modes_compute_and_deterministic_reap(tmp_path: Path) -> None:
    worker = worker_at(tmp_path)
    assert worker._process is None
    try:
        diagnostics = await worker.submit("diagnostics", {}, workcell_id="cell", run_id="run", model_digest="model")
        pid = diagnostics["pid"]
        assert diagnostics["compute_only"] is True
        assert stat.S_IMODE(worker.socket_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(worker.socket_path.parent.stat().st_mode) == 0o700
        assert worker.last_metrics["peak_rss_bytes"] < 512 * 1024 * 1024
        bundle, domain, _ = raw_study()
        result = await worker.submit("solve_raw_response", {"evidence": bundle.to_dict(), "options": {"domain": domain.to_dict()}}, workcell_id="cell", run_id="run", model_digest=bundle.topology.model.digest)
        assert result["estimate"]["metadata"]["response_rank"] == 4
        assert worker._process.pid == pid
    finally:
        process = worker._process
        await worker.close()
    assert process is not None and process.returncode is not None
    assert not worker.socket_path.exists()
    with pytest.raises(CalibrationFailure, match="closed"):
        await worker.submit("diagnostics", {}, workcell_id="cell", run_id="run", model_digest="model")


@pytest.mark.parametrize("maximum", [None, 0.05, 0.001])
async def test_actual_worker_preserves_legacy_and_tightened_update_contract(tmp_path, maximum):
    from leapcalib.estimation.raw_response import estimate_raw_response, update_raw_response
    bundle, domain, _ = raw_study()
    fit = estimate_raw_response(bundle, domain)
    # A disjoint synthetic measured-endpoint fixture; no acquisition/authority claim.
    source = next(o for o in bundle.observations if o.observation_id == "r39")
    def later(state):
        return replace(state, task=replace(state.task, state_id="later:" + state.task.state_id),
            frame_ids=("later:" + state.frame_ids[0],), feedback_ids=("later:" + state.feedback_ids[0],))
    sample = replace(source, observation_id="later-correction", first=later(source.first), second=later(source.second))
    payload = {"fit": fit.to_dict(), "sample": sample.to_dict()}
    if maximum is not None:
        payload["max_update_norm"] = maximum
    worker = worker_at(tmp_path)
    try:
        result = await worker.submit("update_raw", payload, workcell_id="synthetic-cell",
            run_id="later-run", model_digest=fit.topology.model.digest)
        reported = EstimateBundle.from_mapping(result["estimate"])
        expected = update_raw_response(fit, sample, max_update_norm=maximum or 0.05)
        assert reported.digest == expected.digest
        assert reported.holdout_ids == fit.holdout_ids
        assert reported.metadata["holdout_predictions"] == fit.metadata["holdout_predictions"]
    finally:
        await worker.close()


@pytest.mark.parametrize("maximum", [0, -0.001, 0.051, True, "0.01", None, float("nan")])
async def test_invalid_update_bound_refuses_before_process_launch(tmp_path, maximum):
    from leapcalib.estimation.raw_response import estimate_raw_response
    bundle, domain, _ = raw_study()
    fit = estimate_raw_response(bundle, domain)
    worker = worker_at(tmp_path)
    try:
        with pytest.raises(CalibrationFailure) as caught:
            await worker.submit("update_raw", {"fit": fit.to_dict(),
                "sample": bundle.observations[0].to_dict(), "max_update_norm": maximum},
                workcell_id="synthetic-cell", run_id="run", model_digest=fit.topology.model.digest)
        assert caught.value.code == "schema"
        assert worker._process is None and worker._pending == 0
    finally:
        await worker.close()


async def test_malformed_live_socket_request_does_not_crash_worker(tmp_path: Path) -> None:
    worker = worker_at(tmp_path)
    try:
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m")
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        writer.write(encode({"schema": "wrong"}, 1024))
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), 1) == b""
        writer.close()
        await writer.wait_closed()
        assert (await worker.submit("diagnostics", {}, workcell_id="c", run_id="other-run", model_digest="m"))["compute_only"]
    finally:
        await worker.close()


async def test_endpoint_is_private_at_bind_before_chmod(tmp_path, monkeypatch):
    from leapflow.hardware.calibration import worker as owner_worker

    worker = worker_at(tmp_path)
    original_spawn = owner_worker.asyncio.create_subprocess_exec
    code = """
import os, stat, sys, time
from leapcalib.runtime.server import main
os.umask(0o022)
original = os.chmod
def checked(path, mode):
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    time.sleep(0.15)
    original(path, mode)
os.chmod = checked
main(sys.argv[1])
"""

    async def spawn(*args, **kwargs):
        if args[1:3] == ("-m", "leapcalib.cli"):
            return await original_spawn(args[0], "-c", code, str(worker.socket_path), **kwargs)
        return await original_spawn(*args, **kwargs)

    monkeypatch.setattr(owner_worker.asyncio, "create_subprocess_exec", spawn)
    try:
        result = await worker.submit(
            "diagnostics", {}, workcell_id="cell", run_id="bind-mode", model_digest="model"
        )
        assert result["compute_only"]
        assert stat.S_IMODE(worker.socket_path.stat().st_mode) == 0o600
    finally:
        await worker.close()


async def test_disabled_worker_never_launches(tmp_path: Path) -> None:
    worker = CalibrationWorker(tmp_path.resolve() / "worker" / "compute.sock")
    with pytest.raises(CalibrationFailure, match="disabled"):
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m")
    assert worker._process is None


async def test_cancel_queued_job_leaves_active_process_usable(tmp_path: Path) -> None:
    worker = worker_at(tmp_path)
    try:
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m")
        process = worker._process
        await worker._lock.acquire()
        queued = asyncio.create_task(worker.submit("diagnostics", {}, workcell_id="c", run_id="queued", model_digest="m"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        worker._lock.release()
        assert worker._process is process and process.returncode is None
        assert (await worker.submit("diagnostics", {}, workcell_id="c", run_id="later", model_digest="m"))["compute_only"]
    finally:
        await worker.close()


@pytest.mark.parametrize("late_result", [False, True])
async def test_parent_deadline_discards_output_reaps_and_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_result: bool,
) -> None:
    from leapflow.hardware.calibration import worker as owner_worker

    worker = worker_at(tmp_path)
    original_receive = owner_worker.receive
    reads = 0
    try:
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="warmup", model_digest="m")
        process = worker._process
        worker.budgets = replace(worker.budgets, update_s=0.05)

        async def delayed_receive(reader: asyncio.StreamReader, maximum: int) -> dict[str, object]:
            nonlocal reads
            reads += 1
            if late_result:
                response = await original_receive(reader, maximum)
                await asyncio.sleep(0.2)
                return response
            await asyncio.sleep(0.2)
            return await original_receive(reader, maximum)

        monkeypatch.setattr(owner_worker, "receive", delayed_receive)
        with pytest.raises(CalibrationFailure) as caught:
            await worker.submit("diagnostics", {}, workcell_id="c", run_id="expired", model_digest="m")
        assert caught.value.code == "deadline"
        assert "must not be replayed" in caught.value.message
        assert reads == 1
        assert worker._process is None
        assert process is not None and process.returncode is not None
        assert not worker.socket_path.exists()
        assert worker._pending == 0 and worker._active_task is None
        monkeypatch.setattr(owner_worker, "receive", original_receive)
        worker.budgets = WorkerBudgets()
        result = await worker.submit("diagnostics", {}, workcell_id="c", run_id="later-independent", model_digest="m")
        assert result["compute_only"] is True
        assert result["pid"] != process.pid
    finally:
        await worker.close()


async def test_parent_rss_refusal_reaps_and_allows_only_one_later_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = worker_at(tmp_path)
    try:
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m")
        old_rss = worker._rss
        async def excessive(pid: int) -> int:
            return worker.budgets.rss_bytes + 1
        monkeypatch.setattr(worker, "_rss", excessive)
        with pytest.raises(CalibrationFailure, match="RSS"):
            await worker.submit("diagnostics", {}, workcell_id="c", run_id="failure", model_digest="m")
        assert worker._process is None
        monkeypatch.setattr(worker, "_rss", old_rss)
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="fresh", model_digest="m")
        monkeypatch.setattr(worker, "_rss", excessive)
        with pytest.raises(CalibrationFailure):
            await worker.submit("diagnostics", {}, workcell_id="c", run_id="failure2", model_digest="m")
        monkeypatch.setattr(worker, "_rss", old_rss)
        with pytest.raises(CalibrationFailure, match="recovery exhausted"):
            await worker.submit("diagnostics", {}, workcell_id="c", run_id="third", model_digest="m")
    finally:
        await worker.close()


@pytest.mark.parametrize("value", ["plain", "\"\\\n\x00\x7f", "é😀", [None, True, False, 1.25, -3]])
def test_freeze_counts_exact_canonical_bytes_and_detaches_containers(value):
    wire = encode({"value": value}, 1024)
    count = len(wire) - 4
    assert freeze_json({"value": value}, count) == {"value": value}
    with pytest.raises(ValueError, match="byte budget"):
        freeze_json({"value": value}, count - 1)
    source = {"rows": [[1, 2]], "options": {"keep": True}}
    frozen = freeze_json(source, 1024)
    source["rows"][0][0] = 9
    source["options"]["keep"] = False
    assert frozen == {"rows": [[1, 2]], "options": {"keep": True}}


@pytest.mark.parametrize("payload", [{"x": float("nan")}, {1: "bad-key"}, {"x": 1 << 1025}, {"x": [0] * 65537}, {"x": "x" * 1025}])
async def test_malformed_owner_input_never_launches_or_occupies_queue(tmp_path, payload):
    worker = worker_at(tmp_path, budgets=WorkerBudgets(request_bytes=1024))
    with pytest.raises(CalibrationFailure) as caught:
        await worker.submit("diagnostics", payload, workcell_id="c", run_id="r", model_digest="m")
    assert caught.value.code == "schema"
    assert worker._pending == 0 and worker._process is None and not worker._started
    await worker.close()


async def test_invalid_job_preserves_live_process_and_recovery_allowance(tmp_path):
    worker = worker_at(tmp_path)
    try:
        await worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m")
        process = worker._process
        for operation, payload in (("unknown", {}), ("diagnostics", {"extra": []})):
            with pytest.raises(CalibrationFailure) as caught:
                await worker.submit(operation, payload, workcell_id="c", run_id="r", model_digest="m")
            assert caught.value.code == "schema"
        assert worker._process is process and process.returncode is None and worker._recovery_launches == 1
        assert (await worker.submit("diagnostics", {}, workcell_id="c", run_id="later", model_digest="m"))["pid"] == process.pid
    finally:
        await worker.close()


@pytest.mark.parametrize("failure", [{}, {"code": "foreign", "message": "refused", "retryable": False},
    {"code": "schema", "message": "refused", "retryable": 1},
    {"code": "schema", "message": "x" * 513, "retryable": False}])
def test_response_failure_requires_typed_bounded_fields(failure):
    request = WorkerRequest("r", "j", "cell", "run", "model", "diagnostics", time.monotonic() + 1, {}, digest_of({}))
    with pytest.raises(ValueError, match="typed bounded"):
        request.validate_response(request.response(failure=failure))


async def test_request_serialization_runs_off_owner_loop_and_late_preparation_never_launches(tmp_path, monkeypatch):
    from leapflow.hardware.calibration import worker as owner_worker
    worker = worker_at(tmp_path, budgets=WorkerBudgets(update_s=0.05))
    entered = threading.Event()
    release = threading.Event()
    original = owner_worker._prepare_request
    owner_thread = threading.get_ident()
    def slow_prepare(*args):
        assert threading.get_ident() != owner_thread
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(owner_worker, "_prepare_request", slow_prepare)
    task = asyncio.create_task(worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m"))
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 1)
        await asyncio.sleep(0.06)
        assert not task.done() and worker._process is None
        release.set()
        with pytest.raises(CalibrationFailure) as caught:
            await task
        assert caught.value.code == "deadline" and worker._pending == 0 and worker._process is None
    finally:
        release.set()
        await worker.close()


async def test_queue_slots_bound_frozen_jobs_and_cancelled_preparation_settles(tmp_path, monkeypatch):
    from leapflow.hardware.calibration import worker as owner_worker
    worker = worker_at(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = owner_worker._prepare_request
    def slow_prepare(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(owner_worker, "_prepare_request", slow_prepare)
    jobs = [asyncio.create_task(worker.submit("diagnostics", {}, workcell_id="c", run_id=str(i), model_digest="m")) for i in range(3)]
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 1)
        with pytest.raises(CalibrationFailure, match="queue full"):
            await worker.submit("diagnostics", {}, workcell_id="c", run_id="overflow", model_digest="m")
        for job in jobs:
            job.cancel()
        await asyncio.sleep(0)
        jobs[0].cancel()
        assert worker._pending >= 1 and not jobs[0].done()
        release.set()
        results = await asyncio.gather(*jobs, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert worker._pending == 0 and worker._process is None and not worker._lock.locked()
    finally:
        release.set()
        await asyncio.gather(*jobs, return_exceptions=True)
        await worker.close()


async def test_repeated_cancel_during_close_still_reaps_real_process(tmp_path, monkeypatch):
    worker = worker_at(tmp_path)
    await worker.submit("diagnostics", {}, workcell_id="c", run_id="r", model_digest="m")
    process = worker._process
    entered = asyncio.Event()
    release = asyncio.Event()
    original = worker._reap
    async def delayed_reap():
        entered.set()
        await release.wait()
        await original()
    monkeypatch.setattr(worker, "_reap", delayed_reap)
    closing = asyncio.create_task(worker.close())
    await entered.wait()
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert worker._process is None and process.returncode is not None
    assert not worker.socket_path.exists() and not worker._lock.locked()
    await worker.close()
