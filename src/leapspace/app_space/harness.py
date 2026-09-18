# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapAppHarness — end-to-end orchestration of one LeapSpace task run.

Loads a task config, lints it, boots a disposable sandbox, injects the
task's wiring, and drives the apps. The harness owns no verdict: the
in-sandbox expect's PASS/FAIL lines + exit code are the only ground
truth. Host-side only; in-sandbox code never imports this.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from pathlib import Path, PurePath
from typing import Any, Literal

from cua_sandbox import Sandbox
from cua_sandbox.runtime import QEMURuntime

from leapspace.app_space.apps import APP_MODULES
from leapspace.app_space.actor import LeapAppActor
from leapspace.app_space.action_lint import lint_task
from leapspace.app_space.config import AppTaskConfig
from leapspace.app_space.signal import (
    RECORD_DONE_FILE,
    RECORD_START_FILE,
    RECORD_STOP_FILE,
)
from leapspace.app_space.image import get_image
from leapspace.app_space.state import (
    LeapAppImage,
    get_image_venv_python,
    get_sandbox_state_dir,
    load_action,
)

logger = logging.getLogger(__name__)

LAUNCH_READY_TIMEOUT_S = 60.0
LAUNCH_READY_POLL_S = 1.0
SIGNAL_READY_TIMEOUT_S = 60.0
SIGNAL_DONE_TIMEOUT_S = 60.0
SIGNAL_POLL_S = 0.5


class LeapAppHarness:
    """Single-task orchestrator bound to a sandbox image preset.

    Run-invariant choices (image preset, sandbox name) live on the
    instance; the per-run input — the task config — travels through
    run_task().
    """

    def __init__(
        self,
        image: LeapAppImage,
        sandbox_name: str | None = None,
    ) -> None:
        """Bind the image preset; sandbox name defaults to a per-image name."""
        self.image = image
        self.sandbox_name = sandbox_name or f"leapspace-app-{image.value}"

    async def _prepare_files(self, config: AppTaskConfig, actor: LeapAppActor) -> None:
        """Inject the task's wiring into each app's state dir.

        Every app gets hooks.json (the config.hooks mapping, read by the
        in-app registry with stdlib json) and hooks.py, a copy of
        config.action_path — the same file doubles as the in-sandbox
        verdict program (its __main__ runs expect()), so hooks and
        verdict share one source. Writes go through the actor so they
        land inside the sandbox.
        """
        action_source = config.action_path.read_text()
        state_root = get_sandbox_state_dir(in_sandbox=False, system=self.image.value)
        for app_id in config.app_ids:
            state_dir = state_root / app_id
            await actor.fs_mkdir(str(state_dir))
            await actor.fs_write(
                str(state_dir / "hooks.json"),
                json.dumps(config.hooks.get(app_id, {})),
            )
            await actor.fs_write(str(state_dir / "hooks.py"), action_source)

    async def _launch_apps(
        self, config: AppTaskConfig, actor: LeapAppActor
    ) -> dict[str, int]:
        """Start every configured app concurrently and wait until all settle.

        gather() overlaps the slow Qt startups. return_exceptions=True
        collects instead of racing: every failing app is reported (the
        extras logged, the first re-raised), and healthy siblings die
        with the sandbox at teardown — no cleanup needed here.
        Returns the app_id → pid map so later phases can kill apps.
        """
        results = await asyncio.gather(
            *(self._launch_app(app_id, config, actor) for app_id in config.app_ids),
            return_exceptions=True,
        )
        failures = [
            (app_id, result)
            for app_id, result in zip(config.app_ids, results)
            if isinstance(result, BaseException)
        ]
        if failures:
            for app_id, exc in failures[1:]:
                logger.error("app %s also failed to launch", app_id, exc_info=exc)
            raise failures[0][1]
        return dict(zip(config.app_ids, results))

    async def _launch_app(
        self, app_id: str, config: AppTaskConfig, actor: LeapAppActor
    ) -> int:
        """Start one app in the background and wait until it is usable.

        Usable means the first state.json is written (construction,
        before_launch hooks, initial persist), a window with the
        envelope's app_title is on screen, and the app's persisted
        interface covers the names config.interface declares for it.
        Returns the spawned pid so later phases can kill the app.
        """
        app_module = APP_MODULES.get(app_id)
        if app_module is None:
            raise ValueError(
                f"unknown app_id {app_id!r}; must be one of {sorted(APP_MODULES)}"
            )
        # background=True: stdout is the spawned pid, no exit code —
        # startup crashes are caught by the state poll below, not by this call
        python = get_image_venv_python(system=self.image.value)
        launch = await actor.shell_run(f"{python} -m {app_module}", background=True)
        pid = int(launch.stdout.strip())

        envelope = await self._wait_for_app_state(app_id, actor)
        await actor.wait_for_window(envelope["app_title"])

        # config declares the mutation budget; the app's persisted interface
        # is what it actually exposes — the latter must win
        missing = set(config.interface.get(app_id, ())) - set(envelope["interface"])
        if missing:
            raise RuntimeError(
                f"app {app_id!r} does not expose interface {sorted(missing)} "
                f"required by task {config.id}"
            )
        return pid

    async def _wait_for_app_state(
        self, app_id: str, actor: LeapAppActor
    ) -> dict[str, Any]:
        """Poll the app's state.json until its first persist lands; return it.

        background launches report no exit code, so this poll doubles as
        the crash detector: an app that dies during startup shows up as
        a file that never appears.
        """
        state_path = (
            get_sandbox_state_dir(in_sandbox=False, system=self.image.value)
            / app_id
            / "state.json"
        )
        deadline = time.monotonic() + LAUNCH_READY_TIMEOUT_S
        while not await actor.fs_exists(str(state_path)):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"app {app_id!r} never wrote {state_path} within "
                    f"{LAUNCH_READY_TIMEOUT_S}s — startup crash or hung app"
                )
            await asyncio.sleep(LAUNCH_READY_POLL_S)
        # the first persist proves __init__ + before_launch hooks are done;
        # the write is atomic, so an existing file is never half-written
        return json.loads(await actor.fs_read(str(state_path)))

    async def run_task(
        self,
        config_path: str | Path,
        mode: Literal["signal", "e2e"]
    ) -> None:
        """Run one task end-to-end: load → lint → boot → drive → teardown.

        Lint problems are task-authoring bugs — fail fast before paying
        any sandbox cost. The sandbox lives for exactly this run; all
        phases run inside its context and teardown runs on exit.
        """
        config = AppTaskConfig.load(config_path)
        lint_problems = lint_task(config)
        if lint_problems:
            for problem in lint_problems:
                logger.error(problem)
            raise RuntimeError(f"task {config_path} has {len(lint_problems)} lint problems")

        async with Sandbox.ephemeral(
            image=get_image(self.image),
            name=self.sandbox_name,
            local=True,
            runtime=QEMURuntime(mode="bare-metal"),
        ) as sandbox:
            async with LeapAppActor(sandbox=sandbox, system=self.image.value) as actor:
                if mode == "signal":
                    await self._run_signal(config, actor)
                elif mode == "e2e":
                    raise NotImplementedError("e2e mode is not yet implemented")
                else:
                    raise ValueError(f"unknown mode {mode!r}; must be 'signal' or 'e2e'")

    async def _run_signal(self, config: AppTaskConfig, actor: LeapAppActor) -> int:
        """Drive one signal-mode run: record the stimulus, then judge it.

        Launch order encodes the recording window — LeapSignal starts
        after the apps are ready and before the stimulus, so recording
        begins at process start. record_start.json gates the stimulus;
        record_done.json proves the drain finished (stop_recording
        drains internally), so the in-box verdict only reads fully-settled
        state. The verdict itself is `python hooks.py` — action.py's
        __main__ runs expect() — through shell_run(check=False) so a FAIL
        exit code stays a measured result, not an infra failure.

        App-side setup lives here, not in run_task: the e2e path drives a
        completely different flow and must not inherit signal-mode wiring.
        """
        await self._prepare_files(config, actor)
        await self._launch_apps(config, actor)
        # The driver's agent-cursor overlay is a full-screen window
        # above every app: it swallows all pixel input, so it must
        # be gone before any human-real stimulus runs.
        await actor.disable_agent_cursor()

        # Resolve the run's control paths and load the task's action pair.
        state_root = get_sandbox_state_dir(in_sandbox=False, system=self.image.value)
        signal_dir = state_root / config.id / "signal"
        reference, _expect = load_action(config.action_path)

        # background=True: pid only; a startup death surfaces as the death
        # certificate (record_done.json) or not at all, both caught by the poll.
        # The watch surface is the apps' state dirs — the ground-truth writes
        # this run exists to observe — while the control files (this run's
        # signal dir) stay unwatched, so the harness never records its own
        # record_stop handshake as signal.
        python = get_image_venv_python(system=self.image.value)
        watch_flags = "".join(
            f" --watch {shlex.quote(str(state_root / app_id))}"
            for app_id in config.app_ids
        )
        await actor.shell_run(
            f"{python} -m leapspace.app_space.signal {shlex.quote(str(signal_dir))}"
            f" --goal {shlex.quote(config.instruction)}{watch_flags}",
            background=True,
        )
        # Readiness gate: record_start.json proves recording is live before the
        # stimulus starts.
        first, payload = await self._await_sentinel(
            actor,
            signal_dir,
            (RECORD_START_FILE, RECORD_DONE_FILE),
            SIGNAL_READY_TIMEOUT_S,
        )
        if first == RECORD_DONE_FILE:
            raise RuntimeError(
                f"signal: LeapSignal failed during startup: {payload.get('error')}"
            )
        logger.info("signal: recording trajectory %s", payload.get("trajectory_id"))

        # The stimulus: the task's reference actions, the signal being recorded.
        await reference(actor)

        # Close the recording window, then wait for record_done.json — the
        # drain proof that every event has been persisted before the verdict.
        await actor.fs_create(
            str(signal_dir / RECORD_STOP_FILE), json.dumps({"stop": True})
        )
        _, done = await self._await_sentinel(
            actor, signal_dir, (RECORD_DONE_FILE,), SIGNAL_DONE_TIMEOUT_S
        )
        if not done.get("ok"):
            raise RuntimeError(f"signal: LeapSignal failed: {done.get('error')}")
        logger.info("signal: %s", done)

        # hooks.py is action.py's in-box twin; identical copies land in every
        # app's state_dir, expect runs from the first app's copy
        hooks_path = state_root / config.app_ids[0] / "hooks.py"
        result = await actor.shell_run(f"{python} {shlex.quote(str(hooks_path))}", check=False)
        # PASS/FAIL lines are the run's user-facing verdict: print them
        # verbatim and let the exit code travel up as the result.
        if result.stdout:
            print(result.stdout, end="")
        else:
            logger.error(
                "verdict: no PASS/FAIL lines (exit %d): %s",
                result.returncode,
                result.stderr.strip(),
            )
        if result.returncode != 0:
            logger.error(
                "verdict: task %s failed expect (exit %d)", config.id, result.returncode
            )
        return result.returncode

    async def _await_sentinel(
        self,
        actor: LeapAppActor,
        signal_dir: PurePath,
        names: tuple[str, ...],
        timeout_s: float,
    ) -> tuple[str, dict[str, Any]]:
        """Poll until any of the named record_* JSON sentinels lands.

        Waiting for record_start.json also watches record_done.json —
        LeapSignal's death certificate, written even when it dies during
        startup — so a crash surfaces as its real error, not a bare
        timeout.
        """
        paths = {name: str(signal_dir / name) for name in names}
        deadline = time.monotonic() + timeout_s
        while True:
            for name, path in paths.items():
                if await actor.fs_exists(path):
                    return name, json.loads(await actor.fs_read(path))
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"signal: {'/'.join(names)} did not appear within "
                    f"{timeout_s}s — LeapSignal died or hung"
                )
            await asyncio.sleep(SIGNAL_POLL_S)
