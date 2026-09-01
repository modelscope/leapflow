"""Journey runner: one coarse end-to-end test made diagnosable.

The real layer is deliberately small — a handful of journeys, each covering many
user-facing features as ordered phases inside a single session. Coarse tests buy
cross-module coverage at the cost of failure localization, so :meth:`Journey.phase`
buys the localization back: a failure names the phase it happened in and carries
the daemon's own log tail, which is the only place a cross-process cause is
recorded.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests._harness.cassette_proxy import (
    LIVE,
    RECORD,
    CassetteProxy,
    Script,
    store_for,
    upstream_from_env,
)
from tests._harness.leapd import Leapd, journey_root, start_leapd

logger = logging.getLogger(__name__)

DEFAULT_DEADLINE_S = 90.0

# A journey should never need many provider calls. The ceiling is what keeps a
# non-converging turn from burning the engine's whole iteration budget — offline
# that costs minutes, live it costs money.
DEFAULT_MAX_LLM_CALLS = 12

# Token ceiling, checked independently of the call count. Call count alone cannot
# catch prompt growth: a longer system prompt or a bigger tool catalogue raises
# the bill without adding a single round, which is precisely how a live lane's
# cost creeps up unnoticed.
DEFAULT_MAX_LLM_TOKENS = 150_000


class JourneyPhaseError(AssertionError):
    """A journey phase failed; carries the phase trail and daemon log tail."""


@dataclass
class Journey:
    """A running journey: cassette proxy, real leapd, and a phase trail."""

    journey_id: str
    proxy: CassetteProxy
    daemon: Leapd
    deadline_s: float = DEFAULT_DEADLINE_S
    max_llm_calls: int = 0
    max_llm_tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)
    trail: list[str] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        """Seconds since the journey started."""
        return time.monotonic() - self.started_at

    @property
    def is_live(self) -> bool:
        """True when answers come from a real provider rather than a recording.

        Journeys use this to relax the assertions that depend on a *specific*
        model choice ("it called this tool") while keeping the invariants that
        must hold either way ("the turn produced an answer and no error"). The
        strict form still runs on every push, in replay.
        """
        return self.proxy.mode in (LIVE, RECORD)

    def client(self, *, timeout_s: float = 120.0) -> Any:
        """Return a fresh RPC client for the journey's daemon."""
        return self.daemon.client(timeout_s=timeout_s)

    def workspace(self, name: str) -> Path:
        """Create (once) and return an isolated workspace for this journey."""
        return self.daemon.workspace(name)

    @contextlib.contextmanager
    def phase(self, label: str) -> Iterator[None]:
        """Run one named stage, attributing any failure to it.

        Nested phases are supported and render as ``outer > inner``.
        """
        self.trail.append(label)
        crumb = " > ".join(self.trail)
        began = time.monotonic()
        logger.info("journey %s: phase %s", self.journey_id, crumb)
        try:
            yield
        except JourneyPhaseError:
            # An inner phase already attributed the failure; re-wrapping would bury
            # the specific phase under the outer one.
            raise
        except Exception as exc:  # re-raised with cross-process context attached
            raise JourneyPhaseError(
                f"journey {self.journey_id!r} failed in phase {crumb!r} "
                f"after {self.elapsed_s:.1f}s: {type(exc).__name__}: {exc}\n"
                f"--- leapd log tail ---\n{self.daemon.tail_log()}"
            ) from exc
        finally:
            took = time.monotonic() - began
            logger.info("journey %s: phase %s took %.2fs", self.journey_id, crumb, took)
            self.trail.pop()

    def finish(self) -> None:
        """Assert the journey's own contracts: no misses, within call and time budget.

        These budgets are part of the design, not a nicety. The real layer earns
        the right to run on every push only by staying cheap, and a turn that
        stops converging shows up here first — as more provider calls than the
        journey should ever need.
        """
        self.proxy.assert_no_misses()
        if self.proxy.stats.budget_exceeded:
            raise AssertionError(
                f"journey {self.journey_id!r} exhausted its provider-call budget of "
                f"{self.max_llm_calls} after {self.proxy.stats.call_count} calls. "
                "A turn stopped converging, or prompt assembly grew enough to need "
                "extra rounds; investigate rather than raising the ceiling."
            )
        if self.proxy.stats.token_budget_exceeded:
            raise AssertionError(
                f"journey {self.journey_id!r} spent {self.proxy.stats.total_tokens} "
                f"tokens, past its ceiling of {self.max_llm_tokens}. The call count "
                "stayed within budget, so this is prompt growth, not a loop — and it "
                "is what would otherwise raise the live lane's bill in silence."
            )
        logger.info(
            "journey %s: %d provider call(s), %d token(s), %.1fs",
            self.journey_id,
            self.proxy.stats.call_count,
            self.proxy.stats.total_tokens,
            self.elapsed_s,
        )
        if self.elapsed_s > self.deadline_s:
            raise AssertionError(
                f"journey {self.journey_id!r} took {self.elapsed_s:.1f}s, "
                f"over its {self.deadline_s:.0f}s budget — split a phase out or "
                f"reduce the number of turns rather than raising the budget"
            )


class JourneyFactory:
    """Builds journeys and owns their teardown for one test.

    Each journey gets a clean, deterministic scratch root rather than pytest's
    ``tmp_path``: a daemon needs a Unix socket short enough for the kernel, and
    recorded tool calls embed absolute paths that only resolve if the workspace
    lands in the same place on every run.
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._stack = contextlib.ExitStack()
        self._journeys: list[Journey] = []
        self._roots: list[Path] = []

    def __call__(
        self,
        journey_id: str,
        *,
        script: Script | None = None,
        deadline_s: float = DEFAULT_DEADLINE_S,
        max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
        max_llm_tokens: int = DEFAULT_MAX_LLM_TOKENS,
        requires_scripted_responses: bool = False,
        extra_env: dict[str, str] | None = None,
        model: str = "cassette-model",
        profile: str = "default",
    ) -> Journey:
        """Start a cassette proxy and a real leapd, returning the journey handle.

        Args:
            journey_id: Stable id; also names the journey's cassette directory.
            script: Responses used to seed cassettes when none exist yet.
            deadline_s: Wall-clock ceiling asserted by :meth:`Journey.finish`.
            max_llm_calls: Provider-call ceiling. Enforced by the proxy, so a
                turn that stops converging is cut off instead of running to the
                engine's iteration cap.
            max_llm_tokens: Token ceiling, enforced independently. Call count
                cannot catch prompt growth, which raises cost without adding a
                round.
            requires_scripted_responses: Set when the journey's assertions depend
                on responses only a recording can produce — injected 429s, a
                context overflow, a specific tool call. Forwarding modes cannot
                produce those, so the journey skips rather than failing for a
                reason that has nothing to do with the product.
            extra_env: Additional ``LEAPFLOW_*`` overrides.
            model: Model name recorded in cassette fingerprints.
            profile: Profile id to create and activate.
        """
        if requires_scripted_responses and self._mode in (LIVE, RECORD):
            pytest.skip(
                f"journey {journey_id!r} asserts on injected provider behavior, which "
                f"{self._mode!r} mode cannot produce — it forwards every request "
                "upstream. This journey is meaningful in replay only."
            )

        upstream_base_url, upstream_api_key, upstream_model = upstream_from_env()
        proxy = CassetteProxy(
            store_for(journey_id, mode=self._mode),
            mode=self._mode,
            script=script,
            upstream_base_url=upstream_base_url,
            upstream_api_key=upstream_api_key,
            upstream_model=upstream_model,
            max_calls=max_llm_calls,
            max_tokens=max_llm_tokens,
        )
        self._stack.enter_context(proxy)

        # The daemon always sees the journey's stable placeholder model, in every
        # mode. Cassette fingerprints include the model, so letting the real
        # provider's name reach the daemon would make recordings unreplayable; the
        # proxy rewrites the name on the wire instead.
        root = journey_root(journey_id)
        self._roots.append(root)
        # Journey cassettes fingerprint the complete tool catalogue. Product defaults enable
        # passive hardware discovery, which correctly adds ``hw_*`` tools for a real
        # profile -- but that is ambient capability unrelated to seven conversation/
        # lifecycle journeys whose recordings predate it. Keep the harness deterministic by
        # explicitly disabling hardware here; a hardware journey opts in deliberately (R8
        # passes ``LEAPFLOW_HARDWARE_ENABLED=1`` below), exactly like it pins providers to
        # YAML so host-specific devices cannot enter a recorded prompt.
        journey_env = {"LEAPFLOW_HARDWARE_ENABLED": "0", **(extra_env or {})}
        daemon = start_leapd(
            root=root,
            llm_base_url=proxy.base_url,
            llm_model=model,
            profile=profile,
            extra_env=journey_env,
        )
        self._stack.callback(daemon.stop)

        journey = Journey(
            journey_id=journey_id,
            proxy=proxy,
            daemon=daemon,
            deadline_s=deadline_s,
            max_llm_calls=max_llm_calls,
            max_llm_tokens=max_llm_tokens,
        )
        self._journeys.append(journey)
        return journey

    def close(self) -> None:
        """Tear down every journey started through this factory.

        Scratch roots are left in place after teardown of the *processes* only
        long enough to be removed here; the next run also clears them, so a
        crashed run cannot leak state into the next one either way.
        """
        self._stack.close()
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)
