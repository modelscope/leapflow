"""Hermetic tests for the session-analysis dashboard (domain=session watch).

Fakes the analysis services facade (no LLM); exercises producer gating,
the manager path, the session.history RPC, and the session template render.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from leapflow.dashboard import TemplateLibrary
from leapflow.daemon.service import RuntimeLeapService
from leapflow.monitor import MonitorManager, ProducerRegistry, SessionAnalysisProducer, WatchSpec
from leapflow.monitor.types import ProducerContext
from leapflow.storage.connection import LocalConnectionHolder


class _FakeServices:
    def __init__(self, history: dict, analysis: dict, salient: bool = False) -> None:
        self._history = history
        self._analysis = analysis
        self._salient = salient
        self.analyze_calls = 0

    async def session_history(self) -> dict:
        return dict(self._history)

    async def analyze_session(self, messages, *, prior=None) -> dict:
        self.analyze_calls += 1
        return dict(self._analysis)

    async def should_refresh(self, messages) -> bool:
        return self._salient


def _ctx(spec: WatchSpec, services, *, now: float = 1000.0, force: bool = False) -> ProducerContext:
    return ProducerContext(spec=spec, now=now, services=services, force=force)


# ── Producer gating ─────────────────────────────────────────────────────────


async def test_session_producer_first_run_analyzes() -> None:
    prod = SessionAnalysisProducer()
    svc = _FakeServices(
        {"turn_count": 2, "token_count": 100, "messages": [{"role": "user", "content": "hi"}]},
        {"story": "The arc", "insights": [{"title": "a"}]},
    )
    spec = WatchSpec(name="Session", watch_id="w", domain="session", params={"use_model_salience": False, "debounce_s": 0})
    out = list(await prod.observe(_ctx(spec, svc)))
    assert len(out) == 1
    assert out[0].payload["story"] == "The arc"
    assert out[0].payload["usage"]["turns"] == 2
    assert svc.analyze_calls == 1

    # No new turns since last analysis -> skip.
    again = list(await prod.observe(_ctx(spec, svc, now=1001.0)))
    assert again == []
    assert svc.analyze_calls == 1


async def test_session_producer_batch_threshold() -> None:
    prod = SessionAnalysisProducer()
    spec = WatchSpec(name="Session", watch_id="w", domain="session", params={"batch_turns": 6, "use_model_salience": False, "debounce_s": 0})
    await prod.observe(_ctx(spec, _FakeServices({"turn_count": 2, "token_count": 10, "messages": [{"role": "u", "content": "x"}]}, {"story": "s"})))
    svc = _FakeServices({"turn_count": 8, "token_count": 20, "messages": [{"role": "u", "content": "x"}]}, {"story": "s2"})
    out = list(await prod.observe(_ctx(spec, svc, now=2000.0)))
    assert len(out) == 1 and svc.analyze_calls == 1


async def test_session_producer_salience_below_batch() -> None:
    prod = SessionAnalysisProducer()
    spec = WatchSpec(name="Session", watch_id="w2", domain="session", params={"batch_turns": 100, "batch_tokens": 10**9, "use_model_salience": True, "debounce_s": 0})
    await prod.observe(_ctx(spec, _FakeServices({"turn_count": 1, "token_count": 5, "messages": [{"role": "u", "content": "x"}]}, {"story": "s"}, salient=True), now=1.0))
    svc = _FakeServices({"turn_count": 2, "token_count": 8, "messages": [{"role": "u", "content": "y"}]}, {"story": "s"}, salient=True)
    out = list(await prod.observe(_ctx(spec, svc, now=100.0)))
    assert len(out) == 1  # salience forced a refresh below the batch threshold


async def test_session_producer_force_without_new_turns() -> None:
    prod = SessionAnalysisProducer()
    spec = WatchSpec(name="Session", watch_id="w3", domain="session", params={"use_model_salience": False, "debounce_s": 999})
    await prod.observe(_ctx(spec, _FakeServices({"turn_count": 2, "token_count": 10, "messages": [{"role": "u", "content": "x"}]}, {"story": "s"})))
    svc = _FakeServices({"turn_count": 2, "token_count": 10, "messages": [{"role": "u", "content": "x"}]}, {"story": "forced"})
    out = list(await prod.observe(_ctx(spec, svc, now=5.0, force=True)))
    assert len(out) == 1 and out[0].payload["story"] == "forced"


async def test_session_dedup_key_is_session_scoped() -> None:
    prod = SessionAnalysisProducer()
    spec = WatchSpec(name="Session", watch_id="w", domain="session", params={"use_model_salience": False, "debounce_s": 0})
    a = list(await prod.observe(_ctx(spec, _FakeServices(
        {"session_id": "A", "turn_count": 2, "token_count": 10, "messages": [{"role": "u", "content": "x"}]},
        {"story": "a"}), now=1.0)))
    # A different session with the SAME turn_count must not be deduped away.
    b = list(await prod.observe(_ctx(spec, _FakeServices(
        {"session_id": "B", "turn_count": 2, "token_count": 10, "messages": [{"role": "u", "content": "y"}]},
        {"story": "b"}), now=2.0, force=True)))
    assert a[0].dedup_key == "w:A:2"
    assert b[0].dedup_key == "w:B:2"
    assert a[0].dedup_key != b[0].dedup_key


# ── Full manager path ────────────────────────────────────────────────────────


async def test_session_watch_via_manager(tmp_path: Path) -> None:
    producers = ProducerRegistry()
    producers.register(SessionAnalysisProducer())
    svc = _FakeServices(
        {"turn_count": 3, "token_count": 50, "messages": [{"role": "user", "content": "hi"}]},
        {"story": "Story", "insights": [{"title": "i", "summary": "s", "severity": "notable"}]},
    )
    mgr = MonitorManager(
        holder=LocalConnectionHolder(tmp_path / "leap.duckdb"),
        producers=producers,
        services=svc,
    )
    view = await mgr.arm_watch(WatchSpec(name="Session", domain="session", params={"use_model_salience": False, "debounce_s": 0}))
    result = await mgr.run_watch_once(view.watch_id, force=True)
    assert result["ok"] is True and result["findings"] == 1
    findings = mgr.list_findings(watch_id=view.watch_id)
    assert findings[0].payload["story"] == "Story"


# ── session.history RPC ──────────────────────────────────────────────────────


async def test_session_history_reads_engine_transcript() -> None:
    class _WM:
        def as_chat_messages(self):
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    engine = SimpleNamespace(_wm=_WM(), _current_session_id="sid", turn_count=2, context_token_count=42)
    service = RuntimeLeapService(SimpleNamespace())
    service._ctx = SimpleNamespace(engine=engine, _conversation_store=None)
    history = await service.session_history()
    assert history["turn_count"] == 2
    assert history["token_count"] == 42
    assert history["session_id"] == "sid"
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]


async def test_session_history_empty_without_context() -> None:
    service = RuntimeLeapService(SimpleNamespace())
    history = await service.session_history()
    assert history["turn_count"] == 0 and history["messages"] == []


# ── session template render ──────────────────────────────────────────────────


def test_session_template_renders_analysis() -> None:
    lib = TemplateLibrary()
    assert "session.analysis" in lib.names()
    spec = lib.render("session.analysis", {"analysis": {
        "story": "the arc",
        "insights": [{"title": "t", "summary": "s", "severity": "notable"}],
        "action_items": ["do x"],
        "decisions": ["chose y"],
        "open_questions": [],
        "entities": ["Alice"],
        "next_prompts": ["ask z"],
    }})
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for n in nodes:
            flat.append(n)
            _walk(n.get("children") or [])

    _walk(spec["root"])
    types = {n["type"] for n in flat}
    assert "StoryPanel" in types
    assert len([n for n in flat if n["type"] == "InsightCard"]) == 1
