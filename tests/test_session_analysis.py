# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic tests for the session-analysis dashboard (domain=session watch).

Fakes the analysis services facade (no LLM); exercises producer gating,
the manager path, the session.history RPC, and the session template render.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from types import SimpleNamespace

from leapflow.dashboard import TemplateLibrary
from leapflow.daemon.service import RuntimeLeapService
from leapflow.monitor import MonitorManager, ProducerRegistry, SessionAnalysisProducer, WatchSpec
from leapflow.monitor.types import ProducerContext
from leapflow.storage.connection import LocalConnectionHolder
from leapflow.storage.conversation_store import ConversationMessage, ConversationSession


class _FakeServices:
    def __init__(self, history: dict, analysis: dict, salient: bool = False) -> None:
        self._history = history
        self._analysis = analysis
        self._salient = salient
        self.analyze_calls = 0

    async def session_history(self) -> dict:
        return dict(self._history)

    async def analyze_session(self, messages, *, prior=None, artifacts=None) -> dict:
        self.analyze_calls += 1
        data = dict(self._analysis)
        if artifacts is not None:
            data["seen_artifacts"] = list(artifacts)
        return data

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


async def test_session_producer_artifact_change_refreshes_without_new_turns() -> None:
    prod = SessionAnalysisProducer()
    spec = WatchSpec(name="Session", watch_id="w-art", domain="session", params={"use_model_salience": False, "debounce_s": 0})
    base_history = {"session_id": "s", "turn_count": 2, "token_count": 10, "messages": [{"role": "u", "content": "x"}]}
    await prod.observe(_ctx(spec, _FakeServices({**base_history, "artifacts": []}, {"story": "initial"}), now=1.0))
    artifact = {"path": "/workspace/report.md", "status": "included", "mtime": 2.0, "size": 12, "content_excerpt": "new report"}
    svc = _FakeServices({**base_history, "artifacts": [artifact]}, {"story": "with artifact"})
    out = list(await prod.observe(_ctx(spec, svc, now=2.0)))
    assert len(out) == 1
    assert out[0].payload["artifact_context"][0]["content_excerpt"] == "new report"
    assert out[0].payload["observation_status"]["refresh_reason"] == "artifact_changed"
    assert out[0].payload["observation_status"]["artifacts_included"] == 1


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
    service = RuntimeLeapService(SimpleNamespace(workspace_root="."))
    service._ctx = SimpleNamespace(engine=engine, _conversation_store=None, settings=SimpleNamespace(workspace_root="."))
    history = await service.session_history()
    assert history["turn_count"] == 2
    assert history["token_count"] == 42
    assert history["session_id"] == "sid"
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]


async def test_session_history_collects_file_write_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "china_ecommerce_overseas_analysis.md"
    artifact.write_text("# China ecommerce\n\nKey findings from the research.", encoding="utf-8")

    class _Store:
        def get_messages(self, session_id, *, limit=200):
            assert session_id == "sid"
            return [ConversationMessage(
                message_id="m1",
                session_id="sid",
                role="tool",
                content='{"ok": true, "path": "china_ecommerce_overseas_analysis.md", "bytes_written": 45}',
                tool_name="file_write",
                tool_call_id="t1",
            )]

    engine = SimpleNamespace(_wm=None, _current_session_id="sid", turn_count=1, context_token_count=10)
    settings = SimpleNamespace(workspace_root=str(tmp_path))
    service = RuntimeLeapService(settings)
    service._ctx = SimpleNamespace(engine=engine, _conversation_store=_Store(), settings=settings)
    history = await service.session_history()
    assert history["artifacts"][0]["status"] == "included"
    assert history["artifacts"][0]["name"] == "china_ecommerce_overseas_analysis.md"
    assert "Key findings" in history["artifacts"][0]["content_excerpt"]


async def test_session_history_empty_without_context() -> None:
    service = RuntimeLeapService(SimpleNamespace())
    history = await service.session_history()
    assert history["turn_count"] == 0 and history["messages"] == []


async def test_session_detail_reads_persisted_transcript_with_pagination(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()

    class _Store:
        def __init__(self) -> None:
            self.session = ConversationSession(
                session_id="sid",
                title="Historical task",
                created_at=1.0,
                updated_at=2.0,
                cwd=str(workspace.resolve()),
                message_count=3,
                total_tokens=9,
                summary="done",
            )
            self.messages = [
                ConversationMessage("m1", "sid", "user", "first", created_at=1.0, token_count=3),
                ConversationMessage("m2", "sid", "assistant", "second", created_at=2.0, token_count=3),
                ConversationMessage("m3", "sid", "tool", "third", created_at=3.0, token_count=3, tool_name="file_read"),
            ]

        def get_session(self, session_id: str):
            return self.session if session_id == "sid" else None

        def get_messages(self, session_id: str, *, limit: int = 100, active_only: bool = True, offset: int = 0):
            assert session_id == "sid"
            assert active_only is False
            return self.messages[offset:offset + limit]

    settings = SimpleNamespace(workspace_root=str(workspace))
    service = RuntimeLeapService(settings)
    service._ctx = SimpleNamespace(_conversation_store=_Store(), settings=settings)

    detail = await service.session_detail("sid", limit=2, offset=1)

    assert detail["ok"] is True
    assert detail["session"]["session_id"] == "sid"
    assert detail["session"]["message_count"] == 3
    assert detail["messages"][0]["message_id"] == "m2"
    assert [m["content"] for m in detail["messages"]] == ["second", "third"]
    assert detail["has_more"] is False


async def test_session_detail_rejects_different_workspace(tmp_path: Path) -> None:
    work_a = tmp_path / "a"
    work_b = tmp_path / "b"
    work_a.mkdir()
    work_b.mkdir()

    class _Store:
        def get_session(self, session_id: str):
            return ConversationSession(session_id="sid", cwd=str(work_a.resolve()))

        def get_messages(self, session_id: str, *, limit: int = 100, active_only: bool = True, offset: int = 0):
            raise AssertionError("messages must not be read across workspace boundaries")

    settings = SimpleNamespace(workspace_root=str(work_b))
    service = RuntimeLeapService(settings)
    service._ctx = SimpleNamespace(_conversation_store=_Store(), settings=settings)

    detail = await service.session_detail("sid", workspace_root=str(work_b))

    assert detail["ok"] is False
    assert detail["code"] == "workspace_mismatch"
    assert detail["workspace_mismatch"] is True
    assert detail["expected_workspace_root"] == str(work_a.resolve())
    assert detail["requested_workspace_root"] == str(work_b.resolve())


def test_session_detail_rpc_method_is_registered() -> None:
    from leapflow.daemon.protocol import METHOD_REGISTRY

    assert METHOD_REGISTRY["session.detail"] == "session_detail"
    assert METHOD_REGISTRY["session.execution_detail"] == "session_execution_detail"


def _execution_service(tmp_path: Path):
    from leapflow.engine.tools.tool_execution import ToolExecutionRecord
    from leapflow.storage.conversation_store import DuckDBConversationStore

    store = DuckDBConversationStore(tmp_path / "execution.duckdb")
    work_a, work_b = tmp_path / "a", tmp_path / "b"
    for sid, work in (("a", work_a), ("b", work_b)):
        store.create_session(sid, cwd=str(work))
        record = ToolExecutionRecord.create(
            session_id=sid, turn_id="turn-1", command_id="", tool_call_id="call-1",
            tool_name="example", idempotency_key="a" * 64, arguments={}, policy="read_only",
        )
        store.reserve_tool_execution(record)
        store.complete_tool_execution(record.mark_completed({"body": sid * 16000}))
    settings = SimpleNamespace(workspace_root=str(work_a))
    service = RuntimeLeapService(settings)
    service._ctx = SimpleNamespace(_conversation_store=store, settings=settings)
    return service, store, work_a, work_b


async def test_session_execution_full_result_and_concurrent_workspace_isolation(tmp_path: Path) -> None:
    service, store, work_a, work_b = _execution_service(tmp_path)
    try:
        a, b, cross = await asyncio.gather(
            service.session_execution_detail("a", "a" * 64, workspace_root=str(work_a)),
            service.session_execution_detail("b", "a" * 64, workspace_root=str(work_b)),
            service.session_execution_detail("a", "a" * 64, workspace_root=str(work_b)),
        )
        assert a["execution"]["result"]["body"] == "a" * 16000
        assert b["execution"]["result"]["body"] == "b" * 16000
        assert cross == {"ok": False, "code": "workspace_mismatch"}
        assert store.get_tool_execution_by_key("a", "a" * 64).status == "completed"
    finally:
        store.close()


@pytest.mark.parametrize("sid,key,code", [
    ("", "a" * 64, "missing_session_id"),
    ("a", "invalid", "invalid_execution_key"),
    ("missing", "a" * 64, "not_found"),
    ("a", "b" * 64, "not_found"),
])
async def test_session_execution_missing_identity_has_no_fallback(tmp_path: Path, sid, key, code) -> None:
    service, store, _, _ = _execution_service(tmp_path)
    try:
        assert await service.session_execution_detail(sid, key) == {"ok": False, "code": code}
    finally:
        store.close()


async def test_session_execution_atomic_bound_and_structured_redaction(tmp_path: Path, monkeypatch) -> None:
    from leapflow.daemon import _transport

    service, store, _, _ = _execution_service(tmp_path)
    try:
        record = store.get_tool_execution_by_key("a", "a" * 64)
        secret = "sk-" + "z" * 40
        result = {"password": "short", "nested": [{"api_key": secret}], "text": secret}
        store.complete_tool_execution(record.mark_completed(result))
        value = await service.session_execution_detail("a", "a" * 64)
        assert secret not in json.dumps(value)
        assert value["execution"]["result"]["password"] == "<redacted>"
        assert value["execution"]["result"]["nested"][0]["api_key"] == "<redacted>"
        assert store.get_tool_execution_by_key("a", "a" * 64).result == result
        monkeypatch.setattr(_transport, "RPC_STREAM_LIMIT", 100)
        assert await service.session_execution_detail("a", "a" * 64) == {
            "ok": False, "code": "execution_too_large"}
    finally:
        store.close()


async def test_session_execution_foreign_record_and_unbound_workspace_fail_closed(tmp_path: Path, monkeypatch) -> None:
    service, store, _, _ = _execution_service(tmp_path)
    try:
        record = store.get_tool_execution_by_key("a", "a" * 64)
        monkeypatch.setattr(store, "get_tool_execution_by_key", lambda *args: replace(record, session_id="b"))
        assert await service.session_execution_detail("a", "a" * 64) == {
            "ok": False, "code": "execution_scope_mismatch"}
        monkeypatch.setattr(store, "get_session", lambda sid: ConversationSession(session_id=sid, cwd=""))
        assert await service.session_execution_detail("a", "a" * 64) == {
            "ok": False, "code": "workspace_unbound"}
    finally:
        store.close()


async def test_session_execution_store_failure_contains_no_exception_text(tmp_path: Path, monkeypatch) -> None:
    service, store, _, _ = _execution_service(tmp_path)
    try:
        def fail(*args):
            raise RuntimeError("credential details must stay in redacted logs")
        monkeypatch.setattr(store, "get_tool_execution_by_key", fail)
        assert await service.session_execution_detail("a", "a" * 64) == {
            "ok": False, "code": "store_error"}
    finally:
        store.close()


async def test_session_execution_client_rpc_roundtrip(tmp_path: Path) -> None:
    import tempfile

    from leapflow.daemon._transport import get_transport
    from leapflow.daemon.client import DaemonClient
    from test_daemon_rpc import _short_tempdir, _start_server

    service, store, work_a, _ = _execution_service(tmp_path)
    with tempfile.TemporaryDirectory(prefix="lf-exec-", dir=_short_tempdir()) as root:
        server, task, runtime = await _start_server(Path(root), service)
        try:
            client = DaemonClient(get_transport().readiness_path(runtime))
            value = await client.session_execution_detail("a", "a" * 64, workspace_root=str(work_a))
            assert value["execution"]["result"]["body"] == "a" * 16000
        finally:
            task.cancel()
            await server.stop()
            await asyncio.gather(task, return_exceptions=True)
            store.close()


def test_session_tool_workspace_prefers_current_tool_context(tmp_path: Path) -> None:
    from leapflow.cli.context import _active_tool_workspace_root
    from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context

    fallback = tmp_path / "daemon-start-workspace"
    active = tmp_path / "active-tui-workspace"
    fallback.mkdir()
    active.mkdir()
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(active)))
    try:
        resolved = _active_tool_workspace_root(str(fallback))
    finally:
        reset_tool_context(token)

    assert resolved == str(active.resolve())
    assert _active_tool_workspace_root(str(fallback)) == str(fallback)


# ── session template render ──────────────────────────────────────────────────


def test_session_template_renders_analysis() -> None:
    lib = TemplateLibrary()
    assert "generic" in lib.names()
    spec = lib.render("generic", {"analysis": {
        "story": "the arc",
        "insights": [{"title": "t", "summary": "s", "severity": "notable"}],
        "action_items": ["do x"],
        "decisions": ["chose y"],
        "open_questions": [],
        "entities": ["Alice"],
        "next_prompts": ["ask z"],
    }, "observation": {
        "refresh_reason": "artifact_changed",
        "context_scope": "text_and_artifacts",
        "artifacts_included": 1,
        "artifact_count": 1,
    }, "artifact_context": [{"name": "report.md", "status": "included", "reason": ""}]})
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for n in nodes:
            flat.append(n)
            _walk(n.get("children") or [])

    _walk(spec["root"])
    types = {n["type"] for n in flat}
    assert "StoryPanel" in types
    # The severity BarChart is the primary in-view visualization; the artifact
    # Table appears only because artifact_context is non-empty here.
    assert "BarChart" in types
    assert "Table" in types
    assert len([n for n in flat if n["type"] == "InsightCard"]) == 1
