"""Unit tests for the S2 re-entry service (phases N3b–N5): time + event dispatch,
global-budget backstop, disabled gating, and the N5 no-external-send guarantee.

Hermetic: DuckDB store on a temp file + a stub subagent manager; no daemon,
no gateway, no network.
"""
from __future__ import annotations

import asyncio

from leapflow.scheduler.reentry_driver import build_reentry_subagent_config, event_matches
from leapflow.scheduler.reentry_send import SendGovernor, SendRateLimiter, SendTarget
from leapflow.scheduler.reentry_service import ReentryService
from leapflow.security.send_trust import SendTrustLedger, SendTrustLevel
from leapflow.storage.reentry_store import ReentryStore, build_reentry_trigger


class _StubManager:
    def __init__(self) -> None:
        self.configs: list = []

    async def delegate(self, config):
        self.configs.append(config)
        return type("R", (), {"summary": "done", "status": "completed"})()


class _Settings:
    def __init__(self, enabled: bool = True) -> None:
        self.agent_reentry_enabled = enabled


def test_event_matches_pure() -> None:
    assert event_matches({}, platform="feishu", chat="c1", text="deploy done") is False  # empty => no match
    assert event_matches({"platform": "feishu"}, platform="feishu", chat="c1", text="x") is True
    assert event_matches({"platform": "feishu"}, platform="dingtalk", chat="c1", text="x") is False
    assert event_matches({"keyword": "deploy"}, platform="p", chat="c", text="Deploy DONE") is True  # ci substring
    assert event_matches({"keyword": "deploy"}, platform="p", chat="c", text="build ok") is False
    assert event_matches({"chat": "c1"}, platform="p", chat="c2", text="x") is False
    assert event_matches({"platform": "feishu", "keyword": "deploy"}, platform="feishu", chat="c", text="deploy") is True


def test_service_time_tick_dispatches(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    mgr = _StubManager()
    events: list[str] = []
    svc = ReentryService(store=store, manager=mgr, settings=_Settings(True),
                         notify=lambda et, **kw: events.append(et))
    try:
        store.save(build_reentry_trigger(
            task_id="t1", kind="time", delay_seconds=10, now=0.0,
            task_contract={"original_request": "go"},
        ))
        assert asyncio.run(svc.tick(now=100.0)) == 1
        assert len(mgr.configs) == 1 and mgr.configs[0].goal == "go"
        assert "reentry.dispatched" in events and "reentry.completed" in events   # N5 audit
    finally:
        store.close()


def test_service_event_match_dispatches(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    mgr = _StubManager()
    svc = ReentryService(store=store, manager=mgr, settings=_Settings(True))
    try:
        store.save(build_reentry_trigger(
            task_id="e1", kind="event",
            event_match={"platform": "feishu", "keyword": "deploy"},
            task_contract={"original_request": "resume"},
        ))
        assert asyncio.run(svc.on_gateway_message(platform="feishu", chat="c", text="hello")) == 0  # no match
        assert len(mgr.configs) == 0
        assert asyncio.run(svc.on_gateway_message(platform="feishu", chat="c", text="deploy done")) == 1
        assert len(mgr.configs) == 1 and mgr.configs[0].goal == "resume"
    finally:
        store.close()


def test_service_disabled_no_dispatch(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    mgr = _StubManager()
    svc = ReentryService(store=store, manager=mgr, settings=_Settings(False))   # disabled
    try:
        store.save(build_reentry_trigger(task_id="e1", kind="event", event_match={"keyword": "x"}))
        store.save(build_reentry_trigger(task_id="t1", kind="time", delay_seconds=1, now=0.0))
        assert asyncio.run(svc.on_gateway_message(platform="p", chat="c", text="x")) == 0
        assert asyncio.run(svc.tick(now=100.0)) == 0
        assert len(mgr.configs) == 0
    finally:
        store.close()


def test_service_global_budget_backstop(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    mgr = _StubManager()
    svc = ReentryService(store=store, manager=mgr, settings=_Settings(True), global_budget=1)
    try:
        for i in range(3):
            store.save(build_reentry_trigger(task_id=f"t{i}", kind="time", delay_seconds=1, now=0.0))
        asyncio.run(svc.tick(now=100.0))
        assert len(mgr.configs) == 1   # global budget backstops runaway across triggers
    finally:
        store.close()


def test_reentry_subagent_config_blocks_external_send() -> None:
    from leapflow.storage.reentry_store import OrientSnapshot

    cfg = build_reentry_subagent_config(
        OrientSnapshot(task_id="t1", task_contract={"original_request": "x"})
    )
    # N5: an autonomous re-entry runs as an isolated subagent that cannot send
    # messages externally without approval (send_message is blocked).
    assert "send_message" in cfg.blocked_tools


# ── SO3: governed proactive delivery (default-off; opt-in via wiring) ──


def _send_governor(*, enabled: bool = True, verified_at: int = 3) -> SendGovernor:
    return SendGovernor(
        trust=SendTrustLedger(verified_at=verified_at),
        rate=SendRateLimiter(per_hour=10),
        enabled=enabled,
        global_budget=50,
    )


def _save_event_trigger(store) -> None:
    store.save(build_reentry_trigger(
        task_id="e1", kind="event",
        event_match={"platform": "feishu", "chat": "c1", "keyword": "deploy"},
        task_contract={"original_request": "resume"},
    ))


def test_so3_auto_allow_sends_when_verified(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    sent: list = []

    async def send_fn(platform, chat, text):
        sent.append((platform, chat, text))
        return {"ok": True, "message_id": "m1"}

    gov = _send_governor(verified_at=1)
    gov.record_human_allow(SendTarget("feishu", "c1").grant_key("reply"))  # -> VERIFIED
    svc = ReentryService(store=store, manager=_StubManager(), settings=_Settings(True),
                         send_governor=gov, send_fn=send_fn, request_approval=None)
    try:
        _save_event_trigger(store)
        assert asyncio.run(svc.on_gateway_message(platform="feishu", chat="c1", text="deploy done")) == 1
        assert sent == [("feishu", "c1", "done")]   # trust VERIFIED -> auto-sent
    finally:
        store.close()


def test_so3_needs_approval_allow_sends_and_accrues_trust(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    sent: list = []
    approvals: list = []

    async def send_fn(platform, chat, text):
        sent.append((platform, chat, text))
        return {"ok": True}

    async def request_approval(req):
        approvals.append(req)
        return "allow"

    gov = _send_governor(verified_at=3)  # no trust yet -> needs approval
    grant = SendTarget("feishu", "c1").grant_key("reply")
    svc = ReentryService(store=store, manager=_StubManager(), settings=_Settings(True),
                         send_governor=gov, send_fn=send_fn, request_approval=request_approval)
    try:
        _save_event_trigger(store)
        assert asyncio.run(svc.on_gateway_message(platform="feishu", chat="c1", text="deploy done")) == 1
        assert len(approvals) == 1                      # queued for human
        assert sent == [("feishu", "c1", "done")]       # approved -> sent
        assert gov._trust.level(grant) == SendTrustLevel.CANDIDATE   # trust accrued
    finally:
        store.close()


def test_so3_needs_approval_deny_does_not_send(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    sent: list = []

    async def send_fn(platform, chat, text):
        sent.append((platform, chat, text))
        return {"ok": True}

    async def request_approval(req):
        return "deny"

    gov = _send_governor(verified_at=3)
    grant = SendTarget("feishu", "c1").grant_key("reply")
    gov.record_human_allow(grant)  # CANDIDATE, then a deny should freeze it
    svc = ReentryService(store=store, manager=_StubManager(), settings=_Settings(True),
                         send_governor=gov, send_fn=send_fn, request_approval=request_approval)
    try:
        _save_event_trigger(store)
        assert asyncio.run(svc.on_gateway_message(platform="feishu", chat="c1", text="deploy done")) == 1
        assert sent == []                                # denied -> not sent
        assert gov._trust.level(grant) == SendTrustLevel.DRAFT   # frozen by deny
    finally:
        store.close()


def test_so3_disabled_governor_does_not_send(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    sent: list = []

    async def send_fn(platform, chat, text):
        sent.append((platform, chat, text))
        return {"ok": True}

    gov = _send_governor(enabled=False)              # governor off => BLOCKED(disabled)
    svc = ReentryService(store=store, manager=_StubManager(), settings=_Settings(True),
                         send_governor=gov, send_fn=send_fn, request_approval=None)
    try:
        _save_event_trigger(store)
        assert asyncio.run(svc.on_gateway_message(platform="feishu", chat="c1", text="deploy done")) == 1
        assert sent == []                                # dispatched, but never sent
    finally:
        store.close()
