# Copyright (c) Alibaba, Inc. and its affiliates.
"""Provider hardware binding with real providers, registry, evidence and approvals."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import replace
from functools import partial

import httpx
import pytest

from leapflow.hardware.evidence import EvidenceStore
from leapflow.hardware.model_contract import ModelBackend, ModelInfo
from leapflow.hardware.model_qualification import ManagedModelQualification, deployment_action, qualification_binding
from leapflow.hardware.registry import HardwareRegistry
from leapflow.hardware.transport import TransportError
from leapflow.hardware.transports.model_service import build_transport
from leapflow.hardware.verification import EvidenceBundle
from leapflow.llm.provider_registry import LLMProviderRegistry
from leapflow.llm.services.config import parse_services
from leapflow.llm.services.http_client import BoundedTransport
from leapflow.llm.services.runtime import ModelServiceRuntime
from leapflow.security.grants import ApprovalAuditLog
from leapflow.security.orchestrator import ApprovalOrchestrator
from test_model_services import declaration, recorded_response
from test_model_service_transport import _context
from test_model_tools import Human


async def compose(tmp_path, *, policy=None, receipt=True, receipt_fields=None,
                  evidence_device="model_qualification", binding_changes=None):
    task = "grounding" if policy is None else "action_chunk"
    info = (ModelInfo(task, "fixture", "1", "sha256:fixture", "instance-1") if policy is None else
            replace(policy.model_info, checksum="sha256:fixture", instance="instance-1"))
    item = declaration(task=task, processing_location="local",
                       identity={"mode": "server_reported"}, model_info=info.to_dict(),
                       deployment={"engine_version": "fixture-1", "adapter_digest": "sha256:adapter",
                                   "qualification_ref": "qualification-1"})
    spec = parse_services({"native": item})[0]
    store = EvidenceStore(tmp_path / "instrument.duckdb")
    audit = ApprovalAuditLog(tmp_path / "approvals.jsonl")
    gate = ApprovalOrchestrator(Human(), audit=audit)
    action = deployment_action(qualification_binding(spec, info))
    approval = await gate.evaluate(action)
    assert approval.user_consent
    record = {"schema_version": 1, "binding": qualification_binding(spec, info),
              "expires_at": time.time() + 120, "revoked": False, "identity_insertion": "runtime",
              "native_parity": True, "approval_action_id": action.action_id, **(receipt_fields or {})}
    record["binding"].update(binding_changes or {})
    if receipt:
        await store.record(EvidenceBundle("qualification-1", evidence_device, "native", time.time(),
                                         None, None, (), metadata={"model_qualification": record}))
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    state = {"info": info.to_dict(), "echo": info.identity.to_dict(), "delay": 0, "requests": []}

    async def respond(request):
        body = json.loads(request.content)
        text = body["messages"][-1]["content"][0]["text"]
        envelope = json.loads(text)
        state["requests"].append(body)
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        if envelope["operation"] == "infer" and state.get("infer_hook"):
            await state["infer_hook"]()
        if envelope["operation"] == "describe":
            payload = {"model_info": state["info"]}
        elif policy is None:
            payload = {"detections": [], "infer_ms": 1, "model": state["echo"]}
        else:
            rows = policy._chunk(envelope["input"]["state"])
            payload = {"actions": rows, "infer_ms": 1, "model": state["echo"]}
        return httpx.Response(200, json=recorded_response(payload))

    owner = ModelServiceRuntime(providers, (spec,), client_factory=lambda spec: httpx.AsyncClient(
        transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    verifier = ManagedModelQualification(lambda: store, ApprovalAuditLog(tmp_path / "approvals.jsonl"))
    factory = partial(build_transport, model_services=owner, qualification=verifier)
    registry = HardwareRegistry(transport_factories={"model_service": factory})
    config = {"backend": "provider", "service": "native", "task": task}
    context = replace(_context(config), device_id="native_model")
    registry.register_external_contexts("fixture", (context,))
    registry.load()
    registry.scope.async_effect(owner.aclose)
    registry.scope.effect(store.close)
    return registry, owner, state, verifier, record


async def test_provider_factory_pins_generation_and_canonical_images(tmp_path):
    registry, owner, state, _, _ = await compose(tmp_path)
    try:
        transport = await registry.transport("native_model")
        assert isinstance(transport._backend, ModelBackend)
        basis = transport.model_info.details["service_basis"]
        assert basis["generation"] == 1 and basis["qualification_ref"] == "qualification-1"
        original = owner._specs["native"]
        updated = parse_services({"native": {**original.declaration(), "base_url": "http://127.0.0.1:9099/v1"}})
        assert owner.publish(updated)
        assert transport._backend._handle.spec is original
        assert owner.describe("native")["generation"] > basis["generation"]
        result = await transport.infer({"image": base64.b64encode(b"\xff\xd8fixture").decode(), "queries": ["cup"]})
        assert result.detections == ()
        content = state["requests"][-1]["messages"][-1]["content"]
        envelope = json.loads(content[0]["text"])
        assert envelope == {"contract": "leapflow.model.v1.1", "operation": "infer", "task": "grounding",
                            "input": {"queries": ["cup"]}, "image_order": ["image"]}
        assert len(content) == 2 and content[1]["type"] == "image_url"
        assert not (await transport.write("model.info", {})).ok
        assert (await transport.halt()).stop_effect == "none"
    finally:
        await registry.close_all()
    assert not owner._handles


def revoke_service(owner, change):
    original = owner._specs["native"]
    if change == "remove":
        owner.publish(())
    else:
        owner.publish(parse_services({"native": {**original.declaration(), "enabled": False}}))
    if change == "reenable":
        owner.publish((original,))
    return "model_capability_removed" if change == "remove" else "model_service_disabled"


@pytest.mark.parametrize("change", ["disable", "remove", "reenable"])
async def test_catalog_revocation_refuses_pinned_hardware_handle_without_io(tmp_path, change):
    registry, owner, state, _, _ = await compose(tmp_path)
    try:
        transport = await registry.transport("native_model")
        admitted_calls = len(state["requests"])
        expected = revoke_service(owner, change)
        with pytest.raises(TransportError) as refused:
            await transport.infer({"image": base64.b64encode(b"\xff\xd8fixture").decode(), "queries": ["cup"]})
        assert refused.value.failure_code == expected
        assert len(state["requests"]) == admitted_calls
    finally:
        await registry.close_all()
    assert not owner._handles


@pytest.mark.parametrize("change", ["disable", "remove", "reenable"])
async def test_catalog_revocation_during_native_reply_discards_result(tmp_path, change):
    registry, owner, state, _, _ = await compose(tmp_path)
    try:
        transport = await registry.transport("native_model")
        admitted_calls = len(state["requests"])

        async def revoke():
            revoke_service(owner, change)

        state["infer_hook"] = revoke
        with pytest.raises(TransportError) as refused:
            await transport.infer({"image": base64.b64encode(b"\xff\xd8fixture").decode(), "queries": ["cup"]})
        assert refused.value.failure_code == ("model_capability_removed" if change == "remove" else "model_service_disabled")
        assert len(state["requests"]) == admitted_calls + 2
    finally:
        await registry.close_all()
    assert not owner._handles


@pytest.mark.parametrize("phase", ["describe", "result"])
@pytest.mark.parametrize("change", ["disable", "remove", "reenable"])
async def test_catalog_revocation_during_qualification_discards_result(tmp_path, monkeypatch, phase, change):
    registry, owner, state, verifier, _ = await compose(tmp_path)
    try:
        transport = await registry.transport("native_model")
        store = verifier._evidence_store()
        read = store.get
        reads = 0
        admitted_calls = len(state["requests"])

        async def revoke_after_read(receipt_id):
            nonlocal reads
            record = await read(receipt_id)
            reads += 1
            if reads == (1 if phase == "describe" else 2):
                revoke_service(owner, change)
            return record

        monkeypatch.setattr(store, "get", revoke_after_read)
        with pytest.raises(TransportError) as refused:
            await transport.infer({"image": base64.b64encode(b"\xff\xd8fixture").decode(), "queries": ["cup"]})
        assert refused.value.failure_code == ("model_capability_removed" if change == "remove" else "model_service_disabled")
        assert len(state["requests"]) == admitted_calls + (1 if phase == "describe" else 2)
    finally:
        await registry.close_all()
    assert not owner._handles


@pytest.mark.parametrize("change", ["restart", "echo", "manifest", "capability"])
async def test_identity_metadata_and_capability_loss_refuse(tmp_path, change):
    registry, owner, state, _, _ = await compose(tmp_path)
    try:
        transport = await registry.transport("native_model")
        if change == "restart":
            state["info"]["instance"] = "instance-2"
        elif change == "echo":
            state["echo"]["instance"] = "instance-2"
        elif change == "manifest":
            state["info"]["normalization"] = {"changed": True}
        else:
            owner._registry.unregister("openai")
        with pytest.raises(TransportError):
            await transport.infer({"image": base64.b64encode(b"\xff\xd8fixture").decode(), "queries": ["cup"]})
    finally:
        await registry.close_all()


async def test_missing_evidence_and_missing_injection_refuse(tmp_path):
    registry, owner, _, _, _ = await compose(tmp_path, receipt=False)
    try:
        with pytest.raises(TransportError, match="receipt"):
            await registry.transport("native_model")
        transport = build_transport({"backend": "provider", "service": "native", "task": "grounding"})
        with pytest.raises(TransportError, match="owning service"):
            await transport.open(_context({}))
    finally:
        await registry.close_all()
    assert not owner._handles


@pytest.mark.parametrize("field", ["base_url", "checkpoint_checksum", "native_metadata_digest", "adapter_digest", "service_spec_digest"])
async def test_receipt_binding_mismatch_refuses(tmp_path, field):
    registry, owner, _, _, _ = await compose(tmp_path, binding_changes={field: "different"})
    try:
        with pytest.raises(TransportError, match="does not bind"):
            await registry.transport("native_model")
    finally:
        await registry.close_all()


@pytest.mark.parametrize("fields", [{"revoked": True}, {"expires_at": 0}, {"expires_at": True}])
async def test_invalid_qualification_refuses_at_admission(tmp_path, fields):
    registry, owner, _, _, _ = await compose(tmp_path, receipt_fields=fields)
    try:
        with pytest.raises(TransportError, match="expired, revoked"):
            await registry.transport("native_model")
        assert not owner._handles
    finally:
        await registry.close_all()


async def test_qualification_receipt_cannot_come_from_unrelated_evidence(tmp_path):
    registry, owner, _, _, _ = await compose(tmp_path, evidence_device="camera")
    try:
        with pytest.raises(TransportError, match="evidence source"):
            await registry.transport("native_model")
        assert not owner._handles
    finally:
        await registry.close_all()


@pytest.mark.parametrize("change", ["expiry", "invalidation"])
async def test_qualification_loss_during_inference_discards_result(tmp_path, change):
    expiry = time.time() + .5
    fields = {"expires_at": expiry} if change == "expiry" else {}
    registry, owner, state, verifier, receipt = await compose(tmp_path, receipt_fields=fields)
    try:
        transport = await registry.transport("native_model")
        admitted_calls = len(state["requests"])

        async def invalidate():
            if change == "expiry":
                await asyncio.sleep(max(0, expiry - time.time()) + .01)
            else:
                # Append-only evidence cannot silently overwrite the pinned receipt.
                # A second record invalidates resolution rather than selecting either.
                await verifier._evidence_store().record(EvidenceBundle(
                    "qualification-1", "model_qualification", "native", time.time(), None, None, (),
                    metadata={"model_qualification": {**receipt, "revoked": True}}))

        state["infer_hook"] = invalidate
        with pytest.raises(TransportError) as refused:
            await transport.infer({"image": base64.b64encode(b"\xff\xd8fixture").decode(), "queries": ["cup"]})
        assert refused.value.failure_code == "model_unqualified"
        operations = [json.loads(r["messages"][-1]["content"][0]["text"])["operation"]
                      for r in state["requests"][admitted_calls:]]
        assert operations == ["describe", "infer"]  # Qualification recheck has no provider I/O.
    finally:
        await registry.close_all()
    assert not owner._handles


def test_declaration_changes_require_new_qualification():
    info = ModelInfo("grounding", "fixture", "1", "sha256:fixture", "instance-1")
    item = declaration(task="grounding", processing_location="local", identity={"mode": "server_reported"},
                       model_info=info.to_dict(), deployment={"engine_version": "fixture-1",
                       "adapter_digest": "sha256:adapter", "qualification_ref": "qualification-1"})
    spec = parse_services({"native": item})[0]
    baseline = qualification_binding(spec, info)
    for change in ({"output": "tool_call"}, {"request_options": {"temperature": .5}},
                   {"limits": {"deadline_s": 3}}, {"codec": "json.v2"}):
        updated = parse_services({"native": {**item, **change}})[0]
        assert qualification_binding(updated, info) != baseline


async def test_standard_provider_drives_existing_policy_runtime(tmp_path):
    from tests._harness.policy_cell import commanded, execute_run, plan_run, policy_cell, EE
    from tests._harness.skill_stubs import SimPolicy, action_space

    native = SimPolicy(action_space(gripper={}), (0, 0, 1, 0, 0, 0, 0))
    registry, owner, state, _, _ = await compose(tmp_path, policy=native)
    runtime = None
    try:
        transport = await registry.transport("native_model")
        runtime = policy_cell(tmp_path / "cell", native, model_service=transport)
        plan = await plan_run(runtime)
        outcome = await execute_run(runtime, plan)
        assert outcome.ok, outcome
        assert commanded(runtime._registry.sim, EE[2])
        assert transport.model_info.details["service_basis"]["generation"] == 1
        inference = next(json.loads(r["messages"][-1]["content"][0]["text"])
                         for r in state["requests"] if '"operation":"infer"' in r["messages"][-1]["content"][0]["text"])
        assert inference["task"] == "action_chunk" and inference["image_order"] == ["top"]
        assert "images" not in inference["input"]
    finally:
        if runtime is not None:
            await runtime.close()
        await registry.close_all()
