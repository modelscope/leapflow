"""Node-local passive CAN evidence over a synthetic iproute2/decoder boundary."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from piper_ops import health
from piper_ops.config import Settings
from piper_ops.evidence import Refusal
from tests.test_piper_ops import context, snapshot


def link(packets=100):
    return [
        {
            "ifname": "can0",
            "flags": ["UP", "LOWER_UP"],
            "linkinfo": {
                "info_kind": "can",
                "info_data": {
                    "state": "ERROR-ACTIVE",
                    "bittiming": {"bitrate": 1_000_000},
                    "berr_counter": {"rx": 0, "tx": 0},
                },
                "info_xstats": {
                    "restarts": 0,
                    "bus_error": 0,
                    "arbitration_lost": 0,
                    "error_warning": 0,
                    "error_passive": 0,
                    "bus_off": 0,
                },
            },
            "stats64": {d: {"packets": packets, "errors": 0, "dropped": 0} for d in ("rx", "tx")},
        }
    ]


def command_boundary(monkeypatch, bodies):
    calls = []
    rows = iter(bodies)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        body = next(rows)
        if isinstance(body, Exception):
            raise body
        return subprocess.CompletedProcess(command, 0, json.dumps(body), "")

    monkeypatch.setattr(health.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("shape", ["stats64", "legacy", "gs_usb"])
def test_kernel_link_parser_accepts_both_counter_shapes(monkeypatch, shape):
    body = link()
    if shape == "legacy":
        body[0]["stats"] = body[0].pop("stats64")
    elif shape == "gs_usb":
        # gs_usb exports no TEC/REC counter; kernel CAN error statistics remain.
        body[0]["linkinfo"]["info_data"].pop("berr_counter")
    calls = command_boundary(monkeypatch, [body])
    sample = health._link_sample("can0", 2)
    assert sample["counts"]["rx_packets"] == 100
    assert sample["counts"]["can_bus_error"] == 0
    assert sample["bus_error_counters"] == (None if shape == "gs_usb" else {"rx": 0, "tx": 0})
    assert calls == [
        (
            ["ip", "-j", "-s", "-d", "link", "show", "dev", "can0"],
            {"capture_output": True, "text": True, "timeout": 2, "check": False},
        )
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "null",
        "row",
        "duplicate",
        "identity",
        "kind",
        "down",
        "state",
        "bitrate",
        "bus_error",
        "info",
        "timing",
        "stats",
        "direction",
        "missing",
        "negative",
        "fraction",
        "boolean",
        "nan",
        "berr",
        "xstats",
        "xstats_key",
    ],
)
def test_kernel_link_parser_fails_closed(monkeypatch, fault):
    body = link()
    row = body[0]
    data = row["linkinfo"]["info_data"]
    if fault == "null":
        body = None
    elif fault == "row":
        body = [None]
    elif fault == "duplicate":
        body.append(copy.deepcopy(row))
    elif fault == "identity":
        row["ifname"] = "can1"
    elif fault == "kind":
        row["linkinfo"]["info_kind"] = "ether"
    elif fault == "down":
        row["flags"] = []
    elif fault == "state":
        data["state"] = "ERROR-WARNING"
    elif fault == "bitrate":
        data["bittiming"]["bitrate"] = 500_000
    elif fault == "bus_error":
        data["berr_counter"]["rx"] = 1
    elif fault == "info":
        row["linkinfo"] = None
    elif fault == "timing":
        data["bittiming"] = {}
    elif fault == "stats":
        row["stats64"] = None
    elif fault == "direction":
        row["stats64"]["rx"] = None
    elif fault == "missing":
        row["stats64"]["rx"].pop("packets")
    elif fault == "berr":
        data["berr_counter"] = [0, 0]
    elif fault == "xstats":
        row["linkinfo"].pop("info_xstats")
    elif fault == "xstats_key":
        row["linkinfo"]["info_xstats"].pop("bus_off")
    else:
        row["stats64"]["rx"]["packets"] = {
            "negative": -1,
            "fraction": 0.5,
            "boolean": True,
            "nan": float("nan"),
        }[fault]
    command_boundary(monkeypatch, [body])
    with pytest.raises(Refusal):
        health._link_sample("can0", 2)


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "frozen_decoder",
        "packets",
        "reset",
        "errors",
        "dropped",
        *health.CAN_ERROR_STATS,
        "timeout",
        "remote",
        "platform",
        "interface",
        "unbound",
        "undeclared",
    ],
)
async def test_passive_can_requires_two_advancing_local_samples(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(health.sys, "platform", "darwin" if fault == "platform" else "linux")
    settings = replace(Settings(), poll_s=0.001, node_devices=tmp_path)
    local = context()
    # The live owner's listing withholds the node-private binding.
    declared = replace(local, transport=replace(local.transport, config={}))
    first, last = link(), link(101)
    before, after = snapshot(seq=1), snapshot(seq=2)
    if fault == "frozen_decoder":
        after = before
    elif fault == "packets":
        last = first
    elif fault == "reset":
        last = link(1)
    elif fault in ("errors", "dropped"):
        last[0]["stats64"]["rx"][fault] = 1
    elif fault in health.CAN_ERROR_STATS:
        last[0]["linkinfo"]["info_xstats"][fault] = 1
    elif fault == "timeout":
        first = subprocess.TimeoutExpired("ip", 2)
    elif fault == "remote":
        settings = replace(settings, endpoint="tcp://192.0.2.1:8767")
    elif fault == "interface":
        local = replace(local, transport=replace(local.transport, config={"can_name": "bad/name"}))
    elif fault == "unbound":
        declared = replace(declared, display_name="another arm")
    if fault != "undeclared":
        (tmp_path / "piper_arm.yaml").write_text(yaml.safe_dump(local.to_dict()))
    calls = command_boundary(monkeypatch, [first, last])
    session = SimpleNamespace(
        settings=settings, context=declared, snapshot=AsyncMock(side_effect=[before, after])
    )
    result = await health.passive_can(session)
    assert result["qualified"] is (fault == "none")
    if fault in ("remote", "platform", "interface", "unbound", "undeclared"):
        assert not calls
        session.snapshot.assert_not_awaited()
    elif fault == "none":
        assert len(calls) == 2 and result["decoder_groups_advanced"]
        # Bus health does not require an enabled arm; readiness is reported separately.
        session.snapshot.assert_awaited_with(ready=False)
        assert result["counter_changes"]["rx_packets"] == 1
