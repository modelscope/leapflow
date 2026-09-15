# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic tests for cross-platform daemon transport helpers."""
from __future__ import annotations

import pytest

from leapflow.daemon._transport import TcpLoopbackTransport


def test_tcp_loopback_transport_rejects_corrupt_port_file(tmp_path) -> None:
    transport = TcpLoopbackTransport()
    (tmp_path / "leapd.port").write_text("not-a-port", encoding="utf-8")

    with pytest.raises(OSError, match="invalid daemon port file"):
        transport._read_port(tmp_path)
    assert transport.probe_healthy(tmp_path) is False


def test_tcp_loopback_transport_rejects_out_of_range_port(tmp_path) -> None:
    transport = TcpLoopbackTransport()
    (tmp_path / "leapd.port").write_text("70000", encoding="utf-8")

    with pytest.raises(OSError, match="invalid daemon port value"):
        transport._read_port(tmp_path)
    assert transport.probe_healthy(tmp_path) is False
