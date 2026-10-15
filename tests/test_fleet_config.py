"""Fleet Settings parsing keeps edge topology declarative and secret-safe."""

from __future__ import annotations

from leapflow.hardware.fleet_config import configured_nodes_from_settings


def test_settings_fleet_node_resolves_its_token_environment_reference(monkeypatch) -> None:
    monkeypatch.setenv("JETSON_LHP_TOKEN", "test-token")

    nodes = configured_nodes_from_settings(
        '[{"node_id":"yahboom","display_name":"Jetson","endpoint":"tcp://192.168.55.1:8767","token_env":"JETSON_LHP_TOKEN"}]'
    )

    assert len(nodes) == 1
    assert nodes[0].node_id == "yahboom"
    assert nodes[0].endpoint == "tcp://192.168.55.1:8767"
    assert nodes[0].token == "test-token"


def test_settings_nodes_take_precedence_over_legacy_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        '[{"node_id":"legacy","endpoint":"tcp://127.0.0.1:8767","token":"legacy"}]',
    )

    nodes = configured_nodes_from_settings(
        '[{"node_id":"configured","endpoint":"tcp://192.168.55.1:8767"}]'
    )

    assert [node.node_id for node in nodes] == ["configured"]


def test_empty_settings_retain_legacy_environment_fallback(monkeypatch) -> None:
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        '[{"node_id":"yahboom","endpoint":"tcp://192.168.55.1:8767"}]',
    )

    nodes = configured_nodes_from_settings()

    assert [node.node_id for node in nodes] == ["yahboom"]
