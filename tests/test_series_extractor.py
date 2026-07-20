"""Hermetic tests for the anti-hallucination chart extractor.

Pure text parsing: no network, no code execution, no invented numbers.
"""

from __future__ import annotations

from leapflow.dashboard.templates import render_template
from leapflow.monitor.series_extractor import _num, extract_charts


def _artifact(text: str) -> dict:
    return {"status": "included", "name": "data", "content_excerpt": text}


def test_num_parses_floats_including_trailing_dot() -> None:
    # Trailing-dot floats (e.g. '1.') are valid and must parse; inf/nan/garbage reject.
    assert _num("1.") == 1.0
    assert _num("1.5") == 1.5
    assert _num(".5") == 0.5
    assert _num("-3") == -3.0
    assert _num("1e3") == 1000.0
    assert _num("1,234.5") == 1234.5
    assert _num("abc") is None
    assert _num("") is None
    assert _num("inf") is None


def test_json_number_array_becomes_a_series() -> None:
    out = extract_charts(artifacts=[_artifact("[10, 12, 11, 15, 14]")])
    points = out["series"][0]["points"]
    assert len(points) == 5
    assert points[0] == {"x": 0, "y": 10.0}


def test_ohlc_rows_become_candlesticks() -> None:
    rows = (
        '[{"date":"d1","open":1,"high":3,"low":0.5,"close":2},'
        '{"date":"d2","open":2,"high":4,"low":1.5,"close":3}]'
    )
    out = extract_charts(artifacts=[_artifact(rows)])
    bars = out["ohlc"][0]["bars"]
    assert bars[0]["o"] == 1.0 and bars[0]["c"] == 2.0 and bars[0]["t"] == "d1"
    assert "series" not in out  # OHLC classified as candlesticks, not a line


def test_markdown_table_of_labels_becomes_a_distribution() -> None:
    md = "| name | value |\n| --- | --- |\n| A | 3 |\n| B | 5 |\n"
    out = extract_charts(artifacts=[_artifact(md)])
    assert out["distribution"][0]["items"] == [
        {"label": "A", "value": 3.0},
        {"label": "B", "value": 5.0},
    ]


def test_csv_time_and_close_becomes_a_series() -> None:
    csv = "date,close\n2026-07-08,10.0\n2026-07-09,10.5\n2026-07-10,11.2\n"
    out = extract_charts(artifacts=[_artifact(csv)])
    points = out["series"][0]["points"]
    assert points[0] == {"x": "2026-07-08", "y": 10.0}


def test_prose_only_session_yields_no_charts() -> None:
    # No structured numbers anywhere -> nothing is drawn (no fake line ever).
    out = extract_charts(
        messages=[{"role": "assistant", "content": "just prose without any structured data"}],
        artifacts=[],
    )
    assert out == {}


def test_model_intents_only_relabel_extracted_series() -> None:
    # The label comes from the model; the numbers come strictly from the data.
    out = extract_charts(artifacts=[_artifact("[1, 2, 3]")], intents=[{"label": "BABA close"}])
    assert out["series"][0]["label"] == "BABA close"
    assert [p["y"] for p in out["series"][0]["points"]] == [1.0, 2.0, 3.0]


def test_when_conditional_omits_nodes_bound_to_empty_data() -> None:
    template = {
        "layout": [
            {"type": "Card", "when": "charts.series", "props": {"title": "Series"}},
            {"type": "Card", "props": {"title": "Always"}},
        ]
    }
    empty = render_template(template, {"charts": {}})
    present = render_template(template, {"charts": {"series": [{"points": [{"x": 0, "y": 1}]}]}})
    empty_titles = [n["props"].get("title") for n in empty["root"]]
    present_titles = [n["props"].get("title") for n in present["root"]]
    assert empty_titles == ["Always"]
    assert present_titles == ["Series", "Always"]
