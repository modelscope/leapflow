"""Extract structured chart data from a session's captured tool outputs.

Signal-driven and anti-hallucination: this reads ONLY what the session already
produced — tool-result message contents and ``file_write`` artifact excerpts —
and never executes code or touches the network. Numeric values come from the
captured data, never from the model. When nothing reliable is found it returns
an empty mapping so the dashboard omits charts rather than drawing a fake line.

Output contract (all keys optional; absent when no data):
    {
      "series":       [{"id","label","kind":"line|area","points":[{"x","y"}]}],
      "ohlc":         [{"id","label","bars":[{"t","o","h","l","c","v"?}]}],
      "distribution": [{"id","label","items":[{"label","value"}]}],
    }
"""

from __future__ import annotations

import csv as _csv
import io
import json
import re
from typing import Any, Mapping, Optional, Sequence

_MAX_SERIES = 6
_MAX_POINTS = 500
_MAX_BLOCK_CHARS = 20000

_OHLC_ALIASES = {
    "o": "o", "open": "o",
    "h": "h", "high": "h",
    "l": "l", "low": "l",
    "c": "c", "close": "c", "adj close": "c", "adj_close": "c",
    "v": "v", "vol": "v", "volume": "v",
}
_TIME_KEYS = ("t", "time", "date", "datetime", "timestamp", "ts", "x", "period", "label")
_VALUE_KEYS = ("value", "y", "close", "price", "count", "amount", "score", "total")


def extract_charts(
    *,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
    artifacts: Optional[Sequence[Mapping[str, Any]]] = None,
    intents: Optional[Sequence[Mapping[str, Any]]] = None,
    max_series: int = _MAX_SERIES,
    max_points: int = _MAX_POINTS,
) -> dict[str, Any]:
    """Return a ``charts`` mapping extracted from captured session outputs."""
    series: list[dict[str, Any]] = []
    ohlc: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []

    for idx, (label, text) in enumerate(_candidate_blocks(messages or [], artifacts or [])):
        parsed = _parse_block(text)
        if parsed is None:
            continue
        rows, values = parsed
        bars = _as_ohlc(rows, max_points)
        if bars:
            ohlc.append({"id": f"ohlc-{idx}", "label": label, "bars": bars})
            continue
        items = _as_distribution(rows)
        if items:
            distribution.append({"id": f"dist-{idx}", "label": label, "items": items})
            continue
        points = _as_series(rows, values, max_points)
        if points:
            series.append({"id": f"series-{idx}", "label": label, "kind": "line", "points": points})

    _apply_intent_labels(series + ohlc + distribution, intents or [])
    out: dict[str, Any] = {}
    if series:
        out["series"] = series[:max_series]
    if ohlc:
        out["ohlc"] = ohlc[:max_series]
    if distribution:
        out["distribution"] = distribution[:max_series]
    return out


def _candidate_blocks(
    messages: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Collect (label, text) blocks worth parsing, newest tool output first."""
    blocks: list[tuple[str, str]] = []
    for artifact in artifacts:
        if str(artifact.get("status", "")) != "included":
            continue
        excerpt = str(artifact.get("content_excerpt", ""))[:_MAX_BLOCK_CHARS]
        if excerpt.strip():
            name = str(artifact.get("name") or artifact.get("path") or "artifact")
            blocks.append((name, excerpt))
    for message in messages:
        if str(message.get("role", "")) not in ("tool", "assistant"):
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        text = str(text)[:_MAX_BLOCK_CHARS]
        if _looks_structured(text):
            blocks.append((str(message.get("name") or "tool result"), text))
    return blocks


def _looks_structured(text: str) -> bool:
    stripped = text.lstrip()
    return stripped[:1] in ("{", "[") or "|" in text or "," in text


def _parse_block(text: str) -> Optional[tuple[list[dict[str, Any]], list[float]]]:
    """Return (rows, flat_values) parsed from one block, or None."""
    for parser in (_parse_json, _parse_markdown_table, _parse_csv):
        result = parser(text)
        if result is not None:
            return result
    return None


def _parse_json(text: str) -> Optional[tuple[list[dict[str, Any]], list[float]]]:
    obj = _load_json_fragment(text)
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        # A dict of label -> number reads as a distribution/series of pairs.
        rows = [{"label": str(k), "value": v} for k, v in obj.items() if _num(v) is not None]
        return (rows, []) if rows else None
    if isinstance(obj, list):
        if all(isinstance(item, Mapping) for item in obj):
            return ([dict(item) for item in obj], [])
        values = [n for item in obj if (n := _num(item)) is not None]
        return ([], values) if values else None
    return None


def _load_json_fragment(text: str) -> Any:
    stripped = text.strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except (ValueError, TypeError):
                continue
    return None


def _parse_markdown_table(text: str) -> Optional[tuple[list[dict[str, Any]], list[float]]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2 or not re.match(r"^\|[\s:\-|]+\|?$", lines[1]):
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows: list[dict[str, Any]] = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append({header[i]: cells[i] for i in range(len(header))})
    return (rows, []) if rows else None


def _parse_csv(text: str) -> Optional[tuple[list[dict[str, Any]], list[float]]]:
    sample = "\n".join(text.splitlines()[:_MAX_POINTS + 1]).strip()
    if "," not in sample or "\n" not in sample:
        return None
    try:
        reader = list(_csv.reader(io.StringIO(sample)))
    except _csv.Error:
        return None
    if len(reader) < 2:
        return None
    header = [h.strip() for h in reader[0]]
    if not any(_looks_headerish(h) for h in header):
        return None
    rows = [
        {header[i]: row[i] for i in range(min(len(header), len(row)))}
        for row in reader[1:] if row
    ]
    return (rows, []) if rows else None


def _looks_headerish(cell: str) -> bool:
    return bool(cell) and _num(cell) is None


def _as_ohlc(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    keymap = _column_map(rows[0], _OHLC_ALIASES)
    if not {"o", "h", "l", "c"}.issubset(keymap.values()):
        return []
    bars: list[dict[str, Any]] = []
    for row in rows[:max_points]:
        bar: dict[str, Any] = {}
        for src, role in keymap.items():
            num = _num(row.get(src))
            if num is not None:
                bar[role] = num
        if {"o", "h", "l", "c"}.issubset(bar):
            bar["t"] = _time_of(row)
            bars.append(bar)
    return bars if len(bars) >= 2 else []


def _as_series(rows: list[dict[str, Any]], values: list[float], max_points: int) -> list[dict[str, Any]]:
    if values:
        return [{"x": i, "y": v} for i, v in enumerate(values[:max_points])]
    if not rows:
        return []
    value_key = _pick_value_key(rows[0])
    if value_key is None:
        return []
    points: list[dict[str, Any]] = []
    for i, row in enumerate(rows[:max_points]):
        y = _num(row.get(value_key))
        if y is None:
            continue
        x = _time_of(row)
        points.append({"x": x if x is not None else i, "y": y})
    return points if len(points) >= 2 else []


def _as_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or len(rows) > 40:
        return []
    # A categorical label column (not a time axis) + a value column reads as a
    # distribution; time/numeric x-axes fall through to a line series instead.
    label_key = next(
        (k for k in rows[0]
         if str(k).strip().lower() not in _TIME_KEYS and _num(rows[0][k]) is None),
        None,
    )
    value_key = _pick_value_key(rows[0])
    if label_key is None or value_key is None:
        return []
    items = [
        {"label": str(row.get(label_key, "")), "value": v}
        for row in rows if (v := _num(row.get(value_key))) is not None
    ]
    return items if items else []


def _column_map(row: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in row:
        role = aliases.get(str(key).strip().lower())
        if role and role not in out.values():
            out[key] = role
    return out


def _pick_value_key(row: Mapping[str, Any]) -> Optional[str]:
    for hint in _VALUE_KEYS:
        for key in row:
            if str(key).strip().lower() == hint and _num(row[key]) is not None:
                return key
    return next((k for k in row if _num(row[k]) is not None), None)


def _time_of(row: Mapping[str, Any]) -> Any:
    for key in row:
        if str(key).strip().lower() in _TIME_KEYS:
            return str(row[key])
    return None


def _apply_intent_labels(charts: list[dict[str, Any]], intents: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(it.get("label")) for it in intents if isinstance(it, Mapping) and it.get("label")]
    for chart, label in zip(charts, labels):
        if label:
            chart["label"] = label


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("%", "").replace("$", "")
        # Accept ints, decimals, leading-dot (.5) and trailing-dot (1.) floats,
        # with an optional exponent; reject inf/nan/garbage before float().
        if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", cleaned or ""):
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


__all__ = ["extract_charts"]
