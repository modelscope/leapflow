# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fit a tool result into the model's per-result character budget, structure first.

A result over budget is reduced in the order that loses the least meaning:

1. Every string is capped at one shared length -- the longest that fits -- so
   a listing keeps all of its items with shorter descriptions, and a single
   large field (stdout, file content) keeps its head and tail.
2. When that cannot fit, strings inside lists are capped at a floor and lists
   anywhere in the result are pruned from the tail, largest first, each prune
   annotated as ``<key>_omitted`` beside the list. A list keeps at least one
   element: a result holding one large entry (a workcell, a document) is shrunk
   from inside, never replaced by an empty list that reads as "nothing here".
3. The remaining strings are capped again into whatever room is left.

Successful tools may declare ``_context_fields`` as a nested mapping of existing
field names to True leaves. If the full payload does not fit, this atomic view
is selected before any lossy shaping. An unfittable view is explicitly omitted,
never partially pruned; declarations and durable payloads are not mutated.

When nothing fits, a minimal valid-JSON sentinel is returned; a raw cut that
leaves unparseable JSON never is. The functions are pure and never raise.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

# Strings at or below this length are never cut: below it a head+tail excerpt
# carries no more than the elision marker does.
_STRING_FLOOR = 160
_OMITTED_SUFFIX = "_omitted"


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def head_tail_truncate(text: str, allow: int) -> str:
    """Keep the head and tail of a long string with an explicit elision marker.

    The tail of stdout/stderr/tracebacks/test output usually holds the actual
    error, so a naive head-only cut discards the most useful part.
    """
    if len(text) <= allow:
        return text
    keep = max(40, allow - 40)  # leave room for the marker
    head = (keep * 2) // 3
    tail = keep - head
    elided = len(text) - head - tail
    return f"{text[:head]}\n… [{elided} chars elided] …\n{text[-tail:]}"


def fit_result_to_budget(payload: Any, budget: int) -> str:
    """Serialize a tool result to JSON within ``budget``, preserving structure."""
    fields = payload.get("_context_fields") if isinstance(payload, dict) else None
    if fields is not None:
        payload = {key: value for key, value in payload.items() if key != "_context_fields"}
    try:
        text = _dumps(payload)
    except (TypeError, ValueError):
        return str(payload)[:budget]
    if len(text) <= budget:
        return text
    if not isinstance(payload, dict):
        # A bare value has no structure to shrink; the cut is unavoidable.
        return text[:budget]
    original_chars = len(text)
    if payload.get("ok") is True and isinstance(fields, dict):
        # A domain declares an atomic evidence view by selecting existing fields.
        # Never prune its alternatives, scores, identifiers, or safety qualifiers.
        projected = _select_context_fields(payload, fields)
        if projected is not None and "ok" in projected:
            # Execution/recovery facts are never optional presentation fields.
            for key in ("execution_policy", "side_effect_state", "failure", "failure_code",
                        "failure_class", "retryable", "counts_as_failure"):
                if key in payload:
                    projected[key] = payload[key]
            projected["context_compacted"] = True
            fitted = _dumps(projected)
            if len(fitted) <= budget:
                return fitted
            return _dumps({"ok": True, "truncated": True, "context_fields_omitted": True,
                           "original_chars": original_chars, "budget_chars": budget})
    # An independent, JSON-normal copy: shrinking never mutates the caller's payload.
    shrunk: Any = json.loads(text)

    shrunk = _cap_strings(shrunk, budget)
    text = _dumps(shrunk)
    if len(text) <= budget:
        return text

    shrunk = _prune_lists(_capped(shrunk, _STRING_FLOOR, only_in_lists=True), budget)
    shrunk = _cap_strings(shrunk, budget)
    text = _dumps(shrunk)
    if len(text) <= budget:
        return text

    return _dumps(
        {
            "ok": payload.get("ok"),
            "kind": payload.get("kind", ""),
            "truncated": True,
            "original_chars": original_chars,
            "budget_chars": budget,
        }
    )


def _select_context_fields(payload: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any] | None:
    """Select a tool-owned atomic view without synthesizing or changing any value.

    Selectors are nested mappings with literal True leaves. Invalid selectors
    fall back to ordinary budgeting; they cannot change success/failure state.
    """
    selected = {}
    for key, selector in fields.items():
        if not isinstance(key, str) or key == "_context_fields":
            return None
        if selector is not True and not isinstance(selector, dict):
            return None
        if key not in payload:
            continue
        value = payload[key]
        if selector is True:
            selected[key] = value
        elif isinstance(value, dict):
            nested = _select_context_fields(value, selector)
            if nested is None:
                return None
            selected[key] = nested
        else:
            return None
    return selected


def _cap_strings(root: Any, budget: int) -> Any:
    """*root* with every string capped at the longest shared length that fits.

    Returns *root* unchanged when no string is long enough to cut, or when even
    the floor does not fit: cutting strings alone cannot fit, so the caller
    prunes lists before cutting any of them.
    """
    longest = max((len(s) for s in _strings(root)), default=0)
    if longest <= _STRING_FLOOR:
        return root
    floor = _capped(root, _STRING_FLOOR)
    if len(_dumps(floor)) > budget:
        return root
    low, high = _STRING_FLOOR, longest  # low fits, high (the uncapped root) does not
    while high - low > 1:
        middle = (low + high) // 2
        if len(_dumps(_capped(root, middle))) <= budget:
            low = middle
        else:
            high = middle
    return _capped(root, low)


def _capped(value: Any, cap: int, *, only_in_lists: bool = False, in_list: bool = False) -> Any:
    """A copy of *value* with strings longer than *cap* head+tail truncated.

    ``only_in_lists`` limits the cut to strings with a list ancestor: per-item
    descriptions of a listing, not the result's own fields.
    """
    if isinstance(value, str):
        if len(value) > cap and (in_list or not only_in_lists):
            return head_tail_truncate(value, cap)
        return value
    if isinstance(value, dict):
        return {
            key: _capped(item, cap, only_in_lists=only_in_lists, in_list=in_list)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_capped(item, cap, only_in_lists=only_in_lists, in_list=True) for item in value]
    return value


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _prune_lists(root: dict[str, Any], budget: int) -> dict[str, Any]:
    """Drop tail elements of the largest list until *root* fits or every list has one."""
    while True:
        over = len(_dumps(root)) - budget
        if over <= 0:
            return root
        found = _largest_list(root)
        if found is None:
            return root
        parent, key = found
        items = parent[key]
        each = max(1.0, len(_dumps(items)) / len(items))
        # Undershoot and loop rather than overshoot: an element dropped on an
        # estimate is information the model never gets back.
        drop = min(len(items) - 1, max(1, int(over / each)))
        parent[key] = items[: len(items) - drop]
        marker = f"{key}{_OMITTED_SUFFIX}"
        previous = parent.get(marker)
        already = previous if isinstance(previous, int) and not isinstance(previous, bool) else 0
        parent[marker] = already + drop


def _largest_list(value: Any) -> tuple[dict[str, Any], str] | None:
    """The dict-held list with more than one element whose serialization is largest."""
    best: tuple[int, dict[str, Any], str] | None = None
    for parent, key in _list_fields(value):
        size = len(_dumps(parent[key]))
        if best is None or size > best[0]:
            best = (size, parent, key)
    return None if best is None else (best[1], best[2])


def _list_fields(value: Any) -> Iterator[tuple[dict[str, Any], str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, list) and len(item) > 1:
                yield value, key
            yield from _list_fields(item)
    elif isinstance(value, list):
        for item in value:
            yield from _list_fields(item)


__all__ = ["fit_result_to_budget", "head_tail_truncate"]
