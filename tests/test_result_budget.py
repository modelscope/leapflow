# Copyright (c) Alibaba, Inc. and its affiliates.
"""Structure-first fitting of tool results into the per-result character budget.

The regression: a skill listing holding one workcell with 28 skills ran over
budget, and the old shrinker pruned the top-level ``workcells`` list to empty.
The model then read "no workcells" and guessed skill arguments blind.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from leapflow.engine.context.result_budget import fit_result_to_budget, head_tail_truncate


def _listing(skills: int = 28, card_chars: int = 900) -> dict[str, Any]:
    return {
        "ok": True,
        "kind": "skill_list",
        "workcells": [
            {
                "workcell": "piper_cell",
                "skills": [
                    {
                        "skill_id": f"skill.{index:02d}",
                        "available": index % 3 != 0,
                        "card": f"skill {index} " + "x" * card_chars,
                        "args_schema": {"properties": {"camera": {"type": "string"}}},
                    }
                    for index in range(skills)
                ],
            }
        ],
    }


def test_a_result_that_fits_is_serialized_unchanged() -> None:
    payload = {"ok": True, "items": [1, 2, 3]}
    assert fit_result_to_budget(payload, 1_000) == json.dumps(payload)


def test_one_large_entry_is_shrunk_from_inside_not_emptied() -> None:
    payload = _listing()
    original = copy.deepcopy(payload)
    budget = len(json.dumps(payload)) // 2

    text = fit_result_to_budget(payload, budget)
    fitted = json.loads(text)

    assert len(text) <= budget
    (cell,) = fitted["workcells"]
    assert [skill["skill_id"] for skill in cell["skills"]] == [
        f"skill.{index:02d}" for index in range(28)
    ], "every skill survives; only descriptions shorten"
    assert all("chars elided" in skill["card"] for skill in cell["skills"])
    assert [skill["available"] for skill in cell["skills"]] == [
        index % 3 != 0 for index in range(28)
    ]
    assert payload == original, "the caller's payload is never mutated"


def test_lists_are_pruned_from_the_tail_with_an_omission_count() -> None:
    payload = {"ok": True, "kind": "log", "entries": [f"line {index:04d}" for index in range(2_000)]}
    budget = 2_000

    fitted = json.loads(fit_result_to_budget(payload, budget))

    kept = fitted["entries"]
    assert 1 <= len(kept) < 2_000
    assert kept == payload["entries"][: len(kept)], "the head is kept in order"
    assert fitted["entries_omitted"] == 2_000 - len(kept)


def test_nested_lists_keep_at_least_one_element() -> None:
    payload = {
        "ok": True,
        "groups": [{"name": "only", "rows": [{"value": "y" * 120} for _ in range(400)]}],
    }
    budget = 1_500

    text = fit_result_to_budget(payload, budget)
    fitted = json.loads(text)

    assert len(text) <= budget
    (group,) = fitted["groups"]
    assert group["name"] == "only" and group["rows"]
    assert group["rows_omitted"] == 400 - len(group["rows"])


def test_an_unfittable_result_becomes_a_valid_sentinel() -> None:
    payload = {"ok": False, "kind": "dump", **{f"field_{index}": index for index in range(500)}}

    fitted = json.loads(fit_result_to_budget(payload, 120))

    assert fitted["truncated"] is True and fitted["ok"] is False
    assert fitted["original_chars"] == len(json.dumps(payload))


def test_atomic_context_view_keeps_values_and_qualifiers_without_mutation() -> None:
    from leapflow.engine.context.context_control import ToolEvidenceBuilder

    payload = {"ok": True, "result": {"selected": "b", "scores": [0.2, 0.8]},
               "provenance": {"advisory": True, "physical_authority": False, "trace": "x" * 5000},
               "_context_fields": {"ok": True, "result": True,
                                   "provenance": {"advisory": True, "physical_authority": True}}}
    original = copy.deepcopy(payload)
    compact = ToolEvidenceBuilder().build("extension_query", {}, payload)
    fitted = json.loads(fit_result_to_budget(compact, 300))
    assert fitted["result"] == original["result"]
    assert fitted["provenance"] == {"advisory": True, "physical_authority": False}
    assert fitted["context_compacted"] is True and "_context_fields" not in fitted
    assert payload == original
    full = json.loads(fit_result_to_budget(payload, 10000))
    assert full == {key: value for key, value in payload.items() if key != "_context_fields"}


def test_atomic_view_too_large_is_explicitly_omitted_not_pruned() -> None:
    payload = {"ok": True, "result": {"scores": list(range(100))},
               "_context_fields": {"ok": True, "result": True}}
    fitted = json.loads(fit_result_to_budget(payload, 140))
    assert fitted["context_fields_omitted"] is True and "result" not in fitted


def test_context_selector_cannot_replace_values_or_turn_failure_into_success() -> None:
    for fields in ({"ok": True, "result": "invented"}, {"ok": True, "missing": True}):
        payload = {"ok": False, "failure_code": "denied", "retryable": False,
                   "trace": "x" * 5000, "_context_fields": fields}
        fitted = json.loads(fit_result_to_budget(payload, 400))
        assert fitted["ok"] is False and fitted["failure_code"] == "denied"
        assert fitted["retryable"] is False and "context_compacted" not in fitted


def test_a_bare_value_is_cut_and_long_text_keeps_its_tail() -> None:
    assert fit_result_to_budget("z" * 50, 10) == json.dumps("z" * 50)[:10]
    text = "head " + "." * 1_000 + " Traceback: the real error"
    excerpt = head_tail_truncate(text, 200)
    assert excerpt.startswith("head ") and excerpt.endswith("the real error")
    assert "chars elided" in excerpt and len(excerpt) < len(text)
