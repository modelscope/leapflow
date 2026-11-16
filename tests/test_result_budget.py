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


def test_a_bare_value_is_cut_and_long_text_keeps_its_tail() -> None:
    assert fit_result_to_budget("z" * 50, 10) == json.dumps("z" * 50)[:10]
    text = "head " + "." * 1_000 + " Traceback: the real error"
    excerpt = head_tail_truncate(text, 200)
    assert excerpt.startswith("head ") and excerpt.endswith("the real error")
    assert "chars elided" in excerpt and len(excerpt) < len(text)
