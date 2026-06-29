"""Shared JSON extraction for world model modules."""

from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from text, returning empty dict on failure."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}
