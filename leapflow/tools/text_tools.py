"""Text processing utilities — search, replace.

All handlers follow the ToolBridge convention: receive params dict, return result dict.
"""

from __future__ import annotations

import re
from typing import Any, Dict


async def text_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search for regex pattern in text. Returns match positions and groups."""
    text = params.get("text", "")
    pattern = params.get("pattern", "")

    if not pattern:
        return {"ok": False, "error": "Missing required parameter: pattern"}

    try:
        matches = [(m.start(), m.group()) for m in re.finditer(pattern, text)]
        return {"ok": True, "count": len(matches), "matches": matches[:50]}
    except re.error as e:
        return {"ok": False, "error": f"Invalid regex: {e}"}


async def text_replace(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace occurrences of 'old' with 'new' in text."""
    text = params.get("text", "")
    old = params.get("old", "")
    new = params.get("new", "")
    count = int(params.get("count", 0))

    if not old:
        return {"ok": False, "error": "Missing required parameter: old"}

    if count > 0:
        result = text.replace(old, new, count)
    else:
        result = text.replace(old, new)

    replacements = text.count(old) if count == 0 else min(count, text.count(old))
    return {"ok": True, "result": result, "replacements": replacements}
