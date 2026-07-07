"""Smart file organization skill (metadata-first + LLM plan + moves)."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from leapflow.platform.protocol import HostRpc, Methods
from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider

logger = logging.getLogger(__name__)


def _log_progress(msg: str) -> None:
    sys.stderr.write(f"\033[2m→ {msg}\033[0m\n")
    sys.stderr.flush()


def _default_downloads() -> str:
    return str(Path.home() / "Downloads")


def _extract_path_hint(text: str) -> str | None:
    m = re.search(r'(/[^\s"]+)', text)
    if m:
        return m.group(1)
    return None


async def run(
    rpc: HostRpc,
    llm: LLMProvider,
    wm: WorkingMemoryProvider,
    lt: SemanticMemoryProvider,
    *,
    user_goal: str,
) -> str:
    """List PDFs, ask the LLM for category→folder mapping, apply moves via platform."""
    path_hint = _extract_path_hint(user_goal) or _default_downloads()

    _log_progress(f"Listing files in {path_hint}")
    listing = await rpc.call(Methods.FILE_LIST, {"path": path_hint, "include_hidden": False})
    entries: List[Dict[str, Any]] = list(listing.get("entries") or [])
    pdfs = [e for e in entries if str(e.get("name", "")).lower().endswith(".pdf")]
    if not pdfs:
        _log_progress(f"No PDF files found under {path_hint}")
        wm.remember_event("file_organizer", f"No PDFs under {path_hint}", {"path": path_hint})
        return f"No PDF files found under {path_hint}."

    pdf_names = [str(e.get("name", "")) for e in pdfs[:10]]
    remaining = len(pdfs) - len(pdf_names)
    names_display = ", ".join(pdf_names)
    if remaining > 0:
        names_display += f" ... and {remaining} more"
    _log_progress(f"Found {len(pdfs)} PDF(s) in {path_hint}: {names_display}")
    _log_progress("Planning organization with LLM...")
    compact = [
        {"name": e.get("name"), "path": e.get("path"), "size": e.get("size")} for e in pdfs[:50]
    ]
    messages = [
        build_system_message(
            "You are a file organization planner. "
            "Return STRICT JSON only: {\"plan\":[{\"src\":\"...\",\"dst_dir\":\"...\",\"category\":\"...\"}]} "
            "dst_dir must be an absolute path under the user's home."
        ),
        build_user_message_text(
            "Goal:\n"
            f"{user_goal}\n"
            f"Base directory: {path_hint}\n"
            f"Candidates:\n{json.dumps(compact, ensure_ascii=False)}"
        ),
    ]
    resp = await llm.achat(messages, stream=False, enable_thinking=False)
    raw = (resp.content or "").strip()
    plan: List[Dict[str, str]] = []
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        blob = raw[start : end + 1] if start != -1 and end != -1 else raw
        data = json.loads(blob)
        plan = list(data.get("plan") or [])
    except json.JSONDecodeError:
        logger.warning("LLM did not return JSON; using conservative fallback buckets.")
        for item in compact:
            c = "unsorted"
            name = str(item.get("name", "")).lower()
            if "invoice" in name or "发票" in name:
                c = "finance"
            if "spec" in name or "设计" in name or "api" in name:
                c = "engineering"
            src = str(item.get("path"))
            dst_dir = str(Path(path_hint) / "_organized" / c)
            plan.append({"src": src, "dst_dir": dst_dir, "category": c})

    categories = set(str(s.get("category", "unknown")) for s in plan)
    _log_progress(f"Organization plan: {len(plan)} file(s) → categories: {', '.join(sorted(categories))}")
    moves: List[Tuple[str, str]] = []
    for i, step in enumerate(plan):
        src = str(step.get("src", "")).strip()
        dst_dir = str(step.get("dst_dir", "")).strip()
        if not src or not dst_dir:
            continue
        name = Path(src).name
        dst = str(Path(dst_dir) / name)
        _log_progress(f"Moving ({i+1}/{len(plan)}): {name} → {dst_dir}/ [{step.get('category', '')}]")
        await rpc.call(Methods.FILE_MOVE, {"src": src, "dst": dst})
        moves.append((src, dst))
        lt.insert_raw(
            "file_change",
            f"Moved {src} -> {dst}",
            path=dst,
            metadata={"skill": "file_organizer", "category": step.get("category")},
        )
    summary = f"Planned {len(plan)} operations; executed {len(moves)} moves."
    wm.remember_event("file_organizer", summary, {"path": path_hint, "moves": len(moves)})
    logger.info("file_organizer complete: %s", summary)
    return summary + "\n" + "\n".join([f"- {a} → {b}" for a, b in moves[:20]])
