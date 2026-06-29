"""File system operations — list, read, write.

All handlers follow the ToolBridge convention: receive params dict, return result dict.
Safety: paths are resolved and validated before operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


# Safety: block writes to sensitive system directories
_BLOCKED_WRITE_PREFIXES = (
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
    "/var/root",
    "/Library/System",
)


def _is_write_blocked(path: Path) -> bool:
    """Check if writing to the given path should be blocked."""
    resolved = str(path)
    for prefix in _BLOCKED_WRITE_PREFIXES:
        if resolved.startswith(prefix):
            return True
    return False


async def file_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """List directory contents with optional glob pattern."""
    path = params.get("path", ".")
    pattern = params.get("pattern", "*")

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {path}"}
    if not target.is_dir():
        return {"ok": False, "error": f"Not a directory: {path}"}

    entries = []
    for item in sorted(target.glob(pattern)):
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
        })

    return {"ok": True, "path": str(target), "entries": entries[:100]}


async def file_read(params: Dict[str, Any]) -> Dict[str, Any]:
    """Read text file content with line limit."""
    path = params.get("path", "")
    max_lines = int(params.get("max_lines", 200))

    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    if not target.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    try:
        lines = target.read_text(errors="replace").splitlines()
        truncated = len(lines) > max_lines
        content = "\n".join(lines[:max_lines])
        return {
            "ok": True,
            "path": str(target),
            "content": content,
            "lines": len(lines),
            "truncated": truncated,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def file_write(params: Dict[str, Any]) -> Dict[str, Any]:
    """Write content to a file. Supports overwrite and append modes."""
    path = params.get("path", "")
    content = params.get("content", "")
    mode = params.get("mode", "overwrite")

    if not path:
        return {"ok": False, "error": "Missing required parameter: path"}

    target = Path(path).expanduser().resolve()

    if _is_write_blocked(target):
        return {"ok": False, "error": f"Write blocked by safety policy: {target}"}

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with target.open("a") as f:
                f.write(content)
        else:
            target.write_text(content)
        return {"ok": True, "path": str(target), "bytes_written": len(content.encode())}
    except Exception as e:
        return {"ok": False, "error": str(e)}
