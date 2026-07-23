"""Repository orientation map — a compact, read-only project overview.

Grounds the agent when it enters a codebase: languages, detected test/lint
commands, top-level structure, entry points / manifest, and VCS state. Assembled
cheaply (marker files, a shallow top-level listing, manifest parsing, and a
``.git/HEAD`` read) with no subprocess and no deep tree walk. Read-only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

try:  # Python 3.11+ stdlib; the project targets >=3.11.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from leapflow.tools.dev_tools import _detect_lint_command, _detect_test_command
from leapflow.tools.file_operations import _SEARCH_SKIP_DIRS

logger = logging.getLogger(__name__)

_MAX_STRUCTURE_ENTRIES = 60
_LANG_MARKERS = (
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("requirements.txt", "python"),
    ("package.json", "javascript/typescript"),
    ("tsconfig.json", "typescript"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
    ("pom.xml", "java"),
    ("build.gradle", "java/kotlin"),
    ("Gemfile", "ruby"),
    ("composer.json", "php"),
    ("CMakeLists.txt", "c/c++"),
)
_ENTRY_CANDIDATES = (
    "main.py", "__main__.py", "app.py", "manage.py", "cli.py",
    "index.js", "index.ts", "main.go", "src/main.rs",
)


def _detect_languages(root: Path) -> List[str]:
    languages: List[str] = []
    for marker, lang in _LANG_MARKERS:
        if (root / marker).exists() and lang not in languages:
            languages.append(lang)
    return languages


def _vcs_info(root: Path) -> Dict[str, Any]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"git": False}
    branch = ""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            branch = head.split("/", 2)[-1]
    except (OSError, ValueError):
        pass
    return {"git": True, "branch": branch}


def _python_manifest(root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    pyproject = root / "pyproject.toml"
    if tomllib is None or not pyproject.exists():
        return info
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
    except (ValueError, OSError):
        return info
    project = data.get("project", {}) or {}
    if project.get("name"):
        info["project_name"] = str(project["name"])
    scripts = project.get("scripts") or {}
    if isinstance(scripts, dict) and scripts:
        info["scripts"] = list(scripts.keys())[:10]
    return info


def _node_manifest(root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    pkg = root / "package.json"
    if not pkg.exists():
        return info
    try:
        data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
    except (ValueError, OSError):
        return info
    if isinstance(data, dict):
        if data.get("name"):
            info["project_name"] = str(data["name"])
        if data.get("main"):
            info["main"] = str(data["main"])
        scripts = data.get("scripts") or {}
        if isinstance(scripts, dict) and scripts:
            info["scripts"] = list(scripts.keys())[:10]
    return info


async def repo_map(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact project orientation map for a repository root (read-only)."""
    root = Path(str(params.get("path") or ".")).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return {"ok": False, "error": f"Not a directory: {root}", "failure_code": "path_not_found"}

    dirs: List[str] = []
    files: List[str] = []
    try:
        for item in sorted(root.iterdir()):
            if item.name in _SEARCH_SKIP_DIRS:
                continue
            if item.name.startswith(".") and item.name != ".github":
                continue
            if item.is_dir():
                dirs.append(item.name + "/")
            else:
                files.append(item.name)
            if len(dirs) + len(files) >= _MAX_STRUCTURE_ENTRIES:
                break
    except OSError as exc:  # noqa: BLE001 - surface as structured error
        return {"ok": False, "error": str(exc)}

    manifest: Dict[str, Any] = {}
    manifest.update(_python_manifest(root))
    manifest.update(_node_manifest(root))
    readme = next((n for n in ("README.md", "README.rst", "README.txt", "README") if (root / n).exists()), "")

    return {
        "ok": True,
        "tool": "repo_map",
        "root": str(root),
        "languages": _detect_languages(root),
        "test_command": _detect_test_command(root),
        "lint_command": _detect_lint_command(root),
        "structure": {"dirs": dirs, "files": files},
        "entry_points": [c for c in _ENTRY_CANDIDATES if (root / c).exists()],
        "manifest": manifest,
        "readme": readme,
        "vcs": _vcs_info(root),
    }
