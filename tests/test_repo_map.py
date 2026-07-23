"""Tests for the repo_map project-orientation tool (C1)."""
from __future__ import annotations

import asyncio

from leapflow.tools.repo_map import repo_map


def _run(coro):
    return asyncio.run(coro)


def test_repo_map_python_project(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n\n[project.scripts]\ndemo = "demo:main"\n'
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("")

    r = _run(repo_map({"path": str(tmp_path)}))

    assert r["ok"] is True
    assert "python" in r["languages"]
    assert r["test_command"] == "python -m pytest -q"
    assert r["lint_command"] == "ruff check ."
    assert "main.py" in r["entry_points"]
    assert r["manifest"]["project_name"] == "demo"
    assert "demo" in r["manifest"]["scripts"]
    assert r["readme"] == "README.md"
    assert "src/" in r["structure"]["dirs"] and "tests/" in r["structure"]["dirs"]
    assert "node_modules/" not in r["structure"]["dirs"]  # dependency dir skipped


def test_repo_map_node_project(tmp_path) -> None:
    (tmp_path / "package.json").write_text(
        '{"name":"web","main":"index.js","scripts":{"test":"jest","build":"tsc"}}'
    )
    (tmp_path / "index.js").write_text("")

    r = _run(repo_map({"path": str(tmp_path)}))

    assert "javascript/typescript" in r["languages"]
    assert r["test_command"] == "npm test"
    assert r["manifest"]["project_name"] == "web" and r["manifest"]["main"] == "index.js"
    assert "test" in r["manifest"]["scripts"]
    assert "index.js" in r["entry_points"]


def test_repo_map_reads_vcs_branch(tmp_path) -> None:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/feature/x\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    r = _run(repo_map({"path": str(tmp_path)}))

    assert r["vcs"]["git"] is True and r["vcs"]["branch"] == "feature/x"
    assert ".git/" not in r["structure"]["dirs"]  # hidden dir skipped


def test_repo_map_no_vcs(tmp_path) -> None:
    r = _run(repo_map({"path": str(tmp_path)}))
    assert r["vcs"]["git"] is False


def test_repo_map_not_a_directory(tmp_path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    r = _run(repo_map({"path": str(f)}))
    assert r["ok"] is False and r["failure_code"] == "path_not_found"


def test_repo_map_is_read_only() -> None:
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS, _BRIDGE_TOOLS
    from leapflow.tools.name_resolver import ToolRegistry, TOOL_NAME_ALIASES
    from leapflow.engine.tool_execution import execution_policy_for

    reg = ToolRegistry.from_definitions(
        TOOL_DEFINITIONS, TOOL_HANDLERS, bridge_tools=_BRIDGE_TOOLS, aliases=TOOL_NAME_ALIASES,
    )
    assert execution_policy_for("repo_map", reg.specs.get("repo_map")) == "read_only"
