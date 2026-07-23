"""Tests for the P0 coding built-in tools: code_search, file_find, edit_file.

Handlers are pure async functions (params dict -> result dict); exercised
directly. Governance (read-only vs mutating classification) is verified against
the tool registry so a failed read-only search never trips the batch-stop gate.
"""
from __future__ import annotations

import asyncio

from leapflow.tools import file_operations as fo
from leapflow.tools.file_operations import code_search, edit_file, file_find, file_write
from leapflow.tools.code_intel import code_intel


def _run(coro):
    return asyncio.run(coro)


# ── code_search ──────────────────────────────────────────────────────

def test_code_search_finds_matches_with_location(tmp_path) -> None:
    (tmp_path / "a.py").write_text("import os\nprint('hello world')\nx = 1\n")
    (tmp_path / "b.py").write_text("def hello():\n    return 'world'\n")

    result = _run(code_search({"pattern": r"hello", "path": str(tmp_path)}))

    assert result["ok"] is True
    assert result["match_count"] >= 2
    hits = {(m["path"].split("/")[-1], m["line"]) for m in result["matches"]}
    assert ("a.py", 2) in hits and ("b.py", 1) in hits
    assert all("text" in m and m["line"] for m in result["matches"])


def test_code_search_glob_filter(tmp_path) -> None:
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "a.txt").write_text("needle\n")

    result = _run(code_search({"pattern": "needle", "path": str(tmp_path), "glob": "*.py"}))

    assert result["ok"] is True
    assert result["match_count"] == 1
    assert result["matches"][0]["path"].endswith("a.py")


def test_code_search_skips_vcs_and_dep_dirs(tmp_path) -> None:
    (tmp_path / "src.py").write_text("token = 1\n")
    for skip in (".git", "node_modules", "__pycache__"):
        d = tmp_path / skip
        d.mkdir()
        (d / "junk.py").write_text("token = 2\n")

    result = _run(code_search({"pattern": "token", "path": str(tmp_path)}))

    assert result["ok"] is True
    assert result["match_count"] == 1              # only src.py, dep/vcs dirs skipped
    assert result["matches"][0]["path"].endswith("src.py")


def test_code_search_no_match_is_ok_empty(tmp_path) -> None:
    (tmp_path / "a.py").write_text("nothing here\n")
    result = _run(code_search({"pattern": "zzz-not-present", "path": str(tmp_path)}))
    assert result["ok"] is True and result["match_count"] == 0


def test_code_search_invalid_regex_is_structured_error(tmp_path) -> None:
    # Force the Python backend so an invalid pattern surfaces as re.error uniformly.
    import leapflow.tools.file_operations as mod
    orig = mod.shutil.which
    mod.shutil.which = lambda _name: None
    try:
        result = _run(code_search({"pattern": "([unclosed", "path": str(tmp_path)}))
    finally:
        mod.shutil.which = orig
    assert result["ok"] is False and result.get("error_type") == "invalid_regex"


def test_code_search_python_backend_matches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fo.shutil, "which", lambda _name: None)  # force fallback
    (tmp_path / "a.py").write_text("alpha\nbeta needle gamma\n")
    result = _run(code_search({"pattern": "needle", "path": str(tmp_path)}))
    assert result["ok"] is True and result["backend"] == "python"
    assert result["match_count"] == 1 and result["matches"][0]["line"] == 2


# ── file_find ────────────────────────────────────────────────────────

def test_file_find_recursive_glob(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x\n")
    (tmp_path / "top.py").write_text("y\n")
    (tmp_path / "note.md").write_text("z\n")

    result = _run(file_find({"glob": "*.py", "path": str(tmp_path)}))

    assert result["ok"] is True
    found = {p.split("/")[-1] for p in result["files"]}
    assert found == {"mod.py", "top.py"}


def test_file_find_skips_dep_dirs(tmp_path) -> None:
    (tmp_path / "keep.py").write_text("x\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("x\n")

    result = _run(file_find({"glob": "*.py", "path": str(tmp_path)}))

    assert [p.split("/")[-1] for p in result["files"]] == ["keep.py"]


def test_file_find_truncates_at_max_results(tmp_path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x\n")
    result = _run(file_find({"glob": "*.py", "path": str(tmp_path), "max_results": 3}))
    assert result["ok"] is True and result["file_count"] == 3 and result["truncated"] is True


def test_file_find_missing_glob_errors(tmp_path) -> None:
    result = _run(file_find({"path": str(tmp_path)}))
    assert result["ok"] is False


# ── edit_file ────────────────────────────────────────────────────────

def test_edit_file_unique_anchor_replaces(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("a = 1\nb = 2\nc = 3\n")

    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "b = 2", "new_text": "b = 20"}]}))

    assert result["ok"] is True and result["changed"] is True and result["replacements"] == 1
    assert f.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_edit_file_non_unique_anchor_is_rejected(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x = 1\nx = 1\n")

    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "x = 1", "new_text": "x = 2"}]}))

    assert result["ok"] is False and result["error_type"] == "anchor_not_unique"
    assert result["match_count"] == 2
    assert f.read_text() == "x = 1\nx = 1\n"        # unchanged — no silent corruption


def test_edit_file_replace_all(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x = 1\nx = 1\n")
    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "x = 1", "new_text": "x = 9", "replace_all": True}]}))
    assert result["ok"] is True and result["replacements"] == 2
    assert f.read_text() == "x = 9\nx = 9\n"


def test_edit_file_anchor_not_found(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("hello\n")
    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "nope", "new_text": "x"}]}))
    assert result["ok"] is False and result["error_type"] == "anchor_not_found"
    assert f.read_text() == "hello\n"


def test_edit_file_dry_run_does_not_write(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("a = 1\n")
    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "a = 1", "new_text": "a = 2"}], "dry_run": True}))
    assert result["ok"] is True and result.get("dry_run") is True and result["changed"] is True
    assert f.read_text() == "a = 1\n"               # not written


def test_edit_file_sequential_multi_edits(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("one\ntwo\nthree\n")
    result = _run(edit_file({"path": str(f), "edits": [
        {"original_text": "one", "new_text": "1"},
        {"original_text": "three", "new_text": "3"},
    ]}))
    assert result["ok"] is True and result["edits_applied"] == 2
    assert f.read_text() == "1\ntwo\n3\n"


def test_edit_file_missing_file_errors(tmp_path) -> None:
    result = _run(edit_file({"path": str(tmp_path / "nope.py"), "edits": [{"original_text": "a", "new_text": "b"}]}))
    assert result["ok"] is False and result["error_type"] == "file_not_found"


def test_edit_file_single_edit_shorthand(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("k = 1\n")
    result = _run(edit_file({"path": str(f), "original_text": "k = 1", "new_text": "k = 2"}))
    assert result["ok"] is True and f.read_text() == "k = 2\n"


# ── governance: registry classification ──────────────────────────────

def test_new_tools_execution_policy_classification() -> None:
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS, _BRIDGE_TOOLS
    from leapflow.tools.name_resolver import ToolRegistry, TOOL_NAME_ALIASES
    from leapflow.engine.tool_execution import execution_policy_for

    reg = ToolRegistry.from_definitions(
        TOOL_DEFINITIONS, TOOL_HANDLERS, bridge_tools=_BRIDGE_TOOLS, aliases=TOOL_NAME_ALIASES,
    )
    # Read-only search/find must NOT be side-effecting (else a failed search would
    # trip the batch-stop gate); edit_file mutates like file_write.
    assert execution_policy_for("code_search", reg.specs.get("code_search")) == "read_only"
    assert execution_policy_for("file_find", reg.specs.get("file_find")) == "read_only"
    assert execution_policy_for("edit_file", reg.specs.get("edit_file")) == "mutating_idempotent"


# ── ripgrep provisioning: seamless fallback + best-effort auto-install ──

def _reset_rg_cache() -> None:
    fo._RG_PROVISION.update(done=False, available=False)


def test_code_search_install_hint_when_ripgrep_missing(tmp_path, monkeypatch) -> None:
    """With ripgrep absent, code_search still works (Python) and surfaces a
    manual-install hint — the seamless fallback + fallback-to-manual path."""
    monkeypatch.setattr(fo, "ripgrep_path", lambda: None)
    (tmp_path / "a.py").write_text("needle here\n")
    result = _run(code_search({"pattern": "needle", "path": str(tmp_path)}))
    assert result["ok"] is True and result["backend"] == "python"
    assert result["match_count"] == 1
    assert "install_hint" in result and "ripgrep" in result["install_hint"]


def test_ensure_ripgrep_present_does_not_install(monkeypatch) -> None:
    _reset_rg_cache()
    monkeypatch.setattr(fo, "ripgrep_path", lambda: "/usr/bin/rg")

    def _no_install(*_a, **_k):
        raise AssertionError("must not install when ripgrep is already present")

    monkeypatch.setattr(fo.subprocess, "run", _no_install)
    assert fo.ensure_ripgrep_available(autoinstall=True) is True


def test_ensure_ripgrep_autoinstall_disabled_no_install(monkeypatch) -> None:
    _reset_rg_cache()
    monkeypatch.setattr(fo, "ripgrep_path", lambda: None)

    def _no_install(*_a, **_k):
        raise AssertionError("must not install when autoinstall disabled")

    monkeypatch.setattr(fo.subprocess, "run", _no_install)
    assert fo.ensure_ripgrep_available(autoinstall=False) is False


def test_ensure_ripgrep_non_darwin_skips_install(monkeypatch) -> None:
    _reset_rg_cache()
    monkeypatch.setattr(fo, "ripgrep_path", lambda: None)
    monkeypatch.setattr(fo.sys, "platform", "linux")

    def _no_install(*_a, **_k):
        raise AssertionError("no no-sudo installer on non-macOS -> Python fallback")

    monkeypatch.setattr(fo.subprocess, "run", _no_install)
    assert fo.ensure_ripgrep_available(autoinstall=True) is False


def test_ensure_ripgrep_darwin_brew_install_succeeds(monkeypatch) -> None:
    _reset_rg_cache()
    monkeypatch.setattr(fo.sys, "platform", "darwin")
    monkeypatch.setattr(fo.shutil, "which", lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    # ripgrep missing before the install, present after it.
    states = iter([None, "/opt/homebrew/bin/rg"])
    monkeypatch.setattr(fo, "ripgrep_path", lambda: next(states))
    captured: dict = {}

    def _run_install(cmd, **_k):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(fo.subprocess, "run", _run_install)
    assert fo.ensure_ripgrep_available(autoinstall=True) is True
    assert captured["cmd"][:2] == ["brew", "install"]


def test_ensure_ripgrep_is_cached(monkeypatch) -> None:
    _reset_rg_cache()
    fo._RG_PROVISION.update(done=True, available=True)

    def _no_install(*_a, **_k):
        raise AssertionError("cached result must not re-attempt install")

    monkeypatch.setattr(fo.subprocess, "run", _no_install)
    assert fo.ensure_ripgrep_available(autoinstall=True) is True
    _reset_rg_cache()


def test_ripgrep_install_hint_macos(monkeypatch) -> None:
    monkeypatch.setattr(fo.sys, "platform", "darwin")
    assert fo.ripgrep_install_hint() == "brew install ripgrep"


# ── code_intel: precise document symbols (ast for Python, heuristic otherwise) ──

def test_code_intel_python_ast_symbols(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text(
        "import os\n"
        "\n"
        "class Foo:\n"
        "    def method_a(self):\n"
        "        return 1\n"
        "\n"
        "def top_level(x, y):\n"
        "    return x + y\n"
    )
    result = _run(code_intel({"path": str(f)}))
    assert result["ok"] is True and result["engine"] == "ast" and result["language"] == "py"
    by_name = {s["name"]: s for s in result["symbols"]}
    assert by_name["Foo"]["kind"] == "class" and by_name["Foo"]["line"] == 3
    assert by_name["method_a"]["kind"] == "method" and by_name["method_a"]["parent"] == "Foo"
    assert by_name["top_level"]["kind"] == "function" and by_name["top_level"]["line"] == 7
    assert "def top_level(x, y):" in by_name["top_level"]["signature"]


def test_code_intel_non_python_heuristic(tmp_path) -> None:
    f = tmp_path / "m.js"
    f.write_text("function foo() {\n  return 1;\n}\nconst bar = 2;\n")
    result = _run(code_intel({"path": str(f)}))
    assert result["ok"] is True and result["engine"] == "heuristic"
    assert 1 in {s["line"] for s in result["symbols"]}   # 'function foo' line


def test_code_intel_invalid_python_falls_back(tmp_path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n    pass\n")  # syntax error -> heuristic fallback, not failure
    result = _run(code_intel({"path": str(f)}))
    assert result["ok"] is True and result["engine"] == "heuristic-fallback"


def test_code_intel_missing_path() -> None:
    assert _run(code_intel({}))["ok"] is False


def test_code_intel_file_not_found(tmp_path) -> None:
    assert _run(code_intel({"path": str(tmp_path / "nope.py")}))["ok"] is False


def test_code_intel_unsupported_operation(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    assert _run(code_intel({"path": str(f), "operation": "definition"}))["ok"] is False


# ── polish: code_search context_lines + edit_file unified-diff mode ──

def test_code_search_context_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fo, "ripgrep_path", lambda: None)  # deterministic Python backend
    (tmp_path / "a.py").write_text("line1\nline2\nNEEDLE\nline4\nline5\n")
    result = _run(code_search({"pattern": "NEEDLE", "path": str(tmp_path), "context_lines": 2}))
    assert result["ok"] is True and result["match_count"] == 1
    match = result["matches"][0]
    assert match["context_before"] == ["line1", "line2"]
    assert match["context_after"] == ["line4", "line5"]


def test_edit_file_apply_unified_diff(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("alpha\nbeta\ngamma\n")
    diff = "--- a/m.py\n+++ b/m.py\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"
    result = _run(edit_file({"path": str(f), "diff": diff}))
    assert result["ok"] is True and result["changed"] is True
    assert f.read_text() == "alpha\nBETA\ngamma\n"


def test_edit_file_invalid_diff_is_rejected(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x\n")
    result = _run(edit_file({"path": str(f), "diff": "not a diff, no hunks"}))
    assert result["ok"] is False and result["error_type"] == "invalid_diff"
    assert f.read_text() == "x\n"


# ── B1: advisory post-edit syntax verification ──

def test_edit_file_flags_broken_python_syntax(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "x = 1", "new_text": "def broken(:"}]}))
    assert result["ok"] is True and result["changed"] is True  # write is not blocked
    assert result["syntax_ok"] is False and "syntax_error" in result


def test_edit_file_valid_python_syntax_ok(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "x = 1", "new_text": "x = 2"}]}))
    assert result["ok"] is True and result["syntax_ok"] is True


def test_edit_file_non_python_has_no_syntax_field(tmp_path) -> None:
    f = tmp_path / "m.txt"
    f.write_text("hello\n")
    result = _run(edit_file({"path": str(f), "edits": [{"original_text": "hello", "new_text": "world"}]}))
    assert result["ok"] is True and "syntax_ok" not in result


def test_file_write_python_syntax_ok(tmp_path) -> None:
    f = tmp_path / "n.py"
    result = _run(file_write({"path": str(f), "content": "def ok():\n    return 1\n"}))
    assert result["ok"] is True and result["syntax_ok"] is True


def test_verify_edits_toggle_off(tmp_path) -> None:
    fo.set_edit_verification(False)
    try:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n")
        result = _run(edit_file({"path": str(f), "edits": [{"original_text": "x = 1", "new_text": "def broken(:"}]}))
        assert result["ok"] is True and "syntax_ok" not in result
    finally:
        fo.set_edit_verification(True)
