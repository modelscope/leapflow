"""Hermetic unit tests for action_lint (AST only, no Qt, no sandbox).

"""

from leapspace.app_space.action_lint import lint_task, main

CONFIG = """\
id: task-t
title: lint me
apps: [chat]
instruction: x
hooks:
  chat:
    before_launch: seed
    after_message_sent: boss_followup
"""

GOOD_ACTION = '''\
import json
from PyQt6.QtCore import QTimer

def seed(app):
    app.reset({})

def boss_followup(app):
    app.inject_message("boss", "hi")

async def reference(actor):
    await actor.list_apps()

def expect(state_root="/tmp/leapspace"):
    return 0
'''


def make_task(tmp_path, action=GOOD_ACTION, config=CONFIG):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "action.py").write_text(action)
    (tmp_path / "config.yaml").write_text(config)
    return tmp_path


def test_clean_task_passes(tmp_path):
    assert lint_task(make_task(tmp_path)) == []


def test_missing_action_py(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG)
    assert lint_task(tmp_path) == [f"{tmp_path}/action.py: not found"]


def test_syntax_error(tmp_path):
    problems = lint_task(make_task(tmp_path, action="def broken(:\n"))
    assert len(problems) == 1 and "syntax error" in problems[0]


def test_bad_config_reported_not_raised(tmp_path):
    problems = lint_task(make_task(tmp_path, config=CONFIG + "bogus: 1\n"))
    assert len(problems) == 1 and problems[0].startswith("config.yaml:")


def test_hook_missing(tmp_path):
    action = GOOD_ACTION.replace("def boss_followup(app):", "def renamed(app):")
    problems = lint_task(make_task(tmp_path, action=action))
    assert problems == ["hook 'boss_followup' (chat.after_message_sent): not defined in action.py"]


def test_hook_must_be_sync(tmp_path):
    action = GOOD_ACTION.replace("def seed(app):", "async def seed(app):")
    problems = lint_task(make_task(tmp_path, action=action))
    assert any("never awaits hooks" in p for p in problems)


def test_hook_arity(tmp_path):
    for bad in ("def seed():", "def seed(app, extra):"):
        problems = lint_task(make_task(tmp_path, action=GOOD_ACTION.replace("def seed(app):", bad)))
        assert any("exactly one parameter (app)" in p for p in problems)


def test_reference_missing(tmp_path):
    action = GOOD_ACTION.replace("async def reference(actor):", "async def renamed(actor):")
    problems = lint_task(make_task(tmp_path, action=action))
    assert problems == ["reference: missing (async def reference(actor))"]


def test_reference_must_be_async(tmp_path):
    action = GOOD_ACTION.replace("async def reference(actor):", "def reference(actor):")
    problems = lint_task(make_task(tmp_path, action=action))
    assert any("must be async def" in p for p in problems)


def test_expect_missing(tmp_path):
    action = GOOD_ACTION.replace("def expect(", "def renamed(")
    problems = lint_task(make_task(tmp_path, action=action))
    assert any(p.startswith("expect: missing") for p in problems)


def test_expect_params_need_defaults(tmp_path):
    action = GOOD_ACTION.replace('state_root="/tmp/leapspace"', "state_root")
    problems = lint_task(make_task(tmp_path, action=action))
    assert any("must have defaults" in p for p in problems)


def test_host_only_import_at_module_level(tmp_path):
    action = "import cua_sandbox\n" + GOOD_ACTION
    problems = lint_task(make_task(tmp_path, action=action))
    assert any("cua_sandbox" in p and "not in-sandbox safe" in p for p in problems)


def test_host_only_import_inside_function_ok(tmp_path):
    action = GOOD_ACTION.replace(
        "async def reference(actor):",
        "async def reference(actor):\n    import cua_sandbox",
    )
    assert lint_task(make_task(tmp_path, action=action)) == []


def test_relative_import_rejected(tmp_path):
    action = "from . import helper\n" + GOOD_ACTION
    problems = lint_task(make_task(tmp_path, action=action))
    assert any("relative import" in p for p in problems)


def test_cli_exit_codes(tmp_path, capsys):
    assert main([str(make_task(tmp_path / "good"))]) == 0
    bad_dir = make_task(tmp_path / "bad", action="x = 1\n")
    assert main([str(bad_dir)]) == 1
    assert "reference: missing" in capsys.readouterr().err
