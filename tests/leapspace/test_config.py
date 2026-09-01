"""Hermetic unit tests for AppTaskConfig (no Qt, no sandbox).

"""

import pytest
import yaml
from pydantic import ValidationError

from leapspace.app_space.config import AppTaskConfig

VALID = """\
id: task-001
title: Reply to the boss's unread message
apps: [chat]
instruction: "boss 给你发了一条消息，请回复确认收到。"
hooks:
  chat:
    before_launch: seed
    after_message_sent: boss_followup
interface:
  chat: [contact_list, message_history, message_input, send_button]
timeout_s: 120
max_steps: 30
"""

MINIMAL = """\
id: task-002
title: Open the app
apps: [chat]
instruction: "打开聊天应用。"
"""


def write(tmp_path, text=VALID):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_load_full(tmp_path):
    cfg = AppTaskConfig.load(write(tmp_path))
    assert cfg.id == "task-001"
    assert cfg.title == "Reply to the boss's unread message"
    assert cfg.apps == ("chat",)
    assert cfg.hooks == {
        "chat": {"before_launch": "seed", "after_message_sent": "boss_followup"}
    }
    assert cfg.interface == {
        "chat": ("contact_list", "message_history", "message_input", "send_button")
    }
    assert cfg.timeout_s == 120
    assert cfg.max_steps == 30


def test_load_minimal_defaults(tmp_path):
    cfg = AppTaskConfig.load(write(tmp_path, MINIMAL))
    assert cfg.hooks == {}
    assert cfg.interface == {}
    assert cfg.timeout_s is None
    assert cfg.max_steps is None


def test_load_accepts_task_directory(tmp_path):
    write(tmp_path)
    cfg = AppTaskConfig.load(tmp_path)
    assert cfg.id == "task-001"


def test_frozen(tmp_path):
    cfg = AppTaskConfig.load(write(tmp_path))
    with pytest.raises(ValidationError):
        cfg.id = "task-999"


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        AppTaskConfig.load(tmp_path / "config.yaml")


def test_invalid_yaml(tmp_path):
    with pytest.raises(yaml.YAMLError):
        AppTaskConfig.load(write(tmp_path, "id: [unclosed\n"))


def test_non_mapping_top_level(tmp_path):
    with pytest.raises(ValueError, match="top level must be a mapping"):
        AppTaskConfig.load(write(tmp_path, "- just\n- a\n- list\n"))


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValidationError, match="bogus"):
        AppTaskConfig.load(write(tmp_path, VALID + "bogus: 1\n"))


def test_wrong_field_type_rejected(tmp_path):
    with pytest.raises(ValidationError, match="hooks"):
        AppTaskConfig.load(write(tmp_path, MINIMAL + "hooks: [chat]\n"))
