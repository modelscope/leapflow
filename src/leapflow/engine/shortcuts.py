"""User-customizable quick-reply shortcuts.

Shortcuts bypass intent classification and LLM calls entirely,
returning a canned reply for matched patterns.

Storage: `.leapflow/shortcuts.yaml` in the working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_DEFAULTS: dict[str, str] = {
    # English greetings
    "hi": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "hey": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "hello": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "yo": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "sup": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "howdy": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "what's up": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "whats up": "你好！我是 LEAP Agent，有什么可以帮你的？",
    # Chinese greetings
    "你好": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "您好": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "嗨": "你好！我是 LEAP Agent，有什么可以帮你的？",
    "嘿": "你好！我是 LEAP Agent，有什么可以帮你的？",
    # Time-of-day
    "good morning": "早上好！今天有什么可以帮你的？",
    "morning": "早上好！今天有什么可以帮你的？",
    "早": "早上好！今天有什么可以帮你的？",
    "早上好": "早上好！今天有什么可以帮你的？",
    "good afternoon": "下午好！有什么需要帮忙的？",
    "afternoon": "下午好！有什么需要帮忙的？",
    "下午好": "下午好！有什么需要帮忙的？",
    "good evening": "晚上好！需要帮忙就说。",
    "evening": "晚上好！需要帮忙就说。",
    "good night": "晚上好！需要帮忙就说。",
    "晚上好": "晚上好！需要帮忙就说。",
    "晚安": "晚上好！需要帮忙就说。",
    # Thanks
    "thanks": "不客气，随时可以找我 :)",
    "thank you": "不客气，随时可以找我 :)",
    "thx": "不客气，随时可以找我 :)",
    "ty": "不客气，随时可以找我 :)",
    "谢谢": "不客气，随时可以找我 :)",
    "感谢": "不客气，随时可以找我 :)",
    "多谢": "不客气，随时可以找我 :)",
    "辛苦了": "不客气，随时可以找我 :)",
    # Goodbye
    "bye": "再见！需要时随时唤我。",
    "goodbye": "再见！需要时随时唤我。",
    "see you": "再见！需要时随时唤我。",
    "see ya": "再见！需要时随时唤我。",
    "再见": "再见！需要时随时唤我。",
    "拜拜": "再见！需要时随时唤我。",
    "拜": "再见！需要时随时唤我。",
}

_STRIP_CHARS = "!！.。~？?"


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(_STRIP_CHARS)


class ShortcutStore:
    """Manages user-customizable quick-reply shortcuts."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._shortcuts: dict[str, str] = dict(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("shortcuts"), dict):
                self._shortcuts.update(data["shortcuts"])
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.dump(
            {"shortcuts": self._shortcuts},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        self._path.write_text(content, encoding="utf-8")

    def match(self, text: str) -> Optional[str]:
        """Return reply if text matches a shortcut, else None."""
        key = _normalize(text)
        return self._shortcuts.get(key)

    def add(self, pattern: str, reply: str) -> None:
        """Add or update a shortcut mapping and persist."""
        key = _normalize(pattern)
        self._shortcuts[key] = reply
        self._save()

    def remove(self, pattern: str) -> bool:
        """Remove a shortcut. Returns True if it existed."""
        key = _normalize(pattern)
        if key in self._shortcuts:
            del self._shortcuts[key]
            self._save()
            return True
        return False

    def list_all(self) -> dict[str, str]:
        """Return all shortcut mappings."""
        return dict(self._shortcuts)
