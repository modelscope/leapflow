"""Perceptual Field Policy Loader — builtin rules, YAML persistence, goal inference.

Loads and merges rules from multiple sources:
    1. Builtin safety rules (priority=1000, unoverridable)
    2. User explicit rules from YAML (priority=500)
    3. Learned rules from YAML (priority=100)
    4. Goal-seeded rules generated at session start (priority=50)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from leapflow.domain.perception import FieldRule, PerceptionLevel, sort_rules
from leapflow.recording.perceptual_field import FieldPolicy

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# App Category Registry — single source of truth for builtin safety rules
# ═══════════════════════════════════════════════════════════════════════

_APP_CATEGORIES: List[Dict] = [
    {
        "category": "password_manager",
        "patterns": [
            "com.agilebits.onepassword*",
            "com.apple.keychainaccess*",
            "com.lastpass.*",
            "com.bitwarden.*",
            "com.dashlane.*",
        ],
        "level": PerceptionLevel.DENY,
        "priority": 1000,
    },
    {
        "category": "system_security",
        "patterns": [
            "com.apple.systempreferences",
            "com.apple.SystemSettings",
        ],
        "context_rules": [
            (PerceptionLevel.OPAQUE, "*密码*|*Password*|*Security*|*安全*|*Privacy*|*隐私*"),
        ],
        "priority": 1000,
    },
    {
        "category": "browser_private",
        "patterns": [
            "com.google.Chrome*",
            "com.apple.Safari*",
            "org.mozilla.firefox*",
            "com.microsoft.edgemac*",
        ],
        "context_rules": [
            (PerceptionLevel.OPAQUE, "*(Incognito)*|*(无痕)*|*(Private*|*(隐私*|*(InPrivate)*"),
        ],
        "priority": 1000,
    },
    {
        "category": "browser_sensitive",
        "patterns": [
            "com.google.Chrome*",
            "com.apple.Safari*",
            "org.mozilla.firefox*",
            "com.microsoft.edgemac*",
        ],
        "context_rules": [
            (PerceptionLevel.OPAQUE, "*banking*|*alipay*|*wechatpay*|*paypal*"),
        ],
        "priority": 900,
    },
    {
        "category": "video_conference",
        "patterns": [
            "us.zoom.xos*",
            "com.tencent.meeting*",
            "com.microsoft.teams*",
        ],
        "level": PerceptionLevel.OPAQUE,
        "priority": 900,
    },
]


def _build_builtin_rules() -> List[FieldRule]:
    """Generate builtin safety rules from the app category registry."""
    rules: List[FieldRule] = []
    for cat in _APP_CATEGORIES:
        patterns = cat["patterns"]
        priority = cat["priority"]
        if "level" in cat:
            for p in patterns:
                rules.append(FieldRule(p, "*", cat["level"], "builtin", priority))
        for level, ctx_pattern in cat.get("context_rules", []):
            for p in patterns:
                rules.append(FieldRule(p, ctx_pattern, level, "builtin", priority))
    return rules


_BUILTIN_RULES: List[FieldRule] = _build_builtin_rules()

# ═══════════════════════════════════════════════════════════════════════
# Goal → App Hints (structured keyword-to-bundle mapping)
# ═══════════════════════════════════════════════════════════════════════

_GOAL_APP_KEYWORDS: List[Dict] = [
    {"keywords": ["微信", "wechat"], "bundle": "com.tencent.xinWeChat"},
    {"keywords": ["钉钉", "dingtalk"], "bundle": "com.alibaba-inc.DingTalk"},
    {"keywords": ["slack"], "bundle": "com.tinyspeck.slackmacgap"},
    {"keywords": ["chrome"], "bundle": "com.google.Chrome"},
    {"keywords": ["safari"], "bundle": "com.apple.Safari"},
    {"keywords": ["firefox"], "bundle": "org.mozilla.firefox"},
    {"keywords": ["vscode", "vs code"], "bundle": "com.microsoft.VSCode"},
    {"keywords": ["xcode"], "bundle": "com.apple.dt.Xcode"},
    {"keywords": ["finder"], "bundle": "com.apple.finder"},
    {"keywords": ["terminal"], "bundle": "com.apple.Terminal"},
    {"keywords": ["iterm"], "bundle": "com.googlecode.iterm2"},
    {"keywords": ["notion"], "bundle": "notion.id"},
    {"keywords": ["飞书", "lark"], "bundle": "com.bytedance.lark"},
    {"keywords": ["figma"], "bundle": "com.figma.Desktop"},
    {"keywords": ["邮件", "mail"], "bundle": "com.apple.mail"},
    {"keywords": ["outlook"], "bundle": "com.microsoft.Outlook"},
]

_GOAL_APP_HINTS: Dict[str, str] = {
    kw: entry["bundle"]
    for entry in _GOAL_APP_KEYWORDS
    for kw in entry["keywords"]
}

# ═══════════════════════════════════════════════════════════════════════
# Goal NLP Configuration — structured word lists for keyword extraction
# ═══════════════════════════════════════════════════════════════════════

_FUNCTION_WORD_PATTERN = re.compile(
    r"[的在中里从到和与用去做了是被把让给将对向往]|[，。！？、；\s]+"
)

_VERB_BOUNDARY_PATTERN = re.compile(
    r"(发送|接收|打开|关闭|查看|编辑|分享|查找|整理|发消息)"
)

_STOP_WORDS: frozenset = frozenset({
    "消息", "文件", "分享", "查找", "资料", "然后", "整理",
    "发送", "接收", "打开", "关闭", "查看", "编辑", "操作",
    "the", "in", "on", "at", "to", "from", "with", "for", "and",
    "learn", "show", "how", "open", "close", "send",
})


# ═══════════════════════════════════════════════════════════════════════
# Policy Loader
# ═══════════════════════════════════════════════════════════════════════


class FieldPolicyLoader:
    """Loads perceptual field rules from all sources and assembles a FieldPolicy."""

    def load(
        self,
        *,
        config_path: Optional[Path] = None,
        goal: str = "",
        extra_rules: Sequence[FieldRule] = (),
    ) -> FieldPolicy:
        """Load and merge rules from all sources into a FieldPolicy.

        Priority order (highest first):
            1. Builtin safety rules (1000)
            2. User rules from YAML (500)
            3. Learned rules from YAML (100)
            4. Goal-seeded rules (50)
            5. Extra (session-scoped) rules at their own priority
        """
        rules: list[FieldRule] = list(_BUILTIN_RULES)

        if config_path is not None:
            resolved = config_path.expanduser()
            if resolved.exists():
                rules.extend(self._load_yaml(resolved))

        if goal:
            rules.extend(self._generate_goal_rules(goal))

        rules.extend(extra_rules)

        return FieldPolicy(rules=rules)

    def append_user_rule(self, rule: FieldRule, *, config_path: Path) -> None:
        """Append a user rule to the YAML config file."""
        resolved = config_path.expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        data = self._read_yaml_data(resolved)
        if "rules" not in data:
            data["rules"] = []
        data["rules"].append(self._rule_to_dict(rule))
        self._write_yaml_data(resolved, data)

    def remove_user_rule(self, index: int, *, config_path: Path) -> Optional[FieldRule]:
        """Remove a user rule by index. Returns the removed rule or None."""
        resolved = config_path.expanduser()
        if not resolved.exists():
            return None

        data = self._read_yaml_data(resolved)
        entries = data.get("rules", [])
        if index < 0 or index >= len(entries):
            return None

        removed_data = entries.pop(index)
        self._write_yaml_data(resolved, data)
        return self._dict_to_rule(removed_data, "user", 500)

    def clear_learned_rules(self, *, config_path: Path) -> int:
        """Remove all learned rules from config. Returns count removed."""
        resolved = config_path.expanduser()
        if not resolved.exists():
            return 0

        data = self._read_yaml_data(resolved)
        learned = data.get("learned", [])
        count = len(learned)
        data["learned"] = []
        self._write_yaml_data(resolved, data)
        return count

    def generate_template(self, path: Path) -> None:
        """Generate a commented YAML template at the given path."""
        resolved = path.expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(_YAML_TEMPLATE, encoding="utf-8")

    # ── Goal-seeded rule generation ──

    def _generate_goal_rules(self, goal: str) -> list[FieldRule]:
        """Infer perceptual field rules from a goal string."""
        rules: list[FieldRule] = []
        lower = goal.lower()

        target_apps: list[str] = []
        for keyword, bundle in _GOAL_APP_HINTS.items():
            if keyword in lower:
                target_apps.append(bundle)

        context_keywords = self._extract_context_keywords(goal, target_apps)

        for app in target_apps:
            if context_keywords:
                for kw in context_keywords:
                    rules.append(FieldRule(
                        app_pattern=f"{app}*",
                        context_pattern=f"*{kw}*",
                        level=PerceptionLevel.FULL,
                        source="goal",
                        priority=50,
                    ))
            rules.append(FieldRule(
                app_pattern=f"{app}*",
                context_pattern="*",
                level=PerceptionLevel.STRUCTURAL,
                source="goal",
                priority=40,
            ))

        return rules

    def _extract_context_keywords(self, goal: str, target_apps: list[str]) -> list[str]:
        """Extract context-identifying keywords from goal text.

        Strategy: remove app names, split on function words + verb boundaries,
        then extract remaining noun phrases as potential window-title context IDs.
        """
        app_keywords = set(_GOAL_APP_HINTS.keys())

        cleaned = goal
        for kw in sorted(app_keywords, key=len, reverse=True):
            cleaned = cleaned.replace(kw, " ")

        fragments = _FUNCTION_WORD_PATTERN.split(cleaned)

        keywords: list[str] = []
        for frag in fragments:
            frag = frag.strip()
            if not frag or len(frag) < 2:
                continue
            sub_parts = _VERB_BOUNDARY_PATTERN.split(frag)
            for part in sub_parts:
                part = part.strip()
                if not part or len(part) < 2:
                    continue
                if part.lower() in _STOP_WORDS or part.lower() in app_keywords:
                    continue
                if 2 <= len(part) <= 8:
                    keywords.append(part)

        return keywords

    # ── YAML I/O ──

    def _load_yaml(self, path: Path) -> list[FieldRule]:
        """Load rules from a YAML file."""
        data = self._read_yaml_data(path)
        rules: list[FieldRule] = []

        for entry in data.get("rules", []):
            rules.extend(self._expand_entry(entry, "user", 500))

        for entry in data.get("learned", []):
            rules.extend(self._expand_entry(entry, "learned", 100))

        return rules

    def _expand_entry(self, entry: dict, source: str, priority: int) -> list[FieldRule]:
        """Expand a YAML entry (which may have allow/deny lists) into FieldRules."""
        rules: list[FieldRule] = []
        app_pattern = entry.get("app", "*")
        contexts = entry.get("contexts", {})

        for pattern in contexts.get("allow", []):
            rules.append(FieldRule(app_pattern, pattern, PerceptionLevel.FULL, source, priority))

        for pattern in contexts.get("deny", []):
            rules.append(FieldRule(app_pattern, pattern, PerceptionLevel.DENY, source, priority))

        for pattern in contexts.get("opaque", []):
            rules.append(FieldRule(app_pattern, pattern, PerceptionLevel.OPAQUE, source, priority))

        for pattern in contexts.get("structural", []):
            rules.append(FieldRule(app_pattern, pattern, PerceptionLevel.STRUCTURAL, source, priority))

        default_str = entry.get("default_level")
        if default_str:
            level = PerceptionLevel(default_str)
            rules.append(FieldRule(app_pattern, "*", level, source, priority - 10))

        return rules

    def _read_yaml_data(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            import yaml
            text = path.read_text(encoding="utf-8")
            return yaml.safe_load(text) or {}
        except Exception as e:
            logger.warning("Failed to load perceptual field config %s: %s", path, e)
            return {}

    def _write_yaml_data(self, path: Path, data: dict) -> None:
        try:
            import yaml
            path.parent.mkdir(parents=True, exist_ok=True)
            text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write perceptual field config %s: %s", path, e)

    @staticmethod
    def _rule_to_dict(rule: FieldRule) -> dict:
        level_key = {
            PerceptionLevel.FULL: "allow",
            PerceptionLevel.DENY: "deny",
            PerceptionLevel.OPAQUE: "opaque",
            PerceptionLevel.STRUCTURAL: "structural",
        }[rule.level]
        return {
            "app": rule.app_pattern,
            "contexts": {level_key: [rule.context_pattern]},
        }

    @staticmethod
    def _dict_to_rule(data: dict, source: str, priority: int) -> FieldRule:
        app = data.get("app", "*")
        contexts = data.get("contexts", {})
        for level_key, level_val in [
            ("deny", PerceptionLevel.DENY),
            ("opaque", PerceptionLevel.OPAQUE),
            ("structural", PerceptionLevel.STRUCTURAL),
            ("allow", PerceptionLevel.FULL),
        ]:
            patterns = contexts.get(level_key, [])
            if patterns:
                return FieldRule(app, patterns[0], level_val, source, priority)
        return FieldRule(app, "*", PerceptionLevel.FULL, source, priority)


# ═══════════════════════════════════════════════════════════════════════
# YAML Template
# ═══════════════════════════════════════════════════════════════════════

_YAML_TEMPLATE = """\
# Perceptual Field Configuration
# Controls per-context perception depth within apps.
#
# Levels:
#   full       — record everything (default)
#   structural — record action structure only (text stripped)
#   opaque     — record only that activity occurred (no details)
#   deny       — completely exclude from recording
#
# Patterns use glob syntax. Pipe (|) separates alternatives.

version: "1.0"

rules:
  # Example: Record WeChat work groups fully, deny personal groups
  # - app: "com.tencent.xinWeChat"
  #   contexts:
  #     allow: ["*工作*", "*项目*"]
  #     deny: ["*家人*"]
  #   default_level: structural

  # Example: Only record work-related browser tabs
  # - app: "com.google.Chrome*"
  #   contexts:
  #     allow: ["*github*", "*google.com*"]
  #     deny: ["*facebook*", "*twitter*"]
  #   default_level: structural

learned: []
"""
