"""Secret redaction for logs, tool outputs, and display-layer text.

Design (inspired by hermes-agent/redact.py):
- Display-layer only: internal values stay intact for API calls
- Central ``redact_sensitive_text()`` for all trust boundaries
- ``RedactingFormatter`` on log handlers — secrets never hit disk
- Separate modes: default (logs), file_read (sentinel), force (safety boundary)
- Configurable via ``LEAPFLOW_REDACT_SECRETS`` (default: on)
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, NamedTuple, Optional


class _PatternSpec(NamedTuple):
    """A redaction pattern with a fast substring gate."""
    gate: str
    regex: re.Pattern[str]
    label: str


def _compile_patterns() -> List[_PatternSpec]:
    """Build redaction patterns, ordered by frequency for early exit.

    The ``gate`` substring is checked first (fast ``in`` test) — the regex
    is only compiled/run when the gate matches.  This keeps log formatting
    near-zero cost for messages that contain no secrets.
    """
    specs: List[tuple[str, str, str]] = [
        # ── Known API key prefixes ────────────────────────────
        ("sk-", r"\bsk-[A-Za-z0-9_\-]{8,}", "api_key"),
        ("ghp_", r"\bghp_[A-Za-z0-9]{36,}", "github_pat"),
        ("gho_", r"\bgho_[A-Za-z0-9]{36,}", "github_oauth"),
        ("xoxb-", r"\bxoxb-[A-Za-z0-9\-]{30,}", "slack_bot"),
        ("xoxp-", r"\bxoxp-[A-Za-z0-9\-]{30,}", "slack_user"),
        ("AKIA", r"\bAKIA[A-Z0-9]{16}", "aws_access_key"),
        ("hf_", r"\bhf_[A-Za-z0-9]{10,}", "huggingface"),
        # ── Platform tokens (gateway) ─────────────────────────
        # Telegram bot token: 123456789:AABBccDDeeFF…
        ("/bot", r"\d{8,}:[A-Za-z0-9_\-]{30,}", "telegram_bot_token"),
        # Feishu app IDs (cli_a…) are non-secret but Feishu tenant
        # access tokens (t-…) and app_tickets are secret.
        ("t-", r"\bt-[A-Za-z0-9_\-]{20,}", "feishu_access_token"),
        # Encrypted credential values in gateway.yaml
        ("enc:fernet:", r"enc:fernet:[A-Za-z0-9_=\+/\-]{20,}", "fernet_cipher"),
        ("enc:b64:", r"enc:b64:[A-Za-z0-9_=\+/]{8,}", "b64_cipher"),
        # ── JWT tokens ────────────────────────────────────────
        ("eyJ", r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}", "jwt"),
        # ── Auth headers ──────────────────────────────────────
        ("Bearer", r"(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}", "auth_header"),
        ("x-api-key", r"(?i)x-api-key[:\s]+[A-Za-z0-9_\-]{10,}", "api_header"),
        ("Authorization", r"(?i)Authorization[:\s]+\S{10,}", "auth_header"),
        # ── ENV / config assignments ──────────────────────────
        ("=", r"(?i)(?:API_KEY|APP_SECRET|APP_KEY|BOT_TOKEN|SECRET|TOKEN|PASSWORD|CREDENTIAL)\s*=\s*\S{6,}", "env_assignment"),
        # ── JSON/YAML key-value secrets ───────────────────────
        ('"', r'(?i)"(?:api[_-]?key|app[_-]?secret|app[_-]?key|bot[_-]?token|secret|token|password|credential|access[_-]?token|webhook[_-]?secret)"\s*:\s*"[^"]{6,}"', "json_secret"),
        ("secret:", r"(?i)(?:app_secret|app_key|bot_token|password|secret)\s*:\s*\S{6,}", "yaml_secret"),
        # ── Database URLs ─────────────────────────────────────
        ("://", r"(?i)(?:postgres|mysql|mongodb|redis)://\S+:\S+@\S+", "db_url"),
        # ── Private keys ──────────────────────────────────────
        ("-----BEGIN", r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH)?\s*PRIVATE\s+KEY-----", "private_key"),
        # ── URL-embedded tokens ───────────────────────────────
        ("@github.com", r"https?://[A-Za-z0-9_\-]+@github\.com\S*", "url_token"),
    ]
    return [
        _PatternSpec(gate=gate, regex=re.compile(pattern), label=label)
        for gate, pattern, label in specs
    ]


_PATTERNS = _compile_patterns()

_ENABLED: Optional[bool] = None


def _is_enabled() -> bool:
    """Check if redaction is enabled (cached after first call)."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("LEAPFLOW_REDACT_SECRETS", "1").strip().lower() in ("1", "true", "yes")
    return _ENABLED


def _mask_token(token: str, *, preserve_prefix: int = 6, preserve_suffix: int = 4) -> str:
    """Mask a secret token, preserving prefix/suffix for identification."""
    if len(token) < 18:
        return "***"
    return f"{token[:preserve_prefix]}...{token[-preserve_suffix:]}"


def redact_sensitive_text(
    text: str,
    *,
    force: bool = False,
    file_read: bool = False,
) -> str:
    """Redact secrets from text for display/logging.

    Args:
        text: Input text that may contain secrets.
        force: Always redact even if globally disabled (for safety boundaries).
        file_read: Use sentinel markers (non-reusable) for file content mode.

    Returns:
        Text with secrets replaced by masked versions.
    """
    if not force and not _is_enabled():
        return text
    if not text or len(text) < 8:
        return text

    result = text
    for spec in _PATTERNS:
        if spec.gate not in result:
            continue
        def _replacer(m: re.Match[str]) -> str:
            matched = m.group(0)
            if file_read:
                masked = _mask_token(matched, preserve_prefix=4, preserve_suffix=0)
                return f"\u00abredacted:{masked}\u00bb"
            return _mask_token(matched)
        result = spec.regex.sub(_replacer, result)

    return result


class RedactingFormatter(logging.Formatter):
    """Log formatter that scrubs secrets from all log records.

    Install on handlers to prevent secrets from reaching disk or stdout.
    Zero overhead on clean messages (substring gate per pattern).
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return redact_sensitive_text(formatted, force=True)
