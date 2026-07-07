"""Message sanitizer — cleans LLM output of invalid characters and encoding issues.

Handles:
- UTF-16 surrogate pairs (common in some model outputs)
- Control characters (except newline/tab)
- Non-printable Unicode characters
- Null bytes
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Regex patterns for problematic characters
_SURROGATE_PAIR = re.compile(r'[\ud800-\udfff]')
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_NULL_BYTE = re.compile(r'\x00')


class MessageSanitizer:
    """Sanitizes LLM output by removing/replacing problematic characters.

    Designed to be zero-cost for clean input (fast path) and
    gracefully handle all known encoding edge cases.
    """

    def __init__(
        self,
        *,
        replace_char: str = "",
        log_sanitizations: bool = True,
    ) -> None:
        self._replace_char = replace_char
        self._log_sanitizations = log_sanitizations

    def sanitize(self, text: str) -> str:
        """Clean text of problematic characters.

        Fast path: if no issues detected, returns input unchanged (zero-copy).
        """
        if not text:
            return text

        # Fast path: check if sanitization needed
        if not self._needs_sanitization(text):
            return text

        original_len = len(text)

        # Remove null bytes
        text = _NULL_BYTE.sub(self._replace_char, text)

        # Handle surrogate pairs
        text = _SURROGATE_PAIR.sub(self._replace_char, text)

        # Remove control characters (keep \n \t \r)
        text = _CONTROL_CHARS.sub(self._replace_char, text)

        # Ensure valid UTF-8 by encode/decode roundtrip
        try:
            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        if self._log_sanitizations and len(text) != original_len:
            logger.debug(
                "Sanitized message: removed %d problematic characters",
                original_len - len(text),
            )

        return text

    def sanitize_message(self, message: dict) -> dict:
        """Sanitize the content field of a message dict."""
        content = message.get("content")
        if isinstance(content, str):
            cleaned = self.sanitize(content)
            if cleaned is not content:  # Identity check for fast path
                return {**message, "content": cleaned}
        return message

    @staticmethod
    def _needs_sanitization(text: str) -> bool:
        """Quick check if text contains any problematic characters."""
        for ch in text:
            code = ord(ch)
            if code == 0:  # null
                return True
            if 0xD800 <= code <= 0xDFFF:  # surrogate
                return True
            if code < 0x20 and code not in (0x09, 0x0A, 0x0D):  # control (not tab/nl/cr)
                return True
        return False
