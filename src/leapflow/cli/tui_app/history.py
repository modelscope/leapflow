# Copyright (c) Alibaba, Inc. and its affiliates.
"""Input history for the TUI that never persists credentials.

Commands such as ``/config llm set --api-key ...`` carry secrets on the
command line. A plain ``FileHistory`` writes them to disk in clear text and
replays them through up-arrow recall and auto-suggest, so the secret leaks
into every later session.
"""

from __future__ import annotations

import re
from typing import Iterable

from prompt_toolkit.history import FileHistory

from leapflow.security.redact import redact_sensitive_text

_SECRET_OPTION_RE = re.compile(
    r"(?i)(?:^|\s)--?(?:[a-z0-9]+[-_])*"
    r"(?:api[-_]?key|secret|token|password|passwd|credential|private[-_]?key)s?"
    r"(?:\s*=\s*|\s+)\S"
)


def contains_secret(text: str) -> bool:
    """Return True when a submitted line carries a credential value."""
    if _SECRET_OPTION_RE.search(text):
        return True
    return redact_sensitive_text(text, force=True) != text


class SecretAwareFileHistory(FileHistory):
    """``FileHistory`` that drops credential-bearing entries.

    New entries carrying a secret are neither stored nor kept in memory, and
    entries already on disk from older versions are hidden from recall.
    """

    def append_string(self, string: str) -> None:
        if contains_secret(string):
            return
        super().append_string(string)

    def load_history_strings(self) -> Iterable[str]:
        return [item for item in super().load_history_strings() if not contains_secret(item)]
