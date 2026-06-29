"""Message sequence healing — fix invalid patterns before LLM call."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MessageHealer:
    """Repairs invalid message sequences to prevent LLM errors."""

    def heal(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all repair rules in sequence."""
        if not messages:
            return messages
        messages = self._fix_empty_content(messages)
        messages = self._fix_role_alternation(messages)
        messages = self._fix_orphan_tool_results(messages)
        return messages

    def _fix_empty_content(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Replace empty content with placeholder."""
        result = []
        for msg in messages:
            content = msg.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                result.append({**msg, "content": "(no content)"})
            else:
                result.append(msg)
        return result

    def _fix_role_alternation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge consecutive same-role messages (except system and tool)."""
        if len(messages) < 2:
            return messages
        result = [messages[0]]
        for msg in messages[1:]:
            prev = result[-1]
            role = msg.get("role")
            # Never merge system, tool, or messages with tool_calls metadata
            if (
                role == prev.get("role")
                and role not in ("system", "tool")
                and "tool_calls" not in prev
            ):
                merged_content = f"{prev.get('content', '')}\n{msg.get('content', '')}"
                result[-1] = {**prev, "content": merged_content}
            else:
                result.append(msg)
        return result

    def _fix_orphan_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove tool results without preceding assistant message with tool_calls.

        Per OpenAI protocol, a role='tool' message MUST follow an assistant message
        that contains 'tool_calls'. This removes orphaned tool results that would
        cause API errors.
        """
        result = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                # Walk backwards to find the nearest assistant with tool_calls
                has_valid_parent = False
                for j in range(i - 1, -1, -1):
                    if messages[j].get("role") == "assistant":
                        # Valid parent = assistant message with tool_calls metadata
                        if "tool_calls" in messages[j]:
                            has_valid_parent = True
                        break
                    if messages[j].get("role") != "tool":
                        break
                if has_valid_parent:
                    result.append(msg)
                else:
                    logger.debug("message_healer: removed orphan tool result (no tool_calls parent)")
            else:
                result.append(msg)
        return result
