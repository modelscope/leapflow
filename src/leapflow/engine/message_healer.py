"""Message sequence healing — fix invalid patterns before LLM call.

Repairs (inspired by hermes message_sanitization.py):
1. Empty content → placeholder
2. Consecutive same-role merging (respects tool_calls metadata)
3. Orphan tool result removal
4. Malformed tool_call argument JSON repair
5. Interrupted tool sequence closing (tail role=tool gets synthetic assistant)
"""
from __future__ import annotations

import json
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
        messages = self._repair_tool_call_arguments(messages)
        messages = self._fix_role_alternation(messages)
        messages = self._fix_orphan_tool_results(messages)
        messages = self._close_interrupted_tool_sequence(messages)
        return messages

    def _fix_empty_content(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Replace empty content with placeholder."""
        result = []
        for msg in messages:
            content = msg.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                if msg.get("tool_calls"):
                    result.append(msg)
                else:
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
            if (
                role == prev.get("role")
                and role not in ("system", "tool")
                and "tool_calls" not in prev
                and "tool_calls" not in msg
            ):
                prev_content = prev.get("content", "")
                msg_content = msg.get("content", "")
                if isinstance(prev_content, str) and isinstance(msg_content, str):
                    merged_content = f"{prev_content}\n{msg_content}"
                    result[-1] = {**prev, "content": merged_content}
                else:
                    result.append(msg)
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
                has_valid_parent = False
                for j in range(i - 1, -1, -1):
                    if messages[j].get("role") == "assistant":
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

    def _repair_tool_call_arguments(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Repair malformed JSON in tool_call arguments.

        Multi-pass pipeline (inspired by hermes _repair_tool_call_arguments):
        1. Try json.loads(strict=False) + re-serialize
        2. Fix common issues: trailing commas, unbalanced braces
        3. Last resort: replace with "{}"
        """
        result = []
        repaired_count = 0
        for msg in messages:
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                result.append(msg)
                continue

            fixed_calls = []
            needs_fix = False
            for tc in tool_calls:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", "{}")

                if not isinstance(args_raw, str):
                    fixed_calls.append(tc)
                    continue

                fixed_args = self._try_repair_json(args_raw)
                if fixed_args != args_raw:
                    needs_fix = True
                    repaired_count += 1
                    fixed_fn = {**fn, "arguments": fixed_args}
                    fixed_calls.append({**tc, "function": fixed_fn})
                else:
                    fixed_calls.append(tc)

            if needs_fix:
                result.append({**msg, "tool_calls": fixed_calls})
            else:
                result.append(msg)

        if repaired_count:
            logger.debug("message_healer: repaired %d tool_call arguments", repaired_count)
        return result

    @staticmethod
    def _try_repair_json(raw: str) -> str:
        """Attempt to repair a possibly malformed JSON string."""
        if not raw or not raw.strip():
            return "{}"

        try:
            json.loads(raw)
            return raw
        except (json.JSONDecodeError, ValueError):
            pass

        cleaned = raw.strip()

        cleaned = cleaned.rstrip(",")

        open_braces = cleaned.count("{") - cleaned.count("}")
        if open_braces > 0:
            cleaned += "}" * open_braces
        elif open_braces < 0:
            cleaned = cleaned[:cleaned.rfind("}")]

        open_brackets = cleaned.count("[") - cleaned.count("]")
        if open_brackets > 0:
            cleaned += "]" * open_brackets

        try:
            json.loads(cleaned)
            return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("message_healer: unrepairable tool_call arguments, replacing with {}")
        return "{}"

    def _close_interrupted_tool_sequence(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """If the tail is role=tool, append a synthetic assistant to close the sequence.

        Prevents role alternation violations on resume or after compression
        where the last message is a tool result without a following assistant reply.
        """
        if not messages:
            return messages

        if messages[-1].get("role") != "tool":
            return messages

        return messages + [
            {"role": "assistant", "content": "Operation interrupted. Continuing..."}
        ]
