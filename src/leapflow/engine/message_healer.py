# Copyright (c) Alibaba, Inc. and its affiliates.
"""Message sequence healing — fix invalid patterns before LLM call.

Repairs (inspired by hermes message_sanitization.py):
1. Empty content → placeholder
2. Consecutive same-role merging (respects tool_calls metadata)
3. Orphan tool result removal
4. Missing tool result synthesis (assistant tool_calls without a response)
5. Malformed tool_call argument JSON repair
6. Interrupted tool sequence closing (tail role=tool gets synthetic assistant)
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
        messages = self._fix_missing_tool_results(messages)
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

    def _fix_missing_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Synthesize a tool result for any tool_call left without a response.

        Per the OpenAI tool protocol, an assistant message carrying
        ``tool_calls`` must be followed by exactly one ``role="tool"`` message per
        ``tool_call_id``; a missing one triggers HTTP 400 ("insufficient tool
        messages following tool_calls message"). This can arise when a tool batch
        is stopped early (side-effect gating), a turn is cancelled mid-batch, or
        compression drops a result. Rather than let the request fail, emit a
        synthetic non-executed result for each unanswered call, inserted right
        after the existing tool run so the pairing stays contiguous.

        This is the reverse of :meth:`_fix_orphan_tool_results` and completes the
        invariant guard. It is a transient boundary repair on the copy sent to
        the provider; it does not mutate durable history.
        """
        # A tool_call is considered answered if any tool message anywhere carries
        # its id, so a result that survived out of position is not duplicated.
        responded: set[str] = set()
        for msg in messages:
            if msg.get("role") == "tool":
                call_id = str(msg.get("tool_call_id", ""))
                if call_id:
                    responded.add(call_id)

        result: List[Dict[str, Any]] = []
        synthesized = 0
        index = 0
        total = len(messages)
        while index < total:
            msg = messages[index]
            result.append(msg)
            tool_calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
            if not tool_calls:
                index += 1
                continue
            # Copy the contiguous run of tool results that already follow.
            cursor = index + 1
            while cursor < total and messages[cursor].get("role") == "tool":
                result.append(messages[cursor])
                cursor += 1
            # Append a placeholder for each still-unanswered call, in emission order.
            for call in tool_calls:
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id and call_id not in responded:
                    result.append(self._synthetic_tool_result(call_id))
                    responded.add(call_id)
                    synthesized += 1
            index = cursor

        if synthesized:
            logger.debug(
                "message_healer: synthesized %d missing tool result(s)", synthesized
            )
        return result

    @staticmethod
    def _synthetic_tool_result(tool_call_id: str) -> Dict[str, Any]:
        """Build a minimal, provider-valid tool result for an unanswered tool_call."""
        content = json.dumps(
            {
                "ok": False,
                "execution_skipped": True,
                "skipped_reason": "no_result_recorded",
                "note": (
                    "This tool call produced no result (batch stopped, cancelled, "
                    "or truncated). Re-issue it if the action is still needed."
                ),
                "counts_as_failure": False,
            },
            ensure_ascii=False,
        )
        return {"role": "tool", "tool_call_id": str(tool_call_id), "content": content}

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

        Skips insertion when the trailing tool results have a valid preceding
        assistant message with ``tool_calls`` — this is the normal native-tool
        loop where the API expects to generate the next response after tool
        results.  A synthetic assistant injected here would break thinking-mode
        providers (e.g. DeepSeek) that require ``reasoning_content`` on every
        assistant message in the history.
        """
        if not messages:
            return messages

        if messages[-1].get("role") != "tool":
            return messages

        # Walk backwards: if the trailing tool block has a valid parent
        # (an assistant message with tool_calls), the sequence is a normal
        # mid-turn tool call — no synthetic closer needed.
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "tool":
                continue
            if role == "assistant" and msg.get("tool_calls"):
                return messages  # valid parent found — keep the sequence open
            break  # different role without tool_calls — orphaned tail

        return messages + [
            {"role": "assistant", "content": "Operation interrupted. Continuing..."}
        ]
