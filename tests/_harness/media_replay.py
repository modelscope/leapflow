# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resolve symbolic evidence references only at the synthetic provider boundary.

Cassettes retain symbols, not stale random cache IDs. The adapter reads IDs from
actual tool results in the requesting model's own prompt; it cannot retain media,
grant consent, bypass ownership, or call a production tool. Live providers are
never wrapped. Raw traffic is captured for byte-parity assertions in tests only.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from typing import Any

from tests._harness.cassette_proxy import CassetteProxy


def _result_texts(message: dict[str, Any]) -> list[str]:
    """Read only actual tool JSON or the engine's persisted tool-evidence lines."""
    content = message.get("content")
    if not isinstance(content, str):
        return []
    if message.get("role") == "tool":
        candidates = [content]
    elif message.get("role") == "assistant" and "[Tool Evidence —" in content:
        candidates = [line.removeprefix("- hw_skill_run: ") for line in content.splitlines()
                      if line.startswith("- hw_skill_run: ")]
    else:
        return []
    return candidates


def _message_results(message: dict[str, Any]) -> list[Any]:
    results = []
    for candidate in _result_texts(message):
        try:
            results.append(json.loads(candidate))
        except ValueError:
            continue
    return results


def _references(value: Any, result: dict[str, str]) -> None:
    if isinstance(value, dict):
        if value.get("kind") in ("image", "video") and isinstance(value.get("artifact_id"), str):
            result[value["kind"]] = value["artifact_id"]
        for item in value.values():
            _references(item, result)
    elif isinstance(value, list):
        for item in value:
            _references(item, result)


def _stable_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint capture timing/encoder telemetry separately from semantic data.

    Only evidence-bearing skill results are normalized. Actual traffic, media
    bytes, task measurements, ownership checks, and approvals remain unchanged.
    """
    stable = deepcopy(payload)
    for message in stable.get("messages", []):
        for original in _result_texts(message):
            try:
                result = json.loads(original)
            except ValueError:
                continue
            evidence = result.get("evidence") if isinstance(result, dict) else None
            if not isinstance(evidence, dict) or evidence.get("status") != "retained":
                continue
            values = result.get("result")
            if isinstance(values, dict):
                for key in ("liftable_for_s", "valid_for_s", "duration_s"):
                    if key in values:
                        values[key] = "<CAPTURE_TIME>"
                # Snapshot differences retain their geometry and verdict; only
                # elapsed capture time and its derived prose vary with scheduling.
                for change in values.get("changes", {}).values():
                    if not isinstance(change, dict) or "age_s" not in change:
                        continue
                    change["age_s"] = "<CAPTURE_TIME>"
                    summary = change.get("summary")
                    if not isinstance(summary, str):
                        continue
                    stable_summary = re.sub(r"\(\d+ s ago\)", "(<CAPTURE_TIME> s ago)", summary)
                    change["summary"] = stable_summary
                    if isinstance(result.get("message"), str):
                        result["message"] = result["message"].replace(summary, stable_summary)
                    if "situation" in values:
                        values["situation"] = [stable_summary if item == summary else item
                                               for item in values["situation"]]
            if evidence.get("duration_s") is not None:
                evidence["duration_s"] = "<CAPTURE_TIME>"
            if evidence.get("kind") == "video":
                evidence["size_bytes"] = "<ENCODED_SIZE>"
            message["content"] = message["content"].replace(original, json.dumps(result, ensure_ascii=False))
    return stable


def bind_media_response(proxy: CassetteProxy, raw_requests: list[dict]) -> None:
    """Wrap one seeded/replay test proxy; retain its accounting and mode behavior."""
    if proxy.mode not in ("seed", "replay"):
        raise ValueError("symbolic evidence is restricted to offline scripted providers")
    original = proxy.handle_chat

    def respond(payload):
        raw_requests.append(payload)
        response = original(_stable_request(payload))
        if response.is_stream or response.status != 200:
            return response
        decoded = json.loads(response.body)
        references: dict[str, str] = {}
        for message in payload.get("messages", []):
            for result in _message_results(message):
                _references(result, references)
        changed = False
        for choice in decoded.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls", []):
                function = call.get("function", {})
                if function.get("name") != "model_query":
                    continue
                args = json.loads(function["arguments"])
                for label, reference in args.get("artifacts", {}).items():
                    if reference in ("CURRENT_IMAGE", "CURRENT_VIDEO"):
                        kind = "image" if reference == "CURRENT_IMAGE" else "video"
                        if kind not in references:
                            raise ValueError(f"current prompt contains no admitted {kind} reference")
                        args["artifacts"][label] = references[kind]
                        changed = True
                function["arguments"] = json.dumps(args)
        return replace(response, body=json.dumps(decoded).encode()) if changed else response

    proxy.handle_chat = respond
