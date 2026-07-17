"""Slash-command completion for the TUI input area.

Provides interactive slash-command suggestions for the TextArea widget.
The completer keeps command discovery read-only and lightweight: it only uses
the static command registry entries already loaded by the CLI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prompt_toolkit.document import Document


_MAX_DESCRIPTION_WIDTH = 88


def _truncate_meta(text: str, *, width: int = _MAX_DESCRIPTION_WIDTH) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= width:
        return normalized
    return normalized[: max(0, width - 1)].rstrip() + "…"


_CONFIG_ACTIONS: tuple[tuple[str, str], ...] = (
    ("show", "Show effective config values"),
    ("list", "List keys with values, types, scopes, and descriptions"),
    ("keys", "List writable keys only"),
    ("get", "Show one config value"),
    ("set", "Set one config value"),
    ("unset", "Remove one saved config value"),
    ("sources", "Show config source files"),
    ("llm", "Configure the primary LLM provider"),
    ("secret", "Manage stored secret refs"),
)

_CONFIG_LLM_ACTIONS: tuple[tuple[str, str], ...] = (
    ("show", "Show current LLM configuration"),
    ("set", "Set LLM model, base URL, or API key"),
)

_CONFIG_LLM_SET_FLAGS: tuple[tuple[str, str], ...] = (
    ("--model", "Model name (e.g. qwen3.7-plus)"),
    ("--base-url", "OpenAI-compatible endpoint URL"),
    ("--api-key", "API key (stored in the local vault)"),
    ("--context-length", "Context window size (tokens)"),
    ("--max-retries", "Max request retries"),
    ("--scope", "Config scope: profile"),
)

_CONFIG_SECRET_ACTIONS: tuple[tuple[str, str], ...] = (
    ("list", "List stored secret refs"),
    ("set", "Store a secret value: set <ref> <value>"),
    ("get", "Read a secret value (add --reveal to show)"),
    ("delete", "Remove a stored secret"),
)


@dataclass(frozen=True)
class ConfigCompletionField:
    """Compact config field metadata used by slash completion."""

    key: str
    description: str
    value_type: str = ""
    value_hint: str = ""
    category: str = ""


class SlashCommandCompleter(Completer):
    """Complete slash commands with display text and descriptions.

    The returned ``Completion`` objects are intentionally prompt_toolkit-native:
    ``display`` renders the left command column and ``display_meta`` renders the
    right description column used by ``CompletionsMenu``.  This gives LeapFlow a
    Hermes-style slash menu without owning a custom focus or scroll state.
    """

    def __init__(
        self,
        commands: Sequence[tuple[str, str]],
        config_fields: Sequence[object] = (),
    ) -> None:
        self._commands = tuple(commands)
        self._config_fields = tuple(_normalize_config_field(item) for item in config_fields)

    @property
    def commands(self) -> tuple[tuple[str, str], ...]:
        """Return the immutable command catalog used by this completer."""
        return self._commands

    def get_completions(
        self,
        document: "Document",
        complete_event,
    ) -> "Iterable[Completion]":
        text = document.text_before_cursor.lstrip()
        if not text:
            return
        has_slash = text.startswith("/")
        if not has_slash and " " in text:
            return
        if text.startswith("/config "):
            yield from self._config_completions(text)
            return

        query = text.lstrip("/").lower()
        for command, description in self._commands:
            command_lower = command.lower()
            if query and not command_lower.startswith(query):
                continue
            completion = f"/{command}" if has_slash else command
            yield Completion(
                completion,
                start_position=-len(text),
                display=f"/{command}",
                display_meta=_truncate_meta(description),
            )

    def _config_completions(self, text: str) -> "Iterable[Completion]":
        tail = text[len("/config "):]
        parts = tail.split()
        ends_space = tail.endswith(" ")
        if not parts:
            yield from _complete_pairs(_CONFIG_ACTIONS, "", 0)
            return

        action = parts[0]
        if len(parts) == 1 and not ends_space:
            yield from _complete_pairs(_CONFIG_ACTIONS, action, -len(action))
            return

        if action in {"show", "get", "set", "unset"}:
            if len(parts) == 1 and ends_space:
                yield from self._config_key_completions("", 0)
                return
            if len(parts) == 2 and not ends_space:
                yield from self._config_key_completions(parts[1], -len(parts[1]))
                return
            if action == "set" and len(parts) == 2 and ends_space:
                yield from self._config_value_completions(parts[1], "", 0)
                return
            if action == "set" and len(parts) == 3 and not ends_space:
                yield from self._config_value_completions(parts[1], parts[2], -len(parts[2]))
                return

        if action == "list":
            if len(parts) == 1 and ends_space:
                yield from self._config_category_completions("", 0)
                return
            if len(parts) == 2 and not ends_space:
                yield from self._config_category_completions(parts[1], -len(parts[1]))
            return

        if action == "llm":
            if len(parts) == 1 and ends_space:
                yield from _complete_pairs(_CONFIG_LLM_ACTIONS, "", 0)
                return
            if len(parts) == 2 and not ends_space:
                yield from _complete_pairs(_CONFIG_LLM_ACTIONS, parts[1], -len(parts[1]))
                return
            if len(parts) >= 2 and parts[1] == "set":
                last = parts[-1]
                if not ends_space and last.startswith("-"):
                    # Completing a flag name.
                    yield from _complete_pairs(_CONFIG_LLM_SET_FLAGS, last, -len(last))
                elif ends_space and not last.startswith("-"):
                    # Finished a value (or bare `set`); offer the next flag. After a
                    # flag token the user needs to type its value, so suggest nothing.
                    yield from _complete_pairs(_CONFIG_LLM_SET_FLAGS, "", 0)
            return

        if action == "secret":
            if len(parts) == 1 and ends_space:
                yield from _complete_pairs(_CONFIG_SECRET_ACTIONS, "", 0)
                return
            if len(parts) == 2 and not ends_space:
                yield from _complete_pairs(_CONFIG_SECRET_ACTIONS, parts[1], -len(parts[1]))
            return

    def _config_key_completions(self, prefix: str, start_position: int) -> "Iterable[Completion]":
        query = prefix.lower()
        for item in self._config_fields:
            if query and not item.key.lower().startswith(query):
                continue
            yield Completion(
                item.key,
                start_position=start_position,
                display=item.key,
                display_meta=_truncate_meta(item.description),
            )

    def _config_category_completions(self, prefix: str, start_position: int) -> "Iterable[Completion]":
        seen: set[str] = set()
        query = prefix.lower()
        for item in self._config_fields:
            category = item.key.split(".", 1)[0]
            if category in seen or (query and not category.startswith(query)):
                continue
            seen.add(category)
            yield Completion(
                category,
                start_position=start_position,
                display=category,
                display_meta=_truncate_meta(f"List {item.category or category} config fields"),
            )

    def _config_value_completions(self, key: str, prefix: str, start_position: int) -> "Iterable[Completion]":
        field = next((item for item in self._config_fields if item.key == key), None)
        if field is None:
            return
        choices = _value_choices(field)
        query = prefix.lower()
        for choice in choices:
            if query and not choice.lower().startswith(query):
                continue
            yield Completion(
                choice,
                start_position=start_position,
                display=choice,
                display_meta=_truncate_meta(field.value_hint or field.description),
            )


def _complete_pairs(pairs: Sequence[tuple[str, str]], prefix: str, start_position: int) -> "Iterable[Completion]":
    query = prefix.lower()
    for value, description in pairs:
        if query and not value.lower().startswith(query):
            continue
        yield Completion(
            value,
            start_position=start_position,
            display=value,
            display_meta=_truncate_meta(description),
        )


def _normalize_config_field(item: object) -> ConfigCompletionField:
    if isinstance(item, ConfigCompletionField):
        return item
    if isinstance(item, dict):
        return ConfigCompletionField(
            key=str(item.get("key") or ""),
            description=str(item.get("description") or ""),
            value_type=str(item.get("type") or item.get("value_type") or ""),
            value_hint=str(item.get("value_hint") or ""),
            category=str(item.get("category") or ""),
        )
    if hasattr(item, "key"):
        return ConfigCompletionField(
            key=str(getattr(item, "key", "")),
            description=str(getattr(item, "description", "")),
            value_type=str(getattr(item, "value_type", "")),
            value_hint=str(getattr(item, "value_hint", "")),
            category=str(getattr(item, "category", "")),
        )
    if isinstance(item, tuple) and len(item) >= 4:
        return ConfigCompletionField(str(item[0]), str(item[1]), str(item[2]), str(item[3]))
    if isinstance(item, tuple) and len(item) >= 2:
        return ConfigCompletionField(str(item[0]), str(item[1]))
    return ConfigCompletionField(str(item), "")


def _value_choices(field: ConfigCompletionField) -> tuple[str, ...]:
    if field.value_type == "bool" or field.value_hint == "true|false":
        return ("true", "false")
    if "|" in field.value_hint and " " not in field.value_hint:
        return tuple(part for part in field.value_hint.split("|") if part)
    return ()


def build_completer(
    commands: Sequence[tuple[str, str]],
    config_fields: Sequence[object] = (),
) -> ThreadedCompleter:
    """Create a threaded slash-command completer for the TextArea."""
    return ThreadedCompleter(SlashCommandCompleter(commands or [], config_fields=config_fields))
