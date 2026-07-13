"""Feishu action pack for the App Connector execution layer."""
from __future__ import annotations

from leapflow.gateway.connectors.protocol import ActionSpec, BackendKind
from leapflow.security.actions import ActionEffect

FEISHU_ACTION_SPECS: dict[str, ActionSpec] = {
    "im.send_message": ActionSpec(
        name="im.send_message",
        backend_kind=BackendKind.CLI.value,
        description="Send a text message to a Feishu/Lark chat.",
        effect=ActionEffect.SEND.value,
        schema={
            "type": "object",
            "required": ["chat_id", "text"],
            "properties": {
                "chat_id": {"type": "string"},
                "text": {"type": "string"},
                "thread_id": {"type": "string"},
            },
        },
        backend_config={
            "argv": ("im", "+messages-send", "--chat-id", "{chat_id}", "--text", "{text}"),
            "dry_run_argv": (
                "im", "+messages-send", "--chat-id", "{chat_id}", "--text", "{text}", "--dry-run",
            ),
            "approval_summary": (
                "Send a Feishu message to chat {chat_id} as {identity} "
                "using {binary} profile {profile}."
            ),
            "next_steps": [
                "Verify the returned message_id/resource_id if the caller needs delivery tracking.",
                "If sending fails, confirm chat_id and bot/user permissions in the selected Feishu tenant.",
                "Run lark-cli auth status --json to verify the current profile remains authorized.",
            ],
            "timeout_s": 30,
        },
        risk_level="high",
        output_policy="summary",
    ),
    "docs.create_markdown": ActionSpec(
        name="docs.create_markdown",
        backend_kind=BackendKind.CLI.value,
        description="Create a Feishu/Lark document from markdown content.",
        effect=ActionEffect.WRITE.value,
        schema={
            "type": "object",
            "required": ["title", "markdown"],
            "properties": {
                "title": {"type": "string"},
                "markdown": {"type": "string"},
                "folder_token": {"type": "string"},
            },
        },
        backend_config={
            "argv": ("docs", "+create", "--title", "{title}", "--markdown", "{markdown}"),
            "dry_run_argv": (
                "docs", "+create", "--title", "{title}", "--markdown", "{markdown}", "--dry-run",
            ),
            "timeout_s": 60,
        },
        risk_level="medium",
        output_policy="summary",
    ),
    "calendar.create_event": ActionSpec(
        name="calendar.create_event",
        backend_kind=BackendKind.CLI.value,
        description="Create a calendar event.",
        effect=ActionEffect.WRITE.value,
        schema={
            "type": "object",
            "required": ["summary", "start_time", "end_time"],
            "properties": {
                "summary": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "attendees": {"type": "string"},
            },
        },
        backend_config={
            "argv": (
                "calendar", "+events-create", "--summary", "{summary}",
                "--start-time", "{start_time}", "--end-time", "{end_time}",
            ),
            "dry_run_argv": (
                "calendar", "+events-create", "--summary", "{summary}",
                "--start-time", "{start_time}", "--end-time", "{end_time}", "--dry-run",
            ),
            "timeout_s": 45,
        },
        risk_level="high",
        output_policy="summary",
    ),
    "drive.search": ActionSpec(
        name="drive.search",
        backend_kind=BackendKind.CLI.value,
        description="Search Feishu/Lark Drive resources.",
        effect=ActionEffect.READ.value,
        schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        },
        backend_config={"argv": ("drive", "+search", "--query", "{query}"), "timeout_s": 30},
        risk_level="medium",
        output_policy="summary",
    ),
    "sheets.append_row": ActionSpec(
        name="sheets.append_row",
        backend_kind=BackendKind.CLI.value,
        description="Append one row to a Feishu/Lark sheet.",
        effect=ActionEffect.WRITE.value,
        schema={
            "type": "object",
            "required": ["spreadsheet_token", "sheet_id", "values"],
            "properties": {
                "spreadsheet_token": {"type": "string"},
                "sheet_id": {"type": "string"},
                "values": {"type": "array"},
            },
        },
        backend_config={
            "argv": (
                "sheets", "+append-row", "--spreadsheet-token", "{spreadsheet_token}",
                "--sheet-id", "{sheet_id}", "--values", "{values}",
            ),
            "dry_run_argv": (
                "sheets", "+append-row", "--spreadsheet-token", "{spreadsheet_token}",
                "--sheet-id", "{sheet_id}", "--values", "{values}", "--dry-run",
            ),
            "timeout_s": 45,
        },
        risk_level="medium",
        output_policy="summary",
    ),
    "mail.search_unread": ActionSpec(
        name="mail.search_unread",
        backend_kind=BackendKind.CLI.value,
        description="Search unread Feishu/Lark mail.",
        effect=ActionEffect.READ.value,
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        },
        backend_config={"argv": ("mail", "+search", "--unread"), "timeout_s": 30},
        risk_level="high",
        output_policy="summary",
    ),
    "task.create": ActionSpec(
        name="task.create",
        backend_kind=BackendKind.CLI.value,
        description="Create a Feishu/Lark task.",
        effect=ActionEffect.WRITE.value,
        schema={
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "due_time": {"type": "string"},
            },
        },
        backend_config={
            "argv": ("task", "+create", "--title", "{title}"),
            "dry_run_argv": ("task", "+create", "--title", "{title}", "--dry-run"),
            "timeout_s": 30,
        },
        risk_level="medium",
        output_policy="summary",
    ),
}

ACTION_SPECS = FEISHU_ACTION_SPECS


def get_action_spec(action: str) -> ActionSpec | None:
    """Return a Feishu action spec by domain.operation name."""
    return FEISHU_ACTION_SPECS.get(action)
