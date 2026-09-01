"""LeapChat — LeapSpace mock chat app (PyQt6, BaseLeapApp contract).

Communication is mocked: outbound messages only append to local state;
inbound messages arrive via ``<state_dir>/inbox.jsonl`` — one JSON object
per line, ``{"from": <contact>, "text": <body>}`` — which the harness
appends to through the sandbox fs channel. A timer polls the file; this is
the app's only I/O beyond the base class's ground-truth persistence.

Task-facing surface: ``inject_message(sender, text)`` is the public
injection API (task hooks and the inbox transport share it), and the app
declares the ``after_message_sent`` hook point, fired after a send lands.

Run in-sandbox with the leapspace package importable:

    PYTHONPATH=/apps python3 -m leapspace.app_space.apps.chat
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QTextEdit,
)

from leapspace.app_space.apps import BaseLeapApp

INBOX_POLL_MS = 500


class ChatApp(BaseLeapApp):
    """Contact list + per-contact history + input/send. Mocked transport."""

    app_id = "chat"
    app_title = "LeapChat"
    version = "1"
    supported_hooks = ("before_launch", "after_message_sent")

    # ------------------------------------------------------------------
    # BaseLeapApp contract
    # ------------------------------------------------------------------

    def build_ui(self) -> None:
        # Model: {contact: [{"sender", "text", "ts", "read"}]}; contacts are
        # data-driven (reset seed or inbox arrivals), never hardcoded.
        self._conversations: dict[str, list[dict[str, Any]]] = {}
        self._active: str | None = None
        self._inbox_seen = 0

        central = QWidget()
        root = QVBoxLayout(central)

        paned = QHBoxLayout()
        self.contact_list = self.bind(QListWidget(), "contact_list")
        self.contact_list.currentItemChanged.connect(self._on_contact_changed)
        paned.addWidget(self.contact_list)

        right = QVBoxLayout()
        self.history = self.bind(QTextEdit(readOnly=True), "message_history")
        right.addWidget(self.history)

        entry_row = QHBoxLayout()
        self.input = self.bind(
            QLineEdit(placeholderText="Type a message"), "message_input"
        )
        send = self.bind(QPushButton("Send"), "send_button")
        send.clicked.connect(self._on_send)
        self.input.returnPressed.connect(self._on_send)
        entry_row.addWidget(self.input)
        entry_row.addWidget(send)
        right.addLayout(entry_row)

        paned.addLayout(right)
        root.addLayout(paned)
        self.status = QLabel("Ready")
        root.addWidget(self.status)
        self.setCentralWidget(central)
        self.resize(480, 360)

        self._inbox_timer = QTimer(self)
        self._inbox_timer.timeout.connect(self._poll_inbox)
        self._inbox_timer.start(INBOX_POLL_MS)

    def reset(self, data: dict[str, Any]) -> None:
        """Load seed conversations; per-message read flags are honored."""
        self._conversations = {}
        for contact, messages in data.get("conversations", {}).items():
            self._conversations[contact] = [
                {
                    "sender": message["sender"],
                    "text": message["text"],
                    "ts": message.get("ts", 0.0),
                    "read": message.get("read", False),
                }
                for message in messages
            ]
        self.contact_list.clear()
        self.contact_list.addItems(self._conversations.keys())

        # Activate without marking read: switching in the UI is what marks a
        # conversation read, and a seed may deliberately plant it unread.
        self._active = data.get("active") or next(iter(self._conversations), None)
        row = list(self._conversations).index(self._active) if self._active else -1
        self.contact_list.blockSignals(True)
        self.contact_list.setCurrentRow(row)
        self.contact_list.blockSignals(False)
        self._refresh_history()
        self._update_unread_status()

    def state(self) -> dict[str, Any]:
        return {"conversations": self._conversations, "active": self._active}

    # ------------------------------------------------------------------
    # UI handlers
    # ------------------------------------------------------------------

    def _on_contact_changed(self, current, _previous) -> None:
        if current is None:
            return
        self._active = current.text()
        # Opening a conversation marks it read, like any real chat client.
        for message in self._conversations.get(self._active, []):
            message["read"] = True
        self._refresh_history()
        self._update_unread_status()

    def _on_send(self) -> None:
        text = self.input.text().strip()
        if not text or self._active is None:
            return
        self._conversations.setdefault(self._active, []).append(
            {"sender": "me", "text": text, "ts": time.time(), "read": True}
        )
        self.emit("message_sent", to=self._active, text=text)
        # Fired after the emit above: hooks (e.g. a boss follow-up) observe a
        # state.json that already contains the sent message.
        self._execute_hooks("after_message_sent")
        self._refresh_history()
        self.input.clear()
        self.status.setText(f"Sent to {self._active}")

    # ------------------------------------------------------------------
    # Inbound mock transport + injection API
    # ------------------------------------------------------------------

    def inject_message(self, sender: str, text: str) -> None:
        """Inject an inbound message (task hooks and inbox transport share this)."""
        # A message from an unknown contact opens a new conversation, but the
        # view does not auto-switch: the unread count in the title is the
        # signal the agent is supposed to notice.
        conversation = self._conversations.setdefault(sender, [])
        conversation.append(
            {
                "sender": sender,
                "text": text,
                "ts": time.time(),
                "read": sender == self._active,
            }
        )
        if self.contact_list.findItems(sender, Qt.MatchFlag.MatchExactly) == []:
            self.contact_list.addItem(sender)
        if sender == self._active:
            self._refresh_history()
        self.notify(sender, text)
        self.emit("message_received", sender=sender, text=text)
        self._update_unread_status()

    def _poll_inbox(self) -> None:
        inbox = self.state_dir / "inbox.jsonl"
        if not inbox.exists():
            return
        lines = inbox.read_text().splitlines()
        for line in lines[self._inbox_seen :]:
            self._inbox_seen += 1
            try:
                message = json.loads(line)
                self.inject_message(message["from"], message["text"])
            except (json.JSONDecodeError, KeyError) as exc:
                self.emit("inbox_error", line=line, error=str(exc))

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def _refresh_history(self) -> None:
        lines = [
            f"{message['sender']}: {message['text']}"
            for message in self._conversations.get(self._active, [])
        ]
        self.history.setPlainText("\n".join(lines))

    def _update_unread_status(self) -> None:
        unread = sum(
            not message["read"]
            for messages in self._conversations.values()
            for message in messages
        )
        self.set_title_status(str(unread) if unread else "")


if __name__ == "__main__":
    sys.exit(ChatApp.launch())
