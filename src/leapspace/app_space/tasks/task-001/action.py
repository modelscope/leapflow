# Copyright (c) Alibaba, Inc. and its affiliates.
"""task-001 — Reply to the boss's unread message (LeapChat).

Three parts in one file:

- hooks (in-app process, wired by config.yaml): ``seed`` installs the
  precondition at before_launch — alice's conversation is active, and an
  unread message from boss makes the title badge "LeapChat (1)" the signal
  to notice. ``boss_followup`` answers the user's reply at
  after_message_sent.
- reference(actor): the human reference flow, run host-side through the
  actor's action surface, fully human-real: every click is a real pixel
  click on the element's screen-space center (firing Qt's
  currentItemChanged on the contact switch — the driver's element click
  resolves to the AX Toggle action, which moves AT-SPI selection without
  opening the conversation) and the reply is typed one real keystroke at
  a time, so the in-box observers record the whole session.
- expect(): the verdict, run in-sandbox (``python3 action.py``). Reads the
  ground truth under /tmp/leapspace, prints PASS/FAIL per check, and exits
  non-zero on any failure. Interface-name prechecks (config interface ⊆
  the app's persisted interface) belong to the harness's static layer, not
  here.

Module-level imports must stay in-sandbox safe (stdlib / PyQt6 /
leapspace): the app loads this whole file as hooks.py. Host-only imports
(leapspace.app_space.actor -> cua_sandbox) are deferred into reference().
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from leapspace.app_space.state import check

if TYPE_CHECKING:
    from leapspace.app_space.actor import LeapAppActor

APP_TITLE = "LeapChat"
REPLY = "Got it, see you at 3pm tomorrow."
FOLLOWUP = "If you got this, bring last quarter's report too."
SETTLE_TIMEOUT_S = 10

# The unread lives in a background conversation on purpose: opening boss is
# a real switch that marks it read, so the title badge is a load-bearing
# signal rather than decoration.
SEED = {
    "active": "alice",
    "conversations": {
        "alice": [
            {"sender": "alice", "text": "Up for basketball this weekend?", "read": True},
        ],
        "boss": [
            {"sender": "boss", "text": "Tomorrow's meeting moved to 3pm, please confirm.", "read": False},
        ],
    },
}


# ── hooks (in-app process; wired by config.yaml) ─────────────────────────


def seed(app) -> None:
    app.reset(SEED)


def boss_followup(app) -> None:
    app.inject_message("boss", FOLLOWUP)


# ── reference (host side; the human action sequence) ─────────────────────


async def reference(actor: LeapAppActor) -> None:
    from leapspace.app_space.actor import element_center, find_ax_element

    window = await actor.wait_for_window(APP_TITLE)
    # Precondition: exactly one unread surfaces in the title.
    assert window["title"] == f"{APP_TITLE} (1)", window["title"]

    # Open the boss conversation — the unread the badge announced. A real
    # click on the row's center is what marks the seeded message read (the
    # app's read-marking hangs off Qt's currentItemChanged, which the
    # driver's element click never fires).
    elements = await actor.ax_elements()
    x, y = element_center(find_ax_element(elements, "boss", role="list item"))
    await actor.click(x=x, y=y)

    # Click into the input, then type the reply one real keystroke at a
    # time — the input tap records every char and the verdict compares
    # exactly. The typed text is read back through the same ax dump: a
    # human glances at the field before hitting send.
    elements = await actor.ax_elements()
    x, y = element_center(find_ax_element(elements, "message_input", role="text"))
    await actor.click(x=x, y=y)
    await actor.type_keys(REPLY)
    elements = await actor.ax_elements()
    typed = find_ax_element(elements, "message_input", role="text")["value"]
    assert typed == REPLY, f"typed {typed!r}"

    x, y = element_center(find_ax_element(elements, "send_button", role="push button"))
    await actor.click(x=x, y=y)


# ── expect (in-sandbox; the verdict) ─────────────────────────────────────


def expect(state_root: str = "/tmp/leapspace") -> int:
    state_dir = Path(state_root) / "chat"
    ok = True

    # The reply and the follow-up land on the app's Qt loop after reference
    # returns; poll the ground truth until both are persisted.
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    envelope = None
    while time.monotonic() < deadline:
        try:
            envelope = json.loads((state_dir / "state.json").read_text())
        except (OSError, json.JSONDecodeError):
            envelope = None
        else:
            conversation = envelope["data"].get("conversations", {}).get("boss", [])
            texts = {(message["sender"], message["text"]) for message in conversation}
            if ("me", REPLY) in texts and ("boss", FOLLOWUP) in texts:
                break
        time.sleep(0.5)

    if envelope is None:
        check("state-exists", False, f"no readable state.json under {state_dir}")
        return 1

    events = [
        json.loads(line)
        for line in (state_dir / "events.jsonl").read_text().splitlines()
        if line
    ]
    data = envelope["data"] if isinstance(envelope["data"], dict) else {}
    boss = data.get("conversations", {}).get("boss", [])

    ok &= check(
        "no-a11y-violations",
        envelope["a11y_violations"] == [],
        str(envelope["a11y_violations"]),
    )
    ok &= check(
        "reply-sent",
        any(m["sender"] == "me" and m["text"] == REPLY for m in boss),
        f"boss tail: {boss[-2:]}",
    )
    ok &= check(
        "followup-arrived",
        bool(boss) and boss[-1]["sender"] == "boss" and boss[-1]["text"] == FOLLOWUP,
        f"boss tail: {boss[-1:]}",
    )
    ok &= check(
        "conversation-read",
        bool(boss) and all(m["read"] for m in boss),
        str([m["read"] for m in boss]),
    )
    ok &= check(
        "event-reply",
        any(e["kind"] == "message_sent" and e.get("text") == REPLY for e in events),
    )
    ok &= check(
        "event-followup",
        any(e["kind"] == "message_received" and e.get("text") == FOLLOWUP for e in events),
    )
    badge = [e["text"] for e in events if e["kind"] == "title_status"]
    ok &= check("badge-cleared", badge == ["1", ""], str(badge))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(expect())
