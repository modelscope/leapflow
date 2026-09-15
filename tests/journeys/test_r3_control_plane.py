# Copyright (c) Alibaba, Inc. and its affiliates.
"""R3 — the control plane: slash commands, layered config, secrets, cancellation.

`leap config` / `/config` is the only sanctioned way to change durable settings,
so its whole path has to hold across a process boundary: the mutation must be
written to the profile, reloaded by the running daemon, and echoed back on the
mutation payload — a daemon-mode client is a separate process and cannot observe
a reload it was not told about. Secrets must land in the vault as refs, never as
plaintext on disk.

This journey needs no LLM semantics, so it is replay-only: the live lane would
add cost without adding signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._harness.cassette_proxy import answer, scripted
from tests._harness.journey import JourneyFactory
from tests._harness.leapd import await_for

# Journey metadata read by tools/impact.py (see test_r1_conversation.py).
SUBJECT_PATHS = (
    "src/leapflow/config.py",
    "src/leapflow/config_loader.py",
    "src/leapflow/config_service.py",
    "src/leapflow/layout.py",
    "src/leapflow/security/",
    "src/leapflow/cli/commands/",
)

# No LLM semantics: every assertion is about config layering, the vault, and
# slash-command payloads. A live run would spend tokens for no extra signal.
LIVE_SIGNAL = False

SESSION = "r3-control-plane"
SECRET_VALUE = "sk-r3-journey-secret-value"

# A durable, hot-reloadable integer the journey harness does not pin through the
# environment. Env is the highest-priority config layer, so a key the harness
# exports could never change effective value here — which would silently turn
# this phase into a no-op assertion.
MUTABLE_KEY = "memory.working_max_tokens"
MUTATED_VALUE = "4096"

# Pinned by the harness on purpose (cassette fingerprints depend on it), and used
# below to assert the layering contract rather than to test mutation.
ENV_PINNED_KEY = "llm.context_length"


def _plaintext_hits(root: Path, needle: str) -> list[str]:
    """Return every readable file under ``root`` containing ``needle``."""
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if needle in content:
            hits.append(str(path.relative_to(root)))
    return hits


@pytest.mark.asyncio
async def test_r3_control_plane(journeys: JourneyFactory) -> None:
    """Slash commands observe, mutate, reload, and cancel through the daemon."""
    journey = journeys(
        "r3_control_plane",
        script=scripted(answer("Acknowledged.")),
        deadline_s=90.0,
        # One closing turn; the rest of the journey is pure control plane.
        max_llm_calls=6,
        max_llm_tokens=80_000,
    )
    workspace = journey.workspace("ctrl")
    client = journey.client()

    with journey.phase("observe: /status and /usage answer without a turn"):
        status = await client.command_execute("status", session_id=SESSION)
        assert status.get("ok") is not False, f"/status failed: {status}"
        usage = await client.command_execute("usage", session_id=SESSION)
        assert usage.get("ok") is not False, f"/usage failed: {usage}"

    with journey.phase("discover: the config catalog is self-describing"):
        keys = await client.command_execute("config", "keys", session_id=SESSION)
        assert keys.get("ok") is True, f"/config keys failed: {keys}"
        assert keys.get("mode") == "keys"
        assert keys.get("sources"), "the writable-key catalog is empty"

        listing = await client.command_execute("config", "list llm", session_id=SESSION)
        assert listing.get("mode") == "list"
        fields = listing.get("fields") or []
        assert fields, "/config list llm returned no fields"
        described = {str(field.get("key", "")) for field in fields}
        assert "llm.model" in described, f"llm.model is not discoverable: {sorted(described)}"

    with journey.phase("detail: /config show names one field precisely"):
        detail = await client.command_execute("config", "show llm.model", session_id=SESSION)
        assert detail.get("mode") == "show_detail"
        field = detail.get("field") or {}
        assert field.get("key") == "llm.model"

    with journey.phase("mutate: /config set persists, reloads, and echoes runtime state"):
        before = await _get_value(client, MUTABLE_KEY)
        assert before != MUTATED_VALUE, "fixture would not change anything"

        mutation = await client.command_execute(
            "config", f"set {MUTABLE_KEY} {MUTATED_VALUE}", session_id=SESSION
        )
        assert mutation.get("ok") is True, f"/config set failed: {mutation}"
        assert mutation.get("mode") == "mutation"
        assert MUTABLE_KEY in (mutation.get("changed_keys") or []), (
            f"the mutation reported no changed key: {mutation}"
        )

        # A daemon-mode client is a separate process: it can only learn runtime
        # state from what the reply carries, so the mutation payload must ship the
        # values the status bar renders.
        live = await client.status(SESSION)
        assert mutation.get("model") == live.get("model"), (
            f"mutation payload model {mutation.get('model')!r} disagrees with the "
            f"daemon's {live.get('model')!r}"
        )
        assert mutation.get("llm_context_length") == live.get("llm_context_length"), (
            "the mutation payload did not carry the runtime context length the "
            "status bar renders"
        )

    with journey.phase("reload: the running daemon serves the new value"):
        served = await await_for(
            lambda: _value_equals(client, MUTABLE_KEY, MUTATED_VALUE),
            timeout_s=15.0,
            what=f"daemon to serve the reloaded {MUTABLE_KEY}",
        )
        assert served, (
            f"{MUTABLE_KEY} was written but the running daemon never reloaded it "
            "(if the harness now exports it as LEAPFLOW_*, the process override "
            "outranks the durable write and this key can no longer be tested here)"
        )

    with journey.phase("layering: a process override outranks a durable write"):
        # The documented precedence is env > workspace > profile > user, and it is a
        # real footgun: a user who exported LEAPFLOW_* once sees `/config set`
        # report success while nothing changes. The write must still be accepted,
        # and the effective value must still come from the override.
        pinned_before = await _get_value(client, ENV_PINNED_KEY)
        result = await client.command_execute(
            "config", f"set {ENV_PINNED_KEY} 24576", session_id=SESSION
        )
        assert result.get("ok") is True, f"durable write refused: {result}"
        pinned_after = await _get_value(client, ENV_PINNED_KEY)
        assert pinned_after == pinned_before != "24576", (
            f"{ENV_PINNED_KEY} is pinned to {pinned_before!r} by the harness "
            f"environment but a durable write changed the effective value to "
            f"{pinned_after!r} — config precedence changed"
        )

    with journey.phase("secrets: credentials become refs, never plaintext on disk"):
        secret = await client.command_execute(
            "config", f"secret set llm.primary.api_key {SECRET_VALUE}", session_id=SESSION
        )
        assert secret.get("ok") is True, f"/config secret set failed: {secret}"

        leaks = _plaintext_hits(journey.daemon.data_dir, SECRET_VALUE)
        assert leaks == [], (
            f"the secret was written as plaintext to {leaks} — long-lived "
            "credentials must be vault-encrypted and referenced as secret:// refs"
        )

    with journey.phase("cancel: an idle cancel is safe and reports honestly"):
        cancelled = await client.engine_cancel()
        assert isinstance(cancelled, bool), f"engine.cancel returned {cancelled!r}"

    with journey.phase("still usable: a turn works after the control-plane churn"):
        events: list[Any] = []
        async for event in client.engine_chat(
            "Anything to report?", session_id=SESSION, workspace_root=str(workspace)
        ):
            events.append(event)
        errors = [event.content for event in events if event.type == "error"]
        assert not errors, f"turn broke after config mutations: {errors}"

    journey.finish()


async def _get_value(client: Any, key: str) -> str:
    """Return the effective ``/config get`` value for ``key`` as reported.

    ``/config get`` renders values as display strings, so comparisons here stay
    textual rather than guessing at the underlying Python type.
    """
    payload = await client.command_execute("config", f"get {key}", session_id=SESSION)
    values = payload.get("values") or []
    assert values, f"/config get {key} returned nothing: {payload}"
    return str(values[0].get("value"))


async def _value_equals(client: Any, key: str, expected: Any) -> bool:
    """True once the daemon serves ``expected`` for ``key``."""
    return await _get_value(client, key) == str(expected)
