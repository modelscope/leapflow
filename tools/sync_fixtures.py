"""Derive mock-layer LLM fixtures from recorded cassettes.

This is the join between the two test layers. The mock layer keeps its speed and
its ability to enumerate branches, but stops inventing what a provider sends: the
payload *shapes* it feeds into parsers come from real recorded traffic. When the
nightly live lane re-records and a provider has changed its response, the derived
fixtures change with it and the mock layer notices — instead of passing forever
against a body nobody has verified.

Usage::

    python tools/sync_fixtures.py            # write fixtures, report changes
    python tools/sync_fixtures.py --check    # fail if fixtures are out of date

``--check`` is what CI runs: it turns provider drift into a red build with a diff
rather than a silent divergence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests._harness.cassette import CassetteStore  # noqa: E402

CASSETTE_ROOT = REPO_ROOT / "tests" / "_fixtures" / "cassettes"
RECORDING_ROOT = REPO_ROOT / "tests" / "_fixtures" / "recordings"
FIXTURE_ROOT = REPO_ROOT / "tests" / "_fixtures" / "llm_responses"

# Fixture files derived from the store. Each one answers a question the mock layer
# asks: "what does a successful body look like", "what does an error body look
# like", "which usage fields do providers actually send".
SHAPES_FILE = "response_shapes.json"


def _sse_payloads(frames: Iterable[bytes]) -> list[dict[str, Any]]:
    """Parse the JSON objects carried by SSE data frames."""
    payloads: list[dict[str, Any]] = []
    for frame in frames:
        for line in frame.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if not body or body == "[DONE]":
                continue
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
    return payloads


def _key_shape(value: Any) -> Any:
    """Reduce a payload to its key structure, dropping volatile values.

    Only the shape matters: names of fields and their types are what parsers
    depend on, while ids, token counts and text differ on every call.
    """
    if isinstance(value, dict):
        return {key: _key_shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_key_shape(value[0])] if value else []
    return type(value).__name__


def _source_directories() -> list[Path]:
    """Return every per-journey store to distil shapes from.

    Both stores contribute: ``recordings/`` is real provider traffic and is the
    authority on wire shape, while ``cassettes/`` covers the injected failure
    bodies (429, context overflow) that a live provider will not produce on
    demand. Together they describe every shape the parsers must handle.
    """
    directories: list[Path] = []
    for root in (RECORDING_ROOT, CASSETTE_ROOT):
        if root.is_dir():
            directories.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return directories


def collect_shapes() -> dict[str, Any]:
    """Walk every stored exchange and summarize the shapes it contains."""
    successes: list[Any] = []
    errors: list[Any] = []
    chunks: list[Any] = []
    usage_fields: set[str] = set()
    finish_reasons: set[str] = set()
    error_codes: set[str] = set()
    total = 0

    for directory in _source_directories():
        store = CassetteStore(directory)
        for key in store.keys():
            record = store.get(key)
            if record is None:
                continue
            for response in record.responses:
                total += 1
                if response.is_stream:
                    for payload in _sse_payloads(response.frames):
                        chunks.append(_key_shape(payload))
                        usage_fields.update((payload.get("usage") or {}).keys())
                        for choice in payload.get("choices") or []:
                            if choice.get("finish_reason"):
                                finish_reasons.add(str(choice["finish_reason"]))
                    continue
                if not response.body:
                    continue
                try:
                    payload = json.loads(response.body.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if response.status >= 400 or "error" in payload:
                    errors.append(_key_shape(payload))
                    code = (payload.get("error") or {}).get("code")
                    if code:
                        error_codes.add(str(code))
                    continue
                successes.append(_key_shape(payload))
                usage_fields.update((payload.get("usage") or {}).keys())
                for choice in payload.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish_reasons.add(str(choice["finish_reason"]))

    return {
        "_comment": (
            "Generated by tools/sync_fixtures.py from tests/_fixtures/recordings "
            "(real provider traffic) and tests/_fixtures/cassettes (deterministic "
            "replay inputs, including injected failures). Shapes only -- how many "
            "exchanges they were distilled from is reported on stdout, not stored, "
            "because that number changes whenever a journey is added and would make "
            "--check fail on something that is not provider drift. Do not edit by "
            "hand: run `make sync-fixtures` after re-recording."
        ),
        "_stored_responses_seen": total,
        "completion_shapes": _dedupe(successes),
        "chunk_shapes": _dedupe(chunks),
        "error_shapes": _dedupe(errors),
        "usage_fields": sorted(usage_fields),
        "finish_reasons": sorted(finish_reasons),
        "error_codes": sorted(error_codes),
    }


def _contract(shapes: dict[str, Any]) -> dict[str, Any]:
    """Return the part of the summary that --check is allowed to fail on.

    Everything except the corpus inventory. The count of exchanges scanned rises
    whenever a journey is added, and comparing it made ``--check`` red on changes
    that had nothing to do with a provider: the shapes were byte-identical and the
    build failed on ``37`` versus ``200``. A gate that fires on unrelated growth as
    loudly as on real drift stops being read.

    A shrinking corpus needs no gate here: deleting a cassette breaks replay
    immediately and loudly, which is a better signal than a number in a fixture.
    """
    return {key: value for key, value in shapes.items() if not key.startswith("_stored_")}


def _dedupe(shapes: list[Any]) -> list[Any]:
    """Return unique shapes in a stable order."""
    seen: dict[str, Any] = {}
    for shape in shapes:
        seen.setdefault(json.dumps(shape, sort_keys=True), shape)
    return [seen[key] for key in sorted(seen)]


def main(argv: list[str] | None = None) -> int:
    """Write or verify the derived fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed fixtures differ from the cassettes",
    )
    args = parser.parse_args(argv)

    if not CASSETTE_ROOT.is_dir() and not RECORDING_ROOT.is_dir():
        print(
            f"no stored exchanges at {CASSETTE_ROOT} or {RECORDING_ROOT}; "
            "run `make seed-cassettes` first"
        )
        return 1

    shapes = collect_shapes()
    total = int(shapes.get("_stored_responses_seen", 0))
    rendered = json.dumps(_contract(shapes), indent=2, ensure_ascii=False) + "\n"
    target = FIXTURE_ROOT / SHAPES_FILE

    if args.check:
        if not target.is_file():
            print(f"missing derived fixture {target}; run `make sync-fixtures`")
            return 1
        current = target.read_text(encoding="utf-8")
        if current != rendered:
            print(
                f"{target.relative_to(REPO_ROOT)} is out of date with the stored "
                "exchanges — a provider response shape changed.\n"
                "Run `make sync-fixtures`, review the diff, and commit it."
            )
            return 1
        print(f"{target.relative_to(REPO_ROOT)} is up to date")
        return 0

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    changed = not target.is_file() or target.read_text(encoding="utf-8") != rendered
    target.write_text(rendered, encoding="utf-8")
    verb = "updated" if changed else "unchanged"
    print(
        f"{verb}: {target.relative_to(REPO_ROOT)} "
        f"({total} stored responses, "
        f"{len(shapes['completion_shapes'])} completion shapes, "
        f"{len(shapes['error_shapes'])} error shapes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
