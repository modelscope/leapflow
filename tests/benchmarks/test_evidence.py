# Copyright (c) Alibaba, Inc. and its affiliates.
"""Evidence JSONL, content hash, and path containment tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.evidence import EvidenceStore
from benchmarks.models import EvidenceRef


def _artifact(store: EvidenceStore, ref: EvidenceRef) -> Path:
    path = store.root / ref.path
    assert not Path(ref.path).is_absolute()
    assert path.resolve().is_relative_to(store.root)
    return path


def test_add_jsonl_writes_valid_records_and_index_metadata(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    records = ({"step": 1, "value": "开始"}, {"step": 2, "ok": True})

    ref = store.add_jsonl(records, metadata={"scenario": "safe-stop"})

    artifact = _artifact(store, ref)
    assert [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()] == list(
        records
    )
    index_records = [
        json.loads(line) for line in store.index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(index_records) == 1
    assert index_records[0]["path"] == ref.path
    assert index_records[0]["metadata"] == {"scenario": "safe-stop"}
    assert ref.kind == "trajectory"
    assert ref.media_type == "application/x-ndjson"


def test_content_hash_is_sha256_and_duplicate_bytes_are_deduplicated(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    payload = b"raw\x00benchmark-output\n"

    first = store.add_bytes(payload, suffix="dat")
    second = store.add_bytes(payload, suffix="dat")

    expected = hashlib.sha256(payload).hexdigest()
    assert first.content_hash == f"sha256:{expected}"
    assert first.path == second.path
    assert first.size_bytes == len(payload)
    assert _artifact(store, first).read_bytes() == payload
    assert store.verify(first)
    assert len(store.list_refs()) == 2


def test_verify_detects_tampering_and_size_mismatch(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    ref = store.add_text("original")
    _artifact(store, ref).write_text("tampered", encoding="utf-8")

    assert not store.verify(ref)

    fresh = store.add_text("fresh")
    wrong_size = EvidenceRef(
        path=fresh.path,
        content_hash=fresh.content_hash,
        size_bytes=fresh.size_bytes + 1,
    )
    assert not store.verify(wrong_size)


def test_verify_rejects_parent_and_absolute_path_escape(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "store")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()

    parent_escape = EvidenceRef(
        path="../outside.txt",
        content_hash=f"sha256:{digest}",
        size_bytes=outside.stat().st_size,
    )
    absolute_escape = EvidenceRef(
        path=str(outside),
        content_hash=f"sha256:{digest}",
        size_bytes=outside.stat().st_size,
    )

    assert not store.verify(parent_escape)
    assert not store.verify(absolute_escape)


def test_list_refs_skips_invalid_jsonl_and_filters_kind(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    log_ref = store.add_log("stdout")
    json_ref = store.add_json({"status": "ok"}, kind="report")
    with store.index_path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")

    assert store.list_refs(kind="log") == (log_ref,)
    assert store.list_refs(kind="report") == (json_ref,)
    assert len(store.list_refs()) == 2
