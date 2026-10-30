# Copyright (c) Alibaba, Inc. and its affiliates.
"""Append-only JSON Lines evidence store for benchmark artifacts.

The store indexes logs, JSON, trajectories, videos, FailureEnvelopes, and
side-effect evidence.  Every artifact is content-addressed with SHA-256,
relative path, size, media type, and creation timestamp.  Raw external
output is preserved verbatim.

Uses plain JSON Lines rather than DuckDB to avoid additional file locks
when multiple benchmark workers write concurrently.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import mimetypes
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.models import EvidenceRef


_INDEX_FILE = "evidence.jsonl"


def evidence_root(adapter_id: str) -> Path:
    """Resolve a run-isolated evidence path for an adapter.

    CLI and runner invocations install a persistent profile-owned root in the
    runtime context.  Direct adapter calls retain a temporary fallback for
    isolated unit tests and must not be used as release evidence.
    """
    from benchmarks.runtime import current_runtime_context

    context = current_runtime_context()
    if context.evidence_root and context.run_id:
        _validate_path_segment(context.run_id, "run_id")
        _validate_path_segment(adapter_id, "adapter_id")
        return Path(context.evidence_root).expanduser().resolve() / context.run_id / adapter_id
    _validate_path_segment(adapter_id, "adapter_id")
    return Path(tempfile.gettempdir()) / "leapflow-benchmarks" / adapter_id


def _validate_path_segment(value: str, field: str) -> None:
    """Reject path-like values before deriving an evidence directory."""
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field} must be one non-empty path segment")


def _json_default(obj: Any) -> Any:
    """Convert common domain objects to JSON-compatible values."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        return {"__bytes_hex__": obj.hex()}
    return repr(obj)


class EvidenceStore:
    """Content-addressed, append-only evidence store.

    Thread-safe within one process.  Each artifact is stored under
    ``artifacts/<sha256-prefix>/<sha256>.<ext>`` and indexed by one JSON
    object per line in ``evidence.jsonl``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.index_path = self.root / _INDEX_FILE
        self._lock = threading.Lock()

    def _ensure_dirs(self) -> None:
        """Create store directories lazily (never at import time)."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, digest: str, suffix: str) -> Path:
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self.artifacts_dir / digest[:2] / f"{digest}{safe_suffix}"

    def _index_ref(self, ref: EvidenceRef, metadata: Mapping[str, Any] | None = None) -> None:
        """Append one evidence record to the JSON Lines index."""
        record = ref.to_dict()
        if metadata:
            record["metadata"] = dict(metadata)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default)

        with self._lock:
            self._ensure_dirs()
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def add_bytes(
        self,
        data: bytes,
        *,
        kind: str = "file",
        suffix: str = ".bin",
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceRef:
        """Store bytes verbatim and return their evidence reference."""
        digest = hashlib.sha256(data).hexdigest()
        target = self._artifact_path(digest, suffix)
        self._ensure_dirs()
        target.parent.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            with self._lock:
                if not target.exists():
                    target.write_bytes(data)

        ref = EvidenceRef(
            path=str(target.relative_to(self.root)),
            content_hash=f"sha256:{digest}",
            size_bytes=len(data),
            media_type=media_type,
            created_at=time.time(),
            kind=kind,
        )
        self._index_ref(ref, metadata)
        return ref

    def add_text(
        self,
        text: str,
        *,
        kind: str = "log",
        suffix: str = ".log",
        media_type: str = "text/plain; charset=utf-8",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceRef:
        """Store text exactly as UTF-8 bytes (raw external output preserved)."""
        return self.add_bytes(
            text.encode("utf-8"),
            kind=kind,
            suffix=suffix,
            media_type=media_type,
            metadata=metadata,
        )

    def add_json(
        self,
        data: Any,
        *,
        kind: str = "json",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceRef:
        """Store structured data as canonical JSON."""
        text = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        return self.add_text(
            text,
            kind=kind,
            suffix=".json",
            media_type="application/json",
            metadata=metadata,
        )

    def add_jsonl(
        self,
        records: Sequence[Any],
        *,
        kind: str = "trajectory",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceRef:
        """Store a sequence of records in standard JSON Lines format."""
        lines = [
            json.dumps(r, ensure_ascii=False, sort_keys=True, default=_json_default)
            for r in records
        ]
        text = "\n".join(lines) + ("\n" if lines else "")
        return self.add_text(
            text,
            kind=kind,
            suffix=".jsonl",
            media_type="application/x-ndjson",
            metadata=metadata,
        )

    def add_file(
        self,
        source: str | Path,
        *,
        kind: str = "file",
        media_type: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceRef:
        """Copy an existing file into the content-addressed store."""
        path = Path(source).expanduser()
        data = path.read_bytes()
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.add_bytes(
            data,
            kind=kind,
            suffix=path.suffix or ".bin",
            media_type=media_type or guessed,
            metadata={"source_name": path.name, **dict(metadata or {})},
        )

    # ── Typed evidence helpers ────────────────────────────────

    def add_log(self, text: str, **metadata: Any) -> EvidenceRef:
        """Store process output or diagnostic logs verbatim."""
        return self.add_text(text, kind="log", metadata=metadata)

    def add_trajectory(self, records: Sequence[Any], **metadata: Any) -> EvidenceRef:
        """Store an action/observation trajectory as JSON Lines."""
        return self.add_jsonl(records, kind="trajectory", metadata=metadata)

    def add_video(self, source: str | Path, **metadata: Any) -> EvidenceRef:
        """Index and copy a video file into the store."""
        return self.add_file(source, kind="video", metadata=metadata)

    def add_failure_envelope(self, envelope: Any, **metadata: Any) -> EvidenceRef:
        """Store a FailureEnvelope without importing LeapFlow engine internals."""
        return self.add_json(envelope, kind="failure_envelope", metadata=metadata)

    def add_side_effect_evidence(self, evidence: Any, **metadata: Any) -> EvidenceRef:
        """Store side-effect verdict/evidence as canonical JSON."""
        return self.add_json(evidence, kind="side_effect", metadata=metadata)

    # ── Reading ───────────────────────────────────────────────

    def list_refs(self, *, kind: str = "") -> tuple[EvidenceRef, ...]:
        """Read indexed evidence references, optionally filtered by kind."""
        if not self.index_path.exists():
            return ()
        refs: list[EvidenceRef] = []
        try:
            lines = self.index_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                ref = EvidenceRef.from_dict(data)
                if not kind or ref.kind == kind:
                    refs.append(ref)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return tuple(refs)

    def verify(self, ref: EvidenceRef) -> bool:
        """Verify that referenced bytes exist and match the stored hash."""
        path = (self.root / ref.path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        if not path.is_file():
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ref.content_hash.removeprefix("sha256:")
        return digest == expected and path.stat().st_size == ref.size_bytes


__all__ = ["EvidenceStore", "evidence_root"]
