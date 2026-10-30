# Copyright (c) Alibaba, Inc. and its affiliates.
"""File checkpoint interceptor — snapshot files before mutation tools run.

Sits in the tool pipeline AFTER approval (priority 40) and BEFORE audit (100).
On a mutating file tool, ``before()`` snapshots the target files; ``after()``
either persists the TurnCheckpoint on success or auto-rolls-back on failure.

Design:
    - File-path extraction is schema-driven: parameter names matching known path
      patterns (``path``, ``file_path``, ``target_path``, ``dest``, ``destination``,
      ``filename``) are extracted from ``ToolMetadata.parameters_schema``.
      No tool names are hardcoded.
    - Small files (≤ configurable threshold) are stored inline as bytes.
    - Large files are copied to CacheLayout temp; ``temp_ref`` records the path.
    - ``existed=False`` enables rollback of file *creation* (rollback = delete).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Parameter name patterns that indicate a file path argument.
_PATH_PARAM_NAMES = frozenset({
    "path", "file_path", "target_path", "dest", "destination",
    "filename", "filepath", "file", "output_path", "source_path",
    "target", "dest_path",
})


# ════════════════════════════════════════════════════════════════════════
# Domain types
# ════════════════════════════════════════════════════════════════════════


class FileSnapshot(NamedTuple):
    """Immutable snapshot of one file taken before a mutation tool runs."""

    path: str
    content_hash: str        # SHA-256 hex of original content (empty if not existed)
    existed: bool            # Whether the file existed before the tool ran
    inline_content: Optional[bytes]  # For small files; None when using temp_ref
    temp_ref: Optional[str]  # Path to temp copy for large files; None when inline
    size: int                # Original file size (0 if not existed)
    timestamp: float         # Time the snapshot was taken


class TurnCheckpoint(NamedTuple):
    """Immutable checkpoint grouping all file snapshots for one turn."""

    turn_id: str
    session_id: str
    snapshots: tuple[FileSnapshot, ...]
    created_at: float


@dataclass(frozen=True)
class RollbackResult:
    """Result of rolling back a turn's file snapshots."""

    restored: tuple[str, ...]        # Paths successfully restored
    skipped: tuple[str, ...]         # Paths unchanged (hash match)
    failed: tuple[tuple[str, str], ...]  # (path, reason) pairs


# ════════════════════════════════════════════════════════════════════════
# Store protocol
# ════════════════════════════════════════════════════════════════════════


@runtime_checkable
class FileCheckpointStore(Protocol):
    """Durable store for file checkpoint snapshots."""

    def save_turn(self, checkpoint: TurnCheckpoint) -> None:
        """Persist all snapshots for a completed turn."""
        ...

    def get_turn(self, turn_id: str) -> Optional[TurnCheckpoint]:
        """Retrieve the checkpoint for a given turn."""
        ...

    def list_turns(self, session_id: str, *, limit: int = 20) -> list[TurnCheckpoint]:
        """List recent checkpoints for a session, newest first."""
        ...

    def rollback_turn(self, turn_id: str) -> RollbackResult:
        """Restore files from a turn's snapshots.

        For each snapshot:
        - If current on-disk hash matches stored hash -> skip (no-op).
        - If existed=False and the file now exists -> delete it.
        - Otherwise -> restore from inline_content or temp_ref.
        """
        ...

    def cleanup(self, *, max_age_hours: float = 24.0) -> int:
        """Delete checkpoints older than the cutoff. Returns count deleted."""
        ...


# ════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def _extract_file_paths(arguments: Dict[str, Any], parameters_schema: Dict[str, Any]) -> list[str]:
    """Extract file paths from tool arguments using schema-driven heuristics.

    Inspects parameter names in the schema's ``properties`` dict for matches
    against known file-path naming patterns. No tool names are hardcoded.
    """
    properties = parameters_schema.get("properties", {})
    if not properties:
        # Fallback: check argument keys directly against known patterns
        properties = {k: {} for k in arguments}

    paths: list[str] = []
    for param_name in properties:
        normalized = param_name.lower().replace("-", "_")
        if normalized in _PATH_PARAM_NAMES:
            value = arguments.get(param_name)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
    return paths


# ════════════════════════════════════════════════════════════════════════
# Interceptor
# ════════════════════════════════════════════════════════════════════════


class FileCheckpointInterceptor:
    """Waterfall interceptor that snapshots files before mutation tools run.

    Priority 40: after approval (which has no interceptor — it's inline) and
    timeout (10), but before audit (100). This means the checkpoint captures
    the state AFTER approval has been granted and BEFORE the tool runs.
    """

    def __init__(
        self,
        store: FileCheckpointStore,
        *,
        max_inline_bytes: int = 262144,
        temp_dir: Optional[Path] = None,
        get_turn_id: Optional[Any] = None,
        get_session_id: Optional[Any] = None,
        parameters_schema_lookup: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._max_inline_bytes = max_inline_bytes
        self._temp_dir = temp_dir
        self._get_turn_id = get_turn_id
        self._get_session_id = get_session_id
        self._parameters_schema_lookup = parameters_schema_lookup
        # Accumulate snapshots across multiple tool calls within one turn
        self._pending_snapshots: Dict[str, List[FileSnapshot]] = {}

    @property
    def name(self) -> str:
        return "file_checkpoint"

    @property
    def priority(self) -> int:
        return 40

    async def before(self, context: Any) -> Optional[Dict[str, Any]]:
        """Snapshot target files before a mutating file tool runs."""
        metadata = context.metadata or {}
        if not metadata.get("mutates_state", False):
            return None

        # Resolve parameters schema for file-path extraction
        schema: Dict[str, Any] = {}
        if self._parameters_schema_lookup is not None:
            try:
                schema = self._parameters_schema_lookup(context.tool_name) or {}
            except Exception:
                pass

        paths = _extract_file_paths(context.arguments, schema)
        if not paths:
            return None

        turn_id = ""
        if self._get_turn_id is not None:
            try:
                turn_id = str(self._get_turn_id())
            except Exception:
                pass

        snapshots: List[FileSnapshot] = []
        for file_path_str in paths:
            try:
                snapshot = self._snapshot_file(file_path_str)
                if snapshot is not None:
                    snapshots.append(snapshot)
            except Exception as exc:
                logger.warning(
                    "file_checkpoint: failed to snapshot %s: %s",
                    file_path_str, exc,
                )

        if snapshots:
            key = turn_id or "__default__"
            self._pending_snapshots.setdefault(key, []).extend(snapshots)
            # Store snapshots in annotations so after() can access them
            context.annotations["_file_checkpoint_snapshots"] = snapshots
            context.annotations["_file_checkpoint_turn_key"] = key

        return None  # Never short-circuit

    async def after(self, context: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        """On success, persist the checkpoint. On failure, auto-rollback."""
        snapshots = context.annotations.get("_file_checkpoint_snapshots")
        if not snapshots:
            return result

        is_error = isinstance(result, dict) and (
            not result.get("ok", True) or "error" in result
        )

        if is_error:
            # Auto-rollback: restore files to pre-tool state
            for snap in snapshots:
                try:
                    self._restore_file(snap)
                except Exception as exc:
                    logger.warning(
                        "file_checkpoint: auto-rollback failed for %s: %s",
                        snap.path, exc,
                    )
            # Discard the failed turn's snapshots so they don't linger in
            # memory on a long-running daemon (they are never persisted).
            turn_key = context.annotations.get("_file_checkpoint_turn_key", "")
            if turn_key:
                self._pending_snapshots.pop(turn_key, None)
        else:
            # Success: persist the turn checkpoint
            turn_key = context.annotations.get("_file_checkpoint_turn_key", "")
            self._finalize_turn(turn_key)

        return result

    def _snapshot_file(self, file_path_str: str) -> Optional[FileSnapshot]:
        """Create a FileSnapshot for one file."""
        fp = Path(file_path_str)
        now = time.time()

        if not fp.exists():
            return FileSnapshot(
                path=file_path_str,
                content_hash="",
                existed=False,
                inline_content=None,
                temp_ref=None,
                size=0,
                timestamp=now,
            )

        if not fp.is_file():
            return None

        file_size = fp.stat().st_size
        content_hash = _sha256_file(fp)

        if file_size <= self._max_inline_bytes:
            inline_content = fp.read_bytes()
            return FileSnapshot(
                path=file_path_str,
                content_hash=content_hash,
                existed=True,
                inline_content=inline_content,
                temp_ref=None,
                size=file_size,
                timestamp=now,
            )
        else:
            # Large file: copy to temp
            temp_ref = self._copy_to_temp(fp, content_hash)
            return FileSnapshot(
                path=file_path_str,
                content_hash=content_hash,
                existed=True,
                inline_content=None,
                temp_ref=temp_ref,
                size=file_size,
                timestamp=now,
            )

    def _copy_to_temp(self, source: Path, content_hash: str) -> str:
        """Copy a large file to the temp directory, returning its path."""
        if self._temp_dir is None:
            raise RuntimeError("No temp_dir configured for large file checkpoints")
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        dest = self._temp_dir / f"ckpt_{content_hash}_{int(time.time() * 1000)}"
        shutil.copy2(source, dest)
        return str(dest)

    @staticmethod
    def _restore_file(snapshot: FileSnapshot) -> None:
        """Restore a single file from its snapshot."""
        fp = Path(snapshot.path)

        if not snapshot.existed:
            # File was created by the tool: undo by deleting
            if fp.exists():
                fp.unlink()
            return

        if snapshot.inline_content is not None:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(snapshot.inline_content)
        elif snapshot.temp_ref is not None:
            temp = Path(snapshot.temp_ref)
            if temp.exists():
                fp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp, fp)
            else:
                raise FileNotFoundError(
                    f"Temp checkpoint file missing: {snapshot.temp_ref}"
                )

    def _finalize_turn(self, turn_key: str) -> None:
        """Persist accumulated snapshots for a turn and clear the buffer."""
        snapshots = self._pending_snapshots.pop(turn_key, [])
        if not snapshots:
            return

        turn_id = turn_key if turn_key != "__default__" else ""
        session_id = ""
        if self._get_session_id is not None:
            try:
                session_id = str(self._get_session_id())
            except Exception:
                pass

        checkpoint = TurnCheckpoint(
            turn_id=turn_id,
            session_id=session_id,
            snapshots=tuple(snapshots),
            created_at=time.time(),
        )
        try:
            self._store.save_turn(checkpoint)
        except Exception as exc:
            logger.warning("file_checkpoint: failed to persist checkpoint: %s", exc)


def restore_from_snapshot(snapshot: FileSnapshot) -> tuple[bool, str]:
    """Restore a single file, returning (success, reason).

    Used by the store's rollback_turn and the /checkpoint command.
    """
    fp = Path(snapshot.path)

    if not snapshot.existed:
        if fp.exists():
            try:
                fp.unlink()
                return True, "deleted (was created after checkpoint)"
            except OSError as exc:
                return False, f"delete failed: {exc}"
        return True, "already absent"

    # Check if file is unchanged
    if fp.exists() and fp.is_file():
        try:
            current_hash = _sha256_file(fp)
            if current_hash == snapshot.content_hash:
                return True, "unchanged (hash match)"
        except OSError:
            pass

    # Restore content
    try:
        if snapshot.inline_content is not None:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(snapshot.inline_content)
            return True, "restored from inline content"
        elif snapshot.temp_ref is not None:
            temp = Path(snapshot.temp_ref)
            if not temp.exists():
                return False, f"temp file missing: {snapshot.temp_ref}"
            fp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp, fp)
            return True, "restored from temp copy"
        else:
            return False, "no content available (neither inline nor temp_ref)"
    except OSError as exc:
        return False, f"restore failed: {exc}"
