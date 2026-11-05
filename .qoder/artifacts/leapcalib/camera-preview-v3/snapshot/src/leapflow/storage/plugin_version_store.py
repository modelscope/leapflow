# Copyright (c) Alibaba, Inc. and its affiliates.
"""Profile-scoped version store for dynamically installed plugins."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any


class PluginVersionStore:
    """File-backed version snapshots and active pointers for profile plugins."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def record_source(
        self,
        plugin_id: str,
        source_path: Path,
        *,
        version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Copy source into the version store and mark it active."""
        source = Path(source_path)
        code = source.read_bytes()
        version_id = str(version or f"sha-{hashlib.sha256(code).hexdigest()[:12]}")
        plugin_dir = self._plugin_dir(plugin_id)
        versions_dir = plugin_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        target = versions_dir / f"{version_id}.py"
        self._write_bytes(target, code)
        entry = {
            "plugin_id": plugin_id,
            "version": version_id,
            "source_path": str(source),
            "snapshot_path": str(target),
            "sha256": hashlib.sha256(code).hexdigest(),
            "created_at": time.time(),
            "metadata": dict(metadata or {}),
        }
        index = [item for item in self._read_index(plugin_id) if item.get("version") != version_id]
        index.append(entry)
        # Commit the active pointer last. A failed index write may leave an immutable
        # snapshot behind, but it cannot make a partially recorded version active.
        self._write_json(plugin_dir / "versions.json", index)
        self._write_json(plugin_dir / "active.json", entry)
        return entry

    def active(self, plugin_id: str) -> dict[str, Any] | None:
        path = self._plugin_dir(plugin_id) / "active.json"
        data = self._read_json(path)
        return data if isinstance(data, dict) else None

    def versions(self, plugin_id: str) -> list[dict[str, Any]]:
        return self._read_index(plugin_id)

    def snapshot_state(self, plugin_id: str) -> dict[str, Any]:
        """Capture active pointer and version index for transactional rollback."""
        return {
            "active": self.active(plugin_id),
            "versions": self.versions(plugin_id),
        }

    def restore_state(self, plugin_id: str, snapshot: dict[str, Any]) -> None:
        """Restore metadata captured before a failed file/runtime mutation."""
        plugin_dir = self._plugin_dir(plugin_id)
        active_path = plugin_dir / "active.json"
        active = snapshot.get("active")
        if isinstance(active, dict):
            self._write_json(active_path, active)
        else:
            active_path.unlink(missing_ok=True)
        self._write_json(plugin_dir / "versions.json", list(snapshot.get("versions") or ()))

    def restore_source(self, target_path: Path, data: bytes | None) -> None:
        """Atomically restore a source snapshot, or remove a previously absent file."""
        target = Path(target_path)
        if data is None:
            target.unlink(missing_ok=True)
            return
        self._write_bytes(target, data)

    def source_for(self, plugin_id: str, version: str) -> Path | None:
        for item in self._read_index(plugin_id):
            if str(item.get("version")) == str(version):
                path = Path(str(item.get("snapshot_path") or ""))
                return path if path.exists() else None
        return None

    def rollback(self, plugin_id: str, version: str, target_path: Path) -> dict[str, Any]:
        source = self.source_for(plugin_id, version)
        if source is None:
            raise KeyError(f"Plugin version not found: {plugin_id}@{version}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write_bytes(target, source.read_bytes())
        entry = self.record_source(plugin_id, target, version=version, metadata={"rollback": True})
        return entry

    # ── bundle (directory-level) snapshots ────────────────────────────

    def record_bundle(
        self,
        plugin_id: str,
        wrapper_path: Path,
        bundle_dir: Path,
        *,
        version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a tar.gz archive of *wrapper_path* and *bundle_dir* and record it."""
        wrapper = Path(wrapper_path)
        bundle = Path(bundle_dir)
        if not wrapper.is_file():
            raise FileNotFoundError(f"Wrapper not found: {wrapper}")
        if not bundle.is_dir():
            raise NotADirectoryError(f"Bundle directory not found: {bundle}")

        # Derive version from combined content hash when unspecified.
        sha_hasher = hashlib.sha256()
        sha_hasher.update(wrapper.read_bytes())
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                sha_hasher.update(p.read_bytes())
        bundle_sha256 = sha_hasher.hexdigest()
        version_id = str(version or f"sha-{bundle_sha256[:12]}")

        plugin_dir = self._plugin_dir(plugin_id)
        versions_dir = plugin_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        archive_path = versions_dir / f"{version_id}_bundle.tar.gz"

        # Build archive in memory then write atomically.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(wrapper), arcname=f"wrapper/{wrapper.name}")
            tar.add(str(bundle), arcname="bundle")
        archive_bytes = buf.getvalue()
        self._write_bytes(archive_path, archive_bytes)

        entry: dict[str, Any] = {
            "plugin_id": plugin_id,
            "version": version_id,
            "source_path": str(wrapper),
            "snapshot_path": str(archive_path),
            "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "bundle_sha256": bundle_sha256,
            "is_bundle": True,
            "created_at": time.time(),
            "metadata": dict(metadata or {}),
        }
        index = [
            item for item in self._read_index(plugin_id)
            if item.get("version") != version_id
        ]
        index.append(entry)
        self._write_json(plugin_dir / "versions.json", index)
        self._write_json(plugin_dir / "active.json", entry)
        return entry

    def rollback_bundle(
        self,
        plugin_id: str,
        version: str,
        wrapper_target: Path,
        bundle_target_dir: Path,
    ) -> dict[str, Any]:
        """Restore a bundle snapshot previously recorded with *record_bundle*."""
        entry: dict[str, Any] | None = None
        for item in self._read_index(plugin_id):
            if str(item.get("version")) == str(version) and item.get("is_bundle"):
                entry = item
                break
        if entry is None:
            raise KeyError(f"Bundle version not found: {plugin_id}@{version}")
        archive_path = Path(str(entry["snapshot_path"]))
        if not archive_path.exists():
            raise FileNotFoundError(f"Bundle archive missing: {archive_path}")

        expected_sha = str(entry.get("bundle_sha256", ""))
        wrapper_dest = Path(wrapper_target)
        bundle_dest = Path(bundle_target_dir)

        # Extract into a temporary staging directory, then promote.
        staging = bundle_dest.parent / f".rollback_staging_{plugin_id}_{os.getpid()}"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(staging, filter="data")

            # Locate extracted artefacts.
            extracted_wrapper_dir = staging / "wrapper"
            extracted_bundle_dir = staging / "bundle"
            if not extracted_bundle_dir.is_dir():
                raise RuntimeError("Archive does not contain a bundle directory")
            wrapper_files = list(extracted_wrapper_dir.iterdir()) if extracted_wrapper_dir.is_dir() else []
            if not wrapper_files:
                raise RuntimeError("Archive does not contain a wrapper file")
            extracted_wrapper = wrapper_files[0]

            # Verify SHA-256 integrity.
            sha_hasher = hashlib.sha256()
            sha_hasher.update(extracted_wrapper.read_bytes())
            for p in sorted(extracted_bundle_dir.rglob("*")):
                if p.is_file():
                    sha_hasher.update(p.read_bytes())
            actual_sha = sha_hasher.hexdigest()
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(
                    f"Bundle integrity check failed: expected {expected_sha[:16]}…, "
                    f"got {actual_sha[:16]}…"
                )

            # Promote: replace wrapper and bundle directory atomically-ish.
            wrapper_dest.parent.mkdir(parents=True, exist_ok=True)
            self._write_bytes(wrapper_dest, extracted_wrapper.read_bytes())
            if bundle_dest.exists():
                shutil.rmtree(bundle_dest)
            shutil.copytree(extracted_bundle_dir, bundle_dest)

            # Update version index to mark this version active.
            result_entry = dict(entry)
            result_entry["metadata"] = {**result_entry.get("metadata", {}), "rollback": True}
            idx = [
                item for item in self._read_index(plugin_id)
                if item.get("version") != version
            ]
            idx.append(result_entry)
            plugin_dir = self._plugin_dir(plugin_id)
            self._write_json(plugin_dir / "versions.json", idx)
            self._write_json(plugin_dir / "active.json", result_entry)
            return {"ok": True, "plugin_id": plugin_id, "version": version, "bundle": True}
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _plugin_dir(self, plugin_id: str) -> Path:
        return self._root / str(plugin_id)

    def _read_index(self, plugin_id: str) -> list[dict[str, Any]]:
        data = self._read_json(self._plugin_dir(plugin_id) / "versions.json")
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        PluginVersionStore._write_bytes(path, encoded)
