# Copyright (c) Alibaba, Inc. and its affiliates.
"""Profile-scoped version store for dynamically installed plugins."""
from __future__ import annotations

import hashlib
import json
import os
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
        self._write_json(plugin_dir / "active.json", entry)
        index = [item for item in self._read_index(plugin_id) if item.get("version") != version_id]
        index.append(entry)
        self._write_json(plugin_dir / "versions.json", index)
        return entry

    def active(self, plugin_id: str) -> dict[str, Any] | None:
        path = self._plugin_dir(plugin_id) / "active.json"
        data = self._read_json(path)
        return data if isinstance(data, dict) else None

    def versions(self, plugin_id: str) -> list[dict[str, Any]]:
        return self._read_index(plugin_id)

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
