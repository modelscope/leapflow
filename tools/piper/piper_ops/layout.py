"""Inventory, private full backups and manifest-bound relocation.

Deletion requires an injected safe filesystem service. This module never falls
back to shell deletion, recursive unlink, or following external symlinks.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

from .evidence import Refusal


@runtime_checkable
class SafeDeletion(Protocol):
    """An operator-approved safe filesystem service, not a shell command."""

    def delete_file(self, path: Path) -> None: ...


def stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha256(stream)


def _private(path: Path) -> bool:
    return any("token" in p.lower() or p.endswith((".key", ".pem")) or p == ".env"
               for p in path.parts)


def inventory(root: Path, *, private: bool = False) -> dict[str, Any]:
    """Inspect without following symlinks or disclosing credential content."""
    root = root.resolve(strict=True)
    entries = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root)
            info = path.lstat()
            kind = ("symlink" if stat.S_ISLNK(info.st_mode) else
                    "directory" if stat.S_ISDIR(info.st_mode) else
                    "file" if stat.S_ISREG(info.st_mode) else "special")
            row: dict[str, Any] = {"path": str(relative), "kind": kind,
                                   "size": info.st_size, "mode": stat.S_IMODE(info.st_mode),
                                   "mtime_ns": info.st_mtime_ns}
            if kind == "symlink":
                row["target"] = os.readlink(path)
            elif kind == "file" and (private or not _private(relative)):
                row["sha256"] = sha256(path)
                if path.suffix == ".py" and not _private(relative):
                    try:
                        tree = ast.parse(path.read_text())
                        row["path_literals"] = sorted({n.value for n in ast.walk(tree)
                            if isinstance(n, ast.Constant) and isinstance(n.value, str)
                            and ("/" in n.value or n.value.endswith((".json", ".yaml", ".py", ".log")))
                            and len(n.value) < 240})
                        row["imports"] = sorted({n.module for n in ast.walk(tree)
                            if isinstance(n, ast.ImportFrom) and n.module} | {
                            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
                    except (SyntaxError, UnicodeError):
                        row["parse_error"] = True
            entries.append(row)
    return {"root": str(root), "entries": entries,
            "free_bytes": shutil.disk_usage(root).free}


def _reference_bytes(path: Path, max_bytes: int) -> bytes:
    """Open every component without following links, including concurrent replacements."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("no-follow inspection unavailable")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                  dir_fd=descriptor)
        with os.fdopen(file_descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                raise OSError("reference input is not regular or exceeds byte budget")
            body = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
            if (len(body) > max_bytes or len(body) != before.st_size or
                    (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise OSError("reference input changed during inspection")
            return body
    finally:
        os.close(descriptor)


def reference_report(root: Path, locations: list[Path], *, max_bytes: int = 2_000_000) -> dict[str, Any]:
    """Inspect explicitly named external consumers; return paths, never matching text.

    Static matches are dependency candidates, not proof of obsolescence. Skipped,
    unreadable and symlinked inputs remain explicit gaps. No external link is followed.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise Refusal("reference inspection byte budget must be a positive integer")
    root = root.resolve(strict=True)
    sources = sorted(e["path"] for e in inventory(root)["entries"] if e["kind"] == "file"
                     and not _private(Path(e["path"])))
    matches, gaps, inspected = [], [], []
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
                "node_modules", "secrets", "vault", "db", ".ssh", ".gnupg"}
    suffixes = {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".conf", ".service", ".timer"}

    def excluded_path(path: Path) -> bool:
        return (_private(path) or any(part in excluded or part.startswith(".env.")
                                     for part in path.parts))
    patterns = {source: re.compile(r"(?<![\w./-])" + re.escape(source) + r"(?![\w./-])")
                for source in sources}

    def inspect(path: Path) -> None:
        if excluded_path(path):
            gaps.append({"path": str(path), "reason": "sensitive_or_excluded"})
            return
        try:
            if path.is_symlink():
                gaps.append({"path": str(path), "reason": "symlink_not_followed"})
                return
            size = path.stat().st_size
            if not path.is_file() or size > max_bytes:
                gaps.append({"path": str(path), "reason": "not_regular_or_over_budget"})
                return
            text_bytes = _reference_bytes(path, max_bytes)
            text = text_bytes.decode("utf-8")
        except (OSError, UnicodeError) as error:
            gaps.append({"path": str(path), "reason": type(error).__name__})
            return
        inspected.append({"path": str(path), "sha256": hashlib.sha256(text_bytes).hexdigest()})
        if str(root) in text:
            matches.append({"consumer": str(path), "source": ".", "kind": "absolute_root"})
        for source, pattern in patterns.items():
            if str(root / source) in text or pattern.search(text):
                matches.append({"consumer": str(path), "source": source, "kind": "path_candidate"})

    for location in locations:
        path = location.expanduser().absolute()
        if excluded_path(path):
            gaps.append({"path": str(path), "reason": "sensitive_or_excluded"})
        elif any(parent.is_symlink() for parent in (path, *path.parents)):
            gaps.append({"path": str(path), "reason": "symlink_not_followed"})
        elif not path.exists():
            gaps.append({"path": str(path), "reason": "missing"})
        elif path.is_file():
            inspect(path)
        elif path.is_dir():
            def walk_error(error: OSError) -> None:
                gaps.append({"path": str(error.filename), "reason": type(error).__name__})
            for directory, dirs, files in os.walk(path, followlinks=False, onerror=walk_error):
                for name in tuple(dirs):
                    child = Path(directory) / name
                    if excluded_path(child) or child.is_symlink():
                        dirs.remove(name)
                        gaps.append({"path": str(child), "reason": "excluded_or_symlink"})
                for name in sorted(files):
                    child = Path(directory) / name
                    if child.suffix in suffixes or name.startswith("."):
                        inspect(child)
                    else:
                        gaps.append({"path": str(child), "reason": "suffix_not_selected"})
        else:
            gaps.append({"path": str(path), "reason": "not_regular"})
    return {"root": str(root), "locations": [str(p) for p in locations],
            "max_bytes": max_bytes, "inspected": inspected, "matches": matches, "gaps": gaps,
            "scope_complete": not gaps, "cleanup_authorized": False,
            "limitations": "Static path matches only; dynamic imports, environment expansion, "
                           "uncaptured services and future process use require separate inspection."}


def process_references(root: Path, *, proc: Path = Path("/proc")) -> dict[str, Any]:
    """Read same-user Linux process metadata without argv/environment disclosure or signals."""
    root = root.resolve(strict=True)
    if not proc.is_dir() or not hasattr(os, "getuid"):
        return {"supported": False, "matches": [], "gaps": [{"reason": "Linux /proc unavailable"}],
                "cleanup_authorized": False}
    matches, gaps = [], []
    try:
        processes = sorted(proc.iterdir(), key=lambda path: path.name)
    except OSError as error:
        return {"supported": False, "matches": [], "gaps": [{"reason": type(error).__name__}],
                "cleanup_authorized": False}
    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            if process.is_symlink():
                raise OSError("process entry is a symlink")
            if process.stat().st_uid != os.getuid():
                continue
            identity = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
            if not identity.isdecimal():
                raise ValueError("invalid process creation identity")
            references = set()
            try:
                cwd = os.readlink(process / "cwd")
                if cwd == str(root) or cwd.startswith(str(root) + "/"):
                    references.add("cwd")
            except OSError as error:
                gaps.append({"pid": int(process.name), "field": "cwd", "reason": type(error).__name__})
            try:
                with (process / "cmdline").open("rb") as stream:
                    command = stream.read(64_001)
                if len(command) > 64_000:
                    gaps.append({"pid": int(process.name), "reason": "argv_over_budget"})
                if os.fsencode(str(root)) in command:
                    references.add("argv_root")
            except OSError as error:
                gaps.append({"pid": int(process.name), "field": "argv", "reason": type(error).__name__})
            try:
                for fd in (process / "fd").iterdir():
                    try:
                        target = os.readlink(fd)
                        if target == str(root) or target.startswith(str(root) + "/"):
                            references.add("open_file")
                    except OSError as error:
                        gaps.append({"pid": int(process.name), "field": "fd", "reason": type(error).__name__})
            except OSError as error:
                gaps.append({"pid": int(process.name), "field": "fd", "reason": type(error).__name__})
            after = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
            if identity != after:
                gaps.append({"pid": int(process.name), "reason": "process_identity_changed"})
                continue
            if references:
                matches.append({"pid": int(process.name), "process_start": identity,
                                "references": sorted(references)})
        except (OSError, IndexError, ValueError) as error:
            gaps.append({"pid": int(process.name), "reason": type(error).__name__})
    return {"supported": True, "scope": "same-user process snapshot", "matches": matches,
            "gaps": gaps, "cleanup_authorized": False,
            "limitations": "Other users, relative argv, closed files and process races are not "
                           "proof of absent dependencies; do not use this snapshot as deletion authority."}


def backup(root: Path, destination: Path) -> dict[str, Any]:
    """Create a private full tar snapshot and verify each archived member."""
    root, destination = root.resolve(strict=True), destination.resolve()
    if destination == root or root in destination.parents:
        raise Refusal("backup destination must be outside the Piper directory")
    before = inventory(root, private=True)
    if any(e["kind"] == "special" for e in before["entries"]):
        raise Refusal("live special filesystem entries require an approved consistent backup strategy")
    total = sum(e["size"] for e in before["entries"] if e["kind"] == "file")
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < total * 2 + 10_000_000:
        raise Refusal("insufficient space for a complete verified backup")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = destination / f"piper-full-{stamp}.tar.gz"
    descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream, tarfile.open(fileobj=stream, mode="w:gz",
                                                              dereference=False) as tar:
        tar.add(root, arcname="piper", recursive=True)
    members = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            row = {"path": member.name, "size": member.size, "mode": member.mode,
                   "link": member.linkname, "type": member.type.decode("ascii"),
                   "mtime": member.mtime, "uid": member.uid, "gid": member.gid}
            if member.isfile() or member.islnk():
                stream = tar.extractfile(member)
                if stream is None:
                    raise Refusal("backup member unreadable")
                with stream:
                    row["sha256"] = stream_sha256(stream)
            members.append(row)
    archived = {m["path"].removeprefix("piper/"): m for m in members if m["path"] != "piper"}
    missing = [e["path"] for e in before["entries"] if e["path"] not in archived]
    if missing:
        raise Refusal(f"backup is incomplete: {missing}")
    after = inventory(root, private=True)
    if {e["path"] for e in before["entries"]} != {e["path"] for e in after["entries"]}:
        raise Refusal("tree inventory changed during backup; archive retained but not approved")
    for entry in before["entries"]:
        saved = archived[entry["path"]]
        expected_type = {"file": "0", "directory": "5", "symlink": "2"}[entry["kind"]]
        if saved["type"] not in (expected_type, "1" if entry["kind"] == "file" else expected_type):
            raise Refusal("backup member type differs")
        if saved["mode"] != entry["mode"] or (entry["kind"] == "symlink" and
                                               saved["link"] != entry["target"]):
            raise Refusal("backup metadata or symlink target differs")
    after_by_path = {e["path"]: e for e in after["entries"]}
    changed = [e["path"] for e in before["entries"] if "sha256" in e and (
               archived[e["path"]].get("sha256") != e["sha256"] or
               after_by_path[e["path"]].get("sha256") != e["sha256"])]
    # A running node may append logs. Those files are preserved, never cleanup candidates.
    mutable = [p for p in changed if Path(p).suffix in (".log", ".ndjson")]
    if set(changed) - set(mutable):
        raise Refusal(f"non-log files changed during backup: {changed}")
    report = {"ok": True, "archive": str(archive), "archive_sha256": sha256(archive),
              "root": str(root), "mutable_logs": mutable, "members": members}
    manifest = archive.with_suffix(".manifest.json")
    fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(report, stream, indent=2)
    return {"ok": True, "archive": str(archive), "manifest": str(manifest),
            "sha256": report["archive_sha256"], "members": len(members),
            "mutable_logs": mutable}


def checked_path(root: Path, relative: str) -> Path:
    """Reject escapes and symlink traversal before any relocation."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise Refusal("manifest path must be a nonempty relative path")
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or not item.parts or str(item) != relative:
        raise Refusal("manifest path must be a canonical nonempty relative path")
    path = root / item
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise Refusal(f"manifest path traverses symlink: {relative}")
    return path


def active_references(root: Path, sources: set[str]) -> list[dict[str, str]]:
    """Reject retained operational consumers; history and data are not live entry points."""
    blocked = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ("legacy", "archive", "data", "__pycache__", ".git")
                   and not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if relative in sources or path.is_symlink() or path.suffix not in (".py", ".sh", ".yaml", ".yml"):
                continue
            if _private(path.relative_to(root)):
                continue
            text = path.read_text(errors="strict")
            modules = set()
            if path.suffix == ".py":
                try:
                    tree = ast.parse(text)
                    parent = path.parent.relative_to(root).parts
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                modules.add(alias.name)
                                modules.add(".".join((*parent, alias.name)))
                        elif isinstance(node, ast.ImportFrom):
                            base = parent[:len(parent) - node.level + 1] if node.level else ()
                            prefix = ".".join((*base, *((node.module or "").split(".") if node.module else ())))
                            modules.add(prefix)
                            modules.update(".".join(p for p in (prefix, a.name) if p) for a in node.names)
                except SyntaxError:
                    raise Refusal(f"retained script cannot be inspected: {relative}") from None
            for source in sources:
                module = str(Path(source).with_suffix("")).replace("/", ".")
                if (any(m == module or m.startswith(module + ".") for m in modules) or source in text
                        or (len(Path(source).parts) == 1 and Path(source).name in text)):
                    blocked.append({"consumer": relative, "source": source})
    return blocked


def apply(root: Path, manifest: dict[str, Any], *, delete: SafeDeletion | None = None,
          execute: bool = False, evidence: Any = None) -> dict[str, Any]:
    """Prevalidate the complete manifest; never delete without the safe service."""
    root = root.resolve(strict=True)
    if not isinstance(manifest, dict) or manifest.get("root") != str(root):
        raise Refusal("cleanup manifest root differs")
    location = manifest.get("backup_manifest")
    if not isinstance(location, str) or not Path(location).is_absolute():
        raise Refusal("absolute verified backup manifest path required")
    backup_manifest = Path(location)
    if backup_manifest.is_symlink() or not backup_manifest.is_file():
        raise Refusal("backup manifest must be a regular file")
    record = json.loads(backup_manifest.read_text())
    if (not isinstance(record, dict) or record.get("ok") is not True
            or record.get("root") != str(root) or not isinstance(record.get("archive"), str)
            or not isinstance(record.get("members"), list)
            or any(not isinstance(m, dict) or not isinstance(m.get("path"), str) for m in record["members"])
            or not isinstance(record.get("archive_sha256"), str)):
        raise Refusal("malformed verified backup record")
    archive = Path(record["archive"])
    if (not archive.is_absolute() or archive.is_symlink() or not archive.is_file()
            or root == archive.parent or root in archive.parents
            or sha256(archive) != record["archive_sha256"]):
        raise Refusal("backup identity/hash differs; no cleanup")
    rows = manifest.get("actions")
    if not isinstance(rows, list) or any(not isinstance(r, dict) or
        not {"action", "source", "sha256"} <= r.keys() or
        not isinstance(r["action"], str) or not isinstance(r["source"], str) or
        not isinstance(r["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) is None or
        (r["action"] == "move" and not isinstance(r.get("target"), str)) for r in rows):
        raise Refusal("malformed cleanup actions")
    if execute and evidence is None:
        raise Refusal("cleanup execution requires durable per-operation evidence")
    if any(r["action"] == "remove" for r in rows) and not isinstance(delete, SafeDeletion):
        raise Refusal("safe remote deletion service unavailable; no cleanup applied",
                      code="piper_safe_deletion_unavailable")
    archived = {m["path"].removeprefix("piper/"): m for m in record["members"]}
    sources, destinations = set(), set()
    for row in rows:
        path = checked_path(root, row["source"])
        if row["source"] in sources or row["action"] not in ("move", "remove"):
            raise Refusal("duplicate source or unsupported cleanup action")
        sources.add(row["source"])
        if not path.is_file() or _private(path.relative_to(root)) or sha256(path) != row["sha256"]:
            raise Refusal(f"cleanup source changed: {row['source']}")
        if archived.get(row["source"], {}).get("sha256") != row["sha256"]:
            raise Refusal(f"cleanup source not verified in backup: {row['source']}")
        if row["source"] in record.get("mutable_logs", []):
            raise Refusal("live mutable logs are preserved in place")
        if row["action"] == "move":
            if "target" not in row:
                raise Refusal("move requires a target")
            target = checked_path(root, row["target"])
            if target.exists() or row["target"] in destinations:
                raise Refusal("cleanup destination exists or is duplicated")
            destinations.add(row["target"])
    blocked = active_references(root, sources)
    if blocked:
        raise Refusal(f"retained operational dependencies block cleanup: {blocked}")
    for target in destinations:
        if any(target == src or Path(src) in Path(target).parents or
               Path(target) in Path(src).parents for src in sources):
            raise Refusal("cleanup source and target trees overlap")
        if any(target != other and (Path(target) in Path(other).parents or
                                   Path(other) in Path(target).parents) for other in destinations):
            raise Refusal("cleanup destination trees overlap")
    if not execute:
        return {"ok": True, "executed": False, "actions": rows}
    completed = []
    for row in rows:
        source = checked_path(root, row["source"])
        if sha256(source) != row["sha256"]:
            raise Refusal("source changed during cleanup; inspect partial manifest")
        evidence.record("cleanup_attempt", action=row)
        if row["action"] == "move":
            target = checked_path(root, row["target"])
            if target.exists():
                raise Refusal("destination appeared during cleanup; inspect checkpoint")
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        else:
            assert delete is not None
            delete.delete_file(source)
        completed.append(row["source"])
        evidence.record("cleanup_completed", action=row)
    return {"ok": True, "executed": True, "completed": completed,
            "deleted": sum(r["action"] == "remove" for r in rows)}
