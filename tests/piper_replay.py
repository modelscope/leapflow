"""Test-only historical source snapshots, never an operational hardware entry.

Snapshots use .py.txt so they cannot be run/imported as normal SDK scripts.
Only this explicitly installed test finder loads them. Regeneration copies the
exact Python dependency closure; it never executes source or includes secrets,
profiles, calibration records, images, logs, or vendor SDK packages.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import CodeType, ModuleType

from pytest import MonkeyPatch

ROOT = Path(__file__).parent / "_fixtures/piper_legacy"
SEEDS = (
    "cl/mission.py", "cl/config.py", "cl/eye.py", "cl/plan.py",
    "cl/wrist_verify.py", "cl/bridge.py", "cl_board/cl_arm_core.py",
    "cl_board/cl_armd.py", "grasp_execute.py", "c5_jaw_test.py",
    "wrist_handeye.py", "dual_camera_qualify.py", "qualification_report.py",
)


class ReplayLoader(SourceFileLoader):
    """Compile source data without creating bytecode beside immutable snapshots."""

    def get_code(self, fullname: str) -> CodeType:
        return compile(self.get_data(self.path), self.path, "exec")


def load_script(name: str) -> ModuleType:
    """Load immutable replay data under a test-only module identity."""
    path = ROOT / f"{name}.py.txt"
    if name not in {p.removesuffix(".py") for p in SEEDS}:
        raise ValueError("unlisted historical test entry")
    identity = "offline_" + name.replace("/", "_")
    loader = ReplayLoader(identity, str(path))
    spec = importlib.util.spec_from_file_location(identity, path, loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class ReplayFinder(importlib.abc.MetaPathFinder):
    """Resolve only historical namespaces while the pytest fixture is active."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> ModuleSpec | None:
        parts = fullname.split(".")
        if parts[0] in ("piper_sdk", "pyAgxArm", "pyrealsense2", "can"):
            raise ImportError("vendor hardware SDKs are forbidden in historical offline replay")
        if parts[0] in ("cl", "offline_piper_cl"):
            relative = Path("cl", *parts[1:])
        elif len(parts) == 1 and parts[0] in ("cl_arm_core", "c5_jaw_test", "dual_camera_qualify", "grasp_execute"):
            relative = Path("cl_board", parts[0]) if parts[0] == "cl_arm_core" else Path(parts[0])
        else:
            return None
        source = ROOT / relative.with_suffix(".py.txt")
        package = ROOT / relative / "__init__.py.txt"
        if package.is_file():
            source = package
        elif not source.is_file():
            if len(parts) == 1 and (ROOT / relative).is_dir():
                spec = ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(ROOT / relative)]
                return spec
            return None
        return importlib.util.spec_from_file_location(
            fullname, source, loader=ReplayLoader(fullname, str(source)),
            submodule_search_locations=[str(package.parent)] if source == package else None,
        )


def historical_name(name: str) -> bool:
    return name in ("cl", "offline_piper_cl", "cl_arm_core", "c5_jaw_test", "dual_camera_qualify", "grasp_execute") or name.startswith(("cl.", "offline_piper_cl."))


@contextmanager
def imports(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Scope import routing and cached legacy modules to one test, including failures."""
    original = {name: value for name, value in sys.modules.items() if historical_name(name)}
    for name in original:
        sys.modules.pop(name)
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "meta_path", [ReplayFinder(), *sys.meta_path])
        scoped.setattr(sys, "path", list(sys.path))
        try:
            yield
        finally:
            for name in tuple(sys.modules):
                if historical_name(name):
                    sys.modules.pop(name)
            sys.modules.update(original)


def freeze(source: Path) -> None:
    """Bulk mechanical snapshot of inspected test inputs; no source execution."""
    source = source.resolve(strict=True)
    pending = [Path(name) for name in SEEDS]
    content: dict[str, bytes] = {}
    while pending:
        relative = pending.pop()
        if str(relative) in content:
            continue
        path = source / relative
        if path.is_symlink() or source not in path.resolve(strict=True).parents:
            raise ValueError("snapshot source traverses an external symlink")
        body = path.read_bytes()
        tree = ast.parse(body, filename=str(relative))
        content[str(relative)] = body
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates.extend(Path(*alias.name.split(".")) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                parent = relative.parent.parts
                base = parent[:len(parent) - node.level + 1] if node.level else ()
                module = tuple((node.module or "").split(".")) if node.module else ()
                prefix = Path(*base, *module)
                candidates.append(prefix)
                candidates.extend(prefix / alias.name for alias in node.names if alias.name != "*")
        for candidate in candidates:
            for filename in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                if (source / filename).is_file():
                    pending.append(filename)
        # The daemon's historical bare import resolves beside its own source.
        if relative == Path("cl_board/cl_armd.py"):
            pending.append(Path("cl_board/cl_arm_core.py"))
    if ROOT.exists():
        raise ValueError("snapshot already exists; never overwrite historical replay data")
    ROOT.mkdir(parents=True)
    records = []
    for name, body in sorted(content.items()):
        target = ROOT / (name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        records.append({"source": name, "snapshot": name + ".txt", "bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest()})
    (ROOT / "manifest.json").write_text(json.dumps({
        "version": 1, "purpose": "offline historical replay only", "files": records,
    }, indent=2) + "\n")
    print(f"Frozen {len(records)} historical Python inputs; no hardware access.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze test-only source data; never run hardware scripts")
    parser.add_argument("--freeze-source", type=Path, required=True)
    freeze(parser.parse_args().freeze_source)
