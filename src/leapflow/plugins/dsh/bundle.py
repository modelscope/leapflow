"""Safe copying and runtime preparation for DSH source bundles."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from leapflow.learning.compatibility.protocol import PluginSourceKind
from leapflow.learning.compatibility.source_inspector import (
    SourceInspection,
    inspect_plugin_source,
)


class DshBundleError(RuntimeError):
    """A DSH bundle could not be prepared for restricted execution."""


def stage_runtime_bundle(
    inspection: SourceInspection,
    dsh_root: str | Path,
    plugin_id: str,
) -> tuple[Path, str]:
    """Copy one inspected source into a private staging directory.

    Returns ``(staging_root, runtime_entry)``. The caller owns atomic promotion
    to the final directory after restricted discovery succeeds.
    """
    destination_root = Path(dsh_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    staging = destination_root / f".{plugin_id}.staging-{uuid.uuid4().hex}"
    source = Path(inspection.execution_plan.source_root).resolve()
    try:
        staging.mkdir(parents=False, exist_ok=False)
        _copy_regular_files(source, staging)
        # Re-inspect the copied source before adding any generated runtime file.
        # This closes the source-inspection/copy race: approval and audit refer to
        # the exact bytes that are eventually installed.
        copied = inspect_plugin_source(staging)
        if copied.execution_plan.bundle_sha256 != inspection.execution_plan.bundle_sha256:
            raise DshBundleError("DSH source changed while it was being staged")
        if inspection.execution_plan.source_kind == PluginSourceKind.CORDIS_DYNAMIC_EXPORT:
            runtime_entry = "host.runtime.cjs"
            host_source = (staging / "host.js").read_text(encoding="utf-8")
            wrapper = (
                '"use strict";\n'
                "module.exports = (function () {\n"
                f"{host_source}\n"
                "})();\n"
            )
            _atomic_write_text(staging / runtime_entry, wrapper)
        else:
            runtime_entry = inspection.execution_plan.entry_point
        return staging, runtime_entry
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def promote_staging_bundle(staging: Path, final_root: Path) -> None:
    """Atomically make a staged bundle visible; refuse replacement."""
    if final_root.exists():
        raise DshBundleError(f"DSH plugin bundle already exists: {final_root}")
    staging.replace(final_root)


def _copy_regular_files(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise DshBundleError(f"DSH bundle contains a symlink: {item}")
        relative = item.resolve().relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            raise DshBundleError(f"DSH bundle contains a non-regular file: {item}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target, follow_symlinks=False)
        shutil.copymode(item, target, follow_symlinks=False)


def _atomic_write_text(path: Path, content: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
