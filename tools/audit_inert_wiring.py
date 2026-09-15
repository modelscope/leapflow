# Copyright (c) Alibaba, Inc. and its affiliates.
"""Audit for capability that is built but never runs.

Three defects of this shape shipped with a green suite in one week: the lifecycle governor
was never constructed (`self.lifecycle_governor` read through `getattr(..., None)` and
assigned nowhere), the durable trust ledger was never handed to it (so one process held two
divergent views of trust and no plugin could earn PRODUCTION), and the hardware trust gate
was linked to a ledger that was always `None`. None of them raise. None of them fail a
test, because every unit test constructs the collaborator itself. The wiring is the one
thing a unit test cannot see.

So this checks the two shapes those took, and -- importantly -- knows the three ways a
dependency can legitimately arrive, because the first two versions of this audit reported
wired code as inert by only knowing one of them:

1. a keyword argument from another module
2. post-construction attribute assignment (``obj._dep = ...``)
3. ``setattr(obj, "_dep", ...)`` from a composer that must not import the type

Known limitation: purely positional construction is not detected, so a finding still has
to be read before it is believed. That is why the output separates *shape* from *verdict*.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("src/leapflow")

#: Names that denote an injected collaborator rather than a value. A missing threshold
#: degrades a decision; a missing collaborator removes a feature.
SHAPES = (
    "sink", "store", "ledger", "provider", "tracker", "observer", "gate", "queue",
    "actor", "coordinator", "verifier", "advisor", "factory", "extractor", "service",
)

OPTIONAL_PARAM = re.compile(r"^\s{4,}([a-z_][a-z0-9_]*)\s*:\s*[^=]*=\s*None\s*,?\s*$", re.M)
SELF_GETATTR = re.compile(r'getattr\(\s*self\s*,\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']')


def _enclosing(text: str, offset: int) -> str:
    head = text[:offset]
    classes = list(re.finditer(r"^class\s+(\w+)", head, re.M))
    return classes[-1].group(1) if classes else "?"


def _supplied(name: str, owner: Path, repo: dict[Path, str]) -> tuple[int, str]:
    """How many other modules supply this dependency, and by which channel."""
    channels = {
        "keyword": re.compile(rf"(?<![\w.]){re.escape(name)}\s*=(?!=)"),
        "attribute": re.compile(rf"\.\_?{re.escape(name)}\s*=(?!=)"),
        "setattr": re.compile(rf'setattr\([^,]+,\s*["\']_?{re.escape(name)}["\']'),
    }
    found: list[str] = []
    count = 0
    for path, text in repo.items():
        if path == owner:
            continue
        hit = [label for label, pattern in channels.items() if pattern.search(text)]
        if hit:
            count += 1
            found.extend(hit)
    return count, ",".join(sorted(set(found))) or "-"


def audit_optional_injections(repo: dict[Path, str]) -> list[tuple[int, str, str, str, str]]:
    """Optional collaborators no other module supplies through any channel."""
    rows: list[tuple[int, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, text in repo.items():
        module = str(path).replace("src/leapflow/", "")
        for match in OPTIONAL_PARAM.finditer(text):
            name = match.group(1)
            if not any(shape in name for shape in SHAPES):
                continue
            if (module, name) in seen:
                continue
            seen.add((module, name))
            count, channel = _supplied(name, path, repo)
            rows.append((count, module, _enclosing(text, match.start()), name, channel))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def audit_self_getattr(repo: dict[Path, str]) -> list[tuple[int, str, str]]:
    """``getattr(self, "x", ...)`` where nothing anywhere ever sets ``x``."""
    rows: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, text in repo.items():
        module = str(path).replace("src/leapflow/", "")
        for name in sorted(set(SELF_GETATTR.findall(text))):
            if (module, name) in seen:
                continue
            seen.add((module, name))
            assigns = sum(
                len(re.findall(rf"\.{re.escape(name)}\s*=(?!=)", other))
                + len(re.findall(rf'setattr\([^,]+,\s*["\']{re.escape(name)}["\']', other))
                for other in repo.values()
            )
            rows.append((assigns, module, name))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def audit_store_direction(repo: dict[Path, str]) -> list[tuple[str, int, int]]:
    """Stores whose writes and reads do not both have production callers.

    ``ExperienceStore`` is the case this catches: the teacher's grades are written into it
    and its only readers are in the hardware subsystem, so the distillation the docstring
    promised terminated in a store nothing on the student's path consulted. A store written
    and never read is data thrown away with extra steps; one read and never written is a
    feature that can only ever return empty.
    """
    rows: list[tuple[str, int, int]] = []
    for path, text in repo.items():
        if "storage/" not in str(path) and "_store.py" not in path.name:
            continue
        for match in re.finditer(r"^class\s+(\w*Store\w*)", text, re.M):
            name = match.group(1)
            if name.startswith("_"):
                continue
            body = text[match.start():]
            writers = {
                m.group(1)
                for m in re.finditer(
                    r"def\s+((?:add|save|record|write|put|set|append|store|update|"
                    r"retract|remove|delete|resolve)\w*)", body
                )
            }
            readers = {
                m.group(1)
                for m in re.finditer(
                    r"def\s+((?:get|load|read|list|query|find|live|count|all|"
                    r"unresolved|recent|for_)\w*)", body
                )
            }
            outside = {p: t for p, t in repo.items() if p != path}
            used = lambda names: sum(  # noqa: E731 - local predicate, read once
                1
                for t in outside.values()
                if any(re.search(rf"\.{re.escape(n)}\s*\(", t) for n in names)
            )
            rows.append((f"{str(path).replace('src/leapflow/', '')}::{name}",
                         used(writers) if writers else -1,
                         used(readers) if readers else -1))
    return sorted(rows)


def main() -> None:
    repo = {p: p.read_text(encoding="utf-8") for p in sorted(ROOT.rglob("*.py"))}

    print("=" * 100)
    print("A1  getattr(self, \"x\") 且全仓无任何赋值 —— 功能静默不运行")
    print("=" * 100)
    a1 = [row for row in audit_self_getattr(repo) if row[0] == 0]
    for _, module, name in a1:
        print(f"  ⚠  {module:<48} {name}")
    print(f"  合计 {len(a1)} 处")

    print()
    print("=" * 100)
    print("A2  可选协作者，无任何模块通过 keyword/attribute/setattr 供给")
    print("=" * 100)
    rows = audit_optional_injections(repo)
    inert = [row for row in rows if row[0] == 0]
    for _, module, cls, name, _channel in inert:
        print(f"  ⚠  {module:<40} {cls:<28} {name}")
    print(f"  合计 {len(inert)} 处（共检查 {len(rows)} 个可选注入点）")

    print()
    print("=" * 100)
    print("A3  存储的写侧或读侧在生产中无调用方 —— 数据白写，或功能恒空")
    print("=" * 100)
    oneway = 0
    for name, writers, readers in audit_store_direction(repo):
        if writers == 0 or readers == 0:
            oneway += 1
            missing = "写侧无调用方" if writers == 0 else "读侧无调用方"
            print(f"  ⚠  {name:<62} {missing}")
    print(f"  合计 {oneway} 处")


if __name__ == "__main__":
    main()
