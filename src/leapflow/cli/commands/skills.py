"""Skills subcommand — list, show, export, import, disable, delete, audit, sessions."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from leapflow.cli.helpers import require_initialized

if TYPE_CHECKING:
    from leapflow.cli.context import Context


async def cmd_skills(
    ctx: "Context",
    action: str,
    name: Optional[str],
    output: Optional[str] = None,
    limit: int = 20,
    *,
    include_suggestions: bool = False,
) -> int:
    require_initialized(ctx)
    if action == "list":
        skills = ctx.registry.list_all()
        if not skills:
            print("No skills registered.")
        else:
            print(f"{'Name':<25} {'Version':<10} {'Confidence':<12} Description")
            print("-" * 80)
            for s in skills:
                m = s.metadata
                print(f"{s.name:<25} v{m.version:<9} {m.confidence:<11.0%} {s.description[:40]}")
        if include_suggestions and ctx.skill_lib is not None:
            pending = ctx.skill_lib.query_suggestions(limit=20)
            if pending:
                print()
                print(f"[ PENDING SUGGESTIONS ({len(pending)}) ]")
                print(f"{'Title':<35} {'Similarity':<12} Linked skill")
                print("-" * 80)
                for sug in pending:
                    title = (sug.get("candidate_title") or "")[:34]
                    sim = sug.get("similarity_score", 0.0)
                    linked = sug.get("existing_skill_title", "") or "-"
                    print(f"{title:<35} {sim:<11.0%} {linked}")
        return 0

    if action == "show":
        if not name:
            print("Usage: leap skills show <name>")
            return 1
        skill = ctx.registry.get(name)
        if skill is None:
            print(f"Skill '{name}' not found.")
            return 1
        m = skill.metadata
        print(f"Name:        {skill.name}")
        print(f"Description: {skill.description}")
        print(f"Version:     v{m.version}")
        print(f"Confidence:  {m.confidence:.0%}")
        print(f"Source:      {m.source}")
        if skill.triggers:
            print(f"Triggers:    {', '.join(skill.triggers)}")
        if skill.preconditions:
            print(f"Preconditions: {skill.preconditions}")
        return 0

    if action == "export":
        if not name:
            print("Usage: leap skills export <name> [-o output.json]")
            return 1
        return _export_skill(ctx, name, output)

    if action == "import":
        if not name:
            print("Usage: leap skills import <file.json>")
            return 1
        return _import_skill(ctx, name)

    if action == "disable":
        if not name:
            print("Usage: leap skills disable <name>")
            return 1
        found = False
        if ctx.skill_lib and ctx.skill_lib.deactivate_parameterized(name):
            found = True
        if ctx.registry.unregister(name):
            found = True
        print(f"Skill '{name}' disabled." if found else f"Skill '{name}' not found.")
        return 0 if found else 1

    if action == "delete":
        if not name:
            print("Usage: leap skills delete <name>")
            return 1
        found = False
        if ctx.skill_lib:
            stored = ctx.skill_lib.load_skill_by_title(name)
            if stored:
                stored.status = "deleted"
                ctx.skill_lib.update_skill(stored)
                found = True
            if ctx.skill_lib.deactivate_parameterized(name):
                found = True
        if ctx.registry.unregister(name):
            found = True
        print(f"Skill '{name}' deleted." if found else f"Skill '{name}' not found.")
        return 0 if found else 1

    if action == "audit":
        if ctx.skill_lib is None:
            print("Skill library not available.")
            return 1
        skill_filter = name
        history = ctx.skill_lib.query_history(skill_name=skill_filter, limit=limit)
        if not history:
            print("(no execution history)")
            return 0
        print(
            f"{'Time':<20} {'Event':<24} {'Skill':<20} {'Status':<10} Duration"
        )
        print("-" * 90)
        for row in history:
            ts_raw = row.get("ts") or row.get("timestamp") or ""
            if isinstance(ts_raw, (int, float)):
                from datetime import datetime
                ts = datetime.fromtimestamp(ts_raw).strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts = str(ts_raw)[:19]
            ev = row.get("event", "")
            sk = row.get("skill", "") or row.get("skill_name", "")
            ok = row.get("ok")
            if ok is True:
                st = "ok"
            elif ok is False:
                st = "failed"
            else:
                st = row.get("status", "") or row.get("level", "")
            dur_raw = row.get("duration_s", row.get("duration_ms", ""))
            if isinstance(dur_raw, (int, float)):
                dur = f"{dur_raw:.1f}s"
            else:
                dur = str(dur_raw)
            print(f"{ts:<20} {ev:<24} {sk:<20} {st:<10} {dur}")
        return 0

    if action == "sessions":
        sessions = ctx.session_store.list_recent(limit=limit)
        if not sessions:
            print("No learning sessions recorded.")
            return 0
        from datetime import datetime
        print(f"{'ID':<18} {'Status':<12} {'Steps':<7} {'Goal':<30} Started")
        print("-" * 90)
        for s in sessions:
            ts = datetime.fromtimestamp(s["start_time"]).strftime("%Y-%m-%d %H:%M")
            goal = (s["goal"] or "")[:29]
            traj = ctx.imitation.store.load_trajectory(s["trajectory_id"])
            steps = traj.step_count if traj else "?"
            print(f"{s['session_id']:<18} {s['status']:<12} {steps:<7} {goal:<30} {ts}")
        resumable = [s for s in sessions if s["status"] == "recording"]
        if resumable:
            print()
            print(f"Resumable: {len(resumable)} session(s). Use: leap teach --resume <ID>")
        return 0

    print(f"Unknown skills action: {action}")
    return 1


def _export_skill(ctx: "Context", name: str, output_path: Optional[str]) -> int:
    """Export a skill as a standard SKILL.md folder (or JSON fallback)."""
    if ctx.doc_store and ctx.doc_store.exists(name):
        dest = output_path or name
        dest_path = Path(dest)
        if dest_path.suffix == ".json":
            pass
        else:
            import shutil
            src = ctx.doc_store.skills_dir / name
            dest_path.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                shutil.copy2(f, dest_path / f.name)
            print(f"Skill '{name}' exported to: {dest_path}/")
            return 0

    if ctx.skill_lib is None:
        print("Skill library not available.")
        return 1

    all_skills = ctx.skill_lib.load_all_active()
    skill = None
    for s in all_skills:
        if s.title == name or s.skill_id == name:
            skill = s
            break

    if skill is None:
        reg_skill = ctx.registry.get(name)
        if reg_skill is None:
            print(f"Skill '{name}' not found in library or registry.")
            return 1
        export_data = {
            "name": reg_skill.name,
            "description": reg_skill.description,
            "triggers": reg_skill.triggers,
            "version": reg_skill.metadata.version,
            "confidence": reg_skill.metadata.confidence,
            "source": reg_skill.metadata.source,
        }
    else:
        export_data = {
            "skill_id": skill.skill_id,
            "title": skill.title,
            "trigger_phrases": skill.trigger_phrases,
            "steps": skill.steps,
            "parameters": skill.parameters,
            "pre_conditions": skill.pre_conditions,
            "post_conditions": skill.post_conditions,
            "app_sequence": skill.app_sequence,
            "action_names": skill.action_names,
            "confidence": skill.confidence,
            "version": skill.version,
            "status": skill.status,
        }

    dest = output_path or f"{name.replace(' ', '_')}.skill.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    print(f"Skill '{name}' exported to: {dest}")
    return 0


def _import_skill(ctx: "Context", file_path: str) -> int:
    """Import a skill from a SKILL.md folder or JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return 1

    skill_md = path / "SKILL.md" if path.is_dir() else None
    if skill_md and skill_md.exists():
        if ctx.doc_store is None:
            print("Skill document store not available.")
            return 1
        from leapflow.learning.document import SkillDocParser
        content = skill_md.read_text(encoding="utf-8")
        doc = SkillDocParser().parse(content)
        if not doc.name:
            print("Failed to parse SKILL.md: missing skill name.")
            return 1
        ctx.doc_store.save(doc)
        print(f"Skill '{doc.name}' imported from SKILL.md.")
        return 0

    if ctx.skill_lib is None:
        print("Skill library not available.")
        return 1

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to read file: {e}")
        return 1

    from leapflow.storage.skill_library import StoredSkill

    skill = StoredSkill(
        skill_id=data.get("skill_id", uuid.uuid4().hex[:16]),
        title=data.get("title", data.get("name", "Imported skill")),
        trigger_phrases=data.get("trigger_phrases", data.get("triggers", [])),
        steps=data.get("steps", []),
        parameters=data.get("parameters", []),
        pre_conditions=data.get("pre_conditions", []),
        post_conditions=data.get("post_conditions", []),
        app_sequence=data.get("app_sequence", []),
        action_names=data.get("action_names", []),
        confidence=data.get("confidence", 0.5),
        version=data.get("version", 1),
        status=data.get("status", "active"),
        source_trajectory_id=data.get("source_trajectory_id", ""),
        source_episode_id=data.get("source_episode_id", ""),
    )

    ctx.skill_lib.save_skill(skill)
    print(f"Skill '{skill.title}' imported successfully (id: {skill.skill_id}).")
    return 0
