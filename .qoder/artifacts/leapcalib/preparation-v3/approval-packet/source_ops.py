"""Private, board-local fingerprint-bound source operations; no service or hardware control.

Default inspect is read-only. Backup/apply/restore require exact explicit approval.
Source replacement and restoration refuse while the sole supervisor is active.
Restore quarantines newly added files instead of deleting them. No config/key files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path: Path) -> str | None:
    if path.is_symlink():
        raise ValueError('symlink refused: ' + str(path))
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError('regular source file required: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('relative source path required')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('source path escapes declared root')
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError('symlinked parent refused')
        if parent == root:
            break
    return path


def load(plan_path: Path) -> tuple[dict, str]:
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if plan.get('schema') != 'leapflow.private_source_operations.v1':
        raise ValueError('exact private source plan required')
    if not 1 <= len(plan['files']) <= 5000:
        raise ValueError('bounded source manifest required')
    seen = set()
    for row in plan['files']:
        name = row['path']
        if name in seen or not (name.startswith(('src/', 'tools/')) or name in ('pyproject.toml', 'uv.lock')):
            raise ValueError('duplicate or non-source deployment path')
        if name.startswith('src/leapflow/engine/'):
            raise ValueError('LLM engine source is excluded')
        seen.add(name)
        for key in ('before_sha256', 'after_sha256'):
            value = row[key]
            if value is None and key == 'before_sha256':
                continue
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('exact source fingerprints required')
        for key in ('target_root', 'staged_root', 'backup_root'):
            root = Path(plan[key])
            if not root.is_absolute():
                raise ValueError('absolute deployment roots required')
            child(root, name)
    roots = [Path(plan[key]).resolve() for key in ('target_root', 'staged_root', 'backup_root')]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError('source, staging and backup roots must not overlap')
    return plan, hashlib.sha256(raw).hexdigest()


def inspect(plan: dict) -> dict:
    root, staged = Path(plan['target_root']), Path(plan['staged_root'])
    return {'ok': True, 'operation': 'inspect', 'executed': False, 'files': [
        {'path': r['path'], 'before_matches': sha(child(root, r['path'])) == r['before_sha256'],
         'staged_matches': sha(child(staged, r['path'])) == r['after_sha256']} for r in plan['files']]}


def supervisor_stopped(plan: dict) -> None:
    result = subprocess.run(['systemctl', '--user', 'is-active', plan['supervisor']],
        capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 3 or result.stdout.strip() != 'inactive':
        raise ValueError('sole supervisor must be explicitly stopped before source apply/restore')


def record(path: Path, data: dict) -> None:
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_backup(plan: dict, digest: str) -> None:
    backup = Path(plan['backup_root'])
    marker = json.loads((backup / 'backup.json').read_text())
    if marker.get('plan_sha256') != digest:
        raise ValueError('backup is not bound to this exact source plan')
    for row in plan['files']:
        if row['before_sha256'] is not None and sha(child(backup / 'original', row['path'])) != row['before_sha256']:
            raise ValueError('original backup drift')


def backup(plan: dict, digest: str) -> dict:
    root, destination = Path(plan['target_root']), Path(plan['backup_root'])
    if any(sha(child(root, r['path'])) != r['before_sha256'] for r in plan['files']):
        raise ValueError('target changed; no backup or source operation')
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for row in plan['files']:
        if row['before_sha256'] is not None:
            target = child(destination / 'original', row['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child(root, row['path']), target)
            if sha(target) != row['before_sha256']:
                raise ValueError('source changed during backup')
    record(destination / 'backup.json', {'plan_sha256': digest, 'files': plan['files']})
    return {'ok': True, 'operation': 'backup', 'source_changed': False}


def replace_file(source: Path, target: Path, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name('.' + target.name + '.camera-v3.pending')
    with pending.open('xb') as stream:
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    pending.chmod(mode)
    os.replace(pending, target)


def apply(plan: dict, digest: str) -> dict:
    supervisor_stopped(plan)
    verify_backup(plan, digest)
    status = inspect(plan)
    if not all(r['before_matches'] and r['staged_matches'] for r in status['files']):
        raise ValueError('target/staged drift; no source replacement')
    root, staged, destination = (Path(plan[k]) for k in ('target_root', 'staged_root', 'backup_root'))
    record(destination / 'apply.started.json', {'plan_sha256': digest})
    for row in plan['files']:
        target = child(root, row['path'])
        if sha(target) != row['before_sha256']:
            raise ValueError('partial apply: target changed; inspect and request restore approval')
        source = child(staged, row['path'])
        if sha(source) != row['after_sha256']:
            raise ValueError('partial apply: staged source changed; do not replay')
        replace_file(source, target, row['before_mode'] or 0o644)
        if sha(target) != row['after_sha256']:
            raise ValueError('partial apply: post-copy fingerprint mismatch')
    record(destination / 'apply.complete.json', {'plan_sha256': digest})
    return {'ok': True, 'operation': 'apply', 'files': len(plan['files']), 'service_started': False}


def restore(plan: dict, digest: str) -> dict:
    supervisor_stopped(plan)
    verify_backup(plan, digest)
    root, destination = Path(plan['target_root']), Path(plan['backup_root'])
    if not (destination / 'apply.started.json').is_file():
        raise ValueError('no applied source operation to restore')
    for row in plan['files']:
        if sha(child(root, row['path'])) not in (row['before_sha256'], row['after_sha256']):
            raise ValueError('later source change; restore refuses to overwrite it')
    record(destination / 'restore.started.json', {'plan_sha256': digest})
    for row in plan['files']:
        target = child(root, row['path'])
        if sha(target) == row['before_sha256']:
            continue
        if row['before_sha256'] is None:
            quarantine = child(destination / 'quarantine', row['path'])
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, quarantine)
        else:
            replace_file(child(destination / 'original', row['path']), target, row['before_mode'])
        if sha(target) != row['before_sha256']:
            raise ValueError('partial restore: fingerprint mismatch; inspect before continuing')
    record(destination / 'restore.complete.json', {'plan_sha256': digest})
    return {'ok': True, 'operation': 'restore', 'files_deleted': 0, 'service_started': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--operation', choices=('inspect', 'backup', 'apply', 'restore'), default='inspect')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approved-plan-sha256', default='')
    parser.add_argument('--approval-note', default='')
    args = parser.parse_args()
    plan, digest = load(args.plan)
    if args.operation == 'inspect':
        result = inspect(plan)
    else:
        if not args.execute or args.approved_plan_sha256 != digest or not args.approval_note.strip():
            raise ValueError('exact command/impact approval required; no filesystem write')
        os.umask(0o077)
        result = {'backup': backup, 'apply': apply, 'restore': restore}[args.operation](plan, digest)
    print(json.dumps({**result, 'plan_sha256': digest, 'physical_motion_performed': False}, indent=2))


if __name__ == '__main__':
    main()
