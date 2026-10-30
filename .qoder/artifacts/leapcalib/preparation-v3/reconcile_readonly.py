"""Private read-only target comparison; all artifacts stay on the host.

Never uploads, backs up on the board, deploys, restarts, or accesses credentials.
Only explicit source/tool paths from the fingerprinted host inventory are read.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess

ROOT = Path('/Users/jason/work/leap_work/leapflow')
TARGET = '/home/jetson/leap_work/leapflow'
OUTPUT = ROOT / '.qoder/artifacts/leapcalib/preparation-v3/target-reconciliation-v2'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1']
os.umask(0o077)
tool = runpy.run_path(str(ROOT / 'tools/calib_preview.py'))
host = tool['inventory'](ROOT)

REMOTE = r'''
import base64, hashlib, json, pathlib, subprocess, sys
p = json.load(sys.stdin)
root = pathlib.Path(p['root']).resolve(strict=True)
def git(*args):
    return subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args],
        capture_output=True, timeout=15, check=False)
head = git('rev-parse', 'HEAD')
assert head.returncode == 0
head = head.stdout.decode().strip()
assert len(p['paths']) <= 5000
rows = {}
for name, expected in p['paths'].items():
    relative = pathlib.PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert name.startswith(('src/', 'tools/')) or name in ('pyproject.toml', 'uv.lock', 'README.md', 'LICENSE')
    path = root / name
    assert not path.is_symlink()
    assert path.resolve().is_relative_to(root)
    if path.exists():
        assert path.is_file() and path.stat().st_size <= 4 * 1024 * 1024
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
    else:
        raw, actual = None, None
    row = {'sha256': actual, 'mode': path.stat().st_mode & 0o777 if raw is not None else None}
    if actual != expected:
        ancestor = git('show', head + ':' + name)
        row['current'] = base64.b64encode(raw).decode() if raw is not None else None
        row['base'] = base64.b64encode(ancestor.stdout).decode() if ancestor.returncode == 0 else None
    rows[name] = row
status = git('status', '--porcelain=v1', '--untracked-files=all')
assert status.returncode == 0
print(json.dumps({'schema': 'leapflow.private_source_snapshot.v1', 'head': head,
    'status': status.stdout.decode(), 'files': rows, 'remote_changes_performed': False}))
'''
response = subprocess.run(SSH + ['python3', '-c', shlex.quote(REMOTE)],
    input=json.dumps({'root': TARGET, 'paths': host}).encode(),
    capture_output=True, timeout=180, check=True)
if len(response.stdout) > 32 * 1024 * 1024:
    raise ValueError('target snapshot exceeds 32MiB')
snapshot = json.loads(response.stdout)
if tool['inventory'](ROOT) != host:
    raise ValueError('host source drifted during read-only snapshot')
OUTPUT.mkdir(mode=0o700, exist_ok=False)
(OUTPUT / 'snapshot.json').write_bytes(response.stdout)
rows = []
for name, expected in host.items():
    observed = snapshot['files'][name]
    row = {'path': name, 'host_sha256': expected, 'target_sha256': observed['sha256']}
    if observed['sha256'] == expected:
        row['decision'] = 'identical'
    else:
        incoming = (ROOT / name).read_bytes()
        current = base64.b64decode(observed['current']) if observed['current'] is not None else None
        ancestor = base64.b64decode(observed['base']) if observed['base'] is not None else None
        row['base_sha256'] = hashlib.sha256(ancestor).hexdigest() if ancestor is not None else None
        merged = None
        if current == ancestor:
            row['decision'], merged = 'host_only_change', incoming
        elif incoming == ancestor:
            row['decision'], merged = 'retain_target_change', current
        elif current is None:
            row['decision'] = 'target_deleted_host_changed'
        elif ancestor is None:
            row['decision'] = 'independent_addition_conflict'
        else:
            directory = OUTPUT / 'merge-inputs' / name
            directory.mkdir(parents=True)
            for label, data in [('target', current), ('base', ancestor), ('host', incoming)]:
                (directory / label).write_bytes(data)
            result = subprocess.run(['git', 'merge-file', '-p', str(directory / 'target'),
                str(directory / 'base'), str(directory / 'host')], capture_output=True, timeout=15)
            if result.returncode == 0:
                row['decision'], merged = 'clean_three_way_merge', result.stdout
            elif 1 <= result.returncode <= 127:
                row['decision'] = 'three_way_conflict'
                (directory / 'conflict').write_bytes(result.stdout)
            else:
                raise ValueError('three-way merge failed: ' + name)
        if merged is not None:
            destination = OUTPUT / 'merged' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(merged)
            row['merged_sha256'] = hashlib.sha256(merged).hexdigest()
            row['merged_differs_from_host'] = merged != incoming
    rows.append(row)
conflicts = [r['path'] for r in rows if 'conflict' in r['decision'] or r['decision'] == 'target_deleted_host_changed']
retained = [r['path'] for r in rows if r.get('merged_differs_from_host')]
report = {'schema': 'leapflow.private_source_reconciliation.v1', 'target_root': TARGET,
    'head': snapshot['head'], 'target_dirty_entries': len(snapshot['status'].splitlines()),
    'source_inventory_sha256': tool['fingerprint'](json.dumps(host, sort_keys=True).encode()),
    'snapshot_sha256': hashlib.sha256(response.stdout).hexdigest(), 'files': rows,
    'conflicts': conflicts, 'retained_target_differences': retained,
    'remote_backup_created': False, 'remote_changes_performed': False,
    'physical_motion_performed': False, 'deployment_approved': False,
    'state': 'needs_conflict_resolution' if conflicts else 'read_only_reconciled_not_deployed',
    'notes': ['A fresh private board-local backup still requires exact approval.',
        'Recheck target fingerprints before any source replacement.',
        'Host wheel is not the reconciled target tree when target differences remain.']}
(OUTPUT / 'reconciliation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
