"""Task-scoped private backup and three-way staging; never starts a physical owner."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

HOST = Path('/Users/jason/work/leap_work/leapflow')
BOARD = '/home/jetson/leap_work/leapflow'
BASE = 'b3883a1d21b00ee21f136dcffe52556b194fb62e'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1']
os.umask(0o077)
ARTIFACT = HOST / '.qoder/artifacts/leapcalib' / time.strftime('integration-%Y%m%d-%H%M%S')
ARTIFACT.mkdir(mode=0o700)


def remote(code: str, payload: dict) -> dict:
    result = subprocess.run(
        SSH + ['python3', '-c', __import__('shlex').quote(code)],
        input=json.dumps(payload).encode(), capture_output=True, check=True, timeout=120,
    )
    return json.loads(result.stdout)


paths = [str(p.relative_to(HOST)) for root in ('src/leapcalib', 'src/leapflow/hardware/calibration')
         for p in (HOST / root).rglob('*.py')]
paths += [
    'pyproject.toml', 'uv.lock', 'src/leapflow/hardware/registry.py',
    'src/leapflow/hardware/skills/ports.py', 'src/leapflow/hardware/skills/vision.py',
    'src/leapflow/hardware/lhp/runtime.py',
    'src/leapflow/hardware/skills/builtin/metric_readiness.py', 'tools/calib_regression.py',
    'tests/test_leapcalib_contracts.py', 'tests/test_leapcalib_validation.py',
    'tests/test_leapcalib_estimation.py', 'tests/test_leapcalib_readiness.py',
]
paths = sorted(set(paths))

snapshot_code = r'''
import base64, hashlib, json, os, pathlib, subprocess, sys, tarfile
os.umask(0o077)
p = json.load(sys.stdin)
root = pathlib.Path(p['root'])
backup = root.parent / p['backup_name']
backup.mkdir(mode=0o700)
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args])
head = git('rev-parse', 'HEAD').decode().strip()
assert head == p['base'], 'board HEAD changed'
dirty = {x.decode() for x in git('ls-files', '-m', '-o', '--exclude-standard', '-z').split(b'\0') if x}
selected = sorted(dirty | set(p['paths']))
manifest = {}
for name in selected:
    path = root / name
    assert not path.is_symlink(), 'refuse symlink: ' + name
    if path.is_file():
        data = path.read_bytes()
        manifest[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'mode': path.stat().st_mode & 0o777}
with tarfile.open(backup / 'original-files.tar.gz', 'w:gz') as archive:
    for name in selected:
        if (root / name).is_file():
            archive.add(root / name, arcname=name, recursive=False)
os.chmod(backup / 'original-files.tar.gz', 0o600)
for name, args in [('working.patch', ('diff', '--binary')), ('index.patch', ('diff', '--cached', '--binary'))]:
    path = backup / name
    path.write_bytes(git(*args))
    path.chmod(0o600)
record = {'head': head, 'dirty': sorted(dirty), 'files': manifest}
(backup / 'original-manifest.json').write_text(json.dumps(record, sort_keys=True))
(backup / 'original-manifest.json').chmod(0o600)
contents = {}
for name in p['paths']:
    if name in manifest:
        contents[name] = base64.b64encode((root / name).read_bytes()).decode()
print(json.dumps({'backup': str(backup), 'record': record, 'contents': contents}))
'''
snapshot = remote(snapshot_code, {
    'root': BOARD, 'backup_name': 'leapcalib-' + ARTIFACT.name,
    'base': BASE, 'paths': paths,
})
(ARTIFACT / 'board-code-snapshot.json').write_text(json.dumps(snapshot))
(ARTIFACT / 'board-code-snapshot.json').chmod(0o600)
merged = ARTIFACT / 'merged'
merged.mkdir(mode=0o700)
conflicts = []
manifest = []
for name in paths:
    host = (HOST / name).read_bytes()
    board = base64.b64decode(snapshot['contents'][name]) if name in snapshot['contents'] else None
    base = subprocess.run(['git', '-C', str(HOST), 'show', BASE + ':' + name], capture_output=True)
    if board is None:
        result = host
    elif board == host:
        result = board
    elif base.returncode != 0:
        conflicts.append(name)
        continue
    else:
        with tempfile.TemporaryDirectory(dir=ARTIFACT) as work:
            current, ancestor, incoming = [Path(work) / part for part in ('board', 'base', 'host')]
            current.write_bytes(board)
            ancestor.write_bytes(base.stdout)
            incoming.write_bytes(host)
            merge = subprocess.run(['git', 'merge-file', '-p', str(current), str(ancestor), str(incoming)], capture_output=True)
            if merge.returncode != 0:
                conflicts.append(name)
                target = merged / (name + '.conflict')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(merge.stdout)
                continue
            result = merge.stdout
    target = merged / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(result)
    manifest.append({
        'path': name,
        'original_sha256': hashlib.sha256(board).hexdigest() if board is not None else None,
        'merged_sha256': hashlib.sha256(result).hexdigest(),
        'host_sha256': hashlib.sha256(host).hexdigest(),
        'changed': result != board,
    })
report = {
    'backup': snapshot['backup'], 'artifact': str(ARTIFACT),
    'original_dirty_count': len(snapshot['record']['dirty']),
    'original_head': snapshot['record']['head'], 'files': manifest, 'conflicts': conflicts,
}
(ARTIFACT / 'reconcile-manifest.json').write_text(json.dumps(report, indent=2))
(ARTIFACT / 'reconcile-manifest.json').chmod(0o600)
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
print('Clean staged files:', len(manifest))
if conflicts:
    raise SystemExit(1)
