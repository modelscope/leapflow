"""Finalize inspected staging conflicts and upload to a private non-owning test tree."""
from __future__ import annotations
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path('/Users/jason/work/leap_work/leapflow')
ART = ROOT / '.qoder/artifacts/leapcalib/integration-20261027-212408'
report = json.loads((ART / 'reconcile-manifest.json').read_text())
snapshot = json.loads((ART / 'board-code-snapshot.json').read_text())
os.umask(0o077)
pattern = re.compile(r'^<<<<<<< [^\n]+\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]+\n', re.M | re.S)
# Every branch was inspected: these conflicts are formatting or the new READ port.
expected = {
    'src/leapflow/hardware/skills/ports.py': [''],
    'src/leapflow/hardware/skills/vision.py': [
        '    def __init__(\n        self, *, registry: Any, camera: str, target: CharucoTarget | None = None\n    ) -> None:\n',
        '    def for_workcell(cls, registry: Any, spec: WorkcellSpec, *, role: str = ROLE_CAMERA) -> "WorkcellVision":\n',
        '        points = await asyncio.to_thread(\n            cloud_points, frame, near_mm=near_mm, far_mm=far_mm\n        )\n',
        '    async def _capture(self) -> RgbdFrame:\n',
        '                return await source.capture_rgbd()\n',
    ],
}
for name in report['conflicts']:
    source = ART / 'merged' / (name + '.conflict')
    text = source.read_text()
    matches = list(pattern.finditer(text))
    if name in expected:
        assert [m.group(1) for m in matches] == expected[name], 'uninspected conflict: ' + name
        text = pattern.sub(lambda m: m.group(2), text)
    else:
        assert not matches, 'unresolved manually inspected configuration: ' + name
    assert not any(marker in text for marker in ('<<<<<<< ', '\n=======\n', '>>>>>>> '))
    path = ART / 'merged' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    board = base64.b64decode(snapshot['contents'][name])
    report['files'].append({
        'path': name, 'original_sha256': hashlib.sha256(board).hexdigest(),
        'merged_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'host_sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
        'changed': text.encode() != board,
    })
report['resolved_conflicts'] = report.pop('conflicts')
for item in report['files']:
    path = ART / 'merged' / item['path']
    if path.suffix == '.py':
        ast.parse(path.read_text())
report['files'].sort(key=lambda x: x['path'])
(ART / 'deployment-manifest.json').write_text(json.dumps(report, indent=2))
files = {item['path']: base64.b64encode((ART / 'merged' / item['path']).read_bytes()).decode()
         for item in report['files'] if item['changed']}
code = r'''
import base64, hashlib, json, os, pathlib, shutil, sys
os.umask(0o077)
p = json.load(sys.stdin)
root = pathlib.Path('/home/jetson/leap_work/leapflow')
backup = pathlib.Path(p['backup'])
original = json.loads((backup / 'original-manifest.json').read_text())
for name, info in original['files'].items():
    path = root / name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], 'original changed: ' + name
stage = backup / 'stage'
stage.mkdir(mode=0o700)
for name in ('src', 'tests', 'tools'):
    shutil.copytree(root / name, stage / name, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', '.piper-node.lock', 'data'))
for name in ('README.md', 'pyproject.toml', 'uv.lock'):
    shutil.copy2(root / name, stage / name)
for name, encoded in p['files'].items():
    path = stage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(encoded))
(backup / 'deployment-manifest.json').write_text(json.dumps(p['manifest'], indent=2))
print(json.dumps({'stage': str(stage), 'changed_files': len(p['files']), 'original_files_unchanged': True}))
'''
result = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1',
     'python3', '-c', shlex.quote(code)],
    input=json.dumps({'backup': report['backup'], 'files': files, 'manifest': report}).encode(),
    capture_output=True, timeout=180,
)
if result.returncode:
    print(result.stderr.decode())
    raise SystemExit(result.returncode)
print(result.stdout.decode())
