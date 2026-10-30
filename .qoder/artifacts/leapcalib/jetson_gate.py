"""Read-only deployment gate through the existing authenticated node owner."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ART = Path('/Users/jason/work/leap_work/leapflow/.qoder/artifacts/leapcalib/integration-20261027-212408')
CODE = r'''
import asyncio, fnmatch, hashlib, json, os, pathlib, subprocess
from leapflow.config import load_config
from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.skills.runtime_settings import SkillRuntimeSettings
from leapflow.hardware.skills.workcell import WorkcellSpec
import yaml
os.umask(0o077)
root = pathlib.Path('/home/jetson/leap_work/leapflow')
backup = pathlib.Path('/home/jetson/leap_work/leapcalib-integration-20261027-212408')
manifest = json.loads((backup / 'deployment-manifest.json').read_text())
original = json.loads((backup / 'original-manifest.json').read_text())
for name, info in original['files'].items():
    path = root / name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], 'original changed: ' + name
for item in manifest['files']:
    path = root / item['path']
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    assert actual == item['original_sha256'], 'deployment original changed: ' + item['path']
    staged = backup / 'stage' / item['path']
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == item['merged_sha256'], 'stage changed: ' + item['path']
head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert head == manifest['original_head']
config = load_config()
settings = SkillRuntimeSettings.from_settings(config)
assert any(fnmatch.fnmatchcase('calib.metric_readiness', p) for p in settings.enabled), 'global skill allowlist refuses readiness'
cells = []
for path in settings.workcells_dir.glob('*.yaml'):
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get('workcell') == 'piper_cell':
        spec = WorkcellSpec.from_mapping(raw, source=str(path))
        cells.append((path, raw, spec))
assert len(cells) == 1, 'expected one piper_cell declaration'
path, raw, spec = cells[0]
assert spec.roles.get('camera') == 'sensor_realsense_435if'
assert spec.roles.get('camera.wrist') == 'sensor_realsense_d405'
allowed = not spec.skills or any(fnmatch.fnmatchcase('calib.metric_readiness', p) for p in spec.skills)
async def run():
    token = pathlib.Path('/home/jetson/.leapflow/.lhp-token').read_text().strip()
    client = SocketLhpClient('tcp://127.0.0.1:8767', token=token)
    try:
        hello = await client.connect()
        assert hello['node_id'] == 'yahboom'
        assert hello['write_mode'] == 'permit_required', 'unexpected owner write mode'
        runs = await client.request('skill.status', {})
        control = await client.request('control.status', {})
        assert isinstance(runs.get('runs'), list)
        assert control.get('ok') is True and isinstance(control.get('sessions'), list)
        active = [r for r in runs['runs'] if r.get('status') == 'running' or r.get('outcome') is None]
        assert not active, 'active skill run blocks owner restart'
        assert not control['sessions'], 'active control session blocks owner restart'
        catalog = await client.request('skill.list', {'workcell': 'piper_cell'})
        assert len(catalog.get('workcells', [])) == 1
        assert catalog['workcells'][0]['roles'] == dict(spec.roles)
        service = subprocess.check_output(['systemctl', '--user', 'show', 'leap-lhp-node.service', '-p', 'MainPID', '-p', 'ActiveState'], text=True)
        assert 'ActiveState=active' in service
        data = {'node_id': hello['node_id'], 'write_mode': hello['write_mode'], 'service': service.strip().splitlines(), 'active_runs': 0, 'active_control_sessions': 0, 'original_files_unchanged': True, 'staged_files_verified': len(manifest['files']), 'workcell_path': str(path), 'workcell_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'workcell_readiness_allowed': allowed, 'workcell_skill_patterns': list(spec.skills), 'global_skill_patterns': list(settings.enabled), 'catalog': catalog}
        (backup / 'pre-deploy-gate.json').write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
    finally:
        await client.close()
asyncio.run(run())
'''
result = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1',
     '/home/jetson/miniconda3/envs/leap/bin/python', '-'],
    input=CODE.encode(), capture_output=True, timeout=90,
)
if result.returncode:
    print(result.stderr.decode())
    raise SystemExit(result.returncode)
data = json.loads(result.stdout)
(ART / 'pre-deploy-gate.json').write_text(json.dumps(data, indent=2))
print(json.dumps({k: v for k, v in data.items() if k != 'catalog'}, indent=2))
print('existing_catalog_skill_count:', len(data['catalog']['workcells'][0]['skills']))
