"""Apply only fingerprinted, tested milestone files and refresh the same editable owner."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ART = Path('/Users/jason/work/leap_work/leapflow/.qoder/artifacts/leapcalib/integration-20261027-212408')
CODE = r'''
import asyncio, hashlib, importlib.metadata, json, os, pathlib, shutil, subprocess, time
from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.protocol import LhpRemoteError
os.umask(0o077)
root = pathlib.Path('/home/jetson/leap_work/leapflow')
backup = pathlib.Path('/home/jetson/leap_work/leapcalib-integration-20261027-212408')
python = '/home/jetson/miniconda3/envs/leap/bin/python'
service = 'leap-lhp-node.service'
unit = pathlib.Path('/home/jetson/.config/systemd/user') / service
unit_hash = hashlib.sha256(unit.read_bytes()).hexdigest()
manifest = json.loads((backup / 'deployment-manifest.json').read_text())
original = json.loads((backup / 'original-manifest.json').read_text())
gate = json.loads((backup / 'pre-deploy-gate.json').read_text())
assert gate['workcell_readiness_allowed'] is True
workcell = pathlib.Path(gate['workcell_path'])
assert hashlib.sha256(workcell.read_bytes()).hexdigest() == gate['workcell_sha256']
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == manifest['original_head']
for name, info in original['files'].items():
    path = root / name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], 'original changed: ' + name
changed = [item for item in manifest['files'] if item['changed']]
for item in manifest['files']:
    path = root / item['path']
    assert not path.is_symlink()
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    assert actual == item['original_sha256'], 'original changed: ' + item['path']
    assert hashlib.sha256((backup / 'stage' / item['path']).read_bytes()).hexdigest() == item['merged_sha256']
restore = backup / 'pre-apply-files'
restore.mkdir(mode=0o700)
for item in changed:
    path = root / item['path']
    if path.exists():
        dest = restore / item['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
metadata = backup / 'pre-apply-metadata'
metadata.mkdir(mode=0o700)
dist = importlib.metadata.distribution('leapflow')
shutil.copytree(dist._path, metadata / dist._path.name)
for ep in dist.entry_points:
    if ep.group == 'console_scripts':
        path = pathlib.Path(python).parent / ep.name
        if path.exists():
            shutil.copy2(path, metadata / ep.name)
token = pathlib.Path('/home/jetson/.leapflow/.lhp-token').read_text().strip()
async def snapshot():
    client = SocketLhpClient('tcp://127.0.0.1:8767', token=token)
    try:
        hello = await client.connect()
        assert hello['node_id'] == gate['node_id'] and hello['write_mode'] == gate['write_mode']
        runs = await client.request('skill.status', {})
        sessions = await client.request('control.status', {})
        assert isinstance(runs.get('runs'), list) and isinstance(sessions.get('sessions'), list)
        assert sessions.get('ok') is True and not sessions['sessions'], 'active control session blocks deploy'
        assert not any(r.get('status') == 'running' or r.get('outcome') is None for r in runs['runs']), 'active skill blocks deploy'
        return await client.request('skill.list', {'workcell': 'piper_cell'})
    finally:
        await client.close()
before = asyncio.run(snapshot())
state = subprocess.check_output(['systemctl', '--user', 'show', service, '-p', 'MainPID', '-p', 'ActiveState'], text=True)
assert 'ActiveState=active' in state
report = {'backup': str(backup), 'original_service': state.strip().splitlines(), 'changed_files': len(changed), 'no_active_skill_or_control_session': True, 'applied': False, 'metadata_refreshed': False, 'same_owner_verified': False, 'rollback_existing_files': str(restore), 'rollback_metadata': str(metadata)}
(backup / 'apply-report.json').write_text(json.dumps(report, indent=2))
subprocess.run(['systemctl', '--user', 'stop', service], check=True, timeout=45)
try:
    for item in changed:
        path = root / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = backup / 'stage' / item['path']
        temporary = path.with_name(path.name + '.leapcalib-stage')
        shutil.copy2(staged, temporary)
        if item['original_sha256'] is not None:
            os.chmod(temporary, (restore / item['path']).stat().st_mode & 0o777)
        os.replace(temporary, path)
    report['applied'] = True
    log = backup / 'editable-install.log'
    with log.open('wb') as out:
        subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '--editable', str(root)], stdout=out, stderr=subprocess.STDOUT, check=True, timeout=240)
    report['metadata_refreshed'] = True
    subprocess.run(['systemctl', '--user', 'start', service], check=True, timeout=45)
except (OSError, subprocess.SubprocessError):
    for item in changed:
        saved = restore / item['path']
        if saved.exists():
            shutil.copy2(saved, root / item['path'])
    # New orphan source files are retained; the old entry points are restored without deletion.
    with (backup / 'rollback-install.log').open('wb') as out:
        rollback = subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '--editable', str(root)], stdout=out, stderr=subprocess.STDOUT, timeout=240)
    subprocess.run(['systemctl', '--user', 'start', service], check=True, timeout=45)
    report['rolled_back_existing_files'] = True
    report['rollback_metadata_success'] = rollback.returncode == 0
    (backup / 'apply-report.json').write_text(json.dumps(report, indent=2))
    raise
async def verify():
    last = None
    for _ in range(60):
        try:
            catalog = await snapshot()
            assert len(catalog.get('workcells', [])) == 1
            current = catalog['workcells'][0]
            prior = before['workcells'][0]
            assert {k:v for k,v in current.items() if k != 'skills'} == {k:v for k,v in prior.items() if k != 'skills'}, 'workcell or calibration changed'
            old = {s['skill_id']: s for s in prior['skills']}
            new = {s['skill_id']: s for s in current['skills']}
            assert old.keys() <= new.keys(), 'prior skill disappeared'
            for name, spec in old.items():
                assert new[name] == spec, 'prior skill capability changed: ' + name
            ready = new['calib.metric_readiness']
            assert ready['available'] is True and ready['effect'] == 'read'
            return catalog
        except (OSError, TimeoutError, LhpRemoteError) as exc:
            last = type(exc).__name__
            await asyncio.sleep(1)
    raise RuntimeError('same owner did not return: ' + str(last))
after = asyncio.run(verify())
assert hashlib.sha256(unit.read_bytes()).hexdigest() == unit_hash
assert hashlib.sha256(workcell.read_bytes()).hexdigest() == gate['workcell_sha256']
changed_names = {item['path'] for item in changed}
for name, info in original['files'].items():
    if name not in changed_names:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == info['sha256'], 'unrelated original changed: ' + name
for item in changed:
    assert hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['merged_sha256']
report.update({'same_owner_verified': True, 'service': subprocess.check_output(['systemctl', '--user', 'show', service, '-p', 'MainPID', '-p', 'ActiveState'], text=True).strip().splitlines(), 'prior_skills_retained': len(before['workcells'][0]['skills']), 'new_skills': len(after['workcells'][0]['skills']), 'workcell_unchanged': True, 'unit_unchanged': True, 'unrelated_original_files_unchanged': True, 'completed_at': time.time(), 'catalog': after})
(backup / 'apply-report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
'''
result = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1',
     '/home/jetson/miniconda3/envs/leap/bin/python', '-'],
    input=CODE.encode(), capture_output=True, timeout=600,
)
if result.returncode:
    print(result.stderr.decode())
    raise SystemExit(result.returncode)
data = json.loads(result.stdout)
(ART / 'apply-report.json').write_text(json.dumps(data, indent=2))
print(json.dumps({k: v for k, v in data.items() if k != 'catalog'}, indent=2))
