"""Verify private node evidence and isolated final-wheel boundaries without exporting samples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path('/Users/jason/work/leap_work/leapflow')
ART = ROOT / '.qoder/artifacts/leapcalib/integration-20261027-212408'
os.umask(0o077)
WHEEL = ROOT / '.qoder/artifacts/leapcalib/final-package/leapflow-0.7.0+main-py3-none-any.whl'
with zipfile.ZipFile(WHEEL) as archive:
    assert archive.read('leapflow/hardware/lhp/runtime.py') == (ROOT / 'src/leapflow/hardware/lhp/runtime.py').read_bytes()
    for path in (ROOT / 'src/leapcalib').rglob('*.py'):
        assert archive.read(str(path.relative_to(ROOT / 'src'))) == path.read_bytes()
        compile(path.read_bytes(), str(path), 'exec')
ISOLATED = r'''
import importlib.metadata, json, sys
sys.path.insert(0, sys.argv[1])
import leapcalib
import leapcalib.cli
import leapcalib.contracts
from leapcalib.contracts.methods import discover_methods
methods = discover_methods()
assert len(methods) == 2
assert 'numpy' not in sys.modules
assert not any(name == 'leapflow' or name.startswith('leapflow.') for name in sys.modules)
dist = importlib.metadata.distribution('leapflow')
assert 'leapcalib' in dist.metadata.get_all('Provides-Extra')
assert any(ep.group == 'console_scripts' and ep.name == 'leap-calib' for ep in dist.entry_points)
assert any(ep.group == 'leapflow.hardware.skills' and ep.name == 'builtin_metric_readiness' for ep in dist.entry_points)
print(json.dumps({'methods': sorted(methods), 'numpy_loaded': False, 'leapflow_loaded': False, 'final_startup_binding_packaged': True}))
'''
built = subprocess.run([sys.executable, '-I', '-S', '-c', ISOLATED, str(WHEEL)], capture_output=True, text=True, timeout=30)
if built.returncode:
    print(built.stderr)
    raise SystemExit(built.returncode)
package = json.loads(built.stdout)
package.update({'wheel_sha256': hashlib.sha256(WHEEL.read_bytes()).hexdigest(), 'wheel_bytes': WHEEL.stat().st_size})
CODE = r'''
import asyncio, hashlib, importlib.metadata, json, os, pathlib, stat, subprocess, time
import duckdb
from leapflow.config import load_config
from leapflow.hardware.lhp.client import SocketLhpClient
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.methods import discover_methods
os.umask(0o077)
backup = pathlib.Path('/home/jetson/leap_work/leapcalib-integration-20261027-212408')
root = pathlib.Path('/home/jetson/leap_work/leapflow')
config = load_config()
layout = config.profile_layout.cache
assert layout.index_path.is_file()
reports = []
with duckdb.connect(str(layout.index_path), read_only=True) as con:
    for name in ('d435if', 'd405'):
        output = backup / (name + '-readiness.json')
        payload = json.loads(output.read_text())
        assert payload['digest'] == digest_of(payload['content'])
        data = payload['content']
        outcome = data['result']['outcome']
        assert outcome['status'] == 'succeeded' and outcome['side_effect_state'] == 'none'
        summary = outcome['result']
        assert summary['motion_support'] is False and summary['hardware_accuracy_certified'] is False
        assert summary['raw_exported'] is False and summary['syncable'] is False
        coexist = data['preview_coexistence']
        assert coexist['verified'] is True and coexist['overlap_samples'] > 0
        frames = summary['frames']
        assert len(frames) == 8 and all(f['aligned_to_color'] for f in frames)
        seq = [f['frame_sequence'] for f in frames]
        assert all(b > a for a,b in zip(seq, seq[1:]))
        assert all(f['measured_samples'] > 0 and f['valid_ratio'] > 0 for f in frames)
        path = layout.category_dir(scope='profile', category='calibration_evidence') / 'metric_readiness' / summary['local_artifact']
        private = json.loads(path.read_text())
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert private['digest'] == digest_of(private['content']) == summary['evidence_digest']
        assert len(private['content']['observations']) == 8
        assert private['content']['retention'] == {'ttl_s': 3600.0, 'syncable': False, 'sensitivity': 'raw_evidence'}
        row = con.execute('SELECT sensitive, syncable, expires_at, content_hash, size_bytes, owner_component FROM cache_entry WHERE path = ?', [str(path)]).fetchone()
        assert row is not None and row[0] is True and row[1] is False
        assert time.time() < row[2] <= time.time() + 3600
        assert row[3] == private['digest'] and row[4] == path.stat().st_size and row[5] == 'hardware.calibration'
        reports.append({'camera': name, 'frame_count': len(frames), 'frame_sequences': seq, 'valid_ratio_range': [min(f['valid_ratio'] for f in frames), max(f['valid_ratio'] for f in frames)], 'depth_scale_m': frames[0]['depth_scale_m'], 'intrinsics': frames[0]['intrinsics'], 'depth_median_range_m': [min(f['depth_median_m'] for f in frames), max(f['depth_median_m'] for f in frames)], 'receipt_age_range_s': [min(f['receipt_age_s'] for f in frames), max(f['receipt_age_s'] for f in frames)], 'receipt_intervals_s': summary['receipt_intervals_s'], 'compute_s': summary['compute_s'], 'elapsed_s': summary['elapsed_s'], 'process_cpu_s': summary['process_cpu_s'], 'resource_usage': summary['resource_usage'], 'preview_overlap_samples': coexist['overlap_samples'], 'evidence_digest': private['digest'], 'result_digest': payload['digest'], 'artifact_bytes': path.stat().st_size, 'artifact_mode': '0600', 'indexed_sensitive': True, 'indexed_syncable': False, 'indexed_expires_at': row[2], 'motion_support': False, 'hardware_accuracy_certified': False, 'raw_exported': False})
manifest = json.loads((backup / 'deployment-manifest.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['merged_sha256']
async def check():
    token = pathlib.Path('/home/jetson/.leapflow/.lhp-token').read_text().strip()
    client = SocketLhpClient('tcp://127.0.0.1:8767', token=token)
    try:
        hello = await client.connect()
        assert hello['node_id'] == 'yahboom' and hello['write_mode'] == 'permit_required'
        runs = await client.request('skill.status', {})
        sessions = await client.request('control.status', {})
        assert not sessions['sessions']
        assert not any(r.get('status') == 'running' or r.get('outcome') is None for r in runs['runs'])
    finally:
        await client.close()
asyncio.run(check())
assert not any('test-site' in part or '/stage' in part or '/site' in part for part in os.environ.get('PYTHONPATH','').split(':') if part)
methods = discover_methods()
assert len(methods) == 2
report = {'node_id': 'yahboom', 'write_mode': 'permit_required', 'cameras': reports, 'live_method_discovery': sorted(methods), 'deployed_source_fingerprints_verified': True, 'active_runs': 0, 'active_control_sessions': 0, 'service': subprocess.check_output(['systemctl', '--user', 'show', 'leap-lhp-node.service', '-p', 'MainPID', '-p', 'ActiveState'], text=True).strip().splitlines(), 'raw_camera_observations_exported': False}
(backup / 'hardware-acceptance.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
'''
remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1', '/home/jetson/miniconda3/envs/leap/bin/python', '-'], input=CODE.encode(), capture_output=True, timeout=90)
if remote.returncode:
    print(remote.stderr.decode())
    raise SystemExit(remote.returncode)
hardware = json.loads(remote.stdout)
(ART / 'hardware-acceptance.json').write_text(json.dumps(hardware, indent=2))
(ART / 'final-package-check.json').write_text(json.dumps(package, indent=2))
print(json.dumps({'package': package, 'hardware': hardware}, indent=2))
