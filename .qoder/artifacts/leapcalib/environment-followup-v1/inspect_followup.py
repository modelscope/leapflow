"""Private bounded read-only followup; host-only evidence, never calibration factors.

Uses the existing authenticated owner for two diagnostic preview images and
status only. Remote inspection reads manifest hashes and selected nonsecret
configuration fields. No upload, package, service, device or configuration write.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import time

from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.protocol import LhpRemoteError

ROOT = Path('/Users/jason/work/leap_work/leapflow')
OUTPUT = Path(__file__).resolve().parent
PACKET = ROOT / '.qoder/artifacts/leapcalib/preparation-v3/approval-packet'
os.umask(0o077)


def save(name: str, value: dict) -> None:
    with (OUTPUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


REMOTE = r'''
import hashlib, json, pathlib, subprocess, sys, yaml
p = json.load(sys.stdin)
root = pathlib.Path(p['target_root']).resolve(strict=True)
def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
rows = []
for row in p['files']:
    name = pathlib.PurePosixPath(row['path'])
    assert not name.is_absolute() and '..' not in name.parts
    assert str(name).startswith(('src/', 'tools/')) or str(name) in ('pyproject.toml', 'uv.lock')
    path = root / name
    assert not path.is_symlink() and path.resolve().is_relative_to(root)
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    rows.append({'path': str(name), 'sha256': actual,
        'matches_before': actual == row['before_sha256'], 'matches_after': actual == row['after_sha256']})
profile = pathlib.Path('/home/jetson/.leapflow/profiles/default')
def config(name):
    path = profile / name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 1048576
    data = path.read_bytes()
    value = yaml.safe_load(data)
    assert isinstance(value, dict)
    return value, hashlib.sha256(data).hexdigest()
cell, cell_hash = config('hardware/workcells/piper_cell.yaml')
cam, cam_hash = config('hardware/devices/realsense_d405.yaml')
runtime, runtime_hash = config('config/runtime.yaml')
transport = cam.get('transport', {})
hardware = runtime.get('hardware', {})
selected = {'workcell_sha256': cell_hash, 'camera_sha256': cam_hash, 'runtime_sha256': runtime_hash,
    'calibration_acquisition_present': 'calibration_acquisition' in cell,
    'calibration_motion_present': 'calibration_motion' in cell,
    'mount_digest': transport.get('config', {}).get('mount_digest'),
    'calibration_worker_enabled': hardware.get('calibration_worker_enabled',
        runtime.get('hardware_calibration_worker_enabled'))}
properties = ['ActiveState', 'SubState', 'MainPID', 'ExecMainStartTimestamp', 'FragmentPath']
service = command(['systemctl', '--user', 'show', p['supervisor']] + ['--property=' + v for v in properties])
head = command(['git', '--no-optional-locks', '-C', str(root), 'rev-parse', 'HEAD'])
destinations = {v: pathlib.Path(v).exists() for v in p['destinations']}
print(json.dumps({'observed_at_s': __import__('time').time(), 'head': head,
    'files': rows, 'service': service, 'selected_config': selected, 'destinations_exist': destinations,
    'remote_changes_performed': False}))
'''


def inspect_files() -> None:
    plan = json.loads((PACKET / 'source-plan.json').read_text())
    request = {key: plan[key] for key in ('target_root', 'files', 'supervisor')}
    request['destinations'] = [plan['backup_root'], str(Path(plan['staged_root']).parent),
                              '/home/jetson/leap_work/leapcalib-camera-v3-35bca0b9.tar.gz']
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'jetson@192.168.55.1', '/home/jetson/miniconda3/envs/leap/bin/python', '-B',
        '-c', shlex.quote(REMOTE)], input=json.dumps(request), capture_output=True,
        text=True, timeout=90, check=True)
    if len(result.stdout) > 4 * 1024 * 1024:
        raise ValueError('remote inspection exceeded bound')
    observed = json.loads(result.stdout)
    preview_tool = runpy.run_path(str(ROOT / 'tools/calib_preview.py'))
    inventory = preview_tool['inventory'](ROOT)
    host_hash = preview_tool['fingerprint'](json.dumps(inventory, sort_keys=True).encode())
    observed['host_inventory_sha256'] = host_hash
    observed['host_matches_packet_inventory'] = host_hash == plan['source_inventory_sha256']
    observed['target_matches_packet_before'] = all(row['matches_before'] for row in observed['files'])
    observed['target_head_matches_packet'] = observed['head']['stdout'].strip() == plan['target_head']
    observed['archive_sha256'] = hashlib.sha256((PACKET / 'leapcalib-camera-v3-35bca0b9.tar.gz').read_bytes()).hexdigest()
    observed['source_ops_sha256'] = hashlib.sha256((PACKET / 'source_ops.py').read_bytes()).hexdigest()
    observed['source_plan_sha256'] = hashlib.sha256((PACKET / 'source-plan.json').read_bytes()).hexdigest()
    observed['point_in_time_only'] = True
    observed['deployment_approved'] = False
    save('deployment-recheck.json', observed)


async def inspect_owner() -> None:
    client = SocketLhpClient('tcp://192.168.55.1:8767',
        token=(ROOT / 'temp/devices/jetson/.lhp-token').read_text().strip(), timeout_s=15)
    report = {'observed_at_s': time.time(), 'previews_are_diagnostic_only': True,
              'calibration_factor_sources': [], 'remote_changes_performed': False, 'frames': []}
    try:
        hello = await client.connect()
        if hello.get('node_id') != 'yahboom' or hello.get('write_mode') != 'permit_required':
            raise ValueError('exact governed owner required')
        report['node_id'] = hello['node_id']
        report['write_mode'] = hello['write_mode']
        for method in ('control.status', 'skill.status'):
            try:
                report[method] = await client.request(method, {})
            except LhpRemoteError as exc:
                report[method] = {'error_code': exc.code, 'detail': str(exc)}
        for label, device in [('wrist', 'sensor_realsense_d405'), ('global', 'sensor_realsense_435if')]:
            try:
                frame = await client.request('read_frame', {'device_id': device,
                    'channel_id': 'camera.color.preview', 'max_width': 640, 'quality': 85, 'fps': 0.0})
                if frame.get('device_id') != device or frame.get('channel_id') != 'camera.color.preview':
                    raise ValueError('preview identity mismatch')
                data = base64.b64decode(frame['data_b64'], validate=True)
                if not data.startswith(b'\xff\xd8\xff') or not 0 < len(data) <= 2 * 1024 * 1024:
                    raise ValueError('bounded JPEG required')
                name = f'{label}-diagnostic.jpg'
                with (OUTPUT / name).open('xb') as stream:
                    stream.write(data)
                report['frames'].append({**{k: v for k, v in frame.items() if k != 'data_b64'},
                    'file': name, 'sha256': hashlib.sha256(data).hexdigest(), 'not_an_indexed_acquisition': True})
            except LhpRemoteError as exc:
                report['frames'].append({'device_id': device, 'error_code': exc.code, 'detail': str(exc)})
    finally:
        await client.close()
    save('owner-inspection.json', report)


if __name__ == '__main__':
    inspect_files()
    asyncio.run(asyncio.wait_for(inspect_owner(), timeout=90))
    print(json.dumps({'output': str(OUTPUT), 'read_only': True, 'physical_motion_performed': False}))
