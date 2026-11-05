"""Execute the user's exact upload+private-staging approval; never deploy or restart.

Single attempt with host audit. Failed/uncertain effects are recorded and never
retried, cleaned up or rolled back automatically. All later approvals stay closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import shlex
import subprocess
import tarfile
import time

ROOT = Path('/Users/jason/work/leap_work/leapflow')
OUTPUT = Path(__file__).resolve().parent
PACKET = ROOT / '.qoder/artifacts/leapcalib/preparation-v3/approval-packet'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jetson@192.168.55.1']
PYTHON = '/home/jetson/miniconda3/envs/leap/bin/python'


def remote(code: str, data: dict) -> dict:
    result = subprocess.run(SSH + [PYTHON, '-B', '-c', shlex.quote(code)],
        input=json.dumps(data), capture_output=True, text=True, timeout=90, check=True)
    if len(result.stdout) > 4 * 1024 * 1024:
        raise ValueError('bounded read-only response required')
    return json.loads(result.stdout)


VERIFY = r'''
import hashlib, json, pathlib, sys
p = json.load(sys.stdin)
root = pathlib.Path(p['stage'])
def sha(path):
    assert not path.is_symlink() and path.is_file() and path.stat().st_size <= 4194304
    return hashlib.sha256(path.read_bytes()).hexdigest()
rows = []
for name, expected in p['members'].items():
    relative = pathlib.PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = root / relative
    assert path.resolve().is_relative_to(root.resolve())
    actual = sha(path)
    rows.append({'path': name, 'sha256': actual, 'matches': actual == expected})
archive = pathlib.Path(p['archive'])
assert not archive.is_symlink() and archive.is_file() and archive.stat().st_size <= 67108864
archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
print(json.dumps({'archive_sha256': archive_hash, 'members': rows,
    'staged_members_match': all(row['matches'] for row in rows), 'read_only': True}))
'''


def main() -> None:
    handoff = json.loads((PACKET / 'handoff.json').read_text())
    plan_raw = (PACKET / 'source-plan.json').read_bytes()
    plan = json.loads(plan_raw)
    archive = PACKET / 'leapcalib-camera-v3-35bca0b9.tar.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != handoff['archive_sha256']:
        raise ValueError('approved archive fingerprint changed')
    if hashlib.sha256(plan_raw).hexdigest() != handoff['source_plan_sha256']:
        raise ValueError('approved plan fingerprint changed')
    members = {}
    expected = {'source-plan.json': handoff['source_plan_sha256'], 'source_ops.py': handoff['source_ops_sha256']}
    expected.update({'snapshot/' + row['path']: row['after_sha256'] for row in plan['files']})
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or member.size > 4 * 1024 * 1024 or name.is_absolute() or '..' in name.parts or member.name in members:
                raise ValueError('unsafe/duplicate archive member')
            data = stream.extractfile(member).read()
            members[member.name] = hashlib.sha256(data).hexdigest()
    if members != expected:
        raise ValueError('archive contains unapproved or changed files')
    tool = runpy.run_path(str(OUTPUT / 'inspect_followup.py'))
    stage = str(Path(plan['staged_root']).parent)
    remote_archive = '/home/jetson/leap_work/' + archive.name
    request = {key: plan[key] for key in ('target_root', 'files', 'supervisor')}
    request['destinations'] = [stage, remote_archive, plan['backup_root']]
    before = remote(tool['REMOTE'], request)
    if any(before['destinations_exist'].values()) or any(not row['matches_before'] for row in before['files']):
        raise ValueError('fresh absence/target fingerprint precondition failed; no upload')
    if before['head']['returncode'] != 0 or before['head']['stdout'].strip() != plan['target_head']:
        raise ValueError('target HEAD changed; no upload')
    commands = {row['id']: row for row in handoff['commands_not_executed']}
    approval = {'question': 'Staging gate', 'user_selection': 'Approve upload + staging',
        'authorized_ids': ['upload', 'stage'], 'other_command_ids_approved': [],
        'archive_sha256': handoff['archive_sha256']}
    with (OUTPUT / 'staging-audit.jsonl').open('x') as audit:
        def event(data: dict) -> None:
            audit.write(json.dumps({'at_s': time.time(), **data}, ensure_ascii=False) + '\n')
            audit.flush()
        event({'approval': approval, 'preconditions_verified': True})
        results = []
        try:
            for identifier in ('upload', 'stage'):
                command = commands[identifier]['command']
                args = shlex.split(command) if identifier == 'upload' else SSH + [command]
                event({'command_id': identifier, 'command': command, 'state': 'started'})
                result = subprocess.run(args, capture_output=True, text=True, timeout=180, check=False)
                row = {'id': identifier, 'command': command, 'impact': commands[identifier]['impact'],
                    'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
                results.append(row)
                event(row)
                if result.returncode != 0:
                    raise RuntimeError('approved staging command failed; effects require inspection, no retry')
                if identifier == 'upload':
                    archive_check = remote("import hashlib,json,pathlib,sys;p=json.load(sys.stdin);f=pathlib.Path(p['path']);assert not f.is_symlink() and f.is_file();print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}))", {'path': remote_archive})
                    if archive_check['sha256'] != handoff['archive_sha256']:
                        raise ValueError('uploaded archive differs; no extraction')
            staged = remote(VERIFY, {'stage': stage, 'archive': remote_archive, 'members': expected})
            after = remote(tool['REMOTE'], request)
            if not staged['staged_members_match'] or staged['archive_sha256'] != handoff['archive_sha256']:
                raise ValueError('staged fingerprint verification failed; no further action')
            if any(not row['matches_before'] for row in after['files']):
                raise ValueError('canonical source changed concurrently; no further action')
            summary = {'schema': 'leapflow.private_approved_staging.v1', 'state': 'staged_verified_not_deployed',
                'approval': approval, 'commands_executed': results, 'before': before, 'after': after,
                'verification': staged, 'staged_files': len(expected), 'remote_staging_written': True,
                'canonical_source_replaced': False, 'configuration_written': False,
                'backup_created': False, 'metadata_installed': False, 'service_transition_performed': False,
                'physical_motion_performed': False, 'physical_qualification': False, 'activation': 'none'}
            with (OUTPUT / 'staging-result.json').open('x') as output:
                output.write(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
            event({'state': summary['state'], 'staged_files': len(expected)})
            print(json.dumps({key: summary[key] for key in ('state', 'staged_files', 'canonical_source_replaced', 'service_transition_performed', 'physical_motion_performed')}))
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            event({'state': 'staging_failed_or_uncertain_no_retry', 'error_type': type(exc).__name__, 'detail': str(exc)})
            raise


if __name__ == '__main__':
    main()
