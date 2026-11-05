"""Local-only deployment-helper regression; simulated files, no board/service access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ART = Path(__file__).resolve().parent
OPS = ART / 'source_ops.py'
module = runpy.run_path(str(OPS))
work = ART / 'source-ops-test-v2'
work.mkdir(exist_ok=False)
root, staged = work / 'target', work / 'staged'
for directory in (root / 'src', staged / 'src'):
    directory.mkdir(parents=True)
(root / 'src/existing.py').write_text('old = 1\n')
(root / 'src/existing.py').chmod(0o640)
(staged / 'src/existing.py').write_text('new = 2\n')
(staged / 'src/added.py').write_text('added = 3\n')
plan = {'schema': 'leapflow.private_source_operations.v1', 'target_root': str(root),
    'staged_root': str(staged), 'backup_root': str(work / 'backup'),
    'supervisor': 'leap-lhp-node.service', 'files': [
        {'path': name, 'before_sha256': module['sha'](root / name),
         'before_mode': (root / name).stat().st_mode & 0o777 if (root / name).exists() else None,
         'after_sha256': module['sha'](staged / name)} for name in ('src/existing.py', 'src/added.py')]}
path = work / 'plan.json'
path.write_text(json.dumps(plan, indent=2) + '\n')
plan, digest = module['load'](path)
assert all(r['before_matches'] and r['staged_matches'] for r in module['inspect'](plan)['files'])
refusal = subprocess.run([sys.executable, str(OPS), '--plan', str(path), '--operation', 'backup'],
    capture_output=True, timeout=10)
assert refusal.returncode != 0 and not Path(plan['backup_root']).exists()
assert module['backup'](plan, digest)['source_changed'] is False
original_run = subprocess.run
subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a[0], 0, 'active\n', '')
try:
    try:
        module['supervisor_stopped'](plan)
    except ValueError:
        pass
    else:
        raise AssertionError('active supervisor accepted')
finally:
    subprocess.run = original_run
module['apply'].__globals__['supervisor_stopped'] = lambda plan: None
(staged / 'src/added.py').write_text('drift = 4\n')
try:
    module['apply'](plan, digest)
except ValueError:
    pass
else:
    raise AssertionError('staged drift accepted')
assert not (work / 'backup/apply.started.json').exists()
(staged / 'src/added.py').write_text('added = 3\n')
assert module['apply'](plan, digest)['service_started'] is False
assert (root / 'src/existing.py').stat().st_mode & 0o777 == 0o640
assert module['sha'](root / 'src/added.py') == plan['files'][1]['after_sha256']
(root / 'src/existing.py').write_text('later = 4\n')
try:
    module['restore'](plan, digest)
except ValueError:
    pass
else:
    raise AssertionError('later source change overwritten')
assert (root / 'src/existing.py').read_text() == 'later = 4\n'
(root / 'src/existing.py').write_text('new = 2\n')
assert module['restore'](plan, digest)['files_deleted'] == 0
assert (root / 'src/existing.py').read_text() == 'old = 1\n'
assert not (root / 'src/added.py').exists()
assert (work / 'backup/quarantine/src/added.py').read_text() == 'added = 3\n'
result = {'ok': True, 'schema': 'leapflow.private_source_ops_verification.v1',
    'checks': ['read-only inspection', 'missing approval refuses writes', 'hash-bound backup',
        'active supervisor refuses apply/restore', 'staging drift refuses apply', 'existing file modes preserved',
                'later source drift refuses restore', 'no automatic service action',
        'restore original bytes', 'new files quarantined, never deleted'],
    'simulated_files_only': True, 'real_supervisor_called': False,
    'remote_changes_performed': False, 'physical_motion_performed': False,
    'source_ops_sha256': hashlib.sha256(OPS.read_bytes()).hexdigest()}
(ART / 'source-ops-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
