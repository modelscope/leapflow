"""Local source-lineage analysis only; never connects or changes production files."""
from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/jason/work/leap_work/leapflow')
ART = ROOT / '.qoder/artifacts/leapcalib/preparation-v3/target-reconciliation-v2'
PRIOR = ROOT / '.qoder/artifacts/leapcalib/integration-20261027-212408'
report = json.loads((ART / 'reconciliation.json').read_text())
snapshot = json.loads((ART / 'snapshot.json').read_text())
previous = {r['path']: r for r in json.loads((PRIOR / 'deployment-manifest.json').read_text())['files']}
rows, diffs = [], []
for name in report['conflicts']:
    raw = base64.b64decode(snapshot['files'][name]['current'])
    host = (ROOT / name).read_bytes()
    row = {'path': name, 'target_sha256': hashlib.sha256(raw).hexdigest(),
        'host_sha256': hashlib.sha256(host).hexdigest()}
    old = previous.get(name)
    prior = PRIOR / 'merged' / name
    if old and prior.is_file() and hashlib.sha256(prior.read_bytes()).hexdigest() == old['merged_sha256'] == row['target_sha256']:
        row['finding'] = 'target_is_exact_previously_reconciled_task_source'
        row['previous_manifest'] = str(PRIOR / 'deployment-manifest.json')
    elif name.endswith('.py') and ast.dump(ast.parse(raw)) == ast.dump(ast.parse(host)):
        row['finding'] = 'formatting_only_target_host_difference'
    else:
        row['finding'] = 'requires_semantic_inspection'
        diffs.append('\nFILE: ' + name + '\n' + ''.join(difflib.unified_diff(
            raw.decode().splitlines(keepends=True), host.decode().splitlines(keepends=True),
            fromfile='target/' + name, tofile='host/' + name)))
    conflict = ART / 'merge-inputs' / name / 'conflict'
    if conflict.is_file():
        lines, resolved, choosing_host = conflict.read_bytes().splitlines(keepends=True), [], None
        for line in lines:
            if line.startswith(b'<<<<<<< '):
                choosing_host = False
            elif line.startswith(b'=======') and choosing_host is False:
                choosing_host = True
            elif line.startswith(b'>>>>>>> ') and choosing_host is True:
                choosing_host = None
            elif choosing_host is not False:
                resolved.append(line)
        candidate = b''.join(resolved)
        row['host_hunks_equal_host_file'] = candidate == host
    rows.append(row)
result = {'schema': 'leapflow.private_source_lineage_inspection.v1', 'files': rows,
    'requires_semantic_inspection': [r['path'] for r in rows if r['finding'] == 'requires_semantic_inspection'],
    'remote_changes_performed': False, 'deployment_approved': False}
(ART / 'lineage-inspection.json').write_text(json.dumps(result, indent=2) + '\n')
(ART / 'semantic-diffs.txt').write_text(''.join(diffs))
print(json.dumps(result, indent=2))
