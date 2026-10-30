"""Finalize local approval artifacts from verified source/evidence; no remote operations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import shutil
import tarfile

ROOT = Path('/Users/jason/work/leap_work/leapflow')
ART = ROOT / '.qoder/artifacts/leapcalib/preparation-v3'
PREVIEW = ROOT / '.qoder/artifacts/leapcalib/camera-preview-v3'
RECON = ART / 'target-reconciliation-v2'
OUT = ART / 'approval-packet'
OUT.mkdir(mode=0o700, exist_ok=False)
preview = json.loads((PREVIEW / 'deployment-preview.json').read_text())
report = json.loads((RECON / 'reconciliation.json').read_text())
lineage = json.loads((RECON / 'lineage-inspection.json').read_text())
snapshot = json.loads((RECON / 'snapshot.json').read_text())
prepare = json.loads((ART / 'draft/preparation.json').read_text())
ops_verification = json.loads((ART / 'source-ops-verification.json').read_text())
source = runpy.run_path(str(ROOT / 'tools/calib_preview.py'))
assert source['inventory'](ROOT) == preview['source_sha256']
assert preview['source_inventory_sha256'] == report['source_inventory_sha256']
assert hashlib.sha256((ART / 'source_ops.py').read_bytes()).hexdigest() == ops_verification['source_ops_sha256']
notes = {
    'src/leapflow/hardware/health_monitor.py': 'Only adds typed lifecycle receipt/acknowledgment; existing frame health behavior retained.',
    'src/leapflow/hardware/lhp/server.py': 'Existing transport handlers retained; adds busy/stop-use and control-safety feasibility before permit consumption, owner reservation and lifecycle telemetry.',
    'src/leapflow/hardware/skills/protocol.py': 'Adds explicit motion_allowed/uses_motion separation; persisted effects still require permits.',
    'src/leapflow/hardware/skills/runtime.py': 'Replaces old inline admission/sealing/limits with tested delegates; adds model/session checks, governed reload, non-motion capabilities and shutdown draining.',
    'src/leapflow/hardware/skills/workcell.py': 'Only adds strict typed calibration declaration parsing/serialization; existing fields retained.',
    'src/leapflow/hardware/transports/lhp.py': 'Existing probe/FrameHub metadata retained; adds bounded validated lifecycle events/acknowledgments.',
}
for row in lineage['files']:
    if row['finding'] == 'requires_semantic_inspection':
        assert row['path'] in notes and row['host_hunks_equal_host_file'] is True
        row['resolution_note'] = notes[row['path']]
    else:
        assert row['finding'] in ('target_is_exact_previously_reconciled_task_source', 'formatting_only_target_host_difference')
    row['resolution'] = 'host_candidate_for_separately_approved_upgrade'
assert not report['retained_target_differences']
lineage.update({'state': 'source_decisions_resolved_locally_not_applied', 'unresolved_conflicts': [],
    'remote_changes_performed': False, 'deployment_approved': False})
(OUT / 'source-resolution.json').write_text(json.dumps(lineage, indent=2) + '\n')
release = 'leapcalib-camera-v3-' + preview['wheel']['sha256'][:8]
remote_stage = '/home/jetson/leap_work/' + release + '-stage'
plan = {'schema': 'leapflow.private_source_operations.v1', 'target_root': report['target_root'],
    'staged_root': remote_stage + '/snapshot', 'backup_root': '/home/jetson/leap_work/' + release + '-backup',
    'supervisor': 'leap-lhp-node.service', 'target_head': report['head'],
    'source_inventory_sha256': report['source_inventory_sha256'],
    'files': [{'path': r['path'], 'before_sha256': r['target_sha256'],
        'before_mode': snapshot['files'][r['path']]['mode'], 'after_sha256': r['host_sha256']}
        for r in report['files'] if r['host_sha256'] != r['target_sha256']],
    'scope': 'Only manifest-listed source/tools; no engine, config, database, keys, CAN or firmware writes.',
    'remote_changes_performed': False, 'deployment_approved': False}
plan_path = OUT / 'source-plan.json'
plan_path.write_text(json.dumps(plan, indent=2) + '\n')
plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
runpy.run_path(str(ART / 'source_ops.py'))['load'](plan_path)
shutil.copy2(ART / 'source_ops.py', OUT / 'source_ops.py')
archive = OUT / (release + '.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    for name in ('source-plan.json', 'source_ops.py'):
        tar.add(OUT / name, arcname=name, recursive=False)
    for row in plan['files']:
        path = PREVIEW / 'snapshot' / row['path']
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['after_sha256']
        tar.add(path, arcname='snapshot/' + row['path'], recursive=False)
archive.chmod(0o600)
remote_archive = '/home/jetson/leap_work/' + archive.name
python = '/home/jetson/miniconda3/envs/leap/bin/python'
base = f'{python} {remote_stage}/source_ops.py --plan {remote_stage}/source-plan.json'
command_rows = [
    {'id': 'upload', 'command': f'scp {archive} jetson@192.168.55.1:{remote_archive}',
        'impact': 'New private non-owning staging archive only; no source or service replacement.'},
    {'id': 'stage', 'command': f'umask 077 && mkdir {remote_stage} && tar -xzf {remote_archive} -C {remote_stage}',
        'impact': 'Creates private source staging only; never launches an owner.'},
    {'id': 'backup', 'command': base + f' --operation backup --execute --approved-plan-sha256 {plan_sha} --approval-note camera-v3-backup',
        'impact': 'Creates exact original-file backup; target source unchanged.'},
    {'id': 'stop', 'command': 'systemctl --user stop leap-lhp-node.service',
        'impact': 'Disconnects the sole hardware owner; requires fresh idle/no-hold/no-dispatch confirmation first.'},
    {'id': 'apply', 'command': base + f' --operation apply --execute --approved-plan-sha256 {plan_sha} --approval-note camera-v3-source-apply',
        'impact': 'Fingerprint-bound manifest source replacement, only while supervisor is inactive; no configuration/motion.'},
    {'id': 'metadata', 'command': f'{python} -m pip install --no-index --no-deps --no-build-isolation -e {report["target_root"]}',
        'impact': 'Refreshes editable package/entry-point metadata from preserved canonical source, no dependency download.'},
    {'id': 'start', 'command': 'systemctl --user start leap-lhp-node.service',
        'impact': 'Starts the sole existing supervisor before any stationary anchor; new owner session, no implicit enable/motion.'},
]
rollback = {'command': base + f' --operation restore --execute --approved-plan-sha256 {plan_sha} --approval-note camera-v3-source-restore',
    'prerequisites': ['Separate exact rollback approval', 'Sole supervisor explicitly stopped', 'No later source drift'],
    'impact': 'Restore original files/modes, quarantine new source files without deletion; then separately approve metadata refresh and supervisor start. Does not restore physical posture.'}
handoff = {'schema': 'leapflow.private_camera_handoff.v1', 'state': 'ready_for_minimal_inputs_and_explicit_approvals',
    'executing_plan': '/Users/jason/Library/Application Support/Qoder/SharedClientCache/cache/plans/LeapCalib_Governed_Alignment_Activation_cd98b49c.md',
    'scope': 'Camera-only stationary acquisition, separately approved bounded raw-joint collection, one shadow extrinsic and disjoint withheld evaluation; nominal geometry unchanged.',
    'preview': str(PREVIEW / 'deployment-preview.json'), 'wheel': preview['wheel']['file'],
    'wheel_sha256': preview['wheel']['sha256'], 'verification': preview['verification'],
    'source_decisions_resolved': len(lineage['files']), 'manifest_source_files': len(plan['files']),
    'target_dirty_entries': report['target_dirty_entries'], 'source_plan_sha256': plan_sha,
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'source_ops_sha256': ops_verification['source_ops_sha256'],
    'source_ops_verification': str(ART / 'source-ops-verification.json'),
    'target_snapshot_is_point_in_time': True, 'commands_not_executed': command_rows,
    'read_only_after_staging': base, 'rollback_separately_approved': rollback,
    'preconditions': ['Staging/archive/backup destinations must not exist; no overwrite/reuse.',
        'Recheck all target hashes and HEAD immediately before approved staging/backup/apply.',
        'SHA-256-check staged archive/helper/plan before any approved helper operation.',
        'Fresh existing-owner idle/no active lease/run/dispatch evidence before service stop.',
        'Inspect exact installed source, entry points, sole supervisor, hello/list and worker readiness after start.',
        'Existing stale deployment scripts are excluded and must not be executed.'],
    'human_inputs': prepare['human_confirmation'], 'not_human_work': prepare['not_human_work'],
    'draft_readiness_blockers': prepare['readiness_blockers'], 'unresolved_real_evidence': prepare['unresolved'],
    'config_targets': [
        {'path': '/home/jetson/.leapflow/profiles/default/hardware/workcells/piper_cell.yaml',
         'stage_a': 'Merge calibration_acquisition only; no calibration_motion before original indexed anchor.',
         'stage_b': 'Merge genuine anchor/mount/geometry/rates into calibration_motion, then approve skill.reload without restart.'},
        {'path': '/home/jetson/.leapflow/profiles/default/hardware/devices/realsense_d405.yaml', 'field': 'transport.config.mount_digest'},
        {'path': '/home/jetson/.leapflow/profiles/default/config/runtime.yaml', 'field': 'hardware.calibration_worker_enabled'}],
    'future_approval_gates': ['Exact nonsecret config diffs after measured inputs; never apply current draft thresholds as evidence.',
        'Any config-owner restart before anchor', 'Enable/reference positioning/speed setup',
        'Stationary indexed collection', 'Exact Stage B write and same-runtime skill.reload',
        'Each fresh bounded pose-run preview', 'Leased READ shadow trial before raw expiry'],
    'approval_strings_are_not_authority': True, 'deployment_approved': False,
    'remote_changes_performed_by_this_continuation': False, 'physical_motion_performed': False,
    'physical_qualification': False, 'activation': 'none', 'auto_return': False}
(OUT / 'handoff.json').write_text(json.dumps(handoff, indent=2) + '\n')
print(json.dumps({k: handoff[k] for k in ('state', 'source_decisions_resolved', 'manifest_source_files', 'target_dirty_entries', 'source_plan_sha256', 'archive_sha256')}, indent=2))
