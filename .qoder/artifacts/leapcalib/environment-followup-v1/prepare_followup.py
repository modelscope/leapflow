"""Bind supplied images to local unresolved drafts, never to calibration factors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

from PIL import Image

ROOT = Path('/Users/jason/work/leap_work/leapflow')
OUTPUT = Path(__file__).resolve().parent


def describe(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('bounded original regular image required')
    with Image.open(path) as image:
        if image.format != 'JPEG' or max(image.size) > 10000:
            raise ValueError('bounded JPEG dimensions required')
        image.verify()
        dimensions = list(image.size)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size, 'dimensions_px': dimensions,
            'is_metric_measurement': False, 'is_calibration_factor_source': False}


def write(name: str, data: dict) -> None:
    with (OUTPUT / name).open('x') as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    tool = runpy.run_path(str(ROOT / 'tools/calib_prepare.py'))
    review_path = OUTPUT / 'image-observations.json'
    review = json.loads(review_path.read_text())
    images = [ROOT / 'temp/devices/env_images' / row['file'] for row in review['supplied_images']]
    if len(images) != 4 or len(set(images)) != 4:
        raise ValueError('exact reviewed image set required')
    manifest = {'schema': 'leapflow.private_environment_evidence_manifest.v1',
        'images': [describe(path) for path in images],
        'review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
        'diagnostics': [describe(OUTPUT / row) for row in ('wrist-diagnostic.jpg', 'global-diagnostic.jpg')],
        'remote_changes_performed': False, 'physical_qualification': False}
    write('image-manifest.json', manifest)
    inputs = {key: None for key in ('mount_confirmation', 'uncertainty_evidence',
        'whole_link_geometry_evidence', 'rate_bound_evidence', 'wrist_prior_evidence')}
    inputs['supporting_image_manifest'] = 'image-manifest.json'
    write('inputs.json', inputs)
    probe_path = OUTPUT / 'probe.json'
    result = tool['prepare'](tool['load_json'](probe_path), inputs)
    result['supporting_image_evidence'] = manifest
    result['current_onsite_followup'] = review['minimal_operator_actions']
    result['diagnostic_region_verified_for_tracking'] = False
    assert result['stage_b_overlay'] is None and not result['pose_batches']
    assert result['physical_qualification'] is False and result['motion_performed'] is False
    assert result['readiness_blockers'] and result['trial_stamp_bound_s'] > result['trial_stamp_gate_s']
    tool['write_preparation'](OUTPUT / 'draft', result, [probe_path, OUTPUT / 'inputs.json',
        review_path, OUTPUT / 'image-manifest.json'] + images)
    write('preparation-checks.json', {'ok': True, 'physical_qualification': False,
        'photos_promoted_to_measurements': False, 'stage_b_overlay_created': False,
        'pose_batches_created': False, 'timing_gate_relaxed': False,
        'remote_changes_performed': False, 'calibration_factor_sources': [],
        'unresolved': result['unresolved'], 'readiness_blockers': result['readiness_blockers']})
    print(json.dumps({'output': str(OUTPUT / 'draft'), 'state': result['state'],
        'unresolved': result['unresolved'], 'physical_motion_performed': False}))
