"""Audit available native-decoder caches without loading torch or changing training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main(partial=False):
    cfg = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv').set_index('animal_id')
    segmentation = sha(ROOT / 'vendor/seg.py')
    rows = []
    for path in sorted((ROOT / 'data/decoder').glob('*.json')):
        if path.name == 'prepared.json':
            continue
        aid = path.stem; record = meta.loc[aid]
        info = json.loads(path.read_text())
        cache = path.with_suffix('.npz')
        assert sha(cache) == info['cache_sha256'], aid
        assert info['video_sha256'] == record.video_sha256, aid
        assert info['seg_sha256'] == segmentation, aid
        assert [c['condition'] for c in info['conditions']] == cfg['decoder_conditions']
        with np.load(cache, allow_pickle=False) as z:
            for c in info['conditions']:
                name = c['condition']; frames = z[name + '_events']; stats = z[name + '_stats']
                assert frames.shape == (60, 2, 64, 64) and stats.shape == (12,)
                assert np.isfinite(frames).all() and np.isfinite(stats).all()
                assert (frames >= 0).all() and np.equal(frames, np.floor(frames)).all()
                counts = frames.sum(axis=(0, 2, 3), dtype=np.float64)
                assert int(counts.sum()) == c['events'], (aid, name, counts, c['events'])
                np.testing.assert_array_equal(stats[:3],
                    np.array([c['events'], counts[0], counts[1]], dtype=np.float32))
                assert c['cap'] is None and bool(c['empty']) == (c['events'] == 0)
                rows.append(dict(animal_id=aid, condition=name, events=c['events'],
                                 selected_frames=c['selected_frames'], mask_pixels=c['mask_pixels'],
                                 requested_sampling_fps=c['source_fps'], container_fps=4.,
                                 native_resolution=c['source_resolution'], cache_verified=True))
            # Removing exact adjacent duplicate frames with original timestamps must be invariant.
            for representation in ['events', 'stats']:
                np.testing.assert_array_equal(z['graychange_4fps_t002_' + representation],
                                              z['graychange_dedup_t002_' + representation])
            vessel_areas = [c['mask_pixels'] for c in info['conditions']
                            if 'complement_mask' not in c['condition']]
            assert len(set(vessel_areas)) == 1
    records = len({r['animal_id'] for r in rows})
    if not partial:
        assert records == len(meta)
        prepared = json.loads((ROOT / 'data/decoder/prepared.json').read_text())
        assert prepared['n'] == len(meta) and prepared['protocol_sha256'] == sha(ROOT / 'protocol.json')
    target = ROOT / 'audit/native_decoder_cache_audit.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(status='PASS', scope='available completed caches' if partial else 'complete native-decoder cohort',
        records_audited=records, records_expected=len(meta), condition_rows=len(rows),
        exact_duplicate_removal_invariant=True, event_count_conservation=True,
        cap=None, segmentation_sha256=segmentation,
        note='Container fps and requested conversion sampling are different fields; complement support is not known nonvascular tissue.',
        rows=rows), indent=2), encoding='utf-8')
    os.replace(temp, target)
    print(json.dumps(dict(status='PASS', records_audited=records, records_expected=len(meta), condition_rows=len(rows), partial=partial)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
