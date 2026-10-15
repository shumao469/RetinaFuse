"""Screen cached model-input collisions; not proof of native or animal identity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()

def main():
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    rows = []
    for record in meta.itertuples(index=False):
        with np.load(ROOT / 'data' / f'{record.animal_id}.npz', allow_pickle=False) as cache:
            frames = cache['raw_original_frames']
            keep = np.r_[True, np.any(frames[1:] != frames[:-1], axis=tuple(range(1, frames.ndim)))]
            rows.append(dict(animal_id=record.animal_id, label=int(record.label),
                decoded_low_resolution=digest(frames),
                deduplicated_low_resolution=digest(frames[keep]),
                legacy_model_inputs=digest(cache['legacy_events'], cache['legacy_stats'])))
    data = pd.DataFrame(rows)
    collisions = {}
    for key in ['decoded_low_resolution', 'deduplicated_low_resolution', 'legacy_model_inputs']:
        groups = []
        for fingerprint, group in data.groupby(key):
            if len(group) > 1:
                groups.append(dict(sha256=fingerprint, animal_ids=group.animal_id.tolist(), labels=group.label.tolist()))
        collisions[key] = groups
    data.to_csv(ROOT / 'audit/model_input_fingerprints.csv', index=False)
    result = dict(status='COLLISIONS_REQUIRE_REVIEW' if any(collisions.values()) else 'NO_EXACT_CACHED_INPUT_COLLISIONS',
        records=len(rows), collisions=collisions,
        scope='Exact cached low-resolution inputs only; no tolerance-based image similarity. This does not prove native-frame uniqueness or independent animal identity. No cohort/split changes made.')
    (ROOT / 'audit/model_input_fingerprint_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
