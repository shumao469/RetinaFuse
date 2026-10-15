"""Verify all training input caches and vendor identity against saved source provenance."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    cohort = json.loads((ROOT / 'audit/cohort_audit.json').read_text())
    assert sha(ROOT / 'vendor/retinafuse_core.py') == cohort['source_sha256']
    assert meta.animal_id.is_unique and meta.source_family.is_unique
    assert meta.video_sha256.is_unique and meta.event_sha256.is_unique
    assert len(meta) == 120 and meta.label.value_counts().to_dict() == {1: 89, 0: 31}
    rows = []
    for record in meta.itertuples(index=False):
        path = ROOT / f'data/{record.animal_id}.npz'
        saved = json.loads((ROOT / f'data/{record.animal_id}.json').read_text())
        assert sha(path) == saved['cache_sha256'], record.animal_id
        assert record.event_sha256 == saved['event_sha256']
        assert record.video_sha256 == saved['video_sha256']
        assert Path(record.event_path).is_file() and Path(record.video_path).is_file()
        rows.append(dict(source_record=record.animal_id, cache_sha256=saved['cache_sha256']))
    result = dict(status='PASS', unique_source_records=len(rows),
                  input_cache_hashes_verified=True, vendor_identity_verified=True,
                  source_paths_exist=True, source_content_rehashed_in_this_check=False,
                  scope='cache content and retained provenance; not independent animal identity',
                  caches=rows)
    target = ROOT / 'audit/input_integrity_verified.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
    os.replace(tmp, target)
    print(json.dumps({k: v for k, v in result.items() if k != 'caches'}))

if __name__ == '__main__':
    main()
