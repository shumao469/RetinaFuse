"""Check whether migrated locked-test files are genuinely new source recordings."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = Path('H:/2026/深思考/实验2')

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    with (ROOT / 'audit/all_128_record_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        retained = list(csv.DictReader(f))
    lookup = {}
    for row in retained:
        lookup.setdefault(row['event_sha256'], []).append(row)
    with (CANDIDATE / 'locked_test_labels.csv').open(encoding='utf-8-sig', newline='') as f:
        labels = {r['sample_id']: int(r['label']) for r in csv.DictReader(f)}
    rows = []
    for path in sorted((CANDIDATE / 'locked_test_event_csv').glob('*.csv')):
        digest = sha(path)
        matches = lookup.get(digest, [])
        rows.append(dict(sample_id=path.stem, candidate_sha256=digest,
                         supplied_label=labels.get(path.stem),
                         matched_existing_source_ids=[r['animal_id'] for r in matches],
                         labels_agree=all(int(r['label']) == labels[path.stem] for r in matches),
                         overlaps_existing_cohort=bool(matches)))
    assert len(rows) == 2
    assert all(r['overlaps_existing_cohort'] and r['labels_agree'] for r in rows)
    result = dict(status='VERIFIED_OVERLAP', candidate_manifest=str(CANDIDATE / 'locked_test_manifest.csv'),
                  candidate_files=len(rows), existing_cohort_overlaps=len(rows),
                  eligible_as_new_external_cohort=False,
                  interpretation='These two exact event-file copies are not independent new source recordings.',
                  samples=rows)
    (ROOT / 'audit/migrated_locked_test_overlap.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
