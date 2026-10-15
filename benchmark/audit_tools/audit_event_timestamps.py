"""Bounded-memory audit of legacy event timing; no prediction or training changes."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    qc = pd.read_csv(ROOT / 'data/recording_qc.csv').set_index('animal_id')
    rows = []
    for record in meta.itertuples(index=False):
        n = zero = intervals = negative = bad = native_grid = two_grid = 0
        unique = set(); previous = None
        for chunk in pd.read_csv(record.event_path, usecols=['timestamp(s)'], chunksize=25000):
            times = chunk['timestamp(s)'].to_numpy(float)
            bad += int((~np.isfinite(times)).sum())
            times = times[np.isfinite(times)]
            if not len(times):
                continue
            diffs = np.diff(times)
            if previous is not None:
                diffs = np.r_[times[0] - previous, diffs]
            previous = times[-1]
            zero += int((diffs == 0).sum()); negative += int((diffs < 0).sum())
            intervals += len(diffs); n += len(times)
            unique.update(np.unique(times).tolist())
            fps = float(qc.loc[record.animal_id, 'fps'])
            native_grid += int(np.isclose(times * fps, np.round(times * fps), atol=1e-6, rtol=0).sum())
            two_grid += int(np.isclose(times * 2, np.round(times * 2), atol=1e-6, rtol=0).sum())
        stamps = np.array(sorted(unique))
        steps = np.diff(stamps)
        duration = float(qc.loc[record.animal_id, 'duration_s'])
        rows.append(dict(animal_id=record.animal_id, label=int(record.label), finite_events=n,
                         nonfinite_timestamps=bad, csv_negative_steps=negative,
                         zero_interval_fraction=zero / intervals if intervals and not negative else None,
                         unique_timestamps=len(stamps), first_timestamp=float(stamps[0]) if len(stamps) else None,
                         last_timestamp=float(stamps[-1]) if len(stamps) else None,
                         median_unique_step_s=float(np.median(steps)) if len(steps) else None,
                         minimum_unique_step_s=float(steps.min()) if len(steps) else None,
                         within_export_duration=bool(len(stamps) and stamps[0] >= 0 and stamps[-1] <= duration + 1e-6),
                         native_container_grid_fraction=native_grid / n if n else None,
                         phase_zero_2fps_grid_fraction=two_grid / n if n else None,
                         event_source_sha256=record.event_sha256))
        print('TIMESTAMP_QC', len(rows), record.animal_id, len(stamps), flush=True)
    out = pd.DataFrame(rows)
    target = ROOT / 'audit/legacy_event_timestamp_qc.csv'
    tmp = target.with_suffix('.tmp'); out.to_csv(tmp, index=False); os.replace(tmp, target)
    report = dict(records=len(rows), records_with_nonfinite_timestamps=int((out.nonfinite_timestamps > 0).sum()),
                  records_with_unsorted_csv_times=int((out.csv_negative_steps > 0).sum()),
                  records_outside_export_duration=int((~out.within_export_duration).sum()),
                  records_off_phase_zero_2fps_grid=int((out.phase_zero_2fps_grid_fraction < 1 - 1e-6).sum()),
                  records_off_native_container_grid=int((out.native_container_grid_fraction < 1 - 1e-6).sum()),
                  note='Zero inter-event intervals are shared timestamps, not finer temporal measurement. Grid checks assume t=frame_index/container_fps and phase zero; legacy converter provenance must be established separately.',
                  training_changed=False)
    (ROOT / 'audit/legacy_event_timestamp_qc.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
