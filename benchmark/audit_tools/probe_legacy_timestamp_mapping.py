"""Small diagnostic of candidate time mappings, not proof of converter provenance."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    cv2.setNumThreads(1)
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv').set_index('animal_id')
    rows = []
    for aid in ['health_001', 'stroke_001']:
        record = meta.loc[aid]
        pieces = []; first_time = None
        for chunk in pd.read_csv(record.event_path, chunksize=25000):
            if first_time is None:
                first_time = float(chunk['timestamp(s)'].iloc[0])
            pieces.append(chunk[chunk['timestamp(s)'].eq(first_time)])
            if chunk['timestamp(s)'].max() > first_time:
                break
        events = pd.concat(pieces, ignore_index=True)
        indices = np.round(np.linspace(0, len(events) - 1, min(4096, len(events)))).astype(int)
        events = events.iloc[indices]
        cap = cv2.VideoCapture(record.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native = int(round(first_time * fps))
        uniform = np.round(np.linspace(0, total - 1, 16)).astype(int)
        mappings = [('native_adjacent', max(0, native - 1), native),
                    ('uniform_16_full_video_retimed_at_4fps',
                     int(uniform[max(0, native - 1)]), int(uniform[native]))]
        for mapping, a, b in mappings:
            grays = []
            for index in [a, b]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read(); assert ok
                grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.)
            change = np.log(grays[1] + 1e-6) - np.log(grays[0] + 1e-6)
            x = events.x.to_numpy(int); y = events.y.to_numpy(int)
            signed = change[y, x] * events.polarity.to_numpy(float)
            rows.append(dict(animal_id=aid, event_timestamp_s=first_time, mapping=mapping,
                             frame_a=a, frame_b=b, sampled_event_locations=len(events),
                             sign_concordance=float((signed > 0).mean()),
                             fraction_above_current_log_threshold_002=float((signed > .02).mean()),
                             median_signed_log_change=float(np.median(signed))))
        cap.release()
    pd.DataFrame(rows).to_csv(ROOT / 'audit/legacy_timestamp_mapping_probe.csv', index=False)
    (ROOT / 'audit/legacy_timestamp_mapping_probe.json').write_text(json.dumps(dict(
        records=['health_001', 'stroke_001'], selection='first canonical ID in each class, first occupied event timestamp, evenly spaced event-row indices',
        scope='two examples only; candidate correspondence is diagnostic, not converter version certification',
        training_changed=False, results=rows), indent=2), encoding='utf-8')
    print(json.dumps(rows))

if __name__ == '__main__':
    main()
