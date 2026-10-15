"""Pin public converter history without changing training code or inputs."""
import ast
import hashlib
import json
import urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = 'taotao9029/OCTA_Segment-set-attention'

def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'RetinaFuse-provenance-audit'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()

def main():
    history = json.loads(get(f'https://api.github.com/repos/{REPO}/commits?path=Video2Events.py&per_page=100'))
    head = json.loads(get(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    versions = []
    for commit in history:
        raw = get(f'https://raw.githubusercontent.com/{REPO}/{commit["sha"]}/Video2Events.py')
        source = raw.decode('utf-8-sig')
        settings = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for name in node.targets:
                    if isinstance(name, ast.Name) and name.id in {'SAMPLE_FPS', 'MAX_SAMPLE_FRAMES', 'THETA', 'EPS', 'MAX_EVENTS', 'USE_LOG_RELATIVE_CHANGE'}:
                        settings[name.id] = ast.literal_eval(node.value)
        versions.append(dict(commit=commit['sha'], date=commit['commit']['committer']['date'],
            message=commit['commit']['message'], sha256=hashlib.sha256(raw).hexdigest(), settings=settings,
            frame_clock_expression='current_t = frame_index / fps' if 'current_t = frame_index / fps' in source else None,
            url=f'https://github.com/{REPO}/blob/{commit["sha"]}/Video2Events.py'))
    latest_raw = get(f'https://raw.githubusercontent.com/{REPO}/{head}/Video2Events.py')
    timing = pd.read_csv(ROOT / 'audit/legacy_event_timestamp_qc.csv')
    output = dict(repository=REPO, head_commit=head, file_changing_commits=versions,
        history_query_limit=100, history_exhausted=len(history) < 100,
        head_file_sha256=hashlib.sha256(latest_raw).hexdigest(),
        audited_legacy_records=len(timing),
        records_off_phase_zero_2fps_grid=int((timing.phase_zero_2fps_grid_fraction < 1 - 1e-9).sum()),
        legacy_event_count_min=int(timing.finite_events.min()), legacy_event_count_max=int(timing.finite_events.max()),
        status='SOURCE_PINNED_ACTUAL_GENERATION_UNCONFIRMED',
        conclusion='Public defaults and timestamp mapping do not reproduce the off-0.5-second timestamps in the stored legacy CSVs. A modified configuration or different converter may have been used. Repository identity alone does not establish actual generation provenance. The event cap can truncate duration but cannot create off-grid timestamps.',
        scope='Read-only repository/version verification; no training inputs changed.')
    (ROOT / 'audit/converter_repository_provenance.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
