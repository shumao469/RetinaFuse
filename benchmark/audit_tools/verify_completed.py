"""Independent, CPU-only completion audit. Does not modify frozen training code."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def selected_threshold(q):
    candidates = []
    y = q.label.to_numpy(int)
    p = q.probability.to_numpy(float)
    for t in np.arange(0, 1.001, .01):
        pred = p >= t
        key = (balanced_accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
               np.mean(pred == y), -abs(t - .5))
        candidates.append((key, float(t)))
    return max(candidates, key=lambda x: x[0])[1]

def main():
    frozen = json.loads((ROOT / 'audit/training_freeze.json').read_text())
    assert sha(ROOT / 'protocol.json') == frozen['protocol_sha256']
    assert sha(ROOT / 'data/animal_manifest.csv') == frozen['manifest_sha256']
    for name, digest in frozen['implementation'].items():
        assert sha(ROOT / 'code' / name) == digest, name
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    assert meta.source_family.nunique() == len(meta)
    rows = []
    for path in sorted((ROOT / 'runs').glob('*/complete.json')):
        m = json.loads(path.read_text())
        folder = path.parent
        fit = set(m['fit_animals']); test = set(m['test_animals'])
        assert fit.isdisjoint(test)
        assert m['protocol_sha256'] == frozen['protocol_sha256']
        assert m['input_manifest_sha256'] == frozen['manifest_sha256']
        assert sha(folder / 'predictions.csv') == m['predictions_sha256']
        q = pd.read_csv(folder / 'predictions.csv')
        assert set(q.animal_id) == test and q.animal_id.is_unique
        assert np.isfinite(q.probability).all() and q.probability.between(0, 1).all()
        assert ((q.probability >= q.threshold).astype(int) == q.prediction).all()
        assert q.set_index('animal_id').label.equals(
            meta.set_index('animal_id').loc[q.animal_id, 'label'])
        oof = pd.read_csv(folder / 'inner_oof.csv')
        assert oof.animal_id.is_unique and set(oof.animal_id) == fit
        assert np.isclose(selected_threshold(oof), m['threshold'], atol=1e-12)
        assert m['selected_epochs'] == max(1, int(round(np.median(m['inner_selected_epochs']))))
        stats = []
        for aid in meta.animal_id:
            condition = m['task']['condition']
            source = ROOT / ('data/' + aid + '.npz' if condition == 'legacy'
                             else 'data/decoder/' + aid + '.npz')
            with np.load(source, allow_pickle=False) as z:
                stats.append(z['legacy_stats' if condition == 'legacy' else condition + '_stats'])
        stats = np.stack(stats)
        val_ids = []
        for run in [folder / ('inner_' + str(k)) for k in range(1, 6)] + [folder / 'final']:
            info = json.loads((run / 'complete.json').read_text())
            bundle = run / 'bundle.pt'
            assert sha(bundle) == info['checkpoint_sha256']
            b = torch.load(bundle, map_location='cpu', weights_only=False)
            fi = np.array(info['fit_indices'], dtype=int)
            vi = np.array(info['validation_indices'], dtype=int)
            fids = set(meta.iloc[fi].animal_id)
            vids = set(meta.iloc[vi].animal_id)
            assert fids <= fit and vids <= fit and fids.isdisjoint(vids)
            assert fids.isdisjoint(test) and vids.isdisjoint(test)
            if run.name == 'final':
                assert fids == fit and not vids
            else:
                assert fids | vids == fit
                val_ids.extend(vids)
            # Exact training-input statistics; no validation/test contribution allowed.
            np.testing.assert_allclose(b['mean'], stats[fi].mean(0), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(b['std'], np.maximum(stats[fi].std(0), 1e-8),
                                       rtol=1e-6, atol=1e-7)
            del b
        assert len(val_ids) == len(fit) and set(val_ids) == fit
        rows.append(dict(task=m['task']['id'], test_n=len(q), threshold=m['threshold'],
                         no_fit_test_overlap=True, inner_oof_threshold_verified=True,
                         six_checkpoint_hashes_verified=True, training_only_scalers_verified=True,
                         balanced_accuracy_at_selected_threshold=float(balanced_accuracy_score(q.label, q.prediction)),
                         diagnostic_log_threshold=0.5))
    output = dict(status='PASS', completed_tasks_audited=len(rows), tasks=rows,
                  unit='unique source recording; animal identity requires original ledger',
                  scope='completed tasks only, not pooled efficacy or journal acceptance',
                  verifier_sha256=sha(Path(__file__)))
    target = ROOT / 'audit/independent_completed_task_audit.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, target)
    print(json.dumps(dict(status='PASS', completed_tasks_audited=len(rows))))

if __name__ == '__main__':
    main()
