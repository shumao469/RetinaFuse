"""Report threshold metrics from actual held-out classifications, without fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(q):
    y = q.label.to_numpy(int)
    pred = q.prediction.to_numpy(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(n=len(q), healthy=tn + fp, mcao=tp + fn, tn=tn, fp=fp, fn=fn, tp=tp,
        sensitivity=sensitivity, specificity=specificity,
        balanced_accuracy=(sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
        accuracy=(tn + tp) / len(q), errors=fp + fn,
        brier=float(np.mean((q.probability.to_numpy(float) - y) ** 2)),
        threshold_basis='exported inner-training-selected operating points',
        unit='unique source recording; independent animal identity requires original ledger')

def main():
    config = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    task_rows = []
    frames = []
    for path in sorted((ROOT / 'runs').glob('*/complete.json')):
        info = json.loads(path.read_text())
        prediction_path = path.parent / 'predictions.csv'
        assert sha(prediction_path) == info['predictions_sha256']
        q = pd.read_csv(prediction_path)
        assert q.animal_id.is_unique and set(q.animal_id) == set(info['test_animals'])
        assert np.isfinite(q.probability).all() and q.probability.between(0, 1).all()
        assert ((q.probability >= q.threshold).astype(int) == q.prediction).all()
        spec = info['task']
        task_rows.append(dict(task=spec['id'], model=spec['model'], design=spec['design'],
            condition=spec['condition'], seed=spec['seed'], held_out=spec['held_out'],
            threshold=float(info['threshold']), prediction_sha256=info['predictions_sha256'], **metrics(q)))
        frames.append(q)
    output_dir = ROOT / 'results/operating_points'
    output_dir.mkdir(exist_ok=True)
    pd.DataFrame(task_rows).to_csv(output_dir / 'selected_threshold_metrics_by_task.csv', index=False)
    complete = []
    series_rows = []
    if frames:
        all_predictions = pd.concat(frames, ignore_index=True)
        for key, q in all_predictions.groupby(['model', 'design', 'condition', 'seed']):
            model, design, condition, seed = key
            expected = set(meta.loc[meta.filename_date.ne('unknown'), 'animal_id']) if design == 'filename_date' else set(meta.animal_id)
            if not q.animal_id.is_unique or set(q.animal_id) != expected:
                continue
            base = dict(model=model, design=design, condition=condition, seed=int(seed), aggregation='one complete seed')
            series_rows.append(dict(**base, held_out='pooled', **metrics(q)))
            complete.append(q)
            if design == 'filename_date':
                for group, subset in q.groupby('held_out'):
                    series_rows.append(dict(**base, held_out=str(group), **metrics(subset)))
    ensemble_count = 0
    if complete:
        for key, q in pd.concat(complete).groupby(['model', 'design', 'condition']):
            if set(q.seed) != set(config['seeds']):
                continue
            assert (q.groupby('animal_id').size() == len(config['seeds'])).all()
            assert (q.groupby('animal_id').label.nunique() == 1).all()
            assert (q.groupby('animal_id').held_out.nunique() == 1).all()
            r = q.groupby('animal_id', as_index=False).agg(label=('label', 'first'), held_out=('held_out', 'first'),
                probability=('probability', 'mean'), threshold=('threshold', 'mean'))
            r['prediction'] = (r.probability >= r.threshold).astype(int)
            base = dict(model=key[0], design=key[1], condition=key[2], seed='all', aggregation='mean probability and mean selected threshold across fixed seeds')
            series_rows.append(dict(**base, held_out='pooled', **metrics(r)))
            ensemble_count += 1
            if key[1] == 'filename_date':
                for group, subset in r.groupby('held_out'):
                    series_rows.append(dict(**base, held_out=str(group), **metrics(subset)))
    pd.DataFrame(series_rows, columns=None if series_rows else ['model', 'design', 'condition', 'seed', 'held_out', 'balanced_accuracy']).to_csv(output_dir / 'selected_threshold_metrics_complete_series.csv', index=False)
    status = dict(status='PASS', completed_tasks=len(task_rows), complete_model_seed_series=len(complete),
        complete_seed_ensembles=ensemble_count, exporter_sha256=sha(Path(__file__)),
        scope='Actual selected-threshold classifications. Partial outer-fold collections are not reported as pooled model performance. Single-class groups have undefined balanced accuracy.',
        ensemble_rule='Average fixed-seed probabilities and their training-selected thresholds, then classify once.',
        statistical_uncertainty='No confidence intervals in these point-estimate tables; paired bootstrap is provided by the prespecified analysis.')
    (output_dir / 'coverage.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))

if __name__ == '__main__':
    main()
