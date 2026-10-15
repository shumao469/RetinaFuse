"""Independent read-only recomputation of final pooled discrimination metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / 'results/seed_ensemble_predictions.csv'
    predictions = pd.read_csv(path)
    metrics = pd.read_csv(ROOT / 'results/seed_ensemble_metrics.csv')
    checked = []
    for row in metrics.itertuples(index=False):
        q = predictions[(predictions.model == row.model) & (predictions.design == row.design) & (predictions.condition == row.condition)]
        assert q.animal_id.is_unique and len(q) == row.n
        assert len(q) == (118 if row.design == 'filename_date' else 120)
        y, p = q.label.to_numpy(), q.probability.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        computed = [roc_auc_score(y, p), average_precision_score(y, p), np.mean((p-y)**2)]
        np.testing.assert_allclose(computed, [row.auroc, row.ap, row.brier], atol=1e-12, rtol=1e-12)
        checked.append({'model': row.model, 'design': row.design, 'condition': row.condition, 'n': len(q)})
    assert len(checked) == 43
    paired = pd.read_csv(ROOT / 'results/representation_comparisons/paired_pipeline_effects.csv')
    assert len(paired) == 46 and paired.n.eq(120).all()
    for row in paired.itertuples(index=False):
        first = metrics[(metrics.model == row.first_model) & (metrics.condition == row.first_condition) & (metrics.design == row.design)].iloc[0]
        second = metrics[(metrics.model == row.second_model) & (metrics.condition == row.second_condition) & (metrics.design == row.design)].iloc[0]
        np.testing.assert_allclose(first[row.metric]-second[row.metric], row.delta, atol=1e-12, rtol=1e-12)
        assert row.ci95_low <= row.ci95_high
    result = dict(status='PASS', ensembles_recomputed=43, paired_point_estimates_recomputed=46,
                  prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope='AUROC, AP, Brier and secondary paired point estimates; not independent replication of bootstrap intervals or animal identity', series=checked)
    (ROOT / 'audit/final_metric_consistency.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'series'}))

if __name__ == '__main__':
    main()
