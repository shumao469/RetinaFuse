"""Secondary paired pipeline comparisons on complete fixed-seed ensembles."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('RETINAFUSE_WORKDIR', str(Path(__file__).resolve().parents[1]))).resolve()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan_path = ROOT / 'followup/representation_comparison_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    out = ROOT / 'results/representation_comparisons'
    out.mkdir(exist_ok=True)
    freeze = dict(plan_sha256=sha(plan_path), analysis_sha256=sha(Path(__file__)),
        training_protocol_sha256=sha(ROOT / 'protocol.json'))
    frozen_path = ROOT / 'followup/representation_analysis_frozen.json'
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == freeze, 'Analysis plan changed after first execution'
    else:
        frozen_path.write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    pairs = []
    for model in plan['raw_video_comparators']:
        pairs.append(('raw_video_vs_image_representation', 'Raw-video', 'legacy', model, 'legacy'))
    for model in plan['full_export_retinafuse_comparators']:
        pairs.append(('full_export_event_vs_raw_pipeline', 'RetinaFuse', plan['full_export_event_condition'], model, 'legacy'))
    for model in plan['decoder_models']:
        for condition in cfg['decoder_conditions']:
            if condition != plan['reference_event_condition']:
                pairs.append(('decoder_condition', model, condition, model, plan['reference_event_condition']))
    assert len(pairs) == 23
    columns = ['animal_id', 'label', 'model', 'design', 'condition', 'probability']
    prediction_path = ROOT / 'results/seed_ensemble_predictions.csv'
    prediction = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame(columns=columns)
    seed_path = ROOT / 'results/per_seed_metrics.csv'
    seed_metrics = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame(columns=['model', 'design', 'condition', 'seed'])
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv').set_index('animal_id').sort_index()
    rows = []
    pending = []
    for family, first, first_condition, second, second_condition in pairs:
        spec = dict(family=family, first_model=first, first_condition=first_condition,
                    second_model=second, second_condition=second_condition, design=plan['design'])
        selected = []
        for name, condition in [(first, first_condition), (second, second_condition)]:
            q = prediction[prediction.model.eq(name) & prediction.condition.eq(condition) & prediction.design.eq(plan['design'])].set_index('animal_id').sort_index()
            seeds = seed_metrics[seed_metrics.model.eq(name) & seed_metrics.condition.eq(condition) & seed_metrics.design.eq(plan['design'])]
            if q.index.is_unique and q.index.equals(meta.index) and set(seeds.seed.astype(int)) == set(cfg['seeds']):
                np.testing.assert_array_equal(q.label.to_numpy(int), meta.label.to_numpy(int))
                assert np.isfinite(q.probability).all() and q.probability.between(0, 1).all()
                selected.append(q)
        if len(selected) != 2:
            pending.append(spec)
            continue
        a, b = selected
        y = a.label.to_numpy(int)
        pa = a.probability.to_numpy(float)
        pb = b.probability.to_numpy(float)
        rng = np.random.default_rng(plan['bootstrap_seed'])
        classes = [np.flatnonzero(y == value) for value in [0, 1]]
        values = np.empty((plan['bootstrap_resamples'], 2))
        for k in range(len(values)):
            index = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
            yy, aa, bb = y[index], pa[index], pb[index]
            values[k] = [roc_auc_score(yy, aa) - roc_auc_score(yy, bb), np.mean((aa - yy) ** 2) - np.mean((bb - yy) ** 2)]
        endpoints = [roc_auc_score(y, pa) - roc_auc_score(y, pb), np.mean((pa - y) ** 2) - np.mean((pb - y) ** 2)]
        bounds = np.quantile(values, [.025, .975], axis=0)
        for j, endpoint in enumerate(plan['endpoints']):
            rows.append(dict(**spec, metric=endpoint, n=len(y), delta=float(endpoints[j]),
                ci95_low=float(bounds[0, j]), ci95_high=float(bounds[1, j]),
                interval=plan['interval'], direction=plan['difference_direction'],
                unit=plan['unit'], prediction_table_sha256=sha(prediction_path)))
    if rows:
        pd.DataFrame(rows).to_csv(out / 'paired_pipeline_effects.csv', index=False)
    status = dict(**freeze, planned_comparisons=len(pairs), completed_comparisons=len(rows) // 2,
        state='complete_descriptive_analysis' if not pending else 'waiting_for_complete_fixed_seed_ensembles',
        pending=pending, interpretation=plan['interpretation'], multiplicity=plan['multiplicity'])
    (out / 'coverage.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps({key: status[key] for key in ['planned_comparisons', 'completed_comparisons', 'state']}))

if __name__ == '__main__':
    main()
