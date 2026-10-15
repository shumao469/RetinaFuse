"""Export panel-resolved SHAP sources with source-family bootstrap intervals."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(input_dir, lineage_path, output_dir, supplementary_figure=6, draws=5000, seed=20261015):
    src, out = Path(input_dir), Path(output_dir)
    if not (src / 'status.json').exists() or json.loads((src / 'status.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('Summaries require a complete analysis, not a diagnostic subset')
    if draws < 2: raise ValueError('At least two bootstrap draws are required')
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(src / 'descriptor_shap.csv'); g = pd.read_csv(src / 'group_shap.csv')
    animals = d.drop_duplicates('animal_id').sort_values('animal_id').set_index('animal_id')
    ids = animals.index.tolist(); lineage = pd.read_csv(lineage_path).set_index('animal_id')
    if not lineage.index.is_unique: raise ValueError('Ambiguous lineage')
    families = lineage.loc[ids, 'source_family'].to_numpy(); unique = sorted(set(families))
    if any(animals.iloc[np.flatnonzero(families == f)].label.nunique() != 1 for f in unique):
        raise ValueError('A source family has inconsistent labels')
    labels = [int(animals.iloc[np.flatnonzero(families == f)[0]].label) for f in unique]
    members = np.array([[int(f == x) for x in unique] for f in families])
    rng = np.random.default_rng(seed); weights = []
    for _ in range(draws):
        chosen = []
        for label in [0, 1]:
            ix = np.flatnonzero(np.array(labels) == label)
            if not len(ix): raise ValueError('Class-stratified summaries require both classes')
            chosen.extend(rng.choice(ix, len(ix), replace=True))
        weights.append(np.bincount(chosen, minlength=len(unique)))
    weights = np.array(weights) @ members.T; weights = weights / weights.sum(1, keepdims=True)
    def estimate(v, label=None):
        v = np.asarray(v); mask = np.ones(len(v), bool) if label is None else animals.label.to_numpy() == label
        w = weights * mask[None]; w = w / w.sum(1, keepdims=True)
        lo, hi = np.quantile(w @ v, [.025, .975])
        return dict(mean=float(v[mask].mean()), lo=float(lo), hi=float(hi))
    features = []
    for (key, label), q in d.groupby(['feature', 'feature_label']):
        q = q.set_index('animal_id').loc[ids]; v = q.shap_probability.to_numpy()
        features.append(dict(key=key, label=label, importance=estimate(abs(v)), healthy=estimate(v, 0), mcao=estimate(v, 1),
                             points=[dict(id=i, x=float(r.shap_probability), value=float(r.raw_value), percentile=float(r.training_reference_percentile), label=int(r.label)) for i, r in q.iterrows()]))
    features.sort(key=lambda f: -f['importance']['mean'])
    groups = []
    for key, q in g.groupby('group', sort=False):
        q = q.set_index('animal_id').loc[ids]; v = q.group_shap_probability.to_numpy()
        groups.append(dict(key=key, importance=estimate(abs(v)),
                           points=[dict(id=i, x=float(r.sum_feature_shap_probability), y=float(r.group_shap_probability)) for i, r in q.iterrows()]))
    groups.sort(key=lambda f: -f['importance']['mean'])
    for f, color in zip(groups, ['#C58F24', '#52647F', '#4D927B', '#9470AD']): f['color'] = color
    corr = d.pivot(index='animal_id', columns='feature', values='raw_value').loc[ids, [f['key'] for f in features]].corr(method='spearman')
    prefix = f'Supplementary Fig. {supplementary_figure}'
    corr.to_csv(out / 'Fig. 2h_descriptor_Spearman_correlation.csv')
    pd.DataFrame([dict(code=f'{j+1:02d}', feature=f['key'], display_label=f['label']) for j, f in enumerate(features)]).to_csv(out / 'Fig. 2h_feature_code_key.csv', index=False)
    pd.DataFrame([dict(feature=f['key'], label=f['label'], class_label=c, **f[k]) for f in features for c, k in [('Healthy', 'healthy'), ('MCAO', 'mcao')]]).to_csv(out / 'Fig. 2g_class_specific_signed_SHAP.csv', index=False)
    pd.DataFrame([dict(feature=f['key'], label=f['label'], **f['importance']) for f in features]).to_csv(out / (prefix + 'a_mean_absolute_SHAP.csv'), index=False)
    d.to_csv(out / (prefix + 'b_exact_descriptor_SHAP.csv'), index=False)
    g.to_csv(out / (prefix + 'c_exact_group_SHAP.csv'), index=False)
    pd.DataFrame([dict(group=f['key'], **f['importance']) for f in groups]).to_csv(out / (prefix + 'c_group_importance_and_cluster_CI.csv'), index=False)
    g.to_csv(out / (prefix + 'd_joint_group_and_feature_sum.csv'), index=False)
    cases = []
    for label, panel in [(0, 'e'), (1, 'f')]:
        a = animals[(animals.label == label) & (animals.prediction == label)].copy()
        if a.empty: continue
        a['distance'] = (a.probability - a.probability.median()).abs()
        ident = a.reset_index().sort_values(['distance', 'animal_id']).iloc[0].animal_id
        q = d[d.animal_id.eq(ident)].set_index('feature').loc[[f['key'] for f in features]]
        q.reset_index().to_csv(out / (prefix + panel + '_local_SHAP.csv'), index=False)
        cases.append(dict(id=ident, label=label, probability=float(a.loc[ident, 'probability']), baseline=float(a.loc[ident, 'base_probability']),
                          features=[dict(label=r.feature_label, shap=float(r.shap_probability)) for r in q.itertuples()]))
    result = dict(features=features, groups=groups, corr=corr.to_numpy().tolist(), cases=cases,
                  supplementary_figure=supplementary_figure, bootstrap_draws=draws,
                  interval='class-stratified source-family cluster bootstrap; fixed networks and donors')
    (out / 'plot_data.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    tables = sorted(out.glob('*.csv'))
    pd.DataFrame([dict(file=p.name, rows=len(pd.read_csv(p)), analysis='archived original-recording descriptor explanation') for p in tables]).to_csv(out / 'panel_source_index.csv', index=False)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', required=True, type=Path); ap.add_argument('--lineage', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path); ap.add_argument('--supplementary-figure', type=int, default=6)
    ap.add_argument('--bootstrap-draws', type=int, default=5000); ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args(); summarize(a.input_dir, a.lineage, a.output_dir, a.supplementary_figure, a.bootstrap_draws, a.seed)
