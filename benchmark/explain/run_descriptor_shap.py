"""Run archived original-recording descriptor SHAP with explicit local inputs."""
import argparse, ast, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from exact_shap import FEATURES, LABELS, GROUPS, shapley, group_masks, select_donors, coalition_values

ROOT = Path(__file__).resolve().parents[2]


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rearrange(x, pattern):
    if pattern == 'b t c h w -> b c t h w':
        return x.permute(0, 2, 1, 3, 4)
    if pattern == 'b c t h w -> b t (c h w)':
        return x.permute(0, 2, 1, 3, 4).reshape(x.shape[0], x.shape[2], -1)
    raise ValueError(pattern)


def model_class(source):
    """Use only pinned architecture definitions, without running training setup."""
    names = {'PositionalEncoding', 'EventFrameBranch', 'StatBranch', 'AdaptiveFusion', 'EventFusionNet'}
    tree = ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('The supplied source is not the expected RetinaFuse architecture')
    ns = dict(np=np, torch=torch, nn=torch.nn, math=math, rearrange=rearrange,
              ABLATION_MODES=('full', 'wo_frame', 'wo_stat'), SEQ_LEN=60)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns['EventFusionNet']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--limit', type=int, help='Diagnostic subset only; marked incomplete')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    def path(key):
        p = Path(cfg[key]).expanduser()
        return p if p.is_absolute() else (args.config.resolve().parent / p).resolve()
    out = path('output_dir'); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get('device', 'cpu'))
    torch.set_num_threads(int(cfg.get('cpu_threads', 6)))
    torch.manual_seed(int(cfg.get('seed', 20261015)))
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = path('model_source')
    if digest(source) != cfg['model_source_sha256']:
        raise ValueError('Model source SHA-256 mismatch; run scripts/fetch_sources.py')
    records = pd.read_csv(path('predictions'))
    if 'model' in records:
        records = records[records.model.eq('RetinaFuse')]
    records = records.sort_values(['outer_fold', 'animal_id']).reset_index(drop=True)
    required = {'animal_id', 'label', 'outer_fold', 'prediction', 'probability'}
    if not required.issubset(records) or records.empty or records.animal_id.duplicated().any():
        raise ValueError('Expected one archived original-recording prediction per identifier')
    if not set(records.label).issubset({0, 1}) or not np.isfinite(records.probability).all() or not records.probability.between(0, 1).all():
        raise ValueError('Invalid labels or probabilities')
    lineage = pd.read_csv(path('lineage')).set_index('animal_id')
    if not lineage.index.is_unique:
        raise ValueError('Lineage identifiers must be unique')
    records['source_family'] = records.animal_id.map(lineage.source_family)
    if records.source_family.isna().any() or (records.groupby('source_family').label.nunique() > 1).any():
        raise ValueError('Missing or label-inconsistent source families')
    manifest = pd.read_csv(path('checkpoint_manifest'))
    original = np.load(path('input_cache'), allow_pickle=False)
    embeddings = np.load(path('embedding_cache'), allow_pickle=False) if cfg.get('embedding_cache') else None
    netclass = model_class(source)
    inputs = {k: digest(path(k)) for k in ['model_source', 'predictions', 'lineage', 'checkpoint_manifest', 'input_cache']}
    if embeddings is not None: inputs['embedding_cache'] = digest(path('embedding_cache'))
    protocol = dict(config=cfg, inputs_sha256=inputs, method='exact interventional Shapley enumeration',
                    output='P(MCAO), mean-checkpoint-logit aggregation', features=FEATURES, groups=GROUPS,
                    scope='archived original-recording functional explanation; not multiview or corrected-source validation')
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    cache = out / 'coalitions'; cache.mkdir(exist_ok=True)
    prior = out / 'protocol.json'
    if not prior.exists() and any(cache.iterdir()):
        raise ValueError('Existing coalition files have no provenance; use a fresh output_dir')
    if prior.exists() and json.loads(prior.read_text(encoding='utf-8')).get('fingerprint') != fingerprint:
        raise ValueError('Output directory has incompatible provenance; use a new output_dir')
    protocol['fingerprint'] = fingerprint; save(prior, protocol)
    feature_rows, group_rows, diagnostics, donor_rows, checkpoints = [], [], [], [], []
    gmasks = group_masks(GROUPS, len(FEATURES)); done = 0
    for fold in sorted(records.outer_fold.unique()):
        if args.limit is not None and done >= args.limit: break
        q = records[records.outer_fold.eq(fold)]
        donors = select_donors(records, fold, cfg.get('healthy_donors', 4), cfg.get('mcao_donors', 12), cfg.get('seed', 20261015))
        donor_rows.extend(dict(outer_fold=int(fold), **r) for r in donors[['animal_id', 'label', 'source_family']].to_dict('records'))
        ckpt = path('checkpoint_root') / cfg['checkpoint_pattern'].format(fold=int(fold))
        expected = manifest[(manifest.model == 'RetinaFuse') & (manifest.fold == fold)]
        if len(expected) != 1 or digest(ckpt) != expected.iloc[0].sha256:
            raise ValueError(f'Checkpoint SHA-256 mismatch or ambiguous manifest, fold {fold}')
        bundle = torch.load(ckpt, map_location='cpu', weights_only=True)
        if bundle['ablation'] != 'full' or int(bundle['outer_fold']) != fold:
            raise ValueError('Checkpoint mode/fold mismatch')
        models = []
        for state in bundle['model_state_dicts']:
            model = netclass('full').to(device).eval(); model.load_state_dict(state, strict=True); models.append(model)
        checkpoints.append(dict(outer_fold=int(fold), sha256=digest(ckpt), states=len(models)))
        mean = np.asarray(bundle['stat_mean'], np.float32); std = np.asarray(bundle['stat_std'], np.float32) + 1e-8
        bg = np.stack([(original[i + '_stats'].astype(np.float32) - mean) / std for i in donors.animal_id])
        for ri, r in enumerate(q.itertuples()):
            if args.limit is not None and done >= args.limit: break
            if Path(r.animal_id).name != r.animal_id or r.animal_id in {'.', '..'}:
                raise ValueError('Unsafe recording identifier')
            raw = original[r.animal_id + '_stats'].astype(np.float32)
            target = (raw - mean) / std
            with torch.no_grad():
                frames = torch.from_numpy(original[r.animal_id + '_frames'].astype(np.float32))[None].to(device)
                hf = ([torch.from_numpy(embeddings[f'{r.animal_id}_checkpoint{k}_frame'].astype(np.float32)).to(device)[None] for k in range(len(models))]
                      if embeddings is not None else [net.branch_a(frames) for net in models])
                if ri == 0 and embeddings is not None:
                    if any(torch.max(torch.abs(net.branch_a(frames) - f)).item() >= cfg.get('embedding_tolerance', 3e-4) for net, f in zip(models, hf)):
                        raise ValueError('Cached frame embedding does not match fitted model')
                cp = cache / (r.animal_id + '.npy')
                if cp.exists(): values = np.load(cp, allow_pickle=False)
                else:
                    values = coalition_values(models, hf, torch.from_numpy(target).to(device), torch.from_numpy(bg).to(device), cfg.get('coalition_batch', 256)); np.save(cp, values)
            phi = shapley(values, len(FEATURES)); gp = shapley(values[gmasks], len(GROUPS))
            replay = abs(float(values[-1]) - r.probability); residual = abs(values[0] + phi.sum() - values[-1]); gres = abs(values[0] + gp.sum() - values[-1])
            if replay > cfg.get('prediction_tolerance', 3e-6) or max(residual, gres) > 1e-10:
                raise ValueError(f'Prediction replay or additivity failed for {r.animal_id}')
            base = dict(animal_id=r.animal_id, label=int(r.label), outer_fold=int(fold), prediction=int(r.prediction), probability=float(r.probability), base_probability=float(values[0]))
            percentiles = (bg < target[None]).mean(0)
            feature_rows.extend(dict(**base, feature=f, feature_label=LABELS[j], raw_value=float(raw[j]), training_reference_percentile=float(percentiles[j]), shap_probability=float(phi[j]), group=next(g for g, ix in GROUPS.items() if j in ix)) for j, f in enumerate(FEATURES))
            group_rows.extend(dict(**base, group=g, group_shap_probability=float(gp[j]), sum_feature_shap_probability=float(phi[ix].sum())) for j, (g, ix) in enumerate(GROUPS.items()))
            diagnostics.append(dict(animal_id=r.animal_id, baseline_replay_error=replay, additivity_residual=residual, group_additivity_residual=gres)); done += 1
            print(f'Completed {done}/{len(records)}: {r.animal_id}', flush=True)
        del models
    for name, rows in [('descriptor_shap', feature_rows), ('group_shap', group_rows), ('checks', diagnostics), ('reference_donors', donor_rows)]:
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    save(out / 'checkpoints.json', checkpoints)
    save(out / 'status.json', dict(status='COMPLETE' if done == len(records) else 'DIAGNOSTIC_SUBSET', completed=done, total=len(records), fingerprint=fingerprint))


if __name__ == '__main__': main()
