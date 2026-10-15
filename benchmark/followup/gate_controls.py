"""Frozen-network gate interventions; separate from unchanged fitting queue."""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('RETINAFUSE_WORKDIR', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / 'code'))
from common import CFG, sha, write_json
from models import Model
from train import ids_and_inputs

PHASE = ROOT / 'followup/gate_protocol.json'

def table(path, rows):
    temp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)

def main(input_condition='legacy'):
    spec = json.loads(PHASE.read_text(encoding='utf-8'))
    assert input_condition in spec['input_conditions']
    assert sha(ROOT / 'protocol.json') == spec['training_protocol_sha256']
    torch.set_num_threads(CFG['cpu_threads'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = ROOT / ('results/gate_controls' if input_condition == 'legacy' else 'results/gate_controls_' + input_condition)
    output.mkdir(exist_ok=True)
    freeze = dict(phase_sha256=sha(PHASE), implementation_sha256=sha(Path(__file__)),
                  input_manifest_sha256=sha(ROOT / 'data/animal_manifest.csv'), input_condition=input_condition)
    lock = output / 'freeze.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == freeze
    else:
        write_json(lock, freeze)
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    frames, stats = ids_and_inputs(meta, 'RetinaFuse', input_condition)
    conditions = [('joint', t, s, 0.) for t, s in itertools.product(
        spec['joint_temporal_mean_strengths'], spec['joint_summary_zero_strengths'])]
    conditions += [('event_thinning', 0., 0., x) for x in spec['event_thinning_strengths']]
    tables = []
    for runseed in CFG['seeds']:
        for fold in range(1, 6):
            taskdir = ROOT / 'runs' / f'fivefold_{fold}_RetinaFuse_{input_condition}_seed{runseed}'
            completed = json.loads((taskdir / 'complete.json').read_text())
            checkpoint = taskdir / 'final/bundle.pt'
            checkpoint_meta = json.loads((taskdir / 'final/complete.json').read_text())
            assert sha(checkpoint) == checkpoint_meta['checkpoint_sha256']
            target = output / f'seed{runseed}_fold{fold}.csv'
            provenance = target.with_suffix('.json')
            expected = dict(**freeze, checkpoint_sha256=sha(checkpoint))
            if provenance.exists():
                saved = json.loads(provenance.read_text())
                assert all(saved[k] == v for k, v in expected.items())
                assert sha(target) == saved['predictions_sha256']
                tables.append(pd.read_csv(target)); continue
            b = torch.load(checkpoint, map_location='cpu', weights_only=False)
            test = np.flatnonzero(meta.outer_fold.eq(fold))
            original_x = frames[test]
            original_s = (stats[test] - b['mean']) / (b['std'] + 1e-8)
            net = Model('RetinaFuse').cuda().eval()
            rows = []
            # The frozen weights are checkpoint-specific, not an averaged replacement gate.
            clean_weights = []
            with torch.no_grad():
                for state in b['states']:
                    net.load_state_dict(state); weights = []
                    for start in range(0, len(test), CFG['batch_size']):
                        _, gate = net(torch.from_numpy(original_x[start:start + CFG['batch_size']]).cuda(),
                                      torch.from_numpy(original_s[start:start + CFG['batch_size']]).cuda(), return_gate=True)
                        weights.append(gate.cpu().numpy())
                    clean_weights.append(np.concatenate(weights))
                for family, temporal, summary, thinning in conditions:
                    x = original_x.copy(); s = original_s.copy()
                    if family == 'joint':
                        x = (1 - temporal) * x + temporal * x.mean(1, keepdims=True)
                        s = (1 - summary) * s
                    else:
                        x = np.random.default_rng(runseed + fold * 1000 + int(thinning * 100)).binomial(
                            x.astype(np.int64), 1 - thinning).astype(np.float32)
                    for mode in spec['gate_controls']:
                        all_logits = []; all_weights = []
                        for k, state in enumerate(b['states']):
                            net.load_state_dict(state); logits = []; weights = []
                            for start in range(0, len(test), CFG['batch_size']):
                                end = start + CFG['batch_size']
                                out, gate = net(torch.from_numpy(x[start:end]).cuda(),
                                    torch.from_numpy(s[start:end]).cuda(), gate_mode=mode,
                                    clean_weights=torch.from_numpy(clean_weights[k][start:end]).cuda(), return_gate=True)
                                logits.append(out.cpu().numpy()); weights.append(gate.cpu().numpy())
                            all_logits.append(np.concatenate(logits)); all_weights.append(np.concatenate(weights))
                        probability = torch.tensor(np.stack(all_logits).mean(0)).softmax(1)[:, 1].numpy()
                        gate = np.stack(all_weights).mean(0)
                        rows.extend(dict(animal_id=meta.iloc[i].animal_id, label=int(meta.iloc[i].label),
                            seed=runseed, outer_fold=fold, family=family, temporal_strength=temporal,
                            summary_strength=summary, thinning_strength=thinning, gate_mode=mode,
                            probability=float(probability[j]), frame_weight=float(gate[j, 0]),
                            threshold=completed['threshold']) for j, i in enumerate(test))
            q = pd.DataFrame(rows)
            clean = q[q.family.eq('joint') & q.temporal_strength.eq(0) & q.summary_strength.eq(0)]
            original = pd.read_csv(taskdir / 'predictions.csv').set_index('animal_id')
            for mode in ['learned', 'frozen_clean']:
                r = clean[clean.gate_mode.eq(mode)].set_index('animal_id')
                np.testing.assert_allclose(r.probability, original.loc[r.index].probability, atol=1e-7, rtol=1e-6)
            table(target, q); write_json(provenance, dict(**expected, predictions_sha256=sha(target),
                clean_replay_verified=True, checkpoint_count=len(b['states'])))
            tables.append(q); del net, b
            print('GATE_CONTROLS', runseed, fold, flush=True)
    allpred = pd.concat(tables)
    table(output / 'all_seed_predictions.csv', allpred)
    keys = ['animal_id', 'label', 'family', 'temporal_strength', 'summary_strength', 'thinning_strength', 'gate_mode']
    assert allpred.groupby(keys).seed.nunique().eq(len(CFG['seeds'])).all()
    ens = allpred.groupby(keys, as_index=False).agg(probability=('probability', 'mean'), frame_weight=('frame_weight', 'mean'))
    table(output / 'seed_ensemble_predictions.csv', ens)
    rows = []
    for condition, q in ens.groupby(['family', 'temporal_strength', 'summary_strength', 'thinning_strength']):
        ref = q[q.gate_mode.eq('learned')].set_index('animal_id').sort_index()
        assert len(ref) == len(meta)
        for mode in ['frozen_clean', 'equal']:
            control = q[q.gate_mode.eq(mode)].set_index('animal_id').sort_index()
            assert ref.index.equals(control.index)
            y = ref.label.to_numpy(int); a = ref.probability.to_numpy(); c = control.probability.to_numpy()
            groups = [np.flatnonzero(y == label) for label in [0, 1]]
            rng = np.random.default_rng(20261013); values = []
            for _ in range(spec['bootstrap_resamples']):
                index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                yy = y[index]; aa = a[index]; cc = c[index]
                values.append([roc_auc_score(yy, aa) - roc_auc_score(yy, cc),
                               np.mean((aa - yy) ** 2) - np.mean((cc - yy) ** 2)])
            low, high = np.quantile(values, [.025, .975], axis=0)
            rows.append(dict(family=condition[0], temporal_strength=condition[1], summary_strength=condition[2],
                thinning_strength=condition[3], control=mode, n=len(y),
                delta_auroc=float(roc_auc_score(y, a) - roc_auc_score(y, c)), auroc_ci_low=low[0], auroc_ci_high=high[0],
                delta_brier=float(np.mean((a - y) ** 2) - np.mean((c - y) ** 2)), brier_ci_low=low[1], brier_ci_high=high[1],
                direction='learned minus control', interval='pointwise paired-source bootstrap 95%'))
    table(output / 'paired_gate_control_effects.csv', rows)
    write_json(output / 'complete.json', dict(**freeze, source_records=len(meta), conditions=len(conditions),
        modes=spec['gate_controls'], seeds=CFG['seeds'], prediction_rows=len(allpred),
        clean_replay_verified=True, physiological_causation_claimed=False,
        predictions_sha256=sha(output / 'seed_ensemble_predictions.csv'),
        effect_table_sha256=sha(output / 'paired_gate_control_effects.csv')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-condition', default='legacy')
    main(parser.parse_args().input_condition)
