"""CPU-only numeric smoke test; no fitted model or efficacy selection is produced."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from common import CFG, sha, write_json
from models import Model, company
from train import fit_scaler

def main():
    torch.set_num_threads(1)
    meta = pd.read_csv(ROOT / 'data/animal_manifest.csv')
    fit = np.flatnonzero(meta.outer_fold.ne(1))
    qc = json.loads((ROOT / 'audit/native_decoder_cache_audit.json').read_text())
    assert qc['status'] == 'PASS' and qc['records_audited'] == len(meta) == 120
    assert qc['scope'] == 'complete native-decoder cohort'
    results = []
    for condition in CFG['decoder_conditions']:
        # Stress-case choice is input-only and is never used for model selection.
        counts = {r['animal_id']: r['events'] for r in qc['rows'] if r['condition'] == condition}
        chosen = sorted(counts, key=lambda aid: (-counts[aid], aid))[:2]
        stats = []
        frames = {}
        for aid in meta.animal_id:
            with np.load(ROOT / f'data/decoder/{aid}.npz', allow_pickle=False) as z:
                stats.append(z[condition + '_stats'])
                if aid in chosen:
                    frames[aid] = z[condition + '_events'].copy()
        stats = np.stack(stats)
        mean, std = fit_scaler(stats, fit)
        idx = [meta.index[meta.animal_id.eq(aid)][0] for aid in chosen]
        x = torch.from_numpy(np.stack([frames[aid] for aid in chosen]))
        s = torch.from_numpy((stats[idx] - mean) / (std + 1e-8))
        y = torch.tensor(meta.iloc[idx].label.to_numpy(), dtype=torch.long)
        names = ['RetinaFuse', 'Frame-only']
        if condition == 'graychange_4fps_t002':
            names += ['Statistics-only', 'Equal-fusion', 'Global-weight-fusion', 'Concat-fusion']
        for name in names:
            torch.manual_seed(42)
            model = Model(name).cpu().train()
            logits = model(x, s)
            loss = company.FocalLoss(alpha=CFG['focal_alpha'], gamma=CFG['focal_gamma'])(logits, y)
            assert torch.isfinite(logits).all() and torch.isfinite(loss), (condition, name)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients), (condition, name)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            assert torch.isfinite(norm), (condition, name)
            model.eval()
            with torch.no_grad():
                probability = model(x, s).softmax(1)
            assert torch.isfinite(probability).all()
            results.append(dict(condition=condition, model=name, source_ids=chosen,
                input_max=float(x.max()), standardized_summary_abs_max=float(s.abs().max()),
                loss=float(loss.detach()), gradient_norm_before_clipping=float(norm),
                finite_training_and_eval=True))
            del model, gradients
    assert len(results) == 20
    target = ROOT / 'audit/native_numeric_contract.json'
    write_json(target, dict(status='PASS', checks=len(results), device='cpu',
        scope='Input-only high-count stress cases; finite one-step forward/backward/eval, not convergence or efficacy.',
        scaler_fit='Outer-fold 1 development records only; unchanged fit_scaler implementation.',
        trained_model_saved=False, input_selection_used_for_training_selection=False,
        script_sha256=sha(__file__), model_source_sha256=sha(ROOT / 'code/models.py'),
        protocol_sha256=sha(ROOT / 'protocol.json'), results=results))
    print(json.dumps(dict(status='PASS', checks=len(results), report=str(target))))

if __name__ == '__main__':
    main()
