"""CPU-only real-checkpoint contract check; not an efficacy analysis."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from models import Model
from common import sha

def main():
    torch.set_num_threads(1)
    task = ROOT / 'runs/fivefold_1_RetinaFuse_legacy_seed42'
    q = pd.read_csv(task / 'predictions.csv').groupby('label', sort=True).head(1)
    frames = []; stats = []
    for aid in q.animal_id:
        with np.load(ROOT / f'data/{aid}.npz') as z:
            frames.append(z['legacy_events']); stats.append(z['legacy_stats'])
    bundle = torch.load(task / 'final/bundle.pt', map_location='cpu', weights_only=False)
    x = torch.from_numpy(np.stack(frames))
    s = torch.from_numpy((np.stack(stats) - bundle['mean']) / (bundle['std'] + 1e-8))
    model = Model('RetinaFuse').eval()
    clean = []; frozen = []; errors = []
    with torch.no_grad():
        for state in bundle['states']:
            model.load_state_dict(state)
            out, gate = model(x, s, return_gate=True)
            same, weight = model(x, s, gate_mode='frozen_clean', clean_weights=gate, return_gate=True)
            torch.testing.assert_close(out, same, rtol=0, atol=0)
            torch.testing.assert_close(gate, weight, rtol=0, atol=0)
            altered_x = .5 * x + .5 * x.mean(1, keepdim=True)
            _, frozen_weight = model(altered_x, .5 * s, gate_mode='frozen_clean', clean_weights=gate, return_gate=True)
            torch.testing.assert_close(gate, frozen_weight, rtol=0, atol=0)
            equal_out, equal_weight = model(altered_x, .5 * s, gate_mode='equal', return_gate=True)
            torch.testing.assert_close(equal_weight, torch.full_like(equal_weight, .5), rtol=0, atol=0)
            hf = model.frame(altered_x); hs = model.stats(.5 * s)
            manual = model.head(.5 * hf + .5 * hs)
            torch.testing.assert_close(equal_out, manual, rtol=0, atol=0)
            assert torch.isfinite(out).all() and torch.isfinite(equal_out).all()
            clean.append(out.numpy()); frozen.append(same.numpy())
            errors.append(float((equal_out - manual).abs().max()))
    probability = torch.tensor(np.stack(clean).mean(0)).softmax(1)[:, 1].numpy()
    np.testing.assert_allclose(probability, q.probability, rtol=1e-5, atol=2e-6)
    result = dict(status='PASS', examples=q.animal_id.tolist(), checkpoint_count=len(bundle['states']),
                  checks=['clean frozen-gate identity', 'fixed clean weights after input intervention',
                          'equal-weight replacement', 'manual weighted-embedding calculation',
                          'CPU replay of exported GPU probabilities'],
                  maximum_clean_replay_error=float(np.max(np.abs(probability - q.probability.to_numpy()))),
                  maximum_weighted_sum_error=max(errors),
                  phase_implementation_sha256=sha(ROOT / 'followup/gate_controls.py'),
                  scope='two real held-out examples, computational contract only; no population efficacy claim')
    (ROOT / 'audit/frozen_gate_contract_tests.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
