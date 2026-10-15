"""Recompute released metrics from untouched source-level probabilities."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
def recompute():
    p=pd.read_csv(ROOT/'results/corrected/seed_ensemble_predictions.csv');rows=[]
    for key,q in p.groupby(['model','design','condition']):
        assert not q.animal_id.duplicated().any()
        assert np.isfinite(q.probability).all() and q.probability.between(0,1).all()
        y=q.label.to_numpy();pred=q.prediction.to_numpy();assert np.array_equal(pred,(q.probability>=q.threshold).astype(int))
        rows.append(dict(zip(['model','design','condition'],key),n=len(q),healthy=int((y==0).sum()),mcao=int((y==1).sum()),auroc=roc_auc_score(y,q.probability),ap=average_precision_score(y,q.probability),brier=brier_score_loss(y,q.probability),sensitivity=float((pred[y==1]==1).mean()),specificity=float((pred[y==0]==0).mean()),accuracy=float((pred==y).mean()),errors=int((pred!=y).sum())))
    out=pd.DataFrame(rows).sort_values(['model','design','condition']).reset_index(drop=True)
    expected=pd.read_csv(ROOT/'results/corrected/seed_ensemble_metrics.csv').sort_values(['model','design','condition']).reset_index(drop=True)
    pd.testing.assert_frame_equal(out[expected.columns],expected,check_exact=False,rtol=1e-11,atol=1e-12)
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='recomputed_metrics.csv');args=ap.parse_args();q=recompute();q.to_csv(args.output,index=False);print(f'PASS: {len(q)} series independently recomputed')
