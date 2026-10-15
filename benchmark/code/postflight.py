"""Prespecified trained-model perturbations, acquisition-feature baselines and final audit; no plots."""
from pathlib import Path
import json
import numpy as np,pandas as pd,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss
from common import ROOT,CFG,sha,write_json
from models import Model
from train import ids_and_inputs,select_threshold,score
def acquisition_baselines(meta):
    qc=pd.read_csv(ROOT/'data/recording_qc.csv').set_index('animal_id').loc[meta.animal_id]
    numeric=['mean_gray','gray_sd','mean_absolute_difference','repeat_fraction','decoded_frames','duration_s'];X=qc[numeric].to_numpy();y=meta.label.to_numpy(int);split=pd.read_csv(ROOT/'data/split_manifest.csv');rows=[]
    for fold in range(1,6):
        test=np.flatnonzero(meta.outer_fold.eq(fold));fit=np.flatnonzero(~meta.outer_fold.eq(fold));inner=split[split.outer_fold.eq(fold)&split.role.eq('inner_val')];pinner=[];yinner=[]
        for k in range(1,6):
            val=np.flatnonzero(meta.animal_id.isin(inner[inner.inner_fold.eq(k)].animal_id));train=np.setdiff1d(fit,val);net=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=42));net.fit(X[train],y[train]);pinner.extend(net.predict_proba(X[val])[:,1]);yinner.extend(y[val])
        threshold=select_threshold(np.array(yinner),np.array(pinner));net=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=42));net.fit(X[fit],y[fit]);p=net.predict_proba(X[test])[:,1]
        rows.extend(dict(animal_id=meta.iloc[i].animal_id,label=int(y[i]),outer_fold=fold,probability=float(prob),threshold=threshold,prediction=int(prob>=threshold),model='Acquisition-QC-LR')for i,prob in zip(test,p))
    out=pd.DataFrame(rows);out.to_csv(ROOT/'results/acquisition_qc_baseline_predictions.csv',index=False);write_json(ROOT/'results/acquisition_qc_baseline_metrics.json',score(out.label.to_numpy(),out.probability.to_numpy()))
def paired_ci(y,a,b):
    rng=np.random.default_rng(20261013);classes=[np.flatnonzero(y==c)for c in [0,1]];values=[]
    for _ in range(CFG['bootstrap_resamples']):
        index=np.concatenate([rng.choice(g,len(g),replace=True)for g in classes]);yy=y[index];aa=a[index];bb=b[index];values.append([roc_auc_score(yy,aa)-roc_auc_score(yy,bb),np.mean((aa-yy)**2)-np.mean((bb-yy)**2)])
    lo,hi=np.quantile(values,[.025,.975],axis=0);return dict(delta_auroc=float(roc_auc_score(y,a)-roc_auc_score(y,b)),delta_brier=float(np.mean((a-y)**2)-np.mean((b-y)**2)),delta_auroc_low=float(lo[0]),delta_auroc_high=float(hi[0]),delta_brier_low=float(lo[1]),delta_brier_high=float(hi[1]))
def main():
    torch.set_num_threads(CFG['cpu_threads']);meta=pd.read_csv(ROOT/'data/animal_manifest.csv');acquisition_baselines(meta)
    conditions=[('clean',0.),('temporal_mean',.25),('temporal_mean',.5),('temporal_mean',.75),('temporal_mean',1.),('summary_zero',.25),('summary_zero',.5),('summary_zero',.75),('summary_zero',1.),('event_thinning',.25),('event_thinning',.5),('event_thinning',.75),('event_thinning',.9)]
    predictions=[]
    for name in CFG['group_models']:
        frame,stats=ids_and_inputs(meta,name,'legacy')
        for runseed in CFG['seeds']:
            for fold in range(1,6):
                taskid=f'fivefold_{fold}_{name}_legacy_seed{runseed}';taskdir=ROOT/'runs'/taskid;info=json.loads((taskdir/'complete.json').read_text());b=torch.load(taskdir/'final/bundle.pt',map_location='cpu',weights_only=False);test=np.flatnonzero(meta.outer_fold.eq(fold));net=Model(name).cuda().eval()
                target=ROOT/f'results/perturbation_{name}_{runseed}_{fold}.csv'
                if target.exists():predictions.append(pd.read_csv(target));continue
                rows=[]
                for family,strength in conditions:
                    xx=frame[test].copy();ss=(stats[test]-b['mean'])/(b['std']+1e-8)
                    if family=='temporal_mean':xx=(1-strength)*xx+strength*xx.mean(1,keepdims=True)
                    elif family=='summary_zero':ss=(1-strength)*ss
                    elif family=='event_thinning':xx=np.random.default_rng(runseed+fold*1000+int(strength*100)).binomial(xx.astype(np.int64),1-strength).astype(np.float32)
                    logits=[];gates=[]
                    with torch.no_grad():
                        for state in b['states']:
                            net.load_state_dict(state);outputs=[];weights=[]
                            for start in range(0,len(test),CFG['batch_size']):
                                out,g=net(torch.from_numpy(xx[start:start+CFG['batch_size']]).cuda(),torch.from_numpy(ss[start:start+CFG['batch_size']]).cuda(),return_gate=True);outputs.append(out.cpu().numpy())
                                if g is not None:weights.append(g.cpu().numpy())
                            logits.append(np.concatenate(outputs))
                            if weights:gates.append(np.concatenate(weights))
                    prob=torch.tensor(np.stack(logits).mean(0)).softmax(1)[:,1].numpy();gate=np.stack(gates).mean(0)if gates else None
                    rows.extend(dict(animal_id=meta.iloc[i].animal_id,label=int(meta.iloc[i].label),model=name,seed=runseed,outer_fold=fold,family=family,strength=strength,probability=float(p),frame_weight=float(gate[j,0])if gate is not None else np.nan)for j,(i,p)in enumerate(zip(test,prob)))
                q=pd.DataFrame(rows);q.to_csv(target,index=False);predictions.append(q);print('PERTURBATION',taskid,flush=True)
    allpred=pd.concat(predictions);allpred.to_csv(ROOT/'results/trained_fusion_perturbation_predictions.csv',index=False);ensemble=allpred.groupby(['animal_id','label','model','family','strength'],as_index=False).agg(probability=('probability','mean'),frame_weight=('frame_weight','mean'));ensemble.to_csv(ROOT/'results/trained_fusion_perturbation_seed_ensemble.csv',index=False)
    rows=[]
    for (family,strength),q in ensemble.groupby(['family','strength']):
        ref=q[q.model.eq('RetinaFuse')].set_index('animal_id').sort_index()
        for name in ['Frame-only','Equal-fusion','Global-weight-fusion','Concat-fusion']:
            control=q[q.model.eq(name)].set_index('animal_id').sort_index();assert ref.index.equals(control.index)
            rows.append(dict(family=family,strength=float(strength),comparator=name,**paired_ci(ref.label.to_numpy(int),ref.probability.to_numpy(),control.probability.to_numpy())))
    pd.DataFrame(rows).to_csv(ROOT/'results/trained_fusion_perturbation_paired_CI.csv',index=False)
    write_json(ROOT/'audit/postflight_complete.json',dict(unique_source_records=len(meta),protocol_sha256=sha(ROOT/'protocol.json'),external_cohort_pending=CFG['external_cohort']is None,operator_device_out_pending=True,all_outer_predictions_kept=True,journal_acceptance_guaranteed=False))
if __name__=='__main__':main()
