"""Export source tables and paired animal-level uncertainty, never select new models using outer results."""
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from common import ROOT,CFG,sha,write_json
def metrics(q):
    y=q.label.to_numpy(int);p=q.probability.to_numpy(float);pred=q.prediction.to_numpy(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(q),healthy=int((y==0).sum()),mcao=int((y==1).sum()),auroc=float(roc_auc_score(y,p))if len(np.unique(y))==2 else None,ap=float(average_precision_score(y,p))if len(np.unique(y))==2 else None,brier=float(np.mean((p-y)**2)),sensitivity=float(tp/(tp+fn))if tp+fn else None,specificity=float(tn/(tn+fp))if tn+fp else None,accuracy=float((pred==y).mean()),errors=int((pred!=y).sum()))
def main():
    frames=[];lineage=[]
    for p in (ROOT/'runs').glob('*/complete.json'):
        m=json.loads(p.read_text());pred=p.parent/'predictions.csv';assert sha(pred)==m['predictions_sha256'];q=pd.read_csv(pred);assert len(q)==q.animal_id.nunique();assert not set(m['fit_animals'])&set(m['test_animals']);assert set(q.animal_id)==set(m['test_animals']);assert ((q.probability>=q.threshold).astype(int)==q.prediction).all();frames.append(q);lineage.append(dict(task=m['task']['id'],sha256=m['predictions_sha256'],protocol_sha256=m['protocol_sha256'],no_animal_overlap=True))
    if not frames:return
    allpred=pd.concat(frames);allpred.to_csv(ROOT/'results/all_heldout_predictions.csv',index=False);write_json(ROOT/'audit/completed_task_lineage.json',lineage)
    complete=[];summary=[]
    meta=pd.read_csv(ROOT/'data/animal_manifest.csv')
    for key,q in allpred.groupby(['model','design','condition','seed']):
        model,design,condition,seed=key;expected=len(meta)if design!='filename_date'else int(meta.filename_date.ne('unknown').sum())
        if len(q)!=expected or q.animal_id.nunique()!=expected:continue
        complete.append(q);summary.append(dict(model=model,design=design,condition=condition,seed=int(seed),**metrics(q)))
        if design=='filename_date':
            for group,g in q.groupby('held_out'):summary.append(dict(model=model,design='filename_date_specific',condition=condition,seed=int(seed),held_out=group,**metrics(g)))
    if summary:pd.DataFrame(summary).to_csv(ROOT/'results/per_seed_metrics.csv',index=False)
    if not complete:return
    complete=pd.concat(complete);ens=[];summ=[]
    for key,q in complete.groupby(['model','design','condition']):
        if set(q.seed)!=set(CFG['seeds']):continue
        r=q.groupby(['animal_id','label'],as_index=False).agg(probability=('probability','mean'),threshold=('threshold','mean'));r['prediction']=(r.probability>=r.threshold).astype(int)
        for name,val in zip(['model','design','condition'],key):r[name]=val
        ens.append(r);summ.append(dict(model=key[0],design=key[1],condition=key[2],**metrics(r)))
    if not ens:return
    ens=pd.concat(ens);ens.to_csv(ROOT/'results/seed_ensemble_predictions.csv',index=False);pd.DataFrame(summ).to_csv(ROOT/'results/seed_ensemble_metrics.csv',index=False)
    # Frozen primary comparisons and endpoints. 95% descriptive and 99% family-adjusted CIs.
    comparisons=['Frame-only','Global-weight-fusion','Concat-fusion','Unordered-frames','Raw-video'];ci=[]
    for (design,condition),q in ens.groupby(['design','condition']):
        ref=q[q.model.eq('RetinaFuse')].set_index('animal_id').sort_index()
        if ref.empty:continue
        for comparator in comparisons:
            other=q[q.model.eq(comparator)].set_index('animal_id').sort_index()
            if not ref.index.equals(other.index):continue
            signature=sha(ROOT/'protocol.json')+design+condition+comparator
            previous=ROOT/f'results/paired_{design}_{condition}_{comparator}.json'
            if previous.exists():ci.extend(json.loads(previous.read_text()));continue
            y=ref.label.to_numpy(int);a=ref.probability.to_numpy();b=other.probability.to_numpy();rng=np.random.default_rng(20261013);classes=[np.flatnonzero(y==c)for c in [0,1]];values=[]
            for k in range(CFG['bootstrap_resamples']):
                ind=np.concatenate([rng.choice(v,len(v),replace=True)for v in classes]);yy=y[ind];aa=a[ind];bb=b[ind];values.append([roc_auc_score(yy,aa)-roc_auc_score(yy,bb),np.mean((aa-yy)**2)-np.mean((bb-yy)**2)])
            values=np.array(values);rows=[]
            for j,metric in enumerate(['auroc','brier']):
                delta=(roc_auc_score(y,a)-roc_auc_score(y,b))if j==0 else float(np.mean((a-y)**2)-np.mean((b-y)**2));low,high=np.quantile(values[:,j],[.025,.975]);adjlo,adjhi=np.quantile(values[:,j],[.005,.995]);rows.append(dict(design=design,condition=condition,comparator=comparator,metric=metric,delta=float(delta),ci95_low=float(low),ci95_high=float(high),family_ci99_low=float(adjlo),family_ci99_high=float(adjhi),n=len(y),bootstrap_unit='animal',n_resamples=CFG['bootstrap_resamples'],direction='RetinaFuse minus comparator',positive_evidence=bool(adjlo>0 if metric=='auroc'else adjhi<0)))
            write_json(previous,rows);ci.extend(rows)
    if ci:pd.DataFrame(ci).to_csv(ROOT/'results/paired_comparisons.csv',index=False)
    write_json(ROOT/'results/analysis_status.json',dict(completed_tasks=len(frames),complete_model_seed_series=int(len(complete.groupby(['model','design','condition','seed']))),external_validation='not run: no independent cohort identified',journal_acceptance='not certifiable by automatic metrics',primary_protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':main()
