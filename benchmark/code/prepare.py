"""Immutable cohort/split preparation, direct legacy event reconstruction and raw frames."""
import importlib.util,re,shutil,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import ROOT,CFG,sha,write_json,ensure
ensure()
cv2.setNumThreads(CFG['cpu_threads'])
vendor=ROOT/'vendor/retinafuse_core.py'
if not vendor.exists():shutil.copy2(CFG['company_source'],vendor)
spec=importlib.util.spec_from_file_location('company',vendor);company=importlib.util.module_from_spec(spec);spec.loader.exec_module(company)
def decode_video(path):
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);assert abs(fps-4)<1e-5,(path,fps)
    frames=[];hashes=[];means=[];motion=[];previous=None
    while True:
        ok,bgr=cap.read()
        if not ok:break
        g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);hashes.append(__import__('hashlib').sha256(g.tobytes()).hexdigest());means.append(float(g.mean()))
        small=cv2.resize(g,(64,64),interpolation=cv2.INTER_AREA)
        if previous is not None:motion.append(float(np.mean(np.abs(small.astype(float)-previous))))
        previous=small.astype(float);frames.append(small.astype(np.float32)/255.)
    cap.release();assert len(frames)>1,path
    f=np.stack(frames);repeats=np.array([False]+[a==b for a,b in zip(hashes[1:],hashes[:-1])])
    return f,repeats,dict(fps=fps,decoded_frames=len(f),repeat_fraction=float(repeats[1:].mean()),mean_gray=float(np.mean(means)),gray_sd=float(np.std(means)),mean_absolute_difference=float(np.mean(motion)),duration_s=(len(f)-1)/fps)
def aligned(frames):
    reference=frames[0];out=[reference];fail=0
    for f in frames[1:]:
        warp=np.eye(2,3,dtype=np.float32)
        try:
            cv2.findTransformECC(reference,f,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,1e-4))
            out.append(cv2.warpAffine(f,warp,(64,64),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT))
        except cv2.error:out.append(f);fail+=1
    return np.stack(out),fail
def main():
    rename=pd.read_csv(CFG['rename_manifest']);rename=rename[~rename.new_name.str.contains('_aug')].copy();assert len(rename)==128
    rename['animal_id']=rename.new_name.str.replace('.csv','',regex=False)
    rename['label']=rename.animal_id.str.startswith('stroke').astype(int)
    parts=rename.old_name.str.extract(r'_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})')
    rename['filename_datetime']=[f'{int(y):04}-{int(m):02}-{int(d):02} {int(h):02}:{int(mi):02}:{int(s):02}'if pd.notna(y)else 'unknown'for y,m,d,h,mi,s in parts.itertuples(index=False,name=None)]
    rename['filename_date']=rename.filename_datetime.str[:10]
    split=pd.read_csv(CFG['legacy_split']);outer=split[split.role.eq('outer_test')][['animal_id','outer_fold','label']]
    assert outer.animal_id.nunique()==128 and not outer.animal_id.duplicated().any()
    manifest=rename.merge(outer,on=['animal_id','label'],validate='one_to_one').sort_values('animal_id').reset_index(drop=True)
    legacy=pd.read_csv(CFG['legacy_groups']);manifest=manifest.merge(legacy[['animal_id','group','date']].rename(columns={'group':'legacy_group','date':'legacy_group_date'}),on='animal_id',validate='one_to_one')
    rows=[]
    prior=ROOT/'audit/content_duplicate_candidates.csv'
    prior_hashes=pd.read_csv(prior).set_index('animal_id').to_dict(orient='index')if prior.exists()else {}
    for r in manifest.itertuples(index=False):
        event=list(Path(CFG['event_root']).glob('*/'+r.animal_id+'.csv'));video=list(Path(CFG['video_root']).glob('[123]/'+r.animal_id+'.avi'))
        assert len(event)==len(video)==1,(r.animal_id,event,video)
        hashfile=ROOT/f'audit/hash_{r.animal_id}.json'
        if hashfile.exists():q=__import__('json').loads(hashfile.read_text())
        elif r.animal_id in prior_hashes:
            old=prior_hashes[r.animal_id];q=dict(animal_id=r.animal_id,event_path=str(event[0]),video_path=str(video[0]),event_sha256=old['event_sha256'],video_sha256=old['video_sha256']);write_json(hashfile,q)
        else:
            q=dict(animal_id=r.animal_id,event_path=str(event[0]),video_path=str(video[0]),event_sha256=sha(event[0]),video_sha256=sha(video[0]));write_json(hashfile,q)
        rows.append(q);print('HASH',len(rows),r.animal_id,flush=True)
    manifest=manifest.merge(pd.DataFrame(rows),on='animal_id',validate='one_to_one');assert manifest.label.value_counts().to_dict()=={1:97,0:31}
    duplicates=manifest[manifest.event_sha256.duplicated(keep=False)|manifest.video_sha256.duplicated(keep=False)]
    duplicates.to_csv(ROOT/'audit/content_duplicate_candidates.csv',index=False)
    # Conservative exact-content source-family correction, retaining all aliases for audit.
    manifest['source_family']=manifest.groupby('video_sha256').animal_id.transform('min')
    assert manifest.groupby('source_family').label.nunique().eq(1).all(),'Duplicate-content label conflict'
    assert manifest.groupby('source_family').event_sha256.nunique().eq(1).all(),'Same video with different retained events requires investigation'
    manifest.to_csv(ROOT/'audit/all_128_record_manifest.csv',index=False,encoding='utf-8-sig')
    manifest[['animal_id','source_family','outer_fold']].to_csv(ROOT/'audit/duplicate_alias_crosswalk.csv',index=False)
    manifest=manifest[manifest.animal_id.eq(manifest.source_family)].copy().reset_index(drop=True)
    assert len(manifest)==CFG['expected_primary_n']
    manifest=manifest.rename(columns={'outer_fold':'legacy_outer_fold'})
    sk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42);split_rows=[];manifest['outer_fold']=0
    for fold,(fit,test)in enumerate(sk.split(manifest,manifest.label,groups=manifest.source_family),1):
        manifest.loc[test,'outer_fold']=fold
        for i in test:split_rows.append(dict(animal_id=manifest.iloc[i].animal_id,label=int(manifest.iloc[i].label),outer_fold=fold,role='outer_test',inner_fold='',source_family=manifest.iloc[i].source_family,split_seed=42))
        inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42+fold*100)
        for k,(a,b)in enumerate(inner.split(manifest.iloc[fit],manifest.iloc[fit].label,groups=manifest.iloc[fit].source_family),1):
            for role,idx in [('inner_train',fit[a]),('inner_val',fit[b])]:
                split_rows.extend(dict(animal_id=manifest.iloc[i].animal_id,label=int(manifest.iloc[i].label),outer_fold=fold,role=role,inner_fold=k,source_family=manifest.iloc[i].source_family,split_seed=42+fold*100)for i in idx)
    split=pd.DataFrame(split_rows)
    manifest.to_csv(ROOT/'data/animal_manifest.csv',index=False,encoding='utf-8-sig')
    split.to_csv(ROOT/'data/split_manifest.csv',index=False,encoding='utf-8-sig')
    cross=pd.crosstab(manifest.legacy_group,manifest.filename_date);cross.to_csv(ROOT/'audit/legacy_groups_vs_filename_dates.csv')
    write_json(ROOT/'audit/cohort_audit.json',dict(original_records=128,n=len(manifest),healthy=31,mcao=89,exact_duplicate_pairs=8,duplicate_pairs_cross_legacy_folds=8,legacy_group_date_mismatches=int((manifest.filename_date!=manifest.legacy_group_date).sum()),filename_date_class_counts={str(k):{str(l):int(n)for l,n in v.items()}for k,v in pd.crosstab(manifest.filename_date,manifest.label).to_dict(orient='index').items()},date_provenance='original filename timestamp through copy_manifest, not assigned legacy date labels',source_sha256=sha(vendor),protocol_sha256=sha(ROOT/'protocol.json')))
    # All preprocessing is per-recording and label blind. Resumption uses content fingerprints.
    qc=[]
    for n,r in enumerate(manifest.itertuples(index=False),1):
        p=ROOT/f'data/{r.animal_id}.npz';meta=ROOT/f'data/{r.animal_id}.json'
        if p.exists()and meta.exists():
            saved=__import__('json').loads(meta.read_text());assert saved['event_sha256']==r.event_sha256 and saved['video_sha256']==r.video_sha256;qc.append(saved);continue
        parsed=company.load_event_csv(r.event_path);assert parsed is not None and len(parsed[0])>0
        t,x,y,pol=parsed;event_frames=company.build_event_frames(t,x,y,pol);stats=company.extract_stat_features(t,x,y,pol)
        raw,repeated,metrics=decode_video(r.video_path);seq_idx=np.round(np.linspace(0,len(raw)-1,60)).astype(int)
        sampled=raw[seq_idx];reg,reg_fail=aligned(raw)
        support=(event_frames.sum((0,1))>0).astype(np.uint8)
        values=dict(legacy_events=event_frames,legacy_stats=stats,raw_sequence=sampled[:,None],single_frame=raw[len(raw)//2][None],mean_image=reg.mean(0)[None],simple_differences=np.concatenate([np.zeros_like(sampled[:1]),np.diff(sampled,axis=0)])[:,None],raw_original_frames=raw,repeat_flags=repeated,support=support)
        np.savez_compressed(p,**values)
        q=dict(animal_id=r.animal_id,label=int(r.label),event_sha256=r.event_sha256,video_sha256=r.video_sha256,cache_sha256=sha(p),registration_failures=reg_fail,median_unique_event_dt=float(np.median(np.diff(np.unique(t)))),**metrics)
        write_json(meta,q);qc.append(q)
        print(f'PREP {n}/{len(manifest)} {r.animal_id}, {len(raw)} frames, repeat={metrics["repeat_fraction"]:.3f}',flush=True)
    pd.DataFrame(qc).to_csv(ROOT/'data/recording_qc.csv',index=False)
    write_json(ROOT/'data/prepared.json',dict(n=len(manifest),manifest_sha256=sha(ROOT/'data/animal_manifest.csv'),split_sha256=sha(ROOT/'data/split_manifest.csv'),protocol_sha256=sha(ROOT/'protocol.json')))
    print('PREPARED',len(manifest),'unique source records',flush=True)
if __name__=='__main__':main()
