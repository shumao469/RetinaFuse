"""Retinal grayscale-variation event encoding (ReGVE).

Native-resolution, signed log-intensity changes within estimated retinal vascular
support produce a polarity-resolved event volume and twelve linked descriptors.
Renamed from decoder.py; numerical conversion and cache contracts are unchanged.
"""
from pathlib import Path
import importlib.util,shutil
import cv2,numpy as np,pandas as pd
from common import ROOT,CFG,sha,write_json
from models import company
def convert(gray,times,mask,fps,theta,dedup=False):
    indices=[];previous=None;next_t=0.
    for i,t in enumerate(times):
        if t+1e-10<next_t:continue
        if not dedup:next_t+=1/fps
        if previous is not None and dedup and np.array_equal(gray[i],gray[previous]):continue
        indices.append(i);previous=i
    # No frame/event cap: all selected original frames are retained. Every condition uses the same support.
    ts=[];xs=[];ys=[];ps=[]
    for a,b in zip(indices[:-1],indices[1:]):
        difference=np.log(gray[b].astype(np.float32)/255.+1e-6)-np.log(gray[a].astype(np.float32)/255.+1e-6)
        for polarity,active in [(1,(difference>theta)&mask),(-1,(difference<-theta)&mask)]:
            y,x=np.nonzero(active);ts.append(np.full(len(x),times[b],np.float32));xs.append(x);ys.append(y);ps.append(np.full(len(x),polarity,np.float32))
    t=np.concatenate(ts)if ts else np.zeros(0,np.float32);x=np.concatenate(xs)if xs else np.zeros(0,int);y=np.concatenate(ys)if ys else np.zeros(0,int);p=np.concatenate(ps)if ps else np.zeros(0,np.float32)
    if len(t):
        order=np.argsort(t,kind='stable');t,x,y,p=[v[order]for v in [t,x,y,p]]
    frames=company.build_event_frames(t,x,y,p,orig_h=mask.shape[0],orig_w=mask.shape[1]);stats=company.extract_stat_features(t,x,y,p)
    return frames,stats,dict(selected_frames=len(indices),events=len(t),empty=len(t)==0,source_fps=fps,theta=theta,deduplicated=dedup,source_resolution=list(mask.shape),event_order='stable timestamp, positive then negative within sampled frame',cap=None)
def main():
    out=ROOT/'data/decoder';out.mkdir(exist_ok=True)
    vendor=ROOT/'vendor/seg.py';source=Path(CFG.get('mask_source',vendor))
    if not vendor.exists():shutil.copy2(source,vendor)
    spec=importlib.util.spec_from_file_location('seg',vendor);seg=importlib.util.module_from_spec(spec);spec.loader.exec_module(seg)
    cv2.setNumThreads(CFG['cpu_threads']);meta=pd.read_csv(ROOT/'data/animal_manifest.csv');rows=[]
    definitions=[(2,.02,False),(4,.02,False),(4,.02,True),(2,.01,False),(2,.04,False),(2,.02,False),(2,.02,False),(2,.02,False)]
    for n,r in enumerate(meta.itertuples(index=False),1):
        p=out/f'{r.animal_id}.npz';info=out/f'{r.animal_id}.json'
        if p.exists()and info.exists():rows.extend(__import__('json').loads(info.read_text())['conditions']);continue
        cap=cv2.VideoCapture(r.video_path);fps=cap.get(cv2.CAP_PROP_FPS);images=[];first=None
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if first is None:first=bgr
            images.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
        cap.release();assert abs(fps-4)<1e-5
        mask=seg.final_optimized_vessel_segmentation(first,str(out/f'{r.animal_id}_mask.png'))>127;assert mask.any()
        images=np.stack(images);times=np.arange(len(images))/fps;values={};animal=[]
        h,w=mask.shape;yy,xx=np.ogrid[:h,:w];circle=(xx-w/2)**2+(yy-h/2)**2<(min(h,w)/2-35)**2
        refmed=float(np.median(images[0][circle]));thumb=cv2.resize(images[0].astype(np.float32),(64,64));registration_failures=0;max_shift=0.
        for condition,(sampling,theta,dedup)in zip(CFG['decoder_conditions'],definitions):
            active=images;support=mask
            if 'brightness_normalized'in condition:
                active=np.stack([np.clip(g.astype(np.float32)*(refmed/max(1,float(np.median(g[circle])))),0,255).astype(np.uint8)for g in images])
            elif 'translation_registered'in condition:
                regs=[]
                for g in images:
                    shift,response=cv2.phaseCorrelate(thumb,cv2.resize(g.astype(np.float32),(64,64)))
                    if not np.isfinite(shift).all()or response<.05:regs.append(g);registration_failures+=1;continue
                    dx,dy=shift;max_shift=max(max_shift,float(np.hypot(dx*w/64,dy*h/64)))
                    warp=np.array([[1,0,-dx*w/64],[0,1,-dy*h/64]],np.float32);regs.append(cv2.warpAffine(g,warp,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT))
                active=np.stack(regs)
            elif 'complement_mask'in condition:support=circle&~mask
            frames,stats,q=convert(active,times,support,sampling,theta,dedup);values[condition+'_events']=frames;values[condition+'_stats']=stats;animal.append(dict(animal_id=r.animal_id,condition=condition,mask_pixels=int(support.sum()),registration_failures=registration_failures if 'translation'in condition else 0,max_shift_pixels=max_shift if 'translation'in condition else 0.,**q))
            if active is not images:del active
        np.savez_compressed(p,**values);write_json(info,dict(video_sha256=r.video_sha256,seg_sha256=sha(vendor),cache_sha256=sha(p),conditions=animal));rows.extend(animal)
        print(f'DECODER {n}/128 {r.animal_id}',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'audit/decoder_manifest.csv',index=False);write_json(out/'prepared.json',dict(n=len(meta),protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':main()
