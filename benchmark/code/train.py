"""One nested outer task, fully resumable, held-out predictions exported only after inner development."""
import argparse,copy,json,math,os,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import balanced_accuracy_score,average_precision_score,roc_auc_score,brier_score_loss,f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from common import ROOT,CFG,sha,write_json
from models import Model,company
def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def save(p,data):
    tmp=Path(str(p)+'.tmp');torch.save(data,tmp);os.replace(tmp,p)
def ids_and_inputs(meta,name,condition):
    frames=[];stats=[]
    for aid in meta.animal_id:
        with np.load(ROOT/f'data/{aid}.npz',allow_pickle=False)as z:
            if condition!='legacy':
                with np.load(ROOT/f'data/decoder/{aid}.npz',allow_pickle=False)as dz:
                    x=dz[condition+'_events'].copy();s=dz[condition+'_stats'].copy()
            else:x=z['legacy_events'].copy();s=z['legacy_stats'].copy()
            if name=='Single-frame':x=z['single_frame'].copy()
            elif name=='Mean-image':x=z['mean_image'].copy()
            elif name in ['Unordered-frames','Raw-video']:x=z['raw_sequence'].copy()
            elif name=='Frame-difference-video':x=z['simple_differences'].copy()
        frames.append(x);stats.append(s)
    return np.stack(frames),np.stack(stats)
def fit_scaler(stats,index):
    mean=stats[index].mean(0);std=stats[index].std(0);return mean,np.maximum(std,1e-8)
def loader(frames,stats,labels,index,mean,std,training,s):
    dataset=TensorDataset(torch.from_numpy(frames[index]),torch.from_numpy((stats[index]-mean)/(std+1e-8)),torch.from_numpy(labels[index]).long())
    gen=torch.Generator().manual_seed(s)
    if training:
        counts=np.bincount(labels[index],minlength=2);weight=1/np.maximum(counts,1)[labels[index]]
        sampler=WeightedRandomSampler(torch.tensor(weight),len(index),replacement=True,generator=gen)
    else:sampler=None
    return DataLoader(dataset,batch_size=CFG['batch_size'],sampler=sampler,shuffle=False,num_workers=0,pin_memory=True,generator=gen)
@torch.no_grad()
def predict(net,dl,device):
    net.eval();out=[]
    for x,s,y in dl:out.append(net(x.to(device),s.to(device)).cpu().numpy())
    return np.concatenate(out)
def score(y,p):
    auc=roc_auc_score(y,p)if len(np.unique(y))==2 else None
    ap=average_precision_score(y,p)if len(np.unique(y))==2 else None
    return dict(auroc=auc,ap=ap,brier=brier_score_loss(y,p),balanced_accuracy=balanced_accuracy_score(y,p>=.5)if len(np.unique(y))==2 else None)
def select_threshold(y,p):
    best=None;thr=.5
    for t in np.arange(0,1.001,.01):
        pred=p>=t;ba=balanced_accuracy_score(y,pred);f1=f1_score(y,pred,zero_division=0);acc=np.mean(y==pred)
        key=(ba,f1,acc,-abs(t-.5))
        if best is None or key>best:best=key;thr=float(t)
    return thr
def train_run(name,frames,stats,labels,fit,validation,folder,runseed,epochs=None):
    folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json';bundle=folder/'bundle.pt'
    if done.exists():return json.loads(done.read_text()),torch.load(bundle,map_location='cpu',weights_only=False)
    seed(runseed);device=torch.device('cuda:0');mean,std=fit_scaler(stats,fit)
    train_dl=loader(frames,stats,labels,fit,mean,std,True,runseed);val_dl=loader(frames,stats,labels,validation,mean,std,False,runseed+1)if validation is not None else None
    net=Model(name).to(device);optim=torch.optim.AdamW(net.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    lossfn=company.FocalLoss(alpha=CFG['focal_alpha'],gamma=CFG['focal_gamma'])
    total=epochs or CFG['max_epochs'];warm=min(CFG['warmup_epochs'],max(1,total//3))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda ep:(ep+1)/warm if ep<warm else .5*(1+math.cos(math.pi*(ep-warm)/max(1,total-warm))))
    start=0;best=-1;best_epoch=1;bad=0;top=[];logs=[];resume=folder/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False);net.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];bad=state['bad'];top=state['top'];logs=state['logs'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu()for s in state['cuda_rng']]);train_dl.sampler.generator.set_state(state['sampler_rng'].cpu())
    for ep in range(start,total):
        timer=time.monotonic();net.train();losses=[]
        for x,s,y in train_dl:
            x=x.to(device);s=s.to(device);y=y.to(device);optim.zero_grad(set_to_none=True);logits=net(x,s);loss=lossfn(logits,y);assert torch.isfinite(loss),'nonfinite loss';loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optim.step();losses.append(float(loss.detach()))
        scheduler.step();monitor=None;ap=None
        statecpu={k:v.detach().cpu().clone()for k,v in net.state_dict().items()}
        if val_dl is not None:
            logits=predict(net,val_dl,device);p=torch.softmax(torch.tensor(logits),1)[:,1].numpy();monitor=float(balanced_accuracy_score(labels[validation],p>=.5));ap=float(average_precision_score(labels[validation],p));top.append((monitor,ap,ep+1,statecpu));top=sorted(top,key=lambda q:(q[0],q[1]),reverse=True)[:CFG['checkpoint_ensemble']]
            if monitor>best+.002:best=monitor;best_epoch=ep+1;bad=0
            else:bad+=1
        else:top.append((0.,0.,ep+1,statecpu));top=top[-CFG['checkpoint_ensemble']:];best_epoch=ep+1
        logs.append(dict(epoch=ep+1,loss=float(np.mean(losses)),validation_balanced_accuracy=monitor,validation_ap=ap,seconds=time.monotonic()-timer))
        pd.DataFrame(logs).to_csv(folder/'epoch_metrics.csv',index=False)
        save(resume,dict(model=net.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),epoch=ep+1,best=best,best_epoch=best_epoch,bad=bad,top=top,logs=logs,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler_rng=train_dl.sampler.generator.get_state()))
        print(f'{folder.parent.name}/{folder.name} epoch {ep+1}/{total} loss={logs[-1]["loss"]:.4f} innerBA={monitor} sec={logs[-1]["seconds"]:.1f}',flush=True)
        if validation is not None and bad>=CFG['patience']:break
    result=dict(model=name,seed=runseed,selected_epoch=best_epoch,epochs_trained=len(logs),parameter_count=sum(p.numel()for p in net.parameters()),fit_indices=list(map(int,fit)),validation_indices=list(map(int,validation))if validation is not None else [],scaler_fit_only=True)
    b=dict(states=[q[3]for q in top],mean=mean,std=std,selected_epoch=best_epoch,metadata=result)
    save(bundle,b);result['checkpoint_sha256']=sha(bundle);write_json(done,result);return result,b
@torch.no_grad()
def ensemble(name,b,frames,stats,labels,index):
    dl=loader(frames,stats,labels,index,b['mean'],b['std'],False,0);alllog=[];net=Model(name).cuda()
    for state in b['states']:net.load_state_dict(state);alllog.append(predict(net,dl,'cuda:0'))
    return torch.softmax(torch.tensor(np.stack(alllog).mean(0)),1)[:,1].numpy()
def task(spec):
    torch.set_num_threads(CFG['cpu_threads']);meta=pd.read_csv(ROOT/'data/animal_manifest.csv');labels=meta.label.to_numpy(int)
    folder=ROOT/'runs'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    if spec['design']=='fivefold':test=np.flatnonzero(meta.outer_fold.eq(spec['held_out']));development=np.flatnonzero(~meta.outer_fold.eq(spec['held_out']))
    elif spec['design']=='filename_date':
        known=meta.filename_date.ne('unknown');test=np.flatnonzero(meta.filename_date.eq(spec['held_out']));development=np.flatnonzero(known & ~meta.filename_date.eq(spec['held_out']))
    else:test=np.flatnonzero(meta.legacy_group.eq(spec['held_out']));development=np.flatnonzero(~meta.legacy_group.eq(spec['held_out']))
    assert not set(test)&set(development);assert len(np.unique(labels[development]))==2
    frame,stats=ids_and_inputs(meta,spec['model'],spec['condition']);innerps=[];innerids=[];epochs=[];split_rows=[]
    if spec['design']=='fivefold':
        split=pd.read_csv(ROOT/'data/split_manifest.csv');part=split[(split.outer_fold==spec['held_out'])&split.role.eq('inner_val')]
        inner=[(np.array([i for i in development if meta.iloc[i].animal_id not in set(part[part.inner_fold.eq(k)].animal_id)]),np.array([i for i in development if meta.iloc[i].animal_id in set(part[part.inner_fold.eq(k)].animal_id)]))for k in range(1,6)]
    else:
        sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=42);inner=[(development[a],development[b])for a,b in sk.split(development,labels[development])]
    for k,(fit,validation)in enumerate(inner,1):
        assert len(validation)>0 and set(fit).isdisjoint(test)and set(validation).isdisjoint(test)
        info,b=train_run(spec['model'],frame,stats,labels,fit,validation,folder/f'inner_{k}',spec['seed']+k*100+int(spec['partition_index'])*10000)
        innerps.extend(ensemble(spec['model'],b,frame,stats,labels,validation));innerids.extend(validation);epochs.append(info['selected_epoch'])
        for role,idx in [('inner_train',fit),('inner_val',validation)]:split_rows.extend(dict(inner_fold=k,animal_id=meta.iloc[i].animal_id,role=role)for i in idx)
    assert sorted(innerids)==sorted(development.tolist());threshold=select_threshold(labels[innerids],np.array(innerps));duration=max(1,int(round(np.median(epochs))))
    pd.DataFrame(split_rows).to_csv(folder/'inner_split_manifest.csv',index=False)
    info,b=train_run(spec['model'],frame,stats,labels,development,None,folder/'final',spec['seed']+900+int(spec['partition_index'])*10000,epochs=duration)
    probabilities=ensemble(spec['model'],b,frame,stats,labels,test)
    df=meta.iloc[test][['animal_id','label','outer_fold','filename_date','legacy_group']].copy();df['probability']=probabilities;df['threshold']=threshold;df['prediction']=(probabilities>=threshold).astype(int)
    for key in ['model','design','condition','seed','held_out']:df[key]=spec[key]
    df.to_csv(folder/'predictions.csv',index=False)
    pd.DataFrame(dict(animal_id=meta.iloc[i].animal_id,label=int(labels[i]),probability=float(p))for i,p in zip(innerids,innerps)).to_csv(folder/'inner_oof.csv',index=False)
    result=dict(task=spec,threshold=threshold,selected_epochs=duration,inner_selected_epochs=epochs,fit_animals=meta.iloc[development].animal_id.tolist(),test_animals=meta.iloc[test].animal_id.tolist(),test_metrics=score(labels[test],probabilities),protocol_sha256=sha(ROOT/'protocol.json'),input_manifest_sha256=sha(ROOT/'data/animal_manifest.csv'),vendor_sha256=sha(ROOT/'vendor/retinafuse_core.py'),predictions_sha256=sha(folder/'predictions.csv'),parameter_count=info['parameter_count'])
    write_json(folder/'complete.json',result);print('COMPLETED',spec['id'],result['test_metrics'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);args=ap.parse_args();task(json.loads(args.task))
