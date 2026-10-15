import json,time
import numpy as np,pandas as pd,torch
from common import ROOT,CFG,sha,write_json
from models import Model,company
def main():
    torch.set_num_threads(4);meta=pd.read_csv(ROOT/'data/animal_manifest.csv');split=pd.read_csv(ROOT/'data/split_manifest.csv');checks=[]
    assert len(meta)==meta.animal_id.nunique()==CFG['expected_primary_n'] and meta.label.value_counts().to_dict()=={1:89,0:31}
    for fold in range(1,6):
        q=split[split.outer_fold.eq(fold)];test=set(q[q.role.eq('outer_test')].animal_id);expected=set(meta[meta.outer_fold.eq(fold)].animal_id);assert test==expected
        vals=[]
        for k in range(1,6):
            fit=set(q[q.role.eq('inner_train')&q.inner_fold.eq(k)].animal_id);val=set(q[q.role.eq('inner_val')&q.inner_fold.eq(k)].animal_id)
            assert fit.isdisjoint(val)and fit.isdisjoint(test)and val.isdisjoint(test);assert fit|val|test==set(meta.animal_id);vals.extend(val)
        assert len(vals)==len(set(vals))==len(meta)-len(test)
    checks.append(dict(check='120 unique source records, labels and nested splits',passed=True))
    a=meta.iloc[0].animal_id
    with np.load(ROOT/f'data/{a}.npz')as z:x=torch.from_numpy(z['legacy_events'][None].copy());s=torch.from_numpy(z['legacy_stats'][None].copy())
    # Exact network equivalence to the company backbone, including adaptive gate/head.
    old=company.EventFusionNet('full').eval();new=Model('RetinaFuse').eval();mapping={k.replace('branch_a.','frame.').replace('branch_b.','stats.').replace('fusion.','gate.'):v for k,v in old.state_dict().items()};new.load_state_dict(mapping)
    with torch.no_grad():error=float((old(x,s)-new(x,s)).abs().max())
    assert error<1e-5;checks.append(dict(check='RetinaFuse matches company network forward',passed=True,max_absolute_error=error))
    net=Model('Unordered-frames').eval();xx=torch.rand(1,5,1,64,64)
    with torch.no_grad():error=float((net(xx,s)-net(xx[:,[2,4,1,3,0]],s)).abs().max())
    assert error<1e-5;checks.append(dict(check='unordered model permutation invariant',passed=True,max_absolute_error=error))
    for name in CFG['models']:
        net=Model(name).cuda();xx=torch.rand(2,1,64,64,device='cuda')if name in ['Single-frame','Mean-image']else torch.rand(2,60,1 if name in ['Raw-video','Unordered-frames','Frame-difference-video']else 2,64,64,device='cuda');ss=torch.rand(2,12,device='cuda')
        net.train();out=net(xx,ss);loss=company.FocalLoss(alpha=CFG['focal_alpha'],gamma=CFG['focal_gamma'])(out,torch.tensor([0,1],device='cuda'));loss.backward();assert torch.isfinite(loss);checks.append(dict(check=name+' GPU forward/backward',passed=True,parameters=sum(p.numel()for p in net.parameters()),loss=float(loss.detach())))
        del net,xx,ss,out,loss;torch.cuda.empty_cache()
    write_json(ROOT/'audit/preflight_checks.json',checks);print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
