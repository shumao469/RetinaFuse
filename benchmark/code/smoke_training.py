"""Small real-input runtime test, excluded from scientific model comparisons."""
import numpy as np,pandas as pd
from common import ROOT,write_json
from train import ids_and_inputs,train_run,ensemble
def main():
    meta=pd.read_csv(ROOT/'data/animal_manifest.csv');ready=meta[meta.animal_id.map(lambda a:(ROOT/f'data/{a}.npz').exists())]
    meta=pd.concat([ready[ready.label.eq(0)].head(4),ready[ready.label.eq(1)].head(4)]).reset_index(drop=True);assert len(meta)==8
    x,s=ids_and_inputs(meta,'RetinaFuse','legacy');y=meta.label.to_numpy(int);fit=np.array([0,1,4,5]);val=np.array([2,3,6,7]);folder=ROOT/'audit/runtime_smoke'
    info,b=train_run('RetinaFuse',x,s,y,fit,val,folder,100,epochs=2);a=ensemble('RetinaFuse',b,x,s,y,val);info2,b2=train_run('RetinaFuse',x,s,y,fit,val,folder,100,epochs=2);c=ensemble('RetinaFuse',b2,x,s,y,val)
    error=float(np.max(np.abs(a-c)));assert error<1e-6
    write_json(ROOT/'audit/smoke_training_passed.json',dict(runtime_training=True,epochs=2,real_input_records=8,checkpoint_reload_max_difference=error,excluded_from_scientific_benchmark=True))
    print('REAL-INPUT RUNTIME TRAINING AND CHECKPOINT RELOAD PASSED',flush=True)
if __name__=='__main__':main()
